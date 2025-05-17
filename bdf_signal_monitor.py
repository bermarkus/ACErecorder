import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import mne
from mne.io import read_raw_bdf
import queue

class BDFSignalMonitor:
    def __init__(self, parent=None):
        """Initialize the BDF signal monitor window"""
        self.parent = parent
        self.window = None
        self.canvas = None
        self.figure = None
        self.axes = None
        self.lines = []
        self.channel_labels = []
        self.update_interval = 1000  # 1Hz update rate in milliseconds
        self.window_length = 10  # 10 seconds of data
        
        # Threading and monitoring control
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        self.data_queue = queue.Queue()
        
        # BDF file tracking
        self.current_bdf_file = None
        self.last_file_size = 0
        self.last_check_time = 0
        
        # Data display settings
        self.num_channels = 0
        self.channel_offsets = []  # For stacking channels vertically
        self.channel_scales = []   # Individual scaling for each channel
        self.y_scale = 1.0  # Global scaling factor
        self.auto_scale = True  # Enable auto-scaling based on data
        self.per_channel_scale = True  # Scale each channel individually
        
    def create_window(self):
        """Create the signal monitor window if it doesn't exist"""
        if self.window is None:
            # Create a fullscreen window
            self.window = tk.Toplevel(self.parent)
            self.window.title("BDF Signal Monitor")
            # Set to fullscreen
            self.window.attributes("-fullscreen", True)
            
            # Get screen width and height
            screen_width = self.window.winfo_screenwidth()
            screen_height = self.window.winfo_screenheight()
            
            # Make sure window appears on top
            self.window.attributes("-topmost", True)
            
            # Redirect window close to minimize
            self.window.protocol("WM_DELETE_WINDOW", self.minimize_window)
            
            # Create a frame for controls at the top
            control_frame = ttk.Frame(self.window)
            control_frame.pack(fill=tk.X, pady=2)
            
            # Add minimize and close buttons
            minimize_btn = ttk.Button(control_frame, text="Minimize", command=self.minimize_window)
            minimize_btn.pack(side=tk.LEFT, padx=5)
            
            # Exit fullscreen button
            exit_fullscreen_btn = ttk.Button(control_frame, text="Exit Fullscreen", 
                                           command=lambda: self.window.attributes("-fullscreen", False))
            exit_fullscreen_btn.pack(side=tk.LEFT, padx=5)
            
            # Create figure and canvas for plotting - use the full screen dimensions
            # Get the screen DPI to correctly calculate single-pixel lines
            import ctypes
            user32 = ctypes.windll.user32
            try:
                # Get the actual screen DPI (Windows specific)
                LOGPIXELSX = 88
                dc = user32.GetDC(0)
                screen_dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
                user32.ReleaseDC(0, dc)
                print(f"Screen DPI detected: {screen_dpi}")
            except:
                # Fall back to a standard DPI if detection fails
                screen_dpi = 96
                print(f"Using default DPI: {screen_dpi}")
                
            # Calculate true pixel size in points (1/72 inch) for matplotlib
            true_pixel_size = 72.0 / screen_dpi
            print(f"True pixel size for linewidth: {true_pixel_size}")
                
            # Adjust figure size to match screen dimensions (in inches)
            fig_width = screen_width / screen_dpi
            fig_height = screen_height / screen_dpi
            
            # Set dark gray background for the figure with specified DPI
            self.figure, self.axes = plt.subplots(figsize=(fig_width, fig_height), 
                                                  facecolor='#404040', dpi=screen_dpi)
            self.axes.set_facecolor('#404040')
            
            # Store the true pixel size for line drawing
            self.true_pixel_size = true_pixel_size
            
            # Remove ALL margins and padding completely
            self.figure.subplots_adjust(left=0, right=1, top=0.98, bottom=0, hspace=0, wspace=0)
            
            # Remove the axis frame entirely
            self.axes.spines['left'].set_visible(False)
            self.axes.spines['right'].set_visible(False)
            self.axes.spines['top'].set_visible(False)
            self.axes.spines['bottom'].set_visible(False)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial plot setup
            self.axes.set_title("EEG Signal Monitor (Last 10 seconds)", color='white')
            self.axes.set_xlabel("Time (seconds)", color='white')
            self.axes.set_ylabel("Channel", color='white')
            
            # Set grid with vertical lines every second
            self.axes.grid(True, which='major', axis='x', linestyle='-', alpha=0.5)
            self.axes.grid(True, which='major', axis='y', linestyle='-', alpha=0.3)
            
            # Set tick colors to white
            self.axes.tick_params(axis='x', colors='white')
            self.axes.tick_params(axis='y', colors='white')
            
            # Set spine colors to light gray
            for spine in self.axes.spines.values():
                spine.set_color('#808080')
                
            # Ensure window is visible and on top
            self.window.lift()
            self.window.attributes('-topmost', True)
            self.window.after(100, lambda: self.window.attributes('-topmost', False))
            self.window.update()
            
            # Force window to be displayed
            self.window.deiconify()
            print("BDF Signal Monitor window created and displayed")
    
    def minimize_window(self):
        """Minimize the window instead of closing it"""
        self.window.wm_iconify()
    
    def close_monitor(self):
        """Force close the monitor window"""
        if self.window:
            # Allow window to close by temporarily removing protocol
            self.window.protocol("WM_DELETE_WINDOW", lambda: None)
            self.window.destroy()
            self.window = None
            self.canvas = None
            self.figure = None
            self.axes = None
            self.lines = []
    
    def start_monitoring(self, bdf_file_path):
        """Start monitoring the BDF file"""
        # Suppress warnings globally for this process
        import warnings
        warnings.filterwarnings("ignore", message="Number of records from the header does not match the file size")
        
        print(f"Starting BDF monitor with file: {bdf_file_path}")
        if not os.path.exists(os.path.dirname(bdf_file_path)):
            print(f"Directory does not exist: {os.path.dirname(bdf_file_path)}")
            return
            
        # Create window if it doesn't exist
        self.create_window()
        
        # Reset monitoring state
        self.stop_flag.clear()
        self.current_bdf_file = bdf_file_path
        self.last_file_size = 0
        self.last_check_time = time.time()
        
        # Initial placeholder data
        self._setup_initial_display()
        
        # Start the monitoring thread if not already running
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitoring_thread, daemon=True)
            self.monitor_thread.start()
            print("BDF monitor thread started")
            
        # Start the UI update timer with shorter initial interval for faster first update
        print("Starting UI update timer")
        self.window.after(100, self.update_plot)
        
        # Force window to be displayed again
        self.window.lift()
        self.window.deiconify()
    
    def stop_monitoring(self):
        """Stop monitoring the BDF file"""
        print("Stopping BDF monitor")
        self.stop_flag.set()
        
        # Clear the data queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                pass
                
        # Wait for thread to terminate
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None
    
    def _monitoring_thread(self):
        """Background thread that monitors the BDF file and updates the data queue"""
        # Import warnings and filter the specific BDF header warning at the thread level
        import warnings
        warnings.filterwarnings("ignore", message="Number of records from the header does not match the file size")
        
        retry_count = 0
        max_retries = 5
        
        while not self.stop_flag.is_set():
            try:
                # Check if file exists and has been updated
                if not os.path.exists(self.current_bdf_file):
                    time.sleep(0.5)
                    continue
                    
                current_size = os.path.getsize(self.current_bdf_file)
                
                # Only process if file size has changed and file has some minimum size
                if current_size > self.last_file_size and current_size > 1024:
                    # Update file size tracking
                    self.last_file_size = current_size
                    retry_count = 0  # Reset retry counter on successful size check
                    
                    # Attempt to read the BDF file
                    try:
                        print(f"Extracting EDF parameters from {self.current_bdf_file}...")
                        # Use MNE to read the BDF file with appropriate error handling
                        try:
                            # Suppress the specific warning about header record count
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message="Number of records from the header does not match the file size")
                                raw = read_raw_bdf(self.current_bdf_file, preload=True)
                                
                            print(f"BDF loaded successfully with {len(raw.ch_names)} channels")
                            
                            # Get data for the last window_length seconds
                            # Calculate the start index as an integer
                            start_idx = max(0, int(len(raw.times) - self.window_length * raw.info['sfreq']))
                            data, times = raw[:, start_idx:]
                            
                            # Verify data integrity
                            if data.size == 0 or times.size == 0:
                                print("Warning: Empty data or time array")
                                time.sleep(0.5)
                                continue
                                
                            # If this is our first read, set up the channel information
                            if self.num_channels == 0:
                                self.num_channels = len(raw.ch_names)
                                self.channel_labels = raw.ch_names
                                self.channel_offsets = np.arange(self.num_channels) * self.y_scale * 2
                            
                            # Print some debug info about the data
                            data_min = np.min(data)
                            data_max = np.max(data)
                            data_range = data_max - data_min
                            
                            print(f"Read BDF data: shape={data.shape}, time points={len(times)}, sample rate={raw.info['sfreq']}")
                            print(f"Value range: min={data_min:.2f}, max={data_max:.2f}, range={data_range:.2f}")
                            
                            # Auto-adjust y_scale based on actual data range if needed
                            if self.auto_scale and data_range > 0:
                                # Set y_scale to make typical signals about 1.0 units high
                                # with a safety factor of 10 to prevent tiny signals
                                suggested_scale = 2.0 / max(data_range, 0.0001)
                                # Limit how much the scale can change at once
                                if abs(suggested_scale / self.y_scale) > 100 or abs(suggested_scale / self.y_scale) < 0.01:
                                    self.y_scale = suggested_scale
                                    print(f"Auto-adjusted y_scale to {self.y_scale:.8f} based on data range {data_range:.8f}")
                            
                            # Put the data in the queue for the UI thread to consume
                            # Clear queue first to avoid backlog
                            while not self.data_queue.empty():
                                try:
                                    self.data_queue.get_nowait()
                                except:
                                    pass
                                    
                            self.data_queue.put((data, times, raw.info['sfreq']))
                            print("Data successfully added to queue")
                            
                        except ValueError as e:
                            if "slice indices" in str(e):
                                print("Error with slice indices - using alternative approach")
                                # Alternative approach using direct sample calculation
                                # Apply same warning suppression here
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", message="Number of records from the header does not match the file size")
                                    samples_to_read = min(int(self.window_length * raw.info['sfreq']), len(raw.times))
                                    data, times = raw[:, -samples_to_read:]
                                    
                                print(f"Using alternative method: retrieved {samples_to_read} samples")
                                self.data_queue.put((data, times, raw.info['sfreq']))
                            else:
                                raise e
                        
                    except Exception as e:
                        print(f"Error reading BDF file: {e}")
                
                # Sleep to avoid tight loop
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(1.0)
    
    def update_plot(self):
        """Update the signal plot with new data from the queue"""
        if self.window is None or not self.window.winfo_exists():
            print("Window doesn't exist or was closed, exiting update_plot")
            return
            
        try:
            print(f"Checking data queue (empty={self.data_queue.empty()})")
            # Get the latest data from the queue
            if not self.data_queue.empty():
                try:
                    data, times, sfreq = self.data_queue.get_nowait()
                    print(f"Updating plot with data shape: {data.shape}, time points: {len(times)}")
                except Exception as e:
                    print(f"Error getting data from queue: {e}")
                    data, times, sfreq = None, None, None
                
                if data.shape[1] < 2:
                    print("Not enough data points to plot yet")
                    return
                
                # Clear previous plot
                self.axes.clear()
                self.lines = []
                
                # Determine scale factor based on max amplitude
                max_amp = np.max(np.abs(data))
                if max_amp > 0:
                    # Apply both the auto-scale factor and a scaling to use 80% of available channel space
                    scale_factor = 0.8 / max_amp * self.y_scale
                    print(f"Plot scale_factor: {scale_factor:.8f} for max_amp: {max_amp:.8f}")
                else:
                    scale_factor = self.y_scale  # Use current y_scale directly
                
                # Calculate channel offsets, with spacing proportional to the number of channels
                # Use a fixed offset that doesn't depend on y_scale to avoid rescaling issues
                base_offset = 2.0  # Fixed offset between channels regardless of scale
                
                if self.num_channels > 30:
                    channel_spacing = base_offset * 1.5
                else:
                    channel_spacing = base_offset * 2
                    
                self.channel_offsets = np.arange(self.num_channels) * channel_spacing
                print(f"Channel spacing: {channel_spacing}, Using y_scale: {self.y_scale:.8f}")
                
                # Get a colormap for the channels
                cm = plt.cm.get_cmap('tab10')
                
                # Update or initialize per-channel scaling factors
                if len(self.channel_scales) != data.shape[0] or not self.per_channel_scale:
                    # Initialize with default scaling
                    self.channel_scales = [self.y_scale] * data.shape[0]
                
                # Calculate individual channel scales if enabled
                if self.per_channel_scale:
                    for i in range(data.shape[0]):
                        # Get amplitude of this channel
                        channel_max = np.max(np.abs(data[i]))
                        if channel_max > 0:
                            # Aim for a normalized height of 0.8 units per channel
                            target_scale = 0.8 / channel_max
                            # Smooth the scale change to avoid abrupt changes
                            self.channel_scales[i] = self.channel_scales[i] * 0.7 + target_scale * 0.3
                    
                    # Print some debug about the channel scales
                    print(f"Channel scales - min: {min(self.channel_scales):.4f}, max: {max(self.channel_scales):.4f}")
                
                # Plot all channels in yellow (ffff00) as requested
                for i in range(min(self.num_channels, data.shape[0])):
                    # Apply individual channel scaling if enabled
                    if self.per_channel_scale:
                        channel_scale = self.channel_scales[i]
                    else:
                        channel_scale = scale_factor
                        
                    # Center the signal around its offset
                    scaled_data = data[i] * channel_scale
                    
                    # Use yellow color for all lines, exactly one pixel thick with no anti-aliasing
                    line, = self.axes.plot(
                        times, 
                        scaled_data + self.channel_offsets[i],
                        linewidth=self.true_pixel_size,  # Exact single-pixel line
                        color='#ffff00',       # Yellow color as requested
                        antialiased=False,     # Disable anti-aliasing
                        solid_capstyle='butt', # No line caps
                        solid_joinstyle='miter', # Sharp corners
                        snap=True              # Snap to pixel grid
                    )
                    self.lines.append(line)
                
                # Add channel labels on the y-axis (show a subset if there are many channels)
                if self.num_channels > 30:
                    # For many channels, only label every 5th channel to avoid overcrowding
                    shown_indices = range(0, self.num_channels, 5)
                    y_ticks = [self.channel_offsets[i] for i in shown_indices]
                    y_labels = [self.channel_labels[i] for i in shown_indices]
                    self.axes.set_yticks(y_ticks)
                    self.axes.set_yticklabels(y_labels, color='white')
                else:
                    # For fewer channels, show all labels
                    y_ticks = self.channel_offsets
                    self.axes.set_yticks(y_ticks)
                    self.axes.set_yticklabels(self.channel_labels, color='white')
                
                # Set time axis to show EXACTLY last window_length seconds with no padding
                if len(times) > 0:
                    # Calculate precise start and end times
                    end_time = times[-1]
                    start_time = end_time - self.window_length
                    
                    # Set exact limits with no padding whatsoever
                    self.axes.set_xlim([start_time, end_time])
                    
                    # Set vertical grid lines at every second with exact positioning
                    tick_start = int(np.ceil(start_time))
                    tick_end = int(np.floor(end_time))
                    if tick_end - tick_start < self.window_length - 1:
                        tick_end = tick_start + int(self.window_length) - 1
                    self.axes.set_xticks(range(tick_start, tick_end + 1))
                    
                    # Ensure grid shows up properly with the dark background
                    self.axes.grid(True, which='major', axis='x', linestyle='-', color='#808080', alpha=0.8)
                    
                # Expand plot to fill all available space
                self.figure.tight_layout(pad=0)
                
                # Ensure axis spans the full figure
                self.axes.set_position([0, 0, 1, 1])
                
                # Add title with current info and scaling mode
                title = "EEG Signal Monitor"
                if self.current_bdf_file:
                    title += f" - {os.path.basename(self.current_bdf_file)}"
                    
                # Add scaling mode information
                if self.per_channel_scale:
                    title += " (Per-channel scaling)"
                else:
                    title += " (Global scaling)"
                    
                self.axes.set_title(title, color='white', fontsize=14)
                self.axes.set_xlabel("Time (seconds)", color='white')
                
                # Make sure we have the correct background color in case it gets reset
                self.axes.set_facecolor('#404040')
                self.figure.patch.set_facecolor('#404040')
                
                # Configure the grid for better visibility on dark background
                self.axes.grid(True, which='major', axis='x', linestyle='-', color='#808080', alpha=0.8)
                self.axes.grid(True, which='major', axis='y', linestyle='-', color='#606060', alpha=0.4)
                
                # Update tick colors to ensure they remain visible
                self.axes.tick_params(axis='x', colors='white')
                self.axes.tick_params(axis='y', colors='white')
                
                self.canvas.draw()
        
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Always reschedule the next update with shorter interval if no data yet
        update_interval = self.update_interval
        if self.data_queue.empty() and self.num_channels == 0:
            # Use shorter interval if we haven't received any data yet
            update_interval = 200  # Try more frequently until we get data
            
        if not self.stop_flag.is_set() and self.window and self.window.winfo_exists():
            print(f"Scheduling next update in {update_interval}ms")
            self.window.after(update_interval, self.update_plot)
        else:
            print("Not scheduling next update (window closed or monitoring stopped)")
            
    def _setup_initial_display(self):
        """Setup initial display with placeholder data"""
        # Create some initial placeholder data
        sample_rate = 250  # Typical EEG sample rate
        duration = self.window_length  # 10 seconds
        num_samples = int(sample_rate * duration)
        
        # Determine number of channels based on filename pattern or default to 23
        if self.current_bdf_file:
            if "19ch" in self.current_bdf_file.lower():
                self.num_channels = 19
            elif "32ch" in self.current_bdf_file.lower():
                self.num_channels = 32
            elif "64ch" in self.current_bdf_file.lower():
                self.num_channels = 64
            else:
                self.num_channels = 23  # Default to 23 channels
        else:
            self.num_channels = 23
            
        # Create channel labels if not already set
        if not self.channel_labels or len(self.channel_labels) != self.num_channels:
            self.channel_labels = [f"CH{i+1}" for i in range(self.num_channels)]
            
        # Create channel offsets for display
        self.channel_offsets = np.arange(self.num_channels) * self.y_scale * 2
        
        # Create time points
        times = np.linspace(0, duration, num_samples)
        
        # Get a colormap for the channels
        cm = plt.cm.get_cmap('tab10')
        
        # Create placeholder sine waves with different frequencies for each channel
        data = np.zeros((self.num_channels, num_samples))
        for i in range(self.num_channels):
            # Different frequency for each channel
            freq = 1 + (i % 10) * 0.5  # 1-5.5 Hz
            data[i, :] = self.y_scale * 0.5 * np.sin(2 * np.pi * freq * times)
            
        # Clear previous plot
        self.axes.clear()
        self.lines = []
        
        # Plot each channel with a color from the colormap
        for i in range(self.num_channels):
            color_idx = i % 10  # cycle through 10 colors
            line, = self.axes.plot(
                times, 
                data[i] + self.channel_offsets[i],
                linewidth=self.true_pixel_size,  # Exact single-pixel line
                color='#ffff00',        # Yellow color as requested
                antialiased=False,      # Disable anti-aliasing
                solid_capstyle='butt',  # No line caps
                solid_joinstyle='miter',# Sharp corners
                snap=True               # Snap to pixel grid
            )
            self.lines.append(line)
            
        # Add channel labels
        self.axes.set_yticks(self.channel_offsets)
        self.axes.set_yticklabels(self.channel_labels)
        
        # Set axes limits
        self.axes.set_xlim([0, duration])
        
        # Update title
        if self.current_bdf_file:
            self.axes.set_title(f"BDF Signal Monitor - {os.path.basename(self.current_bdf_file)}")
        else:
            self.axes.set_title("BDF Signal Monitor - Waiting for data")
            
        self.axes.set_xlabel("Time (seconds)")
        self.axes.grid(True)
        
        # Update the canvas
        self.canvas.draw()

# For testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Test BDF Monitor")
    root.geometry("300x200")
    
    def test_monitor():
        monitor = BDFSignalMonitor(root)
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG files")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create test file path
        test_file = os.path.join(test_dir, "test_recording.bdf")
        
        # Create or add to the test file
        with open(test_file, "wb") as f:
            f.write(b"TESTBDF" + os.urandom(1024))
            
        # Start monitoring
        monitor.start_monitoring(test_file)
        
        # Simulate file growth
        def update_file():
            if os.path.exists(test_file):
                with open(test_file, "ab") as f:
                    f.write(os.urandom(1024))
            root.after(2000, update_file)
            
        update_file()
    
    test_btn = ttk.Button(root, text="Start Test", command=test_monitor)
    test_btn.pack(pady=20)
    
    root.mainloop()