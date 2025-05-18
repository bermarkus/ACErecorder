import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import time
import threading
import os
import serial
from serial.tools import list_ports
from datetime import datetime
import json
import sys
import traceback
import webbrowser
import pyedflib
import copy
import winsound  # For playing bell sound

# Import electrode monitoring modules
import electrode_monitor
import signal_detect

# Import BDF signal monitor
from bdf_signal_monitor import BDFSignalMonitor

# Try to import MNE-related packages with error handling
try:
    import mne
    import mne_connectivity
    HAS_MNE = True
except ImportError as e:
    print(f"Warning: MNE import failed: {e}")
    HAS_MNE = False

# Try to import pyedflib with error handling
try:
    import pyedflib
except ImportError as e:
    print(f"Warning: pyedflib import failed: {e}")
    pyedflib = None

def compute_coherence(data, fs, fmin=7, fmax=12, duration=1, overlap=0):
    """Compute coherence between two channels (Fp1 and Fp2)"""
    if not HAS_MNE:
        print("MNE library not available for coherence calculation")
        return 0.0
        
    try:
        mne.set_log_level("CRITICAL")
        
        # Print input data stats
        print(f"\nCoherence calculation input:")
        print(f"Data shape: {data.shape}")
        print(f"Sample rate: {fs}")
        print(f"First few samples Fp1: {data[0, :5]}")
        print(f"First few samples Fp2: {data[1, :5]}")
        
        ch_names = ['Fp1', 'Fp2']
        ch_types = ['eeg', 'eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, overlap=overlap, verbose="ERROR")

        con_obj = mne_connectivity.spectral_connectivity_epochs(
            epochs, method='coh', sfreq=fs, fmin=fmin, fmax=fmax,
            faverage=True, mt_adaptive=True, n_jobs=1, verbose="CRITICAL"
        )
        con_values = con_obj.get_data(output='dense')[:, :, 0]
        coherence = 4 * np.mean(con_values)
        
        print(f"Calculated coherence: {coherence:.4f}")
        return coherence
    except Exception as e:
        print(f"Coherence calculation error: {e}")
        traceback.print_exc()  # Add full traceback
        return 0.0

# Channel configurations
INPUT_ORDER_19 = ["O2", "P8", "T8", "F8", "Fp2", "F4", "C4", "P4", "sync", "ch10", "ch11", "A2", 
                  "Pz", "HR", "ch15", "ch16", "Fz", "Cz", "ch19", "ch20", "ch21", "ch22", "ch23", 
                  "A1", "Fp1", "F3", "C3", "P3", "O1", "P7", "T7", "F7"]

INPUT_ORDER_32 = ["O2", "P8", "A2", "F8", "Fp2", "F4", "C4", "P4", "FC6", "CP6", "CP2", "PO4", 
                  "Pz", "HR", "FC2", "AF4", "Fz", "Cz", "FC1", "AF3", "FC5", "CP5", "CP1", "PO3", 
                  "Fp1", "F3", "C3", "P3", "O1", "P7", "A1", "F7"]

CHANNEL_CONFIGS = {
    "2 Channel Headset": {  # Updated to match UI
        "input_order": INPUT_ORDER_19,
        "output_order": ["Fp1", "Fp2", "Coherence"],
        "description": "2 Channel Mode (Fp1, Fp2)"
    },
    "19 Channel": {
        "input_order": INPUT_ORDER_19,
        "output_order": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", 
                        "P7", "P3", "Pz", "P4", "P8", "O1", "O2", "A1", "A2", "HR", "sync"],
        "description": "19 Channel Mode (10-20 System)"
    },
    "32 Channel": {
        "input_order": INPUT_ORDER_32,
        "output_order": ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", 
                        "FC5", "FC6", "C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6", "P7", "P3", 
                        "Pz", "P4", "P8", "PO3", "PO4", "O1", "O2", "A1", "A2", "HR"],
        "description": "32 Channel Mode (Extended 10-20)"
    }
}

class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size and position
        window_width = 300
        window_height = 200
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create a frame with a border
        frame = tk.Frame(self.root, borderwidth=2, relief='solid')
        frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Add loading text
        tk.Label(frame, text="ACErecorder", font=('Helvetica', 16, 'bold')).pack(pady=20)
        tk.Label(frame, text="Loading...", font=('Helvetica', 10)).pack()
        
        # Add a progress bar
        self.progress = ttk.Progressbar(frame, length=200, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()
        
        # Force update
        self.root.update()

    def destroy(self):
        self.root.destroy()

class ACErecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ACErecorder")
        
        # Initialize settings file path first
        try:
            if getattr(sys, 'frozen', False):
                # If the application is run as a bundle
                app_dir = os.path.dirname(sys.executable)
            else:
                # If the application is run from a Python interpreter
                app_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Use a single settings file for all instances
            self.settings_file = os.path.join(app_dir, "settings.json")
        except Exception as e:
            print(f"Error setting up app directory: {e}")
            app_dir = os.path.dirname(os.path.abspath(__file__))
            self.settings_file = os.path.join(app_dir, "settings.json")

        # Load settings
        self.settings = self.load_settings()
        self.custom_configs = self.settings.get("configurations", {}).copy()
        
        # Create variables
        self.channel_mode_var = tk.StringVar(value=self.settings.get("headset", "19 Channel"))
        
        # Handle icon setting
        try:
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = sys._MEIPASS  # Use PyInstaller's special path
            else:
                # Running as script
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            icon_path = os.path.join(base_path, "LogoSquare3.ico")
            if os.path.exists(icon_path):
                # Set window icons
                self.root.iconbitmap(default=icon_path)
                self.root.iconbitmap(icon_path)
                
                # Set taskbar icon
                try:
                    import ctypes
                    myappid = u'alphacoherence.eegrecorder.1.0'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                except Exception as e:
                    print(f"Failed to set app ID: {e}")
            else:
                print(f"Icon not found at: {icon_path}")
        except Exception as e:
            print(f"Could not set icon: {e}")

        # Calculate scaled dimensions (85% of original)
        logo_original_width = 900
        logo_original_height = 97
        scale_factor = 0.85  # Increased to 0.85 for wider window
        logo_width = int(logo_original_width * scale_factor)
        logo_height = int(logo_original_height * scale_factor)

        # Set window size based on logo width and increased height for annotations
        self.root.geometry(f"{logo_width}x800")  # Adjusted height to 800 for better fit
        self.root.resizable(False, False)

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="0")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add logo
        try:
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Head v28 (2).png")
            pil_image = Image.open(logo_path)
            pil_image = pil_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(pil_image)
            logo_label = tk.Label(self.main_frame, image=self.logo_image, cursor="hand2")
            logo_label.grid(row=0, column=0, columnspan=2, pady=(0,10), sticky='ew')
            logo_label.bind("<Button-1>", lambda e: webbrowser.open("https://alphacoherence.com"))
        except Exception as e:
            print(f"Could not load logo: {e}")

        # Content frame with padding
        content_frame = ttk.Frame(self.main_frame, padding="0 0 20 0")
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20)
        
        # Status variables
        self.recording = False
        self.board = None
        self.com_port = None
        self.filename_suffix = ""
        self.port_mapping = {}
        self.recording_start_time = None
        self.total_samples_written = 0  # Track total samples written
        self.current_buffer_samples = 0  # Track samples in current buffer
        self.annotations = []
        self.duration_indices = [None] * 5  # Track up to 5 duration annotations
        self.duration_timers = [None] * 5  # Track timer jobs for auto-stop
        self.bell_states = [tk.BooleanVar(value=True) for _ in range(5)]  # Bell enabled by default
        self.countdown_vars = [tk.StringVar(value="") for _ in range(5)]  # For countdown display
        self.countdown_timers = [None] * 5  # Track countdown update jobs
        
        # Electrode monitoring properties
        self.electrode_monitor_window = None
        self.electrode_check_timer = None
        self.electrode_check_interval = 1000  # Check electrode status every 1 second
        self.connection_status = {}
        self.monitor_electrodes = tk.BooleanVar(value=self.settings.get("monitor_electrodes", True))
        
        # BDF signal monitor properties
        self.bdf_signal_monitor = None
        self.monitor_bdf_signal = tk.BooleanVar(value=self.settings.get("monitor_bdf_signal", True))
        
        # Ensure EEG files directory exists
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            app_dir = os.path.dirname(sys.executable)  # Use executable's directory instead of _MEIPASS
        else:
            # Running as script
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.eeg_dir = os.path.join(app_dir, "EEG files")
        os.makedirs(self.eeg_dir, exist_ok=True)
        
        # Initialize variables
        self.status_var = tk.StringVar(value="Ready to record")  
        self.recording_duration_var = tk.StringVar(value="Duration: 0 seconds")  
        self.sample_rate_var = tk.StringVar(value="Sample Rate: -- Hz")
        self.channel_count_var = tk.StringVar(value="Channels: --")
        self.mode_description_var = tk.StringVar(value="")
        
        # Headset Selection
        ttk.Label(content_frame, text="Headset:").grid(row=0, column=0, pady=5, sticky=tk.W)
        
        # Combine standard and custom configurations
        headset_options = list(CHANNEL_CONFIGS.keys())
        if self.custom_configs:
            headset_options.extend([f"Custom: {name}" for name in self.custom_configs.keys()])
            
        self.channel_mode_combo = ttk.Combobox(content_frame, textvariable=self.channel_mode_var, 
                                             values=headset_options, state="readonly")
        self.channel_mode_combo.grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))
        self.channel_mode_combo.bind('<<ComboboxSelected>>', self.on_channel_mode_change)
        
        # Filename entry
        ttk.Label(content_frame, text="Recording Name (optional):").grid(row=1, column=0, pady=5, sticky=tk.W)
        self.filename_var = tk.StringVar()
        self.filename_entry = ttk.Entry(content_frame, textvariable=self.filename_var)
        self.filename_entry.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E))
        
        # COM Port selection
        ttk.Label(content_frame, text="COM Port:").grid(row=2, column=0, pady=5, sticky=tk.W)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(content_frame, textvariable=self.port_var, state="readonly")
        self.port_combo.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))
        self.port_combo.bind('<<ComboboxSelected>>', lambda e: self.on_port_change())
        
        # Status label with increased wraplength and fixed height
        self.status_var = tk.StringVar(value="Ready to record")  
        status_frame = ttk.Frame(content_frame, height=50)  # Fixed height container
        status_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')
        status_frame.grid_propagate(False)  # Prevent frame from shrinking
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=600)  # Increased wraplength
        self.status_label.grid(row=0, column=0, sticky='w')
        
        # Recording info frame
        info_frame = ttk.LabelFrame(content_frame, text="Recording Info", padding="5")
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Duration
        ttk.Label(info_frame, textvariable=self.recording_duration_var).grid(row=0, column=0, padx=5)  
        
        # Sample Rate
        ttk.Label(info_frame, textvariable=self.sample_rate_var).grid(row=0, column=1, padx=5)
        
        # Channel Count
        ttk.Label(info_frame, textvariable=self.channel_count_var).grid(row=0, column=2, padx=5)
        
        # Mode Description
        ttk.Label(info_frame, textvariable=self.mode_description_var, wraplength=380).grid(row=1, column=0, columnspan=3, padx=5)
        
        # Record button
        self.record_button = ttk.Button(content_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Annotation Controls Frame with increased padding
        annotation_controls_frame = ttk.LabelFrame(content_frame, text="Annotation Controls")
        annotation_controls_frame.grid(row=6, column=0, columnspan=2, pady=(10, 20), sticky='ew')  

        # Load annotation history from settings
        self.duration_history = self.settings.get("duration_history", [])
        self.marker_history = self.settings.get("marker_history", [])
        self.last_duration_labels = self.settings.get("last_duration_labels", [f"Duration {i+1}" for i in range(5)])
        self.last_marker_labels = self.settings.get("last_marker_labels", [f"Marker {i+1}" for i in range(5)])

        # Duration Annotations Section with padding
        duration_section = ttk.LabelFrame(annotation_controls_frame, text="Duration Annotations")
        duration_section.pack(fill='x', pady=5, padx=5)
        
        # Create 5 duration annotation controls with padding
        self.duration_vars = []
        self.duration_entries = []
        self.duration_timer_entries = []
        for i in range(5):
            duration_frame = ttk.Frame(duration_section)
            duration_frame.pack(fill='x', pady=3)
            
            duration_var = tk.BooleanVar(value=False)
            self.duration_vars.append(duration_var)
            
            duration_toggle = ttk.Checkbutton(
                duration_frame,
                text=f"Duration {i+1}",
                variable=duration_var,
                command=lambda idx=i: self.toggle_duration_annotation(idx)
            )
            duration_toggle.pack(side='left', padx=10)
            
            # Create entry with dropdown for history
            duration_entry = ttk.Combobox(duration_frame, width=37)  # Back to original width
            duration_entry.pack(side='left', padx=10)  # Back to original padding
            if self.duration_history:
                duration_entry['values'] = self.duration_history
            duration_entry.set(self.last_duration_labels[i])
            self.duration_entries.append(duration_entry)
            
            # Add timer entry fields with compact spacing
            timer_frame = ttk.Frame(duration_frame)
            timer_frame.pack(side='left', padx=10)
            
            ttk.Label(timer_frame, text="Timer:").pack(side='left', padx=(0,5))
            minutes_entry = ttk.Entry(timer_frame, width=3)
            minutes_entry.pack(side='left', padx=2)
            ttk.Label(timer_frame, text="m").pack(side='left', padx=(0,5))
            
            seconds_entry = ttk.Entry(timer_frame, width=3)
            seconds_entry.pack(side='left', padx=2)
            ttk.Label(timer_frame, text="s").pack(side='left', padx=(0,10))
            
            # Add bell toggle with compact spacing
            bell_frame = ttk.Frame(duration_frame)
            bell_frame.pack(side='left', padx=10)
            
            bell_toggle = ttk.Checkbutton(
                bell_frame,
                text="",
                variable=self.bell_states[i],
                style='Bell.TCheckbutton'
            )
            bell_toggle.pack(side='left', padx=5)

            # Add countdown display
            countdown_label = ttk.Label(duration_frame, textvariable=self.countdown_vars[i], width=8)
            countdown_label.pack(side='left', padx=5)
            
            # Store timer entries
            self.duration_timer_entries.append((minutes_entry, seconds_entry))
        
        # Instant Markers Section
        marker_section = ttk.LabelFrame(annotation_controls_frame, text="Instant Markers")
        marker_section.pack(fill='x', pady=5, padx=5)
        
        # Create 5 instant marker controls
        self.marker_entries = []
        for i in range(5):
            marker_frame = ttk.Frame(marker_section)
            marker_frame.pack(fill='x', pady=3)
            
            marker_button = ttk.Button(
                marker_frame,
                text=f"Marker {i+1}",
                command=lambda idx=i: self.add_instant_marker(idx)
            )
            marker_button.pack(side='left', padx=10)
            
            # Create entry with dropdown for history for markers
            marker_entry = ttk.Combobox(marker_frame, width=37)  # Back to original width
            marker_entry.pack(side='left', padx=10)  # Back to original padding
            if self.marker_history:
                marker_entry['values'] = self.marker_history
            marker_entry.set(self.last_marker_labels[i])
            self.marker_entries.append(marker_entry)
        
        # Configure grid
        content_frame.columnconfigure(1, weight=1)
        
        # Create menu bar
        self.create_menu()
        
        # Initialize COM ports
        self.update_com_ports()
        
        # Update channel info
        self.update_channel_info()
        
        # Refresh COM ports every 5 seconds
        self.root.after(5000, self.periodic_port_update)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_channel_info(self):
        """Update channel count and mode info"""
        mode = self.channel_mode_var.get()
        
        # Handle built-in configurations
        if mode in CHANNEL_CONFIGS:
            config = CHANNEL_CONFIGS[mode]
            channel_count = len(config["output_order"])
            self.channel_count_var.set(f"Channels: {channel_count}")
            self.mode_description_var.set(config["description"])
        # Handle custom configurations (strip "Custom: " prefix if present)
        elif mode.startswith("Custom: "):
            custom_mode = mode[8:]  # Remove "Custom: " prefix
            if custom_mode in self.custom_configs:
                config = self.custom_configs[custom_mode]
                channel_count = len(config["output_order"])
                self.channel_count_var.set(f"Channels: {channel_count}")
                
                # Update description if available
                if "description" in config:
                    self.mode_description_var.set(config["description"])
                else:
                    self.mode_description_var.set("")
        else:
            self.channel_count_var.set("Channels: --")
            self.mode_description_var.set("")
    
    def on_channel_mode_change(self, event=None):
        """Handle channel mode changes"""
        self.update_channel_info()
        self.save_settings()  # Save settings when headset type changes

    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Custom Configuration", command=self.configure_custom_headset)
        file_menu.add_command(label="Manage Configurations", command=self.manage_configurations)
        file_menu.add_separator()
        file_menu.add_command(label="Refresh COM Ports", command=self.update_com_ports)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_checkbutton(label="Monitor Electrodes", 
                                  variable=self.monitor_electrodes, 
                                  command=self.update_electrode_monitoring_setting)
        tools_menu.add_checkbutton(label="BDF Signal Monitor", 
                                  variable=self.monitor_bdf_signal,
                                  command=self.update_bdf_signal_monitoring_setting)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def update_electrode_monitoring_setting(self):
        """Save electrode monitoring preference to settings"""
        self.settings["monitor_electrodes"] = self.monitor_electrodes.get()
        self.save_settings()
        
    def update_bdf_signal_monitoring_setting(self):
        """Save BDF signal monitoring preference to settings"""
        self.settings["monitor_bdf_signal"] = self.monitor_bdf_signal.get()
        self.save_settings()
        
    def initialize_electrode_monitor(self):
        """Initialize the electrode monitor window"""
        # Close any existing window to prevent duplicates
        if self.electrode_monitor_window is not None:
            try:
                self.close_electrode_monitor()
            except:
                pass
            
        # Create a new electrode monitor window
        try:
            # Explicitly ensure matplotlib is in non-interactive mode
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.ioff()
            
            # Create the window
            self.electrode_monitor_window = electrode_monitor.ElectrodeMonitorWindow(
                master=self.root,
                title="EEG Electrode Status Monitor"
            )
            
            # Make sure it's visible
            self.electrode_monitor_window.root.attributes("-topmost", True)
            self.electrode_monitor_window.root.deiconify()
            self.electrode_monitor_window.root.update()
            
            print("Successfully created electrode monitor window")
            
            # Start the electrode check timer
            if self.electrode_check_timer is not None:
                self.root.after_cancel(self.electrode_check_timer)
                
            self.electrode_check_timer = self.root.after(
                self.electrode_check_interval, 
                self.check_electrode_connections
            )
        except Exception as e:
            print(f"Error initializing electrode monitor: {e}")
            import traceback
            traceback.print_exc()
    
    def check_electrode_connections_from_buffer(self, buffer, channel_names):
        """Check electrode connections using the buffer that was just written to BDF
        
        Args:
            buffer: The data buffer that was just written to the BDF file
            channel_names: List of channel names corresponding to the buffer rows
        """
        if self.electrode_monitor_window is None or not self.recording:
            return
            
        try:
            # The buffer already has the correct mapping (output channels)
            # So we can directly use it for connection check without any remapping
            self.connection_status = signal_detect.check_real_time_connection(
                data=buffer,
                channel_names=channel_names,
                window_size=512  # Use all samples in the buffer
            )
            
            # Perform FFT analysis to detect 50Hz noise first, but ensure we're using appropriate buffer size
            try:
                # Get the actual number of samples - use the minimum of what's available
                # This ensures we don't try to use more samples than are actually in the buffer
                available_samples = min(512, len(buffer[0]))
                
                # Only attempt FFT if we have enough samples
                if available_samples >= 250:  # Need enough samples for FFT
                    # Use a slice of the buffer with the appropriate size
                    buffer_slice = buffer[:, :available_samples]
                    threshold = 5.0  # Default threshold
                    self.electrode_monitor_window.update_fft_analysis(buffer_slice, channel_names, threshold)
            except Exception as e:
                print(f"Error in FFT analysis: {e}")
            
            # Then update the electrode monitor display with both connection status and line noise data
            self.electrode_monitor_window.update_electrode_display(self.connection_status)
                
        except Exception as e:
            print(f"Error checking electrode connections from buffer: {e}")
            import traceback
            traceback.print_exc()
    
    def check_electrode_connections(self):
        """Check electrode connections during recording and update the monitor"""
        if self.electrode_monitor_window is None or not self.recording:
            return
        
        try:
            # Get latest data from the board
            data = self.board.get_current_board_data(int(self.sample_rate))  
            
            # Get current configuration
            mode = self.channel_mode_var.get()
            if mode in CHANNEL_CONFIGS:
                config = CHANNEL_CONFIGS[mode]
            elif mode.startswith("Custom: "):
                custom_mode = mode[8:]  # Remove "Custom: " prefix
                if custom_mode in self.custom_configs:
                    config = self.custom_configs[custom_mode]
                else:
                    return
            else:
                return
                
            # Get channel mappings (same logic as in record_data)
            output_channels = config.get("output_order", [])
            input_order = config.get("input_order", [])
            
            # Check connections if we have enough data
            if data.shape[1] > 0 and len(output_channels) > 0:
                # Create proper channel mapping (same as in record_data)
                channel_mapping = {}  # Maps output index to input index
                for out_idx, out_channel in enumerate(output_channels):
                    if out_channel in input_order:
                        channel_mapping[out_idx] = input_order.index(out_channel)
                
                # Get EEG channel indices
                eeg_channels = self.board.get_eeg_channels(self.board.board_id)
                
                # Reorder the data according to the mapping
                mapped_data = np.zeros((len(output_channels), data.shape[1]))
                for out_idx, in_idx in channel_mapping.items():
                    if in_idx < len(eeg_channels):
                        mapped_data[out_idx] = data[eeg_channels[in_idx]]
                
                # Now use properly mapped data for connection check
                self.connection_status = signal_detect.check_real_time_connection(
                    data=mapped_data,
                    channel_names=output_channels,
                    window_size=10  # Check last 10 samples
                )
                
                # Update the electrode monitor display
                self.electrode_monitor_window.update_electrode_display(self.connection_status)
                
            # Schedule next check
            self.electrode_check_timer = self.root.after(
                self.electrode_check_interval, 
                self.check_electrode_connections
            )
            
        except Exception as e:
            print(f"Error checking electrode connections: {e}")
            traceback.print_exc()
            
    def close_electrode_monitor(self):
        """Close the electrode monitor window and cancel timers"""
        # Cancel the electrode check timer
        if self.electrode_check_timer is not None:
            self.root.after_cancel(self.electrode_check_timer)
            self.electrode_check_timer = None
            
        # Close the electrode monitor window
        if self.electrode_monitor_window is not None:
            try:
                # Close matplotlib figure properly first
                if hasattr(self.electrode_monitor_window, 'fig') and self.electrode_monitor_window.fig is not None:
                    import matplotlib.pyplot as plt
                    plt.close(self.electrode_monitor_window.fig)
                    
                # Temporarily allow the window to be closed
                self.electrode_monitor_window.root.protocol("WM_DELETE_WINDOW", self.electrode_monitor_window.root.destroy)
                self.electrode_monitor_window.root.destroy()
            except Exception as e:
                print(f"Error closing electrode monitor: {e}")
                
            self.electrode_monitor_window = None
            
            # Ensure all matplotlib figures are closed
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception as e:
                print(f"Error closing matplotlib figures: {e}")

    def initialize_bdf_signal_monitor(self):
        """Initialize the BDF signal monitor window"""
        # Close any existing window to prevent duplicates
        if self.bdf_signal_monitor is not None:
            try:
                self.close_bdf_signal_monitor()
            except:
                pass
        
        # Store current window state before signal monitor creation
        self._store_window_state()
        
        # Create a new BDF signal monitor
        try:
            # Create the monitor instance with no parent
            self.bdf_signal_monitor = BDFSignalMonitor(parent=None)
            
            # Start monitoring the BDF file
            self.bdf_signal_monitor.start_monitoring(self.output_file)
            
            # Schedule multiple restoration attempts to ensure window size is maintained
            self._schedule_window_restoration()
            
            print("Successfully created BDF signal monitor window")
        except Exception as e:
            print(f"Error initializing BDF signal monitor: {e}")
            import traceback
            traceback.print_exc()
    
    def _store_window_state(self):
        """Store current window state for later restoration"""
        # Save multiple properties to ensure complete restoration
        self._original_geometry = self.root.geometry()
        self._original_width = self.root.winfo_width()
        self._original_height = self.root.winfo_height()
        self._original_x = self.root.winfo_x()
        self._original_y = self.root.winfo_y()
        print(f"Stored window state: {self._original_geometry} ({self._original_width}x{self._original_height})")  
        
    def _schedule_window_restoration(self):
        """Schedule multiple restoration attempts with increasing delays"""
        # Try multiple times with increasing delays to ensure it works
        for delay in [100, 300, 700]:
            self.root.after(delay, self._restore_window_state)
            
    def _restore_window_state(self):
        """Restore the original window state"""
        # Force exact geometry
        self.root.geometry(self._original_geometry)
        
        # Double-check specific dimensions
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        
        if current_width != self._original_width or current_height != self._original_height:
            print(f"Window size mismatch, forcing exact dimensions: {self._original_width}x{self._original_height}")
            # Try explicit resizing if geometry wasn't enough
            self.root.geometry(f"{self._original_width}x{self._original_height}+{self._original_x}+{self._original_y}")
            
            # Update root to process geometry changes
            self.root.update_idletasks()
    
    def close_bdf_signal_monitor(self):
        """Close the BDF signal monitor window"""
        if self.bdf_signal_monitor is not None:
            try:
                # Stop monitoring
                self.bdf_signal_monitor.stop_monitoring()
                
                # Close the window
                self.bdf_signal_monitor.close_monitor()
            except Exception as e:
                print(f"Error closing BDF signal monitor: {e}")
                
            self.bdf_signal_monitor = None

    def update_com_ports(self):
        """Update the list of available COM ports"""
        # Get all active ports with their descriptions
        available_ports = list(serial.tools.list_ports.comports())
        
        # Find all Silicon Labs devices and USB Serial Devices
        eeg_ports = []
        for port in available_ports:
            if port.device:
                if "Silicon Labs CP210x USB to UART Bridge" in port.description:
                    eeg_ports.append((port, "EEG optical dongle"))
                elif "USB Serial Device" in port.description:
                    eeg_ports.append((port, "USB Serial Device"))
        
        if eeg_ports:
            # Show all compatible devices
            descriptions = []
            self.port_mapping = {}
            for port, device_type in eeg_ports:
                description = f"{port.device} - {device_type}"
                descriptions.append(description)
                self.port_mapping[description] = port.device
            
            self.port_combo['values'] = descriptions
            
            # If there's a saved port and it's still available, use it
            saved_port = self.settings.get("com_port")
            if saved_port and saved_port in descriptions:
                self.port_var.set(saved_port)
            elif saved_port:
                # If saved port exists but isn't in current list, try to match just the COM port number
                saved_com = saved_port.split(" - ")[0]  # Get just the COM port part
                for desc in descriptions:
                    if desc.startswith(saved_com):
                        self.port_var.set(desc)
                        break
                else:
                    # If no match found, use first available port
                    self.port_var.set(descriptions[0])
            else:
                # No saved port, use first available
                self.port_var.set(descriptions[0])
            
            self.status_var.set("Ready to record")  
            self.record_button.config(state="normal")
        else:
            # No compatible devices found
            self.port_combo['values'] = []
            self.port_mapping = {}
            self.port_var.set('')
            self.status_var.set("No compatible devices connected")  
            self.record_button.config(state="disabled")

    def periodic_port_update(self):
        """Periodically update COM ports if not recording"""
        if not self.recording:
            self.update_com_ports()
        self.root.after(5000, self.periodic_port_update)

    def show_about(self):
        messagebox.showinfo("About", "ACErecorder\nVersion 1.0\nA simple EEG recording application.")

    def generate_filename(self):
        """Generate filename with date-time prefix"""
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  
        suffix = self.filename_var.get().strip()
        if suffix:
            filename = f"{current_time}_{suffix}.bdf"
        else:
            filename = f"{current_time}.bdf"
        return os.path.join(self.eeg_dir, filename)

    def toggle_recording(self):
        if not self.recording:
            try:
                self.start_recording()
                self.record_button.config(text="Stop Recording")
                self.recording = True
            except Exception as e:
                self.recording = False
                self.status_var.set(f"Recording error: {str(e)}")  
                messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
        else:
            self.stop_recording()
            self.record_button.config(text="Start Recording")
            self.recording = False
            self.status_var.set("Ready to record")  

    def update_duration(self):
        if self.recording:
            duration = time.time() - self.recording_start_time
            self.recording_duration_var.set(f"{duration:.0f} seconds")  
            self.root.after(1000, self.update_duration)
        else:
            self.recording_duration_var.set("0 seconds")  

    def start_recording(self):
        if not isinstance(self.recording, bool):
            raise TypeError(f"Invalid recording state type: {type(self.recording).__name__}")
        
        if self.recording:
            return
        
        try:
            # Set critical timing FIRST
            self.recording_start_time = time.time()
            
            # Then update state
            self.recording = True
            
            # Get the actual COM port from the selection
            selected_port = self.port_mapping[self.port_var.get()]
            
            # Initialize board
            params = BrainFlowInputParams()
            params.serial_port = selected_port
            board_id = brainflow.BoardIds.FREEEEG32_BOARD.value
            self.board = BoardShim(board_id, params)
            
            # Prepare and start session
            self.board.prepare_session()
            self.board.start_stream()
            
            # Wait briefly and check if data is being received
            time.sleep(1)  # Wait for 1 second
            data = self.board.get_board_data()
            if data.size == 0:
                self.board.stop_stream()
                self.board.release_session()
                messagebox.showerror("Error", "No data received from the optical dongle.\nPlease check your connections and try again.")
                return
            
            # Get board info
            self.eeg_channels = self.board.get_eeg_channels(board_id)
            self.num_channels = len(self.eeg_channels)
            self.sample_rate = BoardShim.get_sampling_rate(board_id)
            
            # Update sample rate display
            self.sample_rate_var.set(f"Sample Rate: {self.sample_rate} Hz")
            
            # Channel labels
            self.channel_labels = ['O2', 'P8', 'A2', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'FC6', 'CP6', 
                                 'CP2', 'PO4', 'Pz', 'POz', 'FC2', 'AF4', 'Fz', 'Cz', 'FC1', 'AF3', 
                                 'FC5', 'CP5', 'CP1', 'PO3', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'P7', 'A1', 'F7']
            
            if len(self.channel_labels) < self.num_channels:
                self.channel_labels.extend([f'ch{i + 1}' for i in range(len(self.channel_labels), self.num_channels)])
            
            # Generate output filename
            self.output_file = self.generate_filename()
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self.record_data)
            self.record_thread.start()
            
            # Start duration counter
            self.update_duration()
            
            # Update UI
            self.status_var.set(f"Recording to {self.output_file}...")  
            self.filename_entry.config(state="disabled")
            self.port_combo.config(state="disabled")
            
            # Initialize electrode monitoring if enabled
            if self.monitor_electrodes.get():
                try:
                    self.initialize_electrode_monitor()
                    # Start electrode connection monitoring
                    self.check_electrode_connections()
                    print("Electrode monitoring started")
                except Exception as e:
                    print(f"Error initializing electrode monitor: {e}")
                    traceback.print_exc()
            
            # Initialize BDF signal monitor if enabled
            if self.monitor_bdf_signal.get():
                try:
                    self.initialize_bdf_signal_monitor()
                    print("BDF signal monitor started")
                except Exception as e:
                    print(f"Error initializing BDF signal monitor: {e}")
                    traceback.print_exc()
            
        except Exception as e:
            self.recording = False
            self.recording_start_time = None
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def record_data(self):
        """Record data from the board"""
        try:
            mode = self.channel_mode_var.get()
            
            # Get configuration based on mode
            if mode in CHANNEL_CONFIGS:
                config = CHANNEL_CONFIGS[mode]
            elif mode.startswith("Custom: "):
                custom_mode = mode[8:]  # Remove "Custom: " prefix
                if custom_mode in self.custom_configs:
                    config = self.custom_configs[custom_mode]
                else:
                    self.recording = False
                    messagebox.showerror("Error", f"Custom configuration not found: {custom_mode}")
                    return
            else:
                self.recording = False
                messagebox.showerror("Error", f"Invalid configuration mode: {mode}")
                return
                
            if "input_order" not in config or "output_order" not in config:
                self.recording = False
                messagebox.showerror("Error", f"Invalid configuration format for mode: {mode}")
                return
            
            output_channels = config["output_order"]
            input_order = config["input_order"]
            
            # Ensure we have output channels defined
            if not output_channels:
                self.recording = False
                messagebox.showerror("Error", "No channels configured")
                return
                
            # Get the input channels based on mode
            channel_mapping = {}  # Maps output index to input index
            for out_idx, out_channel in enumerate(output_channels):
                if out_channel in input_order:
                    channel_mapping[out_idx] = input_order.index(out_channel)
                    
            # Initialize buffer for EDF file
            buffer = np.zeros((len(output_channels), self.sample_rate))
            buffer_count = 0
            no_data_count = 0
            self.total_samples_written = 0  # Reset sample counter
            self.current_buffer_samples = 0  # Reset buffer position
            
            # Get indices of EEG channels from board
            self.eeg_channels = self.board.get_eeg_channels(self.board.board_id)
            
            # Create EDF file
            try:
                f = pyedflib.EdfWriter(self.output_file, len(output_channels), file_type=pyedflib.FILETYPE_BDFPLUS)
                
                # Set header fields
                f.setTechnician('X')
                f.setRecordingAdditional('X')
                f.setPatientName('X')
                f.setPatientCode('X')
                f.setPatientAdditional('X')
                f.setAdmincode('X')
                f.setEquipment('FreeEEG32')
                f.setSignalHeaders([{
                    'label': ch_name,
                    'dimension': 'uV',
                    'sample_rate': self.sample_rate,
                    'physical_min': -187500,
                    'physical_max': 187500,
                    'digital_min': -8388608,
                    'digital_max': 8388607,
                    'transducer': '',
                    'prefilter': ''
                } for ch_name in output_channels])
                
                # Flag to indicate we're in stopping state
                stopping = False
                
                while self.recording or (stopping and buffer_count < self.sample_rate):
                    try:
                        data = self.board.get_board_data()
                        new_samples = data.shape[1]
                        
                        # Update stopping state
                        if not self.recording and not stopping:
                            stopping = True
                            print("Completing current data record before stopping...")
                        
                        # Check if we're receiving data
                        if new_samples == 0:
                            no_data_count += 1
                            if no_data_count >= 100:  # After ~1 second of no data (assuming 10ms sleep)
                                self.recording = False
                                stopping = False  # Force stop if we've lost connection
                                messagebox.showerror("Error", "Lost connection with the optical dongle.\nRecording has been stopped.")
                                break
                            time.sleep(0.01)
                            continue
                        else:
                            no_data_count = 0  # Reset counter when data is received
                        
                        if buffer_count + new_samples >= self.sample_rate:
                            samples_to_fill = self.sample_rate - buffer_count
                            
                            # Map input channels to output channels
                            for out_idx, in_idx in channel_mapping.items():
                                if in_idx < len(self.eeg_channels):
                                    buffer[out_idx, buffer_count:] = data[self.eeg_channels[in_idx], :samples_to_fill]
                            
                            # Calculate coherence for 2 Channel mode
                            if mode == "2 Channel Headset":  # Fixed mode name
                                # Get Fp1 and Fp2 data
                                try:
                                    # Get indices for Fp1 and Fp2
                                    fp1_idx = output_channels.index("Fp1")
                                    fp2_idx = output_channels.index("Fp2")
                                    coherence_idx = output_channels.index("Coherence")
                                    
                                    # Prepare data for coherence calculation
                                    coherence_data = np.vstack((buffer[fp1_idx, :], buffer[fp2_idx, :]))
                                    
                                    # Calculate coherence
                                    try:
                                        coherence_value = compute_coherence(coherence_data, self.sample_rate)
                                        # Fill the coherence buffer with the computed value
                                        buffer[coherence_idx, :] = coherence_value
                                    except Exception as e:
                                        print(f"Coherence calculation error: {e}")
                                        traceback.print_exc()  # Add full traceback
                                        buffer[coherence_idx, :] = 0
                                except ValueError as e:
                                    print(f"Error getting channel indices: {e}")
                                    print(f"Available channels: {output_channels}")
                            else:
                                pass
                            
                            # Write the data
                            try:
                                f.writeSamples(buffer)
                                self.total_samples_written += self.sample_rate
                                print(f"Wrote {self.sample_rate} samples to file")
                                
                                # Check electrode connections using this buffer that was just written
                                # But only if electrode monitoring is enabled
                                if self.monitor_electrodes.get() and self.electrode_monitor_window is not None:
                                    # Pass the complete buffer to check electrodes using this data
                                    # This is much more efficient than getting new data from the board
                                    self.check_electrode_connections_from_buffer(buffer, output_channels)
                            except Exception as e:
                                print(f"Error writing samples: {e}")
                                traceback.print_exc()
                            
                            # Handle remaining data
                            if new_samples > samples_to_fill:
                                # Calculate the maximum amount we can copy (limited by buffer size)
                                max_copy_size = min(new_samples-samples_to_fill, buffer.shape[1])
                                
                                # Make sure we don't try to copy more data than is available
                                available_data_size = data.shape[1] - samples_to_fill
                                copy_size = min(max_copy_size, available_data_size)
                                
                                print(f"Buffer handling - Max buffer size: {buffer.shape[1]}, Data size: {data.shape[1]}, Copy size: {copy_size}")
                                
                                # Copy only what will fit in the buffer
                                for out_idx, in_idx in channel_mapping.items():
                                    if in_idx < len(self.eeg_channels):
                                        try:
                                            buffer[out_idx, :copy_size] = data[self.eeg_channels[in_idx], samples_to_fill:samples_to_fill+copy_size]
                                        except Exception as e:
                                            print(f"Error copying data to buffer: {e}")
                                            print(f"Buffer shape: {buffer.shape}, Data shape: {data.shape}, Out idx: {out_idx}, In idx: {in_idx}")
                                            print(f"Copy range - Buffer: 0:{copy_size}, Data: {samples_to_fill}:{samples_to_fill+copy_size}")
                                
                                buffer_count = copy_size
                                self.current_buffer_samples = buffer_count
                            else:
                                buffer_count = 0
                                self.current_buffer_samples = 0
                                buffer.fill(0)
                                
                                # If we were stopping and just wrote a complete buffer, we're done
                                if stopping:
                                    break
                        else:
                            # Map input channels to output channels
                            # Ensure we don't exceed buffer size
                            space_left = buffer.shape[1] - buffer_count
                            copy_size = min(new_samples, space_left)
                            
                            if copy_size > 0:
                                for out_idx, in_idx in channel_mapping.items():
                                    if in_idx < len(self.eeg_channels):
                                        try:
                                            buffer[out_idx, buffer_count:buffer_count+copy_size] = data[self.eeg_channels[in_idx], :copy_size]
                                        except Exception as e:
                                            print(f"Error copying data to buffer: {e}")
                                            print(f"Buffer shape: {buffer.shape}, Data shape: {data.shape}, Out idx: {out_idx}, In idx: {in_idx}")
                                            print(f"Copy range - Buffer: {buffer_count}:{buffer_count+copy_size}, Data: 0:{copy_size}")
                                
                                buffer_count += copy_size
                                self.current_buffer_samples = buffer_count
                                
                                if copy_size < new_samples:
                                    print(f"Warning: Had to trim {new_samples - copy_size} samples to fit in buffer")
                        
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"Recording error: {str(e)}")
                        traceback.print_exc()
                        messagebox.showerror("Error", f"Recording error: {str(e)}")
                        break
                
                # Write annotations to BDF file
                if self.annotations:
                    print("Writing annotations to BDF file...")
                    for ann in self.annotations:
                        try:
                            f.writeAnnotation(ann['onset'], ann['duration'], ann['description'])
                            print(f"Wrote annotation: {ann['description']} @ {ann['onset']:.3f}s (Duration: {ann['duration']:.3f}s)")
                        except Exception as e:
                            print(f"Error writing annotation: {e}")
                            traceback.print_exc()
                
                # Now that annotations are written, clear them for the next recording
                self.annotations = []
                
                # Close the file
                try:
                    f.close()
                    print("BDF file closed successfully")
                except Exception as e:
                    print(f"Error closing BDF file: {e}")
                    traceback.print_exc()
                
            except Exception as e:
                print(f"Error creating BDF file: {e}")
                traceback.print_exc()
                messagebox.showerror("Error", f"Error creating BDF file: {str(e)}")
                self.recording = False
                
        except Exception as e:
            print(f"Recording thread error: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Recording thread error: {str(e)}")
            self.recording = False

    def stop_recording(self):
        if not self.recording:
            return
            
        # Close electrode monitoring if it's active
        self.close_electrode_monitor()
        
        # Close BDF signal monitor if it's active
        self.close_bdf_signal_monitor()
            
        # Calculate end time for annotations (1 second before end of recording)
        annotation_end_time = (self.total_samples_written + self.current_buffer_samples - self.sample_rate) / self.sample_rate
        if annotation_end_time < 0:  # If recording is less than 1 second
            annotation_end_time = 0
            
        # Close any open duration annotations
        for i in range(5):
            if self.duration_vars[i].get():  # If checkbox is checked, annotation is active
                # Calculate end time based on total samples
                if self.duration_indices[i] is not None:
                    start_time = self.annotations[self.duration_indices[i]]['onset']
                    
                    # Ensure annotation doesn't end after the recording
                    end_time = min(annotation_end_time, (self.total_samples_written + self.current_buffer_samples) / self.sample_rate)
                    duration = end_time - start_time
                    
                    if duration <= 0:  # If annotation would have negative duration, remove it
                        self.annotations.pop(self.duration_indices[i])
                        print(f"=== DURATION {i+1} REMOVED: Too short to include ===")
                    else:
                        # Update the annotation
                        self.annotations[self.duration_indices[i]]['duration'] = duration
                        annotation_name = self.annotations[self.duration_indices[i]]['description']
                        print(f"=== DURATION {i+1} STOP: '{annotation_name}' @ {end_time:.3f}s (Duration: {duration:.3f}s) ===")
                    
                    # Reset the checkbox and timer
                    self.duration_vars[i].set(False)
                    if self.duration_timers[i]:
                        self.root.after_cancel(self.duration_timers[i])
                        self.duration_timers[i] = None
                    self.duration_indices[i] = None
                    
                    # Clear countdown if present
                    if hasattr(self, 'countdown_labels') and i < len(self.countdown_labels):
                        self.countdown_labels[i].config(text="")
        
        self.recording = False
        time.sleep(0.5)  # Give time for recording thread to finish
        
        try:
            # Clean up board resources
            if self.board:
                print("Cleaning up board resources...")
                try:
                    self.board.stop_stream()
                    print("Stream stopped")
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                
                try:
                    self.board.release_session()
                    print("Session released")
                except Exception as e:
                    print(f"Error releasing session: {e}")
                    
                self.board = None
                print("Board cleanup complete")
        except Exception as e:
            print(f"Error during board cleanup: {e}")
        
        # Update UI
        self.record_button.config(text="Start Recording")
        self.record_button.config(style="Green.TButton")
        
        # Enable configuration controls
        self.channel_mode_combo.config(state="normal")
        self.port_combo.config(state="readonly")
        self.filename_entry.config(state="normal")
        
        # Update status
        self.status_label.config(text="Recording stopped")
        
        # NOTE: The annotations list will be cleared after record_data() writes them to the BDF file

    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Update old configuration names
                    if settings.get("headset") == "2 Channel":
                        settings["headset"] = "2 Channel Headset"
                    
                    return settings
            return {}
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            return {}

    def save_settings(self):
        """Save settings to file"""
        try:
            # Create a copy of settings to modify
            settings_to_save = copy.deepcopy(self.settings)
            
            # Update the last used configuration
            current_headset = self.channel_mode_var.get()
            
            # Create an ordered dictionary with headset and com_port first
            ordered_settings = {
                "headset": current_headset,
                "com_port": settings_to_save.get("com_port", ""),
            }
            
            # Remove default configurations before adding to ordered settings
            if "configurations" in settings_to_save:
                default_configs = ["2 Channel Headset", "19 Channel", "32 Channel"]
                custom_configs = {
                    name: config 
                    for name, config in settings_to_save["configurations"].items() 
                    if name not in default_configs
                }
                ordered_settings["configurations"] = custom_configs

            # Add any remaining settings
            for key, value in settings_to_save.items():
                if key not in ["headset", "com_port", "configurations"]:
                    ordered_settings[key] = value

            with open(self.settings_file, 'w') as f:
                json.dump(ordered_settings, f, indent=4)

        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_custom_configs(self):
        """Load configurations from settings"""
        try:
            # Get all configurations (both default and custom)
            all_configs = self.settings.get("configurations", CHANNEL_CONFIGS.copy())
            
            # Update the channel configs dictionary
            CHANNEL_CONFIGS.clear()
            CHANNEL_CONFIGS.update(all_configs)
            
            # Update the combobox values
            if hasattr(self, 'channel_mode_combo'):
                current_value = self.channel_mode_var.get()
                self.channel_mode_combo['values'] = list(CHANNEL_CONFIGS.keys())
                if current_value in CHANNEL_CONFIGS:
                    self.channel_mode_var.set(current_value)
                else:
                    self.channel_mode_var.set("19 Channel")
        except Exception as e:
            print(f"Error loading configurations: {e}")

    def reload_configurations(self):
        """Reload configurations from settings file and refresh UI"""
        # Reload settings from file
        self.settings = self.load_settings()
        self.custom_configs = self.settings.get("configurations", {}).copy()
        
        # Update CHANNEL_CONFIGS with custom configurations
        # First remove existing custom configs
        keys_to_remove = [k for k in CHANNEL_CONFIGS.keys() if k.startswith("Custom: ")]
        for k in keys_to_remove:
            del CHANNEL_CONFIGS[k]
        
        # Add custom configs back from settings
        for name, config in self.custom_configs.items():
            CHANNEL_CONFIGS[f"Custom: {name}"] = {
                "input_order": INPUT_ORDER_32,
                "output_order": config["output_order"],
                "description": f"Custom Configuration: {name}"
            }
        
        # Refresh the channel mode menu
        self.refresh_channel_mode_menu()
        
        # Update channel info if current selection is still valid
        current_mode = self.channel_mode_var.get()
        if current_mode not in CHANNEL_CONFIGS:
            self.channel_mode_var.set("2 Channel Headset")
        self.update_channel_info()

    def refresh_channel_mode_menu(self):
        """Refresh the channel mode combo box with updated configurations"""
        current_mode = self.channel_mode_var.get()
        
        # Get standard configurations
        standard_configs = ["2 Channel Headset", "19 Channel", "32 Channel"]
        
        # Combine standard and custom configurations
        all_configs = standard_configs.copy()
        if self.custom_configs:
            all_configs.extend([f"Custom: {name}" for name in self.custom_configs.keys()])
        
        # Update combobox values
        self.channel_mode_combo['values'] = all_configs
        
        # Try to keep the current selection if it still exists
        if current_mode in all_configs:
            self.channel_mode_var.set(current_mode)
        else:
            self.channel_mode_var.set("19 Channel")  # Default to 19 Channel if current selection is invalid
        
        # Update channel info display
        self.update_channel_info()

    def manage_configurations(self):
        """Open dialog to manage custom configurations"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Custom Configurations")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Set dialog position relative to main window
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 500) // 2
        dialog.geometry(f"400x500+{x}+{y}")

        # Add description label
        ttk.Label(dialog, text="Manage your custom channel configurations:", padding=10).pack(fill='x')

        # Create listbox with scrollbar
        self.listbox_frame = ttk.Frame(dialog)
        self.listbox_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        scrollbar = ttk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.config_listbox = tk.Listbox(self.listbox_frame, selectmode='single', yscrollcommand=scrollbar.set)
        self.config_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.config_listbox.yview)

        # Filter out default configurations
        default_configs = ["2 Channel Headset", "19 Channel", "32 Channel"]
        custom_configs = [name for name in self.settings.get("configurations", {}).keys() 
                        if name not in default_configs]
        
        # Populate listbox with custom configurations only
        for config in sorted(custom_configs):
            self.config_listbox.insert('end', config)

        # Create button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=10)

        # Add buttons
        ttk.Button(button_frame, text="Edit", command=self.edit_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Duplicate", command=self.duplicate_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Rename", command=self.start_rename).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", command=lambda: self.on_dialog_close(dialog)).pack(side='left', padx=5)

        # Store dialog reference
        self.config_dialog = dialog
        
        # Add double-click binding to edit
        self.config_listbox.bind('<Double-Button-1>', lambda e: self.edit_config())

    def on_dialog_close(self, dialog):
        """Handle dialog close"""
        dialog.destroy()
        self.reload_configurations()

    def edit_config(self):
        """Edit the selected configuration"""
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to edit")
            return
        
        name = self.config_listbox.get(selection[0])
        config = self.settings["configurations"][name]
        
        # Store the configuration details we need
        edit_config_details = {
            'name': name,
            'channels': config["output_order"],
            'save_flags': config.get("save_flags", [])  # Use get() with default empty list
        }
        
        # Close the current dialog
        self.config_dialog.destroy()
        
        # Open the configuration dialog with the selected config
        self.configure_custom_headset(
            edit_name=edit_config_details['name'],
            edit_channels=edit_config_details['channels'],
            edit_save_flags=edit_config_details['save_flags']
        )

    def duplicate_config(self):
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to duplicate")
            return
        
        original_name = self.config_listbox.get(selection[0])
        new_name = original_name + " (Copy)"
        counter = 1
        
        # Find a unique name
        while new_name in self.custom_configs:
            counter += 1
            new_name = f"{original_name} (Copy {counter})"
        
        # Copy the configuration
        self.custom_configs[new_name] = {
            "output_order": self.custom_configs[original_name]["output_order"][:],
            "save_flags": self.custom_configs[original_name]["save_flags"][:]
        }
        
        # Update CHANNEL_CONFIGS
        CHANNEL_CONFIGS[f"Custom: {new_name}"] = {
            "input_order": INPUT_ORDER_32,
            "output_order": self.custom_configs[new_name]["output_order"],
            "description": f"Custom Configuration: {new_name}"
        }
        
        # Save settings
        self.settings["configurations"] = self.custom_configs
        self.save_settings()
        
        # Update listbox
        self.config_listbox.insert(tk.END, new_name)
        self.config_listbox.selection_clear(0, tk.END)
        self.config_listbox.selection_set(tk.END)
        
        messagebox.showinfo("Success", f"Configuration duplicated as '{new_name}'")

    def delete_config(self):
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to delete")
            return
        
        name = self.config_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", 
                             f"Are you sure you want to delete the configuration '{name}'?"):
            # Remove from custom_configs and CHANNEL_CONFIGS
            del self.custom_configs[name]
            del CHANNEL_CONFIGS[f"Custom: {name}"]
            
            # Update listbox
            self.config_listbox.delete(selection[0])
            
            # Save settings
            self.save_settings()

    def start_rename(self):
        """Start renaming the selected configuration"""
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to rename")
            return
        
        # Get the selected item's position and text
        idx = selection[0]
        old_name = self.config_listbox.get(idx)
        
        # Position and show the entry widget
        bbox = self.config_listbox.bbox(idx)
        if bbox:
            x, y, width, height = bbox
            rename_entry = ttk.Entry(self.listbox_frame)
            rename_entry.place(x=x, y=y, width=width, height=height)
            rename_entry.delete(0, tk.END)
            rename_entry.insert(0, old_name)
            rename_entry.select_range(0, tk.END)
            rename_entry.focus_set()
            
            # Store the original name and index for later use
            rename_entry.old_name = old_name
            rename_entry.item_index = idx

            def finish_rename(event=None):
                if not rename_entry.winfo_ismapped():
                    return
                
                new_name = rename_entry.get().strip()
                old_name = rename_entry.old_name
                idx = rename_entry.item_index
                
                # Hide the entry widget
                rename_entry.place_forget()
                
                # If name is empty or unchanged, do nothing
                if not new_name or new_name == old_name:
                    return
                
                # Check if new name already exists
                if new_name in self.settings["configurations"]:
                    messagebox.showerror("Error", "A configuration with this name already exists")
                    return
                
                # Update configurations
                config = self.settings["configurations"].pop(old_name)
                self.settings["configurations"][new_name] = config
                
                # Update listbox
                self.config_listbox.delete(idx)
                self.config_listbox.insert(idx, new_name)
                self.config_listbox.selection_set(idx)
                
                # Save settings
                self.save_settings()

            def cancel_rename(event=None):
                rename_entry.place_forget()

            # Bind entry events
            rename_entry.bind('<Return>', finish_rename)
            rename_entry.bind('<Escape>', cancel_rename)
            rename_entry.bind('<FocusOut>', finish_rename)

    def configure_custom_headset(self, edit_name=None, edit_channels=None, edit_save_flags=None):
        """Open dialog to configure custom headset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Custom Headset")
        dialog.geometry("800x750")
        dialog.transient(self.root)
        dialog.grab_set()

        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add name field
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(name_frame, text="Configuration Name:").pack(side=tk.LEFT)
        name_var = tk.StringVar(value=edit_name if edit_name else "")
        name_entry = ttk.Entry(name_frame, textvariable=name_var)
        name_entry.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)

        # Add description
        ttk.Label(main_frame, text="Enter channel labels and select which channels to save:", wraplength=780).pack(pady=(0,10))

        # Create frame for channel blocks
        blocks_frame = ttk.Frame(main_frame)
        blocks_frame.pack(fill=tk.BOTH, expand=True)

        # Load existing custom configuration
        custom_channels = edit_channels if edit_channels else [""] * 32
        custom_channels.extend([""] * (32 - len(custom_channels)))
        custom_channels = custom_channels[:32]

        # Create lists to store entries and checkboxes
        channel_entries = []
        channel_save_vars = []

        # Create 4 blocks of 8 channels each
        for block in range(4):
            # Create frame for this block
            block_frame = ttk.LabelFrame(blocks_frame, text=f"Channels {block*8 + 1}-{block*8 + 8}")
            block_frame.grid(row=block//2, column=block%2, padx=10, pady=5, sticky="nsew")

            # Create header row with "Save to BDF" label
            header_frame = ttk.Frame(block_frame)
            header_frame.pack(fill=tk.X, pady=(2,5), padx=5)
            ttk.Label(header_frame, text="Channel", width=12).pack(side=tk.LEFT)
            ttk.Label(header_frame, text="Label", width=25).pack(side=tk.LEFT, padx=(5,10))
            ttk.Label(header_frame, text="Save to BDF").pack(side=tk.LEFT)

            # Create channels in this block
            for i in range(8):
                channel_num = block * 8 + i
                row_frame = ttk.Frame(block_frame)
                row_frame.pack(fill=tk.X, pady=2, padx=5)

                # Channel number label
                ttk.Label(row_frame, text=f"Channel {channel_num + 1}:", width=12).pack(side=tk.LEFT)

                # Entry field
                entry = ttk.Entry(row_frame, width=20)
                entry.pack(side=tk.LEFT, padx=(5, 10))
                entry.insert(0, custom_channels[channel_num])
                channel_entries.append(entry)

                # Save checkbox (without text)
                save_var = tk.BooleanVar(value=bool(custom_channels[channel_num].strip()))
                channel_save_vars.append(save_var)
                save_cb = ttk.Checkbutton(row_frame, variable=save_var)
                save_cb.pack(side=tk.LEFT)
                
                # Bind entry changes to update checkbox
                def on_entry_change(event, entry=entry, save_var=save_var):
                    has_content = bool(entry.get().strip())
                    save_var.set(has_content)
                
                entry.bind('<KeyRelease>', on_entry_change)

        # Configure grid weights
        blocks_frame.columnconfigure(0, weight=1)
        blocks_frame.columnconfigure(1, weight=1)

        def save_custom_config():
            """Save the custom configuration"""
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name for the configuration")
                return
            
            # Get channel entries
            channels = []
            save_flags = []
            for entry, save_var in zip(channel_entries, channel_save_vars):
                channel = entry.get().strip()
                if channel:  # Only add non-empty channels
                    channels.append(channel)
                    save_flags.append(save_var.get())
            
            if not channels:
                messagebox.showerror("Error", "Please enter at least one channel")
                return
            
            # Create configuration
            config = {
                "channels": channels,
                "save_flags": save_flags,
                "input_order": channels.copy(),  # Use same channels as input order
                "output_order": channels.copy()  # Use same channels as output order
            }
            
            # Save to custom configs
            self.custom_configs[name] = config
            self.save_settings()
            
            # Update UI
            self.refresh_channel_mode_menu()
            dialog.destroy()  # Use the local dialog variable instead of self.dialog
            messagebox.showinfo("Success", "Custom channel configuration saved")

        def on_dialog_close():
            dialog.destroy()
            self.reload_configurations()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Add buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        def on_manage_click():
            dialog.destroy()
            self.manage_configurations()

        # Add Save and Manage buttons
        ttk.Button(buttons_frame, text="Save Configuration", command=save_custom_config).pack(side=tk.RIGHT)
        ttk.Button(buttons_frame, text="Manage Configurations", command=on_manage_click).pack(side=tk.LEFT)

    def quit_app(self):
        """Clean up and close the application"""
        if self.recording:
            self.stop_recording()
        self.save_settings()
        self.root.quit()

    def on_port_change(self):
        """Save selected COM port to settings"""
        selected_port = self.port_var.get()
        if selected_port:
            self.settings["com_port"] = selected_port
            self.save_settings()

    def add_instant_marker(self, index):
        """Add an instant marker annotation with no duration"""
        if not self.recording:
            messagebox.showerror("Error", "Start recording first")
            return
            
        annotation_name = self.marker_entries[index].get().strip()
        if not annotation_name:  
            messagebox.showerror("Error", "Annotation name cannot be empty")
            return
        
        # Calculate onset based on samples written plus current buffer position
        onset = (self.total_samples_written + self.current_buffer_samples) / self.sample_rate
        self.annotations.append({
            'onset': onset,
            'duration': 0.0,
            'description': annotation_name
        })
        
        # Add to history if not already present and not a default name
        if annotation_name not in self.marker_history and not annotation_name.startswith("Marker"):
            self.marker_history.append(annotation_name)
            self.save_settings()
            
        # Update last used label
        self.last_marker_labels[index] = annotation_name
        
        # Update entry dropdown
        self.marker_entries[index]['values'] = self.marker_history
        
        print(f"=== INSTANT MARKER {index+1}: '{annotation_name}' @ {onset:.3f}s ===")

    def toggle_duration_annotation(self, index):
        """Toggle a duration annotation on/off"""
        if not self.recording:
            self.duration_vars[index].set(False)  # Reset checkbox
            messagebox.showerror("Error", "Start recording first")
            return
            
        if self.duration_vars[index].get():  # Starting duration
            annotation_name = self.duration_entries[index].get().strip()
            if not annotation_name:  
                messagebox.showerror("Error", "Annotation name cannot be empty")
                self.duration_vars[index].set(False)  # Reset checkbox
                return
                
            # Get timer values if set
            minutes = self.duration_timer_entries[index][0].get().strip()
            seconds = self.duration_timer_entries[index][1].get().strip()
            
            # Calculate timer duration if values are set
            timer_duration = None
            if minutes or seconds:
                try:
                    minutes = int(minutes) if minutes else 0
                    seconds = int(seconds) if seconds else 0
                    if minutes < 0 or seconds < 0 or seconds >= 60:
                        raise ValueError("Invalid time values")
                    timer_duration = minutes * 60 + seconds
                except ValueError:
                    messagebox.showerror("Error", "Invalid timer values")
                    self.duration_vars[index].set(False)
                    return
            
            # Calculate onset based on total samples
            start_time = (self.total_samples_written + self.current_buffer_samples) / self.sample_rate
            
            # Add to annotations list with 0 duration (will be updated when stopped)
            self.annotations.append({
                'onset': start_time,
                'duration': 0.0,
                'description': annotation_name
            })
            
            # Add to history if not already present and not a default name
            if annotation_name not in self.duration_history and not annotation_name.startswith("Duration"):
                self.duration_history.append(annotation_name)
                self.save_settings()
                
            # Update last used label
            self.last_duration_labels[index] = annotation_name
            
            # Update entry dropdown
            self.duration_entries[index]['values'] = self.duration_history
            
            # Store index of this annotation
            self.duration_indices[index] = len(self.annotations) - 1
            
            # Start timer if duration was specified
            if timer_duration is not None:
                end_time = time.time() + timer_duration
                self.update_countdown(index, end_time)
                self.duration_timers[index] = self.root.after(timer_duration * 1000, 
                                                            lambda idx=index: self.auto_stop_duration(idx))
                print(f"=== DURATION {index+1} START: {annotation_name} @ {start_time:.3f}s (Timer: {minutes}m {seconds}s) ===")
            else:
                print(f"=== DURATION {index+1} START: {annotation_name} @ {start_time:.3f}s ===")
        else:  # Stopping duration
            self.stop_duration(index)

    def update_countdown(self, index, end_time):
        """Update the countdown display for a duration annotation"""
        if self.duration_vars[index].get():  # Only if still running
            now = time.time()
            remaining = end_time - now
            
            if remaining > 0:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.countdown_vars[index].set(f"{minutes:02d}:{seconds:02d}")
                
                # Schedule next update
                self.countdown_timers[index] = self.root.after(1000, 
                    lambda: self.update_countdown(index, end_time))
            else:
                self.countdown_vars[index].set("")
                self.countdown_timers[index] = None

    def auto_stop_duration(self, index):
        """Automatically stop a duration annotation when its timer expires"""
        if self.duration_vars[index].get():  # Only if still running
            # First call stop_duration to properly close the annotation
            self.stop_duration(index)
            # Then update the checkbox state
            self.duration_vars[index].set(False)
            self.duration_timers[index] = None
            
            # Play bell sound if enabled for this duration
            if self.bell_states[index].get():
                self.play_bell_sound()

    def play_bell_sound(self):
        """Play a simple bell sound using winsound"""
        try:
            winsound.Beep(1000, 100)  # 1000Hz for 100ms
            time.sleep(0.1)  # Small pause between beeps
            winsound.Beep(1000, 100)  # Second beep
        except:
            print("Could not play bell sound")

    def stop_duration(self, index):
        """Stop a duration annotation (called from toggle or auto-stop)"""
        if self.duration_indices[index] is not None and self.duration_indices[index] < len(self.annotations):
            # Calculate final duration and print status
            now = time.time() - self.recording_start_time
            duration_annotation = self.annotations[self.duration_indices[index]]
            duration_annotation['duration'] = now - duration_annotation['onset']
            print(f"=== DURATION {index+1} END: {duration_annotation['description']} "
                  f"({duration_annotation['duration']:.1f}s) @ {now:.1f}s ===")
            self.duration_indices[index] = None
            
            # Cancel any existing timers
            if self.duration_timers[index]:
                self.root.after_cancel(self.duration_timers[index])
                self.duration_timers[index] = None
            if self.countdown_timers[index]:
                self.root.after_cancel(self.countdown_timers[index])
                self.countdown_timers[index] = None
            self.countdown_vars[index].set("")  # Clear countdown display
        else:
            print(f"Warning: No active duration annotation to end for Duration {index+1}")
            self.duration_vars[index].set(False)

    def on_closing(self):
        """Handle window closing event"""
        if self.recording:
            if messagebox.askokcancel("Quit", "Recording in progress. Stop recording and quit?"):
                self.stop_recording()
                self.quit_app()
        else:
            self.quit_app()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ACErecorder(root)
        root.mainloop()
    except Exception as e:
        import traceback
        error_message = f"Error starting application:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        # Create a basic error window
        error_window = tk.Tk()
        error_window.title("Error")
        error_window.geometry("600x400")
        
        # Add error text
        text_widget = tk.Text(error_window, wrap=tk.WORD)
        text_widget.insert(tk.END, error_message)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Add close button
        close_button = tk.Button(error_window, text="Close", command=error_window.destroy)
        close_button.pack(pady=10)
        
        # Keep window open
        error_window.mainloop()