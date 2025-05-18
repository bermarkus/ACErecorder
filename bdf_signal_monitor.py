import os
import time
import threading
import numpy as np
import queue
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import mne
from mne.io import read_raw_bdf

# Import the signal processor and settings menu
from eeg_signal_processor import EEGSignalProcessor
from eeg_settings_menu import EEGSettingsMenu

class BDFSignalMonitor:
    """A real-time monitoring tool for BDF EEG recordings"""
    
    def __init__(self, parent=None, window_length=10, update_interval=100, live_update=True, signal_processor=None):
        """Initialize the BDF signal monitor window"""
        # Don't store parent reference to avoid any connection between windows
        self.parent = None
        self.window = None
        self.plot_widget = None
        self.plots = []
        self.channel_labels = []
        self.update_interval = 1000  # 1Hz update rate in milliseconds
        self.window_length = window_length  # For locking during updates
        self.lock = threading.RLock()
        
        # Signal processor for filtering and re-referencing
        if signal_processor is None:
            self.signal_processor = EEGSignalProcessor()
        else:
            self.signal_processor = signal_processor
        
        # For storing data (10s window by default)
        self.window_length = window_length
        self.data_queue = []
        
        # Threading and monitoring control
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        
        # BDF file tracking
        self.current_bdf_file = None
        self.last_file_size = 0
        self.last_check_time = 0
        
        # Data display settings
        self.num_channels = 0
        self.channel_offsets = []  # For stacking channels vertically
        self.channel_scales = []   # Individual scaling for each channel
        self.channel_spacing_multipliers = []  # Multipliers for vertical spacing (default 1.0)
        self.y_scale = 30.0  # Default scale in µV (30µV is default)
        self.auto_scale = False  # Default to fixed scaling (not auto-scaling)
        self._auto_scale_value = 1.0  # Separate value for auto-scaling tracking
        self.per_channel_scale = True  # Scale each channel individually
        
        # Create settings menu (initially hidden)
        self.settings_menu = None
        
        # Make sure the application exists - create new instance to isolate from Tkinter
        # Get existing instance but don't use it to avoid Tkinter interaction
        existing_app = QtWidgets.QApplication.instance()
        
        # Only create a new instance if one doesn't exist
        if not existing_app:
            self.app = QtWidgets.QApplication([])
        else:
            self.app = existing_app

    def create_window(self):
        """Create the signal monitor window if it doesn't exist"""
        if self.window is None:
            # Create main window
            self.window = QtWidgets.QMainWindow()
            self.window.setWindowTitle("BDF Signal Monitor")
            self.window.setGeometry(100, 100, 1200, 800)  # Default size
            
            # Set window to stay on top
            self.window.setWindowFlags(
                QtCore.Qt.Window | 
                QtCore.Qt.WindowStaysOnTopHint
            )
            
            # Apply stylesheet to minimize borders in the application
            self.window.setStyleSheet("""
                QMainWindow { background-color: #404040; }
                QMainWindow::separator { width: 0; height: 0; }
                QGraphicsView { border: none; background: #404040; }
                QFrame { border: none; }
                QWidget { background-color: #404040; }
            """)
            
            # Create central widget with minimal layout
            central_widget = QtWidgets.QWidget()
            central_widget.setContentsMargins(0, 0, 0, 0)
            self.window.setCentralWidget(central_widget)
            main_layout = QtWidgets.QVBoxLayout(central_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # Create main widget for the grid layout (plots + settings menu)
            main_widget = QtWidgets.QWidget()
            self.main_grid = QtWidgets.QGridLayout(main_widget)
            self.main_grid.setContentsMargins(0, 0, 0, 0)
            self.main_grid.setSpacing(0)
            main_layout.addWidget(main_widget)
            
            # Setup plot container first (without settings panel to start)
            self.plot_container = QtWidgets.QWidget()
            plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            self.main_grid.addWidget(self.plot_container, 0, 1, 1, 1)  # Takes column 1
            
            # Store references to create UI elements after window is shown
            self.settings_panel = None
            self.settings_icon = None
            
            # Set up PyQtGraph
            pg.setConfigOptions(antialias=False)  # Disable antialiasing for better performance
            
            # Create the plot widget with dark background
            self.plot_widget = pg.PlotWidget(background='#404040')
            self.plot_widget.setBackground('#404040')  # Dark gray background
            plot_layout.addWidget(self.plot_widget)
            
            # CRITICAL: Directly modify the QGraphicsView to remove frame border
            # This is the core issue - PyQtGraph has a built-in frame border that needs to be removed
            graphics_view = self.plot_widget.plotItem.getViewBox().parentItem().getViewWidget()
            graphics_view.setFrameShape(QtWidgets.QFrame.NoFrame)
            graphics_view.setLineWidth(0)
            graphics_view.setContentsMargins(0, 0, 0, 0)
            
            # Disable default grid - we'll add custom grid lines
            self.plot_widget.showGrid(x=False, y=False)
            
            # Create darker gray pen for the grid lines
            self.grid_pen = pg.mkPen(color='#606060', width=1)
            
            # Completely remove all border and margin elements except bottom axis
            for side in ['top', 'right', 'left']:
                self.plot_widget.showAxis(side, False)  # Hide these axes completely
                
            # Apply border removal stylesheet to the whole widget
            self.plot_widget.setStyleSheet("""                
                QGraphicsView { border: 0px; padding: 0px; margin: 0px; }
                QGraphicsScene { border: 0px; padding: 0px; margin: 0px; }
                QFrame { border: 0px; padding: 0px; margin: 0px; }
                QMainWindow { border: 0px; padding: 0px; margin: 0px; }
                QWidget { border: 0px; padding: 0px; margin: 0px; }
            """)
            
            # Add plot widget to main layout
            main_layout.addWidget(self.plot_widget)
            
            # Hide all axes completely
            for side in ['left', 'bottom', 'right', 'top']:
                self.plot_widget.showAxis(side, False)
            
            # Set plot to take up full width by removing all margins and borders
            self.plot_widget.getPlotItem().getViewBox().setDefaultPadding(0)  # Remove padding
            self.plot_widget.setContentsMargins(0, 0, 0, 0)  # Remove content margins
            self.plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)  # Remove plot margins
            
            # Remove all spacing around the plot area
            self.plot_widget.centralWidget.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.centralWidget.layout.setSpacing(0)
            
            # Further eliminate borders and margins on the view box
            view_box = self.plot_widget.getPlotItem().getViewBox()
            view_box.setMouseEnabled(x=True, y=True)  # Keep mouse interactions
            view_box.border = pg.mkPen(color=None)  # No border pen
            
            # Hide the axes completely and remove their space allocation
            self.plot_widget.getPlotItem().showAxis('bottom', False)
            self.plot_widget.getPlotItem().getAxis('bottom').setHeight(0)
            
            # Ensure the window itself doesn't add borders
            if self.window:
                self.window.setContentsMargins(0, 0, 0, 0)
                self.window.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
                self.window.centralWidget().layout().setSpacing(0)
            
            # Remove title completely
            self.plot_widget.setTitle("")
            
            # Set axis colors
            axis_pen = pg.mkPen(color='#808080', width=1)
            self.plot_widget.getAxis('bottom').setPen(axis_pen)
            self.plot_widget.getAxis('left').setPen(axis_pen)
            self.plot_widget.getAxis('bottom').setTextPen('white')
            self.plot_widget.getAxis('left').setTextPen('white')
            
            # Hide the left axis line
            self.plot_widget.getAxis('left').setStyle(tickLength=0)
            
            # Remove padding and margins to make the plot fill the entire space
            self.plot_widget.plotItem.vb.border = pg.mkPen(None)  # No border
            self.plot_widget.getPlotItem().getViewBox().setDefaultPadding(0)
            
            # Configure bottom axis to have zero-width border and no tick marks
            self.plot_widget.getAxis('bottom').setPen(pg.mkPen(None))  # Invisible axis line
            self.plot_widget.getAxis('bottom').setStyle(tickLength=0)  # No tick marks
            
            # Make the entire plot expand to edges
            self.plot_widget.getPlotItem().setMenuEnabled(False)  # Disable right-click menu
            self.plot_widget.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
            
            # Make the ViewBox completely borderless
            self.plot_widget.getPlotItem().getViewBox().setBorder(pen=None)
            
            # Additional styling to remove any remaining borders
            self.plot_widget.setFrameStyle(QtWidgets.QFrame.NoFrame)
            self.plot_widget.viewport().setStyleSheet("border: 0px; padding: 0px; margin: 0px;")
            
            # Add a settings button in the top-left corner
            self.settings_button = QtWidgets.QPushButton("⚙", self.window)
            self.settings_button.setStyleSheet(
                "QPushButton {background-color: rgba(60, 60, 60, 150); color: white; " +
                "border: none; border-radius: 30px; font-size: 32px; padding: 5px;}" +
                "QPushButton:hover {background-color: rgba(80, 80, 80, 200);}"
            )
            self.settings_button.setFixedSize(60, 60)  # 2x larger
            self.settings_button.setToolTip("Open Signal Processing Settings")
            self.settings_button.clicked.connect(self.toggle_settings_menu)
            
            # Position will be set after window is shown
            self.settings_button.hide()
            
            # Handle close event - redirect to minimize
            self.window.closeEvent = self.handle_close_event
            
            # Use a completely separate window process approach
            # Initially hide the window
            self.window.hide()
            
            # Use a more controlled approach to showing the window
            # Delay display to avoid interference with the main window
            QtCore.QTimer.singleShot(1000, self._show_window_safely)
            
            print("BDF Signal Monitor window created and will display shortly")
            
    def handle_close_event(self, event):
        """Handle window close event - redirect to minimize"""
        self.minimize_window()
        event.ignore()
            
    def minimize_window(self):
        """Minimize the window instead of closing it"""
        if self.window:
            self.window.showMinimized()
    
    def _show_window_safely(self):
        """Show the window with precautions to avoid affecting Tkinter"""
        if self.window:
            # Make sure we're completely detached from any parent window
            self.window.setParent(None)
            
            # Process any pending events before showing
            self.app.processEvents()
            
            # Window init completed
            
            # Show in maximized mode (not fullscreen)
            self.window.showMaximized()
            self.fullscreen_mode = False
            
            # Make sure it's visible and on top
            self.window.raise_()
            self.window.activateWindow()
            
            # Initialize the settings menu
            self.init_settings_menu()
            
            # Position the settings button in the top-left corner
            def position_button():
                # Position in top-left with padding
                padding = 15
                self.settings_button.move(
                    padding,
                    padding
                )
                self.settings_button.show()
                self.settings_button.raise_()
                
            # Delay to ensure window is fully shown
            QtCore.QTimer.singleShot(500, position_button)
            
            print("BDF Signal Monitor now displayed in maximized mode")

    def init_settings_menu(self):
        """Initialize the settings menu"""
        if self.settings_menu is None:
            try:
                # Create the settings menu as a separate window with reference to this monitor
                self.settings_menu = EEGSettingsMenu(self.signal_processor, bdf_monitor=self)
                
                # Connect signals
                self.settings_menu.settingsChanged.connect(self.update_title)
                self.settings_menu.settingsChanged.connect(self.update_channel_labels)
                
                print("Settings menu initialized successfully")
            except Exception as e:
                print(f"Error initializing settings menu: {e}")
    
    def toggle_settings_menu(self):
        """Toggle visibility of the settings menu"""
        if self.settings_menu is None:
            self.init_settings_menu()
            
        if self.settings_menu.isVisible():
            self.settings_menu.hide()
        else:
            # Position to the right of the button
            button_pos = self.settings_button.mapToGlobal(QtCore.QPoint(0, 0))
            x = button_pos.x() + self.settings_button.width() + 5  # Position to the right
            y = button_pos.y()  # Same vertical position as the button
            self.settings_menu.move(x, y)
            self.settings_menu.show()
            self.settings_menu.raise_()
        

            

        

            
    def update_title(self):
        """Update window title with current settings"""
        title = "EEG Signal Monitor"
        if self.current_bdf_file:
            title += f" - {os.path.basename(self.current_bdf_file)}"
        
        # Add filter information
        filter_info = []
        if self.signal_processor.bandpass_enabled:
            filter_info.append(f"BP: {self.signal_processor.bandpass_low}-{self.signal_processor.bandpass_high}Hz")
        if self.signal_processor.notch_enabled:
            filter_info.append(f"Notch: {self.signal_processor.notch_freq}Hz")
        if self.signal_processor.reference_mode != 'original':
            filter_info.append(f"Ref: {self.signal_processor.reference_mode}")
            
        if filter_info:
            title += f" [{', '.join(filter_info)}]"
            
        self.window.setWindowTitle(title)
    
    def close_monitor(self):
        """Force close the monitor window"""
        if self.window:
            # Allow window to close by temporarily removing closeEvent handler
            self.window.closeEvent = lambda e: e.accept()
            self.window.close()
            self.window = None
            self.plot_widget = None
            self.plots = []
    
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
            
        # Start the UI update timer
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(100)  # Start with shorter interval for faster first update
        
        # Force window to be displayed again
        self.window.show()
        self.window.raise_()
    
    def stop_monitoring(self):
        """Stop monitoring the BDF file"""
        print("Stopping BDF monitor")
        self.stop_flag.set()
        
        # Stop the timer
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        
        # Clear the data queue
        self.data_queue.clear()
                
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
                                
                            # Always update channel information from the BDF file
                            self.num_channels = len(raw.ch_names)
                            self.original_channel_labels = raw.ch_names.copy()
                            
                            # Get display labels based on reference mode
                            if self.signal_processor:
                                self.channel_labels = self.signal_processor.get_channel_display_labels(self.original_channel_labels)
                            else:
                                self.channel_labels = self.original_channel_labels
                                
                            print(f"Channel labels from BDF: {self.channel_labels}")
                            

                            # Initialize channel spacing multipliers if needed
                            if len(self.channel_spacing_multipliers) != self.num_channels:
                                self.channel_spacing_multipliers = np.ones(self.num_channels)
                            
                            # Create channel offsets - REVERSED so first channel appears at the TOP
                            base_offset = 2.0
                            # Base channel spacing depends on number of channels
                            base_channel_spacing = base_offset * (2 if self.num_channels <= 30 else 1.5)
                            
                            # Calculate cumulative spacing based on multipliers
                            # This ensures channels with higher multipliers get more space
                            # First reverse the array so channel 0 is at the top
                            reversed_multipliers = self.channel_spacing_multipliers[::-1]
                            
                            # Calculate cumulative sum of multipliers (from bottom to top)
                            cumulative_multipliers = np.cumsum(reversed_multipliers)
                            
                            # Normalize by total to maintain overall scale
                            normalized_positions = cumulative_multipliers / np.sum(reversed_multipliers) * (self.num_channels)
                            
                            # Adjust to start from the top
                            self.channel_offsets = normalized_positions[::-1] * base_channel_spacing
                            
                            # Print some debug info about the data
                            data_min = np.min(data)
                            data_max = np.max(data)
                            data_range = data_max - data_min
                            
                            print(f"Read BDF data: shape={data.shape}, time points={len(times)}, sample rate={raw.info['sfreq']}")
                            print(f"Value range: min={data_min:.2f}, max={data_max:.2f}, range={data_range:.2f}")
                            
                            # Auto-adjust scaling value ONLY if auto_scale is enabled
                            # This value is used in update_plot to determine display scaling
                            if self.auto_scale and data_range > 0:
                                # Calculate a new suggested scale based on data range
                                suggested_scale = 2.0 / max(data_range, 0.0001)
                                
                                # Only update if the suggested scale is significantly different
                                # (between 0.1x and 10x the current scale)
                                scale_ratio = suggested_scale / self.y_scale
                                if 0.1 <= scale_ratio <= 10.0:
                                    self.y_scale = suggested_scale
                                    print(f"Auto-adjusted y_scale to {self.y_scale:.8f} based on data range {data_range:.8f}")
                                # If we're in auto mode but not updating, still store the scale for reference
                                self._auto_scale_value = suggested_scale
                            
                            # Put the data in the queue/list for the UI thread to consume
                            # Limit queue size to avoid backlog
                            if len(self.data_queue) > 5:
                                # Keep only the most recent items
                                self.data_queue = self.data_queue[-5:]
                            
                            # Add new data to the list
                            self.data_queue.append((data, times, raw.info['sfreq']))
                            print("Data successfully added to queue, current size:", len(self.data_queue))
                            
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
                                self.data_queue.append((data, times, raw.info['sfreq']))
                                print("Data successfully added to queue, current size:", len(self.data_queue))
                            else:
                                raise e
                        
                    except Exception as e:
                        print(f"Error reading BDF file: {e}")
                
                # Sleep to avoid tight loop
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(1.0)
    
    def update_channel_labels(self):
        """Update channel labels based on current reference mode"""
        if hasattr(self, 'original_channel_labels') and self.original_channel_labels:
            if self.signal_processor:
                self.channel_labels = self.signal_processor.get_channel_display_labels(self.original_channel_labels)
            else:
                self.channel_labels = self.original_channel_labels.copy()
            
            # Force a redraw of the plot to show updated labels
            self.update_plot(force_labels=True)
            
    def set_channel_spacing_multiplier(self, channel_name, multiplier):
        """Set a custom spacing multiplier for a specific channel
        
        Args:
            channel_name: Name of the channel to adjust spacing for
            multiplier: Spacing multiplier (1.0 is default, >1.0 increases space, <1.0 decreases)
        """
        if not hasattr(self, 'original_channel_labels') or not self.original_channel_labels:
            print("Cannot set multiplier: channel labels not initialized")
            return False
            
        if channel_name in self.original_channel_labels:
            channel_idx = self.original_channel_labels.index(channel_name)
            if 0 <= channel_idx < len(self.channel_spacing_multipliers):
                self.channel_spacing_multipliers[channel_idx] = float(multiplier)
                print(f"Set spacing multiplier for {channel_name} to {multiplier}")
                # Update the display to reflect the new spacing
                self.update_plot(force_labels=True)
                return True
                
        print(f"Channel {channel_name} not found")
        return False
        
    def set_channel_group_spacing_multipliers(self, channel_pattern, multiplier):
        """Set spacing multipliers for all channels matching a pattern
        
        Args:
            channel_pattern: String pattern to match channel names (case insensitive)
            multiplier: Spacing multiplier (1.0 is default, >1.0 increases space, <1.0 decreases)
        """
        if not hasattr(self, 'original_channel_labels') or not self.original_channel_labels:
            print("Cannot set multipliers: channel labels not initialized")
            return 0
            
        count = 0
        for i, channel in enumerate(self.original_channel_labels):
            if channel_pattern.lower() in channel.lower():
                if i < len(self.channel_spacing_multipliers):
                    self.channel_spacing_multipliers[i] = float(multiplier)
                    count += 1
                    
        print(f"Set spacing multiplier for {count} channels matching '{channel_pattern}'")
        if count > 0:
            # Update the display to reflect the new spacing
            self.update_plot(force_labels=True)
        return count
        
    def reset_channel_spacing_multipliers(self):
        """Reset all channel spacing multipliers to default (1.0)"""
        self.channel_spacing_multipliers = np.ones(self.num_channels)
        print("Reset all channel spacing multipliers to 1.0")
        # Update the display to reflect the reset spacing
        self.update_plot(force_labels=True)
        return True
    
    def update_plot(self, force_labels=False):
        """Update the visual display of the monitor"""
        # Skip update if no data available and not forcing label update
        if len(self.data_queue) == 0 and not force_labels:
            return
            
        try:
            print(f"Checking data queue (items: {len(self.data_queue)})")
            # Get the latest data from the queue
            if len(self.data_queue) > 0:
                try:
                    # Get the latest item but keep it in the queue
                    data, times, sfreq = self.data_queue[-1]
                    print(f"Updating plot with data shape: {data.shape}, time points: {len(times)}")
                    
                    # Update signal processor sample rate
                    self.signal_processor.set_sample_rate(sfreq)
                    
                except Exception as e:
                    print(f"Error accessing data from queue: {e}")
                    return
                
                if data.shape[1] < 2:
                    print("Not enough data points to plot yet")
                    return
                    
                # Apply signal processing if any processing is enabled
                if (self.signal_processor.bandpass_enabled or 
                    self.signal_processor.notch_enabled or
                    self.signal_processor.reference_mode != 'original'):
                    try:
                        # Get processing info for debug message
                        processing_info = []
                        if self.signal_processor.bandpass_enabled:
                            processing_info.append(f"bandpass {self.signal_processor.bandpass_low}-{self.signal_processor.bandpass_high} Hz")
                        if self.signal_processor.notch_enabled:
                            processing_info.append(f"notch {self.signal_processor.notch_freq} Hz")
                        if self.signal_processor.reference_mode != 'original':
                            processing_info.append(f"{self.signal_processor.reference_mode} reference")
                            
                        # Apply all enabled processing
                        print(f"Applying signal processing: {', '.join(processing_info)}")
                        
                        # We must pass the original channel names, not the display labels with -LE suffix
                        if hasattr(self, 'original_channel_labels'):
                            data = self.signal_processor.process(data, self.original_channel_labels)
                        else:
                            data = self.signal_processor.process(data, self.channel_labels)
                            
                        print(f"Processed data shape: {data.shape}")
                    except Exception as e:
                        print(f"Error applying signal processing: {e}")
                
                # Clear previous plots
                self.plot_widget.clear()
                self.plots = []
                
                # Determine scale factor based on scaling mode
                max_amp = np.max(np.abs(data))
                
                # Data values are likely in millivolts but we want to display in microvolts
                # Need to apply a conversion from mV to µV (x1000)
                conversion_factor = 1000.0  # Convert mV to µV
                
                if self.auto_scale:
                    # Auto scaling: adapt to data amplitude
                    if max_amp > 0:
                        # Use 80% of available channel space
                        scale_factor = 0.8 / max_amp
                        print(f"Auto scaling with factor: {scale_factor:.8f} for max_amp: {max_amp:.8f}")
                    else:
                        # Default scale if no amplitude
                        scale_factor = 0.01
                else:
                    # Fixed scale mode: use the selected µV value
                    # If data is in millivolts but scale is in microvolts, we need to adjust
                    # For example: 30µV scale, data point of 0.03mV = 30µV should be 1.0 unit high
                    # So we multiply by conversion_factor and then divide by scale value
                    scale_factor = conversion_factor / self.y_scale
                    print(f"Fixed scaling at {self.y_scale} µV with factor: {scale_factor:.8f} (data in mV)")
                    
                    # Ensure scale factor isn't too small to see anything
                    if scale_factor < 1.0:
                        print("Warning: Scale factor may be too small - signals might appear flat")
                
                
                # Calculate channel offsets, with spacing proportional to the number of channels
                # Use a fixed offset that doesn't depend on y_scale to avoid rescaling issues
                base_offset = 2.0  # Fixed offset between channels regardless of scale
                
                if self.num_channels > 30:
                    channel_spacing = base_offset * 1.5
                else:
                    channel_spacing = base_offset * 2
                
                # IMPORTANT: Reverse the channel order so first channel is at top, last at bottom
                self.channel_offsets = (self.num_channels - 1 - np.arange(self.num_channels)) * channel_spacing
                print(f"Channel spacing: {channel_spacing}, Using {'auto' if self.auto_scale else 'fixed'} scale: {self.y_scale:.8f} µV")
                
                # Update or initialize per-channel scaling factors
                if len(self.channel_scales) != data.shape[0]:
                    # Initialize with uniform scaling
                    self.channel_scales = [scale_factor] * data.shape[0]
                
                # Calculate individual channel scales ONLY IF in auto-scale mode AND per-channel scaling enabled
                if self.auto_scale and self.per_channel_scale:
                    for i in range(data.shape[0]):
                        # Get amplitude of this channel
                        channel_max = np.max(np.abs(data[i]))
                        if channel_max > 0:
                            # Aim for a normalized height of 0.8 units per channel
                            target_scale = 0.8 / channel_max
                            # Smooth the scale change to avoid abrupt changes
                            self.channel_scales[i] = self.channel_scales[i] * 0.7 + target_scale * 0.3
                elif not self.auto_scale:
                    # In fixed scale mode, ensure all channels use the same fixed scale
                    for i in range(data.shape[0]):
                        self.channel_scales[i] = scale_factor
                    
                    # Print some debug about the channel scales
                    print(f"Channel scales - min: {min(self.channel_scales):.4f}, max: {max(self.channel_scales):.4f}")
                
                # Create yellow pen for all plots
                yellow_pen = pg.mkPen(color='#ffff00', width=1)
                
                # Plots were already created above
                
                # Add channel labels as text items on the right side
                # First clear any existing label items
                for item in self.plot_widget.getPlotItem().items:
                    if hasattr(item, 'isChannelLabel') and item.isChannelLabel:
                        self.plot_widget.getPlotItem().removeItem(item)
                
                # Calculate right edge position
                right_edge = times[-1]
                
                # Determine which channels to show labels for
                label_indices = []
                if self.num_channels > 30:
                    # For many channels, only label every 5th channel
                    label_indices = list(range(0, self.num_channels, 5))
                else:
                    # For fewer channels, show all labels
                    label_indices = list(range(self.num_channels))
                
                # Add text items for labels on right side
                for i in label_indices:
                    if i < data.shape[0]:
                        label = pg.TextItem(
                            text=self.channel_labels[i],
                            color='white',
                            anchor=(0, 0.5)  # Center vertically, left-aligned horizontally
                        )
                        label.isChannelLabel = True  # Custom attribute to identify these items
                        label.setPos(right_edge, self.channel_offsets[i])
                        self.plot_widget.addItem(label)
                
                # Set time axis to show a fixed scale from -10 to 0 seconds
                if len(times) > 0:
                    # Calculate precise start and end times for data display
                    end_time = times[-1]
                    start_time = end_time - self.window_length
                    
                    # Set exact limits for the data
                    self.plot_widget.setXRange(start_time, end_time)
                    
                    # Create ticks with fixed labels from -10 to 0
                    x_ticks = []
                    # Create major ticks at each second
                    for i in range(-self.window_length, 1):
                        # Map the relative time (-10 to 0) to actual data time
                        actual_time = end_time + i  # Maps -10 to end_time-10, -9 to end_time-9, etc.
                        x_ticks.append((actual_time, str(i)))  # Shows -10, -9, ..., -1, 0
                    
                    # Clear the plot for fresh drawing
                    self.plot_widget.clear()  # Clear all previous items 
                    self.plots = []  # Reset plots list
                    
                    # Add vertical grid lines first (so they appear behind the data)
                    for i in range(-self.window_length, 1):
                        grid_line = pg.InfiniteLine(
                            pos=end_time + i,
                            angle=90,
                            pen=self.grid_pen
                        )
                        self.plot_widget.addItem(grid_line)
                        
                    # Update ticks
                    self.plot_widget.getAxis('bottom').setTicks([x_ticks])
                    
                    # Update the bottom axis label to clarify the time scale
                    self.plot_widget.setLabel('bottom', 'Time (seconds ago)', color='white')
                    
                    # Now plot all channels
                    # Create yellow pen for all plots
                    yellow_pen = pg.mkPen(color='#ffff00', width=1)
                    
                    for i in range(min(self.num_channels, data.shape[0])):
                        # Apply individual channel scaling if enabled
                        if self.per_channel_scale:
                            channel_scale = self.channel_scales[i]
                        else:
                            channel_scale = scale_factor
                            
                        # Apply channel scaling to the data
                        scaled_data = data[i] * channel_scale
                        
                        # When using fixed scale, we want to ensure each channel is plotted relative to its own zero line
                        # Each channel gets positioned at its vertical offset position on the screen
                        # This keeps proper calibrated scaling for each channel
                        plot = self.plot_widget.plot(
                            times, 
                            scaled_data + self.channel_offsets[i],  # Add vertical offset for stacking channels
                            pen=yellow_pen,
                            name=self.channel_labels[i]
                        )
                        self.plots.append(plot)
                    
                    # Add channel labels as text items on the right side
                    # Determine which channels to show labels for
                    label_indices = []
                    if self.num_channels > 30:
                        # For many channels, only label every 5th channel
                        label_indices = list(range(0, self.num_channels, 5))
                    else:
                        # For fewer channels, show all labels
                        label_indices = list(range(self.num_channels))
                    
                    # Add text items for labels on left side only
                    # Define label colors and styles - use same yellow as the signal lines
                    label_color = '#ffff00'  # Yellow to match the signal lines
                    background_color = (50, 50, 50, 200)  # Dark gray with opacity (R,G,B,A)
                    
                    # Position labels at the left edge of the display
                    left_edge = start_time + 0.01  # Very close to left edge (1% of window)
                    
                    for i in label_indices:
                        if i < data.shape[0]:
                            # Create left side label only
                            left_label = pg.TextItem(
                                text=self.channel_labels[i],
                                color=label_color,
                                anchor=(0, 0.5),  # Left-aligned, vertically centered
                                fill=background_color  # Add opaque background
                            )
                            left_label.isChannelLabel = True  # Custom attribute to identify these items
                            left_label.setPos(left_edge, self.channel_offsets[i])
                            self.plot_widget.addItem(left_label)
                
                # Title has been removed as requested
                # Set an empty title to remove the header space completely
                self.plot_widget.setTitle("")
                
                # Grid lines are added explicitly as InfiniteLines
                
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Always reschedule the next update with appropriate interval
        if hasattr(self, 'update_timer'):
            # Adjust update interval based on whether we have data
            if len(self.data_queue) == 0 and self.num_channels == 0:
                # Use shorter interval if we haven't received any data yet
                self.update_timer.setInterval(200)  # Try more frequently until we get data
            else:
                # Use normal interval once we're displaying data
                self.update_timer.setInterval(self.update_interval)
            
    def _setup_initial_display(self):
        """Setup initial display with empty grid"""
        duration = self.window_length  # 10 seconds
        
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
            
        # Create empty channel labels if not already set
        if not self.channel_labels or len(self.channel_labels) != self.num_channels:
            self.channel_labels = [f"CH{i+1}" for i in range(self.num_channels)]
            
        # Create channel offsets - REVERSED so first channel is at the top
        base_offset = 2.0
        channel_spacing = base_offset * (2 if self.num_channels <= 30 else 1.5)
        self.channel_offsets = (self.num_channels - 1 - np.arange(self.num_channels)) * channel_spacing
        
        # Create yellow pen for all plots
        yellow_pen = pg.mkPen(color='#ffff00', width=1)
        
        # Clear previous plots
        self.plot_widget.clear()
        self.plots = []
        
        # Set fixed tick spacing from -10 to 0 seconds
        x_ticks = []
        for i in range(-self.window_length, 1):
            # For initial display, map the display time directly
            x_ticks.append((duration + i - self.window_length, str(i)))
        
        # Add explicit vertical grid lines at each second mark first
        for i in range(-self.window_length, 1):
            grid_line = pg.InfiniteLine(
                pos=duration + i - self.window_length,
                angle=90,
                pen=pg.mkPen(color='#606060', width=1)
            )
            self.plot_widget.addItem(grid_line)
            
        # Create empty plots for each channel
        for i in range(self.num_channels):
            plot = self.plot_widget.plot(
                [], [],  # Empty data initially
                pen=yellow_pen,
                name=self.channel_labels[i]
            )
            self.plots.append(plot)
        
        # Add channel labels as text items on the left side
        label_indices = []
        if self.num_channels > 30:
            # For many channels, only label every 5th channel
            label_indices = list(range(0, self.num_channels, 5))
        else:
            # For fewer channels, show all labels
            label_indices = list(range(self.num_channels))
            
        # Define label colors and styles - use same yellow as the signal lines
        label_color = '#ffff00'  # Yellow to match the signal lines
        background_color = (50, 50, 50, 200)  # Dark gray with opacity (R,G,B,A)
        
        # Position labels at the very left edge
        left_edge = duration - self.window_length + 0.01  # Very close to left edge
        
        for i in label_indices:
            label = pg.TextItem(
                text=self.channel_labels[i],
                color=label_color,
                anchor=(0, 0.5),  # Left-aligned, vertically centered
                fill=background_color  # Add opaque background
            )
            label.isChannelLabel = True  # Custom attribute to identify these items
            label.setPos(left_edge, self.channel_offsets[i])
            self.plot_widget.addItem(label)
            
        # Configure bottom axis with ticks but no border
        self.plot_widget.getAxis('bottom').setTicks([x_ticks])
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(None))  # Invisible axis line
        
        # Update the bottom axis label
        self.plot_widget.setLabel('bottom', 'Time (seconds ago)', color='white')
        
        # Update title
        if self.current_bdf_file:
            title = f"BDF Signal Monitor - {os.path.basename(self.current_bdf_file)}"
        else:
            title = "BDF Signal Monitor - Waiting for data"
            
        self.plot_widget.setTitle(title, color='white', size='14pt')
        # Set time axis range with fixed -10 to 0 display
        self.plot_widget.setXRange(duration - self.window_length, duration)


# For testing
if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent appearance
    
    # Create a simple test window
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Test BDF Monitor")
    main_window.setGeometry(100, 100, 300, 200)
    
    # Add a central widget with a button
    central = QtWidgets.QWidget()
    main_window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    
    def test_monitor():
        monitor = BDFSignalMonitor()
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG files")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a test BDF file path
        test_bdf = os.path.join(test_dir, "test_monitor_23ch.bdf")
        
        # If the file doesn't exist, create a placeholder file
        if not os.path.exists(test_bdf):
            # Create a minimal placeholder file so the monitor has something to look for
            with open(test_bdf, 'wb') as f:
                # Write a simple header (just enough bytes to pass the size check)
                f.write(b'\0' * 2048)
        
        # Start monitoring the test file
        monitor.start_monitoring(test_bdf)
    
    # Add a button to start the test
    test_btn = QtWidgets.QPushButton("Start Test")
    test_btn.clicked.connect(test_monitor)
    layout.addWidget(test_btn)
    
    main_window.show()
    sys.exit(app.exec_())
