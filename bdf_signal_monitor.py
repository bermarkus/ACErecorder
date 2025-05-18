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

# Import the signal processor
from eeg_signal_processor import EEGSignalProcessor

class BDFSignalMonitor:
    def __init__(self, parent=None):
        """Initialize the BDF signal monitor window"""
        # Don't store parent reference to avoid any connection between windows
        self.parent = None
        self.window = None
        self.plot_widget = None
        self.plots = []
        self.channel_labels = []
        self.update_interval = 1000  # 1Hz update rate in milliseconds
        self.window_length = 10  # 10 seconds of data
        self.fullscreen_mode = False
        
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
        
        # Initialize signal processor
        self.signal_processor = EEGSignalProcessor()
        self.keyboard_shortcuts = {}
        self.keyboard_shortcuts_enabled = True
        
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
            
            # Create central widget and layout
            central_widget = QtWidgets.QWidget()
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
            
            # Create the plot widget with dark background and no border
            self.plot_widget = pg.PlotWidget(background='#404040')
            self.plot_widget.setBackground('#404040')  # Dark gray background
            plot_layout.addWidget(self.plot_widget)
            
            # Disable default grid - we'll add custom grid lines
            self.plot_widget.showGrid(x=False, y=False)
            
            # Create darker gray pen for the grid lines
            self.grid_pen = pg.mkPen(color='#606060', width=1)
            
            # Completely remove all border and margin elements except bottom axis
            for side in ['top', 'right', 'left']:
                self.plot_widget.showAxis(side, False)  # Hide these axes completely
                
            # Apply border removal stylesheet to the whole widget
            self.plot_widget.setStyleSheet("""                
                QGraphicsView { border: 0px; }
                QGraphicsScene { border: 0px; }
                QFrame { border: 0px; }
            """)
            
            # Add plot widget to main layout
            main_layout.addWidget(self.plot_widget)
            
            # Configure appearance - no left axis for channel labels
            self.plot_widget.setLabel('bottom', 'Time', 'seconds', color='white')
            self.plot_widget.getAxis('left').setStyle(showValues=False)  # Hide values on left axis
            
            # Set plot to take up full width by removing all margins and borders
            self.plot_widget.getPlotItem().getViewBox().setDefaultPadding(0)  # Remove padding
            self.plot_widget.setContentsMargins(0, 0, 0, 0)  # Remove content margins
            self.plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)  # Remove plot margins
            
            # Remove all spacing around the plot area
            self.plot_widget.centralWidget.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.centralWidget.layout.setSpacing(0)
            
            # Initial title
            self.plot_widget.setTitle("EEG Signal Monitor (Last 10 seconds)", color='white', size='14pt')
            
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
            
            # Handle close event - redirect to minimize
            self.window.closeEvent = self.handle_close_event
            
            # Setup keyboard shortcuts
            self.setup_keyboard_shortcuts()
            
            # Schedule initialization will happen in _show_window_safely
            
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
            
            # Setup keyboard event handlers
            self.window.keyPressEvent = self.handle_key_press
            
            # Show in maximized mode (not fullscreen)
            self.window.showMaximized()
            self.fullscreen_mode = False
            
            # Make sure it's visible and on top
            self.window.raise_()
            self.window.activateWindow()
            
            # Setup UI elements after showing the window
            QtCore.QTimer.singleShot(500, self.setup_settings_ui)
            
            print("BDF Signal Monitor now displayed in maximized mode")

    def setup_settings_ui(self):
        """Setup the settings panel and icon after the window is shown"""
        if not self.window:
            return
            
        try:
            # Create settings menu container (initially hidden)
            self.settings_panel = self.create_settings_panel()
            self.settings_panel.setVisible(False)
            
            # Add the settings panel to column 0
            self.main_grid.addWidget(self.settings_panel, 0, 0, 1, 1)
            
            # Adjust grid column stretches (initially settings panel hidden)
            self.main_grid.setColumnStretch(0, 0)  # Settings panel (0% when hidden)
            self.main_grid.setColumnStretch(1, 1)  # Plot area (100% when settings hidden)
            
            # Create settings icon after a delay to ensure window is ready
            QtCore.QTimer.singleShot(300, self.create_settings_button)
            print("Settings panel successfully initialized")
        except Exception as e:
            print(f"Error setting up settings UI: {e}")
    
    def create_settings_button(self):
        """Create the settings button after a delay to ensure window is available"""
        try:
            self.settings_icon = self.create_settings_icon()
            print("Settings button created successfully")
        except Exception as e:
            print(f"Error creating settings button: {e}")
            
    def create_settings_panel(self):
        """Create the settings panel with filter controls"""
        # Create panel widget with dark background
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(
            "background-color: rgba(40, 40, 40, 210); " +
            "color: white; " +
            "border: 1px solid #555;"
        )
        panel.setMinimumWidth(300)  # Set minimum width to ensure readability
        
        # Create layout for the panel
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Add header
        header = QtWidgets.QLabel("Signal Processing")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Add separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #555;")
        layout.addWidget(line)
        
        # Create section for bandpass filter
        bp_group = QtWidgets.QGroupBox("Bandpass Filter")
        bp_group.setStyleSheet("color: white; background-color: rgba(50, 50, 50, 180);")
        bp_layout = QtWidgets.QVBoxLayout(bp_group)
        
        # Bandpass toggle
        self.bp_checkbox = QtWidgets.QCheckBox("Enable Bandpass Filter")
        self.bp_checkbox.setChecked(self.signal_processor.bandpass_enabled)
        self.bp_checkbox.toggled.connect(self.toggle_bandpass_filter)
        bp_layout.addWidget(self.bp_checkbox)
        
        # Bandpass frequency range
        freq_layout = QtWidgets.QHBoxLayout()
        freq_layout.addWidget(QtWidgets.QLabel("Range:"))
        
        self.bp_low_input = QtWidgets.QDoubleSpinBox()
        self.bp_low_input.setRange(0.1, 100)
        self.bp_low_input.setValue(self.signal_processor.bandpass_low)
        self.bp_low_input.setDecimals(1)
        self.bp_low_input.setSuffix(" Hz")
        self.bp_low_input.valueChanged.connect(self.update_bandpass_range)
        freq_layout.addWidget(self.bp_low_input)
        
        freq_layout.addWidget(QtWidgets.QLabel("to"))
        
        self.bp_high_input = QtWidgets.QDoubleSpinBox()
        self.bp_high_input.setRange(1, 200)
        self.bp_high_input.setValue(self.signal_processor.bandpass_high)
        self.bp_high_input.setDecimals(1)
        self.bp_high_input.setSuffix(" Hz")
        self.bp_high_input.valueChanged.connect(self.update_bandpass_range)
        freq_layout.addWidget(self.bp_high_input)
        
        bp_layout.addLayout(freq_layout)
        layout.addWidget(bp_group)
        
        # Create section for notch filter
        notch_group = QtWidgets.QGroupBox("Notch Filter")
        notch_group.setStyleSheet("color: white; background-color: rgba(50, 50, 50, 180);")
        notch_layout = QtWidgets.QVBoxLayout(notch_group)
        
        # Notch toggle
        self.notch_checkbox = QtWidgets.QCheckBox("Enable Notch Filter")
        self.notch_checkbox.setChecked(self.signal_processor.notch_enabled)
        self.notch_checkbox.toggled.connect(self.toggle_notch_filter)
        notch_layout.addWidget(self.notch_checkbox)
        
        # Notch frequency selection
        notch_freq_layout = QtWidgets.QHBoxLayout()
        notch_freq_layout.addWidget(QtWidgets.QLabel("Frequency:"))
        
        self.notch_freq_input = QtWidgets.QDoubleSpinBox()
        self.notch_freq_input.setRange(10, 200)
        self.notch_freq_input.setValue(self.signal_processor.notch_freq)
        self.notch_freq_input.setDecimals(1)
        self.notch_freq_input.setSuffix(" Hz")
        self.notch_freq_input.valueChanged.connect(self.update_notch_freq)
        notch_freq_layout.addWidget(self.notch_freq_input)
        
        # Quick selection buttons for common frequencies
        hz50_btn = QtWidgets.QPushButton("50Hz")
        hz50_btn.setStyleSheet("background-color: #444; min-height: 20px; max-height: 25px;")
        hz50_btn.clicked.connect(lambda: self.set_notch_freq(50))
        notch_freq_layout.addWidget(hz50_btn)
        
        hz60_btn = QtWidgets.QPushButton("60Hz")
        hz60_btn.setStyleSheet("background-color: #444; min-height: 20px; max-height: 25px;")
        hz60_btn.clicked.connect(lambda: self.set_notch_freq(60))
        notch_freq_layout.addWidget(hz60_btn)
        
        notch_layout.addLayout(notch_freq_layout)
        layout.addWidget(notch_group)
        
        # Add spacer at the bottom to push everything up
        spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        layout.addItem(spacer)
        
        # Add close button
        close_btn = QtWidgets.QPushButton("Close Settings")
        close_btn.setStyleSheet("background-color: #555; height: 30px;")
        close_btn.clicked.connect(self.toggle_settings_panel)
        layout.addWidget(close_btn)
        
        return panel
        
    def create_settings_icon(self):
        """Create a translucent settings icon in the top-left corner"""
        try:
            # Create a pushbutton that will float on top of the window
            icon = QtWidgets.QPushButton(self.window)
            icon.setToolTip("Signal Processing Settings")
            
            # Use a simple text icon instead of system icon for better compatibility
            icon.setText("⚙")
            icon.setFont(QtGui.QFont("Arial", 12))
            
            # Style the button to be translucent
            icon.setStyleSheet(
                "QPushButton {background-color: rgba(60, 60, 60, 150); color: white; border: none; " +
                "border-radius: 15px; padding: 5px;}" +
                "QPushButton:hover {background-color: rgba(80, 80, 80, 200);}"
            )
            icon.setFixedSize(30, 30)
            
            # Position in top-left corner
            icon.move(15, 15)
            icon.raise_()
            icon.show()
            
            # Connect click event
            icon.clicked.connect(self.toggle_settings_panel)
            
            return icon
        except Exception as e:
            print(f"Error creating settings icon: {e}")
            return None
        
    def toggle_settings_panel(self):
        """Toggle visibility of the settings panel"""
        if not self.settings_panel:
            print("Settings panel not available yet")
            return
            
        currently_visible = self.settings_panel.isVisible()
        self.settings_panel.setVisible(not currently_visible)
        
        # Adjust the grid layout column sizes
        if not currently_visible:  # Panel becoming visible
            self.main_grid.setColumnStretch(0, 1)  # Settings panel (25%)
            self.main_grid.setColumnStretch(1, 3)  # Plot area (75%)
        else:  # Panel becoming hidden
            self.main_grid.setColumnStretch(0, 0)  # Settings panel (0%)
            self.main_grid.setColumnStretch(1, 1)  # Plot area (100%)
    
    def toggle_bandpass_filter(self, checked):
        """Toggle bandpass filter based on checkbox"""
        self.signal_processor.toggle_bandpass(checked)
        self.update_title()
        print(f"Bandpass filter {'enabled' if checked else 'disabled'}")
        
    def update_bandpass_range(self):
        """Update bandpass filter frequency range"""
        low = self.bp_low_input.value()
        high = self.bp_high_input.value()
        
        # Ensure high is greater than low
        if high <= low:
            high = low + 1
            self.bp_high_input.setValue(high)
            
        self.signal_processor.set_bandpass_range(low, high)
        self.update_title()
        print(f"Bandpass range updated: {low}-{high} Hz")
        
    def toggle_notch_filter(self, checked):
        """Toggle notch filter based on checkbox"""
        if hasattr(self.signal_processor, 'toggle_notch'):
            self.signal_processor.toggle_notch(checked)
            self.update_title()
            print(f"Notch filter {'enabled' if checked else 'disabled'}")
            
    def update_notch_freq(self):
        """Update notch filter frequency"""
        freq = self.notch_freq_input.value()
        if hasattr(self.signal_processor, 'set_notch_freq'):
            self.signal_processor.set_notch_freq(freq)
            self.update_title()
            print(f"Notch frequency updated: {freq} Hz")
            
    def set_notch_freq(self, freq):
        """Set notch frequency from quick buttons"""
        self.notch_freq_input.setValue(freq)
        # update_notch_freq will be called by the valueChanged signal
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.fullscreen_mode:
            # Return to maximized mode (not normal size)
            self.window.showMaximized()
            self.fullscreen_mode = False
        else:
            self.window.showFullScreen()
            self.fullscreen_mode = True
            
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the signal monitor"""
        # F1: Toggle bandpass filter
        self.keyboard_shortcuts['F1'] = lambda: self.toggle_filter('bandpass')
        
    def handle_key_press(self, event):
        """Handle keyboard shortcuts"""
        if not self.keyboard_shortcuts_enabled:
            return
            
        # Get the key that was pressed
        key = event.key()
        key_text = QtCore.Qt.Key(key).name
        
        # Check if this key has a shortcut assigned
        if key_text in self.keyboard_shortcuts:
            self.keyboard_shortcuts[key_text]()
            return True
            
        # Pass the event to the parent handler if no shortcut was found
        return super(QtWidgets.QMainWindow, self.window).keyPressEvent(event)
        
    def toggle_filter(self, filter_type):
        """Toggle a specific filter using keyboard shortcut"""
        if filter_type == 'bandpass':
            enabled = self.signal_processor.toggle_bandpass()
            # Update checkbox if settings panel exists
            if hasattr(self, 'bp_checkbox') and self.bp_checkbox:
                self.bp_checkbox.setChecked(enabled)
            self.update_title()
            print(f"Bandpass filter {'enabled' if enabled else 'disabled'} via shortcut")
            
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
                                self.channel_offsets = np.arange(self.num_channels) * 2.0
                            
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
        if self.window is None:
            print("Window doesn't exist, exiting update_plot")
            return
            
        try:
            print(f"Checking data queue (empty={self.data_queue.empty()})")
            # Get the latest data from the queue
            if not self.data_queue.empty():
                try:
                    data, times, sfreq = self.data_queue.get_nowait()
                    print(f"Updating plot with data shape: {data.shape}, time points: {len(times)}")
                    
                    # Update signal processor sample rate
                    self.signal_processor.set_sample_rate(sfreq)
                    
                except Exception as e:
                    print(f"Error getting data from queue: {e}")
                    return
                
                if data.shape[1] < 2:
                    print("Not enough data points to plot yet")
                    return
                    
                # Apply signal processing if enabled
                if self.signal_processor.bandpass_enabled:
                    try:
                        # Apply bandpass filter
                        print(f"Applying bandpass filter: {self.signal_processor.bandpass_low}-{self.signal_processor.bandpass_high} Hz")
                        data = self.signal_processor.process(data, self.channel_labels)
                        print(f"Filtered data shape: {data.shape}")
                    except Exception as e:
                        print(f"Error applying filter: {e}")
                
                # Clear previous plots
                self.plot_widget.clear()
                self.plots = []
                
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
                            
                        # Center the signal around its offset
                        scaled_data = data[i] * channel_scale
                        
                        # Plot with yellow color
                        plot = self.plot_widget.plot(
                            times, 
                            scaled_data + self.channel_offsets[i],
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
                    
                    # Add text items for labels on right side
                    for i in label_indices:
                        if i < data.shape[0]:
                            label = pg.TextItem(
                                text=self.channel_labels[i],
                                color='white',
                                anchor=(0, 0.5)  # Center vertically, left-aligned horizontally
                            )
                            label.isChannelLabel = True  # Custom attribute to identify these items
                            label.setPos(end_time, self.channel_offsets[i])
                            self.plot_widget.addItem(label)
                
                # Update title with current info and scaling mode
                title = "EEG Signal Monitor"
                if self.current_bdf_file:
                    title += f" - {os.path.basename(self.current_bdf_file)}"
                    
                # Add scaling mode information
                if self.per_channel_scale:
                    title += " (Per-channel scaling)"
                else:
                    title += " (Global scaling)"
                
                self.plot_widget.setTitle(title, color='white', size='14pt')
                
                # Grid lines are added explicitly as InfiniteLines
                
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        # Always reschedule the next update with appropriate interval
        if hasattr(self, 'update_timer'):
            # Adjust update interval based on whether we have data
            if self.data_queue.empty() and self.num_channels == 0:
                # Use shorter interval if we haven't received any data yet
                self.update_timer.setInterval(200)  # Try more frequently until we get data
            else:
                # Use normal interval once we're displaying data
                self.update_timer.setInterval(self.update_interval)
            
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
        base_offset = 2.0
        channel_spacing = base_offset * (2 if self.num_channels <= 30 else 1.5)
        self.channel_offsets = np.arange(self.num_channels) * channel_spacing
        
        # Create time points
        times = np.linspace(0, duration, num_samples)
        
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
        
        # Create placeholder sine waves with different frequencies for each channel
        for i in range(self.num_channels):
            # Different frequency for each channel
            freq = 1 + (i % 10) * 0.5  # 1-5.5 Hz
            signal_data = self.y_scale * 0.5 * np.sin(2 * np.pi * freq * times)
            
            # Plot with yellow pen
            plot = self.plot_widget.plot(
                times, 
                signal_data + self.channel_offsets[i],
                pen=yellow_pen,
                name=self.channel_labels[i]
            )
            self.plots.append(plot)
        
        # Add channel labels as text items on the right side
        label_indices = []
        if self.num_channels > 30:
            # For many channels, only label every 5th channel
            label_indices = list(range(0, self.num_channels, 5))
        else:
            # For fewer channels, show all labels
            label_indices = list(range(self.num_channels))
            
        # Add text items for labels on right side
        right_edge = duration
        for i in label_indices:
            label = pg.TextItem(
                text=self.channel_labels[i],
                color='white',
                anchor=(0, 0.5)  # Center vertically, left-aligned horizontally
            )
            label.isChannelLabel = True  # Custom attribute to identify these items
            label.setPos(right_edge, self.channel_offsets[i])
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
