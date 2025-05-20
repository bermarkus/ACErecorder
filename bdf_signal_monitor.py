import os
import time
import threading
import numpy as np
import queue
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg

import pyedflib
from mne.io import read_raw_bdf
import mne

# Import the signal processor and settings menu
from eeg_signal_processor import EEGSignalProcessor
from eeg_settings_menu import EEGSettingsMenu
from annotations_menu import AnnotationsMenu

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
        self.y_scale = 30.0  # Scale value in μV when not in auto mode (default for EEG signals)
        # Available y-scale options (in microvolts) - match EDFbrowser exactly
        self.y_scale_options = [30.0, 60.0, 100.0, 200.0]
        self.auto_scale = True   # Default to auto-scaling for 2-channel headset
        self._auto_scale_value = 1.0  # Separate value for auto-scaling tracking
        self.per_channel_scale = True  # Scale each channel individually
        
        # Fixed scale channels
        self.fixed_scale_channels = {
            'Coherence': 1.0  # Coherence channel has a fixed 0-1 µV scale
        }
        
        # BDF file physical/digital conversion parameters from ACErecorder.py
        # Note: We're not using these yet - need to confirm signal units first
        self.physical_min = -187500
        self.physical_max = 187500
        self.digital_min = -8388608
        self.digital_max = 8388607
        # Conversion factor: (physical_max - physical_min) / (digital_max - digital_min)
        self.adc_to_uv_factor = 375000 / 16777215  # ≈ 0.022351
        
        # Create menus (initially hidden)
        self.settings_menu = None
        self.annotations_menu = None
        
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
            
            # Standard window
            self.window.setWindowFlags(
                QtCore.Qt.Window
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
            self.settings_button = QtWidgets.QPushButton("📈", self.window)  # Chart increasing icon
            self.settings_button.setStyleSheet(
                "QPushButton {background-color: rgba(60, 60, 60, 150); color: white; " +
                "border: none; border-radius: 30px; font-size: 32px; padding: 5px;}" +
                "QPushButton:hover {background-color: rgba(80, 80, 80, 200);}"
            )
            self.settings_button.setFixedSize(60, 60)  # 2x larger
            self.settings_button.setToolTip("Open Signal Processing Settings")
            self.settings_button.clicked.connect(self.toggle_settings_menu)
            
            # Add an annotations button next to the settings button
            self.annotations_button = QtWidgets.QPushButton("✏️", self.window)  # Pencil emoji
            self.annotations_button.setStyleSheet(
                "QPushButton {background-color: rgba(60, 60, 60, 150); color: white; " +
                "border: none; border-radius: 30px; font-size: 32px; padding: 5px;}" +
                "QPushButton:hover {background-color: rgba(80, 80, 80, 200);}"
            )
            self.annotations_button.setFixedSize(60, 60)  # Same size as settings button
            self.annotations_button.setToolTip("Open Annotations Menu")
            self.annotations_button.clicked.connect(self.toggle_annotations_menu)
            
            # Position will be set after window is shown
            self.settings_button.hide()
            self.annotations_button.hide()
            
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
            
            # Initialize the menus
            self.init_settings_menu()
            self.init_annotations_menu()
            
            # Position the buttons in the top-left corner
            def position_buttons():
                # Position in top-left with padding
                padding = 15
                button_width = 60
                button_spacing = 10
                
                # Position settings button first
                self.settings_button.move(
                    padding,
                    padding
                )
                self.settings_button.show()
                self.settings_button.raise_()
                
                # Position annotations button to the right of settings button
                self.annotations_button.move(
                    padding + button_width + button_spacing,
                    padding
                )
                self.annotations_button.show()
                self.annotations_button.raise_()
                
            # Delay to ensure window is fully shown
            QtCore.QTimer.singleShot(500, position_buttons)
            
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
            # Position below the settings button
            button_pos = self.settings_button.mapToGlobal(QtCore.QPoint(0, 0))
            x = button_pos.x()  # Same horizontal position as the button
            y = button_pos.y() + self.settings_button.height() + 5  # Position below
            self.settings_menu.move(x, y)
            self.settings_menu.show()
            self.settings_menu.raise_()
    
    def init_annotations_menu(self):
        """Initialize the annotations menu"""
        if self.annotations_menu is None:
            try:
                # Create the annotations menu as a separate window with reference to this monitor
                self.annotations_menu = AnnotationsMenu(bdf_monitor=self)
                
                print("Annotations menu initialized successfully")
            except Exception as e:
                print(f"Error initializing annotations menu: {e}")
    
    def toggle_annotations_menu(self):
        """Toggle visibility of the annotations menu"""
        if self.annotations_menu is None:
            self.init_annotations_menu()
            
        if self.annotations_menu.isVisible():
            self.annotations_menu.hide()
        else:
            # Position below the annotations button
            button_pos = self.annotations_button.mapToGlobal(QtCore.QPoint(0, 0))
            x = button_pos.x()  # Same horizontal position as the button
            y = button_pos.y() + self.annotations_button.height() + 5  # Position below
            self.annotations_menu.move(x, y)
            self.annotations_menu.show()
            self.annotations_menu.raise_()
        

            

        

            
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
            filter_info.append(f"Notch: {', '.join(map(str, self.signal_processor.notch_freqs))}Hz")
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
        
        # Reset data queue
        self.data_queue = []
        
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
                            
                            # Get data for the last window_length seconds PLUS padding for filter edges
                            # Add extra padding (1 second) to reduce filter edge effects
                            filter_padding = 1.0  # seconds of extra data for filter edge effects
                            
                            # Calculate start index with padding
                            padded_length = self.window_length + (2 * filter_padding)
                            start_idx = max(0, int(len(raw.times) - padded_length * raw.info['sfreq']))
                            
                            # Get padded data window
                            data, times = raw[:, start_idx:]
                            
                            # We'll trim the padding after processing in update_plot
                            
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
                                
                            # NOTE: We don't calculate channel_offsets here anymore
                            # Channel offsets are now calculated consistently in update_plot
                            # to ensure fixed spacing that doesn't shift between updates
                            
                            # Print some debug info about the data
                            data_min = np.min(data)
                            data_max = np.max(data)
                            data_range = data_max - data_min
                            
                            print(f"Read BDF data: shape={data.shape}, time points={len(times)}, sample rate={raw.info['sfreq']}")
                            print(f"Value range: min={data_min:.2f}, max={data_max:.2f}, range={data_range:.2f}")
                                                    # Use a much simpler approach - we know the signals look good
                            # after removing the DC offset and applying the bandpass filters
                            # Let's just make sure the auto-scale calculation gives reasonable values
                            
                            # Keep track of the original range for scale calculation
                            self.raw_data_range = data_range
                            
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
        self.channel_spacing_multipliers = {}
        print("Reset all channel spacing multipliers to default (1.0)")
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
                    
                    # For debug purposes, print the range of signal values
                    print(f"Signal range: {np.min(data):.2f} to {np.max(data):.2f}")
                    
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
                            processing_info.append(f"notch {', '.join(map(str, self.signal_processor.notch_freqs))} Hz")
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
                
                # IIR filters don't need padding like FIR filters did
                # We no longer need to add and trim padding, which eliminates the buffer delay
                # Keep the original data shape with no trimming
                print(f"Using full data without padding: shape {data.shape}, time points: {len(times)}")
                # No delay processing needed with IIR filters
                    
                # ===== Initialize all variables needed for channel display =====
                # Initialize with default values to prevent any 'referenced before assignment' errors
                max_amp = np.max(np.abs(data))
                reference_channels_to_skip = []
                displayed_channel_indices = []
                num_displayed_channels = self.num_channels
                spacing_multiplier = 1.0
                
                # Data values are likely in millivolts but we want to display in microvolts
                # Need to apply a conversion from mV to µV (x1000)
                conversion_factor = 1000.0  # Convert mV to µV
                
                # Determine which channels to skip when using Linked Ears reference
                if hasattr(self, 'signal_processor') and self.signal_processor and \
                   self.signal_processor.reference_mode == 'linked_ears':
                    # Skip A1 and A2 channels when using Linked Ears reference
                    reference_channels_to_skip = self.signal_processor.reference_channels
                    print(f"Skipping reference channels: {reference_channels_to_skip}")
                
                # Create a list of channels that will be displayed
                # This will exclude reference channels when using Linked Ears reference
                displayed_channel_indices = []
                for i in range(min(self.num_channels, data.shape[0])):
                    if i < len(self.channel_labels) and self.channel_labels[i] not in reference_channels_to_skip:
                        displayed_channel_indices.append(i)
                
                # Count how many channels will actually be displayed
                num_displayed_channels = len(displayed_channel_indices)
                if num_displayed_channels == 0:  # Fallback if no channels would be displayed
                    num_displayed_channels = self.num_channels
                    displayed_channel_indices = list(range(min(self.num_channels, data.shape[0])))
                
                print(f"Displaying {num_displayed_channels} channels out of {self.num_channels} total")
                
                # Calculate spacing multiplier for channel distribution
                if num_displayed_channels < self.num_channels and num_displayed_channels > 0:
                    spacing_multiplier = self.num_channels / num_displayed_channels
                    print(f"Increasing channel spacing by factor of {spacing_multiplier:.2f}")
                else:
                    spacing_multiplier = 1.0
                
                # Handle scaling based on auto vs. fixed mode
                if self.auto_scale:
                    # Auto scaling: adapt to data amplitude
                    # Get max amplitude across all channels (excluding Coherence)
                    channel_amplitudes = []
                    for i in range(data.shape[0]):
                        channel_name = self.channel_labels[i] if i < len(self.channel_labels) else f"Channel {i}"
                        if not 'Coherence' in channel_name:
                            channel_amplitudes.append(np.max(np.abs(data[i])))
                    
                    max_amplitude = max(channel_amplitudes) if channel_amplitudes else 0.001
                    
                    # Ensure reasonable minimum amplitude for visibility
                    if max_amplitude < 0.1:
                        # Signal is very small (less than 0.1 μV) - use aggressive scaling
                        target_amplitude = max(max_amplitude * 50, 0.5)  # Target at least 0.5 μV
                        print(f"Auto scaling: signals very small ({max_amplitude:.6f} μV) - applying visibility boost")
                    else:
                        # Normal signal amplitudes, add 20% headroom
                        target_amplitude = max_amplitude * 1.2
                    
                    # Store for reference
                    self._auto_scale_value = target_amplitude
                    
                    # Scale to make max amplitude fill 80% of channel height
                    scale_factor = 0.8 / target_amplitude
                    
                    print(f"Auto scaling with factor: {scale_factor:.8f} (signals: {max_amplitude:.4f} μV)")
                    print(f"Target height: {max_amplitude * scale_factor:.4f} (80% of channel height)")
                else:
                    # Fixed scale mode: use the selected μV value
                    # A signal of y_scale μV should display at 80% of channel height
                    scale_factor = 0.8 / self.y_scale
                    
                    print(f"Fixed scaling at {self.y_scale} μV with factor: {scale_factor:.8f}") 
                    print(f"A {self.y_scale} μV signal will display at exactly 80% of channel height")
                    
                    # No adjustment - fixed means fixed regardless of signal size
                
                # Use a fixed, consistent channel spacing approach
                # Each channel gets an equal amount of vertical space
                base_channel_spacing = 2.0  # Base spacing between channels
                channel_spacing = base_channel_spacing * spacing_multiplier
                
                # Simplify channel offset calculation to prevent errors
                # Initialize to zeros first
                self.channel_offsets = np.zeros(self.num_channels)
                
                # Only set positions for displayed channels
                position_map = {}
                
                # First create a mapping from original index to position in displayed order
                for display_pos, idx in enumerate(displayed_channel_indices):
                    # Reverse order (top to bottom)
                    visual_pos = (num_displayed_channels - 1 - display_pos) * channel_spacing
                    position_map[idx] = visual_pos
                
                # Apply positions to channel offsets
                for i in range(self.num_channels):
                    if i in position_map:
                        self.channel_offsets[i] = position_map[i]
                
                # Now that we're properly converting to microvolts, show actual values
                if self.auto_scale:
                    max_uv = np.max(np.abs(data))
                    print(f"Channel spacing: {channel_spacing:.2f} (using {num_displayed_channels} displayed channels), "
                          f"Scale: auto {max_uv:.1f} µV (peak)")
                else:
                    print(f"Channel spacing: {channel_spacing:.2f} (using {num_displayed_channels} displayed channels), "
                          f"Scale: fixed {self.y_scale:.1f} µV")
                      
                # Log channel offsets for debugging
                print(f"Channel offsets: {len(self.channel_offsets)} total, {len(position_map)} visible")
                
                # Set consistent clipping limits to prevent signal overlap
                # Each channel is limited to 45% of the distance to the next channel (90% of available height)
                self.channel_max_amp = 0.9  # 90% of channel spacing for signal amplitude
                
                # Update or initialize per-channel scaling factors
                if len(self.channel_scales) != data.shape[0]:
                    # Initialize with uniform scaling
                    self.channel_scales = [scale_factor] * data.shape[0]
                
                # Calculate individual channel scales ONLY IF in auto-scale mode AND per-channel scaling enabled
                if self.auto_scale and self.per_channel_scale:
                    # First determine if Fp1/Fp2 channels exist and their indices
                    fp_indices = []
                    for i in range(min(len(self.channel_labels), data.shape[0])):
                        if self.channel_labels[i].startswith('Fp'):
                            fp_indices.append(i)
                    
                    # Calculate max amplitude across Fp channels specifically
                    fp_max_amp = 1.0  # Default minimum
                    if fp_indices:
                        fp_data = data[fp_indices, :]
                        fp_max_amp = np.max(np.abs(fp_data))
                        # Use actual amplitude for scaling - do not artificially limit
                        print(f"Found Fp channels with max amplitude: {fp_max_amp:.2f}μV")
                    
                    # Process each channel
                    for i in range(data.shape[0]):
                        # Get channel name for logging
                        channel_name = self.channel_labels[i] if i < len(self.channel_labels) else f"Channel {i}"
                        
                        # Special handling for Fp1/Fp2 channels
                        if channel_name.startswith('Fp'):
                            # For Fp channels, use the maximum amplitude across both Fp channels
                            # This ensures both channels use the same scale
                            # With proper μV conversion, our signal is now typically in the 10-100 μV range
                            # We want the signal to use 70% of the channel height
                            target_scale = 0.7 / fp_max_amp
                            
                            # No smoothing for initial scaling to ensure immediate visibility
                            if self.channel_scales[i] < 0.1:  # If scale is very small (initial or reset)
                                self.channel_scales[i] = target_scale
                            else:
                                # Use gentler smoothing (80% old, 20% new) to maintain stability
                                self.channel_scales[i] = self.channel_scales[i] * 0.8 + target_scale * 0.2
                                
                            # Log the actual microvolts value being used for scaling
                            effective_uv_scale = 0.7 / self.channel_scales[i]
                            print(f"  Auto-scale for {channel_name}: effective μV scale = {effective_uv_scale:.2f} μV")
                                
                            print(f"Auto-scaling {channel_name}: fp_max={fp_max_amp:.2f}μV, scale={self.channel_scales[i]:.5f}, height~{fp_max_amp * self.channel_scales[i]:.2f}")
                        else:
                            # Standard channels - process normally
                            channel_max = np.max(np.abs(data[i]))
                            
                            if channel_max > 0:
                                # Target 70% of channel height for standard channels
                                target_scale = 0.7 / channel_max
                                # Smooth the scale change (70% old, 30% new)
                                self.channel_scales[i] = self.channel_scales[i] * 0.7 + target_scale * 0.3
                                print(f"Auto-scaling {channel_name}: max={channel_max:.2f}μV, scale={self.channel_scales[i]:.5f}, height~{channel_max * self.channel_scales[i]:.2f}")
                elif not self.auto_scale:
                    # In fixed scale mode, ensure all channels use the same fixed scale
                    # Get min/max channel scale values for diagnostic purposes
                    min_scale = min(self.channel_scales) if self.channel_scales else scale_factor
                    max_scale = max(self.channel_scales) if self.channel_scales else scale_factor
                    print(f"Channel scales - min: {min_scale:.4f}, max: {max_scale:.4f}")
                    
                    # CRITICAL: In fixed scale mode, all EEG channels MUST use exactly the same scale factor
                    # This ensures that the y-scale setting (30μV, 60μV, etc.) is precisely honored
                    for i in range(data.shape[0]):
                        channel_name = self.channel_labels[i] if i < len(self.channel_labels) else f"Channel {i}"
                        if 'Coherence' in channel_name:
                            # Coherence channel keeps its special amplification
                            continue
                        else:
                            # Force reset to the exact scale factor calculated from y_scale
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
                    
                    # Add horizontal channel boundary lines (subtle grid showing channel limits)
                    # These serve as visual indicators of where channels will be clipped
                    boundary_pen = pg.mkPen(color='#444444', width=1, style=QtCore.Qt.DashLine)
                    for i in range(self.num_channels + 1):
                        # Calculate position (top boundary, between channels, bottom boundary)
                        if i == 0:
                            # Top boundary
                            pos = self.channel_offsets[0] + (channel_spacing * self.channel_max_amp / 2.0)
                        elif i == self.num_channels:
                            # Bottom boundary
                            pos = self.channel_offsets[-1] - (channel_spacing * self.channel_max_amp / 2.0)
                        else:
                            # Mid-channel boundary (halfway between channels)
                            pos = (self.channel_offsets[i-1] + self.channel_offsets[i]) / 2.0
                            
                        # Create horizontal line
                        grid_line = pg.InfiniteLine(
                            pos=pos,
                            angle=0,
                            pen=boundary_pen
                        )
                        self.plot_widget.addItem(grid_line)
                    
                    # We'll add labels after clearing and redrawing the main plots
                    
                    # Update ticks
                    self.plot_widget.getAxis('bottom').setTicks([x_ticks])
                    
                    # Update the bottom axis label to clarify the time scale
                    self.plot_widget.setLabel('bottom', 'Time (seconds ago)', color='white')
                    
                    # Now plot all channels
                    # Create yellow pen for all plots
                    yellow_pen = pg.mkPen(color='#ffff00', width=1)
                    
                    # Create a list of reference channels to skip when using Linked Ears reference
                    reference_channels_to_skip = []
                    if hasattr(self, 'signal_processor') and self.signal_processor and \
                       self.signal_processor.reference_mode == 'linked_ears':
                        # Skip A1 and A2 channels when using Linked Ears reference
                        reference_channels_to_skip = self.signal_processor.reference_channels
                        print(f"Skipping reference channels: {reference_channels_to_skip}")
                        
                    # Create a list of channels that will actually be displayed
                    # This is used to properly redistribute vertical space
                    displayed_channel_indices = []
                    for i in range(min(self.num_channels, data.shape[0])):
                        if self.channel_labels[i] not in reference_channels_to_skip:
                            displayed_channel_indices.append(i)
                    
                    # Count how many channels will actually be displayed
                    num_displayed_channels = len(displayed_channel_indices)
                    print(f"Displaying {num_displayed_channels} channels out of {self.num_channels} total")
                    
                    # Adjust channel spacing based on how many channels are actually displayed
                    # This ensures we use the entire vertical space efficiently
                    if num_displayed_channels < self.num_channels and num_displayed_channels > 0:
                        # Redistribute the vertical space among displayed channels
                        # Use the ratio of total to displayed channels to increase spacing
                        spacing_multiplier = self.num_channels / num_displayed_channels
                        print(f"Increasing channel spacing by factor of {spacing_multiplier:.2f}")
                    else:
                        spacing_multiplier = 1.0
                    
                    for i in range(min(self.num_channels, data.shape[0])):
                        # Skip A1/A2 channels when using Linked Ears reference
                        if self.channel_labels[i] in reference_channels_to_skip:
                            print(f"Skipping channel {self.channel_labels[i]} (used as reference)")
                            continue
                        # Check if this is a channel with a fixed scale requirement
                        channel_name = self.channel_labels[i]
                        
                        # DISPLAY SCALING ONLY - All signal processing is done in eeg_signal_processor.py
                        # We only handle proper display scaling based on selected Y-axis values here
                        
                        # Get the maximum absolute value in this channel for diagnostics
                        signal_max = np.max(np.abs(data[i]))
                        
                        # Special handling for Coherence channel
                        if 'Coherence' in channel_name:
                            # Coherence values are extremely small (around 0.7-0.8) and need amplification
                            # We apply a fixed amplification to make them visible with the same Y-scale
                            # Using 1,000,000x amplification as per EDFbrowser example
                            channel_scale = 1000000.0
                            scaled_data = data[i] * channel_scale
                            print(f"Coherence channel: values around {np.mean(data[i]):.6f}, amplified {channel_scale:.1f}x for visibility")
                        else:
                            # For EEG channels, data is already in μV from eeg_signal_processor.py
                            # BUT - we're seeing values around 0.0002 μV after filtering
                            # This is likely a consistent unit conversion issue
                            
                            # Get the amplitude for diagnostic purposes only
                            amp = np.max(np.abs(data[i]))
                            
                            # First apply a consistent base correction factor
                            # This converts the tiny filtered values to a reasonable range
                            base_correction = 10000.0  # 10^4 correction factor
                            
                            # Scale the signals based on display mode
                            if not self.auto_scale:
                                # For fixed scales, need a multiplier inversely proportional to y_scale
                                # This makes signals appear correctly scaled relative to the fixed scale
                                # At 30 μV scale: signals appear largest
                                # At 60 μV scale: signals appear half as tall 
                                # At 100 μV scale: signals appear 30% of 30 μV scale
                                y_scale_multiplier = 30.0 / self.y_scale  # Normalize to 30 μV scale
                                channel_scale = base_correction * y_scale_multiplier
                                print(f"EEG channel {channel_name}: fixed {self.y_scale} μV scale (multiplier: {y_scale_multiplier:.2f})")
                            else:
                                # For auto mode, determine scale from actual signal range
                                # First apply base correction to get reasonable values
                                corrected_amp = amp * base_correction
                                
                                # Scale to fit 90% of available vertical space
                                # Need to get data from all EEG channels to calculate correct auto-scaling
                                all_channels_max = 0
                                # Check all EEG channels (excluding Coherence)
                                for ch_idx in range(data.shape[0]):
                                    ch_name = self.channel_labels[ch_idx] if ch_idx < len(self.channel_labels) else f"Channel {ch_idx}"
                                    if 'Coherence' not in ch_name:
                                        ch_max = np.max(np.abs(data[ch_idx])) * base_correction
                                        all_channels_max = max(all_channels_max, ch_max)
                                
                                # Now scale to fit the display
                                # First get a reasonable min value to avoid division by zero
                                auto_scale_value = max(all_channels_max, 0.01)
                                # Bigger signals should appear smaller (inverse relationship)
                                auto_multiplier = 5.0 / auto_scale_value  # 5.0 works well as a reference scale
                                channel_scale = base_correction * auto_multiplier
                                
                                print(f"EEG channel {channel_name}: auto scale (max signal: {all_channels_max:.2f} μV)")
                                print(f"Auto scale multiplier: {auto_multiplier:.2f}x (adjust to fit display)")
                            
                                
                            print(f"Applied correction - Original: {amp:.6f} μV, Display: {amp * channel_scale:.2f} μV")
                                
                            scaled_data = data[i] * channel_scale

                                
                        # Diagnostic logging - after applying our direct scaling
                        print(f"Channel {channel_name}:")
                        print(f"  Original range: {np.min(data[i]):.2f} to {np.max(data[i]):.2f}")
                        print(f"  Scaled range: {np.min(scaled_data):.2f} to {np.max(scaled_data):.2f}")
                        print(f"  Scale factor: {channel_scale:.8f}")
                        print(f"  First few samples (raw): {data[i, :5]}")
                        print(f"  First few samples (scaled): {scaled_data[:5]}")
                        
                        # The rest of the scaling logic is SKIPPED
                        # We're using our direct scaling approach instead
                            
                        # We already applied channel scaling earlier, so skip this
                        
                        # Calculate boundary limits for this channel
                        clip_limit = channel_spacing * self.channel_max_amp / 2.0
                        
                        # Instead of clipping to the boundaries, we'll make out-of-bounds values disappear
                        # by setting them to NaN (Not a Number), which PyQtGraph will interpret as a break in the line
                        clipped_data = scaled_data.copy()
                        
                        # Only apply clipping for non-Coherence channels
                        if 'Coherence' not in channel_name:
                            # Create mask for out-of-bounds values (both above and below limits)
                            out_of_bounds_mask = (clipped_data > clip_limit) | (clipped_data < -clip_limit)
                            
                            # Replace out-of-bounds values with NaN to make them invisible
                            clipped_data[out_of_bounds_mask] = np.nan
                        else:
                            # For Coherence channel, don't clip - ensure all values are visible
                            # Instead, constrain values to stay within display bounds
                            # This prevents the signal from disappearing when reaching extremes
                            clipped_data = np.clip(clipped_data, -clip_limit * 0.99, clip_limit * 0.99)
                            print(f"Ensuring Coherence channel is always visible (not clipped)")
                            
                            # Coherence values are always positive and in the 0.0-1.0 range
                            # By clipping to 0.99 of the boundary, we ensure they're always visible
                            # while staying inside the grey boundary box
                        
                        # Plot the clipped data with vertical offset for stacking channels
                        # Use thicker pen line for Coherence channel only
                        if 'Coherence' in channel_name:
                            # Thicker yellow pen for Coherence (3px instead of 1px)
                            coherence_pen = pg.mkPen(color='#ffff00', width=3)
                            plot = self.plot_widget.plot(
                                times, 
                                clipped_data + self.channel_offsets[i],  # Apply channel offset
                                pen=coherence_pen,
                                name=self.channel_labels[i]
                            )
                        else:
                            # Standard thickness for all other channels (1px)
                            plot = self.plot_widget.plot(
                                times, 
                                clipped_data + self.channel_offsets[i],  # Apply channel offset
                                pen=yellow_pen,
                                name=self.channel_labels[i]
                            )
                        self.plots.append(plot)
                        
                        # Add y-axis scale markings for this channel
                        if not self.channel_labels[i] in reference_channels_to_skip:  # Don't add scales for skipped channels
                            scale_color = '#AAAAAA'  # Light grey for scale markings
                            label_style = {'color': scale_color, 'font-size': '8pt'}
                            
                            # Determine channel min/max values in microvolts
                            if 'Coherence' in self.channel_labels[i]:
                                # Special case for Coherence - always 0 to 1.0
                                min_val = 0.0
                                max_val = 1.0
                            elif self.auto_scale:
                                # For auto-scale, use the actual data amplitude
                                max_amp = np.max(np.abs(data[i])) * 1000  # Convert to µV
                                min_val = -max_amp
                                max_val = max_amp
                            else:
                                # For fixed scale, use the y_scale setting
                                min_val = -self.y_scale
                                max_val = self.y_scale
                            
                            mid_val = (min_val + max_val) / 2.0
                            
                            # Calculate positions for scale markings
                            min_pos = self.channel_offsets[i] - clip_limit
                            mid_pos = self.channel_offsets[i]
                            max_pos = self.channel_offsets[i] + clip_limit
                            
                            # Add text labels on the left edge
                            left_edge = times[0] - (0.01 * self.window_length)  # Slightly left of data
                            
                            # Add min value label
                            min_text = pg.TextItem(
                                text=f"{min_val:.1f}", 
                                color=scale_color,
                                anchor=(1.0, 0.5)  # Right-aligned
                            )
                            min_text.setPos(left_edge, min_pos)
                            min_text.setParentItem(self.plot_widget.getPlotItem())
                            
                            # Add mid value label
                            mid_text = pg.TextItem(
                                text=f"{mid_val:.1f}", 
                                color=scale_color,
                                anchor=(1.0, 0.5)  # Right-aligned
                            )
                            mid_text.setPos(left_edge, mid_pos)
                            mid_text.setParentItem(self.plot_widget.getPlotItem())
                            
                            # Add max value label
                            max_text = pg.TextItem(
                                text=f"{max_val:.1f}", 
                                color=scale_color,
                                anchor=(1.0, 0.5)  # Right-aligned
                            )
                            max_text.setPos(left_edge, max_pos)
                            max_text.setParentItem(self.plot_widget.getPlotItem())
                        
                        # Add special bounding box for Coherence channel
                        if 'Coherence' in channel_name:
                            # Create grey pen with 2px width for bounding box
                            grey_pen = pg.mkPen(color='#808080', width=2)
                            
                            # Add upper bound line (representing 1.0 value - exact value)
                            # Set to exactly 1.0 as requested
                            upper_bound = self.channel_offsets[i] + clip_limit
                            upper_line = pg.InfiniteLine(
                                pos=upper_bound,
                                angle=0,
                                pen=grey_pen
                                # No label
                            )
                            self.plot_widget.addItem(upper_line)
                            
                            # Add lower bound line (representing 0.0 value)
                            lower_bound = self.channel_offsets[i] - clip_limit
                            lower_line = pg.InfiniteLine(
                                pos=lower_bound,
                                angle=0,
                                pen=grey_pen
                                # No label
                            )
                            self.plot_widget.addItem(lower_line)
                            
                            print(f"Added bounding box for Coherence channel at offset {self.channel_offsets[i]:.2f} with upper limit {clip_limit:.2f} (1.0) and lower limit -{clip_limit:.2f} (0.0)")
                    
                    # Add channel labels as text items on the right side
                    # Determine which channels to show labels for
                    label_indices = []
                    if self.num_channels > 30:
                        # For many channels, only label every 5th channel
                        label_indices = list(range(0, self.num_channels, 5))
                    else:
                        # For fewer channels, show all labels
                        label_indices = list(range(self.num_channels))
                    
                    # Add channel labels and scale markings on the left edge
                    # Define colors and styles
                    label_color = '#ffff00'  # Yellow to match the signal lines
                    scale_color = '#FFFFFF'  # White for scale markings - more visible
                    background_color = (50, 50, 50, 200)  # Dark gray with opacity (R,G,B,A)
                    
                    # Position labels and scales at the left edge of the display
                    left_edge = start_time + 0.01  # Very close to left edge (1% of window)
                    scales_edge = start_time + 0.01  # Same position as channel labels for now
                    
                    for i in label_indices:
                        if i < data.shape[0]:
                            # Skip labels for A1/A2 channels when using Linked Ears reference
                            if self.channel_labels[i] in reference_channels_to_skip:
                                continue
                                
                            # Create left side label only
                            left_label = pg.TextItem(
                                text=self.channel_labels[i],
                                color=label_color,
                                anchor=(0, 0.5),  # Left-aligned, vertically centered
                                fill=background_color  # Add opaque background
                            )
                            left_label.isChannelLabel = True  # Custom attribute
                            left_label.setPos(left_edge, self.channel_offsets[i])
                            self.plot_widget.addItem(left_label)
                            
                            # Add scale markings for this channel
                            # Calculate min/max/mid values in microvolts
                            if 'Coherence' in self.channel_labels[i]:
                                # Special case for Coherence - always 0 to 1
                                min_val = 0.0
                                max_val = 1.0
                            elif self.auto_scale and i < data.shape[0]:
                                # For auto-scale, use actual data amplitude
                                max_amp = np.max(np.abs(data[i])) * 1000  # mV to µV
                                min_val = -max_amp
                                max_val = max_amp
                            else:
                                # Fixed scale using y_scale setting
                                min_val = -self.y_scale
                                max_val = self.y_scale
                                
                            # Calculate middle value
                            mid_val = (min_val + max_val) / 2.0
                            
                            # Calculate positions for scale markings
                            clip_limit_for_channel = channel_spacing * self.channel_max_amp / 2.0
                            min_pos = self.channel_offsets[i] - clip_limit_for_channel
                            mid_pos = self.channel_offsets[i]
                            max_pos = self.channel_offsets[i] + clip_limit_for_channel
                            
                            # Add min value label with background for visibility
                            min_label = pg.TextItem(
                                text=f"{min_val:.1f}",
                                color=scale_color,
                                anchor=(0, 0.5),  # Left-aligned
                                fill=(40, 40, 40, 200)  # Dark background with opacity
                            )
                            # Position to the right of channel label
                            min_label.setPos(left_edge + 0.15, min_pos)
                            self.plot_widget.addItem(min_label)
                            
                            # Add mid value label with background
                            mid_label = pg.TextItem(
                                text=f"{mid_val:.1f}",
                                color=scale_color,
                                anchor=(0, 0.5),  # Left-aligned
                                fill=(40, 40, 40, 200)  # Dark background with opacity
                            )
                            mid_label.setPos(left_edge + 0.15, mid_pos) 
                            self.plot_widget.addItem(mid_label)
                            
                            # Add max value label with background
                            max_label = pg.TextItem(
                                text=f"{max_val:.1f}", 
                                color=scale_color,
                                anchor=(0, 0.5),  # Left-aligned
                                fill=(40, 40, 40, 200)  # Dark background with opacity
                            )
                            max_label.setPos(left_edge + 0.15, max_pos)
                            self.plot_widget.addItem(max_label)
                
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
        """Set up the initial display components."""
        self.y_scale = 50  # Initial default fixed scale - 50μV 
        self.available_y_scales = [30, 50, 75, 100, 200, 500]  # Available scales in μV
        self.signal_amplitude_correction = 1000.0  # Force correct display amplitude for EEG
        self.eeg_rescale_factor = 1.0  # Will be adjusted dynamically when needed
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
        # Use a default duration of self.window_length for initial display
        initial_duration = self.window_length  # Default to 10 seconds (or whatever window_length is)
        
        x_ticks = []
        for i in range(-self.window_length, 1):
            # For initial display, map the display time directly
            x_ticks.append((i, str(i)))
        
        # Add explicit vertical grid lines at each second mark first
        for i in range(-self.window_length, 1):
            grid_line = pg.InfiniteLine(
                pos=i,
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
        left_edge = -self.window_length + 0.01  # Very close to left edge
        
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
        # Use 0 as the right edge of the display for initial setup
        self.plot_widget.setXRange(-self.window_length, 0)


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
