#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Settings Menu - A separate UI component for controlling EEG signal processing

This module provides a floating/dockable settings window that controls the
signal processing parameters for the EEG signal monitor.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg

class EEGSettingsMenu(QtWidgets.QWidget):
    """
    A floating window for EEG signal processing settings.
    
    This class provides a UI for adjusting bandpass and notch filter settings
    for EEG signal processing. It communicates with an EEGSignalProcessor
    instance to apply the settings.
    """
    
    # Signal emitted when settings are changed
    settingsChanged = QtCore.pyqtSignal()
    
    def __init__(self, signal_processor, parent=None):
        """
        Initialize the settings menu.
        
        Args:
            signal_processor: An instance of EEGSignalProcessor
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Store reference to signal processor
        self.signal_processor = signal_processor
        
        # Setup UI
        self.setWindowTitle("Signal Processing Settings")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setMinimumWidth(300)
        self.setupUI()
        
    def setupUI(self):
        """Create the settings UI elements"""
        # Set dark theme styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D30;
                color: #E1E1E1;
                font-size: 11pt;
            }
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3F3F46;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #0E639C;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #333337;
                border: 1px solid #3F3F46;
                border-radius: 3px;
                padding: 2px 5px;
            }
        """)
        
        # Create main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Add header
        header = QtWidgets.QLabel("EEG Signal Processing")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Create bandpass filter group
        bp_group = QtWidgets.QGroupBox("Bandpass Filter")
        bp_layout = QtWidgets.QVBoxLayout(bp_group)
        
        # Enable/disable checkbox
        self.bp_checkbox = QtWidgets.QCheckBox("Enable Bandpass Filter")
        self.bp_checkbox.setChecked(self.signal_processor.bandpass_enabled)
        self.bp_checkbox.toggled.connect(self.toggle_bandpass_filter)
        bp_layout.addWidget(self.bp_checkbox)
        
        # Frequency range controls
        freq_layout = QtWidgets.QGridLayout()
        
        # Labels
        freq_layout.addWidget(QtWidgets.QLabel("Low cutoff:"), 0, 0)
        freq_layout.addWidget(QtWidgets.QLabel("High cutoff:"), 1, 0)
        
        # Spinboxes
        self.bp_low = QtWidgets.QDoubleSpinBox()
        self.bp_low.setRange(0.1, 100.0)
        self.bp_low.setValue(self.signal_processor.bandpass_low)
        self.bp_low.setSingleStep(0.5)
        self.bp_low.setSuffix(" Hz")
        self.bp_low.valueChanged.connect(self.update_bandpass_range)
        freq_layout.addWidget(self.bp_low, 0, 1)
        
        self.bp_high = QtWidgets.QDoubleSpinBox()
        self.bp_high.setRange(1.0, 200.0)
        self.bp_high.setValue(self.signal_processor.bandpass_high)
        self.bp_high.setSingleStep(1.0)
        self.bp_high.setSuffix(" Hz")
        self.bp_high.valueChanged.connect(self.update_bandpass_range)
        freq_layout.addWidget(self.bp_high, 1, 1)
        
        bp_layout.addLayout(freq_layout)
        
        # Preset buttons for common bands
        presets_layout = QtWidgets.QHBoxLayout()
        presets_layout.addWidget(QtWidgets.QLabel("Presets:"))
        
        # Theta (4-8 Hz)
        theta_btn = QtWidgets.QPushButton("Theta")
        theta_btn.setToolTip("4-8 Hz")
        theta_btn.clicked.connect(lambda: self.set_bandpass_preset(4, 8))
        presets_layout.addWidget(theta_btn)
        
        # Alpha (8-13 Hz)
        alpha_btn = QtWidgets.QPushButton("Alpha")
        alpha_btn.setToolTip("8-13 Hz")
        alpha_btn.clicked.connect(lambda: self.set_bandpass_preset(8, 13))
        presets_layout.addWidget(alpha_btn)
        
        # Beta (13-30 Hz)
        beta_btn = QtWidgets.QPushButton("Beta")
        beta_btn.setToolTip("13-30 Hz")
        beta_btn.clicked.connect(lambda: self.set_bandpass_preset(13, 30))
        presets_layout.addWidget(beta_btn)
        
        bp_layout.addLayout(presets_layout)
        main_layout.addWidget(bp_group)
        
        # Create notch filter group
        notch_group = QtWidgets.QGroupBox("Notch Filter")
        notch_layout = QtWidgets.QVBoxLayout(notch_group)
        
        # Enable/disable checkbox
        self.notch_checkbox = QtWidgets.QCheckBox("Enable Notch Filter")
        self.notch_checkbox.setChecked(self.signal_processor.notch_enabled)
        self.notch_checkbox.toggled.connect(self.toggle_notch_filter)
        notch_layout.addWidget(self.notch_checkbox)
        
        # Frequency control
        notch_freq_layout = QtWidgets.QHBoxLayout()
        notch_freq_layout.addWidget(QtWidgets.QLabel("Frequency:"))
        
        self.notch_freq = QtWidgets.QDoubleSpinBox()
        self.notch_freq.setRange(1.0, 200.0)
        self.notch_freq.setValue(self.signal_processor.notch_freq)
        self.notch_freq.setSingleStep(1.0)
        self.notch_freq.setSuffix(" Hz")
        self.notch_freq.valueChanged.connect(self.update_notch_freq)
        notch_freq_layout.addWidget(self.notch_freq)
        
        # Quick buttons
        notch_freq_layout.addWidget(QtWidgets.QLabel("Quick:"))
        
        # 50Hz button (EU)
        btn_50hz = QtWidgets.QPushButton("50Hz")
        btn_50hz.setMaximumWidth(60)
        btn_50hz.clicked.connect(lambda: self.set_notch_freq(50.0))
        notch_freq_layout.addWidget(btn_50hz)
        
        # 60Hz button (US)
        btn_60hz = QtWidgets.QPushButton("60Hz")
        btn_60hz.setMaximumWidth(60)
        btn_60hz.clicked.connect(lambda: self.set_notch_freq(60.0))
        notch_freq_layout.addWidget(btn_60hz)
        
        notch_layout.addLayout(notch_freq_layout)
        main_layout.addWidget(notch_group)
        
        # Add keyboard shortcut info
        shortcuts_label = QtWidgets.QLabel(
            "Keyboard Shortcuts:\n"
            "F1: Toggle Bandpass Filter\n"
            "F2: Toggle Notch Filter\n"
            "F11: Toggle Fullscreen"
        )
        shortcuts_label.setStyleSheet("color: #BBBBBB; font-size: 10pt;")
        shortcuts_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(shortcuts_label)
        
        # Add close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        main_layout.addWidget(close_btn)
        
        # Set size
        self.resize(350, 500)
        
    def toggle_bandpass_filter(self, checked):
        """Toggle bandpass filter based on checkbox"""
        self.signal_processor.bandpass_enabled = checked
        self.settingsChanged.emit()
        print(f"Bandpass filter {'enabled' if checked else 'disabled'}")
        
    def update_bandpass_range(self):
        """Update bandpass filter frequency range"""
        try:
            # Get values from spinboxes
            low_freq = self.bp_low.value()
            high_freq = self.bp_high.value()
            
            # Ensure high is always greater than low
            if high_freq <= low_freq:
                high_freq = low_freq + 1.0
                self.bp_high.setValue(high_freq)
                
            # Update the processor
            self.signal_processor.set_bandpass_range(low_freq, high_freq)
            self.settingsChanged.emit()
            print(f"Bandpass range updated: {low_freq}-{high_freq} Hz")
        except Exception as e:
            print(f"Error updating bandpass range: {e}")
    
    def set_bandpass_preset(self, low, high):
        """Set bandpass range to a preset"""
        self.bp_checkbox.setChecked(True)  # Enable bandpass
        self.bp_low.setValue(low)
        self.bp_high.setValue(high)
        # update_bandpass_range will be called by the valueChanged signals
            
    def toggle_notch_filter(self, checked):
        """Toggle notch filter based on checkbox"""
        self.signal_processor.notch_enabled = checked
        self.settingsChanged.emit()
        print(f"Notch filter {'enabled' if checked else 'disabled'}")
        
    def update_notch_freq(self):
        """Update notch filter frequency"""
        try:
            # Get value from spinbox
            freq = self.notch_freq.value()
            
            # Update the processor
            self.signal_processor.set_notch_freq(freq)
            self.settingsChanged.emit()
            print(f"Notch frequency updated: {freq} Hz")
        except Exception as e:
            print(f"Error updating notch frequency: {e}")
            
    def set_notch_freq(self, freq):
        """Set notch frequency from quick buttons"""
        try:
            # Update spinbox (which will trigger update_notch_freq)
            self.notch_freq.setValue(freq)
        except Exception as e:
            print(f"Error setting notch frequency: {e}")
            
    def showEvent(self, event):
        """When the window is shown, update controls to match processor state"""
        super().showEvent(event)
        
        # Update controls to match current processor state
        self.bp_checkbox.setChecked(self.signal_processor.bandpass_enabled)
        self.bp_low.setValue(self.signal_processor.bandpass_low)
        self.bp_high.setValue(self.signal_processor.bandpass_high)
        self.notch_checkbox.setChecked(self.signal_processor.notch_enabled)
        self.notch_freq.setValue(self.signal_processor.notch_freq)

# For testing
if __name__ == "__main__":
    import sys
    from eeg_signal_processor import EEGSignalProcessor
    
    app = QtWidgets.QApplication(sys.argv)
    processor = EEGSignalProcessor()
    menu = EEGSettingsMenu(processor)
    menu.show()
    sys.exit(app.exec_())
