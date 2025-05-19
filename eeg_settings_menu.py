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
    
    def __init__(self, signal_processor, parent=None, bdf_monitor=None):
        """
        Initialize the settings menu.
        
        Args:
            signal_processor: An instance of EEGSignalProcessor
            parent: Optional parent widget
            bdf_monitor: Optional reference to BDFSignalMonitor instance
        """
        super().__init__(parent)
        
        # Store references
        self.signal_processor = signal_processor
        self.bdf_monitor = bdf_monitor
        
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
        
        # Note: Theta, Alpha, Beta presets have been removed
        main_layout.addWidget(bp_group)
        
        # Create notch filter group
        notch_group = QtWidgets.QGroupBox("Notch Filter")
        notch_layout = QtWidgets.QVBoxLayout(notch_group)
        
        # Enable/disable checkbox
        self.notch_checkbox = QtWidgets.QCheckBox("Enable Notch Filter")
        self.notch_checkbox.setChecked(self.signal_processor.notch_enabled)
        self.notch_checkbox.toggled.connect(self.toggle_notch_filter)
        notch_layout.addWidget(self.notch_checkbox)
        
        # Checkboxes for common notch frequencies
        notch_freq_layout = QtWidgets.QGridLayout()
        notch_freq_layout.addWidget(QtWidgets.QLabel("Frequencies:"), 0, 0)
        
        # 50Hz checkbox
        self.notch_50hz = QtWidgets.QCheckBox("50 Hz")
        self.notch_50hz.setChecked(50.0 in self.signal_processor.notch_freqs)
        self.notch_50hz.toggled.connect(self.update_notch_freqs)
        notch_freq_layout.addWidget(self.notch_50hz, 0, 1)
        
        # 60Hz checkbox
        self.notch_60hz = QtWidgets.QCheckBox("60 Hz")
        self.notch_60hz.setChecked(60.0 in self.signal_processor.notch_freqs)
        self.notch_60hz.toggled.connect(self.update_notch_freqs)
        notch_freq_layout.addWidget(self.notch_60hz, 0, 2)
        
        # Custom frequency control
        self.notch_custom_layout = QtWidgets.QHBoxLayout()
        self.notch_custom_layout.addWidget(QtWidgets.QLabel("Custom:"))
        
        self.notch_custom_freq = QtWidgets.QDoubleSpinBox()
        self.notch_custom_freq.setRange(1.0, 200.0)
        self.notch_custom_freq.setValue(50.0)
        self.notch_custom_freq.setSingleStep(1.0)
        self.notch_custom_freq.setSuffix(" Hz")
        self.notch_custom_layout.addWidget(self.notch_custom_freq)
        
        # Add button for custom frequency
        btn_add_custom = QtWidgets.QPushButton("Add")
        btn_add_custom.setMaximumWidth(60)
        btn_add_custom.clicked.connect(self.add_custom_notch)
        self.notch_custom_layout.addWidget(btn_add_custom)
        
        # Add the custom frequency controls as a new row
        notch_freq_layout.addLayout(self.notch_custom_layout, 1, 0, 1, 3)
        
        notch_layout.addLayout(notch_freq_layout)
        main_layout.addWidget(notch_group)
        
        # Create reference mode group
        ref_group = QtWidgets.QGroupBox("Reference Mode")
        ref_layout = QtWidgets.QVBoxLayout(ref_group)
        
        # Radio button group for reference mode selection
        self.ref_mode_group = QtWidgets.QButtonGroup(self)
        
        # Original reference (no change)
        self.ref_original = QtWidgets.QRadioButton("Original Reference")
        self.ref_mode_group.addButton(self.ref_original, 0)
        ref_layout.addWidget(self.ref_original)
        
        # Linked Ears reference (A1+A2 average)
        self.ref_linked_ears = QtWidgets.QRadioButton("Linked Ears (A1+A2)")
        self.ref_mode_group.addButton(self.ref_linked_ears, 1)
        ref_layout.addWidget(self.ref_linked_ears)
        
        # Average reference
        self.ref_average = QtWidgets.QRadioButton("Average Reference")
        self.ref_mode_group.addButton(self.ref_average, 2)
        ref_layout.addWidget(self.ref_average)
        
        # Information label
        ref_info = QtWidgets.QLabel("Note: Non-EEG channels (HR, sync) are not re-referenced.\nLinked ears requires A1 and A2 channels.")
        ref_info.setWordWrap(True)
        ref_info.setStyleSheet("font-size: 9pt; color: #aaaaaa;")
        ref_layout.addWidget(ref_info)
        
        # Set initial state based on current signal processor setting
        if self.signal_processor.reference_mode == 'original':
            self.ref_original.setChecked(True)
        elif self.signal_processor.reference_mode == 'linked_ears':
            self.ref_linked_ears.setChecked(True)
        elif self.signal_processor.reference_mode == 'average':
            self.ref_average.setChecked(True)
        else:
            self.ref_original.setChecked(True)
            
        # Connect signal
        self.ref_mode_group.buttonClicked.connect(self.update_reference_mode)
        
        main_layout.addWidget(ref_group)
        
        # Create Y-scale settings group
        scale_group = QtWidgets.QGroupBox("Y-Scale")
        scale_layout = QtWidgets.QVBoxLayout(scale_group)
        
        # Radio button group for vertical scaling options
        self.scale_group = QtWidgets.QButtonGroup(self)
        
        # Fixed scale: 30 µV
        self.scale_30 = QtWidgets.QRadioButton("30 µV")
        self.scale_group.addButton(self.scale_30, 0)
        scale_layout.addWidget(self.scale_30)
        
        # Fixed scale: 60 µV
        self.scale_60 = QtWidgets.QRadioButton("60 µV")
        self.scale_group.addButton(self.scale_60, 1)
        scale_layout.addWidget(self.scale_60)
        
        # Fixed scale: 100 µV
        self.scale_100 = QtWidgets.QRadioButton("100 µV")
        self.scale_group.addButton(self.scale_100, 2)
        scale_layout.addWidget(self.scale_100)
        
        # Auto scale
        self.scale_auto = QtWidgets.QRadioButton("Auto")
        self.scale_group.addButton(self.scale_auto, 3)
        scale_layout.addWidget(self.scale_auto)
        
        # Set default to 30 µV
        if self.bdf_monitor and hasattr(self.bdf_monitor, 'auto_scale'):
            if self.bdf_monitor.auto_scale:
                self.scale_auto.setChecked(True)
            else:
                # Try to match the current scale
                if self.bdf_monitor.y_scale >= 90:
                    self.scale_100.setChecked(True)
                elif self.bdf_monitor.y_scale >= 50:
                    self.scale_60.setChecked(True)
                else:
                    self.scale_30.setChecked(True)
        else:
            # Default to 30 µV
            self.scale_30.setChecked(True)
            
        # Connect signal
        self.scale_group.buttonClicked.connect(self.update_y_scale)
        
        # Add to main layout
        main_layout.addWidget(scale_group)
        
        # Keyboard shortcuts have been removed
        
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
        self.signal_processor.toggle_notch(checked)
        self.settingsChanged.emit()
        print(f"Notch filter {'enabled' if checked else 'disabled'}")
        
    def update_notch_freqs(self):
        """Update the notch filter frequencies based on checkboxes"""
        freqs = []
        if self.notch_50hz.isChecked():
            freqs.append(50.0)
        if self.notch_60hz.isChecked():
            freqs.append(60.0)
            
        if not freqs:  # If no boxes checked, default to both
            self.notch_50hz.setChecked(True)
            self.notch_60hz.setChecked(True)
            freqs = [50.0, 60.0]
        
        # Apply the frequencies
        self.signal_processor.set_notch_freq(freqs)
        print(f"Notch filter frequencies set to {freqs} Hz")
        self.settingsChanged.emit()
        
    def add_custom_notch(self):
        """Add a custom notch filter frequency"""
        value = self.notch_custom_freq.value()
        
        # Get current frequencies
        current_freqs = []
        if self.notch_50hz.isChecked():
            current_freqs.append(50.0)
        if self.notch_60hz.isChecked():
            current_freqs.append(60.0)
            
        # Add the custom frequency if not already present
        if value not in current_freqs:
            current_freqs.append(value)
            
        # Apply to processor
        self.signal_processor.set_notch_freq(current_freqs)
        print(f"Added notch filter at {value} Hz. Active filters: {current_freqs}")
        self.settingsChanged.emit()
    
    def update_reference_mode(self, button):
        """Update reference mode based on radio button selection"""
        btn_id = self.ref_mode_group.id(button)
        
        if btn_id == 0:
            self.signal_processor.reference_mode = 'original'
        elif btn_id == 1:
            self.signal_processor.reference_mode = 'linked_ears'
        elif btn_id == 2:
            self.signal_processor.reference_mode = 'average'
        
        print(f"Reference mode changed to: {self.signal_processor.reference_mode}")
        
        # Update channel labels in the monitor if available
        if self.bdf_monitor:
            self.bdf_monitor.update_channel_labels()
            
        self.settingsChanged.emit()
        
    def update_y_scale(self, button):
        """Update Y-scale based on radio button selection"""
        if not self.bdf_monitor or not hasattr(self.bdf_monitor, 'auto_scale'):
            return
            
        btn_id = self.scale_group.id(button)
        
        if btn_id == 0:  # 30 µV
            self.bdf_monitor.auto_scale = False
            self.bdf_monitor.y_scale = 30.0
        elif btn_id == 1:  # 60 µV
            self.bdf_monitor.auto_scale = False
            self.bdf_monitor.y_scale = 60.0
        elif btn_id == 2:  # 100 µV
            self.bdf_monitor.auto_scale = False
            self.bdf_monitor.y_scale = 100.0
        elif btn_id == 3:  # Auto
            self.bdf_monitor.auto_scale = True
        
        print(f"Y-scale changed: auto={self.bdf_monitor.auto_scale}, scale={self.bdf_monitor.y_scale}")
        
        # Force the monitor to update with new scale
        self.bdf_monitor.update_plot(force_labels=True)
        
        self.settingsChanged.emit()
        
    def showEvent(self, event):
        """When the window is shown, update controls to match processor state"""
        super().showEvent(event)
        
        # Update controls to match current processor state
        self.bp_checkbox.setChecked(self.signal_processor.bandpass_enabled)
        self.bp_low.setValue(self.signal_processor.bandpass_low)
        self.bp_high.setValue(self.signal_processor.bandpass_high)
        self.notch_checkbox.setChecked(self.signal_processor.notch_enabled)
        # This is outdated - we now use checkboxes instead of a single frequency value
        # Old code: self.notch_freq.setValue(self.signal_processor.notch_freq)
        
        # Update reference mode radio buttons
        if self.signal_processor.reference_mode == 'original':
            self.ref_original.setChecked(True)
        elif self.signal_processor.reference_mode == 'linked_ears':
            self.ref_linked_ears.setChecked(True)
        elif self.signal_processor.reference_mode == 'average':
            self.ref_average.setChecked(True)
        else:
            self.ref_original.setChecked(True)

# For testing
if __name__ == "__main__":
    import sys
    from eeg_signal_processor import EEGSignalProcessor
    
    app = QtWidgets.QApplication(sys.argv)
    processor = EEGSignalProcessor()
    menu = EEGSettingsMenu(processor)
    menu.show()
    sys.exit(app.exec_())
