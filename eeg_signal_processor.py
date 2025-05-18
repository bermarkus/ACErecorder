import numpy as np
import mne

class EEGSignalProcessor:
    """
    Signal processor for EEG data with filtering and re-referencing capabilities
    using MNE-Python.
    """
    def __init__(self, sample_rate=250):
        """
        Initialize signal processor with default settings
        
        Args:
            sample_rate: sampling rate in Hz, defaults to 250 Hz
        """
        self.sample_rate = sample_rate
        
        # Filter settings
        self.bandpass_enabled = False
        self.bandpass_low = 1.0   # Hz
        self.bandpass_high = 40.0  # Hz
        
        self.notch_enabled = False
        self.notch_freq = 50.0  # Hz (or 60.0 for US)
        
        self.reference_mode = 'original'  # 'original', 'average', 'linked_mastoids'
        self.reference_channels = ['A1', 'A2']  # For linked mastoids
        
    def process(self, data, channel_names=None):
        """
        Process EEG data with current filter and reference settings
        
        Args:
            data: numpy array of shape (channels, samples)
            channel_names: list of channel names (optional)
            
        Returns:
            Processed data of the same shape
        """
        if data.size == 0:
            return data
            
        # If no channel names provided, create generic ones
        if channel_names is None:
            channel_names = [f'Ch{i+1}' for i in range(data.shape[0])]
            
        # No processing needed if all features disabled
        if not (self.bandpass_enabled or self.notch_enabled or 
                self.reference_mode != 'original'):
            return data
            
        # Create MNE info object
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sample_rate,
            ch_types=['eeg'] * len(channel_names)
        )
        
        # Create Raw object from data
        raw = mne.io.RawArray(data, info)
        
        # Apply bandpass filter if enabled
        if self.bandpass_enabled:
            raw.filter(
                l_freq=self.bandpass_low, 
                h_freq=self.bandpass_high,
                method='iir',
                iir_params=dict(order=4, ftype='butter'),
                verbose=False
            )
            
        # Apply notch filter if enabled
        if self.notch_enabled:
            raw.notch_filter(
                freqs=self.notch_freq,
                method='iir',
                iir_params=dict(order=4, ftype='butter'),
                verbose=False
            )
        
        # Apply re-referencing if needed
        if self.reference_mode == 'average':
            # Re-reference to average reference
            raw.set_eeg_reference('average', verbose=False)
        elif self.reference_mode == 'linked_mastoids':
            # Check if reference channels exist
            ref_picks = [i for i, ch in enumerate(channel_names) 
                        if ch in self.reference_channels]
            if ref_picks:
                # Re-reference to linked mastoids (average of A1 and A2)
                raw.set_eeg_reference(ref_picks, verbose=False)
        
        # Return the processed data
        return raw.get_data()
        
    def apply_bandpass(self, data, channel_names=None):
        """
        Apply only bandpass filter to data
        
        Args:
            data: numpy array of shape (channels, samples)
            channel_names: list of channel names (optional)
            
        Returns:
            Filtered data of the same shape
        """
        # Store current state
        orig_bandpass = self.bandpass_enabled
        orig_notch = self.notch_enabled
        orig_ref = self.reference_mode
        
        # Set temporary state for bandpass only
        self.bandpass_enabled = True
        self.notch_enabled = False
        self.reference_mode = 'original'
        
        # Process data
        result = self.process(data, channel_names)
        
        # Restore original state
        self.bandpass_enabled = orig_bandpass
        self.notch_enabled = orig_notch
        self.reference_mode = orig_ref
        
        return result
        
    def toggle_bandpass(self, enabled=None):
        """Toggle or set bandpass filter state"""
        if enabled is None:
            self.bandpass_enabled = not self.bandpass_enabled
        else:
            self.bandpass_enabled = enabled
        return self.bandpass_enabled
        
    def set_bandpass_range(self, low_freq, high_freq):
        """Set bandpass filter frequency range"""
        self.bandpass_low = float(low_freq)
        self.bandpass_high = float(high_freq)
        
    def set_sample_rate(self, sample_rate):
        """Update the sample rate"""
        self.sample_rate = float(sample_rate)
