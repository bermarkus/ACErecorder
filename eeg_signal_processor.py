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
        
        # Reference settings
        self.reference_mode = 'original'  # 'original', 'average', 'linked_ears'
        self.reference_channels = ['A1', 'A2']  # For linked ears
        self.non_eeg_channels = ['HR', 'sync']  # Channels to exclude from re-referencing
        
    def set_reference_mode(self, mode):
        """Set the referencing mode"""
        if mode in ['original', 'average', 'linked_ears']:
            self.reference_mode = mode
            
    def get_channel_display_labels(self, channel_names):
        """Get channel labels with reference mode suffix
        
        Args:
            channel_names: Original channel names
            
        Returns:
            List of channel names with appropriate reference suffix
        """
        if self.reference_mode == 'original' or not channel_names:
            return channel_names.copy()
        
        display_labels = []
        for ch in channel_names:
            # Don't change non-EEG channel names
            if ch in self.non_eeg_channels or ch in self.reference_channels:
                display_labels.append(ch)
            else:
                # Add appropriate suffix based on reference mode
                if self.reference_mode == 'average':
                    display_labels.append(f"{ch}-Avg")
                elif self.reference_mode == 'linked_ears':
                    display_labels.append(f"{ch}-LE")
                else:
                    display_labels.append(ch)
        
        return display_labels
        
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
            # Identify which channels to re-reference
            non_eeg = [ch for ch in self.non_eeg_channels if ch in channel_names]
            print(f"Non-EEG channels excluded from average reference: {non_eeg}")
            
            # Get list of EEG channels (excluding non-EEG channels)
            eeg_channels = [ch for ch in channel_names if ch not in non_eeg]
            print(f"EEG channels for average reference: {eeg_channels}")
            
            try:
                # Get indices of EEG channels
                eeg_indices = [idx for idx, ch in enumerate(channel_names) if ch in eeg_channels]
                
                # Manual average reference implementation
                if eeg_indices:
                    # Calculate the average of EEG channels - make sure it's a 1D array
                    eeg_avg = np.mean(raw._data[eeg_indices, :], axis=0)
                    
                    # Subtract the average from EEG channels only
                    for idx in eeg_indices:
                        raw._data[idx, :] -= eeg_avg
                    
                    print("Average reference applied successfully, non-EEG channels preserved")
            except Exception as e:
                print(f"Error applying average reference: {e}")
        elif self.reference_mode == 'linked_ears':
            # Find reference channels
            ref_ch_indices = []
            ref_channels = []
            for ref_name in self.reference_channels:
                # Look for exact match first
                if ref_name in channel_names:
                    ref_ch_indices.append(channel_names.index(ref_name))
                    ref_channels.append(ref_name)
                else:
                    # Try flexible matching for channels like A1, EarL, etc.
                    for i, ch in enumerate(channel_names):
                        if ref_name.lower() in ch.lower() or (
                            ref_name.lower() == 'a1' and ch.lower().endswith('1')) or (
                            ref_name.lower() == 'a2' and ch.lower().endswith('2')):
                            ref_ch_indices.append(i)
                            ref_channels.append(ch)
                            break
            
            # Identify non-EEG channels
            non_eeg = [ch for ch in self.non_eeg_channels if ch in channel_names]
            print(f"Non-EEG channels: {non_eeg}")
            
            if not ref_ch_indices:
                print(f"Warning: No reference channels found for {self.reference_channels}")
                return raw.get_data()
                
            # Try to find A1/A2 if not already found
            if 'A1' not in ref_channels and any(ch.endswith('1') for ch in channel_names):
                new_ref = next(ch for ch in channel_names if ch.endswith('1'))
                if new_ref not in ref_channels:  # Avoid duplicates
                    ref_channels.append(new_ref)
                    ref_ch_indices.append(channel_names.index(new_ref))
                    print(f"Substituting {new_ref} for A1")
            
            if 'A2' not in ref_channels and any(ch.endswith('2') for ch in channel_names):
                new_ref = next(ch for ch in channel_names if ch.endswith('2'))
                if new_ref not in ref_channels:  # Avoid duplicates
                    ref_channels.append(new_ref)
                    ref_ch_indices.append(channel_names.index(new_ref))
                    print(f"Substituting {new_ref} for A2")
            
            if len(ref_channels) >= 1:  # At least one reference channel
                # Exclude non-EEG channels (HR, sync) and reference channels from re-referencing
                eeg_channels = [ch for ch in channel_names if ch not in ref_channels and ch not in non_eeg]
                print(f"EEG channels to re-reference: {eeg_channels}")
                
                # Re-reference EEG channels only
                if eeg_channels:
                    try:
                        # Get indices of channels to re-reference
                        eeg_indices = [idx for idx, ch in enumerate(channel_names) if ch in eeg_channels]
                        
                        # Manual linked ears reference implementation
                        if eeg_indices and ref_ch_indices:
                            # Calculate the average of reference channels - without keepdims to avoid broadcasting issues
                            ref_avg = np.mean(raw._data[ref_ch_indices, :], axis=0)
                            
                            # Apply linked ears reference to EEG channels only
                            print(f"Applying linked ears reference using: {ref_channels} to channels: {eeg_channels}")
                            for idx in eeg_indices:
                                raw._data[idx, :] -= ref_avg
                            
                        print("Reference applied successfully, non-EEG channels preserved")
                    except Exception as e:
                        print(f"Error applying reference: {e}")
            else:
                print("No reference channels available for linked-ears referencing")
                
        
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
        
    def toggle_notch(self, enabled=None):
        """Toggle or set notch filter state"""
        if enabled is None:
            self.notch_enabled = not self.notch_enabled
        else:
            self.notch_enabled = enabled
        return self.notch_enabled
        
    def set_notch_freq(self, freq):
        """Set notch filter frequency"""
        self.notch_freq = float(freq)
        
    def set_sample_rate(self, sample_rate):
        """Update the sample rate"""
        self.sample_rate = float(sample_rate)
