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
        
        # Filter settings - enabled by default for 2-channel headset
        self.bandpass_enabled = True  # Default to enabled
        self.bandpass_low = 3.0   # Hz - match EDFbrowser to better remove DC offset
        self.bandpass_high = 45.0  # Hz - match EDFbrowser's upper limit
        
        self.notch_enabled = True  # Default to enabled
        self.notch_freqs = [50.0, 60.0]  # Both 50Hz (Europe/Asia) and 60Hz (US) notch filters enabled
        
        # Reference settings
        self.reference_mode = 'original'  # 'original', 'average', 'linked_ears'
        self.reference_channels = ['A1', 'A2']  # For linked ears
        self.non_eeg_channels = ['sync', 'Coherence']  # Special channels to exclude from processing
        self.preserve_from_filter = ['sync', 'Coherence']  # Channels to exclude from filtering (HR should be filtered)
        
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
        
        # Check if the data might be in a different unit scale than expected
        max_signal = np.max(np.abs(data))
        
        # ADC-to-microvolts conversion for BDF files
        # BDF files store ADC values that need conversion to microvolts
        # Physical range: -187500 to 187500 μV (from ACErecorder.py)
        # Digital range: -8388608 to 8388607 (24-bit ADC)
        adc_to_uv_factor = 375000.0 / 16777215.0  # ≈ 0.022351
        
        # Apply conversion if we detect large values (ADC range)
        if max_signal > 1000:  # If values are very large (definitely ADC values)
            print(f"Converting from ADC values to microvolts (max={max_signal:.2f})")
            # Apply proper conversion to all channels
            for i in range(data.shape[0]):
                # Convert from ADC units to microvolts
                data[i, :] = data[i, :] * adc_to_uv_factor
            print(f"After conversion: range {np.min(data):.2f} to {np.max(data):.2f} μV")
            
        # Create Raw object from data
        raw = mne.io.RawArray(data, info)
        
        # Identify channels to filter and those to preserve
        preserve_channel_indices = []
        filter_channel_indices = []
        
        for i, ch in enumerate(channel_names):
            # Skip filtering for special channels like sync and Coherence
            if any(special_ch in ch for special_ch in self.preserve_from_filter):
                preserve_channel_indices.append(i)
                print(f"Preserving channel from filtering: {ch}")
            else:
                filter_channel_indices.append(i)
                
        # Explicitly remove DC offset (mean) from all channels before filtering
        # This is CRITICAL to match EDFbrowser's display scaling
        if filter_channel_indices:
            print("Explicitly removing DC offset (mean) from signals")
            for idx in filter_channel_indices:
                # Store original data mean for debugging
                channel_mean = np.mean(raw._data[idx])
                # Remove the mean (DC offset)
                raw._data[idx] = raw._data[idx] - channel_mean
                print(f"  Channel {channel_names[idx]}: removed DC offset of {channel_mean:.2f} μV")
        
        # Store original data for channels that shouldn't be filtered
        preserved_data = {}
        for idx in preserve_channel_indices:
            preserved_data[idx] = raw._data[idx, :].copy()
        
        # Apply bandpass filter if enabled (only to EEG channels and HR)
        if self.bandpass_enabled and filter_channel_indices:
            print(f"Applying IIR Butterworth bandpass filter to {len(filter_channel_indices)} channels")
            # Use picks parameter to only filter selected channels
            raw.filter(
                l_freq=self.bandpass_low, 
                h_freq=self.bandpass_high,
                method='iir',                  # IIR filter (Butterworth) like EDFbrowser uses
                iir_params={'order': 4,         # 4th order Butterworth filter (2 poles per octave)
                          'ftype': 'butter',    # Butterworth filter type
                          'output': 'sos',      # Second-order sections for better numerical stability
                          },
                picks=filter_channel_indices,  # Only apply to EEG channels and HR
                phase='zero',                  # Zero phase filter (no delay)
                verbose=False
            )
            
        # Apply notch filter if enabled (only to EEG channels and HR)
        if self.notch_enabled and filter_channel_indices:
            # MNE doesn't support multiple frequencies with IIR notch filters
            # So we need to apply them one at a time
            for freq in self.notch_freqs:
                print(f"Applying IIR notch filter at {freq} Hz to {len(filter_channel_indices)} channels")
                # Use picks parameter to only filter selected channels
                raw.notch_filter(
                    freqs=freq,                  # Single frequency
                    method='iir',                # IIR notch filter for consistency with bandpass
                    iir_params={'order': 4,      # 4th order filter
                              'ftype': 'butter',  # Butterworth filter type
                              'output': 'sos'},   # Second-order sections for stability
                    picks=filter_channel_indices, # Only apply to EEG channels and HR
                    verbose=False
                )
            
        # Restore original data for unfiltered channels
        for idx, original_data in preserved_data.items():
            raw._data[idx, :] = original_data
            
        # For BDF files, we need to use the physical and digital min/max settings
        # BDF standard calibration:
        # physical_min = -187500 μV, physical_max = 187500 μV (from ACErecorder.py)
        # digital_min = -8388608, digital_max = 8388607 (24-bit ADC values)
        
        # EDFbrowser approach:
        # 1. Keep data in raw ADC values during filtering
        # 2. Convert to physical units (μV) at display time
        # 3. Let the BDF calibration factors handle the conversion
        #
        # MNE-Python's filter can cause scaling issues - we'll normalize when needed
        # but avoid any unnecessary scaling
        
        # Check if MNE's filtering has severely attenuated signals
        min_signal = float('inf')
        for i in filter_channel_indices:
            max_abs = np.max(np.abs(raw._data[i]))
            if max_abs < min_signal:
                min_signal = max_abs
        
        # Only apply minimal correction if filter has severely attenuated signals
        # This ensures filter output is in same general range as input but retains proper relationships
        if min_signal < 0.001 and min_signal > 0:
            print("Filter normalized signals to very small values - applying minimal correction")
            
            # Calculate a small normalization factor to avoid numerical issues
            # This does NOT attempt to scale to μV - that happens in the display renderer
            correction_factor = 1.0 / min_signal  # Just bring values back to ~1.0 range
            
            # Apply this minimal correction to all filtered channels
            for i in filter_channel_indices:
                raw._data[i] = raw._data[i] * correction_factor
                
            print(f"Applied minimal normalization factor of {correction_factor:.2f}x")
            print(f"Let BDF calibration in display handle conversion to μV")
        
        # Print diagnostic information about signal amplitudes after filtering
        for i, idx in enumerate(filter_channel_indices):
            if i < 2:  # Only print for first two channels to avoid log spam
                channel_name = channel_names[idx] if idx < len(channel_names) else f"Channel {idx}"
                signal = raw._data[idx]
                print(f"DIAGNOSTIC | {channel_name} after all filtering:")
                print(f"  Min: {np.min(signal):.2f} μV")
                print(f"  Max: {np.max(signal):.2f} μV")
                print(f"  Range: {np.max(signal) - np.min(signal):.2f} μV")
                print(f"  RMS: {np.sqrt(np.mean(np.square(signal))):.2f} μV")
                print(f"  First few samples: {signal[:5]}")
                
                # Calculate and print spectral characteristics
                from scipy import signal as scipy_signal
                f, Pxx = scipy_signal.welch(signal, fs=self.sample_rate, nperseg=min(2048, len(signal)))
                dominant_freq_idx = np.argmax(Pxx[1:]) + 1  # Skip DC (0 Hz)
                print(f"  Dominant frequency: {f[dominant_freq_idx]:.1f} Hz (power: {Pxx[dominant_freq_idx]:.2f})")
                print(f"  Total spectral power: {np.sum(Pxx):.2f}")
                print(f"  3-45 Hz band power: {np.sum(Pxx[(f >= 3) & (f <= 45)]):.2f}")
                print()
        
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
        """Set notch filter frequency - backwards compatibility for UI"""
        # Convert to list if single value
        if isinstance(freq, (int, float)):
            self.notch_freqs = [float(freq)]
        else:
            self.notch_freqs = [float(f) for f in freq]
        
    def set_sample_rate(self, sample_rate):
        """Update the sample rate"""
        self.sample_rate = float(sample_rate)
