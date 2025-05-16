"""
fft_analysis.py - Frequency analysis for EEG signals

This module provides functions to perform FFT analysis on EEG data and
detect common interference patterns such as 50Hz and 60Hz line noise.
Focused on real-time analysis of live EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy.fft import fft, fftfreq
import signal_detect  # Import to reuse channel configurations

# Constants
SAMPLE_RATE = 250  # Default sample rate in Hz for most EEG devices
LINE_FREQ_RANGES = {
    "50Hz": (49.0, 51.0),  # European/Asian power line frequency range (widened)
    "60Hz": (59.0, 61.0)   # North American power line frequency range (widened)
}
# Ratios relative to the average spectrum power to detect significant line noise
LINE_NOISE_THRESHOLD_RATIO = 5.0  # Noise peak must be this times higher than average power


def perform_fft(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Fast Fourier Transform on a signal.
    
    Args:
        signal: 1D array of signal values
        sample_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    # Number of samples
    n = len(signal)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal * window
    
    # Perform FFT
    fft_result = fft(windowed_signal)
    
    # Calculate magnitude spectrum (take absolute value of complex FFT result)
    # Only keep first half of the spectrum (since it's symmetrical for real signals)
    # Note: We're using the squared magnitude for proper power spectrum
    power_spectrum = np.abs(fft_result[:n//2])**2 / (n**2)  # Adjust normalization for power spectrum
    
    # Calculate frequency axis
    frequencies = fftfreq(n, 1/sample_rate)[:n//2]
    
    # Debug - look for expected 50Hz peak
    indices_50hz = np.where((frequencies >= 49.0) & (frequencies <= 51.0))[0]
    if len(indices_50hz) > 0:
        max_idx = np.argmax(power_spectrum[indices_50hz])
        max_freq = frequencies[indices_50hz[max_idx]]
        max_power = power_spectrum[indices_50hz[max_idx]]
        print(f"FFT found peak at {max_freq:.2f} Hz with power {max_power:.2e}")
    
    return frequencies, power_spectrum


def detect_line_noise(power_spectrum: np.ndarray, frequencies: np.ndarray, threshold_ratio=None) -> Dict[str, float]:
    """
    Detect power line interference at 50Hz and 60Hz.
    
    Args:
        power_spectrum: Power spectrum from FFT
        frequencies: Frequency array corresponding to the power spectrum
        threshold_ratio: Optional custom threshold ratio to override the default
        
    Returns:
        Dictionary of noise type to relative power ratio
    """
    noise_ratios = {}
    
    # For each line frequency, calculate the ratio of peak power to average power
    # in the entire spectrum (excluding DC component and very low frequencies)
    
    # Compute average power in different regions to get a better baseline
    # Use the frequency range from 10Hz to 40Hz as the baseline
    baseline_indices = np.where((frequencies >= 10) & (frequencies <= 40))[0]
    
    if len(baseline_indices) > 0:
        baseline_power = np.mean(power_spectrum[baseline_indices])
    else:
        # Fallback to using most of the spectrum if baseline range wasn't found
        baseline_power = np.mean(power_spectrum[5:])  # Skip first few bins
    
    # Make sure baseline isn't zero to avoid division issues
    if baseline_power < 1e-15:
        baseline_power = 1e-15
        
    print(f"Baseline power: {baseline_power:.2e}")
    
    # Check each line frequency range
    for noise_type, (freq_min, freq_max) in LINE_FREQ_RANGES.items():
        # Find indices corresponding to the frequency range
        indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max))[0]
        
        if len(indices) > 0:
            # Get the peak power in this frequency range
            peak_idx = np.argmax(power_spectrum[indices])
            peak_freq = frequencies[indices[peak_idx]]
            peak_power = power_spectrum[indices[peak_idx]]
            
            # Calculate ratio to average power
            ratio = peak_power / baseline_power
            noise_ratios[noise_type] = ratio
            
            print(f"{noise_type} peak at {peak_freq:.2f}Hz: {peak_power:.2e} (ratio: {ratio:.2f}x baseline)")
            
    return noise_ratios


def analyze_channel_noise(data: np.ndarray, 
                          channel_names: List[str], 
                          sample_rate: int = SAMPLE_RATE,
                          threshold_ratio: float = LINE_NOISE_THRESHOLD_RATIO,
                          debug: bool = True) -> Dict[str, Dict[str, float]]:
    print(f"\nAnalyzing {len(channel_names)} channels for line noise with threshold {threshold_ratio:.2f}x")
    """
    Analyze each EEG channel for power line interference.
    
    Args:
        data: Array of EEG data with shape (channels, samples)
        channel_names: List of channel names corresponding to rows in the data
        sample_rate: Sampling rate in Hz
        
    Returns:
        Dictionary mapping channel names to noise analysis results
    """
    results = {}
    
    for i, channel_name in enumerate(channel_names):
        if i < len(data):
            channel_data = data[i]
            
            # Skip if insufficient data points for FFT
            if len(channel_data) < 50:  # Minimum number of samples needed
                print(f"Skipping {channel_name} - insufficient data points: {len(channel_data)}")
                continue
                
            # Perform FFT
            frequencies, power_spectrum = perform_fft(channel_data, sample_rate)
            
            if debug:
                print(f"\n--- Channel {channel_name} Analysis ---")
                # Look for any unusually strong frequencies
                sorted_indices = np.argsort(power_spectrum)[::-1]  # Sort by power, descending
                top_freqs = frequencies[sorted_indices[:5]]  # Top 5 frequencies
                top_powers = power_spectrum[sorted_indices[:5]]  # Power at those frequencies
                print(f"Strongest frequencies: {', '.join([f'{freq:.1f}Hz' for freq in top_freqs])}")
                
                # Calculate signal statistics
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                print(f"Signal stats - mean: {mean_val:.2e}, std dev: {std_val:.2e}, range: {np.max(channel_data)-np.min(channel_data):.2e}")
            
            # Detect line noise
            noise_ratios = detect_line_noise(power_spectrum, frequencies)
            
            # Store results
            results[channel_name] = {
                "noise_ratios": noise_ratios,
                "has_50hz_noise": noise_ratios.get("50Hz", 0) > threshold_ratio,
                "has_60hz_noise": noise_ratios.get("60Hz", 0) > threshold_ratio,
                "50hz_ratio": noise_ratios.get("50Hz", 0),
                "60hz_ratio": noise_ratios.get("60Hz", 0),
                "threshold_used": threshold_ratio  # Store the threshold that was used
            }
            
            # Print diagnostics
            noise_types = []
            if results[channel_name]["has_50hz_noise"]:
                noise_types.append(f"50Hz ({noise_ratios['50Hz']:.1f}x)")
            if results[channel_name]["has_60hz_noise"]:
                noise_types.append(f"60Hz ({noise_ratios['60Hz']:.1f}x)")
                
            if noise_types:
                print(f"Channel {channel_name} has significant noise: {', '.join(noise_types)}")
            else:
                print(f"Channel {channel_name} has no significant line noise")
                
    return results





def get_noisy_channels_summary(noise_results: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    """
    Get summary of channels affected by different types of noise.
    
    Args:
        noise_results: Results from analyze_channel_noise
        
    Returns:
        Dictionary mapping noise type to list of affected channels
    """
    summary = {
        "50Hz": [],
        "60Hz": [],
        "clean": []
    }
    
    for channel, results in noise_results.items():
        if results.get("has_50hz_noise", False):
            summary["50Hz"].append(channel)
        if results.get("has_60hz_noise", False):
            summary["60Hz"].append(channel)
        if not results.get("has_50hz_noise", False) and not results.get("has_60hz_noise", False):
            summary["clean"].append(channel)
            
    return summary


def plot_channel_spectrum(data: np.ndarray, 
                          channel_names: List[str],
                          channel_indices: Optional[List[int]] = None,
                          sample_rate: int = SAMPLE_RATE,
                          max_freq: int = 100,
                          highlight_line_noise: bool = True,
                          fig=None, 
                          axes=None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the frequency spectrum for selected channels.
    
    Args:
        data: Array of EEG data with shape (channels, samples)
        channel_names: List of channel names corresponding to rows in the data
        channel_indices: Optional list of channel indices to plot, if None plots all
        sample_rate: Sampling rate in Hz
        max_freq: Maximum frequency to display (Hz)
        highlight_line_noise: Whether to highlight the 50Hz and 60Hz ranges
        fig: Optional existing figure to plot on
        axes: Optional existing axes to plot on
        
    Returns:
        Tuple of (figure, axes) that were used for plotting
    """
    if channel_indices is None:
        channel_indices = list(range(min(len(channel_names), len(data))))
    
    # Calculate number of rows and columns for subplots
    n_plots = len(channel_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create or use existing figure and axes
    if fig is None or axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Make it indexable
        axes = axes.flatten()
    
    for i, channel_idx in enumerate(channel_indices):
        if i < len(axes) and channel_idx < len(data) and channel_idx < len(channel_names):
            channel_data = data[channel_idx]
            channel_name = channel_names[channel_idx]
            
            # Perform FFT
            frequencies, power_spectrum = perform_fft(channel_data, sample_rate)
            
            # Plot only up to max_freq
            max_idx = np.where(frequencies <= max_freq)[0][-1]
            ax = axes[i]
            ax.clear()  # Clear existing content
            ax.plot(frequencies[:max_idx], power_spectrum[:max_idx])
            
            # Highlight line noise regions if requested
            if highlight_line_noise:
                for noise_type, (freq_min, freq_max) in LINE_FREQ_RANGES.items():
                    ax.axvspan(freq_min, freq_max, alpha=0.2, color='red')
                    ax.text(freq_min + (freq_max-freq_min)/2, ax.get_ylim()[1]*0.9, 
                            noise_type, ha='center', color='red')
            
            ax.set_title(f"Channel: {channel_name}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.grid(True)
    
    # Hide empty subplots
    for i in range(len(channel_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def generate_noise_report(noise_results: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a human-readable report of line noise analysis.
    
    Args:
        noise_results: Results from analyze_channel_noise
        
    Returns:
        String containing the noise analysis report
    """
    summary = get_noisy_channels_summary(noise_results)
    
    report = "EEG Line Noise Analysis Report\n"
    report += "============================\n\n"
    
    # 50Hz noise summary
    report += f"Channels with 50Hz noise ({len(summary['50Hz'])} channels):\n"
    if summary['50Hz']:
        for channel in summary['50Hz']:
            ratio = noise_results[channel]['50hz_ratio']
            report += f"  - {channel}: {ratio:.1f}x above average power\n"
    else:
        report += "  None\n"
    
    report += "\n"
    
    # 60Hz noise summary
    report += f"Channels with 60Hz noise ({len(summary['60Hz'])} channels):\n"
    if summary['60Hz']:
        for channel in summary['60Hz']:
            ratio = noise_results[channel]['60hz_ratio']
            report += f"  - {channel}: {ratio:.1f}x above average power\n"
    else:
        report += "  None\n"
    
    report += "\n"
    
    # Clean channels
    report += f"Clean channels ({len(summary['clean'])} channels):\n"
    if summary['clean']:
        report += "  " + ", ".join(summary['clean']) + "\n"
    else:
        report += "  None\n"
    
    report += "\n"
    
    # Overall assessment
    total_channels = len(noise_results)
    if total_channels > 0:
        pct_50hz = len(summary['50Hz']) / total_channels * 100
        pct_60hz = len(summary['60Hz']) / total_channels * 100
        pct_clean = len(summary['clean']) / total_channels * 100
        
        report += "Overall Assessment:\n"
        report += f"  - {pct_50hz:.1f}% of channels show 50Hz line noise\n"
        report += f"  - {pct_60hz:.1f}% of channels show 60Hz line noise\n"
        report += f"  - {pct_clean:.1f}% of channels are clean\n\n"
        
        if pct_50hz > 50:
            report += "ISSUE: Widespread 50Hz interference detected. Check grounding and shielding.\n"
        if pct_60hz > 50:
            report += "ISSUE: Widespread 60Hz interference detected. Check grounding and shielding.\n"
        
        if pct_clean > 80:
            report += "GOOD: Most channels are clean of line noise interference.\n"
    
    return report


def check_real_time_noise(data: np.ndarray, 
                         channel_names: List[str],
                         sample_rate: int = SAMPLE_RATE,
                         threshold_ratio: float = LINE_NOISE_THRESHOLD_RATIO,
                         debug: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Analyze most recent EEG data for line noise (for real-time monitoring).
    
    Args:
        data: Array of most recent EEG data samples with shape (channels, samples)
        channel_names: List of channel names corresponding to rows in the data
        sample_rate: Sampling rate in Hz
        threshold_ratio: Ratio threshold for line noise detection (default=LINE_NOISE_THRESHOLD_RATIO)
        
    Returns:
        Dictionary mapping channel names to noise analysis results
    """
    return analyze_channel_noise(data, channel_names, sample_rate, threshold_ratio, debug)


if __name__ == "__main__":
    # This module is designed to be imported, not run directly
    print("This module provides functions for EEG frequency analysis.")
    print("Import it into your application to use the functionality.")
