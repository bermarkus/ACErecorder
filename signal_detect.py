"""
signal_detect.py - Utilities for detecting poorly connected EEG sensors

This module provides functions to detect disconnected or poorly connected EEG electrodes
by analyzing the signal patterns and identifying flatlined channels with the value -187500.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
import os
import json

# Constants
DISCONNECT_VALUE = -187500.0  # The value that indicates a disconnected sensor in converted data
RAW_DISCONNECT_VALUES = [-8388608, -187500.0]  # Possible disconnect values in different formats
TOLERANCE = 0.001  # Tolerance for floating point comparison
FLATLINE_THRESHOLD = 0.05  # Threshold for identifying flatlined signals (standard deviation)

# Channel configurations from the main application
# These match the configurations in the main app
INPUT_ORDER_19 = ["O2", "P8", "T8", "F8", "Fp2", "F4", "C4", "P4", "sync", "ch10", "ch11", "A2", 
                  "Pz", "HR", "ch15", "ch16", "Fz", "Cz", "ch19", "ch20", "ch21", "ch22", "ch23", 
                  "A1", "Fp1", "F3", "C3", "P3", "O1", "P7", "T7", "F7"]

INPUT_ORDER_32 = ["O2", "P8", "A2", "F8", "Fp2", "F4", "C4", "P4", "FC6", "CP6", "CP2", "PO4", 
                  "Pz", "HR", "FC2", "AF4", "Fz", "Cz", "FC1", "AF3", "FC5", "CP5", "CP1", "PO3", 
                  "Fp1", "F3", "C3", "P3", "O1", "P7", "A1", "F7"]

CHANNEL_CONFIGS = {
    "2 Channel Headset": {
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


def is_disconnected(value: float) -> bool:
    """
    Check if a signal value indicates a disconnected sensor.
    
    Args:
        value: Signal value to check
        
    Returns:
        bool: True if the sensor appears to be disconnected, False otherwise
    """
    # Check against all possible disconnect values
    for disconnect_val in RAW_DISCONNECT_VALUES:
        if abs(value - disconnect_val) < TOLERANCE:
            return True
    return False


def detect_disconnected_channels(data: np.ndarray, channel_names: List[str]) -> Dict[str, bool]:
    """
    Analyze EEG data to detect disconnected channels by checking for signal variation patterns.
    
    Args:
        data: Array of EEG data with shape (channels, samples)
        channel_names: List of channel names corresponding to rows in the data
        
    Returns:
        Dictionary mapping channel names to connection status (True=connected, False=disconnected)
    """
    connection_status = {}
    
    # Print diagnostic information
    print(f"Analyzing {len(channel_names)} channels with data shape: {data.shape}")
    
    # We know channels like F3, F4, A1, A2, sync, HR are working properly
    # So we'll analyze their characteristics to better detect all channels
    
    # Thresholds for signal analysis
    # If a signal's standard deviation is below this, it's considered flat
    std_dev_threshold = 0.5
    # Multiplier for the range threshold based on standard deviation
    range_multiplier = 5
    # Minimum meaningful change between consecutive samples
    min_change_threshold = 0.1
    
    for i, channel_name in enumerate(channel_names):
        if i < len(data):
            channel_data = data[i]
            
            # Skip if insufficient data points
            if len(channel_data) < 2:
                connection_status[channel_name] = None
                continue
                
            # Calculate statistics about the signal
            std_dev = np.std(channel_data)
            data_range = np.max(channel_data) - np.min(channel_data)
            
            # Count meaningful changes between consecutive samples
            changes = 0
            for j in range(1, len(channel_data)):
                if abs(channel_data[j] - channel_data[j-1]) > min_change_threshold:
                    changes += 1
            change_ratio = changes / (len(channel_data) - 1) if len(channel_data) > 1 else 0
            
            # Check for patterns that indicate disconnection
            is_flatline = std_dev < std_dev_threshold and data_range < std_dev_threshold * range_multiplier
            low_changes = change_ratio < 0.3  # Less than 30% of samples show meaningful change
            
            # Examine actual values for common disconnection patterns
            # Check if all values are extremely close to each other
            values_close = True
            first_val = channel_data[0]
            for val in channel_data[1:]:
                if abs(val - first_val) > min_change_threshold:
                    values_close = False
                    break
                    
            # Decision logic for connected vs disconnected
            disconnected = (is_flatline and low_changes) or values_close
            
            if disconnected:
                print(f"Channel {channel_name} appears disconnected. StdDev: {std_dev:.6f}, Range: {data_range:.6f}, Changes: {changes}")
                print(f"  Sample values: {channel_data[:5]}")
                connection_status[channel_name] = False
            else:
                print(f"Channel {channel_name} appears connected. StdDev: {std_dev:.6f}, Range: {data_range:.6f}, Changes: {changes}")
                connection_status[channel_name] = True
            
        else:
            # Skip channels that don't have data
            connection_status[channel_name] = None
    
    # Print summary
    connected = sum(1 for status in connection_status.values() if status is True)
    disconnected = sum(1 for status in connection_status.values() if status is False)
    print(f"Connection summary: {connected} connected, {disconnected} disconnected")
    
    return connection_status


def detect_disconnected_from_csv(csv_file: str, 
                                 headset_type: str = "19 Channel") -> Dict[str, bool]:
    """
    Analyze EEG data from a CSV file to detect disconnected channels.
    
    Args:
        csv_file: Path to CSV file with EEG data
        headset_type: Type of headset/configuration used for recording
        
    Returns:
        Dictionary mapping channel names to connection status (True=connected, False=disconnected)
    """
    try:
        # Load data from CSV
        df = pd.read_csv(csv_file)
        
        # Skip the first column (time) and convert to numpy array
        data = df.iloc[:, 1:].values.T  # Transpose to get channels as rows
        
        # Get channel mapping based on headset type
        if headset_type in CHANNEL_CONFIGS:
            config = CHANNEL_CONFIGS[headset_type]
            output_channels = config["output_order"]
        else:
            # If headset type not recognized, use column numbers as channel names
            output_channels = [f"Channel {i+1}" for i in range(data.shape[0])]
        
        # Skip coherence channel if it exists (calculated, not recorded)
        if "Coherence" in output_channels:
            output_channels = [ch for ch in output_channels if ch != "Coherence"]
        
        # Make sure we only use as many channel names as we have data
        output_channels = output_channels[:data.shape[0]]
        
        return detect_disconnected_channels(data, output_channels)
    
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return {}


def get_disconnected_summary(connection_status: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    """
    Get lists of connected and disconnected channels.
    
    Args:
        connection_status: Dictionary mapping channel names to connection status
        
    Returns:
        Tuple of (connected_channels, disconnected_channels)
    """
    connected = [ch for ch, status in connection_status.items() if status is True]
    disconnected = [ch for ch, status in connection_status.items() if status is False]
    
    return connected, disconnected


def generate_connection_report(csv_file: str, headset_type: str = "19 Channel") -> str:
    """
    Generate a human-readable report on channel connection status.
    
    Args:
        csv_file: Path to CSV file with EEG data
        headset_type: Type of headset/configuration used for recording
        
    Returns:
        String containing the connection report
    """
    connection_status = detect_disconnected_from_csv(csv_file, headset_type)
    connected, disconnected = get_disconnected_summary(connection_status)
    
    report = "EEG CHANNEL CONNECTION REPORT\n"
    report += "===========================\n\n"
    report += f"Headset type: {headset_type}\n"
    report += f"Data source: {os.path.basename(csv_file)}\n\n"
    
    if disconnected:
        report += f"ALERT: {len(disconnected)} channels appear to be disconnected!\n"
        report += "Please check the connection of the following electrodes:\n"
        for ch in sorted(disconnected):
            report += f"  - {ch}\n"
        report += "\n"
    else:
        report += "Good news! All channels appear to be properly connected.\n\n"
    
    report += f"Connected channels ({len(connected)}):\n"
    for ch in sorted(connected):
        report += f"  - {ch}\n"
    
    return report


def check_real_time_connection(data: np.ndarray, 
                               channel_names: List[str], 
                               window_size: int = 10) -> Dict[str, bool]:
    """
    Monitor real-time EEG data to detect disconnected channels by checking for signal variation patterns.
    
    Args:
        data: Array of most recent EEG data samples
        channel_names: List of channel names corresponding to rows in the data
        window_size: Number of recent samples to consider
        
    Returns:
        Dictionary mapping channel names to connection status (True=connected, False=disconnected)
    """
    # Print brief diagnostic information
    print(f"\n===== Real-time connection check =====")
    print(f"Data shape: {data.shape}, Channels: {len(channel_names)}")
    
    # Check if we have enough data to make a determination
    if data.shape[1] < 1:
        print("Not enough data samples for connection analysis")
        return {ch: None for ch in channel_names}  # Return all as unknown
        
    # Only look at the most recent window_size samples
    if data.shape[1] > window_size:
        recent_data = data[:, -window_size:]
    else:
        recent_data = data
    
    # Special handling for single-sample case - can't measure variation with just one sample
    if recent_data.shape[1] == 1:
        print("Only one sample available - using single-sample analysis")
        connection_status = {}
        
        # Parameters for single-sample analysis
        min_acceptable_value = 0.2  # Values below this might indicate disconnection
        max_acceptable_value = 10000  # Values above this might indicate disconnection
        
        for i, channel_name in enumerate(channel_names):
            if i < len(recent_data):
                # For single samples, use value range checking
                value = abs(recent_data[i][0])  # Absolute value for simpler comparison
                
                # Consider disconnected if value is extremely large or extremely small
                if value > max_acceptable_value or value < min_acceptable_value:
                    print(f"Channel {channel_name} appears disconnected - unusual value: {recent_data[i][0]}")
                    connection_status[channel_name] = False
                else:
                    print(f"Channel {channel_name} appears connected - value: {recent_data[i][0]}")
                    connection_status[channel_name] = True
            else:
                connection_status[channel_name] = None
                
        return connection_status
    
    # For multiple samples, use the full detection algorithm
    return detect_disconnected_channels(recent_data, channel_names)


def plot_connection_status(connection_status: Dict[str, bool], 
                           title: str = "EEG Channel Connection Status"):
    """
    Create a visual representation of channel connection status.
    
    Args:
        connection_status: Dictionary mapping channel names to connection status
        title: Title for the plot
    """
    channels = list(connection_status.keys())
    status = [1 if connection_status[ch] else 0 for ch in channels]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(channels, status, color=['green' if s else 'red' for s in status])
    
    plt.title(title)
    plt.ylabel('Status (1=Connected, 0=Disconnected)')
    plt.xticks(rotation=90)
    plt.ylim(0, 1.2)
    
    # Add status labels above each bar
    for i, bar in enumerate(bars):
        if status[i]:
            label = 'Connected'
            color = 'darkgreen'
        else:
            label = 'Disconnected'
            color = 'darkred'
        plt.text(bar.get_x() + bar.get_width()/2, 1.05, label, 
                 ha='center', va='bottom', rotation=90, color=color, fontsize=8)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze EEG data for disconnected channels")
    parser.add_argument("csv_file", help="Path to CSV file containing EEG data")
    parser.add_argument("--headset", default="19 Channel", 
                        help="Headset type/configuration (default: 19 Channel)")
    parser.add_argument("--plot", action="store_true", 
                        help="Generate visualization of connection status")
    
    args = parser.parse_args()
    
    # Generate and print the connection report
    report = generate_connection_report(args.csv_file, args.headset)
    print(report)
    
    # Optionally show a plot
    if args.plot:
        connection_status = detect_disconnected_from_csv(args.csv_file, args.headset)
        plot_connection_status(connection_status, 
                              f"Channel Status - {os.path.basename(args.csv_file)}")
