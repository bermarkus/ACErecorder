"""
electrode_monitor.py - Visual monitoring of EEG electrode connections

This module provides a tkinter-based GUI for visualizing EEG electrode connection status
in a topographic layout matching the standard 10-20 system.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import signal_detect

class ElectrodeMonitorWindow:
    """
    A window that displays the status of EEG electrodes in a standard 10-20 system layout.
    Shows electrodes in green when properly connected and red when disconnected.
    """
    
    def __init__(self, master=None, title="EEG Electrode Connection Monitor"):
        """
        Initialize the electrode monitor window.
        
        Args:
            master: Parent tkinter window or None for standalone
            title: Window title
        """
        # Create a new window if no master provided
        if master is None:
            self.root = tk.Tk()
            self.root.title(title)
            self.standalone = True
        else:
            self.root = tk.Toplevel(master)
            self.root.title(title)
            self.root.transient(master)
            self.standalone = False
        
        # Set window size and position
        window_width = 500
        window_height = 500
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(True, True)
        
        # Standard 10-20 system electrode positions
        self.positions = {
            # Front row
            'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
            # F row
            'F7': (-0.8, 0.6), 'F3': (-0.4, 0.6), 'Fz': (0, 0.6), 'F4': (0.4, 0.6), 'F8': (0.8, 0.6),
            # T and C row
            'T7': (-0.8, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.8, 0),
            # P row
            'P7': (-0.8, -0.6), 'P3': (-0.4, -0.6), 'Pz': (0, -0.6), 'P4': (0.4, -0.6), 'P8': (0.8, -0.6),
            # Back row
            'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
            # Ear electrodes
            'A1': (-1.0, 0), 'A2': (1.0, 0),
            # Additional positions for 32 channel systems
            'AF3': (-0.3, 0.7), 'AF4': (0.3, 0.7),
            'FC1': (-0.2, 0.3), 'FC2': (0.2, 0.3),
            'FC5': (-0.6, 0.3), 'FC6': (0.6, 0.3),
            'CP1': (-0.2, -0.3), 'CP2': (0.2, -0.3),
            'CP5': (-0.6, -0.3), 'CP6': (0.6, -0.3),
            'PO3': (-0.3, -0.7), 'PO4': (0.3, -0.7),
            # Special channels - placed at the bottom of display
            'HR': (-0.5, -1.1), 'sync': (0.5, -1.1)
        }
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for the plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create and place summary label
        self.summary_label = ttk.Label(
            self.main_frame, 
            text="Ready to monitor electrode connections", 
            font=('Arial', 10, 'bold')
        )
        self.summary_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Close button for standalone window
        if self.standalone:
            close_button = ttk.Button(self.main_frame, text="Close", command=self.root.destroy)
            close_button.pack(side=tk.BOTTOM, pady=5)
        
        # Create initial plot with all electrodes gray (unknown status)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with all electrodes as unknown (gray)
        unknown_status = {electrode: None for electrode in self.positions.keys()}
        self.update_electrode_display(unknown_status)
    
    def update_electrode_display(self, connection_status: Dict[str, Optional[bool]]):
        """
        Update the electrode display with the current connection status.
        
        Args:
            connection_status: Dictionary mapping electrode names to connection status
                               (True=connected, False=disconnected, None=unknown)
        """
        # Clear the previous plot
        self.ax.clear()
        
        # Draw a head outline
        circle = plt.Circle((0, 0), 0.95, fill=False, color='black', linewidth=2)
        self.ax.add_patch(circle)
        
        # Draw nose as a small triangle at the top
        nose_x = [0, -0.1, 0.1, 0]
        nose_y = [1.0, 1.15, 1.15, 1.0]
        self.ax.plot(nose_x, nose_y, 'k-', linewidth=2)
        
        # Draw ears as small shapes on the sides
        left_ear_x = [-1.05, -1.15, -1.15, -1.05]
        left_ear_y = [0.1, 0.05, -0.05, -0.1]
        right_ear_x = [1.05, 1.15, 1.15, 1.05]
        right_ear_y = [0.1, 0.05, -0.05, -0.1]
        self.ax.plot(left_ear_x, left_ear_y, 'k-', linewidth=2)
        self.ax.plot(right_ear_x, right_ear_y, 'k-', linewidth=2)
        
        # Plot each electrode with appropriate color
        for electrode, position in self.positions.items():
            x, y = position
            
            # Determine color and size based on connection status
            if electrode in connection_status:
                status = connection_status[electrode]
                if status is True:  # Connected
                    color = 'green'
                    size = 300
                    alpha = 1.0
                elif status is False:  # Disconnected
                    color = 'red'
                    size = 300
                    alpha = 1.0
                else:  # Unknown or None
                    color = 'gray'
                    size = 200
                    alpha = 0.7
            else:
                # Electrode not in the current montage
                color = 'lightgray'
                size = 100
                alpha = 0.5
            
            # Plot electrode with appropriate color
            self.ax.scatter(x, y, s=size, c=color, alpha=alpha, edgecolors='black', zorder=2)
            
            # Add electrode label
            self.ax.text(x, y, electrode, ha='center', va='center', fontsize=8, 
                         weight='bold', color='black', zorder=3)
        
        # Set plot limits and remove axes
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off')
        
        # Update the canvas
        self.canvas.draw()
        
        # Update summary label
        if connection_status:
            connected = sum(1 for status in connection_status.values() if status is True)
            disconnected = sum(1 for status in connection_status.values() if status is False)
            total = sum(1 for status in connection_status.values() if status is not None)
            
            if disconnected > 0:
                self.summary_label.config(
                    text=f"ALERT: {disconnected} electrodes disconnected! ({connected}/{total} connected)",
                    foreground='red'
                )
            else:
                self.summary_label.config(
                    text=f"All electrodes connected properly ({connected}/{total})",
                    foreground='green'
                )
        else:
            self.summary_label.config(
                text="No connection data available",
                foreground='black'
            )

    def show(self):
        """Show the window and start the main loop if standalone"""
        if self.standalone:
            self.root.mainloop()
        else:
            self.root.grab_set()  # Make window modal

    def update_from_data(self, data: np.ndarray, 
                         channel_names: List[str], 
                         window_size: int = 10):
        """
        Update electrode display from raw EEG data.
        
        Args:
            data: Array of EEG data with shape (channels, samples)
            channel_names: List of channel names corresponding to rows in the data
            window_size: Number of recent samples to analyze
        """
        connection_status = signal_detect.check_real_time_connection(
            data, channel_names, window_size)
        self.update_electrode_display(connection_status)

    def update_from_csv(self, csv_file: str, headset_type: str = "19 Channel"):
        """
        Update electrode display from a CSV file.
        
        Args:
            csv_file: Path to CSV file with EEG data
            headset_type: Type of headset/configuration used for recording
        """
        connection_status = signal_detect.detect_disconnected_from_csv(csv_file, headset_type)
        self.update_electrode_display(connection_status)


def show_connection_monitor(master=None, data=None, channel_names=None, 
                           csv_file=None, headset_type="19 Channel"):
    """
    Show the electrode connection monitor window with the provided data.
    
    Args:
        master: Parent tkinter window or None for standalone
        data: (Optional) Array of EEG data with shape (channels, samples)
        channel_names: (Optional) List of channel names if data is provided
        csv_file: (Optional) Path to CSV file with EEG data
        headset_type: Type of headset/configuration (default: 19 Channel)
    
    Returns:
        The ElectrodeMonitorWindow instance
    """
    monitor = ElectrodeMonitorWindow(master)
    
    if data is not None and channel_names is not None:
        # Update from raw data
        monitor.update_from_data(data, channel_names)
    elif csv_file is not None:
        # Update from CSV file
        monitor.update_from_csv(csv_file, headset_type)
    
    monitor.show()
    return monitor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize EEG electrode connection status")
    parser.add_argument("--csv", help="Path to CSV file containing EEG data")
    parser.add_argument("--headset", default="19 Channel", 
                        help="Headset type/configuration (default: 19 Channel)")
    
    args = parser.parse_args()
    
    # Show the electrode monitor window
    if args.csv:
        monitor = ElectrodeMonitorWindow()
        monitor.update_from_csv(args.csv, args.headset)
        monitor.show()
    else:
        # Just show the window with unknown status
        monitor = ElectrodeMonitorWindow()
        monitor.show()
