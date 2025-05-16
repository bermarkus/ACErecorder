"""
signal_monitor.py - Real-time visualization of EEG signal waveforms

This module provides a tkinter-based GUI for visualizing streaming EEG signal data
in real-time, displaying multiple channels as time-series plots.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Make sure to use TkAgg backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple
import time
import signal_detect

class SignalMonitorWindow:
    """A window that displays real-time EEG signal waveforms for multiple channels."""
    
    def __init__(self, master=None, title="EEG Signal Monitor", sample_rate=None, time_window=10):
        """
        Initialize the signal monitor window.
        
        Args:
            master: Parent tkinter window or None for standalone
            title: Window title
            sample_rate: Sampling rate in Hz (must be provided by the caller)
            time_window: Time window to display in seconds (default: 10 seconds)
        """
        # Ensure sample_rate is provided
        if sample_rate is None:
            raise ValueError("sample_rate must be specified when creating SignalMonitorWindow")
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
        window_width = 800
        window_height = 600
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(True, True)
        
        # Make window always stay on top
        self.root.attributes("-topmost", True)
        
        # Configure window to have proper controls
        self.root.update()
        self._configure_window_controls()
            
        # Set the close button to minimize instead of close
        self.root.protocol("WM_DELETE_WINDOW", self.minimize)
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for controls at the top
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Add time window slider
        self.time_frame = ttk.Frame(self.controls_frame)
        self.time_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.time_label = ttk.Label(
            self.time_frame,
            text=f"Time Window: {time_window} seconds",
            font=("Arial", 9)
        )
        self.time_label.pack(side=tk.TOP, anchor=tk.W)
        
        self.time_var = tk.DoubleVar(value=time_window)
        self.time_slider = ttk.Scale(
            self.time_frame,
            from_=1.0,  # Minimum 1 second
            to=30.0,    # Maximum 30 seconds
            orient=tk.HORIZONTAL,
            variable=self.time_var,
            length=150,
            command=self._update_time_label
        )
        self.time_slider.pack(side=tk.TOP, fill=tk.X)
        
        # Add a status label
        self.status_label = ttk.Label(
            self.controls_frame,
            text="Signal monitor ready",
            font=("Arial", 10, "bold")
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Add minimize button
        self.minimize_button = ttk.Button(
            self.controls_frame, 
            text="Minimize", 
            command=self.minimize
        )
        self.minimize_button.pack(side=tk.RIGHT, padx=5)
        
        # Create frame for the plots
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Store parameters
        self.sample_rate = sample_rate
        self.time_window = time_window
        self.buffer_size = int(time_window * sample_rate)
        
        # Initialize data buffer for each channel (will be populated later)
        self.channel_data = {}
        self.channel_plots = {}
        self.active_channels = []
        
        # Create initial plot
        self._create_figure()
        
        # Register cleanup handler for when the window is destroyed
        self.root.bind("<Destroy>", self._on_destroy)
        
    def _create_figure(self):
        """Create the matplotlib figure and subplots for signal display."""
        # Ensure matplotlib is in non-interactive mode
        plt.ioff()
        
        # Create a figure with subplots (will be adjusted based on channel count)
        self.fig = Figure(figsize=(10, 8), constrained_layout=True)
        
        # Store the figure canvas in our frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _update_time_label(self, *args):
        """Update the time window label when the slider changes."""
        value = self.time_var.get()
        self.time_window = value
        self.buffer_size = int(value * self.sample_rate)
        self.time_label.config(text=f"Time Window: {value:.1f} seconds")
        # Resize the data buffers if needed
        self._resize_buffers()
        
    def _resize_buffers(self):
        """Resize the data buffers when the time window changes."""
        for channel in self.channel_data:
            current_data = self.channel_data[channel]
            if len(current_data) > self.buffer_size:
                # Keep only the most recent data
                self.channel_data[channel] = current_data[-self.buffer_size:]
            elif len(current_data) < self.buffer_size:
                # Extend with zeros if needed
                padding = np.zeros(self.buffer_size - len(current_data))
                self.channel_data[channel] = np.concatenate([padding, current_data])
    
    def minimize(self):
        """Minimize the window instead of closing it."""
        self.root.iconify()
    
    def _configure_window_controls(self):
        """Configure window to have proper minimize button on Windows."""
        if sys.platform == 'win32':
            try:
                # Import Windows API modules
                import ctypes
                from ctypes import windll
                from ctypes.wintypes import HWND, LONG
                
                # Window style constants
                GWL_STYLE = -16
                WS_MINIMIZEBOX = 0x00020000
                WS_MAXIMIZEBOX = 0x00010000
                
                # Get window handle
                hwnd = HWND(int(self.root.winfo_id()))
                
                # Get current style
                style = windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
                
                # Add minimize box to style, remove maximize box
                style |= WS_MINIMIZEBOX
                style &= ~WS_MAXIMIZEBOX
                
                # Set the new style
                windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            except Exception as e:
                print(f"Could not configure window controls: {e}")
    
    def _on_destroy(self, event):
        """Clean up matplotlib resources when window is destroyed."""
        # Only process if this is our window being destroyed
        if event.widget == self.root:
            print("Cleaning up signal monitor resources...")
            try:
                # Clean up plot resources
                plt.close(self.fig)
                self.fig = None
                self.canvas = None
                self.channel_plots = {}
            except Exception as e:
                print(f"Error cleaning up matplotlib: {e}")
    
    def initialize_channels(self, channel_names: List[str]):
        """
        Initialize the display with the specified channels.
        
        Args:
            channel_names: List of channel names to display
        """
        # Store active channels
        self.active_channels = channel_names
        
        # Create empty data buffers for each channel
        for channel in channel_names:
            self.channel_data[channel] = np.zeros(self.buffer_size)
        
        # Create subplots for each channel
        self._create_channel_plots()
        
    def _create_channel_plots(self):
        """Create matplotlib subplots for each active channel."""
        # Clear any existing plots
        self.fig.clf()
        self.channel_plots = {}
        
        # Determine layout based on number of channels
        n_channels = len(self.active_channels)
        if n_channels <= 0:
            return
            
        # Create time vector (in seconds)
        self.time_vec = np.linspace(-self.time_window, 0, self.buffer_size)
        
        # Create subplots in a vertical stack
        subplot_height = 0.8 / n_channels  # Allow space for labels
        
        for i, channel in enumerate(self.active_channels):
            # Create subplot
            ax = self.fig.add_subplot(n_channels, 1, i+1)
            
            # Set y-label with channel name
            ax.set_ylabel(channel, rotation=0, labelpad=10, fontsize=9)
            
            # Only show x-axis on bottom plot
            if i < n_channels - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (seconds)")
            
            # Create the line plot
            line, = ax.plot(self.time_vec, self.channel_data[channel], 'b-', linewidth=0.8)
            
            # Set reasonable y-limits
            ax.set_ylim(-150, 150)
            
            # Set x-limits based on time window
            ax.set_xlim(-self.time_window, 0)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Store the plot
            self.channel_plots[channel] = line
            
        # Adjust layout
        self.fig.tight_layout()
        
        # Update canvas
        self.canvas.draw_idle()
    
    def update_from_data(self, data: np.ndarray, channel_names: List[str], max_points: int = None):
        """
        Update signal display with new data.
        
        Args:
            data: Array of EEG data with shape (channels, samples)
            channel_names: List of channel names corresponding to rows in the data
            max_points: Maximum number of new points to add (default is all points)
        """
        if data is None or len(data) == 0 or len(channel_names) == 0:
            return
        
        # Make a copy of the data to prevent modification during processing
        data_copy = data.copy()
        names_copy = channel_names.copy()
        
        # Schedule the actual processing to happen in the main thread
        self.root.after_idle(lambda: self._thread_safe_update(data_copy, names_copy, max_points))
    
    def _thread_safe_update(self, data: np.ndarray, channel_names: List[str], max_points: int = None):
        """Thread-safe implementation of data update that runs on the main thread."""
        try:
            # If channels aren't initialized yet, initialize them
            if not self.active_channels and len(channel_names) > 0:
                self.initialize_channels(channel_names)
            
            # If no channels or no data, return
            if not self.active_channels or data.shape[0] == 0:
                return
            
            # Get number of new samples
            n_samples = data.shape[1]
            if max_points is not None:
                n_samples = min(n_samples, max_points)
            
            if n_samples <= 0:
                return
            
            # Update data buffers for each channel
            for i, channel in enumerate(channel_names):
                if i >= data.shape[0] or channel not in self.channel_data:
                    continue
                    
                # Get current buffer
                buffer = self.channel_data[channel]
                
                # Shift data to make room for new samples
                buffer = np.roll(buffer, -n_samples)
                
                # Add new samples at the end
                buffer[-n_samples:] = data[i, :n_samples]
                
                # Update buffer
                self.channel_data[channel] = buffer
                
                # Update plot if it exists
                if channel in self.channel_plots:
                    self.channel_plots[channel].set_ydata(buffer)
            
            # Update the canvas - use draw_idle for better performance
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating signal display: {e}")
            import traceback
            traceback.print_exc()
    
    def show(self):
        """Show the window and start the main loop if standalone."""
        if self.standalone:
            self.root.mainloop()
        else:
            self.root.deiconify()  # Ensure window is visible
            self.root.lift()       # Bring to front


def show_signal_monitor(master=None, data=None, channel_names=None, sample_rate=250, time_window=10):
    """
    Show the signal monitor window with the provided data.
    
    Args:
        master: Parent tkinter window or None for standalone
        data: (Optional) Initial EEG data with shape (channels, samples)
        channel_names: (Optional) List of channel names if data is provided
        sample_rate: Sampling rate in Hz (default: 250 Hz)
        time_window: Time window to display in seconds (default: 10 seconds)
    
    Returns:
        The SignalMonitorWindow instance
    """
    monitor = SignalMonitorWindow(master, sample_rate=sample_rate, time_window=time_window)
    
    if data is not None and channel_names is not None:
        # Initialize with provided data
        monitor.initialize_channels(channel_names)
        monitor.update_from_data(data, channel_names)
    
    monitor.show()
    return monitor


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Visualize real-time EEG signals")
    parser.add_argument("--time", type=float, default=10.0, 
                        help="Time window in seconds (default: 10)")
    parser.add_argument("--sample-rate", type=int, default=512, 
                        help="Sample rate in Hz (default: 512)")
    
    args = parser.parse_args()
    
    # Create demo data - simulated sine waves
    sample_rate = args.sample_rate
    time_window = args.time
    n_samples = int(time_window * sample_rate)
    t = np.linspace(0, time_window, n_samples)
    
    # Create 4 channels of simulated data
    channel_names = ["Ch1", "Ch2", "Ch3", "Ch4"]
    data = np.zeros((len(channel_names), n_samples))
    
    # Channel 1: 10 Hz sine wave
    data[0, :] = 100 * np.sin(2 * np.pi * 10 * t)
    
    # Channel 2: 5 Hz sine wave
    data[1, :] = 80 * np.sin(2 * np.pi * 5 * t)
    
    # Channel 3: 15 Hz sine wave
    data[2, :] = 120 * np.sin(2 * np.pi * 15 * t)
    
    # Channel 4: 50 Hz sine wave (simulated noise)
    data[3, :] = 50 * np.sin(2 * np.pi * 50 * t)
    
    # Just show the window with demo data
    monitor = SignalMonitorWindow(sample_rate=sample_rate, time_window=time_window)
    monitor.initialize_channels(channel_names)
    monitor.update_from_data(data, channel_names)
    monitor.show()
