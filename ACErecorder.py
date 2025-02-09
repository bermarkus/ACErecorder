import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import time
import threading
import os
import serial
from serial.tools import list_ports
from datetime import datetime
import json
import sys
import traceback
import webbrowser
import pyedflib
import copy

# Try to import MNE-related packages with error handling
try:
    import mne
    import mne_connectivity
    HAS_MNE = True
except ImportError as e:
    print(f"Warning: MNE import failed: {e}")
    HAS_MNE = False

# Try to import pyedflib with error handling
try:
    import pyedflib
except ImportError as e:
    print(f"Warning: pyedflib import failed: {e}")
    pyedflib = None

def compute_coherence(data, fs, fmin=7, fmax=12, duration=1, overlap=0):
    """Compute coherence between two channels (Fp1 and Fp2)"""
    if not HAS_MNE:
        return 0.0  # Return default value if MNE is not available
        
    try:
        mne.set_log_level("CRITICAL")
        
        ch_names = ['Fp1', 'Fp2']
        ch_types = ['eeg', 'eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, overlap=overlap, verbose="ERROR")

        con_obj = mne_connectivity.spectral_connectivity_epochs(
            epochs, method='coh', sfreq=fs, fmin=fmin, fmax=fmax,
            faverage=True, mt_adaptive=True, n_jobs=1, verbose="CRITICAL"
        )
        con_values = con_obj.get_data(output='dense')[:, :, 0]
        return 4 * np.mean(con_values)
    except Exception as e:
        print(f"Coherence calculation error: {e}")
        return 0.0

# Channel configurations
INPUT_ORDER_19 = ["O2", "P8", "T8", "F8", "Fp2", "F4", "C4", "P4", "sync", "ch10", "ch11", "A2", 
                  "Pz", "HR", "ch15", "ch16", "Fz", "Cz", "ch19", "ch20", "ch21", "ch22", "ch23", 
                  "A1", "Fp1", "F3", "C3", "P3", "O1", "P7", "T7", "F7"]

INPUT_ORDER_32 = ["O2", "P8", "A2", "F8", "Fp2", "F4", "C4", "P4", "FC6", "CP6", "CP2", "PO4", 
                  "Pz", "HR", "FC2", "AF4", "Fz", "Cz", "FC1", "AF3", "FC5", "CP5", "CP1", "PO3", 
                  "Fp1", "F3", "C3", "P3", "O1", "P7", "A1", "F7"]

CHANNEL_CONFIGS = {
    "2 Channel": {
        "input_order": INPUT_ORDER_19,  # Using 19ch input order for 2ch mode
        "output_order": ["Fp1", "Fp2", "Coherence"],  # Removed HR and sync
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

class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size and position
        window_width = 300
        window_height = 200
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create a frame with a border
        frame = tk.Frame(self.root, borderwidth=2, relief='solid')
        frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Add loading text
        tk.Label(frame, text="ACErecorder", font=('Helvetica', 16, 'bold')).pack(pady=20)
        tk.Label(frame, text="Loading...", font=('Helvetica', 10)).pack()
        
        # Add a progress bar
        self.progress = ttk.Progressbar(frame, length=200, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()
        
        # Force update
        self.root.update()

    def destroy(self):
        self.root.destroy()

class ACErecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ACErecorder")
        
        # Initialize settings file path first
        try:
            if getattr(sys, 'frozen', False):
                # If the application is run as a bundle
                app_dir = os.path.dirname(sys.executable)
            else:
                # If the application is run from a Python interpreter
                app_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Use a single settings file for all instances
            self.settings_file = os.path.join(app_dir, "settings.json")
        except Exception as e:
            print(f"Error setting up app directory: {e}")
            app_dir = os.path.dirname(os.path.abspath(__file__))
            self.settings_file = os.path.join(app_dir, "settings.json")

        # Load settings
        self.settings = self.load_settings()
        self.custom_configs = self.settings.get("configurations", {}).copy()
        
        # Create variables
        self.channel_mode_var = tk.StringVar(value=self.settings.get("headset", "19 Channel"))
        
        # Handle icon setting
        try:
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = sys._MEIPASS  # Use PyInstaller's special path
            else:
                # Running as script
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            icon_path = os.path.join(base_path, "LogoSquare3.ico")
            if os.path.exists(icon_path):
                # Set window icons
                self.root.iconbitmap(default=icon_path)
                self.root.iconbitmap(icon_path)
                
                # Set taskbar icon
                try:
                    import ctypes
                    myappid = u'alphacoherence.eegrecorder.1.0'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                except Exception as e:
                    print(f"Failed to set app ID: {e}")
            else:
                print(f"Icon not found at: {icon_path}")
        except Exception as e:
            print(f"Could not set icon: {e}")

        # Calculate scaled dimensions (60% of original)
        logo_original_width = 900
        logo_original_height = 97
        scale_factor = 0.60
        logo_width = int(logo_original_width * scale_factor)
        logo_height = int(logo_original_height * scale_factor)

        # Set window size based on logo width
        self.root.geometry(f"{logo_width}x450")
        self.root.resizable(False, False)

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="0")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add logo
        try:
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Head v28 (2).png")
            pil_image = Image.open(logo_path)
            pil_image = pil_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(pil_image)
            logo_label = tk.Label(self.main_frame, image=self.logo_image, cursor="hand2")
            logo_label.grid(row=0, column=0, columnspan=2, pady=(0,10), sticky='ew')
            logo_label.bind("<Button-1>", lambda e: webbrowser.open("https://alphacoherence.com"))
        except Exception as e:
            print(f"Could not load logo: {e}")

        # Content frame with padding
        content_frame = ttk.Frame(self.main_frame, padding="0 0 20 0")
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20)
        
        # Status variables
        self.recording = False
        self.board = None
        self.com_port = None
        self.filename_suffix = ""
        self.port_mapping = {}
        self.recording_start_time = None
        self.annotations = []
        self.current_duration_index = None  # Track current duration annotation
        
        # Ensure EEG files directory exists
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            app_dir = os.path.dirname(sys.executable)  # Use executable's directory instead of _MEIPASS
        else:
            # Running as script
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.eeg_dir = os.path.join(app_dir, "EEG files")
        os.makedirs(self.eeg_dir, exist_ok=True)
        
        # Initialize variables
        self.status_var = tk.StringVar(value="Initializing...")
        self.recording_duration_var = tk.StringVar(value="Duration: 0 seconds")  # Renamed from duration_var
        self.sample_rate_var = tk.StringVar(value="Sample Rate: -- Hz")
        self.channel_count_var = tk.StringVar(value="Channels: --")
        self.mode_description_var = tk.StringVar(value="")
        
        # Headset Selection
        ttk.Label(content_frame, text="Headset:").grid(row=0, column=0, pady=5, sticky=tk.W)
        self.channel_mode_combo = ttk.Combobox(content_frame, textvariable=self.channel_mode_var, 
                                             values=list(self.custom_configs.keys()), state="readonly")
        self.channel_mode_combo.grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))
        self.channel_mode_combo.bind('<<ComboboxSelected>>', self.on_channel_mode_change)
        
        # Filename entry
        ttk.Label(content_frame, text="Recording Name (optional):").grid(row=1, column=0, pady=5, sticky=tk.W)
        self.filename_var = tk.StringVar()
        self.filename_entry = ttk.Entry(content_frame, textvariable=self.filename_var)
        self.filename_entry.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E))
        
        # COM Port selection
        ttk.Label(content_frame, text="COM Port:").grid(row=2, column=0, pady=5, sticky=tk.W)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(content_frame, textvariable=self.port_var, state="readonly")
        self.port_combo.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))
        self.port_combo.bind('<<ComboboxSelected>>', lambda e: self.on_port_change())
        
        # Status label
        self.status_label = ttk.Label(content_frame, textvariable=self.status_var, wraplength=380)
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10, sticky='w')
        
        # Recording info frame
        info_frame = ttk.LabelFrame(content_frame, text="Recording Info", padding="5")
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Duration
        ttk.Label(info_frame, textvariable=self.recording_duration_var).grid(row=0, column=0, padx=5)  # Updated reference
        
        # Sample Rate
        ttk.Label(info_frame, textvariable=self.sample_rate_var).grid(row=0, column=1, padx=5)
        
        # Channel Count
        ttk.Label(info_frame, textvariable=self.channel_count_var).grid(row=0, column=2, padx=5)
        
        # Mode Description
        ttk.Label(info_frame, textvariable=self.mode_description_var, wraplength=380).grid(row=1, column=0, columnspan=3, padx=5)
        
        # Record button
        self.record_button = ttk.Button(content_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Annotation Controls Frame
        annotation_controls_frame = ttk.LabelFrame(content_frame, text="Annotation Controls")
        annotation_controls_frame.grid(row=6, column=0, columnspan=2, pady=5, sticky='ew')

        # Duration Annotation Section
        duration_frame = ttk.Frame(annotation_controls_frame)
        duration_frame.pack(fill='x', pady=2)
        
        ttk.Label(duration_frame, text="Duration Annotation:").pack(side='left')
        
        self.duration_var = tk.BooleanVar(value=False)
        self.duration_toggle = ttk.Checkbutton(
            duration_frame,
            text="Start/Stop",
            variable=self.duration_var,
            command=self.toggle_duration_annotation
        )
        self.duration_toggle.pack(side='left', padx=(5,5))
        
        self.duration_entry = ttk.Entry(duration_frame, width=25)
        self.duration_entry.pack(side='left')
        self.duration_entry.insert(0, 'Duration name')

        # Instant Marker Section
        marker_frame = ttk.Frame(annotation_controls_frame)
        marker_frame.pack(fill='x', pady=2)
        
        ttk.Label(marker_frame, text="Instant Marker:").pack(side='left')
        
        self.marker_button = ttk.Button(
            marker_frame,
            text="Add Marker",
            command=self.add_instant_marker
        )
        self.marker_button.pack(side='left', padx=(5,5))
        
        self.marker_entry = ttk.Entry(marker_frame, width=25)
        self.marker_entry.pack(side='left')
        self.marker_entry.insert(0, 'Marker name')
        
        # Configure grid
        content_frame.columnconfigure(1, weight=1)
        
        # Create menu bar
        self.create_menu()
        
        # Initialize COM ports
        self.update_com_ports()
        
        # Update channel info
        self.update_channel_info()
        
        # Refresh COM ports every 5 seconds
        self.root.after(5000, self.periodic_port_update)

    def update_channel_info(self):
        """Update channel count and mode info"""
        mode = self.channel_mode_var.get()
        if mode in self.custom_configs:
            channel_count = len(self.custom_configs[mode]["output_order"])
            self.channel_count_var.set(f"Channels: {channel_count}")
            
            # Update description if available
            if "description" in self.custom_configs[mode]:
                self.mode_description_var.set(self.custom_configs[mode]["description"])
            else:
                self.mode_description_var.set("")
        
    def on_channel_mode_change(self, event=None):
        """Handle channel mode changes"""
        self.update_channel_info()
        self.save_settings()  # Save settings when headset type changes

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Custom Configuration", command=self.configure_custom_headset)
        file_menu.add_command(label="Manage Configurations", command=self.manage_configurations)
        file_menu.add_separator()
        file_menu.add_command(label="Refresh COM Ports", command=self.update_com_ports)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def update_com_ports(self):
        """Update the list of available COM ports"""
        # Get all active ports with their descriptions
        available_ports = list(serial.tools.list_ports.comports())
        
        # Find all Silicon Labs devices and USB Serial Devices
        eeg_ports = []
        for port in available_ports:
            if port.device:
                if "Silicon Labs CP210x USB to UART Bridge" in port.description:
                    eeg_ports.append((port, "EEG optical dongle"))
                elif "USB Serial Device" in port.description:
                    eeg_ports.append((port, "USB Serial Device"))
        
        if eeg_ports:
            # Show all compatible devices
            descriptions = []
            self.port_mapping = {}
            for port, device_type in eeg_ports:
                description = f"{port.device} - {device_type}"
                descriptions.append(description)
                self.port_mapping[description] = port.device
            
            self.port_combo['values'] = descriptions
            
            # If there's a saved port and it's still available, use it
            saved_port = self.settings.get("com_port")
            if saved_port and saved_port in descriptions:
                self.port_var.set(saved_port)
            elif saved_port:
                # If saved port exists but isn't in current list, try to match just the COM port number
                saved_com = saved_port.split(" - ")[0]  # Get just the COM port part
                for desc in descriptions:
                    if desc.startswith(saved_com):
                        self.port_var.set(desc)
                        break
                else:
                    # If no match found, use first available port
                    self.port_var.set(descriptions[0])
            else:
                # No saved port, use first available
                self.port_var.set(descriptions[0])
            
            self.status_var.set("Ready to record")
            self.record_button.config(state="normal")
        else:
            # No compatible devices found
            self.port_combo['values'] = []
            self.port_mapping = {}
            self.port_var.set('')
            self.status_var.set("No compatible devices connected")
            self.record_button.config(state="disabled")

    def periodic_port_update(self):
        """Periodically update COM ports if not recording"""
        if not self.recording:
            self.update_com_ports()
        self.root.after(5000, self.periodic_port_update)

    def show_about(self):
        messagebox.showinfo("About", "ACErecorder\nVersion 1.0\nA simple EEG recording application.")

    def generate_filename(self):
        """Generate filename with date-time prefix"""
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # Added seconds with %S
        suffix = self.filename_var.get().strip()
        if suffix:
            filename = f"{current_time}_{suffix}.bdf"
        else:
            filename = f"{current_time}.bdf"
        return os.path.join(self.eeg_dir, filename)

    def toggle_recording(self):
        if not self.recording:
            try:
                self.start_recording()
                self.record_button.config(text="Stop Recording")
                self.recording = True
            except Exception as e:
                self.recording = False
                messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
        else:
            self.stop_recording()
            self.record_button.config(text="Start Recording")
            self.recording = False

    def update_duration(self):
        if self.recording:
            duration = time.time() - self.recording_start_time
            self.recording_duration_var.set(f"{duration:.0f} seconds")  # Updated reference
            self.root.after(1000, self.update_duration)
        else:
            self.recording_duration_var.set("0 seconds")  # Updated reference

    def start_recording(self):
        if not isinstance(self.recording, bool):
            raise TypeError(f"Invalid recording state type: {type(self.recording).__name__}")
        
        if self.recording:
            return
        
        try:
            # Set critical timing FIRST
            self.recording_start_time = time.time()
            
            # Then update state
            self.recording = True
            
            # Get the actual COM port from the selection
            selected_port = self.port_mapping[self.port_var.get()]
            
            # Initialize board
            params = BrainFlowInputParams()
            params.serial_port = selected_port
            board_id = brainflow.BoardIds.FREEEEG32_BOARD.value
            self.board = BoardShim(board_id, params)
            
            # Prepare and start session
            self.board.prepare_session()
            self.board.start_stream()
            
            # Wait briefly and check if data is being received
            time.sleep(1)  # Wait for 1 second
            data = self.board.get_board_data()
            if data.size == 0:
                self.board.stop_stream()
                self.board.release_session()
                messagebox.showerror("Error", "No data received from the optical dongle.\nPlease check your connections and try again.")
                return
            
            # Get board info
            self.eeg_channels = self.board.get_eeg_channels(board_id)
            self.num_channels = len(self.eeg_channels)
            self.sample_rate = BoardShim.get_sampling_rate(board_id)
            
            # Update sample rate display
            self.sample_rate_var.set(f"Sample Rate: {self.sample_rate} Hz")
            
            # Channel labels
            self.channel_labels = ['O2', 'P8', 'A2', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'FC6', 'CP6', 
                                 'CP2', 'PO4', 'Pz', 'POz', 'FC2', 'AF4', 'Fz', 'Cz', 'FC1', 'AF3', 
                                 'FC5', 'CP5', 'CP1', 'PO3', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'P7', 'A1', 'F7']
            
            if len(self.channel_labels) < self.num_channels:
                self.channel_labels.extend([f'ch{i + 1}' for i in range(len(self.channel_labels), self.num_channels)])
            
            # Generate output filename
            self.output_file = self.generate_filename()
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self.record_data)
            self.record_thread.start()
            
            # Start duration counter
            self.update_duration()
            
            # Update UI
            self.status_var.set(f"Recording to {self.output_file}...")
            self.filename_entry.config(state="disabled")
            self.port_combo.config(state="disabled")
            
        except Exception as e:
            self.recording = False
            self.recording_start_time = None
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def record_data(self):
        """Record data from the board"""
        mode = self.channel_mode_var.get()
        output_channels = self.custom_configs[mode]["output_order"]
        
        # Ensure we have output channels defined
        if not output_channels:
            messagebox.showerror("Error", "No channels configured. Please configure channels in File > Configure Custom Headset")
            self.recording = False
            return
            
        # Get the input channels based on mode
        input_order = self.custom_configs[mode]["input_order"]
        
        # Create mapping from input to output channels
        channel_mapping = {}  # Maps output index to input index
        for out_idx, out_channel in enumerate(output_channels):
            if out_channel in input_order:
                channel_mapping[out_idx] = input_order.index(out_channel)
        
        # Initialize buffer for EDF file
        buffer = np.zeros((len(output_channels), self.sample_rate))
        buffer_count = 0
        no_data_count = 0
        
        # Get indices of EEG channels from board
        self.eeg_channels = self.board.get_eeg_channels(self.board.board_id)
        
        # Create EDF file
        try:
            f = pyedflib.EdfWriter(self.output_file, len(output_channels), file_type=pyedflib.FILETYPE_BDFPLUS)
            
            # Set header fields
            f.setTechnician('X')
            f.setRecordingAdditional('X')
            f.setPatientName('X')
            f.setPatientCode('X')
            f.setPatientAdditional('X')
            f.setAdmincode('X')
            f.setEquipment('FreeEEG32')
            f.setSignalHeaders([{
                'label': ch_name,
                'dimension': 'uV',
                'sample_rate': self.sample_rate,
                'physical_min': -187500,
                'physical_max': 187500,
                'digital_min': -8388608,
                'digital_max': 8388607,
                'transducer': '',
                'prefilter': ''
            } for ch_name in output_channels])
            
            while self.recording:
                try:
                    data = self.board.get_board_data()
                    new_samples = data.shape[1]
                    
                    # Check if we're receiving data
                    if new_samples == 0:
                        no_data_count += 1
                        if no_data_count >= 100:  # After ~1 second of no data (assuming 10ms sleep)
                            self.recording = False
                            messagebox.showerror("Error", "Lost connection with the optical dongle.\nRecording has been stopped.")
                            break
                        time.sleep(0.01)
                        continue
                    else:
                        no_data_count = 0  # Reset counter when data is received
                    
                    if buffer_count + new_samples >= self.sample_rate:
                        samples_to_fill = self.sample_rate - buffer_count
                        
                        # Map input channels to output channels
                        for out_idx, in_idx in channel_mapping.items():
                            if in_idx < len(self.eeg_channels):
                                buffer[out_idx, buffer_count:] = data[self.eeg_channels[in_idx], :samples_to_fill]
                        
                        # Calculate coherence for 2 Channel mode
                        if mode == "2 Channel":
                            # Get Fp1 and Fp2 data
                            fp1_idx = output_channels.index("Fp1")
                            fp2_idx = output_channels.index("Fp2")
                            coherence_idx = output_channels.index("Coherence")
                            
                            # Prepare data for coherence calculation
                            coherence_data = np.vstack((buffer[fp1_idx, :], buffer[fp2_idx, :]))
                            
                            # Calculate coherence
                            try:
                                coherence_value = compute_coherence(coherence_data, self.sample_rate)
                                # Fill the coherence buffer with the computed value
                                buffer[coherence_idx, :] = coherence_value
                            except Exception as e:
                                print(f"Coherence calculation error: {e}")
                                buffer[coherence_idx, :] = 0
                        
                        # Write the data
                        try:
                            f.writeSamples(buffer)
                            print(f"Wrote {self.sample_rate} samples to file")
                        except Exception as e:
                            print(f"Error writing samples: {e}")
                            traceback.print_exc()
                        
                        # Handle remaining data
                        if new_samples > samples_to_fill:
                            for out_idx, in_idx in channel_mapping.items():
                                if in_idx < len(self.eeg_channels):
                                    buffer[out_idx, :new_samples-samples_to_fill] = data[self.eeg_channels[in_idx], samples_to_fill:]
                            buffer_count = new_samples - samples_to_fill
                        else:
                            buffer_count = 0
                            buffer.fill(0)
                    else:
                        # Map input channels to output channels
                        for out_idx, in_idx in channel_mapping.items():
                            if in_idx < len(self.eeg_channels):
                                buffer[out_idx, buffer_count:buffer_count+new_samples] = data[self.eeg_channels[in_idx], :]
                        buffer_count += new_samples
                    
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Recording error: {str(e)}")
                    traceback.print_exc()
                    messagebox.showerror("Error", f"Recording error: {str(e)}")
                    break
        except Exception as e:
            print(f"Error writing EDF file: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Error writing EDF file: {str(e)}")
            self.recording = False
        
        # Save annotations to BDF
        if self.annotations:
            for ann in self.annotations:
                f.writeAnnotation(
                    onset_in_seconds=ann['onset'],
                    duration_in_seconds=ann['duration'],
                    description=ann['description']
                )

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.recording_start_time = None
            self.record_thread.join()
            self.board.stop_stream()
            self.board.release_session()
            self.board = None
            
            self.status_var.set("Not Recording")
            self.filename_entry.config(state="normal")
            self.port_combo.config(state="readonly")
            self.recording_duration_var.set("0 seconds")  # Updated reference
            messagebox.showinfo("Success", f"Recording saved to {self.output_file}")

    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            else:
                self.settings = {}

            # Initialize configurations if not present
            if "configurations" not in self.settings:
                self.settings["configurations"] = {}

            # Add any custom configurations from the old format
            if "custom_configs" in self.settings:
                self.settings["configurations"].update(self.settings.pop("custom_configs"))

            # Add default configurations to the settings dictionary but don't save them
            self.settings["configurations"] = {
                **{
                    "2 Channel Headset": CHANNEL_CONFIGS["2 Channel"],
                    "19 Channel": CHANNEL_CONFIGS["19 Channel"],
                    "32 Channel": CHANNEL_CONFIGS["32 Channel"]
                },
                **self.settings["configurations"]
            }

            # Use a single consistent setting for last used configuration
            if "last_channel_mode" in self.settings:
                self.settings["headset"] = self.settings.pop("last_channel_mode")
            if "headset" not in self.settings:
                self.settings["headset"] = "19 Channel"

            return self.settings

        except Exception as e:
            print(f"Error loading settings: {e}")
            return {"headset": "19 Channel", "configurations": {}}

    def save_settings(self):
        """Save settings to file"""
        try:
            # Create a copy of settings to modify
            settings_to_save = copy.deepcopy(self.settings)
            
            # Update the last used configuration
            current_headset = self.channel_mode_var.get()
            
            # Create an ordered dictionary with headset and com_port first
            ordered_settings = {
                "headset": current_headset,
                "com_port": settings_to_save.get("com_port", ""),
            }
            
            # Remove default configurations before adding to ordered settings
            if "configurations" in settings_to_save:
                default_configs = ["2 Channel Headset", "19 Channel", "32 Channel"]
                custom_configs = {
                    name: config 
                    for name, config in settings_to_save["configurations"].items() 
                    if name not in default_configs
                }
                ordered_settings["configurations"] = custom_configs

            # Add any remaining settings
            for key, value in settings_to_save.items():
                if key not in ["headset", "com_port", "configurations"]:
                    ordered_settings[key] = value

            with open(self.settings_file, 'w') as f:
                json.dump(ordered_settings, f, indent=4)

        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_custom_configs(self):
        """Load configurations from settings"""
        try:
            # Get all configurations (both default and custom)
            all_configs = self.settings.get("configurations", CHANNEL_CONFIGS.copy())
            
            # Update the channel configs dictionary
            CHANNEL_CONFIGS.clear()
            CHANNEL_CONFIGS.update(all_configs)
            
            # Update the combobox values
            if hasattr(self, 'channel_mode_combo'):
                current_value = self.channel_mode_var.get()
                self.channel_mode_combo['values'] = list(CHANNEL_CONFIGS.keys())
                if current_value in CHANNEL_CONFIGS:
                    self.channel_mode_var.set(current_value)
                else:
                    self.channel_mode_var.set("19 Channel")
        except Exception as e:
            print(f"Error loading configurations: {e}")

    def reload_configurations(self):
        """Reload configurations from settings file and refresh UI"""
        # Reload settings from file
        self.settings = self.load_settings()
        self.custom_configs = self.settings.get("configurations", {}).copy()
        
        # Update CHANNEL_CONFIGS with custom configurations
        # First remove existing custom configs
        keys_to_remove = [k for k in CHANNEL_CONFIGS.keys() if k.startswith("Custom: ")]
        for k in keys_to_remove:
            del CHANNEL_CONFIGS[k]
        
        # Add custom configs back from settings
        for name, config in self.custom_configs.items():
            CHANNEL_CONFIGS[f"Custom: {name}"] = {
                "input_order": INPUT_ORDER_32,
                "output_order": config["output_order"],
                "description": f"Custom Configuration: {name}"
            }
        
        # Refresh the channel mode menu
        self.refresh_channel_mode_menu()
        
        # Update channel info if current selection is still valid
        current_mode = self.channel_mode_var.get()
        if current_mode not in CHANNEL_CONFIGS:
            self.channel_mode_var.set("2 Channel Headset")
        self.update_channel_info()

    def manage_configurations(self):
        """Open dialog to manage custom configurations"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Custom Configurations")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Set dialog position relative to main window
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 500) // 2
        dialog.geometry(f"400x500+{x}+{y}")

        # Add description label
        ttk.Label(dialog, text="Manage your custom channel configurations:", padding=10).pack(fill='x')

        # Create listbox with scrollbar
        self.listbox_frame = ttk.Frame(dialog)
        self.listbox_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        scrollbar = ttk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.config_listbox = tk.Listbox(self.listbox_frame, selectmode='single', yscrollcommand=scrollbar.set)
        self.config_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.config_listbox.yview)

        # Filter out default configurations
        default_configs = ["2 Channel Headset", "19 Channel", "32 Channel"]
        custom_configs = [name for name in self.settings.get("configurations", {}).keys() 
                        if name not in default_configs]
        
        # Populate listbox with custom configurations only
        for config in sorted(custom_configs):
            self.config_listbox.insert('end', config)

        # Create button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=10)

        # Add buttons
        ttk.Button(button_frame, text="Edit", command=self.edit_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Duplicate", command=self.duplicate_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Rename", command=self.start_rename).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", command=lambda: self.on_dialog_close(dialog)).pack(side='left', padx=5)

        # Store dialog reference
        self.config_dialog = dialog
        
        # Add double-click binding to edit
        self.config_listbox.bind('<Double-Button-1>', lambda e: self.edit_config())

    def on_dialog_close(self, dialog):
        """Handle dialog close"""
        dialog.destroy()
        self.reload_configurations()

    def edit_config(self):
        """Edit the selected configuration"""
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to edit")
            return
        
        name = self.config_listbox.get(selection[0])
        config = self.settings["configurations"][name]
        
        # Store the configuration details we need
        edit_config_details = {
            'name': name,
            'channels': config["output_order"],
            'save_flags': config.get("save_flags", [])  # Use get() with default empty list
        }
        
        # Close the current dialog
        self.config_dialog.destroy()
        
        # Open the configuration dialog with the selected config
        self.configure_custom_headset(
            edit_name=edit_config_details['name'],
            edit_channels=edit_config_details['channels'],
            edit_save_flags=edit_config_details['save_flags']
        )

    def duplicate_config(self):
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to duplicate")
            return
        
        original_name = self.config_listbox.get(selection[0])
        new_name = original_name + " (Copy)"
        counter = 1
        
        # Find a unique name
        while new_name in self.custom_configs:
            counter += 1
            new_name = f"{original_name} (Copy {counter})"
        
        # Copy the configuration
        self.custom_configs[new_name] = {
            "output_order": self.custom_configs[original_name]["output_order"][:],
            "save_flags": self.custom_configs[original_name]["save_flags"][:]
        }
        
        # Update CHANNEL_CONFIGS
        CHANNEL_CONFIGS[f"Custom: {new_name}"] = {
            "input_order": INPUT_ORDER_32,
            "output_order": self.custom_configs[new_name]["output_order"],
            "description": f"Custom Configuration: {new_name}"
        }
        
        # Save settings
        self.settings["configurations"] = self.custom_configs
        self.save_settings()
        
        # Update listbox
        self.config_listbox.insert(tk.END, new_name)
        self.config_listbox.selection_clear(0, tk.END)
        self.config_listbox.selection_set(tk.END)
        
        messagebox.showinfo("Success", f"Configuration duplicated as '{new_name}'")

    def delete_config(self):
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to delete")
            return
        
        name = self.config_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", 
                             f"Are you sure you want to delete the configuration '{name}'?"):
            # Remove from custom_configs and CHANNEL_CONFIGS
            del self.custom_configs[name]
            del CHANNEL_CONFIGS[f"Custom: {name}"]
            
            # Update listbox
            self.config_listbox.delete(selection[0])
            
            # Save settings
            self.save_settings()

    def start_rename(self):
        """Start renaming the selected configuration"""
        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a configuration to rename")
            return
        
        # Get the selected item's position and text
        idx = selection[0]
        old_name = self.config_listbox.get(idx)
        
        # Position and show the entry widget
        bbox = self.config_listbox.bbox(idx)
        if bbox:
            x, y, width, height = bbox
            rename_entry = ttk.Entry(self.listbox_frame)
            rename_entry.place(x=x, y=y, width=width, height=height)
            rename_entry.delete(0, tk.END)
            rename_entry.insert(0, old_name)
            rename_entry.select_range(0, tk.END)
            rename_entry.focus_set()
            
            # Store the original name and index for later use
            rename_entry.old_name = old_name
            rename_entry.item_index = idx

            def finish_rename(event=None):
                if not rename_entry.winfo_ismapped():
                    return
                
                new_name = rename_entry.get().strip()
                old_name = rename_entry.old_name
                idx = rename_entry.item_index
                
                # Hide the entry widget
                rename_entry.place_forget()
                
                # If name is empty or unchanged, do nothing
                if not new_name or new_name == old_name:
                    return
                
                # Check if new name already exists
                if new_name in self.settings["configurations"]:
                    messagebox.showerror("Error", "A configuration with this name already exists")
                    return
                
                # Update configurations
                config = self.settings["configurations"].pop(old_name)
                self.settings["configurations"][new_name] = config
                
                # Update listbox
                self.config_listbox.delete(idx)
                self.config_listbox.insert(idx, new_name)
                self.config_listbox.selection_set(idx)
                
                # Save settings
                self.save_settings()

            def cancel_rename(event=None):
                rename_entry.place_forget()

            # Bind entry events
            rename_entry.bind('<Return>', finish_rename)
            rename_entry.bind('<Escape>', cancel_rename)
            rename_entry.bind('<FocusOut>', finish_rename)

    def configure_custom_headset(self, edit_name=None, edit_channels=None, edit_save_flags=None):
        """Open dialog to configure custom headset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Custom Headset")
        dialog.geometry("800x750")
        dialog.transient(self.root)
        dialog.grab_set()

        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add name field
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(name_frame, text="Configuration Name:").pack(side=tk.LEFT)
        name_var = tk.StringVar(value=edit_name if edit_name else "")
        name_entry = ttk.Entry(name_frame, textvariable=name_var)
        name_entry.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)

        # Add description
        ttk.Label(main_frame, text="Enter channel labels and select which channels to save:", wraplength=780).pack(pady=(0,10))

        # Create frame for channel blocks
        blocks_frame = ttk.Frame(main_frame)
        blocks_frame.pack(fill=tk.BOTH, expand=True)

        # Load existing custom configuration
        custom_channels = edit_channels if edit_channels else [""] * 32
        custom_channels.extend([""] * (32 - len(custom_channels)))
        custom_channels = custom_channels[:32]

        # Create lists to store entries and checkboxes
        channel_entries = []
        channel_save_vars = []

        # Create 4 blocks of 8 channels each
        for block in range(4):
            # Create frame for this block
            block_frame = ttk.LabelFrame(blocks_frame, text=f"Channels {block*8 + 1}-{block*8 + 8}")
            block_frame.grid(row=block//2, column=block%2, padx=10, pady=5, sticky="nsew")

            # Create header row with "Save to BDF" label
            header_frame = ttk.Frame(block_frame)
            header_frame.pack(fill=tk.X, pady=(2,5), padx=5)
            ttk.Label(header_frame, text="Channel", width=12).pack(side=tk.LEFT)
            ttk.Label(header_frame, text="Label", width=25).pack(side=tk.LEFT, padx=(5,10))
            ttk.Label(header_frame, text="Save to BDF").pack(side=tk.LEFT)

            # Create channels in this block
            for i in range(8):
                channel_num = block * 8 + i
                row_frame = ttk.Frame(block_frame)
                row_frame.pack(fill=tk.X, pady=2, padx=5)

                # Channel number label
                ttk.Label(row_frame, text=f"Channel {channel_num + 1}:", width=12).pack(side=tk.LEFT)

                # Entry field
                entry = ttk.Entry(row_frame, width=20)
                entry.pack(side=tk.LEFT, padx=(5, 10))
                entry.insert(0, custom_channels[channel_num])
                channel_entries.append(entry)

                # Save checkbox (without text)
                save_var = tk.BooleanVar(value=bool(custom_channels[channel_num].strip()))
                channel_save_vars.append(save_var)
                save_cb = ttk.Checkbutton(row_frame, variable=save_var)
                save_cb.pack(side=tk.LEFT)
                
                # Bind entry changes to update checkbox
                def on_entry_change(event, entry=entry, save_var=save_var):
                    has_content = bool(entry.get().strip())
                    save_var.set(has_content)
                
                entry.bind('<KeyRelease>', on_entry_change)

        # Configure grid weights
        blocks_frame.columnconfigure(0, weight=1)
        blocks_frame.columnconfigure(1, weight=1)

        def save_custom_config():
            # Get configuration name
            config_name = name_var.get().strip()
            if not config_name:
                messagebox.showerror("Error", "Please enter a name for this configuration")
                return
            
            # If we're editing and the name hasn't changed, we can overwrite
            # Otherwise, check if the name exists
            if config_name != edit_name and config_name in self.custom_configs:
                if not messagebox.askyesno(
                    "Confirm Overwrite", 
                    f"A configuration named '{config_name}' already exists. Do you want to overwrite it?"
                ):
                    return

            # Get channel labels and save flags
            channel_labels = [entry.get().strip() for entry in channel_entries]
            save_flags = [var.get() for var in channel_save_vars]
            
            # Check for duplicate channel labels
            used_labels = {}  # label -> channel numbers
            for i, label in enumerate(channel_labels):
                if label and save_flags[i]:  # Only check non-empty labels that are selected to save
                    if label in used_labels:
                        used_labels[label].append(i + 1)
                    else:
                        used_labels[label] = [i + 1]
            
            # Find any duplicates
            duplicates = {label: channels for label, channels in used_labels.items() if len(channels) > 1}
            if duplicates:
                duplicate_msg = []
                for label, channels in duplicates.items():
                    channel_str = ", ".join(str(ch) for ch in channels)
                    duplicate_msg.append(f"Label '{label}' is used in channels: {channel_str}")
                
                messagebox.showerror(
                    "Duplicate Channel Labels",
                    "Each channel must have a unique label.\n\n" + "\n".join(duplicate_msg) + "\n\nPlease provide unique labels for all channels."
                )
                return
            
            # Check for channels that are selected to save but have no label
            unlabeled_channels = []
            for i, (label, save) in enumerate(zip(channel_labels, save_flags)):
                if save and not label:
                    unlabeled_channels.append(i + 1)
            
            if unlabeled_channels:
                if len(unlabeled_channels) == 1:
                    channel_str = f"Channel {unlabeled_channels[0]}"
                else:
                    channel_str = "Channels " + ", ".join(str(ch) for ch in unlabeled_channels)
                
                messagebox.showerror(
                    "Configuration Error",
                    f"{channel_str} {'is' if len(unlabeled_channels) == 1 else 'are'} selected to save but {'has' if len(unlabeled_channels) == 1 else 'have'} no label.\n\n"
                    "Please either:\n"
                    "- Enter labels for these channels, or\n"
                    "- Uncheck them from being saved to BDF"
                )
                return
            
            # Get list of channels to actually save
            channels_to_save = [label for label, save in zip(channel_labels, save_flags) if label and save]
            
            if not channels_to_save:
                messagebox.showerror("Error", "Please enter at least one channel label and select it to save")
                return

            # If we're editing, remove the old configuration if the name changed
            if edit_name and edit_name != config_name:
                del self.custom_configs[edit_name]
                del CHANNEL_CONFIGS[f"Custom: {edit_name}"]

            # Save configuration
            self.custom_configs[config_name] = {
                "output_order": channels_to_save,
                "save_flags": save_flags
            }
            
            # Update CHANNEL_CONFIGS
            CHANNEL_CONFIGS[f"Custom: {config_name}"] = {
                "input_order": INPUT_ORDER_32,
                "output_order": channels_to_save,
                "description": f"Custom Configuration: {config_name}"
            }
            
            # Save settings
            self.settings["configurations"] = self.custom_configs
            self.settings["headset"] = f"Custom: {config_name}"
            self.save_settings()
            
            # Update channel info if this configuration is selected
            self.channel_mode_var.set(f"Custom: {config_name}")
            self.update_channel_info()
            
            # Refresh the channel mode menu
            self.refresh_channel_mode_menu()
            
            dialog.destroy()
            self.reload_configurations()
            messagebox.showinfo("Success", "Custom channel configuration saved")

        def on_dialog_close():
            dialog.destroy()
            self.reload_configurations()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Add buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        def on_manage_click():
            dialog.destroy()
            self.manage_configurations()

        # Add Save and Manage buttons
        ttk.Button(buttons_frame, text="Save Configuration", command=save_custom_config).pack(side=tk.RIGHT)
        ttk.Button(buttons_frame, text="Manage Configurations", command=on_manage_click).pack(side=tk.LEFT)

    def refresh_channel_mode_menu(self):
        # Get the menu widget
        menu = self.channel_mode_combo["menu"]
        
        # Delete all existing menu items
        menu.delete(0, "end")
        
        # Add standard options
        for mode in ["2 Channel Headset", "19 Channel Cap", "32 Channel Cap"]:
            menu.add_command(label=mode, 
                           command=lambda m=mode: self.channel_mode_var.set(m))
        
        # Add separator if we have custom configs
        if self.custom_configs:
            menu.add_separator()
        
        # Add custom configurations
        for config_name in sorted(self.custom_configs.keys()):
            menu_label = f"Custom: {config_name}"
            menu.add_command(label=menu_label,
                           command=lambda m=menu_label: self.channel_mode_var.set(m))

    def quit_app(self):
        """Clean up and close the application"""
        if self.recording:
            self.stop_recording()
        self.save_settings()
        self.root.quit()

    def on_port_change(self):
        """Save selected COM port to settings"""
        selected_port = self.port_var.get()
        if selected_port:
            self.settings["com_port"] = selected_port
            self.save_settings()

    def add_instant_marker(self):
        """Add an instant marker annotation with no duration"""
        if not self.recording:
            messagebox.showerror("Error", "Start recording first")
            return
            
        annotation_name = self.marker_entry.get().strip()
        if not annotation_name or annotation_name == 'Marker name':
            messagebox.showerror("Error", "Valid annotation name required")
            return
        
        # Add marker with 0 duration
        now = time.time() - self.recording_start_time
        self.annotations.append({
            'onset': now,
            'duration': 0.0,
            'description': f"[MARKER] {annotation_name}"
        })
        print(f"\n=== INSTANT MARKER: '{annotation_name}' @ {now:.1f}s ===")

    def toggle_duration_annotation(self):
        if self.duration_var.get():
            if not self.recording or self.recording_start_time is None:
                self.duration_var.set(False)
                messagebox.showerror("Error", "Recording not properly initialized")
                return
            
            annotation_name = self.duration_entry.get().strip()
            if not annotation_name or annotation_name == 'Duration name':
                messagebox.showerror("Error", "Valid annotation name required")
                self.duration_var.set(False)
                return
            
            # Record start time and print status
            start_time = time.time() - self.recording_start_time
            self.annotations.append({
                'onset': start_time,
                'duration': 0.0,
                'description': f"[DURATION] {annotation_name}"  # Add DURATION tag
            })
            self.current_duration_index = len(self.annotations) - 1  # Track the index
            print(f"\n=== DURATION START: {annotation_name} @ {start_time:.1f}s ===")
        else:
            # End duration annotation
            if self.current_duration_index is not None and self.current_duration_index < len(self.annotations):
                # Calculate final duration and print status
                now = time.time() - self.recording_start_time
                duration_annotation = self.annotations[self.current_duration_index]
                duration_annotation['duration'] = now - duration_annotation['onset']
                print(f"\n=== DURATION END: {duration_annotation['description']} "
                      f"({duration_annotation['duration']:.1f}s) @ {now:.1f}s ===")
                self.current_duration_index = None  # Clear the tracking
            else:
                print("\nWarning: No active duration annotation to end")
                self.duration_var.set(False)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ACErecorder(root)
        root.mainloop()
    except Exception as e:
        import traceback
        error_message = f"Error starting application:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        # Create a basic error window
        error_window = tk.Tk()
        error_window.title("Error")
        error_window.geometry("600x400")
        
        # Add error text
        text_widget = tk.Text(error_window, wrap=tk.WORD)
        text_widget.insert(tk.END, error_message)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Add close button
        close_button = tk.Button(error_window, text="Close", command=error_window.destroy)
        close_button.pack(pady=10)
        
        # Keep window open
        error_window.mainloop()