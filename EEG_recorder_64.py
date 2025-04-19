"""
EEG_recorder_64.py

Handles 64-channel recording using two FreeEEG32 boards, synchronizing data streams using package number channels,
and merging the data for saving as a single BDF file.
"""
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
import threading
import time
import os
import pyedflib

class EEGRecorder64:
    def __init__(self, port1, port2, output_file, sample_rate=250):
        self.port1 = port1
        self.port2 = port2
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.board1 = None
        self.board2 = None
        self.running = False
        self.thread = None
        self._stop_event = threading.Event()

    def start(self):
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop(self):
        print("[EEGRecorder64] stop() called. Attempting to stop recording thread...")
        self.running = False
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                print("[EEGRecorder64] WARNING: Recording thread did not exit after join().")
            else:
                print("[EEGRecorder64] Recording thread exited cleanly.")
        if self.board1:
            try:
                self.board1.stop_stream()
            except Exception as e:
                print(f"[EEGRecorder64] Exception stopping board1 stream: {e}")
            try:
                self.board1.release_session()
            except Exception as e:
                print(f"[EEGRecorder64] Exception releasing board1 session: {e}")
        if self.board2:
            try:
                self.board2.stop_stream()
            except Exception as e:
                print(f"[EEGRecorder64] Exception stopping board2 stream: {e}")
            try:
                self.board2.release_session()
            except Exception as e:
                print(f"[EEGRecorder64] Exception releasing board2 session: {e}")

    def _record(self):
        print("[EEGRecorder64] _record() thread started.")
        # Initialize both boards
        params1 = BrainFlowInputParams()
        params1.serial_port = self.port1
        params2 = BrainFlowInputParams()
        params2.serial_port = self.port2
        board_id = BoardIds.FREEEEG32_BOARD.value
        self.board1 = BoardShim(board_id, params1)
        self.board2 = BoardShim(board_id, params2)
        self.board1.prepare_session()
        self.board2.prepare_session()
        self.board1.start_stream()
        self.board2.start_stream()
        print("Both boards streaming. Synchronizing...")
        time.sleep(2)  # Let buffers fill

        pkg_idx1 = BoardShim.get_package_num_channel(board_id, 0)
        pkg_idx2 = BoardShim.get_package_num_channel(board_id, 0)

        # Logging variables for dropped/matched samples
        matched_samples = 0
        dropped_1_only = 0
        dropped_2_only = 0
        total_written_samples = 0

        # Main acquisition loop (simple version)
        buffer1 = []
        file_initialized = False
        edf_writer = None
        try:
            def mod_diff(a, b):
                return (a - b) % 256

            BUFFER_SIZE = 20
            buf1 = []
            buf2 = []
            last_eeg1 = np.zeros(32)
            last_eeg2 = np.zeros(32)

            while self.running and not self._stop_event.is_set():
                data1 = self.board1.get_board_data()
                data2 = self.board2.get_board_data()
                if data1.shape[1] > 0:
                    for i in range(data1.shape[1]):
                        eeg = data1[1:33, i]
                        pkg = int(data1[pkg_idx1, i])
                        buf1.append((pkg, eeg))
                        last_eeg1 = eeg
                        if len(buf1) > BUFFER_SIZE:
                            buf1.pop(0)
                if data2.shape[1] > 0:
                    for i in range(data2.shape[1]):
                        eeg = data2[1:33, i]
                        pkg = int(data2[pkg_idx2, i])
                        buf2.append((pkg, eeg))
                        last_eeg2 = eeg
                        if len(buf2) > BUFFER_SIZE:
                            buf2.pop(0)
                if len(buf1) == 0 or len(buf2) == 0:
                    time.sleep(0.01)
                    continue
                # Synchronize buffers
                while buf1 and buf2:
                    p1, eeg1 = buf1[0]
                    p2, eeg2 = buf2[0]
                    if p1 == p2:
                        pkg1_centered = ((p1 + 128) % 256) - 128
                        pkg2_centered = ((p2 + 128) % 256) - 128
                        merged = np.concatenate([eeg1, eeg2, [pkg1_centered], [pkg2_centered]]).astype(np.int32)
                        buffer1.append(merged)
                        matched_samples += 1
                        buf1.pop(0)
                        buf2.pop(0)
                    elif mod_diff(p1, p2) < 128:
                        # p1 ahead of p2: look for p1 in buf2
                        idx2 = next((i for i, (pkg, _) in enumerate(buf2) if pkg == p1), -1)
                        if 0 <= idx2 < BUFFER_SIZE:
                            # Pad missing samples in buf2 with last_eeg2
                            for _ in range(idx2):
                                pkg2_miss, _ = buf2.pop(0)
                                pkg1_centered = ((p1 + 128) % 256) - 128
                                pkg2_centered = ((pkg2_miss + 128) % 256) - 128
                                merged = np.concatenate([eeg1, last_eeg2, [pkg1_centered], [pkg2_centered]]).astype(np.int32)
                                buffer1.append(merged)
                                dropped_2_only += 1
                                print(f"[EEGRecorder64] WARNING: True missing sample from board2 at pkg {pkg2_miss}, padding with last value.")
                            continue  # Re-evaluate after popping
                        else:
                            if len(buf2) > BUFFER_SIZE:
                                pkg2_miss, _ = buf2.pop(0)
                                pkg1_centered = ((p1 + 128) % 256) - 128
                                pkg2_centered = ((pkg2_miss + 128) % 256) - 128
                                merged = np.concatenate([eeg1, last_eeg2, [pkg1_centered], [pkg2_centered]]).astype(np.int32)
                                buffer1.append(merged)
                                dropped_2_only += 1
                                print(f"[EEGRecorder64] WARNING: True missing sample from board2 at pkg {pkg2_miss}, padding with last value.")
                            else:
                                break  # Wait for more data
                    else:
                        # p2 ahead of p1: look for p2 in buf1
                        idx1 = next((i for i, (pkg, _) in enumerate(buf1) if pkg == p2), -1)
                        if 0 <= idx1 < BUFFER_SIZE:
                            for _ in range(idx1):
                                pkg1_miss, _ = buf1.pop(0)
                                pkg1_centered = ((pkg1_miss + 128) % 256) - 128
                                pkg2_centered = ((p2 + 128) % 256) - 128
                                merged = np.concatenate([last_eeg1, eeg2, [pkg1_centered], [pkg2_centered]]).astype(np.int32)
                                buffer1.append(merged)
                                dropped_1_only += 1
                                print(f"[EEGRecorder64] WARNING: True missing sample from board1 at pkg {pkg1_miss}, padding with last value.")
                            continue
                        else:
                            if len(buf1) > BUFFER_SIZE:
                                pkg1_miss, _ = buf1.pop(0)
                                pkg1_centered = ((pkg1_miss + 128) % 256) - 128
                                pkg2_centered = ((p2 + 128) % 256) - 128
                                merged = np.concatenate([last_eeg1, eeg2, [pkg1_centered], [pkg2_centered]]).astype(np.int32)
                                buffer1.append(merged)
                                dropped_1_only += 1
                                print(f"[EEGRecorder64] WARNING: True missing sample from board1 at pkg {pkg1_miss}, padding with last value.")
                            else:
                                break
                    # Handle buffer overflow: write full blocks, keep remainder
                    while len(buffer1) >= self.sample_rate:
                        if not file_initialized:
                            edf_writer = pyedflib.EdfWriter(self.output_file, n_channels=66, file_type=pyedflib.FILETYPE_BDFPLUS)
                            channel_info = []
                            for i in range(64):
                                channel_info.append({
                                    'label': f"Ch{i+1}",
                                    'dimension': 'uV',
                                    'sample_rate': self.sample_rate,
                                    'physical_min': -8388608,
                                    'physical_max': 8388607,
                                    'digital_min': -8388608,
                                    'digital_max': 8388607,
                                    'transducer': '',
                                    'prefilter': ''
                                })
                            for i in range(2):
                                channel_info.append({
                                    'label': f"PkgNum{i+1}",
                                    'dimension': 'count',
                                    'sample_rate': self.sample_rate,
                                    'physical_min': -128,
                                    'physical_max': 127,
                                    'digital_min': -8388608,
                                    'digital_max': 8388607,
                                    'transducer': '',
                                    'prefilter': ''
                                })
                            edf_writer.setSignalHeaders(channel_info)
                            file_initialized = True
                        # Write a full block
                        block = np.array(buffer1[:self.sample_rate]).T
                        edf_writer.writeSamples(block)
                        total_written_samples += self.sample_rate
                        print(f"Appended {self.sample_rate} samples to {self.output_file}")
                        buffer1 = buffer1[self.sample_rate:]  # Keep remainder
                        print(f"Buffer overflow handled, {len(buffer1)} samples remain in buffer after write")
                if self._stop_event.is_set():
                    break
            # Save any remaining samples (even if not a full block)
            if len(buffer1) > 0 and edf_writer is not None:
                data = np.array(buffer1).T
                edf_writer.writeSamples(data)
                total_written_samples += len(buffer1)
                print(f"Final append of {len(buffer1)} samples to {self.output_file}")
        finally:
            print("\n=== Dual-board Synchronization Report ===")
            print(f"Matched samples written: {matched_samples}")
            print(f"Dropped (present only on board 1): {dropped_1_only}")
            print(f"Dropped (present only on board 2): {dropped_2_only}")
            print(f"Total samples written to file: {total_written_samples}")
            if edf_writer is not None:
                edf_writer.close()
                print(f"Closed BDF file {self.output_file}")
            print("[EEGRecorder64] _record() thread exiting.")
