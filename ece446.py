from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowShelfFilter, HighShelfFilter, Phaser
from pythonosc import dispatcher, osc_server
from datetime import datetime
import threading
import csv
import os
import time
import ast
import sounddevice as sd
import numpy as np

class MuseStreamManager:
    # --- Configuration ---
    EEG_BANDS = [
        "/muse/elements/alpha_relative", "/muse/elements/beta_relative",
        "/muse/elements/delta_absolute", "/muse/elements/theta_absolute",
        "/muse/elements/gamma_absolute", "/muse/elements/alpha_absolute",
        "/muse/elements/beta_absolute", "/muse/elements/delta_relative",
        "/muse/elements/theta_relative", "/muse/elements/gamma_relative"
    ]
    FIXED_PLAYBACK_DELAY = 0.5 
    ALPHA_ADDRESS = "/muse/elements/alpha_relative"
    BETA_ADDRESS = "/muse/elements/beta_relative"
    # ---------------------

    def __init__(self, ip="0.0.0.0", port=3001, csv_path="muse_stream.csv", timeout=3.0):
        self.ip = ip 
        self.port = port
        self.csv_path = csv_path
        self.timeout = timeout
        self.data_received_event = threading.Event()
        self.mode = "detecting"
        self.latest_alpha = None
        self.latest_beta = None
        self.audio_thread = None
        self.stop_audio = threading.Event()
        self.audio_file = None
        self.board = None
        self.samplerate = None
        self.num_channels = None
        
        # Audio effect parameters that will change based on focus
        self.current_focus_index = 0.5  # Default middle value
        self.effects_lock = threading.Lock()  # Thread safety for real-time parameter updates
        
        # Initialize effects with default parameters
        self.setup_audio_effects()

        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "address", "values"])

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("*", self.live_handler)

    def setup_audio_effects(self):
        """Initialize audio effects with default parameters"""
        # These will be updated in real-time based on focus index
        self.board = Pedalboard([
            Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=100),
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=0),
            HighShelfFilter(cutoff_frequency_hz=5000, gain_db=0),
            Chorus(rate_hz=0.5, depth=0.1, centre_delay_ms=7, feedback=0.3, mix=0.3),
            Phaser(rate_hz=0.3, depth=0.5, centre_frequency_hz=800, feedback=0.2, mix=0.1),
            Reverb(room_size=0.3, damping=0.5, wet_level=0.1, dry_level=0.9, width=0.8)
        ])

    def clamp(self, value, min_val, max_val):
        """Ensure value stays within specified range"""
        return max(min_val, min(value, max_val))

    def update_audio_effects_based_on_focus(self, focus_index):
        """Update audio effects parameters based on the current focus index"""
        # Normalize focus index to 0-1 range (assuming typical FI range is 0.5-3.0)
        normalized_focus = self.clamp((focus_index - 0.5) / 2.5, 0.0, 1.0)
        
        with self.effects_lock:
            # HIGH FOCUS: Clear, punchy, concentrated sound
            if normalized_focus > 0.6:
                # Less reverb, more compression, bass boost
                self.board[0].threshold_db = -15  # More compression
                self.board[0].ratio = 6
                self.board[1].gain_db = 4.0 + (normalized_focus * 4)  # Bass boost
                self.board[2].gain_db = 2.0  # Slight treble boost
                self.board[3].mix = self.clamp(0.1, 0.0, 1.0)
                self.board[4].mix = self.clamp(0.05, 0.0, 1.0)
                self.board[5].wet_level = self.clamp(0.05, 0.0, 1.0)
                self.board[5].room_size = self.clamp(0.2, 0.0, 1.0)
                
            # MEDIUM FOCUS: Balanced sound
            elif normalized_focus > 0.3:
                self.board[0].threshold_db = -18
                self.board[0].ratio = 4
                self.board[1].gain_db = 2.0
                self.board[2].gain_db = 1.0
                self.board[3].mix = self.clamp(0.2, 0.0, 1.0)
                self.board[4].mix = self.clamp(0.1, 0.0, 1.0)
                self.board[5].wet_level = self.clamp(0.15, 0.0, 1.0)
                self.board[5].room_size = self.clamp(0.4, 0.0, 1.0)
                
            # LOW FOCUS: Dreamy, spacious, airy sound
            else:
                # More reverb, chorus, and phaser for spaciousness
                self.board[0].threshold_db = -25  # Less compression
                self.board[0].ratio = 2
                self.board[1].gain_db = 0.0  # Less bass
                self.board[2].gain_db = -1.0  # Softer treble
                
                # Calculate and clamp mix values
                chorus_mix = 0.4 + ((0.3 - normalized_focus) * 1.5)
                phaser_mix = 0.3 + ((0.3 - normalized_focus) * 1.5)
                reverb_wet = 0.3 + ((0.3 - normalized_focus) * 1.5)
                reverb_room = 0.7 + ((0.3 - normalized_focus) * 0.3)
                
                self.board[3].mix = self.clamp(chorus_mix, 0.0, 1.0)
                self.board[3].rate_hz = 0.3  # Slower chorus
                self.board[4].mix = self.clamp(phaser_mix, 0.0, 1.0)
                self.board[4].rate_hz = 0.2  # Slower phaser
                self.board[5].wet_level = self.clamp(reverb_wet, 0.0, 1.0)
                self.board[5].room_size = self.clamp(reverb_room, 0.0, 1.0)

    def calculate_average_focus_index(self):
        """Calculate average focus index from all electrodes"""
        if self.latest_alpha is None or self.latest_beta is None:
            return 1.0  # Default neutral value
            
        if len(self.latest_alpha) != len(self.latest_beta):
            return 1.0
            
        valid_focus_values = []
        for alpha_val, beta_val in zip(self.latest_alpha, self.latest_beta):
            try:
                if (isinstance(alpha_val, (int, float)) and 
                    isinstance(beta_val, (int, float)) and 
                    alpha_val > 0):
                    fi = beta_val / alpha_val
                    if 0 < fi < 10:  # Reasonable range check
                        valid_focus_values.append(fi)
            except (TypeError, ZeroDivisionError):
                continue
                
        if valid_focus_values:
            avg_focus = sum(valid_focus_values) / len(valid_focus_values)
            self.current_focus_index = avg_focus
            return avg_focus
        else:
            return 1.0

    def process_audio_in_realtime(self):
        """Process and play audio in real-time with dynamic effects"""
        try:
            # Open an audio file for reading
            self.audio_file = AudioFile('music.wav')
            self.samplerate = self.audio_file.samplerate
            self.num_channels = self.audio_file.num_channels
                
            print(f"🎵 Starting real-time audio processing: {self.samplerate}Hz, {self.num_channels} channels")
            
            # Define callback function for real-time playback
            def audio_callback(outdata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                
                if self.stop_audio.is_set():
                    raise sd.CallbackStop()
                
                # Update effects based on current focus before processing
                current_focus = self.calculate_average_focus_index()
                self.update_audio_effects_based_on_focus(current_focus)
                
                # Read audio chunk
                chunk = self.audio_file.read(frames)
                
                if chunk.size == 0:
                    # End of file - stop playback
                    raise sd.CallbackStop()
                
                # Apply effects with current parameters
                with self.effects_lock:
                    effected = self.board(chunk, self.samplerate, reset=False)
                
                # Fix array shape
                if effected.ndim == 2 and effected.shape[0] == self.num_channels:
                    effected = effected.T
                
                # Ensure output matches expected shape
                if effected.shape[0] < frames:
                    silence_length = frames - effected.shape[0]
                    silence = np.zeros((silence_length, self.num_channels))
                    outdata[:effected.shape[0]] = effected
                    outdata[effected.shape[0]:] = silence
                else:
                    outdata[:] = effected[:frames]

            # Start real-time playback
            with sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.num_channels,
                callback=audio_callback,
                blocksize=1024
            ):
                print("🔊 Audio playback started - Press Ctrl+C to stop")
                while not self.stop_audio.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
        finally:
            if self.audio_file:
                self.audio_file.close()
            print("🔇 Audio playback finished")

    def start_audio_processing(self):
        """Start audio processing in a separate thread"""
        self.audio_thread = threading.Thread(target=self.process_audio_in_realtime, daemon=True)
        self.audio_thread.start()

    def _calculate_focus_index(self):
        """Calculates Focus Index (Beta / Alpha) across all four electrodes."""
        if self.latest_alpha is None or self.latest_beta is None:
            return "N/A"
            
        if len(self.latest_alpha) != len(self.latest_beta):
            return "Error: Length mismatch"

        focus_indices = []
        for alpha_val, beta_val in zip(self.latest_alpha, self.latest_beta):
            try:
                if isinstance(alpha_val, (int, float)) and alpha_val > 0:
                    fi = beta_val / alpha_val
                    focus_indices.append(f"{fi:.3f}")
                elif alpha_val == 0:
                    focus_indices.append("Inf") 
                else:
                    focus_indices.append("Err")
            except TypeError:
                    focus_indices.append("Err")

        return ", ".join(focus_indices)

    def start(self):
        """Starts the detection and streaming process."""
        print(f"⏳ Attempting to connect to live stream on {self.ip}:{self.port}...")
        print("🎵 Dynamic audio effects: Low focus = dreamy/reverb | High focus = punchy/bass boosted")
        
        # Start audio processing
        self.start_audio_processing()
        
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()

        stream_found = self.data_received_event.wait(timeout=self.timeout)

        if stream_found:
            self.mode = "live"
            print("✅ Live stream detected. Displaying Focus Index...")
            print("🎵 Real-time audio processing with dynamic effects is running...")
            try:
                while True:
                    time.sleep(1) 
            except KeyboardInterrupt:
                print("\n🛑 Stopping live stream and audio.")
                self.stop_audio.set()
        else:
            self.mode = "playback"
            print(f"⚠️ No live stream detected after {self.timeout}s.")
            print(f"🔄 Switching to CSV playback from: {self.csv_path}")
            print("🎵 Real-time audio processing with dynamic effects is running...")
            try:
                self.play_from_csv()
            except KeyboardInterrupt:
                print("\n🛑 Stopping playback and audio.")
            finally:
                self.stop_audio.set()
                print("\n🏁 Program finished.")

    def _run_server(self):
        try:
            server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
            server.serve_forever()
        except OSError as e:
            print(f"❌ Could not bind to port {self.port} on IP {self.ip}: {e}")

    def live_handler(self, address: str, *args):
        """Handles live incoming OSC messages, filters, logs, and calculates FI."""
        
        if not self.data_received_event.is_set():
            self.data_received_event.set()

        is_relevant_fi_data = False
        if address == self.ALPHA_ADDRESS and args:
            self.latest_alpha = list(args)
            is_relevant_fi_data = True
        elif address == self.BETA_ADDRESS and args:
            self.latest_beta = list(args)
            is_relevant_fi_data = True

        if address in self.EEG_BANDS:
            timestamp = datetime.now().isoformat()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, address, list(args)])

        if is_relevant_fi_data:
            fi_value = self._calculate_focus_index()
            avg_focus = self.calculate_average_focus_index()
            focus_category = "HIGH" if avg_focus > 1.5 else "MEDIUM" if avg_focus > 1.0 else "LOW"
            print(f"LIVE FI [{datetime.now().isoformat()}] | Focus: {focus_category} ({avg_focus:.2f}) | Electrodes: [{fi_value}]")

    def play_from_csv(self):
        """Reads and replays data from the CSV file with a fixed 0.5s delay, printing only FI."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) < 100: 
            print("❌ CSV file is empty or missing data. Cannot replay.")
            return

        print("--- Starting CSV Playback Simulation (0.5s fixed delay) ---")
        
        count = 0
        self.latest_alpha = None
        self.latest_beta = None

        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)

                for row in reader:
                    if len(row) < 3: continue
                    
                    current_timestamp_str = row[0]
                    address = row[1]
                    values_str = row[2]

                    is_relevant_fi_data = False
                    if address == self.ALPHA_ADDRESS:
                        self.latest_alpha = ast.literal_eval(values_str)
                        is_relevant_fi_data = True
                    elif address == self.BETA_ADDRESS:
                        self.latest_beta = ast.literal_eval(values_str)
                        is_relevant_fi_data = True
                        
                    if is_relevant_fi_data:
                        fi_value = self._calculate_focus_index()
                        avg_focus = self.calculate_average_focus_index()
                        focus_category = "HIGH" if avg_focus > 1.5 else "MEDIUM" if avg_focus > 1.0 else "LOW"
                        print(f"PLAYBACK FI [{current_timestamp_str}] | Focus: {focus_category} ({avg_focus:.2f}) | Electrodes: [{fi_value}]")
                        count += 1
                        time.sleep(self.FIXED_PLAYBACK_DELAY)
                        
            print(f"--- Playback Finished. Processed {count} relevant Focus Index updates. ---")

        except Exception as e:
            print(f"❌ Critical Error during CSV playback: {e}")
        finally:
            self.stop_audio.set()

if __name__ == "__main__":
    # Create the output file for reference (optional)
    try:
        board = Pedalboard([Chorus(), Reverb(room_size=0.25)])
        with AudioFile('music.wav') as f:
            with AudioFile('output.wav', 'w', f.samplerate, f.num_channels) as o:
                while f.tell() < f.frames:
                    chunk = f.read(f.samplerate)
                    effected = board(chunk, f.samplerate, reset=False)
                    o.write(effected)
        print("✅ Created output.wav for reference")
    except Exception as e:
        print(f"⚠️ Could not create output.wav: {e}")
        print("🎵 Continuing with real-time processing only...")
    
    manager = MuseStreamManager(ip="0.0.0.0", port=3001, timeout=3.0) 
    manager.start()