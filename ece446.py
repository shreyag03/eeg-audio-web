from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowShelfFilter, HighShelfFilter, Phaser, Delay, Gain
from pythonosc import dispatcher, osc_server
from datetime import datetime
import threading
import csv
import os
import time
import ast
import sounddevice as sd
import numpy as np
import sys
from collections import deque
from PyQt6.QtWidgets import (QApplication, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout, QComboBox,
                             QLabel, QPushButton, QDoubleSpinBox, QFileDialog, QGridLayout)
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
import pyqtgraph as pg

# Global data store for thread-safe communication
class DataStore(QObject):
    # Emit: mental_state, attention, relaxation, creativity, drowsiness,
    #        delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm
    data_updated = pyqtSignal(str, float, float, float, float, float, float, float, float, float)
    
    def __init__(self):
        super().__init__()
        self.latest_data = None

class BrainwavePlotter:
    def __init__(self, data_store):
        self.data_store = data_store
        self.data_store.data_updated.connect(self.on_data_updated)
        
        # Data storage for plotting
        self.timestamps = deque(maxlen=1000)
        self.brainwave_data = {
            'delta': deque(maxlen=1000),
            'theta': deque(maxlen=1000), 
            'alpha': deque(maxlen=1000),
            'beta': deque(maxlen=1000),
            'gamma': deque(maxlen=1000)
        }
        self.mental_states = deque(maxlen=1000)
        self.scores = {
            'attention': deque(maxlen=1000),
            'relaxation': deque(maxlen=1000),
            'creativity': deque(maxlen=1000),
            'drowsiness': deque(maxlen=1000)
        }
        
        # PyQt setup
        self.app = QApplication(sys.argv)
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        
        # Control panel
        self.ctrl_widget = QWidget()
        self.ctrl_layout = QGridLayout(self.ctrl_widget)
        self.layout.addWidget(self.ctrl_widget)
        
        # Time range control
        self.time_range_label = QLabel("Time Range (s):")
        self.ctrl_layout.addWidget(self.time_range_label, 0, 0)
        self.time_range_spin = QDoubleSpinBox()
        self.time_range_spin.setRange(5, 300)
        self.time_range_spin.setValue(30)
        self.time_range_spin.setSingleStep(5)
        self.ctrl_layout.addWidget(self.time_range_spin, 0, 1)
        
        # Status display
        self.status_label = QLabel("Status: Waiting for data...")
        self.ctrl_layout.addWidget(self.status_label, 0, 2, 1, 2)
        
        # Current state display
        self.state_label = QLabel("Current State: ---")
        self.state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.ctrl_layout.addWidget(self.state_label, 1, 0, 1, 4)
        
        # Save button
        self.save_button = QPushButton("Save Data")
        self.ctrl_layout.addWidget(self.save_button, 2, 0)
        self.save_button.clicked.connect(self.save_data)
        
        # Scroll area for plots
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll)
        
        # Create plots
        self.create_plots()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)  # Update every 100ms
        
        self.main_widget.show()
        self.main_widget.resize(1200, 800)
        self.main_widget.setWindowTitle("Brainwave & Mental State Monitor")

    def on_data_updated(self, mental_state, attention, relaxation, creativity, drowsiness, delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm):
        """Handle new data from the Muse manager"""
        current_time = time.time()
        
        # Store data
        self.timestamps.append(current_time)
        self.mental_states.append(mental_state)
        
        self.brainwave_data['delta'].append(delta_norm)
        self.brainwave_data['theta'].append(theta_norm)
        self.brainwave_data['alpha'].append(alpha_norm)
        self.brainwave_data['beta'].append(beta_norm)
        self.brainwave_data['gamma'].append(gamma_norm)
        
        self.scores['attention'].append(attention)
        self.scores['relaxation'].append(relaxation)
        self.scores['creativity'].append(creativity)
        self.scores['drowsiness'].append(drowsiness)
        
        # Debug: Print received data
        # print(f"Plotter received: {mental_state} at {current_time}")

    def create_plots(self):
        """Create all the plots for brainwaves and mental states"""
        
        # Brainwave amplitudes plot
        self.brainwave_plot = pg.PlotWidget(title="Brainwave Amplitudes (Normalized)")
        self.brainwave_plot.setLabel('left', 'Normalized Amplitude')
        self.brainwave_plot.setLabel('bottom', 'Time (seconds)')
        self.brainwave_plot.addLegend()
        self.brainwave_plot.showGrid(x=True, y=True)
        self.brainwave_plot.setYRange(0, 1)
        
        self.brainwave_curves = {
            'delta': self.brainwave_plot.plot(pen=pg.mkPen('red', width=2), name='Delta'),
            'theta': self.brainwave_plot.plot(pen=pg.mkPen('blue', width=2), name='Theta'),
            'alpha': self.brainwave_plot.plot(pen=pg.mkPen('green', width=2), name='Alpha'),
            'beta': self.brainwave_plot.plot(pen=pg.mkPen('orange', width=2), name='Beta'),
            'gamma': self.brainwave_plot.plot(pen=pg.mkPen('purple', width=2), name='Gamma')
        }
        
        self.scroll_layout.addWidget(self.brainwave_plot)
        
        # Mental state scores plot
        self.scores_plot = pg.PlotWidget(title="Mental State Scores")
        self.scores_plot.setLabel('left', 'Score')
        self.scores_plot.setLabel('bottom', 'Time (seconds)')
        self.scores_plot.addLegend()
        self.scores_plot.showGrid(x=True, y=True)
        self.scores_plot.setYRange(0, 1)
        
        self.score_curves = {
            'attention': self.scores_plot.plot(pen=pg.mkPen('red', width=2), name='Attention'),
            'relaxation': self.scores_plot.plot(pen=pg.mkPen('blue', width=2), name='Relaxation'),
            'creativity': self.scores_plot.plot(pen=pg.mkPen('green', width=2), name='Creativity'),
            'drowsiness': self.scores_plot.plot(pen=pg.mkPen('yellow', width=2), name='Drowsiness')
        }
        
        self.scroll_layout.addWidget(self.scores_plot)
        
        # Mental state history plot (categorical)
        self.state_plot = pg.PlotWidget(title="Mental State History")
        self.state_plot.setLabel('left', 'State')
        self.state_plot.setLabel('bottom', 'Time (seconds)')
        self.state_plot.showGrid(x=True, y=True)
        
        # State mapping for plotting
        self.state_mapping = {
            'neutral': 0,
            'focused': 1,
            'relaxed': 2,
            'creative': 3,
            'drowsy': 4
        }
        
        self.state_curve = self.state_plot.plot(pen=pg.mkPen('black', width=3), name='Mental State')
        self.state_plot.setYRange(-0.5, 4.5)
        
        # Add state labels to y-axis
        y_ticks = [(v, k.upper()) for k, v in self.state_mapping.items()]
        self.state_plot.getPlotItem().getAxis('left').setTicks([y_ticks])
        
        # Add colored regions for each state
        self.state_regions = []
        colors = ['gray', 'red', 'blue', 'green', 'orange']
        for i, (state, value) in enumerate(self.state_mapping.items()):
            region = pg.LinearRegionItem(values=[value-0.45, value+0.45], orientation='horizontal', 
                                       brush=pg.mkBrush(color=colors[i], alpha=50), movable=False)
            self.state_plot.addItem(region)
            self.state_regions.append(region)
        
        self.scroll_layout.addWidget(self.state_plot)

    def update_plots(self):
        """Update all plots with current data"""
        if not self.timestamps:
            return
            
        # Convert timestamps to relative seconds for x-axis
        current_time = time.time()
        time_range = self.time_range_spin.value()
        
        # Ensure we have data to plot
        if len(self.timestamps) == 0:
            return
            
        # Calculate time window
        start_time = current_time - time_range
        x_data = []
        valid_indices = []
        
        # Filter data within time window
        for i, t in enumerate(self.timestamps):
            if t >= start_time:
                x_data.append(t - start_time)
                valid_indices.append(i)
        
        if not x_data:
            return
            
        # Update brainwave plot
        for wave, curve in self.brainwave_curves.items():
            if self.brainwave_data[wave] and len(self.brainwave_data[wave]) > valid_indices[0]:
                # Get only the data within our time window
                wave_data = [self.brainwave_data[wave][i] for i in valid_indices if i < len(self.brainwave_data[wave])]
                if len(wave_data) == len(x_data):
                    curve.setData(x_data, wave_data)
        
        # Update scores plot
        for score_type, curve in self.score_curves.items():
            if self.scores.get(score_type) and len(self.scores[score_type]) > valid_indices[0]:
                # Get only the data within our time window
                score_data = [self.scores[score_type][i] for i in valid_indices if i < len(self.scores[score_type])]
                if len(score_data) == len(x_data):
                    curve.setData(x_data, score_data)
        
        # Update state plot
        if self.mental_states and len(self.mental_states) > valid_indices[0]:
            # Get only the states within our time window
            state_data = [self.mental_states[i] for i in valid_indices if i < len(self.mental_states)]
            state_numeric = [self.state_mapping.get(state, 0) for state in state_data]
            
            if len(state_numeric) == len(x_data):
                # Use scatter plot for better visibility of state changes
                scatter = pg.ScatterPlotItem(x=x_data, y=state_numeric, 
                                           pen=pg.mkPen('black', width=2),
                                           brush=pg.mkBrush('black'),
                                           size=10)
                
                # Clear old plot and add new scatter
                self.state_plot.clear()
                
                # Re-add regions
                for region in self.state_regions:
                    self.state_plot.addItem(region)
                
                # Add scatter plot
                self.state_plot.addItem(scatter)
                
                # Update y-axis labels
                y_ticks = [(v, k.upper()) for k, v in self.state_mapping.items()]
                self.state_plot.getPlotItem().getAxis('left').setTicks([y_ticks])
                self.state_plot.setYRange(-0.5, 4.5)
        
        # Update status and current state
        if self.mental_states:
            current_state = self.mental_states[-1]
            state_colors = {
                'neutral': 'gray',
                'focused': 'red', 
                'relaxed': 'blue',
                'creative': 'green',
                'drowsy': 'orange'
            }
            color = state_colors.get(current_state, 'black')
            self.state_label.setText(f"Current State: <span style='color: {color}; font-weight: bold;'>{current_state.upper()}</span>")
            
            # Update status with latest scores
            if self.scores['attention']:
                status_text = (f"Status: Attention: {self.scores['attention'][-1]:.2f} | "
                             f"Relaxation: {self.scores['relaxation'][-1]:.2f} | "
                             f"Creativity: {self.scores['creativity'][-1]:.2f} | "
                             f"Drowsiness: {self.scores['drowsiness'][-1]:.2f}")
                self.status_label.setText(status_text)

    def save_data(self):
        """Save current data to CSV file"""
        if not self.timestamps:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            None, "Save Brainwave Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if save_path:
            with open(save_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header (include drowsiness)
                writer.writerow(['timestamp', 'mental_state', 'attention', 'relaxation', 'creativity', 'drowsiness',
                               'delta', 'theta', 'alpha', 'beta', 'gamma'])
                
                # Convert deques to lists for indexing
                timestamps_list = list(self.timestamps)
                mental_states_list = list(self.mental_states)
                scores_attention = list(self.scores['attention'])
                scores_relaxation = list(self.scores['relaxation'])
                scores_creativity = list(self.scores['creativity'])
                scores_drowsiness = list(self.scores['drowsiness'])
                brainwave_delta = list(self.brainwave_data['delta'])
                brainwave_theta = list(self.brainwave_data['theta'])
                brainwave_alpha = list(self.brainwave_data['alpha'])
                brainwave_beta = list(self.brainwave_data['beta'])
                brainwave_gamma = list(self.brainwave_data['gamma'])
                
                # Write data
                for i in range(len(timestamps_list)):
                    writer.writerow([
                        timestamps_list[i],
                        mental_states_list[i] if i < len(mental_states_list) else '',
                        scores_attention[i] if i < len(scores_attention) else '',
                        scores_relaxation[i] if i < len(scores_relaxation) else '',
                        scores_creativity[i] if i < len(scores_creativity) else '',
                        scores_drowsiness[i] if i < len(scores_drowsiness) else '',
                        brainwave_delta[i] if i < len(brainwave_delta) else '',
                        brainwave_theta[i] if i < len(brainwave_theta) else '',
                        brainwave_alpha[i] if i < len(brainwave_alpha) else '',
                        brainwave_beta[i] if i < len(brainwave_beta) else '',
                        brainwave_gamma[i] if i < len(brainwave_gamma) else ''
                    ])
            print(f"✅ Data saved to {save_path}")

    def run(self):
        """Start the plotter application"""
        return self.app.exec()

class MuseStreamManager:
    # --- Configuration ---
    EEG_BANDS = [
        "/muse/elements/delta_absolute", "/muse/elements/theta_absolute",
        "/muse/elements/alpha_absolute", "/muse/elements/beta_absolute", 
        "/muse/elements/gamma_absolute", "/muse/elements/delta_relative",
        "/muse/elements/theta_relative", "/muse/elements/alpha_relative",
        "/muse/elements/beta_relative", "/muse/elements/gamma_relative"
    ]
    FIXED_PLAYBACK_DELAY = 0.5 
    
    # Brainwave addresses for comprehensive analysis
    DELTA_ABS = "/muse/elements/delta_absolute"
    THETA_ABS = "/muse/elements/theta_absolute" 
    ALPHA_ABS = "/muse/elements/alpha_absolute"
    BETA_ABS = "/muse/elements/beta_absolute"
    GAMMA_ABS = "/muse/elements/gamma_absolute"
    ALPHA_REL = "/muse/elements/alpha_relative"
    BETA_REL = "/muse/elements/beta_relative"
    # ---------------------

    def __init__(self, ip="0.0.0.0", port=3001, csv_path="muse_stream.csv", timeout=3.0, data_store=None):
        self.ip = ip 
        self.port = port
        self.csv_path = csv_path
        self.timeout = timeout
        self.data_received_event = threading.Event()
        self.mode = "detecting"
        self.data_store = data_store
        
        # Store all brainwave data
        self.latest_delta = None
        self.latest_theta = None
        self.latest_alpha = None
        self.latest_beta = None
        self.latest_gamma = None
        
        self.audio_thread = None
        self.stop_audio = threading.Event()
        self.audio_file = None
        self.board = None
        self.samplerate = None
        self.num_channels = None
        
        # Advanced brain state tracking
        self.current_mental_state = "neutral"
        self.attention_score = 0.5
        self.relaxation_score = 0.5
        self.creativity_score = 0.5
        self.effects_lock = threading.Lock()
        
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
        self.board = Pedalboard([
            Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=100),
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=0),
            HighShelfFilter(cutoff_frequency_hz=5000, gain_db=0),
            Chorus(rate_hz=0.5, depth=0.1, centre_delay_ms=7, feedback=0.3, mix=0.3),
            Phaser(rate_hz=0.3, depth=0.5, centre_frequency_hz=800, feedback=0.2, mix=0.1),
            Delay(delay_seconds=0.3, feedback=0.3, mix=0.1),
            Reverb(room_size=0.3, damping=0.5, wet_level=0.1, dry_level=0.9, width=0.8),
            Gain(gain_db=0)
        ])

    def clamp(self, value, min_val, max_val):
        """Ensure value stays within specified range"""
        return max(min_val, min(value, max_val))

    def calculate_mental_states(self):
        """Calculate comprehensive mental states based on all brainwaves"""
        if None in [self.latest_delta, self.latest_theta, self.latest_alpha, 
               self.latest_beta, self.latest_gamma]:
            return "neutral", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        
        try:
            # Calculate averages across electrodes
            delta_avg = np.mean([x for x in self.latest_delta if isinstance(x, (int, float))])
            theta_avg = np.mean([x for x in self.latest_theta if isinstance(x, (int, float))])
            alpha_avg = np.mean([x for x in self.latest_alpha if isinstance(x, (int, float))])
            beta_avg = np.mean([x for x in self.latest_beta if isinstance(x, (int, float))])
            gamma_avg = np.mean([x for x in self.latest_gamma if isinstance(x, (int, float))])
            
            # Total power for normalization
            total_power = delta_avg + theta_avg + alpha_avg + beta_avg + gamma_avg
            
            if total_power == 0:
                return "neutral", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            
            # Calculate relative percentages
            delta_rel = delta_avg / total_power
            theta_rel = theta_avg / total_power
            alpha_rel = alpha_avg / total_power
            beta_rel = beta_avg / total_power
            gamma_rel = gamma_avg / total_power
            
            # Keep original normalization for plotting (scaled for visualization)
            delta_norm = self.clamp(delta_avg / 100.0, 0.0, 1.0)  # Scale for plotting
            theta_norm = self.clamp(theta_avg / 50.0, 0.0, 1.0)
            alpha_norm = self.clamp(alpha_avg / 30.0, 0.0, 1.0)
            beta_norm = self.clamp(beta_avg / 20.0, 0.0, 1.0)
            gamma_norm = self.clamp(gamma_avg / 15.0, 0.0, 1.0)
            
            # Mental state calculations based on brainwave research:
            
            # 1. ATTENTION SCORE: Beta dominance = focused attention
            attention_score = self.clamp((beta_rel * 0.6 + gamma_rel * 0.4) * 2.5, 0.0, 1.0)
            
            # 2. RELAXATION SCORE: Alpha dominance = relaxed, calm
            relaxation_score = self.clamp(alpha_rel * 1.5, 0.0, 1.0)
            
            # 3. CREATIVITY SCORE: Theta/Alpha balance = creative, meditative
            creativity_score = self.clamp((theta_rel * 0.5 + alpha_rel * 0.5) * 1.5, 0.0, 1.0)
            
            # 4. DROWSINESS: Theta/Delta dominance = sleepy, drowsy
            drowsiness_score = self.clamp((theta_rel * 0.6 + delta_rel * 0.4) * 1, 0.0, 1.0)
            
            # Determine primary mental state
            scores = {
                "focused": attention_score,
                "relaxed": relaxation_score,
                "creative": creativity_score,
                "drowsy": drowsiness_score
            }
            
            primary_state = max(scores, key=scores.get)
            
            # Only classify if score is significant
            if scores[primary_state] < 0.3:
                primary_state = "neutral"

            return (primary_state, attention_score, relaxation_score, creativity_score, drowsiness_score,
                   delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)
            
        except Exception as e:
            print(f"❌ Error calculating mental states: {e}")
            return "neutral", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5

    def update_audio_effects_based_on_mental_state(self, mental_state, attention, relaxation, creativity):
        """Update audio effects based on comprehensive mental state analysis"""
        with self.effects_lock:
            if mental_state == "focused":
                # HIGH FOCUS: Clear, analytical, precise
                print("🎯 FOCUSED STATE: Clear, punchy audio")
                self.board[0].threshold_db = -15  # More compression
                self.board[0].ratio = 6
                self.board[1].gain_db = 6.0  # Strong bass boost
                self.board[2].gain_db = 3.0  # Treble boost for clarity
                self.board[3].mix = self.clamp(0.05, 0.0, 1.0)  # Minimal chorus
                self.board[4].mix = self.clamp(0.02, 0.0, 1.0)  # Minimal phaser
                self.board[5].mix = self.clamp(0.05, 0.0, 1.0)  # Minimal delay
                self.board[6].wet_level = self.clamp(0.05, 0.0, 1.0)  # Minimal reverb
                self.board[6].room_size = self.clamp(0.2, 0.0, 1.0)
                self.board[7].gain_db = 2.0  # Slight volume boost
                
            elif mental_state == "relaxed":
                # RELAXED: Smooth, warm, comfortable
                print("😌 RELAXED STATE: Warm, smooth audio")
                self.board[0].threshold_db = -22  # Less compression
                self.board[0].ratio = 3
                self.board[1].gain_db = 2.0  # Moderate bass
                self.board[2].gain_db = 0.0  # Neutral treble
                self.board[3].mix = self.clamp(0.2, 0.0, 1.0)  # Gentle chorus
                self.board[4].mix = self.clamp(0.1, 0.0, 1.0)  # Light phaser
                self.board[5].mix = self.clamp(0.15, 0.0, 1.0)  # Some delay
                self.board[6].wet_level = self.clamp(0.25, 0.0, 1.0)  # Moderate reverb
                self.board[6].room_size = self.clamp(0.5, 0.0, 1.0)
                self.board[7].gain_db = 0.0  # Normal volume
                
            elif mental_state == "creative":
                # CREATIVE: Expansive, imaginative, flowing
                print("🎨 CREATIVE STATE: Expansive, modulated audio")
                self.board[0].threshold_db = -25  # Minimal compression
                self.board[0].ratio = 2
                self.board[1].gain_db = -1.0  # Reduced bass for airiness
                self.board[2].gain_db = 1.0  # Slight treble emphasis
                self.board[3].mix = self.clamp(0.4, 0.0, 1.0)  # Strong chorus
                self.board[3].rate_hz = 0.3  # Slower modulation
                self.board[4].mix = self.clamp(0.35, 0.0, 1.0)  # Strong phaser
                self.board[5].mix = self.clamp(0.3, 0.0, 1.0)  # Noticeable delay
                self.board[6].wet_level = self.clamp(0.4, 0.0, 1.0)  # Lots of reverb
                self.board[6].room_size = self.clamp(0.8, 0.0, 1.0)
                self.board[7].gain_db = -1.0  # Slightly quieter
                
            elif mental_state == "drowsy":
                # DROWSY: Soft, distant, dreamlike
                print("💤 DROWSY STATE: Soft, distant audio")
                self.board[0].threshold_db = -30  # Very little compression
                self.board[0].ratio = 1.5
                self.board[1].gain_db = -3.0  # Reduced bass
                self.board[2].gain_db = -2.0  # Reduced treble
                self.board[3].mix = self.clamp(0.5, 0.0, 1.0)  # Maximum chorus
                self.board[3].rate_hz = 0.2  # Very slow
                self.board[4].mix = self.clamp(0.45, 0.0, 1.0)  # Maximum phaser
                self.board[5].mix = self.clamp(0.4, 0.0, 1.0)  # Strong delay
                self.board[6].wet_level = self.clamp(0.6, 0.0, 1.0)  # Heavy reverb
                self.board[6].room_size = self.clamp(0.9, 0.0, 1.0)
                self.board[7].gain_db = -3.0  # Quieter
                
            else:  # neutral
                # NEUTRAL: Balanced, natural
                print("⚖️ NEUTRAL STATE: Balanced audio")
                self.board[0].threshold_db = -20
                self.board[0].ratio = 4
                self.board[1].gain_db = 0.0
                self.board[2].gain_db = 0.0
                self.board[3].mix = self.clamp(0.15, 0.0, 1.0)
                self.board[4].mix = self.clamp(0.08, 0.0, 1.0)
                self.board[5].mix = self.clamp(0.1, 0.0, 1.0)
                self.board[6].wet_level = self.clamp(0.15, 0.0, 1.0)
                self.board[6].room_size = self.clamp(0.4, 0.0, 1.0)
                self.board[7].gain_db = 0.0

    def process_audio_in_realtime(self):
        """Process and play audio in real-time with dynamic effects"""
        try:
            self.audio_file = AudioFile('music.wav')
            self.samplerate = self.audio_file.samplerate
            self.num_channels = self.audio_file.num_channels
                
            print(f"🎵 Starting real-time audio processing: {self.samplerate}Hz, {self.num_channels} channels")
            
            def audio_callback(outdata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                
                if self.stop_audio.is_set():
                    raise sd.CallbackStop()
                
                # Update effects based on current mental state
                (mental_state, attention, relaxation, creativity, drowsiness,
                 delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()
                
                # Send data to plotter if available (but less frequently to avoid overloading)
                if self.data_store and hasattr(self, 'last_plot_time'):
                    current_time = time.time()
                    if current_time - self.last_plot_time > 0.1:  # Send every 100ms
                        self.data_store.data_updated.emit(mental_state, attention, relaxation, creativity, drowsiness,
                                                        delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)
                        self.last_plot_time = current_time
                elif self.data_store:
                    # First time
                    self.last_plot_time = time.time()
                    self.data_store.data_updated.emit(mental_state, attention, relaxation, creativity, drowsiness,
                                                    delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)
                
                self.update_audio_effects_based_on_mental_state(mental_state, attention, relaxation, creativity)
                
                # Read and process audio
                chunk = self.audio_file.read(frames)
                
                if chunk.size == 0:
                    raise sd.CallbackStop()
                
                with self.effects_lock:
                    effected = self.board(chunk, self.samplerate, reset=False)
                
                if effected.ndim == 2 and effected.shape[0] == self.num_channels:
                    effected = effected.T
                
                if effected.shape[0] < frames:
                    silence_length = frames - effected.shape[0]
                    silence = np.zeros((silence_length, self.num_channels))
                    outdata[:effected.shape[0]] = effected
                    outdata[effected.shape[0]:] = silence
                else:
                    outdata[:] = effected[:frames]

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

    def start(self):
        """Starts the detection and streaming process."""
        print(f"⏳ Attempting to connect to live stream on {self.ip}:{self.port}...")
        print("🧠 Advanced Brainwave Analysis Active:")
        print("   🎯 FOCUSED: Beta/Gamma = Clear, punchy audio")
        print("   😌 RELAXED: Alpha = Warm, smooth audio") 
        print("   🎨 CREATIVE: Theta/Alpha = Expansive, modulated audio")
        print("   💤 DROWSY: Theta/Delta = Soft, distant audio")
        print("   ⚖️ NEUTRAL: Balanced audio")
        
        self.start_audio_processing()
        
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()

        stream_found = self.data_received_event.wait(timeout=self.timeout)

        if stream_found:
            self.mode = "live"
            print("✅ Live stream detected. Analyzing mental states...")
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
        """Handles live incoming OSC messages for all brainwave types."""
        
        if not self.data_received_event.is_set():
            self.data_received_event.set()

        # Store all brainwave data
        if address == self.DELTA_ABS and args:
            self.latest_delta = list(args)
        elif address == self.THETA_ABS and args:
            self.latest_theta = list(args)
        elif address == self.ALPHA_ABS and args:
            self.latest_alpha = list(args)
        elif address == self.BETA_ABS and args:
            self.latest_beta = list(args)
        elif address == self.GAMMA_ABS and args:
            self.latest_gamma = list(args)

        # Log all brainwave data
        if address in self.EEG_BANDS:
            timestamp = datetime.now().isoformat()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, address, list(args)])

        # Only print when we have a complete set of brainwaves
        if (self.latest_delta is not None and self.latest_theta is not None and 
            self.latest_alpha is not None and self.latest_beta is not None and 
            self.latest_gamma is not None):
            
            (mental_state, attention, relaxation, creativity, drowsiness,
             delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()

            print(f"🧠 [{datetime.now().strftime('%H:%M:%S')}] {mental_state.upper():8} | "
                f"Attention: {attention:.2f} | Relaxation: {relaxation:.2f} | Creativity: {creativity:.2f} | Drowsiness: {drowsiness:.2f}")

            # Send data to plotter if available
            if self.data_store:
                self.data_store.data_updated.emit(mental_state, attention, relaxation, creativity, drowsiness,
                                    delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)

    def play_from_csv(self):
        """Enhanced CSV playback with mental state analysis"""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) < 100: 
            print("❌ CSV file is empty or missing data. Cannot replay.")
            return

        print("--- Starting CSV Playback Simulation ---")
        
        count = 0
        # Reset all brainwave data
        self.latest_delta = self.latest_theta = self.latest_alpha = self.latest_beta = self.latest_gamma = None

        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)

                for row in reader:
                    if len(row) < 3: continue
                    
                    address = row[1]
                    values_str = row[2]

                    # Update brainwave data
                    if address == self.DELTA_ABS:
                        self.latest_delta = ast.literal_eval(values_str)
                    elif address == self.THETA_ABS:
                        self.latest_theta = ast.literal_eval(values_str)
                    elif address == self.ALPHA_ABS:
                        self.latest_alpha = ast.literal_eval(values_str)
                    elif address == self.BETA_ABS:
                        self.latest_beta = ast.literal_eval(values_str)
                    elif address == self.GAMMA_ABS:
                        self.latest_gamma = ast.literal_eval(values_str)
                        
                    # Analyze when we have complete data
                    if (self.latest_delta is not None and self.latest_theta is not None and 
                        self.latest_alpha is not None and self.latest_beta is not None and 
                        self.latest_gamma is not None):
                        
                        (mental_state, attention, relaxation, creativity, drowsiness,
                         delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()

                        # Send data to plotter if available
                        if self.data_store:
                            self.data_store.data_updated.emit(mental_state, attention, relaxation, creativity, drowsiness,
                                                delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)

                        print(f"PLAYBACK 🧠 {mental_state.upper():8} | "
                            f"Attention: {attention:.2f} | Relaxation: {relaxation:.2f} | Creativity: {creativity:.2f} | Drowsiness: {drowsiness:.2f}")
                        count += 1
                        time.sleep(self.FIXED_PLAYBACK_DELAY)
                        
            print(f"--- Playback Finished. Processed {count} mental state updates. ---")

        except Exception as e:
            print(f"❌ Critical Error during CSV playback: {e}")
        finally:
            self.stop_audio.set()

def main():
    """Main function that runs everything in the correct thread"""
    # Create thread-safe data store
    data_store = DataStore()
    
    # Create and start plotter in main thread
    plotter = BrainwavePlotter(data_store)
    
    # Create Muse manager in a separate thread
    muse_manager = MuseStreamManager(ip="0.0.0.0", port=3001, timeout=3.0, data_store=data_store)
    
    # Start Muse processing in background thread
    muse_thread = threading.Thread(target=muse_manager.start, daemon=True)
    muse_thread.start()
    
    # Run the plotter in main thread (this blocks until window is closed)
    plotter.run()

if __name__ == "__main__":
    main()