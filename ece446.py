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
from collections import defaultdict
from PyQt6.QtWidgets import (QApplication, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout, QComboBox,
                             QLabel, QPushButton, QDoubleSpinBox, QFileDialog, QGridLayout)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

class BrainwavePlotter:
    def __init__(self, muse_manager, ip="127.0.0.1", port=7001):
        self.muse_manager = muse_manager
        self.ip = ip
        self.port = port
        
        # Data storage for plotting
        self.timestamps = []
        self.brainwave_data = {
            'delta': [],
            'theta': [], 
            'alpha': [],
            'beta': [],
            'gamma': []
        }
        self.mental_states = []
        self.scores = {
            'attention': [],
            'relaxation': [],
            'creativity': []
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

    def create_plots(self):
        """Create all the plots for brainwaves and mental states"""
        
        # Brainwave amplitudes plot
        self.brainwave_plot = pg.PlotWidget(title="Brainwave Amplitudes (Normalized)")
        self.brainwave_plot.setLabel('left', 'Normalized Amplitude')
        self.brainwave_plot.setLabel('bottom', 'Time (seconds)')
        self.brainwave_plot.addLegend()
        self.brainwave_plot.showGrid(x=True, y=True)
        
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
            'creativity': self.scores_plot.plot(pen=pg.mkPen('green', width=2), name='Creativity')
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
        
        self.scroll_layout.addWidget(self.state_plot)

    def update_data(self, mental_state, attention, relaxation, creativity, 
                   delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm):
        """Update data for plotting"""
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
        
        # Keep only recent data based on time range
        time_range = self.time_range_spin.value()
        cutoff_time = current_time - time_range
        
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.pop(0)
            self.mental_states.pop(0)
            for wave in self.brainwave_data:
                self.brainwave_data[wave].pop(0)
            for score in self.scores:
                self.scores[score].pop(0)

    def update_plots(self):
        """Update all plots with current data"""
        if not self.timestamps:
            return
            
        # Convert timestamps to relative seconds for x-axis
        current_time = time.time()
        time_range = self.time_range_spin.value()
        x_data = [t - (current_time - time_range) for t in self.timestamps]
        
        # Update brainwave plot
        for wave, curve in self.brainwave_curves.items():
            if self.brainwave_data[wave]:
                curve.setData(x_data, self.brainwave_data[wave])
        
        # Update scores plot
        for score_type, curve in self.score_curves.items():
            if self.scores[score_type]:
                curve.setData(x_data, self.scores[score_type])
        
        # Update state plot
        if self.mental_states:
            state_numeric = [self.state_mapping.get(state, 0) for state in self.mental_states]
            self.state_curve.setData(x_data, state_numeric)
        
        # Update status and current state
        if self.mental_states:
            current_state = self.mental_states[-1]
            state_colors = {
                'neutral': 'black',
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
                             f"Creativity: {self.scores['creativity'][-1]:.2f}")
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
                # Write header
                writer.writerow(['timestamp', 'mental_state', 'attention', 'relaxation', 'creativity',
                               'delta', 'theta', 'alpha', 'beta', 'gamma'])
                
                # Write data
                for i, timestamp in enumerate(self.timestamps):
                    writer.writerow([
                        timestamp,
                        self.mental_states[i],
                        self.scores['attention'][i],
                        self.scores['relaxation'][i],
                        self.scores['creativity'][i],
                        self.brainwave_data['delta'][i],
                        self.brainwave_data['theta'][i],
                        self.brainwave_data['alpha'][i],
                        self.brainwave_data['beta'][i],
                        self.brainwave_data['gamma'][i]
                    ])
            print(f"✅ Data saved to {save_path}")

    def run(self):
        """Start the plotter application"""
        return self.app.exec()

class MuseStreamManager:
    # ... (previous MuseStreamManager code remains the same until the calculate_mental_states method)
    
    def __init__(self, ip="0.0.0.0", port=3001, csv_path="muse_stream.csv", timeout=3.0, enable_plotting=True):
        self.ip = ip 
        self.port = port
        self.csv_path = csv_path
        self.timeout = timeout
        self.data_received_event = threading.Event()
        self.mode = "detecting"
        self.enable_plotting = enable_plotting
        
        # Store all brainwave data
        self.latest_delta = None
        self.latest_theta = None
        self.latest_alpha = None
        self.latest_beta = None
        self.latest_gamma = None
        
        # More conservative wave-specific normalization baselines
        self.wave_baselines = {
            'delta': 100.0,   # Delta waves are typically highest amplitude
            'theta': 50.0,    # Theta is medium-high
            'alpha': 25.0,    # Alpha is medium
            'beta': 15.0,     # Beta is lower
            'gamma': 8.0      # Gamma is typically lowest
        }
        
        # Dynamic calibration (will adjust based on actual data)
        self.dynamic_baselines = self.wave_baselines.copy()
        self.calibration_samples = []
        self.is_calibrated = False
        
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
        
        # For plotting
        self.plotter = None
        self.plotter_thread = None
        
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

    def wave_specific_normalization(self, delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg):
        """Apply wave-specific normalization with more conservative scaling"""
        
        # Use sigmoid-like function for more gradual normalization
        def sigmoid_normalize(value, baseline):
            # Sigmoid function that maps to 0-1 range more gradually
            x = value / baseline
            return 1 / (1 + np.exp(-(x - 0.5) * 4))  # More gradual slope
        
        # Apply sigmoid normalization to each wave
        delta_norm = sigmoid_normalize(delta_avg, self.dynamic_baselines['delta'])
        theta_norm = sigmoid_normalize(theta_avg, self.dynamic_baselines['theta'])
        alpha_norm = sigmoid_normalize(alpha_avg, self.dynamic_baselines['alpha'])
        beta_norm = sigmoid_normalize(beta_avg, self.dynamic_baselines['beta'])
        gamma_norm = sigmoid_normalize(gamma_avg, self.dynamic_baselines['gamma'])
        
        # Apply additional scaling to prevent maxing out
        scale_factor = 0.7  # Keep values in more reasonable range
        delta_norm *= scale_factor
        theta_norm *= scale_factor
        alpha_norm *= scale_factor
        beta_norm *= scale_factor
        gamma_norm *= scale_factor
        
        return (
            self.clamp(delta_norm, 0.0, 1.0),
            self.clamp(theta_norm, 0.0, 1.0),
            self.clamp(alpha_norm, 0.0, 1.0),
            self.clamp(beta_norm, 0.0, 1.0),
            self.clamp(gamma_norm, 0.0, 1.0)
        )

    def update_dynamic_baselines(self, delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg):
        """Update baselines based on recent data for adaptive normalization"""
        if len(self.calibration_samples) < 30:  # Reduced to 30 samples for faster calibration
            self.calibration_samples.append((delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg))
            return False
        
        if not self.is_calibrated:
            # Calculate moving averages for each wave
            deltas = [s[0] for s in self.calibration_samples]
            thetas = [s[1] for s in self.calibration_samples]
            alphas = [s[2] for s in self.calibration_samples]
            betas = [s[3] for s in self.calibration_samples]
            gammas = [s[4] for s in self.calibration_samples]
            
            # Update baselines with exponential moving average
            alpha = 0.05  # More conservative smoothing
            self.dynamic_baselines['delta'] = (1 - alpha) * self.dynamic_baselines['delta'] + alpha * np.percentile(deltas, 75)  # Use 75th percentile
            self.dynamic_baselines['theta'] = (1 - alpha) * self.dynamic_baselines['theta'] + alpha * np.percentile(thetas, 75)
            self.dynamic_baselines['alpha'] = (1 - alpha) * self.dynamic_baselines['alpha'] + alpha * np.percentile(alphas, 75)
            self.dynamic_baselines['beta'] = (1 - alpha) * self.dynamic_baselines['beta'] + alpha * np.percentile(betas, 75)
            self.dynamic_baselines['gamma'] = (1 - alpha) * self.dynamic_baselines['gamma'] + alpha * np.percentile(gammas, 75)
            
            self.is_calibrated = True
            print(f"🎯 Dynamic calibration complete! Baselines: Delta={self.dynamic_baselines['delta']:.1f}, "
                  f"Theta={self.dynamic_baselines['theta']:.1f}, Alpha={self.dynamic_baselines['alpha']:.1f}, "
                  f"Beta={self.dynamic_baselines['beta']:.1f}, Gamma={self.dynamic_baselines['gamma']:.1f}")
        
        return True

    def calculate_mental_states(self):
        """Calculate comprehensive mental states with improved normalization"""
        if None in [self.latest_delta, self.latest_theta, self.latest_alpha, 
                   self.latest_beta, self.latest_gamma]:
            return "neutral", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        
        try:
            # Calculate averages across electrodes
            delta_avg = np.mean([x for x in self.latest_delta if isinstance(x, (int, float))])
            theta_avg = np.mean([x for x in self.latest_theta if isinstance(x, (int, float))])
            alpha_avg = np.mean([x for x in self.latest_alpha if isinstance(x, (int, float))])
            beta_avg = np.mean([x for x in self.latest_beta if isinstance(x, (int, float))])
            gamma_avg = np.mean([x for x in self.latest_gamma if isinstance(x, (int, float))])
            
            # Update dynamic baselines
            self.update_dynamic_baselines(delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg)
            
            # Apply wave-specific normalization
            delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm = self.wave_specific_normalization(
                delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg
            )
            
            # Calculate relative percentages from NORMALIZED values
            total_normalized = delta_norm + theta_norm + alpha_norm + beta_norm + gamma_norm
            
            if total_normalized == 0:
                return "neutral", 0.5, 0.5, 0.5, delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm
            
            delta_rel = delta_norm / total_normalized
            theta_rel = theta_norm / total_normalized
            alpha_rel = alpha_norm / total_normalized
            beta_rel = beta_norm / total_normalized
            gamma_rel = gamma_norm / total_normalized
            
            # MORE CONSERVATIVE Mental state calculations:
            
            # 1. ATTENTION SCORE: Use Beta dominance with saturation control
            attention_score = self.clamp(beta_rel * 1.5, 0.0, 0.9)  # Cap at 0.9 to prevent maxing out
            
            # 2. RELAXATION SCORE: Alpha dominance with saturation control
            relaxation_score = self.clamp(alpha_rel * 1.3, 0.0, 0.9)
            
            # 3. CREATIVITY SCORE: Theta + Alpha combination
            creativity_score = self.clamp((theta_rel * 0.5 + alpha_rel * 0.5) * 1.2, 0.0, 0.9)
            
            # 4. DROWSINESS: Only when Theta/Delta are dominant
            drowsiness_score = self.clamp((theta_rel * 0.7 + delta_rel * 0.3) * 1.1, 0.0, 0.9)
            
            # Determine primary mental state
            scores = {
                "focused": attention_score,
                "relaxed": relaxation_score,
                "creative": creativity_score,
                "drowsy": drowsiness_score
            }
            
            primary_state = max(scores, key=scores.get)
            
            # More conservative thresholds
            classification_thresholds = {
                "focused": 0.3,    # Higher threshold for focused
                "relaxed": 0.28,   # Higher threshold for relaxed  
                "creative": 0.25,  # Higher threshold for creative
                "drowsy": 0.35     # Much higher threshold for drowsy
            }
            
            # Only classify if score meets threshold AND is clearly dominant
            if scores[primary_state] < classification_thresholds[primary_state]:
                primary_state = "neutral"
            else:
                # Check if this state is clearly dominant (15% higher than next)
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.12:
                    primary_state = "neutral"  # Too close to call
                    
            return (primary_state, attention_score, relaxation_score, creativity_score,
                   delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)
            
        except Exception as e:
            print(f"❌ Error calculating mental states: {e}")
            return "neutral", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5

    def update_audio_effects_based_on_mental_state(self, mental_state, attention, relaxation, creativity):
        """Update audio effects with more gradual scaling"""
        with self.effects_lock:
            if mental_state == "focused":
                # HIGH FOCUS: Clear, analytical, precise
                focus_strength = self.clamp(attention * 1.5, 0.0, 1.0)  # Reduced multiplier
                print(f"🎯 FOCUSED STATE (strength: {focus_strength:.2f}): Clear, punchy audio")
                self.board[0].threshold_db = -20 + (focus_strength * -3)  # Reduced effect
                self.board[0].ratio = 4 + (focus_strength * 1.5)
                self.board[1].gain_db = 1.0 + (focus_strength * 4.0)  # Reduced bass boost
                self.board[2].gain_db = 0.5 + (focus_strength * 1.5)
                self.board[3].mix = self.clamp(0.2 - (focus_strength * 0.15), 0.0, 1.0)
                self.board[4].mix = self.clamp(0.1 - (focus_strength * 0.06), 0.0, 1.0)
                self.board[5].mix = self.clamp(0.15 - (focus_strength * 0.12), 0.0, 1.0)
                self.board[6].wet_level = self.clamp(0.2 - (focus_strength * 0.15), 0.0, 1.0)
                self.board[6].room_size = self.clamp(0.4 - (focus_strength * 0.15), 0.0, 1.0)
                self.board[7].gain_db = focus_strength * 1.5
                
            elif mental_state == "relaxed":
                # RELAXED: Smooth, warm, comfortable
                relax_strength = self.clamp(relaxation * 1.2, 0.0, 1.0)  # Reduced multiplier
                print(f"😌 RELAXED STATE (strength: {relax_strength:.2f}): Warm, smooth audio")
                self.board[0].threshold_db = -22
                self.board[0].ratio = 3
                self.board[1].gain_db = 0.5 + (relax_strength * 0.8)
                self.board[2].gain_db = -0.3 + (relax_strength * 0.3)
                self.board[3].mix = self.clamp(0.15 + (relax_strength * 0.08), 0.0, 1.0)
                self.board[4].mix = self.clamp(0.08 + (relax_strength * 0.05), 0.0, 1.0)
                self.board[5].mix = self.clamp(0.1 + (relax_strength * 0.08), 0.0, 1.0)
                self.board[6].wet_level = self.clamp(0.15 + (relax_strength * 0.15), 0.0, 1.0)
                self.board[6].room_size = self.clamp(0.4 + (relax_strength * 0.2), 0.0, 1.0)
                self.board[7].gain_db = -0.5 + (relax_strength * 0.5)
                
            elif mental_state == "creative":
                # CREATIVE: Expansive, imaginative, flowing
                creative_strength = self.clamp(creativity * 1.1, 0.0, 1.0)  # Reduced multiplier
                print(f"🎨 CREATIVE STATE (strength: {creative_strength:.2f}): Expansive, modulated audio")
                self.board[0].threshold_db = -24
                self.board[0].ratio = 2.5
                self.board[1].gain_db = -0.5
                self.board[2].gain_db = 0.3 + (creative_strength * 0.4)
                self.board[3].mix = self.clamp(0.2 + (creative_strength * 0.2), 0.0, 1.0)
                self.board[3].rate_hz = 0.4 - (creative_strength * 0.15)
                self.board[4].mix = self.clamp(0.12 + (creative_strength * 0.18), 0.0, 1.0)
                self.board[5].mix = self.clamp(0.15 + (creative_strength * 0.15), 0.0, 1.0)
                self.board[6].wet_level = self.clamp(0.2 + (creative_strength * 0.2), 0.0, 1.0)
                self.board[6].room_size = self.clamp(0.45 + (creative_strength * 0.3), 0.0, 1.0)
                self.board[7].gain_db = -0.5
                
            elif mental_state == "drowsy":
                # DROWSY: Soft, distant, dreamlike
                drowsy_strength = self.clamp(creativity * 1.1, 0.0, 1.0)  # Reduced multiplier
                print(f"💤 DROWSY STATE (strength: {drowsy_strength:.2f}): Soft, distant audio")
                self.board[0].threshold_db = -26
                self.board[0].ratio = 2.0
                self.board[1].gain_db = -1.5
                self.board[2].gain_db = -1.0
                self.board[3].mix = self.clamp(0.25 + (drowsy_strength * 0.15), 0.0, 1.0)
                self.board[3].rate_hz = 0.35
                self.board[4].mix = self.clamp(0.2 + (drowsy_strength * 0.15), 0.0, 1.0)
                self.board[5].mix = self.clamp(0.2 + (drowsy_strength * 0.12), 0.0, 1.0)
                self.board[6].wet_level = self.clamp(0.25 + (drowsy_strength * 0.25), 0.0, 1.0)
                self.board[6].room_size = self.clamp(0.5 + (drowsy_strength * 0.25), 0.0, 1.0)
                self.board[7].gain_db = -1.5
                
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
                (mental_state, attention, relaxation, creativity,
                 delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()
                
                # Send data to plotter if enabled
                if self.enable_plotting and self.plotter:
                    self.plotter.update_data(mental_state, attention, relaxation, creativity,
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

    def start_plotter(self):
        """Start the plotter in a separate thread"""
        if self.enable_plotting:
            self.plotter = BrainwavePlotter(self)
            # Run plotter in main thread (required for PyQt)
            self.plotter.run()

    def start(self):
        """Starts the detection and streaming process."""
        print(f"⏳ Attempting to connect to live stream on {self.ip}:{self.port}...")
        print("🧠 WAVE-SPECIFIC NORMALIZATION ACTIVE:")
        print("   📊 Delta: 50 (high amplitude) | Theta: 30 | Alpha: 20 | Beta: 10 | Gamma: 5 (low)")
        print("   🎯 Dynamic calibration will adapt to your brain's unique patterns")
        print("   🧠 States: FOCUSED • RELAXED • CREATIVE • DROWSY • NEUTRAL")
        
        # Start plotter if enabled
        if self.enable_plotting:
            plotter_thread = threading.Thread(target=self.start_plotter, daemon=True)
            plotter_thread.start()
            time.sleep(2)  # Give plotter time to initialize
        
        self.start_audio_processing()
        
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()

        stream_found = self.data_received_event.wait(timeout=self.timeout)

        if stream_found:
            self.mode = "live"
            print("✅ Live stream detected. Calibrating and analyzing mental states...")
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
            
            (mental_state, attention, relaxation, creativity,
             delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()
            
            calibration_status = "CALIBRATED" if self.is_calibrated else "CALIBRATING"
            print(f"🧠 [{datetime.now().strftime('%H:%M:%S')}] {mental_state.upper():8} [{calibration_status}] | "
                  f"Attention: {attention:.2f} | Relaxation: {relaxation:.2f} | Creativity: {creativity:.2f}")

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
                        
                        (mental_state, attention, relaxation, creativity,
                         delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm) = self.calculate_mental_states()
                        
                        # Send data to plotter if enabled
                        if self.enable_plotting and self.plotter:
                            self.plotter.update_data(mental_state, attention, relaxation, creativity,
                                                   delta_norm, theta_norm, alpha_norm, beta_norm, gamma_norm)
                        
                        calibration_status = "CALIBRATED" if self.is_calibrated else "CALIBRATING"
                        print(f"PLAYBACK 🧠 {mental_state.upper():8} [{calibration_status}] | "
                              f"Attention: {attention:.2f} | Relaxation: {relaxation:.2f} | Creativity: {creativity:.2f}")
                        count += 1
                        time.sleep(self.FIXED_PLAYBACK_DELAY)
                        
            print(f"--- Playback Finished. Processed {count} mental state updates. ---")

        except Exception as e:
            print(f"❌ Critical Error during CSV playback: {e}")
        finally:
            self.stop_audio.set()

if __name__ == "__main__":
    # Create reference output file
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
    
    manager = MuseStreamManager(ip="0.0.0.0", port=3001, timeout=3.0, enable_plotting=True) 
    manager.start()