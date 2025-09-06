import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import joblib
import traceback
import sys
import pylsl
import seaborn as sns
from datetime import datetime
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Configuration ===
DATA_DIR = 'data'
MODEL_DIR = 'models'
LOG_DIR = 'logs'
BENCHMARK_DIR = 'benchmarks'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

MODEL_CNN = os.path.join(MODEL_DIR, 'project4_cnn_model.keras')
MODEL_SVM = os.path.join(MODEL_DIR, 'project4_svm.pkl')

CHANNELS = 8
FS = 256
DEFAULT_WIN_SIZE = 2
WIN_LEN = FS * DEFAULT_WIN_SIZE

# === Preprocessing Functions ===
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(fs, freq=50, q=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, q)
    return b, a

def preprocess(sig):
    """Apply bandpass and notch filtering to EEG signal"""
    # Bandpass filter (1-45 Hz)
    b, a = butter_bandpass(1, 45, FS)
    filtered = lfilter(b, a, sig)
    
    # Notch filter (50 Hz)
    bn, an = notch_filter(FS)
    filtered = lfilter(bn, an, filtered)
    
    # Remove DC offset
    return filtered - np.mean(filtered)

# === Model Architecture ===
def build_cnn(input_shape):
    """Build 1D CNN model for EEG classification with adaptive pooling"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(32, kernel_size=32, strides=4, 
                               padding='same', activation='relu'),
        tf.keras.layers.AveragePooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=16, strides=2, 
                               padding='same', activation='relu'),
        tf.keras.layers.AveragePooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# === Data Generation ===
def generate_sample_data():
    """Generate sample EEG data with validation checks"""
    n_samples = FS * 60 * 5  # 5 minutes of data per file
    
    for label, name in [(0, 'low'), (1, 'high')]:
        for i in range(3):  # 3 files per class
            filename = f"eeg_{name}_{i+1}.csv"
            filepath = os.path.join(DATA_DIR, filename)
            
            if os.path.exists(filepath):
                continue
                
            data = np.zeros((n_samples, CHANNELS))
            
            # Simulate EEG with different frequency characteristics
            t = np.arange(n_samples) / FS
            for ch in range(CHANNELS):
                # Base signal (alpha/beta waves)
                base = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
                
                # Add class-specific characteristics
                if label == 1:  # High cognitive load
                    base += 0.4 * np.sin(2 * np.pi * 40 * t) + 0.2 * np.random.normal(size=n_samples)
                else:  # Low cognitive load
                    base += 0.1 * np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.normal(size=n_samples)
                
                # Add noise and drift
                data[:, ch] = base + 0.05 * np.random.normal(size=n_samples) + 0.01 * t
                
            # Create DataFrame and save
            df = pd.DataFrame(data, columns=[f'Channel_{i}' for i in range(CHANNELS)])
            df.to_csv(filepath, index=False)
            
    return True

# === Training Routine ===
def train_models(status_callback):
    """Train both CNN and SVM models with comprehensive validation"""
    # Create sample data if none exists
    if not os.listdir(DATA_DIR):
        status_callback('Generating sample EEG data...')
        if not generate_sample_data():
            return 0, 0, "Failed to generate sample data"
    
    # Load and prepare dataset
    X_cnn, X_svm, y = [], [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not files:
        return 0, 0, "No valid EEG files found in data directory"
    
    status_callback(f'Processing {len(files)} EEG files...')
    
    valid_files = 0
    for file in files:
        try:
            filepath = os.path.join(DATA_DIR, file)
            df = pd.read_csv(filepath)
            
            # Validate data shape
            if df.shape[1] != CHANNELS:
                status_callback(f"Skipping {file}: Expected {CHANNELS} channels, got {df.shape[1]}")
                continue
                
            # Get label from filename
            label = 0 if 'low' in file.lower() else 1
            data = df.values.T  # Transpose to (channels, samples)
            
            # Create windows with 50% overlap
            step = WIN_LEN // 2
            num_windows = (data.shape[1] - WIN_LEN) // step + 1
            
            if num_windows < 1:
                status_callback(f"Skipping {file}: Data too short ({data.shape[1]} samples) for window size {WIN_LEN}")
                continue
                
            valid_files += 1
            
            for i in range(0, data.shape[1] - WIN_LEN + 1, step):
                window_data = data[:, i:i+WIN_LEN]
                
                # Preprocess for CNN
                proc_window = np.stack([preprocess(ch) for ch in window_data])
                X_cnn.append(proc_window.T)
                
                # Feature extraction for SVM
                freqs, psd = welch(window_data, fs=FS, nperseg=WIN_LEN, axis=1)
                features = []
                for ch in range(CHANNELS):
                    for band in [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]:
                        band_mask = (freqs >= band[0]) & (freqs <= band[1])
                        band_power = np.trapz(psd[ch][band_mask], freqs[band_mask])
                        features.append(band_power)
                X_svm.append(features)
                y.append(label)
                
        except Exception as e:
            status_callback(f"Error processing {file}: {str(e)}")
            continue
    
    # Check if we have enough data
    if not X_cnn or not X_svm:
        return 0, 0, "No valid windows extracted from data"
    
    status_callback(f'Training models with {len(X_cnn)} samples...')
    
    # Convert to arrays
    X_cnn = np.array(X_cnn)
    X_svm = np.array(X_svm)
    y = np.array(y)
    
    # Train-test split with validation
    try:
        (X_cnn_tr, X_cnn_te, X_svm_tr, X_svm_te, 
         y_tr, y_te) = train_test_split(X_cnn, X_svm, y, test_size=0.2, 
                                        stratify=y, random_state=42)
    except Exception as e:
        return 0, 0, f"Train-test split failed: {str(e)}"
    
    # Train CNN
    try:
        cnn = build_cnn((WIN_LEN, CHANNELS))
        cnn.fit(X_cnn_tr, y_tr, validation_data=(X_cnn_te, y_te), 
                epochs=15, batch_size=32, verbose=0)
        cnn.save(MODEL_CNN)
        cnn_acc = cnn.evaluate(X_cnn_te, y_te, verbose=0)[1]
    except Exception as e:
        cnn_acc = 0
        status_callback(f"CNN training failed: {str(e)}")
    
    # Train SVM
    try:
        svm = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        )
        svm.fit(X_svm_tr, y_tr)
        joblib.dump(svm, MODEL_SVM)
        svm_acc = svm.score(X_svm_te, y_te)
    except Exception as e:
        svm_acc = 0
        status_callback(f"SVM training failed: {str(e)}")
    
    # Return results
    status = (f"Training complete! CNN Acc: {cnn_acc:.2f}, SVM Acc: {svm_acc:.2f}\n"
              f"Trained on {valid_files} files with {len(X_cnn)} windows")
    return cnn_acc, svm_acc, status

# === LSL Integration ===
class LSLStream:
    def __init__(self, status_callback):
        self.status_callback = status_callback
        self.inlet = None
        self.running = False
        
    def connect(self):
        try:
            # Resolve EEG streams
            streams = pylsl.resolve_stream('type', 'EEG')
            if not streams:
                self.status_callback("No EEG streams found via LSL")
                return False
                
            # Connect to the first stream
            self.inlet = pylsl.StreamInlet(streams[0])
            info = self.inlet.info()
            self.status_callback(f"Connected to LSL stream: {info.name()}")
            return True
        except Exception as e:
            self.status_callback(f"LSL connection error: {str(e)}")
            return False
            
    def get_window(self, win_len):
        """Get a window of EEG data from LSL stream"""
        samples = []
        start_time = pylsl.local_clock()
        
        # Collect samples until we have a full window
        while len(samples) < win_len:
            sample, timestamp = self.inlet.pull_sample(timeout=0.1)
            if sample:
                samples.append(sample)
                
        # Convert to numpy array and transpose
        return np.array(samples).T
    
    def start(self):
        """Start LSL connection"""
        if not self.connect():
            return False
        self.running = True
        return True
        
    def stop(self):
        """Stop LSL connection"""
        if self.inlet:
            self.inlet.close_stream()
        self.running = False

# === Real-time Processing Thread ===
class EEGProcessor:
    def __init__(self, status_callback, update_ui_callback):
        self.running = False
        self.thread = None
        self.log = []
        self.data_files = []
        self.current_data = None
        self.file_index = 0
        self.position = 0
        self.cnn_model = None
        self.svm_model = None
        self.active_model = None
        self.status_callback = status_callback
        self.update_ui_callback = update_ui_callback
        self.lsl_stream = LSLStream(status_callback)
        self.win_len = WIN_LEN
        self.confidence_history = []
        self.latency_history = []
        # Don't load models immediately - wait until UI is ready
        self.models_loaded = False
        self.data_loaded = False
        
    def ensure_models_loaded(self):
        """Load models when UI is ready"""
        if not self.models_loaded:
            self.load_models()
            self.models_loaded = True
            
    def ensure_data_loaded(self):
        """Load data when UI is ready"""
        if not self.data_loaded:
            self.load_data()
            self.data_loaded = True
    
    def load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists(MODEL_CNN):
                self.cnn_model = tf.keras.models.load_model(MODEL_CNN)
                self.status_callback("CNN model loaded successfully")
            else:
                self.status_callback("CNN model not found")
                
            if os.path.exists(MODEL_SVM):
                self.svm_model = joblib.load(MODEL_SVM)
                self.status_callback("SVM model loaded successfully")
            else:
                self.status_callback("SVM model not found")
                
        except Exception as e:
            self.status_callback(f"Model loading error: {str(e)}")
    
    def load_data(self):
        """Load EEG data from CSV files"""
        try:
            self.data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            if not self.data_files:
                self.status_callback("No EEG data files found. Generating sample data...")
                generate_sample_data()
                self.data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
                if not self.data_files:
                    raise Exception("Failed to generate sample data")
                    
            self.status_callback(f"Loaded {len(self.data_files)} EEG files")
            return True
            
        except Exception as e:
            self.status_callback(f"Data loading error: {str(e)}")
            return False
    
    def start(self, ax_ts, ax_psd, params):
        """Start the real-time processing thread"""
        # Ensure models and data are loaded
        self.ensure_models_loaded()
        self.ensure_data_loaded()
        
        # Store parameters for thread
        self.current_params = params
        model_type = params['model_type']
        win_size = int(params['win_size'])
        self.win_len = win_size * FS
        
        # Set active model
        if model_type == 'CNN' and self.cnn_model:
            self.active_model = 'CNN'
            self.update_ui_callback('model_active', 'CNN (Loaded)')
        elif model_type == 'SVM' and self.svm_model:
            self.active_model = 'SVM'
            self.update_ui_callback('model_active', 'SVM (Loaded)')
        else:
            self.status_callback('Model not available!')
            return False
            
        # Set data source
        if params['source'] == 'simulation':
            self.update_ui_callback('source_active', 'Simulation')
            if not self.data_files:
                if not self.load_data():
                    return False
                
            # Get selected file
            selected_file = params['selected_file']
            if selected_file != 'All Files (Loop)':
                if selected_file in self.data_files:
                    self.data_files = [selected_file]
                else:
                    messagebox.showerror("Error", f"Selected file not found: {selected_file}")
                    return False
                    
            # Load initial data file
            self.file_index = 0
            self.position = 0
            if not self.load_current_file():
                return False
        else:
            self.update_ui_callback('source_active', 'Live EEG (LSL)')
            if not self.lsl_stream.start():
                return False
                
        try:
            self.running = True
            self.thread = threading.Thread(
                target=self.process, 
                args=(ax_ts, ax_psd),
                daemon=True
            )
            self.thread.start()
            return True
            
        except Exception as e:
            self.status_callback(f"Error starting processing thread: {str(e)}")
            return False
    
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.lsl_stream.stop()
    
    def process(self, ax_ts, ax_psd):
        """Real-time processing loop"""
        try:
            # Use parameters stored in main thread
            params = self.current_params
            noise_level = params['noise_level'] / 1000.0
            speed = params['speed']
            threshold = params['threshold'] / 100.0
            step_size = max(1, self.win_len // 4)  # Ensure at least 1
            
            while self.running:
                start_time = time.time()
                
                # Get EEG data based on source
                if params['source'] == 'simulation':
                    # Check if we need to switch to next file
                    if self.position + self.win_len > self.current_data.shape[1]:
                        self.file_index = (self.file_index + 1) % len(self.data_files)
                        self.position = 0
                        self.load_current_file()
                    
                    # Extract current window
                    window_data = self.current_data[:, self.position:self.position+self.win_len]
                    self.position += step_size
                    
                    # Add noise
                    noise = noise_level * np.random.normal(size=window_data.shape)
                    noisy_data = window_data + noise
                    
                    # Preprocess
                    processed = np.array([preprocess(ch) for ch in noisy_data])
                    current_file = self.data_files[self.file_index]
                    
                else:  # Live EEG
                    window_data = self.lsl_stream.get_window(self.win_len)
                    processed = np.array([preprocess(ch) for ch in window_data])
                    current_file = "Live EEG"
                
                # Update time series plot
                for i, ax in enumerate(ax_ts.flatten()):
                    if i < CHANNELS:
                        ax.clear()
                        ax.plot(processed[i], linewidth=0.8)
                        ax.set_ylim(-3, 3)
                        ax.set_title(f'Ch {i+1}', fontsize=8)
                        ax.set_xticks([])
                        ax.grid(True, linestyle='--', alpha=0.6)
                
                fig_ts = ax_ts[0, 0].get_figure()
                fig_ts.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
                fig_ts.canvas.draw()
                
                # Update PSD plot
                ax_psd.clear()
                for ch in range(min(2, CHANNELS)):  # Show first 2 channels to avoid clutter
                    freqs, psd = welch(processed[ch], fs=FS, nperseg=256)
                    ax_psd.semilogy(freqs, psd, label=f'Ch {ch+1}', alpha=0.7)
                
                ax_psd.set_xlim(1, 45)
                ax_psd.set_xlabel('Frequency (Hz)', fontsize=9)
                ax_psd.set_ylabel('Power', fontsize=9)
                ax_psd.set_title('Power Spectral Density', fontsize=10)
                ax_psd.legend(fontsize=8)
                ax_psd.grid(True, linestyle='--', alpha=0.6)
                fig_psd = ax_psd.get_figure()
                fig_psd.tight_layout()
                fig_psd.canvas.draw()
                
                # Perform classification
                if self.active_model == 'CNN':
                    # CNN expects (samples, channels) format
                    input_data = processed.T[np.newaxis, ...]
                    prediction = self.cnn_model.predict(input_data, verbose=0)[0]
                    confidence = np.max(prediction)
                    class_idx = np.argmax(prediction)
                else:  # SVM
                    # Extract PSD features
                    features = []
                    for ch in range(CHANNELS):
                        freqs, psd = welch(processed[ch], fs=FS, nperseg=self.win_len)
                        for band in [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]:
                            band_mask = (freqs >= band[0]) & (freqs <= band[1])
                            band_power = np.trapz(psd[band_mask], freqs[band_mask])
                            features.append(band_power)
                    
                    # Predict
                    proba = self.svm_model.predict_proba([features])[0]
                    confidence = np.max(proba)
                    class_idx = np.argmax(proba)
                
                # Update GUI
                label = "Low Load" if class_idx == 0 else "High Load"
                self.update_ui_callback('prediction', label)
                self.update_ui_callback('confidence', int(confidence * 100))
                
                # Apply confidence threshold
                if confidence < threshold:
                    self.update_ui_callback('prediction_color', 'red')
                else:
                    self.update_ui_callback('prediction_color', 'white')
                
                # Calculate and display latency
                latency = (time.time() - start_time) * 1000
                self.update_ui_callback('latency', f'{latency:.1f} ms')
                
                # Store for metrics
                self.confidence_history.append(confidence)
                self.latency_history.append(latency)
                
                # Log results
                self.log.append([
                    pd.Timestamp.now(),
                    current_file,
                    self.active_model,
                    label,
                    confidence,
                    latency
                ])
                
                # Control processing speed
                time.sleep(max(0, 1/speed - latency/1000))
                
        except Exception as e:
            self.status_callback(f"Processing error: {str(e)}")
            self.running = False
            traceback.print_exc()
    
    def load_current_file(self):
        """Load the current EEG file with validation"""
        try:
            filepath = os.path.join(DATA_DIR, self.data_files[self.file_index])
            df = pd.read_csv(filepath)
            
            # Validate data shape
            if df.shape[1] != CHANNELS:
                self.status_callback(
                    f"Invalid channel count in {self.data_files[self.file_index]}: "
                    f"Expected {CHANNELS}, got {df.shape[1]}"
                )
                return False
                
            self.current_data = df.values.T
            self.status_callback(
                f"Loaded {self.data_files[self.file_index]} "
                f"({self.current_data.shape[1]} samples)"
            )
            return True
            
        except Exception as e:
            self.status_callback(
                f"Error loading {self.data_files[self.file_index]}: {str(e)}"
            )
            return False
            
    def run_benchmark(self):
        """Run comprehensive performance benchmark without CNN for small windows"""
        try:
            self.status_callback("Starting benchmark...")
            
            # Test different window sizes
            window_sizes = [1, 2, 3]  # in seconds
            noise_levels = [0, 20, 50, 80]  # noise percentage
            
            results = []
            for win_size in window_sizes:
                win_len_samples = win_size * FS
                
                for noise in noise_levels:
                    # Test SVM only for this benchmark
                    svm_time, svm_acc = self.test_model(
                        'SVM', win_len_samples, noise/100.0
                    )
                    
                    results.append({
                        'window_size': win_size,
                        'noise_level': noise,
                        'svm_time': svm_time,
                        'svm_acc': svm_acc
                    })
            
            # Create results DataFrame
            df = pd.DataFrame(results)
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_file = os.path.join(BENCHMARK_DIR, f'benchmark_{timestamp}.csv')
            df.to_csv(benchmark_file, index=False)
            
            # Format results for display
            result_text = "=== Benchmark Results (SVM Only) ===\n"
            result_text += f"Generated: {timestamp}\n"
            result_text += f"Tested on window sizes: {window_sizes}s\n"
            result_text += f"Tested noise levels: {noise_levels}%\n\n"
            
            # Add summary statistics
            svm_avg_time = df['svm_time'].mean()
            svm_avg_acc = df['svm_acc'].mean()
            
            result_text += f"SVM Average: {svm_avg_time:.2f} ms latency, {svm_avg_acc:.2%} accuracy\n\n"
            
            # Add detailed results
            result_text += "Detailed Results:\n"
            result_text += df.to_string()
            
            self.update_ui_callback('benchmark_results', result_text)
            self.status_callback(f"Benchmark complete! Results saved to {benchmark_file}")
            
            return True
            
        except Exception as e:
            self.status_callback(f"Benchmark failed: {str(e)}")
            return False
    
    def test_model(self, model_type, win_len, noise_level):
        """Test a single model configuration"""
        # Load sample data
        sample_file = os.path.join(DATA_DIR, "eeg_high_1.csv")
        df = pd.read_csv(sample_file)
        data = df.values.T[:, :win_len]  # First window
        
        # Add noise
        noise = noise_level * np.random.normal(size=data.shape)
        noisy_data = data + noise
        
        # Preprocess
        processed = np.array([preprocess(ch) for ch in noisy_data])
        
        # Time prediction
        start_time = time.perf_counter()
        
        if model_type == 'CNN':
            # Skip CNN for benchmark to avoid errors
            return 0, 0
        else:  # SVM
            # Extract features
            features = []
            for ch in range(CHANNELS):
                freqs, psd = welch(processed[ch], fs=FS, nperseg=win_len)
                for band in [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]:
                    band_mask = (freqs >= band[0]) & (freqs <= band[1])
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    features.append(band_power)
            
            # Predict
            proba = self.svm_model.predict_proba([features])[0]
            confidence = np.max(proba)
            class_idx = np.argmax(proba)
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Since we're using high load data, accuracy is 1 if classified as high
        accuracy = 1.0 if class_idx == 1 else 0.0
        
        return latency, accuracy
        
    def show_session_metrics(self):
        """Show metrics from current session with robust error handling"""
        if not self.log:
            return False
            
        try:
            # Create new figure for metrics
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Session Performance Metrics', fontsize=14)
            
            # Prepare data
            df = pd.DataFrame(
                self.log,
                columns=['Timestamp', 'File', 'Model', 'Prediction', 'Confidence', 'Latency']
            )
            
            # Prediction distribution
            prediction_counts = df['Prediction'].value_counts()
            axs[0, 0].bar(prediction_counts.index, prediction_counts.values, color=['blue', 'orange'])
            axs[0, 0].set_title('Prediction Distribution', fontsize=10)
            axs[0, 0].set_xlabel('Class', fontsize=9)
            axs[0, 0].set_ylabel('Count', fontsize=9)
            axs[0, 0].grid(True, linestyle='--', alpha=0.3)
            
            # Confidence over time
            axs[0, 1].plot(df['Confidence'], 'b-')
            axs[0, 1].set_title('Confidence Over Time', fontsize=10)
            axs[0, 1].set_xlabel('Window', fontsize=9)
            axs[0, 1].set_ylabel('Confidence', fontsize=9)
            axs[0, 1].set_ylim(0, 1.05)
            axs[0, 1].grid(True, linestyle='--', alpha=0.3)
            
            # Latency distribution
            axs[1, 0].hist(df['Latency'], bins=20, color='green', alpha=0.7)
            axs[1, 0].set_title('Latency Distribution', fontsize=10)
            axs[1, 0].set_xlabel('Latency (ms)', fontsize=9)
            axs[1, 0].set_ylabel('Frequency', fontsize=9)
            axs[1, 0].grid(True, linestyle='--', alpha=0.3)
            
            # Model comparison (if multiple models used)
            if 'CNN' in df['Model'].values and 'SVM' in df['Model'].values:
                model_stats = df.groupby('Model')['Confidence'].agg(['mean', 'std'])
                model_stats.plot(kind='bar', y='mean', yerr='std', 
                               ax=axs[1, 1], capsize=4, alpha=0.7, color=['blue', 'orange'])
                axs[1, 1].set_title('Model Performance', fontsize=10)
                axs[1, 1].set_ylabel('Average Confidence', fontsize=9)
                axs[1, 1].set_xticklabels(model_stats.index, rotation=0, fontsize=9)
            else:
                model = df['Model'].iloc[0] if len(df) > 0 else 'Unknown'
                axs[1, 1].text(0.5, 0.5, f'Single model used: {model}', 
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[1, 1].transAxes, fontsize=11)
                axs[1, 1].set_title('Model Comparison', fontsize=10)
                axs[1, 1].axis('off')
            
            fig.tight_layout(pad=3.0)
            
            # Create new window for metrics
            metrics_window = tk.Toplevel()
            metrics_window.title("Session Metrics")
            metrics_window.geometry("800x600")
            
            # Embed figure
            canvas = FigureCanvasTkAgg(fig, metrics_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, metrics_window)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Close button
            close_btn = ttk.Button(metrics_window, text="Close", command=metrics_window.destroy)
            close_btn.pack(side=tk.BOTTOM, pady=10)
            
            return True
            
        except Exception as e:
            self.status_callback(f"Error showing metrics: {str(e)}")
            return False

# === Tkinter GUI ===
class EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Classifier - Phase 2/3")
        self.root.geometry("1200x800")
        
        # Create matplotlib figures
        self.fig_ts = plt.figure(figsize=(10, 6))
        self.ax_ts = self.fig_ts.subplots(4, 2, sharex=True)
        self.fig_ts.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
        self.fig_ts.suptitle('EEG Time Series', fontsize=12)
        
        self.fig_psd = plt.figure(figsize=(6, 4))
        self.ax_psd = self.fig_psd.subplots(1, 1)
        self.ax_psd.set_title('Power Spectral Density', fontsize=10)
        self.ax_psd.set_xlabel('Frequency (Hz)', fontsize=9)
        self.ax_psd.set_ylabel('Power', fontsize=9)
        self.fig_psd.tight_layout()
        
        # Initialize variables
        self.data_source = tk.StringVar(value="simulation")
        self.model_type = tk.StringVar(value="CNN")
        self.win_size = tk.StringVar(value="2")
        self.selected_file = tk.StringVar(value="All Files (Loop)")
        
        # Create GUI
        self.create_widgets()
        
        # Initialize processor after UI is fully created
        self.processor = EEGProcessor(self.update_status, self.update_ui)
        
        # Populate file list
        self.update_file_list()
        
    def create_widgets(self):
        # Main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.S), padx=(0, 10))
        
        # Right panel - display
        display_frame = ttk.LabelFrame(main_frame, text="Display", padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.N, tk.E, tk.S, tk.W))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Control widgets
        ttk.Label(control_frame, text="Adaptive EEG Classifier", font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(control_frame, text="Data Source:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(control_frame, text="Simulation", variable=self.data_source, value="simulation").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(control_frame, text="Live EEG (LSL)", variable=self.data_source, value="live").grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Model Selection:").grid(row=3, column=0, sticky=tk.W, pady=(10, 2))
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_type, values=["CNN", "SVM"], state="readonly", width=15)
        model_combo.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Noise Level:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.noise_slider = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.noise_slider.set(20)
        self.noise_slider.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.noise_value = ttk.Label(control_frame, text="20")
        self.noise_value.grid(row=7, column=2, sticky=tk.W, pady=2)
        self.noise_slider.configure(command=lambda v: self.noise_value.config(text=f"{float(v):.0f}"))
        
        ttk.Label(control_frame, text="Speed (Hz):").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.speed_slider = ttk.Scale(control_frame, from_=1, to=30, orient=tk.HORIZONTAL)
        self.speed_slider.set(10)
        self.speed_slider.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.speed_value = ttk.Label(control_frame, text="10")
        self.speed_value.grid(row=9, column=2, sticky=tk.W, pady=2)
        self.speed_slider.configure(command=lambda v: self.speed_value.config(text=f"{float(v):.0f}"))
        
        ttk.Label(control_frame, text="Confidence Threshold (%):").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.threshold_slider = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.threshold_slider.set(70)
        self.threshold_slider.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.threshold_value = ttk.Label(control_frame, text="70")
        self.threshold_value.grid(row=11, column=2, sticky=tk.W, pady=2)
        self.threshold_slider.configure(command=lambda v: self.threshold_value.config(text=f"{float(v):.0f}"))
        
        ttk.Label(control_frame, text="Window Size (s):").grid(row=12, column=0, sticky=tk.W, pady=2)
        win_size_combo = ttk.Combobox(control_frame, textvariable=self.win_size, values=["1", "2", "3"], state="readonly", width=15)
        win_size_combo.grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Select File:").grid(row=14, column=0, sticky=tk.W, pady=2)
        self.file_combo = ttk.Combobox(control_frame, textvariable=self.selected_file, state="readonly", width=25)
        self.file_combo.grid(row=15, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=16, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_btn.grid(row=17, column=0, pady=5, sticky=tk.W)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=17, column=1, pady=5, sticky=tk.W)
        
        self.train_btn = ttk.Button(control_frame, text="Train Models", command=self.train_models)
        self.train_btn.grid(row=18, column=0, pady=5, sticky=tk.W)
        
        self.benchmark_btn = ttk.Button(control_frame, text="Run Benchmark", command=self.run_benchmark)
        self.benchmark_btn.grid(row=18, column=1, pady=5, sticky=tk.W)
        
        self.exit_btn = ttk.Button(control_frame, text="Exit", command=self.root.quit)
        self.exit_btn.grid(row=19, column=0, pady=5, sticky=tk.W)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=20, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Status area
        ttk.Label(control_frame, text="Status:", font=('Arial', 10, 'bold')).grid(row=21, column=0, sticky=tk.W, pady=2)
        
        self.status_text = scrolledtext.ScrolledText(control_frame, width=30, height=5, state=tk.DISABLED)
        self.status_text.grid(row=22, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Display widgets - Time series plot
        ts_frame = ttk.Frame(display_frame)
        ts_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.ts_canvas = FigureCanvasTkAgg(self.fig_ts, ts_frame)
        self.ts_canvas.draw()
        self.ts_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Display widgets - PSD plot
        psd_frame = ttk.Frame(display_frame)
        psd_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.psd_canvas = FigureCanvasTkAgg(self.fig_psd, psd_frame)
        self.psd_canvas.draw()
        self.psd_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Display widgets - Prediction info
        info_frame = ttk.Frame(display_frame)
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(info_frame, text="Prediction:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.prediction_label = ttk.Label(info_frame, text="N/A", font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        self.confidence_bar = ttk.Progressbar(info_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.confidence_bar.grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(info_frame, text="Latency:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.latency_label = ttk.Label(info_frame, text="0.0 ms")
        self.latency_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text="Active Model:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.model_label = ttk.Label(info_frame, text="None")
        self.model_label.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text="Data Source:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.source_label = ttk.Label(info_frame, text="Simulation")
        self.source_label.grid(row=3, column=1, sticky=tk.W)
        
        # Action buttons
        action_frame = ttk.Frame(display_frame)
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.save_btn = ttk.Button(action_frame, text="Save Log", command=self.save_log)
        self.save_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.metrics_btn = ttk.Button(action_frame, text="Show Metrics", command=self.show_metrics)
        self.metrics_btn.grid(row=0, column=1)
        
        # Metrics section (initially hidden)
        self.metrics_frame = ttk.LabelFrame(main_frame, text="Metrics", padding="10")
        self.metrics_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.S), pady=(10, 0))
        self.metrics_frame.grid_remove()  # Hide initially
        
        # Benchmark results
        ttk.Label(self.metrics_frame, text="Benchmark Results:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.benchmark_text = scrolledtext.ScrolledText(self.metrics_frame, width=70, height=10, state=tk.DISABLED)
        self.benchmark_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for metrics frame
        self.metrics_frame.columnconfigure(0, weight=1)
        self.metrics_frame.rowconfigure(1, weight=1)
    
    def update_status(self, message):
        """Update status text area"""
        # Check if status_text exists before trying to update it
        if hasattr(self, 'status_text'):
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
    
    def update_ui(self, element, value):
        """Update UI elements from processor thread"""
        if element == 'prediction':
            self.prediction_label.config(text=value)
        elif element == 'prediction_color':
            self.prediction_label.config(foreground=value)
        elif element == 'confidence':
            self.confidence_bar.config(value=value)
        elif element == 'latency':
            self.latency_label.config(text=value)
        elif element == 'model_active':
            self.model_label.config(text=value)
        elif element == 'source_active':
            self.source_label.config(text=value)
        elif element == 'benchmark_results':
            if hasattr(self, 'benchmark_text'):
                self.benchmark_text.config(state=tk.NORMAL)
                self.benchmark_text.delete(1.0, tk.END)
                self.benchmark_text.insert(tk.END, value)
                self.benchmark_text.config(state=tk.DISABLED)
    
    def update_file_list(self):
        """Update the file list dropdown"""
        files = ['All Files (Loop)'] + sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
        self.file_combo.config(values=files)
        if files:
            self.selected_file.set(files[0])
    
    def start_processing(self):
        """Start processing button handler"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED)
        self.benchmark_btn.config(state=tk.DISABLED)
        
        # Capture current parameters
        params = {
            'source': self.data_source.get(),
            'model_type': self.model_type.get(),
            'noise_level': self.noise_slider.get(),
            'speed': self.speed_slider.get(),
            'threshold': self.threshold_slider.get(),
            'win_size': self.win_size.get(),
            'selected_file': self.selected_file.get()
        }
        
        if not self.processor.start(self.ax_ts, self.ax_psd, params):
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.NORMAL)
            self.benchmark_btn.config(state=tk.NORMAL)
    
    def stop_processing(self):
        """Stop processing button handler"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.NORMAL)
        self.benchmark_btn.config(state=tk.NORMAL)
        self.processor.stop()
        self.update_status("Processing stopped")
    
    def train_models(self):
        """Train models button handler"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.benchmark_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
        # Train models in a separate thread to avoid blocking UI
        def train_thread():
            cnn_acc, svm_acc, status = train_models(self.update_status)
            self.update_status(status)
            
            # Reload models and update file list
            self.processor.load_models()
            self.processor.load_data()
            self.update_file_list()
            
            # Re-enable buttons
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.NORMAL)
            self.benchmark_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def run_benchmark(self):
        """Run benchmark button handler"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.benchmark_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
        # Show metrics frame
        self.metrics_frame.grid()
        
        # Run benchmark in a separate thread
        def benchmark_thread():
            self.processor.run_benchmark()
            
            # Re-enable buttons
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.NORMAL)
            self.benchmark_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=benchmark_thread, daemon=True).start()
    
    def save_log(self):
        """Save log button handler"""
        if not self.processor.log:
            messagebox.showerror("Error", "No data to save!")
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(LOG_DIR, f'eeg_log_{timestamp}.csv')
        
        df = pd.DataFrame(
            self.processor.log,
            columns=['Timestamp', 'File', 'Model', 'Prediction', 'Confidence', 'Latency (ms)']
        )
        df.to_csv(log_file, index=False)
        self.update_status(f"Log saved as {log_file}")
        messagebox.showinfo("Success", f'Log saved as:\n{log_file}')
    
    def show_metrics(self):
        """Show metrics button handler"""
        if not self.processor.log:
            messagebox.showerror("Error", "No session data to show!")
            return
        
        self.processor.show_session_metrics()

# === Main Application ===
def main():
    try:
        root = tk.Tk()
        app = EEGApp(root)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Critical Error", f"Critical error: {str(e)}\n\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()