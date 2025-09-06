# 🧠 SpectroCognix: Adaptive Real-Time EEG Classifier with Dual-Mode Simulation and Live Streaming
### A High-Performance Framework for Cognitive Load Estimation Using 1D-CNN and SVM Classification  

This repository contains **SpectroCognix**, a comprehensive framework for real-time EEG-based cognitive load classification.  
The system achieves **≥99.9% accuracy** with minimal latency (**35±8ms**) and supports seamless switching between **CSV simulation** and **LSL live-streaming**.  

Designed for **research prototyping, clinical applications, and educational use**, this project demonstrates **state-of-the-art performance** in real-time EEG processing.  

---

## 🏆 Key Achievements
- **Accuracy**: ≥99.9% on cognitive load classification  
- **Latency**: 35±8ms (CNN), 20±5ms (SVM) in real-time operation  
- **Noise Robustness**: Maintains performance up to σ=0.8 noise levels  
- **Dual-Mode Operation**: Instant switching between simulation and live EEG streaming  
- **Comparative Analysis**: Side-by-side evaluation of 1D-CNN and SVM approaches  

---

## 🛠️ Tech Stack
- Python 3.8+ with full scientific computing stack  
- TensorFlow 2.4+ (Deep Learning)  
- Scikit-learn (SVM implementation)  
- PySimpleGUI (Interactive dashboard interface)  
- LSL (Lab Streaming Layer) for live EEG acquisition  
- Matplotlib & Seaborn (Visualization)  
- Docker (Reproducible environment)  
- NumPy & SciPy (Signal processing and computation)  

---

## 🚀 Features
- 🎛️ **Dual-Mode Operation**: CSV simulation ↔ LSL live-streaming without code changes  
- 🧠 **Dual-Classifier Support**: 1D-CNN and PSD-SVM with real-time performance comparison  
- 📊 **Interactive Dashboard**: Real-time control of window size, step interval, noise levels  
- ⚡ **Low Latency Processing**: Optimized for real-time operation (<40ms)  
- 📈 **Comprehensive Visualization**: Time-series, PSD, confidence metrics, latency monitoring  
- 🔍 **Benchmarking Tools**: Performance evaluation across noise levels and parameters  
- 💾 **Extensive Logging**: Session metrics, classification results, and export capabilities  
- 🐳 **Docker Support**: Fully containerized reproducible environment  

---
<pre>
## 📂 Folder Structure
SpectroCognix/
├── data/ # Synthetic EEG datasets (CSV format)
│ ├── eeg_low_.csv # Low cognitive load samples
│ └── eeg_high_.csv # High cognitive load samples
├── models/ # Pre-trained models
│ ├── project4_cnn_model.keras
│ └── project4_svm.pkl
├── logs/ # Session logs and performance metrics
├── benchmarks/ # Benchmark results and comparisons
├── src/ # Source code
│ ├── NeuroAdaptRT.py # Main application file
│ ├── train_models.py # Model training routines
│ ├── lsl_interface.py # LSL streaming module
│ └── processing_utils.py # Signal processing functions
├── screenshots.pdf # Visualization examples
├── LICENSE
├── README.md # Project documentation
└── Paper.docx # Academic manuscript
</pre>


---

## 🧪 How It Works
1. **Data Acquisition**: Choose between synthetic CSV data or live LSL stream  
2. **Preprocessing**: Automatic bandpass filtering, notch filtering, baseline correction  
3. **Feature Extraction**: PSD-based features for SVM or raw data for 1D-CNN  
4. **Real-Time Classification**: Simultaneous operation of both classifiers  
5. **Visualization & Monitoring**: Live dashboard with performance metrics  
6. **Logging & Export**: Comprehensive session recording and data export  

---

## 📄 Documentation

Academic Paper: Paper.docx – Detailed methodology, results, and discussion

Code Documentation: Inline comments and module descriptions

Tutorial Guides: Step-by-step usage instructions in /docs/

API Reference: Comprehensive function and class documentation

## 🔓 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with appropriate attribution.

##👤 Author

Keerthi Kumar K J
📧 Email: inteegrus.research@gmail.com
