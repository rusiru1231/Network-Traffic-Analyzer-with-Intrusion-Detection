# ML-Powered Network Intrusion Detection System

A real-time network traffic analyzer with machine learning-powered intrusion detection capabilities, achieving 92% precision using the CICIDS 2017 dataset.

## 🚀 Features

- **Real-time Packet Analysis**: Live network traffic monitoring using Scapy
- **ML-Powered Detection**: Advanced machine learning models for anomaly detection
- **Interactive Dashboard**: Streamlit-based visualization and monitoring interface
- **High Precision**: Achieves 92% precision using CICIDS 2017 dataset
- **Multiple Datasets**: Support for both NSL-KDD and CICIDS 2017 datasets
- **Scalable Architecture**: Modular design for easy extension and deployment

## 🛠️ Tech Stack

- **Backend**: Python 3.8+
- **ML Frameworks**: TensorFlow, PyTorch, Scikit-learn
- **Network Analysis**: Scapy, PyShark
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Additional Tools**: Wireshark (for packet analysis reference)

## 📊 Datasets

- **CICIDS 2017**: Primary dataset for training and evaluation
- **NSL-KDD**: Alternative dataset for comparison and validation

## 🏗️ Project Structure

```
├── src/
│   ├── data/              # Data preprocessing and loading
│   ├── models/            # ML model implementations
│   ├── utils/             # Utility functions
│   └── dashboard/         # Streamlit dashboard components
├── data/                  # Dataset storage
├── models/                # Trained model artifacts
├── config/                # Configuration files
├── logs/                  # Application logs
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Administrative privileges (for packet capture)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rusiru1231/Network-Traffic-Analyzer-with-Intrusion-Detection.git
   cd "Network Traffic Analyzer with Intrusion Detection"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets** (Optional - for training)
   - CICIDS 2017: [Download from UNB](https://www.unb.ca/cic/datasets/ids-2017.html)
   - NSL-KDD: [Download from UNB](https://www.unb.ca/cic/datasets/nsl.html)
   
   Place datasets in the `data/` directory.

### Usage

1. **Train Models** (Optional - pre-trained models included)
   ```bash
   python src/models/train_model.py
   ```

2. **Start Real-time Detection**
   ```bash
   python src/main.py
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run src/dashboard/app.py
   ```

4. **Access the Dashboard**
   Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

Configuration files are located in the `config/` directory:

- `model_config.yaml`: ML model parameters
- `network_config.yaml`: Network interface settings
- `dashboard_config.yaml`: Dashboard customization

## 📈 Performance Metrics

- **Precision**: 92%
- **Recall**: 89%
- **F1-Score**: 90.5%
- **Accuracy**: 94%

## 🔍 Features Overview

### Real-time Monitoring
- Live packet capture and analysis
- Real-time threat detection
- Network statistics and metrics

### Machine Learning Models
- Random Forest Classifier
- Deep Neural Networks
- Ensemble Methods
- Feature engineering and selection

### Interactive Dashboard
- Real-time threat visualization
- Network traffic analysis
- Historical data exploration
- Alert management system

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Important Notes

- **Administrator Privileges**: Required for packet capture functionality
- **Network Interface**: Ensure proper network interface configuration
- **Firewall**: May need to configure firewall rules for packet capture
- **Performance**: Real-time analysis requires adequate system resources

## 📞 Support

For questions and support, please open an issue in the repository.

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local laws and regulations when monitoring network traffic.
