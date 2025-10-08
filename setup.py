"""
Setup script for the Network Intrusion Detection System.
Initialize the environment and prepare for first use.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

def setup_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'models',
        'logs',
        'reports',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_sample_data():
    """Create sample dataset for testing."""
    print("Creating sample dataset...")
    
    try:
        sys.path.append('src')
        from data.preprocessing import create_sample_data
        
        data_path = Path('data') / 'sample_cicids2017.csv'
        create_sample_data(str(data_path), n_samples=10000)
        
        print(f"✓ Sample dataset created: {data_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def test_installation():
    """Test the installation by running basic imports."""
    print("Testing installation...")
    
    try:
        # Test basic imports
        import numpy
        import pandas
        import sklearn
        import scapy
        import streamlit
        print("✓ All required packages imported successfully")
        
        # Test custom modules
        sys.path.append('src')
        from models.intrusion_detector import IntrusionDetector
        from data.preprocessing import DataPreprocessor
        from utils.config_manager import ConfigManager
        
        print("✓ Custom modules imported successfully")
        
        # Test basic functionality
        detector = IntrusionDetector()
        config_manager = ConfigManager()
        
        print("✓ Basic functionality test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🛡️  NETWORK INTRUSION DETECTION SYSTEM SETUP COMPLETE")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("\n1. Start Real-time Detection:")
    print("   python main.py")
    
    print("\n2. Launch Dashboard:")
    print("   streamlit run src/dashboard/app.py")
    
    print("\n3. Train Custom Model (optional):")
    print("   python src/models/train_model.py --data-path data/your_dataset.csv")
    
    print("\n4. Run Tests:")
    print("   python -m pytest tests/")
    
    print("\n📊 SAMPLE DATA:")
    print("   - Sample dataset created in: data/sample_cicids2017.csv")
    print("   - Use this for testing and demonstration")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("   - Administrator privileges required for packet capture")
    print("   - Configure network interface in config/model_config.yaml")
    print("   - Check firewall settings if packet capture fails")
    
    print("\n🔗 DASHBOARD ACCESS:")
    print("   - After launching dashboard, open: http://localhost:8501")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   src/          - Source code")
    print("   data/         - Datasets")
    print("   models/       - Trained models")
    print("   config/       - Configuration files")
    print("   logs/         - Application logs")
    print("   reports/      - Training reports")

def main():
    """Main setup function."""
    print("🛡️  Network Intrusion Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please install dependencies manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Please check error messages above.")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()
    
    print(f"\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()