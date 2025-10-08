"""
Unit tests for the intrusion detection system.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.intrusion_detector import IntrusionDetector
from data.preprocessing import DataPreprocessor
from utils.config_manager import ConfigManager


class TestIntrusionDetector(unittest.TestCase):
    """Test intrusion detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = IntrusionDetector()
    
    def test_default_model_initialization(self):
        """Test that default model initializes correctly."""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.scaler)
    
    def test_prediction(self):
        """Test model prediction functionality."""
        # Create test feature vector
        features = np.random.random((1, 20))
        
        prediction, confidence = self.detector.predict(features)
        
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        # Create batch of test features
        features = np.random.random((10, 20))
        
        for i in range(features.shape[0]):
            prediction, confidence = self.detector.predict(features[i:i+1])
            self.assertIn(prediction, [0, 1])
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor('cicids2017')
    
    def test_sample_data_creation(self):
        """Test sample data creation."""
        from data.preprocessing import create_sample_data
        
        output_path = "test_sample.csv"
        create_sample_data(output_path, n_samples=100)
        
        # Verify file was created
        self.assertTrue(Path(output_path).exists())
        
        # Load and verify data
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 100)
        self.assertIn('Label', df.columns)
        
        # Clean up
        Path(output_path).unlink()
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Create test data with issues
        data = {
            'feature1': [1, 2, np.inf, 4, 5],
            'feature2': [1, np.nan, 3, 4, 5],
            'feature3': [1, 1, 1, 1, 1],  # Constant column
            'Label': ['BENIGN', 'ATTACK', 'BENIGN', 'ATTACK', 'BENIGN']
        }
        df = pd.DataFrame(data)
        
        # Clean data
        df_clean = self.preprocessor._clean_data(df)
        
        # Verify cleaning
        self.assertFalse(df_clean.isnull().any().any())
        self.assertFalse(np.isinf(df_clean.select_dtypes(include=[np.number])).any().any())


class TestConfigManager(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.config_manager.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('network', config)
        self.assertIn('logging', config)
    
    def test_value_access(self):
        """Test configuration value access."""
        # Test getting values
        model_type = self.config_manager.get_value('model.type')
        self.assertIsNotNone(model_type)
        
        # Test setting values
        self.config_manager.set_value('test.value', 'test')
        test_value = self.config_manager.get_value('test.value')
        self.assertEqual(test_value, 'test')


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_flow(self):
        """Test complete data flow from preprocessing to prediction."""
        # Create sample data
        from data.preprocessing import create_sample_data
        
        sample_path = "test_integration.csv"
        create_sample_data(sample_path, n_samples=100)
        
        try:
            # Initialize preprocessor
            preprocessor = DataPreprocessor('cicids2017')
            
            # Load and preprocess data
            df = preprocessor.load_data(sample_path)
            datasets = preprocessor.preprocess_data(df, test_size=0.3)
            
            # Initialize detector
            detector = IntrusionDetector()
            
            # Make predictions on test data
            for i in range(min(5, len(datasets['X_test']))):
                features = datasets['X_test'].iloc[i:i+1].values
                prediction, confidence = detector.predict(features)
                
                self.assertIn(prediction, [0, 1])
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
        
        finally:
            # Clean up
            if Path(sample_path).exists():
                Path(sample_path).unlink()


if __name__ == '__main__':
    # Create test results directory
    Path('test_results').mkdir(exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)