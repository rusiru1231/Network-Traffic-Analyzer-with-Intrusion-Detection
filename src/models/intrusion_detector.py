"""
Intrusion Detection Models
ML-powered intrusion detection using various algorithms with CICIDS 2017 dataset.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetector:
    """ML-powered intrusion detection system."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize intrusion detector.
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metadata = {}
        
        # Load pre-trained model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.logger.info("No pre-trained model found. Training new model...")
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize a default model for immediate use."""
        self.logger.info("Initializing default Random Forest model...")
        
        # Create a basic model with default parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Create synthetic training data for initial model
        self._train_default_model()
    
    def _train_default_model(self):
        """Train a basic model with synthetic data for immediate functionality."""
        try:
            # Generate synthetic data based on typical network traffic patterns
            n_samples = 1000
            n_features = 20  # Match the feature vector size from packet capture
            
            # Normal traffic patterns
            normal_data = np.random.normal(0, 1, (int(n_samples * 0.8), n_features))
            normal_labels = np.zeros(int(n_samples * 0.8))
            
            # Anomalous traffic patterns (higher variance, different means)
            anomaly_data = np.random.normal(2, 3, (int(n_samples * 0.2), n_features))
            anomaly_labels = np.ones(int(n_samples * 0.2))
            
            # Combine data
            X = np.vstack([normal_data, anomaly_data])
            y = np.hstack([normal_labels, anomaly_labels])
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            self.logger.info("Default model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training default model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict if traffic is malicious.
        
        Args:
            features: Feature vector from packet analysis
            
        Returns:
            Tuple of (prediction, confidence) where prediction is 0/1 and confidence is 0-1
        """
        try:
            if self.model is None:
                raise ValueError("No model available for prediction")
            
            # Ensure features are in correct shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
            else:
                # For models without probability prediction
                confidence = 0.8 if prediction == 1 else 0.2
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0, 0.0
    
    def train_on_cicids2017(self, data_path: str) -> Dict[str, float]:
        """
        Train model on CICIDS 2017 dataset.
        
        Args:
            data_path: Path to CICIDS 2017 dataset
            
        Returns:
            Training metrics dictionary
        """
        try:
            self.logger.info(f"Loading CICIDS 2017 dataset from {data_path}")
            
            # Load dataset
            df = self._load_cicids2017_data(data_path)
            
            # Preprocess data
            X, y = self._preprocess_cicids2017(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            self.model = self._create_ensemble_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            self.logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
    
    def _load_cicids2017_data(self, data_path: str) -> pd.DataFrame:
        """Load and combine CICIDS 2017 dataset files."""
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Single file
            return pd.read_csv(data_path)
        elif data_path.is_dir():
            # Multiple files in directory
            dfs = []
            for file_path in data_path.glob("*.csv"):
                self.logger.info(f"Loading {file_path.name}")
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            if not dfs:
                raise FileNotFoundError(f"No CSV files found in {data_path}")
            
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def _preprocess_cicids2017(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess CICIDS 2017 dataset."""
        self.logger.info("Preprocessing CICIDS 2017 dataset...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Separate features and labels
        label_column = 'Label'  # Adjust based on actual dataset
        if label_column not in df.columns:
            # Try alternative label column names
            possible_labels = ['Label', 'Attack', 'Class', 'Target']
            for col in possible_labels:
                if col in df.columns:
                    label_column = col
                    break
            else:
                raise ValueError("Label column not found in dataset")
        
        # Extract features and labels
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        # Encode labels (convert to binary: normal=0, attack=1)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert multi-class to binary (normal vs attack)
        y_binary = (y_encoded > 0).astype(int)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Convert to numpy arrays
        X = X.select_dtypes(include=[np.number]).values
        
        self.logger.info(f"Dataset preprocessed. Shape: {X.shape}, Classes: {np.unique(y_binary)}")
        
        return X, y_binary
    
    def _create_ensemble_model(self):
        """Create ensemble model for high accuracy."""
        from sklearn.ensemble import VotingClassifier
        
        # Individual models
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            learning_rate_init=0.001
        )
        
        # Ensemble model
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('mlp', mlp_model)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Log detailed classification report
        self.logger.info("Classification Report:")
        self.logger.info(f"\n{classification_report(y_true, y_pred)}")
        
        return metrics
    
    def save_model(self, model_path: str):
        """Save trained model to file."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'metadata': self.model_metadata
            }
            
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model_data, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        try:
            self.logger.info(f"Loading model from {model_path}")
            
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder')
            self.feature_names = model_data.get('feature_names')
            self.model_metadata = model_data.get('metadata', {})
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'):
                # For ensemble models, get importance from first estimator
                importances = self.model.estimators_[0].feature_importances_
            else:
                return None
            
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            else:
                return {f'feature_{i}': imp for i, imp in enumerate(importances)}
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None


class AnomalyDetector:
    """Unsupervised anomaly detection for unknown attacks."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.logger = logging.getLogger(__name__)
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit anomaly detector on normal traffic."""
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.is_fitted = True
            self.logger.info("Anomaly detector fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting anomaly detector: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Returns:
            Tuple of (predictions, scores) where -1 indicates anomaly
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector not fitted")
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            scores = self.model.score_samples(X_scaled)
            
            return predictions, scores
            
        except Exception as e:
            self.logger.error(f"Error in anomaly prediction: {e}")
            raise