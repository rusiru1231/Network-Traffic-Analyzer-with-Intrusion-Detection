"""
Data preprocessing utilities for CICIDS 2017 and NSL-KDD datasets.
Handle data loading, cleaning, and feature engineering for ML models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle data preprocessing for intrusion detection datasets."""
    
    def __init__(self, dataset_name: str = 'cicids2017'):
        """
        Initialize data preprocessor.
        
        Args:
            dataset_name: Name of dataset ('cicids2017' or 'nsl_kdd')
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = None
        self.preprocessing_stats = {}
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'cicids2017': {
                'label_column': 'Label',
                'normal_class': 'BENIGN',
                'time_columns': [],
                'categorical_columns': [],
                'drop_columns': ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
            },
            'nsl_kdd': {
                'label_column': 'class',
                'normal_class': 'normal',
                'time_columns': [],
                'categorical_columns': ['protocol_type', 'service', 'flag'],
                'drop_columns': []
            }
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load dataset from file or directory.
        
        Args:
            data_path: Path to dataset file or directory
            
        Returns:
            Combined dataset as DataFrame
        """
        try:
            data_path = Path(data_path)
            
            if data_path.is_file():
                self.logger.info(f"Loading dataset from file: {data_path}")
                df = pd.read_csv(data_path)
            elif data_path.is_dir():
                self.logger.info(f"Loading dataset from directory: {data_path}")
                df = self._load_multiple_files(data_path)
            else:
                raise FileNotFoundError(f"Data path not found: {data_path}")
            
            self.logger.info(f"Dataset loaded. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _load_multiple_files(self, data_dir: Path) -> pd.DataFrame:
        """Load and combine multiple CSV files."""
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        dataframes = []
        for file_path in csv_files:
            self.logger.info(f"Loading {file_path.name}")
            df = pd.read_csv(file_path)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(csv_files)} files into dataset with shape: {combined_df.shape}")
        
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame, 
                       test_size: float = 0.2,
                       validation_size: float = 0.1,
                       feature_selection: bool = True,
                       n_features: int = 20) -> Dict[str, Any]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            df: Raw dataset DataFrame
            test_size: Test set proportion
            validation_size: Validation set proportion
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Dictionary containing processed datasets and metadata
        """
        try:
            self.logger.info("Starting data preprocessing pipeline...")
            
            # 1. Initial data cleaning
            df_clean = self._clean_data(df)
            
            # 2. Handle categorical variables
            df_encoded = self._encode_categorical_features(df_clean)
            
            # 3. Separate features and labels
            X, y = self._separate_features_labels(df_encoded)
            
            # 4. Feature engineering
            X_engineered = self._engineer_features(X)
            
            # 5. Handle class imbalance information
            class_distribution = self._analyze_class_distribution(y)
            
            # 6. Split data
            datasets = self._split_data(X_engineered, y, test_size, validation_size)
            
            # 7. Feature scaling
            datasets = self._scale_features(datasets)
            
            # 8. Feature selection
            if feature_selection:
                datasets = self._select_features(datasets, n_features)
            
            # Store preprocessing statistics
            self.preprocessing_stats = {
                'original_shape': df.shape,
                'processed_shape': X_engineered.shape,
                'class_distribution': class_distribution,
                'feature_names': self.feature_names,
                'n_selected_features': n_features if feature_selection else X_engineered.shape[1]
            }
            
            # Add metadata to results
            datasets['metadata'] = self.preprocessing_stats
            datasets['preprocessor'] = self
            
            self.logger.info("Data preprocessing completed successfully!")
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        self.logger.info("Cleaning data...")
        
        # Remove duplicates
        original_size = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {original_size - len(df)} duplicate rows")
        
        # Handle missing values
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            self.logger.info(f"Found missing values: {missing_info[missing_info > 0].to_dict()}")
            # Fill missing values with median for numeric, mode for categorical
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
        
        # Handle infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Remove constant columns (zero variance)
        constant_columns = [col for col in numeric_columns if df[col].nunique() <= 1]
        if constant_columns:
            self.logger.info(f"Removing constant columns: {constant_columns}")
            df = df.drop(columns=constant_columns)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        config = self.dataset_configs[self.dataset_name]
        categorical_columns = config.get('categorical_columns', [])
        
        if not categorical_columns:
            return df
        
        self.logger.info(f"Encoding categorical features: {categorical_columns}")
        
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                # Use label encoding for categorical features
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
        
        return df_encoded
    
    def _separate_features_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and labels."""
        config = self.dataset_configs[self.dataset_name]
        label_column = config['label_column']
        drop_columns = config.get('drop_columns', [])
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Extract labels
        y = df[label_column].copy()
        
        # Extract features
        feature_columns = [col for col in df.columns 
                          if col != label_column and col not in drop_columns]
        X = df[feature_columns].copy()
        
        # Convert labels to binary (normal=0, attack=1)
        normal_class = config['normal_class']
        y_binary = (y != normal_class).astype(int)
        
        self.logger.info(f"Features shape: {X.shape}, Labels shape: {y_binary.shape}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        return X, y_binary
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        self.logger.info("Engineering features...")
        
        X_engineered = X.copy()
        
        # Convert to numeric only
        numeric_columns = X_engineered.select_dtypes(include=[np.number]).columns
        X_engineered = X_engineered[numeric_columns]
        
        # Add feature interactions (only for important features to avoid explosion)
        important_features = X_engineered.columns[:10]  # Top 10 features
        
        for i, feat1 in enumerate(important_features):
            for feat2 in important_features[i+1:i+3]:  # Limit interactions
                if feat1 != feat2:
                    interaction_name = f"{feat1}_x_{feat2}"
                    X_engineered[interaction_name] = X_engineered[feat1] * X_engineered[feat2]
        
        # Add statistical features
        X_engineered['feature_sum'] = X_engineered.sum(axis=1)
        X_engineered['feature_mean'] = X_engineered.mean(axis=1)
        X_engineered['feature_std'] = X_engineered.std(axis=1)
        
        self.logger.info(f"Feature engineering completed. New shape: {X_engineered.shape}")
        
        return X_engineered
    
    def _analyze_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """Analyze class distribution."""
        class_counts = y.value_counts()
        class_proportions = y.value_counts(normalize=True)
        
        distribution = {
            'counts': class_counts.to_dict(),
            'proportions': class_proportions.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min()
        }
        
        self.logger.info(f"Class distribution: {distribution}")
        
        return distribution
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float, validation_size: float) -> Dict[str, Any]:
        """Split data into train, validation, and test sets."""
        self.logger.info("Splitting data...")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        datasets = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Log split information
        for split_name, data in datasets.items():
            self.logger.info(f"{split_name} shape: {data.shape}")
        
        return datasets
    
    def _scale_features(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Scale features using StandardScaler."""
        self.logger.info("Scaling features...")
        
        # Fit scaler on training data
        self.scaler.fit(datasets['X_train'])
        
        # Transform all datasets
        datasets['X_train'] = pd.DataFrame(
            self.scaler.transform(datasets['X_train']),
            columns=datasets['X_train'].columns,
            index=datasets['X_train'].index
        )
        
        datasets['X_val'] = pd.DataFrame(
            self.scaler.transform(datasets['X_val']),
            columns=datasets['X_val'].columns,
            index=datasets['X_val'].index
        )
        
        datasets['X_test'] = pd.DataFrame(
            self.scaler.transform(datasets['X_test']),
            columns=datasets['X_test'].columns,
            index=datasets['X_test'].index
        )
        
        return datasets
    
    def _select_features(self, datasets: Dict[str, Any], n_features: int) -> Dict[str, Any]:
        """Select best features using statistical tests."""
        self.logger.info(f"Selecting top {n_features} features...")
        
        # Use mutual information for feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        
        # Fit on training data
        self.feature_selector.fit(datasets['X_train'], datasets['y_train'])
        
        # Transform all datasets
        datasets['X_train'] = pd.DataFrame(
            self.feature_selector.transform(datasets['X_train']),
            index=datasets['X_train'].index
        )
        
        datasets['X_val'] = pd.DataFrame(
            self.feature_selector.transform(datasets['X_val']),
            index=datasets['X_val'].index
        )
        
        datasets['X_test'] = pd.DataFrame(
            self.feature_selector.transform(datasets['X_test']),
            index=datasets['X_test'].index
        )
        
        # Update feature names
        selected_features = datasets['X_train'].columns
        self.feature_names = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        
        self.logger.info(f"Selected features: {self.feature_names}")
        
        return datasets
    
    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        try:
            # Ensure same columns as training data
            if self.feature_names:
                # Keep only features that were in training
                available_features = [feat for feat in self.feature_names if feat in X.columns]
                missing_features = [feat for feat in self.feature_names if feat not in X.columns]
                
                if missing_features:
                    self.logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with zeros
                    for feat in missing_features:
                        X[feat] = 0
                
                X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Select features if feature selector was used
            if self.feature_selector:
                X_scaled = self.feature_selector.transform(X_scaled)
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming new data: {e}")
            raise
    
    def save_preprocessor(self, save_path: str):
        """Save preprocessor objects."""
        try:
            preprocessor_data = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'dataset_name': self.dataset_name,
                'preprocessing_stats': self.preprocessing_stats
            }
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(preprocessor_data, save_path)
            self.logger.info(f"Preprocessor saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {e}")
            raise
    
    def load_preprocessor(self, load_path: str):
        """Load preprocessor objects."""
        try:
            preprocessor_data = joblib.load(load_path)
            
            self.scaler = preprocessor_data['scaler']
            self.label_encoder = preprocessor_data['label_encoder']
            self.feature_selector = preprocessor_data['feature_selector']
            self.feature_names = preprocessor_data['feature_names']
            self.dataset_name = preprocessor_data['dataset_name']
            self.preprocessing_stats = preprocessor_data['preprocessing_stats']
            
            self.logger.info(f"Preprocessor loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {e}")
            raise


def create_sample_data(output_path: str, n_samples: int = 10000):
    """Create sample dataset for testing purposes."""
    np.random.seed(42)
    
    # Generate synthetic network traffic features
    data = {
        'Flow Duration': np.random.exponential(1000, n_samples),
        'Total Fwd Packets': np.random.poisson(10, n_samples),
        'Total Backward Packets': np.random.poisson(8, n_samples),
        'Total Length of Fwd Packets': np.random.exponential(500, n_samples),
        'Total Length of Bwd Packets': np.random.exponential(400, n_samples),
        'Fwd Packet Length Max': np.random.gamma(2, 100, n_samples),
        'Fwd Packet Length Min': np.random.gamma(1, 50, n_samples),
        'Fwd Packet Length Mean': np.random.normal(100, 30, n_samples),
        'Fwd Packet Length Std': np.random.gamma(1, 20, n_samples),
        'Bwd Packet Length Max': np.random.gamma(2, 80, n_samples),
        'Bwd Packet Length Min': np.random.gamma(1, 40, n_samples),
        'Bwd Packet Length Mean': np.random.normal(80, 25, n_samples),
        'Bwd Packet Length Std': np.random.gamma(1, 15, n_samples),
        'Flow Bytes/s': np.random.exponential(1000, n_samples),
        'Flow Packets/s': np.random.exponential(10, n_samples),
        'Flow IAT Mean': np.random.exponential(100, n_samples),
        'Flow IAT Std': np.random.exponential(200, n_samples),
        'Flow IAT Max': np.random.exponential(1000, n_samples),
        'Flow IAT Min': np.random.exponential(10, n_samples),
        'Fwd IAT Total': np.random.exponential(500, n_samples),
        'Fwd IAT Mean': np.random.exponential(50, n_samples),
        'Fwd IAT Std': np.random.exponential(100, n_samples),
        'Fwd IAT Max': np.random.exponential(500, n_samples),
        'Fwd IAT Min': np.random.exponential(5, n_samples),
        'Bwd IAT Total': np.random.exponential(400, n_samples),
        'Bwd IAT Mean': np.random.exponential(40, n_samples),
        'Bwd IAT Std': np.random.exponential(80, n_samples),
        'Bwd IAT Max': np.random.exponential(400, n_samples),
        'Bwd IAT Min': np.random.exponential(4, n_samples)
    }
    
    # Create labels (90% benign, 10% malicious)
    labels = np.random.choice(['BENIGN', 'ATTACK'], n_samples, p=[0.9, 0.1])
    data['Label'] = labels
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution: {df['Label'].value_counts()}")


if __name__ == "__main__":
    # Create sample data for testing
    output_path = "data/sample_cicids2017.csv"
    create_sample_data(output_path)
    
    # Test preprocessing
    preprocessor = DataPreprocessor('cicids2017')
    df = preprocessor.load_data(output_path)
    datasets = preprocessor.preprocess_data(df)
    
    print("Preprocessing completed successfully!")
    print(f"Training set shape: {datasets['X_train'].shape}")
    print(f"Test set shape: {datasets['X_test'].shape}")