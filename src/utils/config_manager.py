"""
Configuration management utilities.
Handle loading and managing configuration files for the intrusion detection system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Manage configuration files and settings."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        
        # Load user configuration if exists
        if self.config_path.exists():
            self._load_config()
        else:
            # Create default config file
            self._save_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'model': {
                'type': 'ensemble',
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'mlp': {
                    'hidden_layer_sizes': [100, 50],
                    'max_iter': 500,
                    'learning_rate_init': 0.001,
                    'random_state': 42
                }
            },
            'network': {
                'interface': 'auto',
                'filter': None,
                'capture_timeout': 1,
                'buffer_size': 1000
            },
            'detection': {
                'threshold': 0.5,
                'anomaly_threshold': -0.5,
                'enable_anomaly_detection': True,
                'flow_timeout': 60
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'logs/intrusion_detection.log',
                'security_log_file': 'logs/security_events.log',
                'console_output': True,
                'max_file_size': 10485760,  # 10 MB
                'backup_count': 5
            },
            'data': {
                'dataset_path': 'data/cicids2017',
                'model_save_path': 'models/intrusion_model.joblib',
                'scaler_save_path': 'models/scaler.joblib'
            },
            'dashboard': {
                'host': 'localhost',
                'port': 8501,
                'auto_refresh_interval': 5,
                'max_display_packets': 100
            },
            'alerts': {
                'enable_email': False,
                'email_smtp_server': 'smtp.gmail.com',
                'email_smtp_port': 587,
                'email_username': '',
                'email_password': '',
                'email_recipients': [],
                'enable_syslog': False,
                'syslog_server': 'localhost',
                'syslog_port': 514
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    user_config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    user_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            # Merge with default config
            self._deep_merge(self.config, user_config)
            
        except Exception as e:
            logging.error(f"Error loading config from {self.config_path}: {e}")
            logging.info("Using default configuration")
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            # Create config directory
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logging.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logging.error(f"Error saving config to {self.config_path}: {e}")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get specific configuration section."""
        return self.config.get(section, {})
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'model.random_forest.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_value(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config_dict = self.config
        
        # Navigate to parent dict
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
        
        # Set value
        config_dict[keys[-1]] = value
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_merge(self.config, updates)
    
    def save(self):
        """Save current configuration to file."""
        self._save_config()
    
    def reload(self):
        """Reload configuration from file."""
        if self.config_path.exists():
            self._load_config()


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load dataset-specific configuration."""
    dataset_configs = {
        'cicids2017': {
            'name': 'CICIDS 2017',
            'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
            'files': [
                'Monday-WorkingHours.pcap_ISCX.csv',
                'Tuesday-WorkingHours.pcap_ISCX.csv',
                'Wednesday-workingHours.pcap_ISCX.csv',
                'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
            ],
            'label_column': 'Label',
            'normal_class': 'BENIGN',
            'preprocessing': {
                'remove_duplicates': True,
                'handle_missing': True,
                'remove_infinite': True,
                'feature_selection': True
            }
        },
        'nsl_kdd': {
            'name': 'NSL-KDD',
            'url': 'https://www.unb.ca/cic/datasets/nsl.html',
            'files': [
                'KDDTrain+.txt',
                'KDDTest+.txt'
            ],
            'label_column': 'class',
            'normal_class': 'normal',
            'preprocessing': {
                'remove_duplicates': True,
                'handle_missing': True,
                'encode_categorical': True,
                'feature_selection': True
            }
        }
    }
    
    return dataset_configs.get(dataset_name, {})