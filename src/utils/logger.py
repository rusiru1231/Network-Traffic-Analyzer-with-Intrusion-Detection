"""
Logging configuration and utilities.
Centralized logging setup for the intrusion detection system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any
import os


def setup_logging(config: Dict[str, Any] = None):
    """
    Setup logging configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    # Default configuration
    default_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'logs/intrusion_detection.log',
        'max_file_size': 10 * 1024 * 1024,  # 10 MB
        'backup_count': 5,
        'console_output': True
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    # Create logs directory
    log_file_path = Path(default_config['log_file'])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, default_config['level'].upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(default_config['format'])
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        default_config['log_file'],
        maxBytes=default_config['max_file_size'],
        backupCount=default_config['backup_count']
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if default_config['console_output']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('scapy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, log_file: str = "logs/security_events.log"):
        """Initialize security logger."""
        self.logger = logging.getLogger('security')
        
        # Create security log directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup security file handler
        security_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=10
        )
        
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        
        self.logger.addHandler(security_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_intrusion_attempt(self, packet_info: Dict[str, Any], confidence: float):
        """Log intrusion attempt."""
        message = f"INTRUSION DETECTED - Confidence: {confidence:.2%} - {packet_info}"
        self.logger.warning(message)
    
    def log_anomaly(self, packet_info: Dict[str, Any], anomaly_score: float):
        """Log network anomaly."""
        message = f"ANOMALY DETECTED - Score: {anomaly_score:.3f} - {packet_info}"
        self.logger.info(message)
    
    def log_system_event(self, event_type: str, details: str):
        """Log system events."""
        message = f"{event_type} - {details}"
        self.logger.info(message)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with specified name."""
    return logging.getLogger(name)