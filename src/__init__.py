"""
Package initialization for the intrusion detection system.
"""

__version__ = "1.0.0"
__author__ = "Network Security Team"
__description__ = "ML-powered Network Intrusion Detection System"

# Import main components
from .models.intrusion_detector import IntrusionDetector
from .data.preprocessing import DataPreprocessor
from .utils.config_manager import ConfigManager
from .utils.packet_capture import PacketCapture

__all__ = [
    'IntrusionDetector',
    'DataPreprocessor', 
    'ConfigManager',
    'PacketCapture'
]