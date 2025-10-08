#!/usr/bin/env python3
"""
ML-Powered Network Intrusion Detection System
Main application entry point for real-time packet analysis and threat detection.
"""

import sys
import logging
import argparse
from pathlib import Path
import threading
import signal
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.packet_capture import PacketCapture
from src.models.intrusion_detector import IntrusionDetector
from src.utils.logger import setup_logging
from src.utils.config_manager import ConfigManager

class NetworkIntrusionDetectionSystem:
    """Main class for the Network Intrusion Detection System."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the NIDS system."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.packet_capture = None
        self.intrusion_detector = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def initialize_components(self):
        """Initialize packet capture and intrusion detection components."""
        try:
            self.logger.info("Initializing Network Intrusion Detection System...")
            
            # Initialize intrusion detector
            self.intrusion_detector = IntrusionDetector(
                model_path=self.config.get('model_path', 'models/intrusion_model.joblib')
            )
            
            # Initialize packet capture
            network_config = self.config.get('network', {})
            self.packet_capture = PacketCapture(
                interface=network_config.get('interface', 'auto'),
                filter_expression=network_config.get('filter', None)
            )
            
            self.logger.info("All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_monitoring(self):
        """Start real-time network monitoring and threat detection."""
        try:
            self.logger.info("Starting network monitoring...")
            self.running = True
            
            # Start packet capture in a separate thread
            capture_thread = threading.Thread(
                target=self._packet_capture_worker,
                daemon=True
            )
            capture_thread.start()
            
            # Main monitoring loop
            self._monitoring_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
        finally:
            self.stop_monitoring()
    
    def _packet_capture_worker(self):
        """Worker thread for packet capture."""
        try:
            self.packet_capture.start_capture(
                packet_callback=self._process_packet
            )
        except Exception as e:
            self.logger.error(f"Packet capture error: {e}")
    
    def _process_packet(self, packet_data):
        """Process captured packet and detect intrusions."""
        try:
            # Extract features from packet
            features = self.packet_capture.extract_features(packet_data)
            
            if features is not None:
                # Detect intrusion
                prediction, confidence = self.intrusion_detector.predict(features)
                
                if prediction == 1:  # Intrusion detected
                    self._handle_intrusion_alert(packet_data, confidence)
                    
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    def _handle_intrusion_alert(self, packet_data, confidence):
        """Handle intrusion detection alert."""
        alert_msg = f"INTRUSION DETECTED! Confidence: {confidence:.2%}"
        self.logger.warning(alert_msg)
        
        # Log packet details
        packet_info = self.packet_capture.get_packet_info(packet_data)
        self.logger.warning(f"Packet details: {packet_info}")
        
        # TODO: Implement alert notifications (email, SMS, etc.)
        # TODO: Save to database for dashboard visualization
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Network monitoring active. Press Ctrl+C to stop.")
        
        while self.running:
            try:
                # Display periodic statistics
                stats = self.packet_capture.get_statistics()
                self.logger.info(f"Packets captured: {stats.get('total_packets', 0)}")
                
                time.sleep(10)  # Update every 10 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def stop_monitoring(self):
        """Stop network monitoring."""
        self.logger.info("Stopping network monitoring...")
        self.running = False
        
        if self.packet_capture:
            self.packet_capture.stop_capture()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML-Powered Network Intrusion Detection System"
    )
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--interface",
        help="Network interface to monitor (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize NIDS
        nids = NetworkIntrusionDetectionSystem(config_path=args.config)
        
        # Override interface if specified
        if args.interface:
            nids.config['network']['interface'] = args.interface
        
        # Initialize and start monitoring
        nids.initialize_components()
        nids.start_monitoring()
        
    except Exception as e:
        logging.error(f"Failed to start NIDS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()