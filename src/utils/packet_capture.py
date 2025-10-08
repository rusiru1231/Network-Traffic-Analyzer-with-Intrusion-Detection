"""
Packet capture and network analysis utilities.
Real-time network packet capture using Scapy with feature extraction for ML models.
"""

import logging
import time
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from scapy.all import sniff, get_if_list, conf
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
import threading
from collections import defaultdict, deque
import psutil


class PacketCapture:
    """Handle real-time packet capture and feature extraction."""
    
    def __init__(self, interface: str = "auto", filter_expression: str = None):
        """
        Initialize packet capture.
        
        Args:
            interface: Network interface to capture on ('auto' for automatic selection)
            filter_expression: BPF filter expression for packet filtering
        """
        self.logger = logging.getLogger(__name__)
        self.interface = self._select_interface(interface)
        self.filter_expression = filter_expression
        
        # Statistics tracking
        self.stats = {
            'total_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'other_packets': 0,
            'start_time': None,
            'intrusions_detected': 0
        }
        
        # Packet storage for feature extraction
        self.packet_buffer = deque(maxlen=1000)
        self.flow_tracker = defaultdict(list)
        
        # Control flags
        self.capturing = False
        self.capture_thread = None
        
        self.logger.info(f"PacketCapture initialized on interface: {self.interface}")
    
    def _select_interface(self, interface: str) -> str:
        """Select appropriate network interface."""
        if interface != "auto":
            return interface
        
        # Get available interfaces
        interfaces = get_if_list()
        
        # Try to find the best interface
        for iface in interfaces:
            try:
                # Skip loopback interfaces
                if 'loopback' in iface.lower() or 'lo' in iface.lower():
                    continue
                
                # Check if interface is active
                stats = psutil.net_if_stats().get(iface)
                if stats and stats.isup:
                    self.logger.info(f"Auto-selected interface: {iface}")
                    return iface
            except:
                continue
        
        # Fallback to first available interface
        if interfaces:
            self.logger.warning(f"Using fallback interface: {interfaces[0]}")
            return interfaces[0]
        
        raise RuntimeError("No suitable network interface found")
    
    def start_capture(self, packet_callback: Callable = None):
        """
        Start packet capture.
        
        Args:
            packet_callback: Callback function to process each packet
        """
        if self.capturing:
            self.logger.warning("Packet capture already in progress")
            return
        
        self.capturing = True
        self.stats['start_time'] = time.time()
        
        self.logger.info(f"Starting packet capture on {self.interface}")
        
        try:
            # Start sniffing packets
            sniff(
                iface=self.interface,
                filter=self.filter_expression,
                prn=lambda pkt: self._packet_handler(pkt, packet_callback),
                stop_filter=lambda x: not self.capturing,
                store=False
            )
        except Exception as e:
            self.logger.error(f"Packet capture error: {e}")
            self.capturing = False
            raise
    
    def stop_capture(self):
        """Stop packet capture."""
        self.logger.info("Stopping packet capture...")
        self.capturing = False
    
    def _packet_handler(self, packet, callback: Callable = None):
        """Handle captured packets."""
        try:
            self.stats['total_packets'] += 1
            
            # Update protocol statistics
            if packet.haslayer(TCP):
                self.stats['tcp_packets'] += 1
            elif packet.haslayer(UDP):
                self.stats['udp_packets'] += 1
            elif packet.haslayer(ICMP):
                self.stats['icmp_packets'] += 1
            else:
                self.stats['other_packets'] += 1
            
            # Store packet for flow tracking
            self.packet_buffer.append(packet)
            
            # Call user callback if provided
            if callback:
                callback(packet)
                
        except Exception as e:
            self.logger.error(f"Error handling packet: {e}")
    
    def extract_features(self, packet) -> Optional[np.ndarray]:
        """
        Extract ML features from packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Feature vector as numpy array or None if extraction fails
        """
        try:
            features = {}
            
            # Basic packet features
            features['packet_length'] = len(packet)
            features['protocol'] = self._get_protocol_number(packet)
            
            # IP layer features
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                features['src_ip'] = self._ip_to_int(ip_layer.src)
                features['dst_ip'] = self._ip_to_int(ip_layer.dst)
                features['ttl'] = ip_layer.ttl
                features['flags'] = ip_layer.flags
                features['frag'] = ip_layer.frag
            else:
                features.update({
                    'src_ip': 0, 'dst_ip': 0, 'ttl': 0, 
                    'flags': 0, 'frag': 0
                })
            
            # Transport layer features
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                features.update({
                    'src_port': tcp_layer.sport,
                    'dst_port': tcp_layer.dport,
                    'tcp_flags': tcp_layer.flags,
                    'window_size': tcp_layer.window,
                    'seq_num': tcp_layer.seq,
                    'ack_num': tcp_layer.ack
                })
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                features.update({
                    'src_port': udp_layer.sport,
                    'dst_port': udp_layer.dport,
                    'tcp_flags': 0,
                    'window_size': 0,
                    'seq_num': 0,
                    'ack_num': 0
                })
            else:
                features.update({
                    'src_port': 0, 'dst_port': 0, 'tcp_flags': 0,
                    'window_size': 0, 'seq_num': 0, 'ack_num': 0
                })
            
            # Flow-based features
            flow_features = self._extract_flow_features(packet)
            features.update(flow_features)
            
            # Convert to feature vector
            feature_vector = self._features_to_vector(features)
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None
    
    def _get_protocol_number(self, packet) -> int:
        """Get protocol number from packet."""
        if packet.haslayer(TCP):
            return 6
        elif packet.haslayer(UDP):
            return 17
        elif packet.haslayer(ICMP):
            return 1
        else:
            return 0
    
    def _ip_to_int(self, ip_str: str) -> int:
        """Convert IP address string to integer."""
        try:
            parts = ip_str.split('.')
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
                   (int(parts[2]) << 8) + int(parts[3])
        except:
            return 0
    
    def _extract_flow_features(self, packet) -> Dict[str, float]:
        """Extract flow-based statistical features."""
        features = {}
        
        try:
            # Create flow key
            flow_key = self._get_flow_key(packet)
            
            if flow_key:
                # Add packet to flow
                self.flow_tracker[flow_key].append({
                    'timestamp': time.time(),
                    'size': len(packet),
                    'packet': packet
                })
                
                # Calculate flow statistics
                flow_packets = self.flow_tracker[flow_key]
                
                # Keep only recent packets (last 60 seconds)
                current_time = time.time()
                flow_packets = [p for p in flow_packets 
                              if current_time - p['timestamp'] < 60]
                self.flow_tracker[flow_key] = flow_packets
                
                if len(flow_packets) > 1:
                    # Flow duration
                    duration = flow_packets[-1]['timestamp'] - flow_packets[0]['timestamp']
                    features['flow_duration'] = duration
                    
                    # Packet rate
                    features['flow_packet_rate'] = len(flow_packets) / max(duration, 0.1)
                    
                    # Byte rate
                    total_bytes = sum(p['size'] for p in flow_packets)
                    features['flow_byte_rate'] = total_bytes / max(duration, 0.1)
                    
                    # Packet size statistics
                    sizes = [p['size'] for p in flow_packets]
                    features['flow_avg_packet_size'] = np.mean(sizes)
                    features['flow_std_packet_size'] = np.std(sizes)
                    features['flow_min_packet_size'] = np.min(sizes)
                    features['flow_max_packet_size'] = np.max(sizes)
                else:
                    # Single packet flow
                    features.update({
                        'flow_duration': 0,
                        'flow_packet_rate': 1,
                        'flow_byte_rate': len(packet),
                        'flow_avg_packet_size': len(packet),
                        'flow_std_packet_size': 0,
                        'flow_min_packet_size': len(packet),
                        'flow_max_packet_size': len(packet)
                    })
            else:
                # No flow information available
                features.update({
                    'flow_duration': 0, 'flow_packet_rate': 0, 'flow_byte_rate': 0,
                    'flow_avg_packet_size': 0, 'flow_std_packet_size': 0,
                    'flow_min_packet_size': 0, 'flow_max_packet_size': 0
                })
                
        except Exception as e:
            self.logger.error(f"Flow feature extraction error: {e}")
            features.update({
                'flow_duration': 0, 'flow_packet_rate': 0, 'flow_byte_rate': 0,
                'flow_avg_packet_size': 0, 'flow_std_packet_size': 0,
                'flow_min_packet_size': 0, 'flow_max_packet_size': 0
            })
        
        return features
    
    def _get_flow_key(self, packet) -> Optional[str]:
        """Generate flow key for packet tracking."""
        try:
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                src_ip, dst_ip = ip_layer.src, ip_layer.dst
                
                if packet.haslayer(TCP) or packet.haslayer(UDP):
                    transport_layer = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
                    src_port, dst_port = transport_layer.sport, transport_layer.dport
                    protocol = "TCP" if packet.haslayer(TCP) else "UDP"
                    
                    # Create bidirectional flow key
                    flow_tuple = tuple(sorted([
                        f"{src_ip}:{src_port}",
                        f"{dst_ip}:{dst_port}"
                    ]))
                    
                    return f"{protocol}_{flow_tuple[0]}_{flow_tuple[1]}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Flow key generation error: {e}")
            return None
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector."""
        # Define feature order (must match training data)
        feature_names = [
            'packet_length', 'protocol', 'src_ip', 'dst_ip', 'ttl', 'flags', 'frag',
            'src_port', 'dst_port', 'tcp_flags', 'window_size', 'seq_num', 'ack_num',
            'flow_duration', 'flow_packet_rate', 'flow_byte_rate',
            'flow_avg_packet_size', 'flow_std_packet_size',
            'flow_min_packet_size', 'flow_max_packet_size'
        ]
        
        # Create feature vector
        vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0)
            # Handle potential None values
            if value is None:
                value = 0
            vector.append(float(value))
        
        return np.array(vector).reshape(1, -1)
    
    def get_packet_info(self, packet) -> Dict[str, Any]:
        """Extract human-readable packet information."""
        info = {'timestamp': time.time()}
        
        try:
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                info.update({
                    'src_ip': ip_layer.src,
                    'dst_ip': ip_layer.dst,
                    'protocol': ip_layer.proto,
                    'length': len(packet)
                })
                
                if packet.haslayer(TCP):
                    tcp_layer = packet[TCP]
                    info.update({
                        'src_port': tcp_layer.sport,
                        'dst_port': tcp_layer.dport,
                        'tcp_flags': tcp_layer.flags
                    })
                elif packet.haslayer(UDP):
                    udp_layer = packet[UDP]
                    info.update({
                        'src_port': udp_layer.sport,
                        'dst_port': udp_layer.dport
                    })
            
        except Exception as e:
            self.logger.error(f"Error extracting packet info: {e}")
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capture statistics."""
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
            if stats['uptime'] > 0:
                stats['packets_per_second'] = stats['total_packets'] / stats['uptime']
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_capture()
        self.packet_buffer.clear()
        self.flow_tracker.clear()