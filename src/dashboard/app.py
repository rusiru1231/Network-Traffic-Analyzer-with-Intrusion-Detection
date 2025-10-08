"""
Streamlit Dashboard for Network Intrusion Detection System
Real-time visualization and monitoring interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import threading
from collections import deque

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.intrusion_detector import IntrusionDetector
from utils.packet_capture import PacketCapture
from utils.logger import setup_logging, SecurityLogger
from utils.config_manager import ConfigManager


class NetworkDashboard:
    """Main dashboard class for real-time monitoring."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self.packet_capture = None
        self.intrusion_detector = None
        self.security_logger = SecurityLogger()
        
        # Data storage for visualization
        self.packet_data = deque(maxlen=1000)
        self.threat_data = deque(maxlen=100)
        self.network_stats = {}
        
        # Dashboard state
        self.monitoring_active = False
        self.last_update = datetime.now()
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Network Intrusion Detection System",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .threat-alert {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 10px;
            margin: 10px 0;
        }
        .normal-status {
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render dashboard header."""
        st.title("🛡️ Network Intrusion Detection System")
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢 Active" if self.monitoring_active else "🔴 Inactive"
            st.metric("System Status", status)
        
        with col2:
            total_packets = len(self.packet_data)
            st.metric("Total Packets", f"{total_packets:,}")
        
        with col3:
            threat_count = len(self.threat_data)
            st.metric("Threats Detected", threat_count)
        
        with col4:
            uptime = datetime.now() - self.last_update
            st.metric("Last Update", f"{uptime.seconds}s ago")
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title("Control Panel")
        
        # System controls
        st.sidebar.subheader("System Controls")
        
        if st.sidebar.button("Start Monitoring", disabled=self.monitoring_active):
            self.start_monitoring()
        
        if st.sidebar.button("Stop Monitoring", disabled=not self.monitoring_active):
            self.stop_monitoring()
        
        if st.sidebar.button("Clear Data"):
            self.clear_data()
        
        # Configuration
        st.sidebar.subheader("Configuration")
        
        # Network interface selection
        interface = st.sidebar.selectbox(
            "Network Interface",
            options=["auto", "eth0", "wlan0", "lo"],
            index=0
        )
        
        # Detection threshold
        threshold = st.sidebar.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=5
        )
        
        # Model information
        st.sidebar.subheader("Model Information")
        if self.intrusion_detector and hasattr(self.intrusion_detector, 'model_metadata'):
            metadata = self.intrusion_detector.model_metadata
            st.sidebar.write(f"**Model Type:** {metadata.get('type', 'Unknown')}")
            st.sidebar.write(f"**Accuracy:** {metadata.get('accuracy', 'N/A')}")
            st.sidebar.write(f"**Precision:** {metadata.get('precision', 'N/A')}")
        
        return {
            'interface': interface,
            'threshold': threshold,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval
        }
    
    def render_real_time_metrics(self):
        """Render real-time metrics."""
        st.subheader("📊 Real-time Network Metrics")
        
        if not self.packet_data:
            st.info("No packet data available. Start monitoring to see real-time metrics.")
            return
        
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        # Protocol distribution
        with col1:
            st.markdown("**Protocol Distribution**")
            protocol_counts = self._get_protocol_distribution()
            
            if protocol_counts:
                fig_protocol = px.pie(
                    values=list(protocol_counts.values()),
                    names=list(protocol_counts.keys()),
                    title="Protocol Distribution"
                )
                fig_protocol.update_layout(height=300)
                st.plotly_chart(fig_protocol, use_container_width=True)
        
        # Packet rate over time
        with col2:
            st.markdown("**Packet Rate (packets/sec)**")
            packet_rates = self._get_packet_rates()
            
            if packet_rates:
                fig_rate = px.line(
                    x=list(range(len(packet_rates))),
                    y=packet_rates,
                    title="Packet Rate Over Time"
                )
                fig_rate.update_layout(height=300)
                st.plotly_chart(fig_rate, use_container_width=True)
        
        # Threat detection timeline
        with col3:
            st.markdown("**Threat Detection Timeline**")
            if self.threat_data:
                threat_times = [threat['timestamp'] for threat in self.threat_data]
                threat_confidences = [threat['confidence'] for threat in self.threat_data]
                
                fig_threats = px.scatter(
                    x=threat_times,
                    y=threat_confidences,
                    title="Threat Detection Timeline",
                    labels={'x': 'Time', 'y': 'Confidence'}
                )
                fig_threats.update_layout(height=300)
                st.plotly_chart(fig_threats, use_container_width=True)
            else:
                st.info("No threats detected yet.")
    
    def render_threat_alerts(self):
        """Render threat alerts section."""
        st.subheader("🚨 Threat Alerts")
        
        if not self.threat_data:
            st.markdown(
                '<div class="normal-status">✅ No threats detected. System operating normally.</div>',
                unsafe_allow_html=True
            )
            return
        
        # Display recent threats
        for threat in list(self.threat_data)[-5:]:  # Show last 5 threats
            confidence_pct = threat['confidence'] * 100
            timestamp = threat['timestamp'].strftime("%H:%M:%S")
            
            alert_html = f"""
            <div class="threat-alert">
                <strong>🚨 THREAT DETECTED</strong><br>
                <strong>Time:</strong> {timestamp}<br>
                <strong>Confidence:</strong> {confidence_pct:.1f}%<br>
                <strong>Source:</strong> {threat.get('src_ip', 'Unknown')}<br>
                <strong>Destination:</strong> {threat.get('dst_ip', 'Unknown')}<br>
                <strong>Protocol:</strong> {threat.get('protocol', 'Unknown')}
            </div>
            """
            st.markdown(alert_html, unsafe_allow_html=True)
    
    def render_network_analysis(self):
        """Render detailed network analysis."""
        st.subheader("🔍 Network Analysis")
        
        if not self.packet_data:
            st.info("No data available for analysis.")
            return
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Traffic Analysis", "Geographic View", "Port Analysis"])
        
        with tab1:
            self._render_traffic_analysis()
        
        with tab2:
            self._render_geographic_view()
        
        with tab3:
            self._render_port_analysis()
    
    def _render_traffic_analysis(self):
        """Render traffic analysis charts."""
        # Packet size distribution
        packet_sizes = [packet.get('size', 0) for packet in self.packet_data]
        
        if packet_sizes:
            fig_sizes = px.histogram(
                x=packet_sizes,
                title="Packet Size Distribution",
                labels={'x': 'Packet Size (bytes)', 'y': 'Count'}
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
        
        # Top source IPs
        src_ips = [packet.get('src_ip', 'Unknown') for packet in self.packet_data]
        src_ip_counts = pd.Series(src_ips).value_counts().head(10)
        
        if not src_ip_counts.empty:
            fig_src_ips = px.bar(
                x=src_ip_counts.index,
                y=src_ip_counts.values,
                title="Top Source IP Addresses",
                labels={'x': 'Source IP', 'y': 'Packet Count'}
            )
            fig_src_ips.update_xaxes(tickangle=45)
            st.plotly_chart(fig_src_ips, use_container_width=True)
    
    def _render_geographic_view(self):
        """Render geographic analysis (placeholder)."""
        st.info("Geographic analysis requires IP geolocation data.")
        st.write("This feature would show:")
        st.write("- Geographic distribution of traffic sources")
        st.write("- Attack origins on world map")
        st.write("- Regional threat patterns")
    
    def _render_port_analysis(self):
        """Render port usage analysis."""
        # Extract port information
        src_ports = [packet.get('src_port', 0) for packet in self.packet_data if packet.get('src_port')]
        dst_ports = [packet.get('dst_port', 0) for packet in self.packet_data if packet.get('dst_port')]
        
        if dst_ports:
            # Top destination ports
            dst_port_counts = pd.Series(dst_ports).value_counts().head(15)
            
            fig_ports = px.bar(
                x=dst_port_counts.values,
                y=dst_port_counts.index,
                orientation='h',
                title="Top Destination Ports",
                labels={'x': 'Connection Count', 'y': 'Port Number'}
            )
            st.plotly_chart(fig_ports, use_container_width=True)
    
    def start_monitoring(self):
        """Start network monitoring."""
        try:
            st.info("Initializing network monitoring...")
            
            # Initialize components
            self.intrusion_detector = IntrusionDetector()
            self.packet_capture = PacketCapture(
                interface=self.config['network']['interface']
            )
            
            self.monitoring_active = True
            st.success("Network monitoring started successfully!")
            
        except Exception as e:
            st.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop network monitoring."""
        self.monitoring_active = False
        if self.packet_capture:
            self.packet_capture.stop_capture()
        st.success("Network monitoring stopped.")
    
    def clear_data(self):
        """Clear all stored data."""
        self.packet_data.clear()
        self.threat_data.clear()
        st.success("Data cleared successfully!")
    
    def _get_protocol_distribution(self):
        """Get protocol distribution from packet data."""
        protocols = [packet.get('protocol', 'Unknown') for packet in self.packet_data]
        return pd.Series(protocols).value_counts().to_dict()
    
    def _get_packet_rates(self):
        """Calculate packet rates over time."""
        if len(self.packet_data) < 2:
            return []
        
        # Simple rate calculation (packets per second)
        rates = []
        window_size = 10
        
        for i in range(window_size, len(self.packet_data)):
            window_packets = list(self.packet_data)[i-window_size:i]
            rates.append(len(window_packets))
        
        return rates
    
    def simulate_data(self):
        """Simulate network data for demonstration."""
        # Generate synthetic packet data
        protocols = ['TCP', 'UDP', 'ICMP']
        
        for _ in range(50):
            packet = {
                'timestamp': datetime.now(),
                'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22, 21, 53, 25]),
                'protocol': np.random.choice(protocols),
                'size': np.random.randint(64, 1500)
            }
            self.packet_data.append(packet)
        
        # Generate some threat data
        if np.random.random() < 0.1:  # 10% chance of threat
            threat = {
                'timestamp': datetime.now(),
                'confidence': np.random.uniform(0.6, 0.9),
                'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                'protocol': np.random.choice(protocols)
            }
            self.threat_data.append(threat)
    
    def run(self):
        """Run the dashboard."""
        self.setup_page()
        
        # Render dashboard components
        self.render_header()
        sidebar_config = self.render_sidebar()
        
        # Add demo data button
        if st.sidebar.button("Generate Demo Data"):
            self.simulate_data()
            st.rerun()
        
        # Main content
        self.render_real_time_metrics()
        st.markdown("---")
        self.render_threat_alerts()
        st.markdown("---")
        self.render_network_analysis()
        
        # Auto refresh
        if sidebar_config['auto_refresh']:
            time.sleep(sidebar_config['refresh_interval'])
            st.rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = NetworkDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()