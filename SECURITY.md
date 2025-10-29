# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these guidelines:

### 🔒 **Responsible Disclosure**

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by emailing: **rusirubandara408@gmail.com**

### 📋 **What to Include**

When reporting a vulnerability, please provide:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if known)
5. **Your contact information** for follow-up

### ⏰ **Response Timeline**

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: 2-4 weeks (depending on severity)
- **Disclosure**: After fix is released

### 🏆 **Recognition**

- Security researchers who report valid vulnerabilities will be credited
- We maintain a security contributors list
- Consider responsible disclosure bounties for critical issues

## Security Considerations

### 🛡️ **Network Monitoring**

This system performs network packet capture, which requires:

- **Administrator privileges** on most systems
- **Network interface access** permissions
- **Firewall configuration** considerations

### 🔐 **Data Privacy**

- **Packet data** is processed in-memory and not stored by default
- **Log files** may contain network metadata
- **Configuration files** should not contain sensitive credentials
- **Model files** do not contain raw packet data

### ⚙️ **System Security**

- Run with **minimum required privileges**
- Use **virtual environments** for isolation
- **Validate input data** from network sources
- **Encrypt stored models** if containing sensitive patterns

### 🌐 **Network Security**

- **Monitor network interface selection** carefully
- **Validate packet sources** in production environments
- **Consider network segmentation** for deployment
- **Review firewall rules** for dashboard access

## Security Best Practices

### For Users:

1. **Environment Isolation**
   ```bash
   python -m venv intrusion_detection_env
   source intrusion_detection_env/bin/activate  # Linux/Mac
   # or
   intrusion_detection_env\Scripts\activate  # Windows
   ```

2. **Configuration Security**
   ```yaml
   # Don't store credentials in config files
   alerts:
     email_password: ${EMAIL_PASSWORD}  # Use environment variables
   ```

3. **Network Interface**
   ```yaml
   network:
     interface: "eth0"  # Specify exact interface
     filter: "not host 127.0.0.1"  # Filter localhost traffic
   ```

### For Developers:

1. **Input Validation**
   ```python
   def validate_packet(packet):
       # Always validate network input
       if not packet or len(packet) > MAX_PACKET_SIZE:
           raise ValueError("Invalid packet")
   ```

2. **Secure Defaults**
   ```python
   # Use secure defaults
   DEFAULT_CONFIG = {
       'logging': {'level': 'INFO'},  # Not DEBUG in production
       'network': {'filter': 'not port 22'}  # Exclude SSH
   }
   ```

3. **Error Handling**
   ```python
   try:
       process_packet(packet)
   except Exception as e:
       # Don't leak sensitive info in errors
       logger.error("Packet processing failed", exc_info=False)
   ```

## Compliance

This project aims to comply with:

- **GDPR** - Data protection and privacy
- **HIPAA** - Healthcare data security (when applicable)
- **SOC 2** - Security controls for service organizations
- **NIST Cybersecurity Framework** - Security best practices

## Security Tools

We recommend using these tools for security assessment:

- **Static Analysis**: `bandit`, `safety`
- **Dependency Scanning**: `pip-audit`, `snyk`
- **Secret Detection**: `detect-secrets`, `truffleHog`
- **Container Scanning**: `docker-bench-security`

## Contact

For security-related questions or concerns:

- **Email**: security@example.com
- **GPG Key**: [Public key link]
- **Response Time**: 48 hours maximum

---

**Remember**: Security is everyone's responsibility! 🛡️
