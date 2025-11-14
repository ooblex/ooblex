# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Ooblex seriously. If you have discovered a security vulnerability in our codebase, please report it to us following the guidelines below.

### Where to Report

Please report security vulnerabilities via one of the following methods:

1. **Email**: security@ooblex.com
2. **GitHub Security Advisories**: [Create a security advisory](https://github.com/ooblex/ooblex/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any proof-of-concept code
- Your contact information for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Target**: 
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report
2. **Assessment**: Our security team will assess the vulnerability
3. **Communication**: We will keep you informed about the progress
4. **Resolution**: We will work on a fix and coordinate disclosure
5. **Recognition**: With your permission, we will acknowledge your contribution

## Security Measures

### Dependencies

- All dependencies are regularly scanned for vulnerabilities
- Automated dependency updates via Dependabot
- Security patches are prioritized and deployed quickly

### Code Security

- Static code analysis with CodeQL
- Container scanning for Docker images
- Secret scanning enabled on all repositories
- Regular security audits

### Infrastructure

- End-to-end encryption for all data in transit
- Encryption at rest for sensitive data
- Regular penetration testing
- Compliance with industry standards

## Best Practices for Users

1. **Keep Dependencies Updated**: Regularly update all Ooblex dependencies
2. **Use Strong Authentication**: Enable MFA where available
3. **Secure API Keys**: Never commit API keys or secrets to repositories
4. **Monitor Security Advisories**: Subscribe to our security advisories
5. **Report Suspicious Activity**: Contact us immediately if you notice unusual behavior

## Security Features

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- OAuth 2.0 support
- API key management

### Data Protection
- AES-256 encryption for sensitive data
- TLS 1.3 for all communications
- Secure WebRTC implementation
- Privacy-preserving ML processing

### Monitoring & Compliance
- Real-time security monitoring
- Audit logging
- GDPR compliance
- SOC 2 Type II certification (in progress)

## Vulnerability Disclosure Policy

We follow a coordinated vulnerability disclosure process:

1. **Private Disclosure**: Vulnerabilities are first disclosed privately to give us time to address them
2. **Patch Development**: We develop and test patches
3. **Notification**: We notify users about the vulnerability and available patches
4. **Public Disclosure**: After users have had time to update, we publicly disclose the vulnerability

## Contact

For any security-related questions or concerns, please contact:

- **Email**: security@ooblex.com
- **PGP Key**: [Download our PGP key](https://ooblex.com/pgp-key.asc)

## Acknowledgments

We thank the following security researchers for responsibly disclosing vulnerabilities:

- [Contributors will be listed here with their permission]

---

Last Updated: January 2025