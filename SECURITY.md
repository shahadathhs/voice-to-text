# Security Policy

## Supported Versions

Currently, only the latest version of Voice-to-Text is supported with security updates.

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public issue**.

### How to Report

1. **Email**: Send a detailed report to the project maintainer
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### What to Expect

- **Response time**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity
- **Public disclosure**: After fix is released

### Security Best Practices

This project follows these security practices:

- ✅ **Fully Local**: Audio files never leave your machine
- ✅ **No API Keys**: No cloud services or external APIs
- ✅ **File Validation**: All uploads are validated for type and size
- ✅ **Input Sanitization**: File paths and user inputs are sanitized
- ✅ **Dependency Scanning**: Automated security scanning with Bandit
- ✅ **Regular Updates**: Dependencies are kept up to date

### Severity Levels

- **Critical**: Remote code execution, data exposure
- **High**: Local privilege escalation, data loss
- **Medium**: DoS, minor data exposure
- **Low**: Information disclosure, minor issues

## Security Features

### File Upload Security

- File type validation (WAV, MP3, OGG, M4A, FLAC, AAC only)
- File size limits (configurable, default 500MB)
- Safe filename sanitization
- Temporary file isolation

### API Security

- CORS configuration
- Request validation with Pydantic
- Error message sanitization
- No sensitive data in logs

### Deployment Security

- No hardcoded secrets
- Environment-based configuration
- Docker container isolation
- Health check endpoints

## Dependency Security

We use automated tools to scan for vulnerabilities:

```bash
make security  # Run Bandit security scanner
```

Security-focused dependencies:
- **FastAPI**: Built-in security features
- **Pydantic**: Input validation
- **Bandit**: Security linting

## Private Deployment

For maximum security, deploy entirely offline:

```bash
# Run locally without internet
make dev

# Deploy offline Docker image
docker compose -f compose.yaml up -d
```

No internet connection required after initial setup.

## Questions?

For security questions or concerns, please open an issue with the `security` tag.
