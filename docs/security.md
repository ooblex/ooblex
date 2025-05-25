# Security Best Practices

## Overview

This guide covers comprehensive security practices for Ooblex deployments, including authentication, encryption, network security, compliance requirements, and threat mitigation strategies. Security is built into every layer of the Ooblex platform.

## Getting Started

### Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimal access rights for users and services
3. **Zero Trust**: Never trust, always verify
4. **End-to-End Encryption**: Protect data in transit and at rest
5. **Continuous Monitoring**: Real-time threat detection and response

### Quick Security Checklist

```bash
#!/bin/bash
# Quick security audit script

echo "🔒 Ooblex Security Audit"
echo "========================"

# Check SSL/TLS
echo -n "✓ SSL/TLS Configuration: "
openssl s_client -connect localhost:443 -tls1_2 2>/dev/null | grep -q "Verify return code: 0" && echo "PASS" || echo "FAIL"

# Check firewall
echo -n "✓ Firewall Status: "
sudo ufw status | grep -q "Status: active" && echo "PASS" || echo "FAIL"

# Check API authentication
echo -n "✓ API Authentication: "
curl -s http://localhost:8080/api/v1/streams | grep -q "401" && echo "PASS" || echo "FAIL"

# Check database encryption
echo -n "✓ Database Encryption: "
sudo -u postgres psql -c "SHOW ssl;" | grep -q "on" && echo "PASS" || echo "FAIL"

# Check service permissions
echo -n "✓ Service Permissions: "
ps aux | grep ooblex | grep -v grep | awk '{print $1}' | grep -q "ooblex" && echo "PASS" || echo "FAIL"
```

## Detailed Security Implementation

### Authentication and Authorization

#### JWT Implementation

```python
# auth/jwt_handler.py
import jwt
import datetime
from typing import Optional, Dict
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.refresh_secret = self._derive_refresh_secret(secret_key)
    
    def _derive_refresh_secret(self, base_secret: str) -> str:
        """Derive refresh token secret from base secret"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ooblex_refresh_salt',
            iterations=100000,
        )
        return kdf.derive(base_secret.encode()).hex()
    
    def create_access_token(self, user_id: str, permissions: List[str]) -> str:
        """Create access token with short expiration"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
            "iat": datetime.datetime.utcnow(),
            "jti": secrets.token_hex(16),  # Unique token ID
            "type": "access"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token with long expiration"""
        payload = {
            "user_id": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
            "iat": datetime.datetime.utcnow(),
            "jti": secrets.token_hex(16),
            "type": "refresh"
        }
        return jwt.encode(payload, self.refresh_secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        """Verify and decode token"""
        try:
            secret = self.secret_key if token_type == "access" else self.refresh_secret
            payload = jwt.decode(token, secret, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != token_type:
                return None
            
            # Check if token is blacklisted
            if self._is_blacklisted(payload.get("jti")):
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        # Implementation depends on your blacklist storage
        # Could use Redis, database, or in-memory store
        return redis_client.sismember("blacklisted_tokens", jti)
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            # Add to blacklist with expiration
            ttl = exp - datetime.datetime.utcnow().timestamp()
            if ttl > 0:
                redis_client.setex(f"blacklist:{jti}", int(ttl), "1")
        except:
            pass
```

#### OAuth2/OIDC Integration

```python
# auth/oauth2.py
from authlib.integrations.fastapi_client import OAuth
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

oauth = OAuth()

# Configure OAuth providers
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'}
)

class OAuth2Handler:
    def __init__(self):
        self.providers = ['google', 'github', 'microsoft', 'okta']
    
    async def authenticate(self, provider: str, code: str) -> Dict:
        """Authenticate with OAuth provider"""
        if provider not in self.providers:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        client = oauth.create_client(provider)
        
        try:
            # Exchange code for token
            token = await client.authorize_access_token(code=code)
            
            # Get user info
            user_info = token.get('userinfo')
            if not user_info:
                resp = await client.get('user')
                user_info = resp.json()
            
            # Create or update user in database
            user = await self.create_or_update_user(provider, user_info)
            
            # Generate JWT tokens
            access_token = jwt_handler.create_access_token(
                user_id=user.id,
                permissions=user.permissions
            )
            refresh_token = jwt_handler.create_refresh_token(user_id=user.id)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "user": user
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
```

#### API Key Management

```python
# auth/api_keys.py
import hashlib
import secrets
from typing import Optional

class APIKeyManager:
    def __init__(self, db_session):
        self.db = db_session
        self.key_prefix = "ooblex_"
        self.hash_iterations = 100000
    
    def generate_api_key(self, user_id: str, name: str, permissions: List[str]) -> Dict:
        """Generate new API key"""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        api_key = f"{self.key_prefix}{raw_key}"
        
        # Hash key for storage
        key_hash = self._hash_key(api_key)
        
        # Store in database
        key_record = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            created_at=datetime.utcnow(),
            last_used=None,
            expires_at=datetime.utcnow() + timedelta(days=365)
        )
        self.db.add(key_record)
        self.db.commit()
        
        return {
            "id": key_record.id,
            "api_key": api_key,  # Only returned once
            "name": name,
            "permissions": permissions,
            "expires_at": key_record.expires_at
        }
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key using PBKDF2"""
        return hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            b'ooblex_api_salt',
            self.hash_iterations
        ).hex()
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return associated record"""
        if not api_key.startswith(self.key_prefix):
            return None
        
        key_hash = self._hash_key(api_key)
        
        # Look up key in database
        key_record = self.db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.expires_at > datetime.utcnow(),
            APIKey.revoked == False
        ).first()
        
        if key_record:
            # Update last used timestamp
            key_record.last_used = datetime.utcnow()
            self.db.commit()
            
            # Check rate limits
            if not self._check_rate_limit(key_record):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        return key_record
    
    def revoke_api_key(self, key_id: str, user_id: str):
        """Revoke API key"""
        key_record = self.db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if key_record:
            key_record.revoked = True
            key_record.revoked_at = datetime.utcnow()
            self.db.commit()
```

### Encryption

#### End-to-End Encryption for WebRTC

```javascript
// crypto/e2ee.js
class E2EEncryption {
    constructor() {
        this.keyPair = null;
        this.sharedSecrets = new Map();
        this.cryptoKey = null;
    }
    
    async initialize() {
        // Generate ECDH key pair
        this.keyPair = await crypto.subtle.generateKey(
            {
                name: 'ECDH',
                namedCurve: 'P-256'
            },
            true,
            ['deriveKey']
        );
        
        // Export public key
        const publicKey = await crypto.subtle.exportKey(
            'spki',
            this.keyPair.publicKey
        );
        
        return btoa(String.fromCharCode(...new Uint8Array(publicKey)));
    }
    
    async deriveSharedSecret(remotePublicKeyBase64, streamId) {
        // Import remote public key
        const remotePublicKey = await crypto.subtle.importKey(
            'spki',
            Uint8Array.from(atob(remotePublicKeyBase64), c => c.charCodeAt(0)),
            {
                name: 'ECDH',
                namedCurve: 'P-256'
            },
            false,
            []
        );
        
        // Derive shared secret
        const sharedSecret = await crypto.subtle.deriveKey(
            {
                name: 'ECDH',
                public: remotePublicKey
            },
            this.keyPair.privateKey,
            {
                name: 'AES-GCM',
                length: 256
            },
            false,
            ['encrypt', 'decrypt']
        );
        
        this.sharedSecrets.set(streamId, sharedSecret);
        return sharedSecret;
    }
    
    async encryptFrame(encodedFrame, controller, streamId) {
        const sharedSecret = this.sharedSecrets.get(streamId);
        if (!sharedSecret) {
            controller.enqueue(encodedFrame);
            return;
        }
        
        // Generate IV
        const iv = crypto.getRandomValues(new Uint8Array(12));
        
        // Encrypt frame data
        const encrypted = await crypto.subtle.encrypt(
            {
                name: 'AES-GCM',
                iv: iv,
                additionalData: new Uint8Array([encodedFrame.type])
            },
            sharedSecret,
            encodedFrame.data
        );
        
        // Prepend IV to encrypted data
        const encryptedData = new Uint8Array(iv.length + encrypted.byteLength);
        encryptedData.set(iv);
        encryptedData.set(new Uint8Array(encrypted), iv.length);
        
        encodedFrame.data = encryptedData.buffer;
        controller.enqueue(encodedFrame);
    }
    
    async decryptFrame(encodedFrame, controller, streamId) {
        const sharedSecret = this.sharedSecrets.get(streamId);
        if (!sharedSecret) {
            controller.enqueue(encodedFrame);
            return;
        }
        
        const data = new Uint8Array(encodedFrame.data);
        
        // Extract IV
        const iv = data.slice(0, 12);
        const encryptedData = data.slice(12);
        
        try {
            // Decrypt frame data
            const decrypted = await crypto.subtle.decrypt(
                {
                    name: 'AES-GCM',
                    iv: iv,
                    additionalData: new Uint8Array([encodedFrame.type])
                },
                sharedSecret,
                encryptedData
            );
            
            encodedFrame.data = decrypted;
            controller.enqueue(encodedFrame);
        } catch (e) {
            console.error('Decryption failed:', e);
            // Drop frame on decryption failure
        }
    }
}

// Apply E2EE to WebRTC
function applyE2EE(pc, encryption) {
    // Encrypt outgoing streams
    pc.getSenders().forEach(sender => {
        if (sender.track) {
            const streams = sender.createEncodedStreams();
            const transformStream = new TransformStream({
                transform: (frame, controller) => 
                    encryption.encryptFrame(frame, controller, pc.streamId)
            });
            
            streams.readable
                .pipeThrough(transformStream)
                .pipeTo(streams.writable);
        }
    });
    
    // Decrypt incoming streams
    pc.getReceivers().forEach(receiver => {
        const streams = receiver.createEncodedStreams();
        const transformStream = new TransformStream({
            transform: (frame, controller) => 
                encryption.decryptFrame(frame, controller, pc.streamId)
        });
        
        streams.readable
            .pipeThrough(transformStream)
            .pipeTo(streams.writable);
    });
}
```

#### Data Encryption at Rest

```python
# crypto/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os

class DataEncryption:
    def __init__(self, master_key: str):
        self.master_key = master_key
        self.backend = default_backend()
    
    def derive_key(self, salt: bytes, context: str) -> bytes:
        """Derive encryption key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt + context.encode(),
            iterations=100000,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt file with unique key"""
        # Generate unique salt for this file
        salt = os.urandom(16)
        
        # Derive file-specific key
        file_key = self.derive_key(salt, f"file:{file_path}")
        cipher = Fernet(file_key)
        
        # Encrypt file in chunks
        chunk_size = 64 * 1024  # 64KB chunks
        
        with open(file_path, 'rb') as infile, open(output_path, 'wb') as outfile:
            # Write salt as header
            outfile.write(salt)
            
            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break
                
                encrypted_chunk = cipher.encrypt(chunk)
                outfile.write(len(encrypted_chunk).to_bytes(4, 'big'))
                outfile.write(encrypted_chunk)
    
    def decrypt_file(self, encrypted_path: str, output_path: str):
        """Decrypt file"""
        with open(encrypted_path, 'rb') as infile:
            # Read salt from header
            salt = infile.read(16)
            
            # Derive file-specific key
            file_key = self.derive_key(salt, f"file:{output_path}")
            cipher = Fernet(file_key)
            
            with open(output_path, 'wb') as outfile:
                while True:
                    # Read chunk size
                    size_bytes = infile.read(4)
                    if not size_bytes:
                        break
                    
                    chunk_size = int.from_bytes(size_bytes, 'big')
                    encrypted_chunk = infile.read(chunk_size)
                    
                    decrypted_chunk = cipher.decrypt(encrypted_chunk)
                    outfile.write(decrypted_chunk)
    
    def encrypt_database_field(self, value: str, field_name: str) -> Dict:
        """Encrypt database field with deterministic encryption for searchability"""
        # Generate deterministic salt for this field
        salt = hashlib.sha256(f"{self.master_key}:{field_name}".encode()).digest()[:16]
        
        # Derive field-specific key
        field_key = self.derive_key(salt, f"field:{field_name}")
        cipher = Fernet(field_key)
        
        # Encrypt value
        encrypted = cipher.encrypt(value.encode())
        
        # Generate blind index for searching
        blind_index = self._generate_blind_index(value, field_name)
        
        return {
            "encrypted_value": base64.b64encode(encrypted).decode(),
            "blind_index": blind_index
        }
    
    def _generate_blind_index(self, value: str, field_name: str) -> str:
        """Generate searchable blind index"""
        index_key = self.derive_key(b'blind_index', field_name)
        return hashlib.hmac(index_key, value.encode(), hashlib.sha256).hexdigest()[:16]
```

### Network Security

#### Firewall Configuration

```bash
#!/bin/bash
# Advanced firewall configuration

# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH with rate limiting
sudo ufw limit 22/tcp comment 'SSH rate limited'

# HTTP/HTTPS
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# WebRTC ports
sudo ufw allow 3478/tcp comment 'STUN/TURN TCP'
sudo ufw allow 3478/udp comment 'STUN/TURN UDP'
sudo ufw allow 10000:60000/udp comment 'WebRTC media'

# Internal services (restrict to private networks)
sudo ufw allow from 10.0.0.0/8 to any port 5432 comment 'PostgreSQL internal'
sudo ufw allow from 172.16.0.0/12 to any port 6379 comment 'Redis internal'
sudo ufw allow from 192.168.0.0/16 to any port 9090 comment 'Prometheus internal'

# DDoS protection
sudo iptables -A INPUT -p tcp --dport 80 -m conntrack --ctstate NEW -m limit --limit 60/s --limit-burst 100 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -m conntrack --ctstate NEW -m limit --limit 60/s --limit-burst 100 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -m conntrack --ctstate NEW -j DROP
sudo iptables -A INPUT -p tcp --dport 443 -m conntrack --ctstate NEW -j DROP

# SYN flood protection
sudo iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j RETURN

# Save iptables rules
sudo iptables-save > /etc/iptables/rules.v4

# Enable firewall
sudo ufw --force enable

# Configure fail2ban
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = security@ooblex.com
sendername = Fail2Ban
action = %(action_mwl)s

[sshd]
enabled = true
port = 22
logpath = /var/log/auth.log

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-badbots]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
EOF

sudo systemctl restart fail2ban
```

#### SSL/TLS Configuration

```nginx
# nginx/ssl.conf
# Modern SSL configuration

ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/nginx/ssl/ca-bundle.crt;

# SSL session caching
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
ssl_session_tickets off;

# HSTS
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

# Additional security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' wss://ooblex.com https://api.ooblex.com; media-src 'self'; object-src 'none'; frame-ancestors 'self';" always;
add_header Permissions-Policy "camera=(self), microphone=(self), geolocation=(), payment=()";

# Certificate configuration
ssl_certificate /etc/nginx/ssl/ooblex.crt;
ssl_certificate_key /etc/nginx/ssl/ooblex.key;
ssl_dhparam /etc/nginx/ssl/dhparam.pem;
```

### Input Validation and Sanitization

```python
# security/validation.py
from typing import Any, Dict, List
import re
import bleach
from pydantic import BaseModel, validator, constr, conint
import magic

class InputValidator:
    def __init__(self):
        self.file_mime_types = {
            'image': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
            'video': ['video/mp4', 'video/webm', 'video/ogg'],
            'audio': ['audio/mpeg', 'audio/ogg', 'audio/wav']
        }
        
        self.sql_injection_patterns = [
            r'(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\s',
            r'(;|\s)(--|\#|\/\*)',
            r'(\'|\")\s*(OR|AND)\s*\d+\s*=\s*\d+',
            r'(\'|\")\s*(OR|AND)\s*(\'|\")\d+(\'|\")=(\'|\")\d+'
        ]
        
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
    
    def validate_sql_input(self, value: str) -> bool:
        """Check for SQL injection attempts"""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        return True
    
    def validate_xss_input(self, value: str) -> bool:
        """Check for XSS attempts"""
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        return True
    
    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML content"""
        allowed_tags = [
            'p', 'br', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'strong', 'em', 'u', 'a', 'img', 'ul', 'ol', 'li', 'blockquote'
        ]
        allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height'],
            'div': ['class'],
            'span': ['class']
        }
        
        return bleach.clean(
            html,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
    
    def validate_file_upload(self, file_path: str, file_type: str) -> Dict:
        """Validate uploaded files"""
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = 100 * 1024 * 1024  # 100MB
        
        if file_size > max_size:
            return {"valid": False, "error": "File size exceeds limit"}
        
        # Check MIME type
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_file(file_path)
        
        if file_type not in self.file_mime_types:
            return {"valid": False, "error": "Invalid file type"}
        
        if detected_mime not in self.file_mime_types[file_type]:
            return {"valid": False, "error": f"Invalid MIME type: {detected_mime}"}
        
        # Check for malicious content
        if self._check_malicious_content(file_path):
            return {"valid": False, "error": "Malicious content detected"}
        
        return {"valid": True, "mime_type": detected_mime, "size": file_size}
    
    def _check_malicious_content(self, file_path: str) -> bool:
        """Check for malicious content in files"""
        # Check for embedded executables
        with open(file_path, 'rb') as f:
            header = f.read(1024)
            
        # Check for common executable signatures
        exe_signatures = [
            b'MZ',  # DOS/Windows executable
            b'\x7fELF',  # Linux ELF
            b'\xfe\xed\xfa\xce',  # Mach-O (macOS)
            b'\xce\xfa\xed\xfe',  # Mach-O (macOS)
            b'#!/',  # Shell script
        ]
        
        for sig in exe_signatures:
            if header.startswith(sig):
                return True
        
        return False

# Pydantic models for request validation
class StreamCreateRequest(BaseModel):
    name: constr(min_length=1, max_length=100, regex=r'^[\w\s-]+$')
    resolution: constr(regex=r'^\d+x\d+$')
    bitrate: conint(ge=100000, le=10000000)
    ai_models: List[constr(regex=r'^[a-z_]+$')]
    
    @validator('name')
    def validate_name(cls, v):
        if not InputValidator().validate_sql_input(v):
            raise ValueError('Invalid characters in name')
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        width, height = map(int, v.split('x'))
        if width > 3840 or height > 2160:
            raise ValueError('Resolution too high')
        if width < 320 or height < 240:
            raise ValueError('Resolution too low')
        return v
```

### Security Monitoring and Logging

```python
# security/monitoring.py
import logging
import json
from datetime import datetime
from typing import Dict, Optional
import geoip2.database
from collections import defaultdict
import asyncio

class SecurityMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logger()
        self.geoip_reader = geoip2.database.Reader(config['geoip_db_path'])
        self.threat_scores = defaultdict(int)
        self.blocked_ips = set()
    
    def _setup_logger(self):
        """Setup security logger"""
        logger = logging.getLogger('security')
        logger.setLevel(logging.INFO)
        
        # File handler for security events
        fh = logging.FileHandler('/var/log/ooblex/security.log')
        fh.setLevel(logging.INFO)
        
        # Format for security logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        return logger
    
    def log_auth_attempt(self, request, success: bool, user_id: Optional[str] = None):
        """Log authentication attempt"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'auth_attempt',
            'success': success,
            'user_id': user_id,
            'ip_address': request.client.host,
            'user_agent': request.headers.get('User-Agent'),
            'method': request.method,
            'path': request.url.path
        }
        
        # Add geolocation
        try:
            geo = self.geoip_reader.city(request.client.host)
            event['location'] = {
                'country': geo.country.iso_code,
                'city': geo.city.name,
                'latitude': geo.location.latitude,
                'longitude': geo.location.longitude
            }
        except:
            event['location'] = None
        
        # Check for suspicious activity
        if not success:
            self._check_failed_auth(request.client.host)
        
        self.logger.info(json.dumps(event))
    
    def _check_failed_auth(self, ip_address: str):
        """Check for brute force attempts"""
        self.threat_scores[ip_address] += 1
        
        if self.threat_scores[ip_address] >= 5:
            self.block_ip(ip_address, reason='brute_force')
    
    def log_api_request(self, request, response, duration: float):
        """Log API request for analysis"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'api_request',
            'ip_address': request.client.host,
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'duration_ms': duration * 1000,
            'user_id': getattr(request.state, 'user_id', None)
        }
        
        # Flag suspicious patterns
        if self._is_suspicious_request(request):
            event['suspicious'] = True
            self.threat_scores[request.client.host] += 0.5
        
        self.logger.info(json.dumps(event))
    
    def _is_suspicious_request(self, request) -> bool:
        """Detect suspicious request patterns"""
        suspicious_patterns = [
            '/etc/passwd',
            '../',
            '<script',
            'SELECT * FROM',
            'DROP TABLE',
            'eval(',
            'base64_decode'
        ]
        
        url = str(request.url)
        for pattern in suspicious_patterns:
            if pattern in url:
                return True
        
        return False
    
    def log_security_event(self, event_type: str, details: Dict):
        """Log general security event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            **details
        }
        
        self.logger.warning(json.dumps(event))
        
        # Send alerts for critical events
        if event_type in ['intrusion_detected', 'data_breach', 'malware_detected']:
            asyncio.create_task(self._send_security_alert(event))
    
    async def _send_security_alert(self, event: Dict):
        """Send security alerts"""
        # Email alert
        await send_email(
            to=self.config['security_team_email'],
            subject=f"Security Alert: {event['event_type']}",
            body=json.dumps(event, indent=2)
        )
        
        # Webhook alert
        await send_webhook(
            url=self.config['security_webhook_url'],
            data=event
        )
        
        # SMS alert for critical events
        if event['event_type'] == 'data_breach':
            await send_sms(
                to=self.config['security_team_phone'],
                message=f"CRITICAL: {event['event_type']} detected at {event['timestamp']}"
            )
    
    def block_ip(self, ip_address: str, reason: str):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        
        # Add to firewall
        os.system(f"sudo ufw deny from {ip_address}")
        
        # Log blocking
        self.log_security_event('ip_blocked', {
            'ip_address': ip_address,
            'reason': reason,
            'threat_score': self.threat_scores[ip_address]
        })
    
    def is_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips

# SIEM Integration
class SIEMConnector:
    def __init__(self, siem_config: Dict):
        self.siem_url = siem_config['url']
        self.api_key = siem_config['api_key']
        self.batch_size = 100
        self.event_queue = []
    
    async def send_event(self, event: Dict):
        """Send event to SIEM"""
        self.event_queue.append(event)
        
        if len(self.event_queue) >= self.batch_size:
            await self._flush_events()
    
    async def _flush_events(self):
        """Flush events to SIEM"""
        if not self.event_queue:
            return
        
        try:
            response = await http_client.post(
                f"{self.siem_url}/api/events",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"events": self.event_queue}
            )
            response.raise_for_status()
            self.event_queue = []
        except Exception as e:
            logger.error(f"Failed to send events to SIEM: {e}")
```

### Compliance and Auditing

```python
# security/compliance.py
from typing import Dict, List
import hashlib
from datetime import datetime, timedelta

class ComplianceManager:
    def __init__(self):
        self.audit_log = []
        self.retention_policies = {
            'video_recordings': timedelta(days=90),
            'ai_results': timedelta(days=365),
            'user_data': timedelta(days=730),
            'logs': timedelta(days=180)
        }
    
    def log_data_access(self, user_id: str, resource_type: str, 
                       resource_id: str, action: str, metadata: Dict = None):
        """Log data access for audit trail"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'action': action,
            'metadata': metadata or {},
            'hash': None
        }
        
        # Create tamper-proof hash
        audit_entry['hash'] = self._create_audit_hash(audit_entry)
        
        # Store in audit log
        self.audit_log.append(audit_entry)
        
        # Persist to database
        self._persist_audit_entry(audit_entry)
    
    def _create_audit_hash(self, entry: Dict) -> str:
        """Create cryptographic hash of audit entry"""
        # Get previous hash for chaining
        prev_hash = self.audit_log[-1]['hash'] if self.audit_log else '0'
        
        # Create hash including previous hash
        content = f"{prev_hash}{entry['timestamp']}{entry['user_id']}{entry['action']}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_audit_integrity(self) -> bool:
        """Verify audit log hasn't been tampered with"""
        prev_hash = '0'
        
        for entry in self.audit_log:
            # Recreate hash
            content = f"{prev_hash}{entry['timestamp']}{entry['user_id']}{entry['action']}"
            expected_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if entry['hash'] != expected_hash:
                return False
            
            prev_hash = entry['hash']
        
        return True
    
    def gdpr_data_export(self, user_id: str) -> Dict:
        """Export all user data for GDPR compliance"""
        user_data = {
            'user_id': user_id,
            'export_date': datetime.utcnow().isoformat(),
            'profile': self._get_user_profile(user_id),
            'streams': self._get_user_streams(user_id),
            'ai_results': self._get_user_ai_results(user_id),
            'access_logs': self._get_user_access_logs(user_id),
            'api_keys': self._get_user_api_keys(user_id)
        }
        
        # Log data export
        self.log_data_access(
            user_id=user_id,
            resource_type='user_data',
            resource_id=user_id,
            action='gdpr_export'
        )
        
        return user_data
    
    def gdpr_data_deletion(self, user_id: str) -> Dict:
        """Delete user data for GDPR compliance"""
        deletion_report = {
            'user_id': user_id,
            'deletion_date': datetime.utcnow().isoformat(),
            'deleted_items': {}
        }
        
        # Delete user data
        deletion_report['deleted_items']['profile'] = self._delete_user_profile(user_id)
        deletion_report['deleted_items']['streams'] = self._delete_user_streams(user_id)
        deletion_report['deleted_items']['ai_results'] = self._delete_user_ai_results(user_id)
        deletion_report['deleted_items']['api_keys'] = self._delete_user_api_keys(user_id)
        
        # Log deletion
        self.log_data_access(
            user_id='system',
            resource_type='user_data',
            resource_id=user_id,
            action='gdpr_deletion',
            metadata=deletion_report
        )
        
        return deletion_report
    
    def enforce_retention_policies(self):
        """Enforce data retention policies"""
        for data_type, retention_period in self.retention_policies.items():
            cutoff_date = datetime.utcnow() - retention_period
            
            # Delete old data
            deleted_count = self._delete_old_data(data_type, cutoff_date)
            
            # Log retention enforcement
            self.log_data_access(
                user_id='system',
                resource_type=data_type,
                resource_id='*',
                action='retention_cleanup',
                metadata={'deleted_count': deleted_count, 'cutoff_date': cutoff_date.isoformat()}
            )
    
    def generate_compliance_report(self, report_type: str) -> Dict:
        """Generate compliance reports"""
        reports = {
            'gdpr': self._generate_gdpr_report,
            'hipaa': self._generate_hipaa_report,
            'sox': self._generate_sox_report,
            'pci': self._generate_pci_report
        }
        
        if report_type in reports:
            return reports[report_type]()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
```

### Threat Detection and Response

```python
# security/threat_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import asyncio

class ThreatDetector:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.threat_patterns = self._load_threat_patterns()
        self.baseline_behavior = {}
        self.active_threats = []
    
    def _load_threat_patterns(self) -> Dict:
        """Load known threat patterns"""
        return {
            'ddos': {
                'indicators': ['high_request_rate', 'distributed_sources'],
                'threshold': 1000,  # requests per minute
                'response': 'rate_limit'
            },
            'brute_force': {
                'indicators': ['failed_auth_attempts', 'sequential_passwords'],
                'threshold': 5,  # failed attempts
                'response': 'block_ip'
            },
            'data_exfiltration': {
                'indicators': ['unusual_download_volume', 'off_hours_access'],
                'threshold': 1000000000,  # 1GB
                'response': 'alert_and_monitor'
            },
            'injection_attack': {
                'indicators': ['sql_patterns', 'script_injection'],
                'threshold': 1,
                'response': 'block_request'
            },
            'lateral_movement': {
                'indicators': ['unusual_service_access', 'privilege_escalation'],
                'threshold': 3,
                'response': 'isolate_account'
            }
        }
    
    async def analyze_traffic(self, traffic_data: Dict) -> Dict:
        """Analyze network traffic for threats"""
        features = self._extract_features(traffic_data)
        
        # Check against known patterns
        pattern_matches = self._check_threat_patterns(features)
        
        # Anomaly detection
        anomaly_score = self._detect_anomalies(features)
        
        # Combine results
        threat_level = self._calculate_threat_level(pattern_matches, anomaly_score)
        
        if threat_level > 0.7:
            threat = {
                'timestamp': datetime.utcnow().isoformat(),
                'threat_level': threat_level,
                'patterns_matched': pattern_matches,
                'anomaly_score': anomaly_score,
                'source_ip': traffic_data.get('source_ip'),
                'recommended_action': self._recommend_action(pattern_matches)
            }
            
            self.active_threats.append(threat)
            await self._respond_to_threat(threat)
        
        return {
            'threat_detected': threat_level > 0.7,
            'threat_level': threat_level,
            'details': pattern_matches
        }
    
    def _extract_features(self, traffic_data: Dict) -> np.ndarray:
        """Extract features for ML analysis"""
        features = [
            traffic_data.get('request_rate', 0),
            traffic_data.get('unique_ips', 0),
            traffic_data.get('failed_auth_count', 0),
            traffic_data.get('data_volume', 0),
            traffic_data.get('unusual_endpoints', 0),
            traffic_data.get('time_of_day', 0),
            traffic_data.get('geo_diversity', 0)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies using ML"""
        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = self.anomaly_detector.predict(features)[0]
        
        # Get anomaly score
        score = self.anomaly_detector.score_samples(features)[0]
        
        # Normalize to 0-1 range
        return 1 - (score + 1) / 2
    
    async def _respond_to_threat(self, threat: Dict):
        """Automated threat response"""
        action = threat['recommended_action']
        
        if action == 'block_ip':
            await self._block_ip(threat['source_ip'])
        elif action == 'rate_limit':
            await self._apply_rate_limit(threat['source_ip'])
        elif action == 'isolate_account':
            await self._isolate_account(threat.get('user_id'))
        elif action == 'alert_and_monitor':
            await self._send_alert(threat)
            await self._enable_enhanced_monitoring(threat['source_ip'])
    
    def train_baseline(self, historical_data: List[Dict]):
        """Train baseline behavior model"""
        features = []
        for data in historical_data:
            features.append(self._extract_features(data))
        
        features = np.vstack(features)
        self.anomaly_detector.fit(features)
        
        # Calculate baseline statistics
        self.baseline_behavior = {
            'avg_request_rate': np.mean(features[:, 0]),
            'std_request_rate': np.std(features[:, 0]),
            'avg_data_volume': np.mean(features[:, 3]),
            'std_data_volume': np.std(features[:, 3])
        }

# Incident Response
class IncidentResponse:
    def __init__(self):
        self.playbooks = self._load_playbooks()
        self.active_incidents = []
    
    def _load_playbooks(self) -> Dict:
        """Load incident response playbooks"""
        return {
            'data_breach': {
                'severity': 'critical',
                'steps': [
                    'isolate_affected_systems',
                    'preserve_evidence',
                    'assess_scope',
                    'notify_stakeholders',
                    'implement_containment',
                    'begin_recovery',
                    'post_incident_review'
                ]
            },
            'ransomware': {
                'severity': 'critical',
                'steps': [
                    'disconnect_network',
                    'identify_variant',
                    'preserve_evidence',
                    'assess_backups',
                    'notify_authorities',
                    'begin_recovery',
                    'strengthen_defenses'
                ]
            },
            'ddos_attack': {
                'severity': 'high',
                'steps': [
                    'enable_ddos_protection',
                    'increase_capacity',
                    'filter_traffic',
                    'contact_isp',
                    'monitor_attack',
                    'post_attack_analysis'
                ]
            }
        }
    
    async def handle_incident(self, incident_type: str, details: Dict):
        """Handle security incident"""
        if incident_type not in self.playbooks:
            raise ValueError(f"Unknown incident type: {incident_type}")
        
        playbook = self.playbooks[incident_type]
        incident = {
            'id': str(uuid.uuid4()),
            'type': incident_type,
            'severity': playbook['severity'],
            'start_time': datetime.utcnow(),
            'status': 'active',
            'details': details,
            'steps_completed': []
        }
        
        self.active_incidents.append(incident)
        
        # Execute playbook steps
        for step in playbook['steps']:
            try:
                await self._execute_step(incident, step)
                incident['steps_completed'].append({
                    'step': step,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'completed'
                })
            except Exception as e:
                incident['steps_completed'].append({
                    'step': step,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'failed',
                    'error': str(e)
                })
                
                # Critical step failed, escalate
                if playbook['severity'] == 'critical':
                    await self._escalate_incident(incident)
        
        incident['status'] = 'resolved'
        incident['end_time'] = datetime.utcnow()
        
        return incident
```

## Best Practices

### Security Development Lifecycle

1. **Secure Coding Practices**
```python
# Use parameterized queries
def get_user_streams(user_id: str):
    query = "SELECT * FROM streams WHERE user_id = %s"
    return db.execute(query, (user_id,))

# Input validation
def validate_stream_name(name: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Invalid stream name")
    return name

# Output encoding
def render_user_content(content: str) -> str:
    return html.escape(content)
```

2. **Dependency Management**
```bash
# Regular dependency updates
pip install --upgrade pip-audit
pip-audit --fix

# Check for vulnerabilities
safety check --json

# Lock dependencies
pip freeze > requirements.lock.txt
```

3. **Code Review Process**
```yaml
# .github/workflows/security.yml
name: Security Checks

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run security scan
        uses: ShiftLeftSecurity/scan-action@master
        
      - name: Run dependency check
        run: |
          pip install safety
          safety check
          
      - name: Run static analysis
        run: |
          pip install bandit
          bandit -r . -f json -o bandit.json
          
      - name: Check secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
```

### Infrastructure Security

1. **Container Security**
```dockerfile
# Secure Dockerfile
FROM python:3.9-slim

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash ooblex

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application
WORKDIR /app
COPY --chown=ooblex:ooblex . .

# Use non-root user
USER ooblex

# Use dumb-init to handle signals
ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "main.py"]
```

2. **Kubernetes Security**
```yaml
# k8s/security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Incident Response Plan

1. **Detection Phase**
- Monitor security alerts
- Analyze threat indicators
- Verify incident validity

2. **Containment Phase**
- Isolate affected systems
- Preserve evidence
- Prevent lateral movement

3. **Eradication Phase**
- Remove malicious artifacts
- Patch vulnerabilities
- Update security controls

4. **Recovery Phase**
- Restore systems from backups
- Verify system integrity
- Resume normal operations

5. **Lessons Learned**
- Document incident timeline
- Identify improvement areas
- Update security procedures

## Troubleshooting

### Common Security Issues

1. **SSL/TLS Errors**
```bash
# Test SSL configuration
openssl s_client -connect ooblex.com:443 -tls1_2

# Check certificate expiration
openssl x509 -in /etc/ssl/certs/ooblex.crt -noout -dates

# Verify certificate chain
openssl verify -CAfile /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ooblex.crt
```

2. **Authentication Failures**
```python
# Debug JWT issues
def debug_jwt(token: str):
    try:
        # Decode without verification
        header = jwt.get_unverified_header(token)
        payload = jwt.decode(token, options={"verify_signature": False})
        
        print(f"Header: {header}")
        print(f"Payload: {payload}")
        
        # Check expiration
        exp = datetime.fromtimestamp(payload['exp'])
        print(f"Expires: {exp}")
        print(f"Valid: {exp > datetime.utcnow()}")
        
    except Exception as e:
        print(f"JWT Error: {e}")
```

3. **Firewall Issues**
```bash
# Check firewall rules
sudo iptables -L -n -v

# Test connectivity
nc -zv ooblex.com 443

# Check blocked IPs
sudo fail2ban-client status
sudo fail2ban-client status nginx-http-auth
```

### Security Resources

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CIS Benchmarks: https://www.cisecurity.org/cis-benchmarks/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- Security Headers: https://securityheaders.com/
- SSL Labs: https://www.ssllabs.com/ssltest/