#!/bin/bash

# Generate self-signed certificates for development
# DO NOT use these in production!

echo "Generating self-signed certificates for development..."

# Generate private key
openssl genrsa -out server.key 2048

# Generate certificate signing request
openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=Ooblex/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Create combined certificate file for some services
cat server.crt server.key > server.pem

# Clean up CSR
rm server.csr

# Set appropriate permissions
chmod 600 server.key
chmod 644 server.crt
chmod 600 server.pem

echo "Certificates generated successfully!"
echo "Files created:"
echo "  - server.key (private key)"
echo "  - server.crt (certificate)"
echo "  - server.pem (combined cert+key)"
echo ""
echo "WARNING: These are self-signed certificates for development only!"
echo "For production, use proper certificates from a trusted CA."