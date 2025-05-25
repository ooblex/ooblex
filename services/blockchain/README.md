# Blockchain Service for Ooblex

This service provides cryptographic proof of content authenticity for AI-processed videos, crucial for combating deepfakes and ensuring trust.

## Features

### Content Verification
- **Cryptographic Hashing**: SHA-256, SHA3-256, BLAKE2b algorithms
- **Perceptual Hashing**: Robust against minor modifications
- **Blockchain Registration**: Multi-chain support (Ethereum, Polygon, Avalanche, Arbitrum, Optimism, BSC)
- **IPFS Integration**: Distributed content storage
- **Watermarking**: Invisible DCT-based watermarks

### Chain of Custody
- Complete provenance tracking
- Derivative content linking
- Timestamp verification
- Actor identification

### Smart Contracts
- Solidity-based ContentRegistry contract
- Access control with verifier roles
- Batch verification support
- Emergency pause functionality

## API Endpoints

### Register Content
```bash
POST /blockchain/register
{
  "content_hash": "sha256_hash",
  "content_type": "video|frame|processed",
  "creator": "username",
  "device_id": "device123",
  "location": {"lat": 0.0, "lon": 0.0},
  "ai_processing": ["face_swap", "style_transfer"],
  "parent_hash": "parent_sha256_hash",
  "network": "polygon"
}
```

### Verify Content
```bash
POST /blockchain/verify
{
  "content_hash": "sha256_hash",
  "ipfs_cid": "QmXxx...",
  "content_data": "base64_encoded_data"
}
```

### Get Provenance
```bash
GET /blockchain/provenance/{content_hash}?network=polygon
```

### Generate Fingerprint
```bash
POST /blockchain/fingerprint
{
  "video_path": "/path/to/video.mp4",
  "sample_rate": 30
}
```

## Environment Variables

```bash
# Blockchain Networks
BLOCKCHAIN_PRIVATE_KEY=your_private_key
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
POLYGON_RPC_URL=https://polygon-rpc.com
AVALANCHE_RPC_URL=https://api.avax.network/ext/bc/C/rpc
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
OPTIMISM_RPC_URL=https://mainnet.optimism.io
BSC_RPC_URL=https://bsc-dataseed.binance.org/

# IPFS Configuration
IPFS_NODE_URL=http://localhost:5001
IPFS_GATEWAY_URL=https://ipfs.io

# Pinning Services
PINATA_API_KEY=your_pinata_key
INFURA_PROJECT_ID=your_infura_id
INFURA_PROJECT_SECRET=your_infura_secret
```

## Smart Contract Deployment

1. Install dependencies:
```bash
npm install -g truffle
npm install @openzeppelin/contracts
```

2. Compile contract:
```bash
truffle compile
```

3. Deploy to network:
```bash
truffle migrate --network polygon
```

## Watermarking

The service embeds invisible watermarks using DCT (Discrete Cosine Transform):
- Watermarks contain content hash, timestamp, and signature
- Robust against compression and minor modifications
- Can be extracted for verification

## Security Considerations

1. **Private Key Management**: Store private keys securely (use HSM in production)
2. **RPC Endpoints**: Use authenticated endpoints for production
3. **Access Control**: Implement proper authentication for API endpoints
4. **Smart Contract Auditing**: Audit contracts before mainnet deployment

## Development

### Running Tests
```bash
pytest tests/test_blockchain_service.py
```

### Local IPFS Node
```bash
docker run -d --name ipfs -p 5001:5001 -p 8080:8080 ipfs/kubo:latest
```

### Contract Verification
```bash
truffle run verify ContentRegistry --network polygon
```

## Integration Example

```python
from services.blockchain import BlockchainService, ContentMetadata

# Initialize service
blockchain = BlockchainService(config)

# Register content
metadata = ContentMetadata(
    content_hash="sha256_hash",
    timestamp=int(time.time()),
    creator="user123",
    ai_processing=["face_swap"]
)

tx_result = await blockchain.register_content(metadata)

# Verify content
result = await blockchain.verify_content(content_bytes, expected_hash)
print(f"Authentic: {result.is_authentic}, Confidence: {result.confidence_score}")
```

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check RPC URLs and network connectivity
2. **Transaction Failed**: Ensure sufficient gas and correct network
3. **IPFS Timeout**: Verify IPFS node is running and accessible
4. **Watermark Not Found**: Check if content was properly watermarked

### Monitoring

- Check service health: `GET /health`
- View metrics: `GET /metrics`
- Monitor blockchain transactions in respective explorers

## License

Part of the Ooblex project - see main LICENSE file.