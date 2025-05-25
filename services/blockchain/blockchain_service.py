"""
Blockchain Service for Content Verification and Authenticity

This service provides cryptographic proof of content authenticity,
crucial for combating deepfakes and ensuring trust in AI-processed videos.
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


class BlockchainNetwork(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BINANCE_SMART_CHAIN = "bsc"


@dataclass
class ContentMetadata:
    """Metadata for registered content"""
    content_hash: str
    timestamp: int
    creator: str
    device_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    ai_processing: Optional[List[str]] = None
    parent_hash: Optional[str] = None  # For tracking modifications
    watermark_hash: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of content verification"""
    is_authentic: bool
    blockchain_record: Optional[Dict] = None
    confidence_score: float = 0.0
    tampering_detected: bool = False
    watermark_valid: bool = False
    chain_of_custody: Optional[List[Dict]] = None


class BlockchainService:
    """Main blockchain service for content verification"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.networks = self._initialize_networks()
        self.contract_addresses = config.get('contract_addresses', {})
        self.private_key = config.get('private_key', os.environ.get('BLOCKCHAIN_PRIVATE_KEY'))
        self.ipfs_gateway = config.get('ipfs_gateway', 'https://ipfs.io/ipfs/')
        
        # Initialize watermarking key
        self.watermark_key = self._generate_watermark_key()
        
    def _initialize_networks(self) -> Dict[str, Web3]:
        """Initialize connections to blockchain networks"""
        networks = {}
        
        network_configs = {
            BlockchainNetwork.ETHEREUM: {
                'rpc_url': os.environ.get('ETHEREUM_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID'),
                'chain_id': 1
            },
            BlockchainNetwork.POLYGON: {
                'rpc_url': os.environ.get('POLYGON_RPC_URL', 'https://polygon-rpc.com'),
                'chain_id': 137
            },
            BlockchainNetwork.AVALANCHE: {
                'rpc_url': os.environ.get('AVALANCHE_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc'),
                'chain_id': 43114
            },
            BlockchainNetwork.ARBITRUM: {
                'rpc_url': os.environ.get('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),
                'chain_id': 42161
            },
            BlockchainNetwork.OPTIMISM: {
                'rpc_url': os.environ.get('OPTIMISM_RPC_URL', 'https://mainnet.optimism.io'),
                'chain_id': 10
            },
            BlockchainNetwork.BINANCE_SMART_CHAIN: {
                'rpc_url': os.environ.get('BSC_RPC_URL', 'https://bsc-dataseed.binance.org/'),
                'chain_id': 56
            }
        }
        
        for network, config in network_configs.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config['rpc_url']))
                
                # Add middleware for PoA chains
                if network in [BlockchainNetwork.BINANCE_SMART_CHAIN]:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                if w3.is_connected():
                    networks[network.value] = w3
                    logger.info(f"Connected to {network.value} network")
                else:
                    logger.warning(f"Failed to connect to {network.value} network")
                    
            except Exception as e:
                logger.error(f"Error initializing {network.value}: {e}")
                
        return networks
    
    def _generate_watermark_key(self) -> rsa.RSAPrivateKey:
        """Generate RSA key for watermarking"""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
    
    def hash_content(self, content: Union[bytes, np.ndarray], 
                    algorithm: str = 'sha256') -> str:
        """
        Generate cryptographic hash of content
        
        Args:
            content: Raw content bytes or numpy array (for video frames)
            algorithm: Hashing algorithm to use
            
        Returns:
            Hex string of content hash
        """
        if isinstance(content, np.ndarray):
            content = content.tobytes()
            
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha3_256':
            hasher = hashlib.sha3_256()
        elif algorithm == 'blake2b':
            hasher = hashlib.blake2b()
        else:
            raise ValueError(f"Unsupported hashing algorithm: {algorithm}")
            
        hasher.update(content)
        return hasher.hexdigest()
    
    def generate_perceptual_hash(self, frame: np.ndarray) -> str:
        """
        Generate perceptual hash for video frame
        More robust against minor modifications
        """
        # Convert to PIL Image
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        
        # Generate multiple hash types for robustness
        avg_hash = str(imagehash.average_hash(pil_image))
        p_hash = str(imagehash.phash(pil_image))
        d_hash = str(imagehash.dhash(pil_image))
        w_hash = str(imagehash.whash(pil_image))
        
        # Combine hashes
        combined = f"{avg_hash}:{p_hash}:{d_hash}:{w_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def embed_watermark(self, frame: np.ndarray, content_hash: str) -> Tuple[np.ndarray, str]:
        """
        Embed invisible watermark in video frame
        
        Args:
            frame: Video frame as numpy array
            content_hash: Hash of content to embed
            
        Returns:
            Watermarked frame and watermark hash
        """
        # Create watermark data
        watermark_data = {
            'content_hash': content_hash,
            'timestamp': int(time.time()),
            'service': 'ooblex'
        }
        
        # Sign watermark data
        watermark_bytes = json.dumps(watermark_data).encode()
        signature = self.watermark_key.sign(
            watermark_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Embed watermark using LSB steganography in DCT domain
        watermarked_frame = self._embed_dct_watermark(frame, signature)
        
        # Generate watermark hash
        watermark_hash = hashlib.sha256(signature).hexdigest()
        
        return watermarked_frame, watermark_hash
    
    def _embed_dct_watermark(self, frame: np.ndarray, data: bytes) -> np.ndarray:
        """Embed watermark using DCT (Discrete Cosine Transform)"""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)
        
        # Apply DCT
        dct = cv2.dct(y_channel)
        
        # Embed data in mid-frequency coefficients
        # This is simplified - production would use more sophisticated embedding
        data_bits = ''.join(format(byte, '08b') for byte in data[:32])  # Limit size
        
        # Embed bits
        h, w = dct.shape
        bit_index = 0
        for i in range(10, min(50, h)):
            for j in range(10, min(50, w)):
                if bit_index < len(data_bits):
                    if data_bits[bit_index] == '1':
                        dct[i, j] = abs(dct[i, j]) * 1.01  # Slight modification
                    else:
                        dct[i, j] = abs(dct[i, j]) * 0.99
                    bit_index += 1
        
        # Apply inverse DCT
        y_channel = cv2.idct(dct)
        ycrcb[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        watermarked = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return watermarked
    
    def extract_watermark(self, frame: np.ndarray) -> Optional[bytes]:
        """Extract watermark from frame"""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)
        
        # Apply DCT
        dct = cv2.dct(y_channel)
        
        # Extract bits from same positions
        extracted_bits = []
        h, w = dct.shape
        for i in range(10, min(50, h)):
            for j in range(10, min(50, w)):
                if len(extracted_bits) < 256:  # 32 bytes * 8 bits
                    if dct[i, j] > 0:
                        extracted_bits.append('1')
                    else:
                        extracted_bits.append('0')
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        for i in range(0, len(extracted_bits), 8):
            byte_bits = ''.join(extracted_bits[i:i+8])
            if len(byte_bits) == 8:
                extracted_bytes.append(int(byte_bits, 2))
        
        return bytes(extracted_bytes)
    
    async def register_content(self, content_metadata: ContentMetadata, 
                             network: BlockchainNetwork = BlockchainNetwork.POLYGON) -> Dict:
        """
        Register content on blockchain
        
        Args:
            content_metadata: Metadata about the content
            network: Blockchain network to use
            
        Returns:
            Transaction details
        """
        w3 = self.networks.get(network.value)
        if not w3:
            raise ValueError(f"Network {network.value} not available")
        
        # Get contract instance
        contract_address = self.contract_addresses.get(network.value)
        if not contract_address:
            raise ValueError(f"No contract address for {network.value}")
        
        # Load contract ABI
        with open('services/blockchain/smart_contracts/ContentRegistry.json', 'r') as f:
            contract_abi = json.load(f)['abi']
        
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Prepare transaction
        account = Account.from_key(self.private_key)
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Call contract method
        transaction = contract.functions.registerContent(
            content_metadata.content_hash,
            content_metadata.timestamp,
            content_metadata.creator,
            json.dumps(asdict(content_metadata))
        ).build_transaction({
            'nonce': nonce,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'from': account.address,
        })
        
        # Sign and send transaction
        signed_txn = account.sign_transaction(transaction)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'tx_hash': receipt['transactionHash'].hex(),
            'block_number': receipt['blockNumber'],
            'network': network.value,
            'contract_address': contract_address,
            'gas_used': receipt['gasUsed']
        }
    
    async def verify_content(self, content: Union[bytes, np.ndarray], 
                           expected_hash: Optional[str] = None) -> VerificationResult:
        """
        Verify content authenticity
        
        Args:
            content: Content to verify
            expected_hash: Expected content hash (if known)
            
        Returns:
            VerificationResult with verification details
        """
        # Generate content hash
        content_hash = self.hash_content(content)
        
        # Check if hash matches expected
        if expected_hash and content_hash != expected_hash:
            return VerificationResult(
                is_authentic=False,
                tampering_detected=True,
                confidence_score=0.0
            )
        
        # Search for content on blockchains
        blockchain_record = None
        for network_name, w3 in self.networks.items():
            try:
                record = await self._search_blockchain(w3, content_hash, network_name)
                if record:
                    blockchain_record = record
                    break
            except Exception as e:
                logger.error(f"Error searching {network_name}: {e}")
        
        # Extract and verify watermark if it's a frame
        watermark_valid = False
        if isinstance(content, np.ndarray):
            try:
                watermark_data = self.extract_watermark(content)
                if watermark_data:
                    # Verify watermark signature
                    public_key = self.watermark_key.public_key()
                    try:
                        public_key.verify(
                            watermark_data,
                            content_hash.encode(),
                            padding.PSS(
                                mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.MAX_LENGTH
                            ),
                            hashes.SHA256()
                        )
                        watermark_valid = True
                    except:
                        watermark_valid = False
            except:
                pass
        
        # Calculate confidence score
        confidence_score = 0.0
        if blockchain_record:
            confidence_score += 0.5
        if watermark_valid:
            confidence_score += 0.3
        if expected_hash and content_hash == expected_hash:
            confidence_score += 0.2
        
        # Get chain of custody if available
        chain_of_custody = None
        if blockchain_record:
            chain_of_custody = await self._get_chain_of_custody(
                blockchain_record['network'],
                content_hash
            )
        
        return VerificationResult(
            is_authentic=blockchain_record is not None,
            blockchain_record=blockchain_record,
            confidence_score=confidence_score,
            tampering_detected=False,
            watermark_valid=watermark_valid,
            chain_of_custody=chain_of_custody
        )
    
    async def _search_blockchain(self, w3: Web3, content_hash: str, 
                               network_name: str) -> Optional[Dict]:
        """Search for content hash on blockchain"""
        contract_address = self.contract_addresses.get(network_name)
        if not contract_address:
            return None
        
        # Load contract ABI
        with open('services/blockchain/smart_contracts/ContentRegistry.json', 'r') as f:
            contract_abi = json.load(f)['abi']
        
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        try:
            # Call view function
            result = contract.functions.getContent(content_hash).call()
            
            if result[0] != '':  # Content exists
                return {
                    'network': network_name,
                    'content_hash': result[0],
                    'timestamp': result[1],
                    'creator': result[2],
                    'metadata': json.loads(result[3]) if result[3] else {},
                    'block_number': result[4] if len(result) > 4 else None
                }
        except:
            return None
        
        return None
    
    async def _get_chain_of_custody(self, network: str, content_hash: str) -> List[Dict]:
        """Get full chain of custody for content"""
        w3 = self.networks.get(network)
        if not w3:
            return []
        
        contract_address = self.contract_addresses.get(network)
        if not contract_address:
            return []
        
        # Load contract ABI
        with open('services/blockchain/smart_contracts/ContentRegistry.json', 'r') as f:
            contract_abi = json.load(f)['abi']
        
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        try:
            # Get content history
            history = contract.functions.getContentHistory(content_hash).call()
            
            chain_of_custody = []
            for record in history:
                chain_of_custody.append({
                    'hash': record[0],
                    'timestamp': record[1],
                    'actor': record[2],
                    'action': record[3],
                    'metadata': json.loads(record[4]) if record[4] else {}
                })
            
            return chain_of_custody
        except:
            return []
    
    def generate_content_fingerprint(self, video_path: str, 
                                   sample_rate: int = 30) -> Dict[str, str]:
        """
        Generate comprehensive fingerprint for video content
        
        Args:
            video_path: Path to video file
            sample_rate: Frames to sample per second
            
        Returns:
            Dictionary of fingerprint components
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_hashes = []
        perceptual_hashes = []
        frame_count = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate) if fps > sample_rate else 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Generate cryptographic hash
                frame_hash = self.hash_content(frame)
                frame_hashes.append(frame_hash)
                
                # Generate perceptual hash
                p_hash = self.generate_perceptual_hash(frame)
                perceptual_hashes.append(p_hash)
            
            frame_count += 1
        
        cap.release()
        
        # Combine all hashes
        combined_hash = hashlib.sha256(
            ''.join(frame_hashes).encode()
        ).hexdigest()
        
        perceptual_fingerprint = hashlib.sha256(
            ''.join(perceptual_hashes).encode()
        ).hexdigest()
        
        return {
            'content_hash': combined_hash,
            'perceptual_fingerprint': perceptual_fingerprint,
            'frame_count': frame_count,
            'sample_rate': sample_rate,
            'algorithm': 'sha256',
            'version': '1.0'
        }
    
    async def batch_verify(self, content_hashes: List[str]) -> Dict[str, VerificationResult]:
        """Verify multiple content items in batch"""
        results = {}
        
        for content_hash in content_hashes:
            # Mock content for hash verification
            mock_content = content_hash.encode()
            result = await self.verify_content(mock_content, content_hash)
            results[content_hash] = result
        
        return results