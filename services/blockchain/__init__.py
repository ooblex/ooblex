"""
Blockchain Service for Ooblex

Provides content verification and authenticity for AI-processed videos
"""

from .blockchain_service import (
    BlockchainService,
    BlockchainNetwork,
    ContentMetadata,
    VerificationResult
)
from .ipfs_client import IPFSClient, IPFSFile, PinningService

__all__ = [
    'BlockchainService',
    'BlockchainNetwork',
    'ContentMetadata',
    'VerificationResult',
    'IPFSClient',
    'IPFSFile',
    'PinningService'
]

__version__ = '1.0.0'