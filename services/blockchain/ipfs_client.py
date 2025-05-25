"""
IPFS Client for Distributed Content Storage

Provides integration with IPFS (InterPlanetary File System) for
decentralized content storage and addressing.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Union, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import aiofiles
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class IPFSFile:
    """Represents a file stored in IPFS"""
    cid: str  # Content Identifier
    name: Optional[str] = None
    size: Optional[int] = None
    type: Optional[str] = None
    pinned: bool = False
    timestamp: Optional[datetime] = None


@dataclass
class PinningService:
    """Configuration for remote pinning service"""
    name: str
    endpoint: str
    access_token: str
    
    
class IPFSClient:
    """
    IPFS client for content storage and retrieval
    
    Supports:
    - Local IPFS node via HTTP API
    - Remote pinning services (Pinata, Infura, etc.)
    - Content addressing and verification
    - Clustering for redundancy
    """
    
    def __init__(self, config: Dict):
        self.node_url = config.get('node_url', 'http://localhost:5001')
        self.gateway_url = config.get('gateway_url', 'https://ipfs.io')
        self.pinning_services = self._init_pinning_services(config.get('pinning_services', []))
        self.cluster_nodes = config.get('cluster_nodes', [])
        self.timeout = config.get('timeout', 300)
        
        # Initialize session
        self.session = None
        
    def _init_pinning_services(self, services_config: List[Dict]) -> List[PinningService]:
        """Initialize remote pinning services"""
        services = []
        
        # Default services from environment variables
        if os.environ.get('PINATA_API_KEY'):
            services.append(PinningService(
                name='pinata',
                endpoint='https://api.pinata.cloud',
                access_token=os.environ.get('PINATA_API_KEY')
            ))
            
        if os.environ.get('INFURA_PROJECT_ID'):
            services.append(PinningService(
                name='infura',
                endpoint=f'https://ipfs.infura.io:5001',
                access_token=os.environ.get('INFURA_PROJECT_SECRET')
            ))
            
        # Add configured services
        for service_config in services_config:
            services.append(PinningService(**service_config))
            
        return services
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def add_file(self, file_path: str, pin: bool = True) -> IPFSFile:
        """
        Add file to IPFS
        
        Args:
            file_path: Path to file to add
            pin: Whether to pin the file locally
            
        Returns:
            IPFSFile object with CID and metadata
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        # Read file
        async with aiofiles.open(file_path, 'rb') as f:
            file_data = await f.read()
            
        # Prepare multipart data
        data = aiohttp.FormData()
        data.add_field('file',
                      file_data,
                      filename=os.path.basename(file_path),
                      content_type='application/octet-stream')
        
        # Add to IPFS
        url = urljoin(self.node_url, '/api/v0/add')
        params = {'pin': str(pin).lower()}
        
        try:
            async with self.session.post(url, data=data, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                
                ipfs_file = IPFSFile(
                    cid=result['Hash'],
                    name=result.get('Name', os.path.basename(file_path)),
                    size=int(result.get('Size', 0)),
                    pinned=pin,
                    timestamp=datetime.utcnow()
                )
                
                # Pin to remote services if requested
                if pin:
                    await self._pin_to_services(ipfs_file)
                    
                return ipfs_file
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to add file to IPFS: {e}")
            raise
    
    async def add_json(self, data: Dict, pin: bool = True) -> IPFSFile:
        """
        Add JSON data to IPFS
        
        Args:
            data: Dictionary to store as JSON
            pin: Whether to pin the content
            
        Returns:
            IPFSFile object with CID
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        json_str = json.dumps(data, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        # Prepare multipart data
        form_data = aiohttp.FormData()
        form_data.add_field('file',
                           json_bytes,
                           filename='data.json',
                           content_type='application/json')
        
        # Add to IPFS
        url = urljoin(self.node_url, '/api/v0/add')
        params = {'pin': str(pin).lower()}
        
        try:
            async with self.session.post(url, data=form_data, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                
                ipfs_file = IPFSFile(
                    cid=result['Hash'],
                    name='data.json',
                    size=len(json_bytes),
                    type='application/json',
                    pinned=pin,
                    timestamp=datetime.utcnow()
                )
                
                # Pin to remote services if requested
                if pin:
                    await self._pin_to_services(ipfs_file)
                    
                return ipfs_file
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to add JSON to IPFS: {e}")
            raise
    
    async def add_bytes(self, data: bytes, filename: str = 'data', 
                       content_type: str = 'application/octet-stream',
                       pin: bool = True) -> IPFSFile:
        """
        Add raw bytes to IPFS
        
        Args:
            data: Bytes to store
            filename: Filename for the data
            content_type: MIME type
            pin: Whether to pin the content
            
        Returns:
            IPFSFile object with CID
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        # Prepare multipart data
        form_data = aiohttp.FormData()
        form_data.add_field('file',
                           data,
                           filename=filename,
                           content_type=content_type)
        
        # Add to IPFS
        url = urljoin(self.node_url, '/api/v0/add')
        params = {'pin': str(pin).lower()}
        
        try:
            async with self.session.post(url, data=form_data, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                
                ipfs_file = IPFSFile(
                    cid=result['Hash'],
                    name=filename,
                    size=len(data),
                    type=content_type,
                    pinned=pin,
                    timestamp=datetime.utcnow()
                )
                
                # Pin to remote services if requested
                if pin:
                    await self._pin_to_services(ipfs_file)
                    
                return ipfs_file
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to add bytes to IPFS: {e}")
            raise
    
    async def get_file(self, cid: str) -> bytes:
        """
        Retrieve file from IPFS by CID
        
        Args:
            cid: Content Identifier
            
        Returns:
            File content as bytes
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        # Try local node first
        url = urljoin(self.node_url, f'/api/v0/cat?arg={cid}')
        
        try:
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
                
        except aiohttp.ClientError:
            # Fallback to public gateway
            gateway_url = f'{self.gateway_url}/ipfs/{cid}'
            
            try:
                async with self.session.get(gateway_url) as resp:
                    resp.raise_for_status()
                    return await resp.read()
                    
            except aiohttp.ClientError as e:
                logger.error(f"Failed to retrieve file from IPFS: {e}")
                raise
    
    async def get_json(self, cid: str) -> Dict:
        """
        Retrieve JSON data from IPFS
        
        Args:
            cid: Content Identifier
            
        Returns:
            Parsed JSON data
        """
        content = await self.get_file(cid)
        return json.loads(content.decode('utf-8'))
    
    async def pin(self, cid: str, recursive: bool = True) -> bool:
        """
        Pin content to prevent garbage collection
        
        Args:
            cid: Content Identifier to pin
            recursive: Pin recursively for directories
            
        Returns:
            Success status
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        url = urljoin(self.node_url, '/api/v0/pin/add')
        params = {
            'arg': cid,
            'recursive': str(recursive).lower()
        }
        
        try:
            async with self.session.post(url, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                
                # Also pin to remote services
                ipfs_file = IPFSFile(cid=cid, pinned=True)
                await self._pin_to_services(ipfs_file)
                
                return True
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to pin content: {e}")
            return False
    
    async def unpin(self, cid: str, recursive: bool = True) -> bool:
        """
        Unpin content to allow garbage collection
        
        Args:
            cid: Content Identifier to unpin
            recursive: Unpin recursively for directories
            
        Returns:
            Success status
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        url = urljoin(self.node_url, '/api/v0/pin/rm')
        params = {
            'arg': cid,
            'recursive': str(recursive).lower()
        }
        
        try:
            async with self.session.post(url, params=params) as resp:
                resp.raise_for_status()
                return True
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to unpin content: {e}")
            return False
    
    async def list_pins(self, type: str = 'all') -> List[str]:
        """
        List pinned content
        
        Args:
            type: Pin type ('all', 'direct', 'indirect', 'recursive')
            
        Returns:
            List of pinned CIDs
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        url = urljoin(self.node_url, '/api/v0/pin/ls')
        params = {'type': type}
        
        try:
            async with self.session.get(url, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                
                return list(result.get('Keys', {}).keys())
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to list pins: {e}")
            return []
    
    async def dag_put(self, data: Dict) -> str:
        """
        Store data as IPLD DAG
        
        Args:
            data: Dictionary to store as DAG
            
        Returns:
            CID of the DAG
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        url = urljoin(self.node_url, '/api/v0/dag/put')
        
        # Prepare data
        json_data = json.dumps(data)
        
        try:
            async with self.session.post(url, data=json_data) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result['Cid']['/']
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to store DAG: {e}")
            raise
    
    async def dag_get(self, cid: str, path: str = '') -> Dict:
        """
        Retrieve data from IPLD DAG
        
        Args:
            cid: CID of the DAG
            path: Optional path within the DAG
            
        Returns:
            Retrieved data
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager")
            
        url = urljoin(self.node_url, '/api/v0/dag/get')
        params = {'arg': f'{cid}{path}'}
        
        try:
            async with self.session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to retrieve DAG: {e}")
            raise
    
    async def _pin_to_services(self, ipfs_file: IPFSFile):
        """Pin content to remote pinning services"""
        for service in self.pinning_services:
            try:
                await self._pin_to_service(ipfs_file, service)
            except Exception as e:
                logger.error(f"Failed to pin to {service.name}: {e}")
    
    async def _pin_to_service(self, ipfs_file: IPFSFile, service: PinningService):
        """Pin to specific service"""
        if service.name == 'pinata':
            await self._pin_to_pinata(ipfs_file, service)
        elif service.name == 'infura':
            await self._pin_to_infura(ipfs_file, service)
        else:
            logger.warning(f"Unknown pinning service: {service.name}")
    
    async def _pin_to_pinata(self, ipfs_file: IPFSFile, service: PinningService):
        """Pin to Pinata"""
        url = f"{service.endpoint}/pinning/pinByHash"
        
        headers = {
            'Authorization': f'Bearer {service.access_token}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'hashToPin': ipfs_file.cid,
            'pinataMetadata': {
                'name': ipfs_file.name or 'ooblex-content',
                'keyvalues': {
                    'timestamp': ipfs_file.timestamp.isoformat() if ipfs_file.timestamp else None,
                    'service': 'ooblex'
                }
            }
        }
        
        async with self.session.post(url, headers=headers, json=data) as resp:
            resp.raise_for_status()
            logger.info(f"Pinned {ipfs_file.cid} to Pinata")
    
    async def _pin_to_infura(self, ipfs_file: IPFSFile, service: PinningService):
        """Pin to Infura"""
        url = f"{service.endpoint}/api/v0/pin/add"
        
        params = {'arg': ipfs_file.cid}
        auth = aiohttp.BasicAuth('', service.access_token)
        
        async with self.session.post(url, params=params, auth=auth) as resp:
            resp.raise_for_status()
            logger.info(f"Pinned {ipfs_file.cid} to Infura")
    
    def get_gateway_url(self, cid: str) -> str:
        """Get HTTP gateway URL for CID"""
        return f"{self.gateway_url}/ipfs/{cid}"
    
    def get_ipfs_url(self, cid: str) -> str:
        """Get IPFS protocol URL"""
        return f"ipfs://{cid}"