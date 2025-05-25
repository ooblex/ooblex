"""
Ooblex API Gateway - Modern FastAPI implementation
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis.asyncio as redis
from aio_pika import connect_robust, Message, DeliveryMode
import json
import aiofiles
import hashlib
from pathlib import Path

from config import settings
from logger import setup_logger

# Import blockchain service
import sys
sys.path.append('/mnt/c/Users/steve/Code/claude/ooblex')
from services.blockchain.blockchain_service import (
    BlockchainService, ContentMetadata, VerificationResult, BlockchainNetwork
)
from services.blockchain.ipfs_client import IPFSClient

# Setup logging
logger = setup_logger(__name__)

# Metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
websocket_connections = Counter('websocket_connections_total', 'Total WebSocket connections')
active_connections = Counter('websocket_active_connections', 'Active WebSocket connections')

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global connections
redis_client: Optional[redis.Redis] = None
rabbitmq_connection = None
rabbitmq_channel = None
blockchain_service: Optional[BlockchainService] = None
ipfs_client: Optional[IPFSClient] = None


# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class ProcessRequest(BaseModel):
    stream_token: str
    process_type: str = Field(..., regex="^(face_swap|style_transfer|object_detection|background_removal)$")
    model_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


class ProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]


class ContentRegistrationRequest(BaseModel):
    content_hash: str
    content_type: str = Field(..., regex="^(video|frame|processed)$")
    creator: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    ai_processing: Optional[List[str]] = None
    parent_hash: Optional[str] = None
    network: Optional[str] = "polygon"


class ContentRegistrationResponse(BaseModel):
    tx_hash: str
    block_number: int
    network: str
    contract_address: str
    ipfs_cid: Optional[str] = None
    timestamp: datetime


class ContentVerificationRequest(BaseModel):
    content_hash: Optional[str] = None
    ipfs_cid: Optional[str] = None
    content_data: Optional[str] = None  # Base64 encoded content


class ContentVerificationResponse(BaseModel):
    is_authentic: bool
    confidence_score: float
    tampering_detected: bool
    watermark_valid: bool
    blockchain_record: Optional[Dict] = None
    chain_of_custody: Optional[List[Dict]] = None
    message: str


class ProvenanceRequest(BaseModel):
    content_hash: str
    network: Optional[str] = "polygon"


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, rabbitmq_connection, rabbitmq_channel, blockchain_service, ipfs_client
    
    logger.info("Starting Ooblex API Gateway")
    
    # Initialize Redis
    try:
        redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    
    # Initialize RabbitMQ
    try:
        rabbitmq_connection = await connect_robust(settings.RABBITMQ_URL)
        rabbitmq_channel = await rabbitmq_connection.channel()
        await rabbitmq_channel.declare_queue("tasks", durable=True)
        logger.info("Connected to RabbitMQ")
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        raise
    
    # Initialize Blockchain Service
    try:
        blockchain_config = {
            'contract_addresses': getattr(settings, 'BLOCKCHAIN_CONTRACTS', {}),
            'private_key': getattr(settings, 'BLOCKCHAIN_PRIVATE_KEY', None)
        }
        blockchain_service = BlockchainService(blockchain_config)
        logger.info("Initialized Blockchain Service")
    except Exception as e:
        logger.error(f"Failed to initialize Blockchain Service: {e}")
        # Non-critical, continue without blockchain
    
    # Initialize IPFS Client
    try:
        ipfs_config = {
            'node_url': getattr(settings, 'IPFS_NODE_URL', 'http://localhost:5001'),
            'gateway_url': getattr(settings, 'IPFS_GATEWAY_URL', 'https://ipfs.io'),
            'pinning_services': []
        }
        ipfs_client = IPFSClient(ipfs_config)
        logger.info("Initialized IPFS Client")
    except Exception as e:
        logger.error(f"Failed to initialize IPFS Client: {e}")
        # Non-critical, continue without IPFS
    
    yield
    
    # Cleanup
    logger.info("Shutting down Ooblex API Gateway")
    if redis_client:
        await redis_client.close()
    if rabbitmq_connection:
        await rabbitmq_connection.close()


# Initialize FastAPI app
app = FastAPI(
    title="Ooblex API",
    description="Real-time AI video processing platform",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user(username: str):
    """Get user from database"""
    # TODO: Implement database lookup
    # For now, using a mock user
    if username == "demo":
        return UserInDB(
            username="demo",
            email="demo@ooblex.com",
            full_name="Demo User",
            hashed_password=get_password_hash("demo123"),
            disabled=False,
        )
    return None


async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Ooblex API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check Redis
    try:
        await redis_client.ping()
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"
    
    # Check RabbitMQ
    try:
        if rabbitmq_channel and not rabbitmq_channel.is_closed:
            services["rabbitmq"] = "healthy"
        else:
            services["rabbitmq"] = "unhealthy"
    except Exception:
        services["rabbitmq"] = "unhealthy"
    
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services
    )


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return current_user


@app.post("/process", response_model=ProcessResponse)
async def start_process(
    request: ProcessRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Start a new processing task"""
    # Generate task ID
    task_id = f"{request.stream_token}:{request.process_type}:{datetime.utcnow().timestamp()}"
    
    # Create task message
    task_data = {
        "task_id": task_id,
        "stream_token": request.stream_token,
        "process_type": request.process_type,
        "model_name": request.model_name,
        "parameters": request.parameters,
        "user": current_user.username,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Send to RabbitMQ
    try:
        message = Message(
            body=json.dumps(task_data).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json"
        )
        await rabbitmq_channel.default_exchange.publish(
            message,
            routing_key="tasks"
        )
        
        # Store task status in Redis
        await redis_client.setex(
            f"task:{task_id}",
            3600,  # 1 hour TTL
            json.dumps({"status": "queued", "created_at": datetime.utcnow().isoformat()})
        )
        
        logger.info(f"Task {task_id} queued for processing")
        
        return ProcessResponse(
            task_id=task_id,
            status="queued",
            message="Task queued for processing"
        )
    except Exception as e:
        logger.error(f"Failed to queue task: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue task")


@app.get("/process/{task_id}", response_model=ProcessResponse)
async def get_process_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get processing task status"""
    task_data = await redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = json.loads(task_data)
    return ProcessResponse(
        task_id=task_id,
        status=task_info.get("status", "unknown"),
        message=task_info.get("message", "")
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


# Blockchain endpoints
@app.post("/blockchain/register", response_model=ContentRegistrationResponse)
async def register_content(
    request: ContentRegistrationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Register content on blockchain for authenticity verification"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Create content metadata
        metadata = ContentMetadata(
            content_hash=request.content_hash,
            timestamp=int(datetime.utcnow().timestamp()),
            creator=request.creator or current_user.username,
            device_id=request.device_id,
            location=request.location,
            ai_processing=request.ai_processing,
            parent_hash=request.parent_hash
        )
        
        # Register on blockchain
        network = BlockchainNetwork(request.network)
        tx_result = await blockchain_service.register_content(metadata, network)
        
        # Store metadata in IPFS if available
        ipfs_cid = None
        if ipfs_client:
            try:
                async with ipfs_client as client:
                    ipfs_file = await client.add_json({
                        "content_hash": request.content_hash,
                        "metadata": metadata.__dict__,
                        "blockchain_tx": tx_result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    ipfs_cid = ipfs_file.cid
            except Exception as e:
                logger.error(f"Failed to store in IPFS: {e}")
        
        # Track metrics
        request_count.labels(method="POST", endpoint="/blockchain/register", status="success").inc()
        
        return ContentRegistrationResponse(
            tx_hash=tx_result['tx_hash'],
            block_number=tx_result['block_number'],
            network=tx_result['network'],
            contract_address=tx_result['contract_address'],
            ipfs_cid=ipfs_cid,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        request_count.labels(method="POST", endpoint="/blockchain/register", status="error").inc()
        logger.error(f"Failed to register content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blockchain/verify", response_model=ContentVerificationResponse)
async def verify_content(
    request: ContentVerificationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Verify content authenticity using blockchain"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Get content data
        content_data = None
        expected_hash = request.content_hash
        
        if request.content_data:
            # Decode base64 content
            import base64
            content_data = base64.b64decode(request.content_data)
            
        elif request.ipfs_cid and ipfs_client:
            # Retrieve from IPFS
            try:
                async with ipfs_client as client:
                    content_data = await client.get_file(request.ipfs_cid)
            except Exception as e:
                logger.error(f"Failed to retrieve from IPFS: {e}")
                
        elif request.content_hash:
            # Just verify the hash exists on blockchain
            content_data = request.content_hash.encode()
            
        else:
            raise HTTPException(status_code=400, detail="No content provided for verification")
        
        # Verify content
        result = await blockchain_service.verify_content(content_data, expected_hash)
        
        # Prepare response message
        if result.is_authentic:
            message = f"Content verified with {result.confidence_score:.1%} confidence"
        else:
            message = "Content could not be verified"
            if result.tampering_detected:
                message += " - tampering detected"
        
        # Track metrics
        request_count.labels(method="POST", endpoint="/blockchain/verify", status="success").inc()
        
        return ContentVerificationResponse(
            is_authentic=result.is_authentic,
            confidence_score=result.confidence_score,
            tampering_detected=result.tampering_detected,
            watermark_valid=result.watermark_valid,
            blockchain_record=result.blockchain_record,
            chain_of_custody=result.chain_of_custody,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_count.labels(method="POST", endpoint="/blockchain/verify", status="error").inc()
        logger.error(f"Failed to verify content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/blockchain/provenance/{content_hash}")
async def get_provenance(
    content_hash: str,
    network: Optional[str] = "polygon",
    current_user: User = Depends(get_current_active_user)
):
    """Get complete provenance chain for content"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Get chain of custody
        chain_of_custody = await blockchain_service._get_chain_of_custody(network, content_hash)
        
        if not chain_of_custody:
            raise HTTPException(status_code=404, detail="Content not found on blockchain")
        
        # Track metrics
        request_count.labels(method="GET", endpoint="/blockchain/provenance", status="success").inc()
        
        return {
            "content_hash": content_hash,
            "network": network,
            "chain_of_custody": chain_of_custody,
            "total_events": len(chain_of_custody)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        request_count.labels(method="GET", endpoint="/blockchain/provenance", status="error").inc()
        logger.error(f"Failed to get provenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blockchain/fingerprint")
async def generate_fingerprint(
    video_path: str,
    sample_rate: Optional[int] = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Generate content fingerprint for video"""
    if not blockchain_service:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Validate file exists
        if not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Generate fingerprint
        fingerprint = blockchain_service.generate_content_fingerprint(video_path, sample_rate)
        
        # Track metrics
        request_count.labels(method="POST", endpoint="/blockchain/fingerprint", status="success").inc()
        
        return fingerprint
        
    except HTTPException:
        raise
    except Exception as e:
        request_count.labels(method="POST", endpoint="/blockchain/fingerprint", status="error").inc()
        logger.error(f"Failed to generate fingerprint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        active_connections.inc()
        websocket_connections.inc()
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            active_connections.dec()
            logger.info(f"Client {client_id} disconnected")
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Parse message
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    await manager.send_personal_message(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                        client_id
                    )
                elif message_type == "subscribe":
                    # Subscribe to specific events
                    stream_token = message.get("stream_token")
                    await redis_client.sadd(f"subscribers:{stream_token}", client_id)
                    await manager.send_personal_message(
                        json.dumps({"type": "subscribed", "stream_token": stream_token}),
                        client_id
                    )
                else:
                    # Echo back for now
                    await manager.send_personal_message(data, client_id)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON"}),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Clean up subscriptions
        # TODO: Remove client from all Redis subscription sets


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)