"""
gRPC server implementation for API service
"""
import asyncio
from concurrent import futures
from datetime import datetime, timedelta
import logging
from typing import AsyncIterator

import grpc
from grpc_reflection.v1alpha import reflection
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp

# Import generated protobuf files
import ooblex_pb2
import ooblex_pb2_grpc

from config import settings
from logger import setup_logger
from main import (
    authenticate_user,
    create_access_token,
    get_user,
    get_password_hash
)

logger = setup_logger(__name__)


class APIServicer(ooblex_pb2_grpc.APIServiceServicer):
    """gRPC API service implementation"""
    
    def __init__(self, redis_client, rabbitmq_channel):
        self.redis_client = redis_client
        self.rabbitmq_channel = rabbitmq_channel
        self.active_streams = {}
    
    async def Login(self, request, context):
        """Handle login request"""
        user = await authenticate_user(request.username, request.password)
        
        if not user:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid credentials")
        
        # Create tokens
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        refresh_token = create_access_token(
            data={"sub": user.username, "type": "refresh"},
            expires_delta=timedelta(days=7)
        )
        
        # Create response
        response = ooblex_pb2.LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
            user=ooblex_pb2.User(
                id=str(user.id) if hasattr(user, 'id') else user.username,
                username=user.username,
                email=user.email or "",
                full_name=user.full_name or "",
                disabled=user.disabled or False
            )
        )
        
        return response
    
    async def RefreshToken(self, request, context):
        """Handle token refresh"""
        try:
            # Verify refresh token
            payload = jwt.decode(
                request.refresh_token, 
                settings.jwt_secret, 
                algorithms=[settings.jwt_algorithm]
            )
            
            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            
            username = payload.get("sub")
            if not username:
                raise ValueError("Invalid token")
            
            # Create new access token
            access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
            access_token = create_access_token(
                data={"sub": username}, 
                expires_delta=access_token_expires
            )
            
            return ooblex_pb2.RefreshTokenResponse(
                access_token=access_token,
                expires_in=settings.access_token_expire_minutes * 60
            )
            
        except Exception as e:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(e))
    
    async def Logout(self, request, context):
        """Handle logout"""
        # TODO: Implement token blacklisting
        return Empty()
    
    async def GetUser(self, request, context):
        """Get user information"""
        user = await get_user(request.user_id)
        
        if not user:
            context.abort(grpc.StatusCode.NOT_FOUND, "User not found")
        
        return ooblex_pb2.User(
            id=str(user.id) if hasattr(user, 'id') else user.username,
            username=user.username,
            email=user.email or "",
            full_name=user.full_name or "",
            disabled=user.disabled or False
        )
    
    async def CreateTask(self, request, context):
        """Create processing task"""
        task_id = f"{request.stream_token}:{request.process_type}:{datetime.utcnow().timestamp()}"
        
        # Create task
        task = ooblex_pb2.ProcessingTask(
            task_id=task_id,
            stream_token=request.stream_token,
            process_type=request.process_type,
            parameters=request.parameters,
            status=ooblex_pb2.TASK_STATUS_QUEUED
        )
        
        # Set timestamps
        now = Timestamp()
        now.GetCurrentTime()
        task.created_at.CopyFrom(now)
        task.updated_at.CopyFrom(now)
        
        # Store in Redis
        await self.redis_client.setex(
            f"task:{task_id}",
            3600,
            task.SerializeToString()
        )
        
        # Send to RabbitMQ
        await self.rabbitmq_channel.default_exchange.publish(
            Message(body=task.SerializeToString()),
            routing_key="tasks"
        )
        
        return task
    
    async def GetTask(self, request, context):
        """Get task status"""
        task_data = await self.redis_client.get(f"task:{request.task_id}")
        
        if not task_data:
            context.abort(grpc.StatusCode.NOT_FOUND, "Task not found")
        
        task = ooblex_pb2.ProcessingTask()
        task.ParseFromString(task_data)
        
        return task
    
    async def ListTasks(self, request, context):
        """List tasks for stream"""
        # TODO: Implement pagination
        tasks = []
        
        # Get all tasks for stream
        cursor = 0
        pattern = f"task:{request.stream_token}:*"
        
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor, 
                match=pattern, 
                count=100
            )
            
            for key in keys:
                task_data = await self.redis_client.get(key)
                if task_data:
                    task = ooblex_pb2.ProcessingTask()
                    task.ParseFromString(task_data)
                    
                    if request.status == ooblex_pb2.TASK_STATUS_UNKNOWN or task.status == request.status:
                        tasks.append(task)
            
            if cursor == 0:
                break
        
        # Sort by created_at
        tasks.sort(key=lambda t: t.created_at.seconds, reverse=True)
        
        # Pagination
        start = 0
        if request.page_token:
            start = int(request.page_token)
        
        end = start + request.page_size if request.page_size > 0 else len(tasks)
        page_tasks = tasks[start:end]
        
        response = ooblex_pb2.ListTasksResponse(
            tasks=page_tasks,
            total_count=len(tasks)
        )
        
        if end < len(tasks):
            response.next_page_token = str(end)
        
        return response
    
    async def StreamStatus(self, request, context) -> AsyncIterator[ooblex_pb2.TaskStatusUpdate]:
        """Stream task status updates"""
        stream_id = str(uuid.uuid4())
        self.active_streams[stream_id] = True
        
        try:
            # Subscribe to Redis pubsub
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(f"task_updates:{request.stream_token}")
            
            async for message in pubsub.listen():
                if not self.active_streams.get(stream_id):
                    break
                
                if message['type'] == 'message':
                    task = ooblex_pb2.ProcessingTask()
                    task.ParseFromString(message['data'])
                    
                    update = ooblex_pb2.TaskStatusUpdate(task=task)
                    update.timestamp.GetCurrentTime()
                    
                    yield update
                    
        finally:
            del self.active_streams[stream_id]
            await pubsub.unsubscribe()
            await pubsub.close()


async def serve_grpc(redis_client, rabbitmq_channel, port: int = 50051):
    """Start gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Add servicer
    servicer = APIServicer(redis_client, rabbitmq_channel)
    ooblex_pb2_grpc.add_APIServiceServicer_to_server(servicer, server)
    
    # Enable reflection
    SERVICE_NAMES = (
        ooblex_pb2.DESCRIPTOR.services_by_name['APIService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Start server
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    
    logger.info(f"gRPC server started on port {port}")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(5)


# Interceptor for authentication
class AuthInterceptor(grpc.aio.ServerInterceptor):
    """Authentication interceptor for gRPC"""
    
    def __init__(self, excluded_methods=None):
        self.excluded_methods = excluded_methods or [
            '/ooblex.APIService/Login',
            '/ooblex.APIService/RefreshToken'
        ]
    
    async def intercept_service(self, continuation, handler_call_details):
        # Skip auth for excluded methods
        if handler_call_details.method in self.excluded_methods:
            return await continuation(handler_call_details)
        
        # Check for authorization header
        metadata = dict(handler_call_details.invocation_metadata)
        auth_header = metadata.get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    'Missing authorization header'
                )
            )
        
        token = auth_header.split(' ')[1]
        
        try:
            # Verify token
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            
            # Add user info to context
            handler_call_details.invocation_metadata.append(
                ('user', payload.get('sub'))
            )
            
        except Exception as e:
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    f'Invalid token: {str(e)}'
                )
            )
        
        return await continuation(handler_call_details)