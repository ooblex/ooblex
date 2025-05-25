"""
Global pytest configuration and fixtures
"""
import asyncio
import os
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.rabbitmq import RabbitMQContainer
import redis.asyncio as redis
from aio_pika import connect_robust

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from api.main import app
from api.config import settings


# Override event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for testing"""
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def redis_container():
    """Start Redis container for testing"""
    with RedisContainer("redis:7-alpine") as redis_cont:
        yield redis_cont


@pytest.fixture(scope="session")
def rabbitmq_container():
    """Start RabbitMQ container for testing"""
    with RabbitMQContainer("rabbitmq:3.12-management-alpine") as rabbitmq:
        yield rabbitmq


@pytest_asyncio.fixture(scope="session")
async def test_db(postgres_container):
    """Create test database"""
    # Update settings with test database URL
    settings.database_url = postgres_container.get_connection_url().replace("psycopg2", "asyncpg")
    
    # Create engine and tables
    engine = create_async_engine(settings.database_url)
    
    # Import models and create tables
    from api.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    async_session = async_sessionmaker(
        test_db,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="session")
async def redis_client(redis_container):
    """Create Redis client for testing"""
    settings.redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}"
    
    client = await redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    
    yield client
    
    await client.close()


@pytest_asyncio.fixture(scope="session")
async def rabbitmq_connection(rabbitmq_container):
    """Create RabbitMQ connection for testing"""
    settings.rabbitmq_url = f"amqp://guest:guest@localhost:{rabbitmq_container.get_exposed_port(5672)}"
    
    connection = await connect_robust(settings.rabbitmq_url)
    
    yield connection
    
    await connection.close()


@pytest_asyncio.fixture
async def api_client(test_db, redis_client, rabbitmq_connection) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for API"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_ml_model(mocker):
    """Mock ML model for testing"""
    mock_model = mocker.MagicMock()
    mock_model.run.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return mock_model


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    import numpy as np
    import cv2
    
    # Create a simple 100x100 RGB image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some patterns
    cv2.rectangle(image, (20, 20), (80, 80), (255, 0, 0), -1)
    cv2.circle(image, (50, 50), 20, (0, 255, 0), -1)
    
    return image


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing"""
    from api.main import create_access_token
    
    token = create_access_token(data={"sub": "testuser"})
    return {"Authorization": f"Bearer {token}"}


# Test data fixtures
@pytest.fixture
def user_data():
    """Sample user data"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpass123"
    }


@pytest.fixture
def process_request_data():
    """Sample process request data"""
    return {
        "stream_token": "test_token_123",
        "process_type": "face_swap",
        "model_name": "face_swap_v1",
        "parameters": {
            "confidence_threshold": 0.5
        }
    }