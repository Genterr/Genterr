"""Test module for the API client."""
# src/tests/test_api_client.py
# Created: 2025-02-01 22:13:07
# Author: Genterr

import pytest
import aiohttp
import asyncio
from datetime import datetime, UTC
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any
from pathlib import Path
import json
from src.utils.api.api_client import (
    APIClient, 
    APIConfig, 
    APIResponse,
    SecurityProvider,
    RequestMethod,
    RequestError,
    APIError,
    AuthenticationError
)

class MockSecurityProvider(SecurityProvider):
    """Mock security provider for testing."""
    async def get_auth_header(self) -> Dict[str, str]:
        return {"Authorization": "Bearer test-token"}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def api_config():
    """Fixture for API configuration"""
    return APIConfig(
        base_url="https://api.example.com",
        timeout=5.0,
        max_retries=2,
        retry_delay=0.1,
        rate_limit=10,
        enable_caching=True,
        cache_ttl=1
    )

@pytest.fixture
def security_provider():
    """Create mock security provider."""
    return MockSecurityProvider()

@pytest.fixture
async def api_client(api_config, security_provider):
    """Fixture for API client"""
    client = APIClient(config=api_config, security=security_provider)
    yield client
    await client.close()

@pytest.mark.asyncio
class TestAPIClient:
    """Test cases for APIClient class."""

    async def test_initialization(self, api_client, api_config):
        """Test client initialization."""
        assert api_client.config == api_config
        assert api_client._session is None
        assert api_client._rate_limit_tokens == api_config.rate_limit

    async def test_get_session(self, api_client):
        """Test session creation and reuse"""
        session1 = await api_client._get_session()
        assert isinstance(session1, aiohttp.ClientSession)
        
        session2 = await api_client._get_session()
        assert session1 is session2

        await api_client.close()

    async def test_rate_limiting(self, api_client):
        """Test rate limiting functionality"""
        for _ in range(api_client.config.rate_limit):
            assert await api_client._check_rate_limit() is True
        assert await api_client._check_rate_limit() is False

    async def test_successful_get_request(self, api_client):
        """Test successful GET request"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "success"})
        mock_response.headers = {"Content-Type": "application/json"}

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        with patch.object(aiohttp.ClientSession, 'request', return_value=mock_cm):
            response = await api_client.get("/test")
            
            assert isinstance(response, APIResponse)
            assert response.status == 200
            assert response.data == {"message": "success"}
            assert "Content-Type" in response.headers

    async def test_failed_request_retry(self, api_client):
        """Test request retry on failure"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "success"})
        mock_response.headers = {"Content-Type": "application/json"}

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.request.side_effect = [
            aiohttp.ClientError(),
            aiohttp.ClientError(),
            mock_cm
        ]

        with patch('aiohttp.ClientSession', return_value=mock_session):
            response = await api_client.get("/test")
            assert response.status == 200
            assert mock_session.request.call_count == 3

    async def test_request_max_retries_exceeded(self, api_client):
        """Test exception when max retries exceeded"""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.request.side_effect = aiohttp.ClientError()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RequestError):
                await api_client.get("/test")
            assert mock_session.request.call_count == api_client.config.max_retries + 1

    async def test_cache_functionality(self, api_client):
        """Test response caching"""
        calls = []
        
        async def mock_response():
            return {
                "message": "success"
            }

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value.status = 200
        mock_context.__aenter__.return_value.json = mock_response
        mock_context.__aenter__.return_value.headers = {"Content-Type": "application/json"}
        
        def record_call(*args, **kwargs):
            calls.append((args, kwargs))
            return mock_context
        
        with patch.object(aiohttp.ClientSession, 'request', side_effect=record_call):
            # First request should hit the API
            response1 = await api_client.get("/test", use_cache=True)
            
            # Second request should use cache
            response2 = await api_client.get("/test", use_cache=True)
            
            # Verify results
            assert len(calls) == 1  # API should only be called once
            assert response1.data == {"message": "success"}
            assert response2.data == {"message": "success"}
            assert response1.status == 200
            assert response2.status == 200
            assert "Content-Type" in response1.headers
            assert "Content-Type" in response2.headers

    async def test_authentication(self, api_client):
        """Test authentication header injection"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "success"})
        mock_response.headers = {"Content-Type": "application/json"}

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        with patch.object(aiohttp.ClientSession, 'request', return_value=mock_cm) as mock_request:
            await api_client.get("/test")
            
            # Verify auth header was included
            call_kwargs = mock_request.call_args[1]
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"

    def test_cache_key_generation(self, api_client):
        """Test cache key generation"""
        key1 = api_client._get_cache_key(
            RequestMethod.GET,
            "https://api.example.com/test",
            {"param": "value"}
        )
        key2 = api_client._get_cache_key(
            RequestMethod.GET,
            "https://api.example.com/test",
            {"param": "value"}
        )
        assert key1 == key2
        
        key3 = api_client._get_cache_key(
            RequestMethod.POST,
            "https://api.example.com/test",
            {"param": "value"}
        )
        assert key1 != key3

    async def test_request_methods(self, api_client):
        """Test all HTTP methods"""
        methods = [
            (api_client.get, "/test"),
            (api_client.post, "/test", {"data": "test"}),
            (api_client.put, "/test", {"data": "test"}),
            (api_client.patch, "/test", {"data": "test"}),
            (api_client.delete, "/test")
        ]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "success"})
        mock_response.headers = {"Content-Type": "application/json"}

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        with patch.object(aiohttp.ClientSession, 'request', return_value=mock_cm):
            for method_info in methods:
                method = method_info[0]
                args = method_info[1:]
                response = await method(*args)
                assert isinstance(response, APIResponse)
                assert response.status == 200

if __name__ == "__main__":
    pytest.main(["-v"])