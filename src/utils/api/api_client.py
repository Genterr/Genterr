# src/utils/api/api_client.py
# Created: 2025-02-01 22:05:50
# Author: Genterr

from typing import Dict, Any, Optional, List, Union, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from pathlib import Path
import json
import aiohttp
import yarl
from asyncio import Lock
from datetime import datetime, timedelta, UTC

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base exception for API-related errors"""
    pass

class RequestError(APIError):
    """Raised when an API request fails"""
    pass

class ResponseError(APIError):
    """Raised when processing an API response fails"""
    pass

class AuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass

class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

@dataclass
class APIConfig:
    """Configuration for API client"""
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    user_agent: str = "Genterr/1.0"
    rate_limit: int = 100  # requests per minute
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    max_connections: int = 100
    connection_timeout: float = 10.0

class SecurityProvider(Protocol):
    """Protocol for security providers"""
    async def get_auth_header(self) -> Dict[str, str]:
        """Get authentication header"""
        ...

@dataclass
class APIResponse:
    """Container for API response data"""
    status: int
    data: Any
    headers: Dict[str, str]
    timestamp: datetime
    duration: float

class APIClient:
    """
    Handles API requests and responses.
    
    This class provides:
    - Asynchronous HTTP requests
    - Automatic retry logic
    - Rate limiting
    - Response caching
    - Error handling
    - Authentication management
    - Request/Response logging
    """

    def __init__(
        self,
        config: APIConfig,
        security: Optional[SecurityProvider] = None
    ):
        self.config = config
        self.security = security
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_tokens = self.config.rate_limit
        self._last_token_refresh = datetime.now(UTC)
        self._cache: Dict[str, Tuple[APIResponse, datetime]] = {}
        self._rate_limit_lock = Lock()
        self._rate_limit_queue: List[datetime] = []

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure API-related logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('api.log'),
                logging.StreamHandler()
            ]
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={"User-Agent": self.config.user_agent}
            )
        return self._session

    async def close(self) -> None:
        """Close the API client session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _check_rate_limit(self) -> bool:
        """
        Check if request is within rate limits using sliding window algorithm
        """
        async with self._rate_limit_lock:
            now = datetime.now(UTC)
            window_start = now - timedelta(minutes=1)
        
            # Remove expired timestamps
            self._rate_limit_queue = [
                ts for ts in self._rate_limit_queue 
                if ts > window_start
            ]
            
            # Check if we're within limit
            if len(self._rate_limit_queue) >= self.config.rate_limit:
                return False
                
            # Add new timestamp
            self._rate_limit_queue.append(now)
            return True

    def _get_cache_key(
        self,
        method: RequestMethod,
        url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for request"""
        key_parts = [method.value, url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        return ":".join(key_parts)

    async def request(
        self,
        method: RequestMethod,
        endpoint: str,
        params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True
    ) -> APIResponse:
        """
        Make an API request
        
        Args:
            method: HTTP method to use
            endpoint: API endpoint to call
            params: Query parameters
            data: Request body data
            headers: Additional headers
            use_cache: Whether to use response caching
            
        Returns:
            APIResponse object containing response data
        """
        if not await self._check_rate_limit():
            raise RequestError("Rate limit exceeded")
            
        url = yarl.URL(self.config.base_url) / endpoint.lstrip('/')
        cache_key = self._get_cache_key(method, str(url), params)
        
        # Check cache
        if use_cache and self.config.enable_caching:
            cached = self._cache.get(cache_key)
            if cached and (datetime.now(UTC) - cached[1]).total_seconds() < self.config.cache_ttl:
                return cached[0]
        
        # Prepare request
        session = await self._get_session()
        request_headers = headers or {}
        if self.security:
            # Add authentication if available
            auth_header = await self.security.get_auth_header()
            request_headers.update(auth_header)
            
        retries = 0
        while retries <= self.config.max_retries:
            try:
                start_time = datetime.now(UTC)
                
                async with session.request(
                    method.value,
                    url,
                    params=params,
                    json=data,
                    headers=request_headers,
                    ssl=self.config.verify_ssl
                ) as response:
                    duration = (datetime.now(UTC) - start_time).total_seconds()
                    
                    # Check for error status codes
                    if response.status >= 400:
                        error_data = await response.json()
                        raise RequestError(f"API request failed: {response.status} - {error_data}")
                        
                    response_data = await response.json()
                    api_response = APIResponse(
                        status=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        timestamp=datetime.now(UTC),
                        duration=duration
                    )
                    
                    # Update cache
                    if use_cache and self.config.enable_caching:
                        self._cache[cache_key] = (api_response, datetime.now(UTC))
                        
                    return api_response
                    
            except Exception as e:
                logger.error(f"API request failed: {str(e)}")
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * retries)
                else:
                    raise RequestError(f"Max retries exceeded: {str(e)}")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        **kwargs: Any
    ) -> APIResponse:
        """Perform GET request"""
        return await self.request(RequestMethod.GET, endpoint, params=params, **kwargs)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> APIResponse:
        """Perform POST request"""
        return await self.request(RequestMethod.POST, endpoint, data=data, **kwargs)

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> APIResponse:
        """Perform PUT request"""
        return await self.request(RequestMethod.PUT, endpoint, data=data, **kwargs)

    async def delete(
        self,
        endpoint: str,
        **kwargs: Any
    ) -> APIResponse:
        """Perform DELETE request"""
        return await self.request(RequestMethod.DELETE, endpoint, **kwargs)

    async def patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> APIResponse:
        """Perform PATCH request"""
        return await self.request(RequestMethod.PATCH, endpoint, data=data, **kwargs)