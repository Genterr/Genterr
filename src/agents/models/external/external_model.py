from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import asyncio
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass
import json
import aiohttp
import backoff
import tiktoken
from pathlib import Path
import hashlib
import ssl
import certifi
from ratelimit import limits, sleep_and_retry
import jwt
from cryptography.fernet import Fernet
import numpy as np

class ExternalModelError(Exception):
    """Base exception for external model-related errors"""
    pass

class APIError(ExternalModelError):
    """Raised when API calls fail"""
    pass

class AuthenticationError(ExternalModelError):
    """Raised when authentication fails"""
    pass

class RateLimitError(ExternalModelError):
    """Raised when rate limits are exceeded"""
    pass

class ModelNotFoundError(ExternalModelError):
    """Raised when requested model is not available"""
    pass

class ModelStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

@dataclass
class APIConfig:
    """Configuration for API connections"""
    base_url: str
    api_key: str
    api_version: str
    timeout: int = 30
    max_retries: int = 3
    rate_limit_calls: int = 60
    rate_limit_period: int = 60  # in seconds
    verify_ssl: bool = True
    proxy_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

@dataclass
class ModelConfig:
    """Configuration for external model behavior"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = None
    cache_responses: bool = True
    cache_ttl: int = 3600  # in seconds
    enable_logging: bool = True
    log_level: str = "INFO"

class ResponseCache:
    """Cache for API responses"""
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                return result
            del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, datetime.utcnow())

    def clear_expired(self) -> None:
        current_time = datetime.utcnow()
        expired_keys = [
            k for k, (_, t) in self.cache.items()
            if current_time - t >= timedelta(seconds=self.ttl)
        ]
        for k in expired_keys:
            del self.cache[k]

class TokenCounter:
    """Counts tokens for different model types"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class ExternalModel(ABC):
    """
    Base class for external AI model integrations.
    Provides common functionality for API-based model interactions.
    
    Supported Models:
    - OpenAI GPT models
    - Google PaLM
    - Anthropic Claude
    - Hugging Face models
    - Custom API-based models
    """

    def __init__(
        self,
        model_name: str,
        api_config: APIConfig,
        model_config: Optional[ModelConfig] = None,
        encryption_key: Optional[str] = None
    ):
        self.model_id: UUID = uuid4()
        self.model_name: str = model_name
        self.api_config: APIConfig = api_config
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.status: ModelStatus = ModelStatus.DISCONNECTED
        self.created_at: datetime = datetime.utcnow()
        
        # Setup encryption
        self.encryption_key = encryption_key
        self.cipher_suite = Fernet(encryption_key.encode()) if encryption_key else None
        
        # Setup logging
        self.logger = logging.getLogger(f"external_model.{model_name}")
        self.logger.setLevel(self.model_config.log_level)
        
        # Setup caching
        self.cache = ResponseCache(self.model_config.cache_ttl)
        
        # Setup token counter
        self.token_counter = TokenCounter(model_name)
        
        # Setup metrics
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "average_latency": 0.0,
            "rate_limit_hits": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_error": None,
            "last_error_time": None
        }
        
        # Setup session
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    @sleep_and_retry
    @limits(calls=60, period=60)
    async def connect(self) -> bool:
        """Establish connection to the API"""
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    base_url=self.api_config.base_url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.api_config.timeout)
                )
            
            # Test connection
            async with self._session.get("/health") as response:
                if response.status == 200:
                    self.status = ModelStatus.CONNECTED
                    self.logger.info(f"Successfully connected to {self.model_name}")
                    return True
                else:
                    self.status = ModelStatus.ERROR
                    return False
                    
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.logger.error(f"Connection failed: {str(e)}")
            raise ConnectionError(f"Failed to connect to {self.model_name}: {str(e)}")

    async def disconnect(self) -> None:
        """Close API connection"""
        if self._session:
            await self._session.close()
            self._session = None
        self.status = ModelStatus.DISCONNECTED

    def _get_headers(self) -> Dict[str, str]:
        """Prepare API headers with authentication"""
        headers = {
            "Authorization": f"Bearer {self.api_config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"GENTERR-AI/{self.model_name}"
        }
        if self.api_config.headers:
            headers.update(self.api_config.headers)
        return headers

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.cipher_suite:
            return self.cipher_suite.encrypt(data.encode()).decode()
        return data

    def _decrypt_data(self, data: str) -> str:
        """Decrypt sensitive data"""
        if self.cipher_suite:
            return self.cipher_suite.decrypt(data.encode()).decode()
        return data

    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_data = f"{prompt}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, RateLimitError),
        max_tries=3
    )
    async def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make API request with retry logic"""
        try:
            if not self._session:
                await self.connect()

            start_time = datetime.utcnow()
            
            async with self._session.request(
                method=method,
                url=endpoint,
                json=data,
                params=params,
                ssl=self._ssl_context if self.api_config.verify_ssl else False,
                proxy=self.api_config.proxy_url
            ) as response:
                
                self.metrics["total_requests"] += 1
                
                if response.status == 429:
                    self.metrics["rate_limit_hits"] += 1
                    self.status = ModelStatus.RATE_LIMITED
                    raise RateLimitError("Rate limit exceeded")
                
                response_data = await response.json()
                
                if response.status != 200:
                    self.metrics["failed_requests"] += 1
                    self.metrics["last_error"] = response_data.get("error", str(response.status))
                    self.metrics["last_error_time"] = datetime.utcnow()
                    raise APIError(f"API request failed: {response_data.get('error', 'Unknown error')}")
                
                self.metrics["successful_requests"] += 1
                
                # Update latency metrics
                latency = (datetime.utcnow() - start_time).total_seconds()
                self.metrics["average_latency"] = (
                    (self.metrics["average_latency"] * (self.metrics["successful_requests"] - 1) + latency)
                    / self.metrics["successful_requests"]
                )
                
                return response_data
                
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion from the model.
        
        Args:
            prompt: Input text
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing the model's response
        """
        raise NotImplementedError("Subclasses must implement generate method")

    @abstractmethod
    async def embed(
        self,
        text: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for the input text.
        
        Args:
            text: Input text
            **kwargs: Additional model-specific parameters
            
        Returns:
            Numpy array of embeddings
        """
        raise NotImplementedError("Subclasses must implement embed method")

    async def generate_with_cache(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with caching"""
        if not self.model_config.cache_responses:
            return await self.generate(prompt, **kwargs)
            
        cache_key = self._generate_cache_key(prompt, **kwargs)
        cached_response = self.cache.get(cache_key)
        
        if cached_response:
            self.metrics["cache_hits"] += 1
            return cached_response
            
        self.metrics["cache_misses"] += 1
        response = await self.generate(prompt, **kwargs)
        self.cache.set(cache_key, response)
        return response

    def update_api_config(self, **kwargs) -> None:
        """Update API configuration"""
        for key, value in kwargs.items():
            if hasattr(self.api_config, key):
                setattr(self.api_config, key, value)

    def update_model_config(self, **kwargs) -> None:
        """Update model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "model_id": str(self.model_id),
            "model_name": self.model_name,
            "status": self.status.value,
            "uptime": (datetime.utcnow() - self.created_at).total_seconds(),
            **self.metrics
        }

    async def health_check(self) -> bool:
        """Check if the model API is healthy"""
        try:
            async with self._session.get("/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def validate_response(
        self,
        response: Dict[str, Any],
        expected_fields: List[str]
    ) -> bool:
        """Validate API response format"""
        return all(field in response for field in expected_fields)

    def estimate_cost(
        self,
        prompt: str,
        max_tokens: int
    ) -> float:
        """Estimate cost of the API call based on token count"""
        input_tokens = self.token_counter.count_tokens(prompt)
        # Implementation depends on specific model pricing
        return 0.0

    async def backup_configuration(self, path: Path) -> bool:
        """Backup API and model configuration"""
        try:
            config_data = {
                "api_config": {
                    k: v for k, v in vars(self.api_config).items()
                    if k != "api_key"  # Don't backup sensitive data
                },
                "model_config": vars(self.model_config),
                "metrics": self.metrics,
                "created_at": self.created_at.isoformat()
            }
            
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            return False

    def __repr__(self) -> str:
        return f"ExternalModel(name='{self.model_name}', id={self.model_id}, status={self.status})"