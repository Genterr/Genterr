# src/platform/integration/platform_integrator.py
# Created: 2025-01-29 19:13:51
# Author: Genterr

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import logging
from pathlib import Path
import json
import aiohttp
import yaml
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

class IntegrationError(Exception):
    """Base exception for integration-related errors"""
    pass

class IntegrationConfigError(IntegrationError):
    """Raised when integration configuration is invalid"""
    pass

class IntegrationConnectionError(IntegrationError):
    """Raised when connection to external service fails"""
    pass

class IntegrationStatus(Enum):
    """Status of external service integration"""
    ACTIVE = "active"         # Integration is active and working
    INACTIVE = "inactive"     # Integration is configured but not active
    ERROR = "error"          # Integration is experiencing errors
    MAINTENANCE = "maintenance"  # Integration is under maintenance
    DEPRECATED = "deprecated"    # Integration is marked for removal

@dataclass
class IntegrationConfig:
    """Configuration for external service integration"""
    config_path: Path
    backup_path: Path
    max_retries: int = 3
    timeout: int = 30
    max_workers: int = 4
    enable_ssl_verify: bool = True
    debug_mode: bool = False

@dataclass
class ServiceCredentials:
    """Credentials for external service authentication"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None

class PlatformIntegrator:
    """
    Manages integrations with external platforms and services.
    
    This class handles:
    - Service configuration management
    - Authentication and authorization
    - API request handling
    - Rate limiting and quotas
    - Error handling and retry logic
    - Monitoring and logging
    """

    def __init__(self, config: IntegrationConfig):
        """Initialize PlatformIntegrator with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._session = None
        self._services: Dict[str, Dict[str, Any]] = {}
        self._credentials: Dict[str, ServiceCredentials] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self._load_configurations()

    def _setup_logging(self) -> None:
        """Configure logging for integration management"""
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug_mode else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'platform_integrator.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _load_configurations(self) -> None:
        """Load service configurations from yaml files"""
        try:
            config_file = self.config.config_path / 'services.yaml'
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._services = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configurations: {str(e)}")
            raise IntegrationConfigError(f"Configuration loading failed: {str(e)}")

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is created"""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close all connections and cleanup"""
        if self._session:
            await self._session.close()
        self._executor.shutdown(wait=True)

    async def integrate_service(
        self,
        service_name: str,
        credentials: ServiceCredentials,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Integrate a new external service"""
        if service_name in self._services:
            raise IntegrationError(f"Service {service_name} already integrated")
            
        # Validate credentials
        if not self._validate_credentials(service_name, credentials):
            raise IntegrationConfigError(
                f"Invalid credentials for service {service_name}"
            )
            
        # Store service configuration
        self._services[service_name] = {
            "status": IntegrationStatus.INACTIVE.value,
            "config": config or {},
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Store credentials securely
        self._credentials[service_name] = credentials
        
        # Save updated configurations
        await self._save_configurations()
        
        logger.info(f"Successfully integrated service: {service_name}")

    def _validate_credentials(
        self,
        service_name: str,
        credentials: ServiceCredentials
    ) -> bool:
        """Validate service credentials"""
        # Basic validation - should be extended based on service requirements
        if not any([
            credentials.api_key,
            credentials.api_secret,
            credentials.access_token
        ]):
            return False
            
        return True

    async def _save_configurations(self) -> None:
        """Save current service configurations"""
        try:
            config_file = self.config.config_path / 'services.yaml'
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._services, f)
        except Exception as e:
            logger.error(f"Failed to save configurations: {str(e)}")
            raise IntegrationConfigError(f"Configuration saving failed: {str(e)}")

    async def get_service_status(self, service_name: str) -> IntegrationStatus:
        """Get current status of integrated service"""
        if service_name not in self._services:
            raise IntegrationError(f"Service {service_name} not found")
            
        return IntegrationStatus(self._services[service_name]["status"])

    async def update_service_status(
        self,
        service_name: str,
        status: IntegrationStatus
    ) -> None:
        """Update status of integrated service"""
        if service_name not in self._services:
            raise IntegrationError(f"Service {service_name} not found")
            
        self._services[service_name]["status"] = status.value
        self._services[service_name]["last_updated"] = datetime.utcnow().isoformat()
        
        await self._save_configurations()
        
        logger.info(f"Updated {service_name} status to: {status.value}")
