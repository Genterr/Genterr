# src/utils/api/response_handler.py
# Created: 2025-01-29 20:59:38
# Author: Genterr

from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from .api_client import APIResponse, APIError

T = TypeVar('T')
logger = logging.getLogger(__name__)

class ResponseHandlerError(APIError):
    """Raised when response handling fails"""
    pass

class ValidationError(ResponseHandlerError):
    """Raised when response validation fails"""
    pass

@dataclass
class ProcessedResponse(Generic[T]):
    """Container for processed API response"""
    data: T
    metadata: Dict[str, Any]
    processed_at: datetime

class ResponseHandler:
    """
    Handles processing and validation of API responses.
    
    This class provides:
    - Response data extraction
    - Data validation
    - Type conversion
    - Error handling
    - Response metadata processing
    """

    def __init__(self):
        """Initialize ResponseHandler"""
        self._validators: Dict[str, Callable] = {}
        self._processors: Dict[str, Callable] = {}

    def register_validator(
        self,
        name: str,
        validator: Callable[[Any], bool]
    ) -> None:
        """
        Register a validation function
        
        Args:
            name: Name of the validator
            validator: Validation function
        """
        self._validators[name] = validator

    def register_processor(
        self,
        name: str,
        processor: Callable[[Any], Any]
    ) -> None:
        """
        Register a processing function
        
        Args:
            name: Name of the processor
            processor: Processing function
        """
        self._processors[name] = processor

    def process_response(
        self,
        response: APIResponse,
        expected_type: Optional[type] = None,
        validator_name: Optional[str] = None,
        processor_name: Optional[str] = None
    ) -> ProcessedResponse:
        """
        Process an API response
        
        Args:
            response: APIResponse to process
            expected_type: Expected type of response data
            validator_name: Name of validator to use
            processor_name: Name of processor to use
            
        Returns:
            ProcessedResponse containing processed data
        """
        try:
            # Extract response data
            data = response.data
            
            # Validate response if validator specified
            if validator_name:
                validator = self._validators.get(validator_name)
                if validator and not validator(data):
                    raise ValidationError(f"Response validation failed using {validator_name}")
            
            # Process data if processor specified
            if processor_name:
                processor = self._processors.get(processor_name)
                if processor:
                    data = processor(data)
            
            # Type conversion if expected_type specified
            if expected_type:
                try:
                    if isinstance(data, dict):
                        data = expected_type(**data)
                    elif isinstance(data, list):
                        data = [expected_type(**item) if isinstance(item, dict) else item for item in data]
                    else:
                        data = expected_type(data)
                except Exception as e:
                    raise ValidationError(f"Type conversion failed: {str(e)}")
            
            # Extract metadata
            metadata = {
                "status": response.status,
                "headers": response.headers,
                "timestamp": response.timestamp,
                "duration": response.duration
            }
            
            return ProcessedResponse(
                data=data,
                metadata=metadata,
                processed_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to process response: {str(e)}")
            raise ResponseHandlerError(f"Failed to process response: {str(e)}")

    def extract_error(self, response: APIResponse) -> Dict[str, Any]:
        """
        Extract error information from failed response
        
        Args:
            response: Failed APIResponse
            
        Returns:
            Dict containing error details
        """
        try:
            error_data = {
                "status": response.status,
                "timestamp": response.timestamp.isoformat(),
                "headers": response.headers
            }
            
            if isinstance(response.data, dict):
                error_data.update({
                    "error": response.data.get("error", "Unknown error"),
                    "message": response.data.get("message", "No message provided"),
                    "code": response.data.get("code", "NO_CODE"),
                    "details": response.data.get("details", {})
                })
            else:
                error_data.update({
                    "error": "Invalid Response",
                    "message": str(response.data),
                    "code": "INVALID_RESPONSE",
                    "details": {}
                })
                
            return error_data
            
        except Exception as e:
            logger.error(f"Failed to extract error information: {str(e)}")
            return {
                "error": "Error Extraction Failed",
                "message": str(e),
                "code": "ERROR_EXTRACTION_FAILED",
                "details": {}
            }

    def validate_response_schema(
        self,
        response: APIResponse,
        schema: Dict[str, Any]
    ) -> bool:
        """
        Validate response data against a schema
        
        Args:
            response: APIResponse to validate
            schema: Schema to validate against
            
        Returns:
            bool indicating if validation passed
        """
        try:
            def validate_type(value: Any, expected_type: str) -> bool:
                if expected_type == "string":
                    return isinstance(value, str)
                elif expected_type == "number":
                    return isinstance(value, (int, float))
                elif expected_type == "boolean":
                    return isinstance(value, bool)
                elif expected_type == "array":
                    return isinstance(value, list)
                elif expected_type == "object":
                    return isinstance(value, dict)
                return False

            def validate_object(obj: Dict[str, Any], schema_obj: Dict[str, Any]) -> bool:
                for key, schema_value in schema_obj.items():
                    if key not in obj:
                        if schema_value.get("required", False):
                            return False
                        continue
                        
                    if not validate_type(obj[key], schema_value["type"]):
                        return False
                        
                    if schema_value["type"] == "object" and "properties" in schema_value:
                        if not validate_object(obj[key], schema_value["properties"]):
                            return False
                            
                    if schema_value["type"] == "array" and "items" in schema_value:
                        if not all(validate_type(item, schema_value["items"]["type"]) for item in obj[key]):
                            return False
                            
                return True

            return validate_object(response.data, schema)
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False