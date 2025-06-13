#!/usr/bin/env python3
"""
JSON RAG System - Custom Exceptions Module
Defines custom exception classes for better error handling
"""

class JSONRAGSystemError(Exception):
    """Base exception class for all JSON RAG System errors"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'SYSTEM_ERROR'
        self.details = details or {}
        
    def to_dict(self) -> dict:
        """Convert exception to dictionary format"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }

class DatabaseConnectionError(JSONRAGSystemError):
    """Raised when database connection fails"""
    def __init__(self, message: str = "Failed to connect to database", **kwargs):
        super().__init__(message, error_code='DB_CONNECTION_ERROR', **kwargs)

class DatabaseQueryError(JSONRAGSystemError):
    """Raised when database query execution fails"""
    def __init__(self, message: str = "Database query failed", **kwargs):
        super().__init__(message, error_code='DB_QUERY_ERROR', **kwargs)

class IndexNotFoundError(JSONRAGSystemError):
    """Raised when required index files are not found"""
    def __init__(self, message: str = "Index files not found", **kwargs):
        super().__init__(message, error_code='INDEX_NOT_FOUND', **kwargs)

class IndexCreationError(JSONRAGSystemError):
    """Raised when index creation fails"""
    def __init__(self, message: str = "Failed to create search index", **kwargs):
        super().__init__(message, error_code='INDEX_CREATION_ERROR', **kwargs)

class EmbeddingError(JSONRAGSystemError):
    """Raised when text embedding generation fails"""
    def __init__(self, message: str = "Failed to generate text embeddings", **kwargs):
        super().__init__(message, error_code='EMBEDDING_ERROR', **kwargs)

class VocabularyError(JSONRAGSystemError):
    """Raised when vocabulary processing fails"""
    def __init__(self, message: str = "Vocabulary processing error", **kwargs):
        super().__init__(message, error_code='VOCABULARY_ERROR', **kwargs)

class QueryProcessingError(JSONRAGSystemError):
    """Raised when query processing fails"""
    def __init__(self, message: str = "Query processing failed", **kwargs):
        super().__init__(message, error_code='QUERY_PROCESSING_ERROR', **kwargs)

class SearchError(JSONRAGSystemError):
    """Raised when search execution fails"""
    def __init__(self, message: str = "Search execution failed", **kwargs):
        super().__init__(message, error_code='SEARCH_ERROR', **kwargs)

class ConfigurationError(JSONRAGSystemError):
    """Raised when configuration is invalid or missing"""
    def __init__(self, message: str = "Invalid configuration", **kwargs):
        super().__init__(message, error_code='CONFIGURATION_ERROR', **kwargs)

class FileOperationError(JSONRAGSystemError):
    """Raised when file operations fail"""
    def __init__(self, message: str = "File operation failed", **kwargs):
        super().__init__(message, error_code='FILE_OPERATION_ERROR', **kwargs)

class SessionError(JSONRAGSystemError):
    """Raised when session management fails"""
    def __init__(self, message: str = "Session management error", **kwargs):
        super().__init__(message, error_code='SESSION_ERROR', **kwargs)

class ValidationError(JSONRAGSystemError):
    """Raised when input validation fails"""
    def __init__(self, message: str = "Input validation failed", **kwargs):
        super().__init__(message, error_code='VALIDATION_ERROR', **kwargs)

class TextProcessingError(JSONRAGSystemError):
    """Raised when text processing fails"""
    def __init__(self, message: str = "Text processing failed", **kwargs):
        super().__init__(message, error_code='TEXT_PROCESSING_ERROR', **kwargs)

class ModelLoadError(JSONRAGSystemError):
    """Raised when ML model loading fails"""
    def __init__(self, message: str = "Failed to load ML model", **kwargs):
        super().__init__(message, error_code='MODEL_LOAD_ERROR', **kwargs)

class ResourceNotFoundError(JSONRAGSystemError):
    """Raised when required resource is not found"""
    def __init__(self, message: str = "Required resource not found", **kwargs):
        super().__init__(message, error_code='RESOURCE_NOT_FOUND', **kwargs)

class MemoryError(JSONRAGSystemError):
    """Raised when memory operations fail"""
    def __init__(self, message: str = "Memory operation failed", **kwargs):
        super().__init__(message, error_code='MEMORY_ERROR', **kwargs)

class TimeoutError(JSONRAGSystemError):
    """Raised when operations timeout"""
    def __init__(self, message: str = "Operation timed out", **kwargs):
        super().__init__(message, error_code='TIMEOUT_ERROR', **kwargs)

class RateLimitError(JSONRAGSystemError):
    """Raised when rate limits are exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, error_code='RATE_LIMIT_ERROR', **kwargs)

class AuthenticationError(JSONRAGSystemError):
    """Raised when authentication fails"""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code='AUTHENTICATION_ERROR', **kwargs)

class AuthorizationError(JSONRAGSystemError):
    """Raised when authorization fails"""
    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(message, error_code='AUTHORIZATION_ERROR', **kwargs)

# Exception handler decorator
def handle_exceptions(default_return=None, log_errors=True):
    """Decorator to handle exceptions gracefully"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except JSONRAGSystemError as e:
                if log_errors:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"JSONRAGSystem error in {func.__name__}: {e.message}", 
                               extra={'error_code': e.error_code, 'details': e.details})
                if default_return is not None:
                    return default_return
                raise
            except Exception as e:
                if log_errors:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
                if default_return is not None:
                    return default_return
                raise JSONRAGSystemError(f"Unexpected error in {func.__name__}: {str(e)}", 
                                        error_code='UNEXPECTED_ERROR')
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

# Error context manager
class ErrorContext:
    """Context manager for error handling"""
    def __init__(self, operation_name: str, logger=None, reraise=True):
        self.operation_name = operation_name
        self.logger = logger
        self.reraise = reraise
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"Error in {self.operation_name}: {str(exc_val)}"
            
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            
            if self.reraise:
                if isinstance(exc_val, JSONRAGSystemError):
                    return False  # Re-raise the original exception
                else:
                    raise JSONRAGSystemError(error_msg, error_code='OPERATION_ERROR')
            
        return not self.reraise

# Validation helpers
def validate_not_none(value, name: str):
    """Validate that value is not None"""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value

def validate_not_empty(value, name: str):
    """Validate that value is not empty"""
    if not value:
        raise ValidationError(f"{name} cannot be empty")
    return value

def validate_type(value, expected_type, name: str):
    """Validate that value is of expected type"""
    if not isinstance(value, expected_type):
        raise ValidationError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")
    return value

def validate_range(value, min_val=None, max_val=None, name: str = "value"):
    """Validate that value is within specified range"""
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value

def validate_length(value, min_len=None, max_len=None, name: str = "value"):
    """Validate that value length is within specified range"""
    length = len(value)
    if min_len is not None and length < min_len:
        raise ValidationError(f"{name} length must be >= {min_len}, got {length}")
    if max_len is not None and length > max_len:
        raise ValidationError(f"{name} length must be <= {max_len}, got {length}")
    return value
