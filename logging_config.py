#!/usr/bin/env python3
"""
JSON RAG System - Fixed Logging Configuration Module
Centralizes logging configuration for the entire system
Fixes LogRecord module overwrite issue
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Log levels mapping
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the log level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the record
        formatted = super().format(record)
        
        # Reset colors at the end
        return formatted + self.COLORS['RESET']

class SystemLogFilter(logging.Filter):
    """Custom filter to add system information to log records"""
    
    def filter(self, record):
        # Add system information
        record.system_name = 'JSON_RAG_System'
        record.pid = os.getpid()
        
        # Add custom fields if they exist
        if hasattr(record, 'session_id'):
            record.session_id = getattr(record, 'session_id', 'unknown')
        if hasattr(record, 'user_id'):
            record.user_id = getattr(record, 'user_id', 'anonymous')
        if hasattr(record, 'operation'):
            record.operation = getattr(record, 'operation', 'general')
        
        return True

def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = 'logs',
    console_output: bool = True,
    file_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_style: str = 'detailed'
) -> Dict[str, Any]:
    """Setup logging configuration for the entire system"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Define log formats
    formats = {
        'simple': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'detailed': '%(asctime)s - %(system_name)s[%(pid)d] - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        'json': '{"timestamp": "%(asctime)s", "system": "%(system_name)s", "pid": %(pid)d, "logger": "%(name)s", "level": "%(levelname)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
        'console': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    # Get the log level
    numeric_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create system filter
    system_filter = SystemLogFilter()
    
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            fmt=formats['console'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(system_filter)
        
        root_logger.addHandler(console_handler)
        handlers.append('console')
    
    # File handlers
    if file_output:
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / 'main.log',
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(numeric_level)
        
        main_formatter = logging.Formatter(
            fmt=formats[format_style],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        main_handler.addFilter(system_filter)
        
        root_logger.addHandler(main_handler)
        handlers.append('main_file')
        
        # Error log (ERROR and CRITICAL only)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / 'error.log',
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = logging.Formatter(
            fmt=formats['detailed'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(system_filter)
        
        root_logger.addHandler(error_handler)
        handlers.append('error_file')
        
        # Debug log (DEBUG level only, if debug is enabled)
        if numeric_level <= logging.DEBUG:
            debug_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / 'debug.log',
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            
            debug_formatter = logging.Formatter(
                fmt=formats['detailed'],
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            debug_handler.setFormatter(debug_formatter)
            debug_handler.addFilter(system_filter)
            
            root_logger.addHandler(debug_handler)
            handlers.append('debug_file')
    
    # Configure specific loggers
    configure_specific_loggers(numeric_level)
    
    # Log the setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed - Level: {log_level}, Handlers: {handlers}")
    
    return {
        'level': log_level,
        'handlers': handlers,
        'log_dir': str(log_path),
        'format_style': format_style
    }

def configure_specific_loggers(level: int):
    """Configure specific loggers with appropriate levels"""
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    # Set appropriate levels for system components
    component_loggers = {
        'core_system': level,
        'database': level,
        'indexing': level,
        'search': level,
        'vocabulary': level,
        'session': level,
        'api': level
    }
    
    for logger_name, logger_level in component_loggers.items():
        logging.getLogger(logger_name).setLevel(logger_level)

def get_logger(name: str, **kwargs) -> logging.Logger:
    """Get a logger with optional context information"""
    logger = logging.getLogger(name)
    
    # Create a logger adapter if context is provided
    if kwargs:
        return logging.LoggerAdapter(logger, kwargs)
    
    return logger

def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def log_performance(func):
    """Decorator to log function performance"""
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f} seconds: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# FIXED: StructuredLogger class with proper LogRecord attribute handling
class StructuredLogger:
    """Structured logger for consistent log formatting"""
    
    def __init__(self, name: str, context: Dict[str, Any] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with structured format - FIXED to handle LogRecord conflicts"""
        # Handle reserved LogRecord attributes by renaming them
        log_data = {**self.context, **kwargs}
        
        # Rename 'module' to 'source_module' to avoid LogRecord conflict
        if 'module' in log_data:
            log_data['source_module'] = log_data.pop('module')
        
        # Create extra dict for LogRecord, avoiding reserved names
        extra_dict = {}
        reserved_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
            'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
            'relativeCreated', 'thread', 'threadName', 'processName', 
            'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info',
            'taskName'  # Added in Python 3.12+
        }
        
        for key, value in log_data.items():
            if key not in reserved_attrs:
                extra_dict[key] = value
        
        if log_data:
            extra_info = ' | '.join(f"{k}={v}" for k, v in log_data.items())
            formatted_message = f"{message} | {extra_info}"
        else:
            formatted_message = message
        
        # Use extra parameter to pass custom fields safely
        try:
            self.logger.log(level, formatted_message, extra=extra_dict)
        except (TypeError, ValueError) as e:
            # Fallback to simple logging if extra fields cause issues
            fallback_message = f"{message} | {extra_info}" if log_data else message
            self.logger.log(level, f"[LOGGING_ERROR] {fallback_message} | original_error={e}")
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

# Context manager for logging operations
class LogOperation:
    """Context manager for logging operations"""
    
    def __init__(self, logger: logging.Logger, operation_name: str, 
                 log_start: bool = True, log_success: bool = True, log_errors: bool = True):
        self.logger = logger
        self.operation_name = operation_name
        self.log_start = log_start
        self.log_success = log_success
        self.log_errors = log_errors
        self.start_time = None
    
    def __enter__(self):
        if self.log_start:
            self.logger.info(f"Starting operation: {self.operation_name}")
        self.start_time = __import__('time').time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = __import__('time').time() - self.start_time
        
        if exc_type is None:
            if self.log_success:
                self.logger.info(f"Operation completed: {self.operation_name} ({duration:.3f}s)")
        else:
            if self.log_errors:
                self.logger.error(f"Operation failed: {self.operation_name} after {duration:.3f}s - {exc_val}")
        
        return False  # Don't suppress exceptions

# Initialize logging on module import
if __name__ != '__main__':
    # Setup basic logging when module is imported
    setup_logging()
