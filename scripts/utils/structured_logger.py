#!/usr/bin/env python3
"""
Structured logging utility for ExcuseMyFrench project.

Provides JSON-formatted structured logging that's easier to parse and monitor
in production environments. Can be enabled via environment variable.

Usage:
    from scripts.utils.structured_logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing video", extra={
        "video_id": "vid_123",
        "duration": 45.2,
        "character": "Butcher"
    })

Environment Variables:
    STRUCTURED_LOGGING - Set to "true" to enable JSON logging (default: false)
    LOG_LEVEL - Logging level (default: INFO)
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects with consistent fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add thread/process info if available
        if record.process:
            log_data["process_id"] = record.process
        if record.thread:
            log_data["thread_id"] = record.thread

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": "".join(traceback.format_exception(*record.exc_info))
            }

        # Add any extra fields passed via the extra parameter
        # These are custom fields specific to the log message
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add any other custom attributes from the record
        # (excluding built-in logging attributes)
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'msecs', 'relativeCreated',
            'levelname', 'levelno', 'pathname', 'filename', 'module',
            'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'process', 'processName', 'thread', 'threadName', 'extra_fields',
            'message', 'asctime'
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                # Only add JSON-serializable values
                try:
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


class StructuredAdapter(logging.LoggerAdapter):
    """
    Adapter that adds extra fields to log records for structured logging.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message and kwargs to extract extra fields.

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            Tuple of (message, processed_kwargs)
        """
        # Extract 'extra' dict if present
        extra = kwargs.get('extra', {})

        # Merge with adapter's extra context
        if self.extra:
            extra = {**self.extra, **extra}

        # Store extra fields in a way that the formatter can access
        if extra:
            kwargs['extra'] = {'extra_fields': extra}

        return msg, kwargs


def get_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None,
    force_structured: bool = False
) -> logging.Logger:
    """
    Get a logger instance with optional structured logging.

    Args:
        name: Logger name (usually __name__)
        context: Optional context dict to include in all log messages
        force_structured: Force structured logging regardless of env var

    Returns:
        Configured logger instance
    """
    # Check if structured logging is enabled
    use_structured = force_structured or os.getenv('STRUCTURED_LOGGING', 'false').lower() == 'true'

    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Get log level from environment or default to INFO
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logger.level)

        if use_structured:
            # Use structured JSON formatter
            formatter = StructuredFormatter()
        else:
            # Use standard formatter
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    # Wrap in adapter if context is provided
    if context:
        logger = StructuredAdapter(logger, context)

    return logger


def configure_root_logger(
    level: Optional[str] = None,
    structured: Optional[bool] = None
):
    """
    Configure the root logger for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured logging (overrides env var)
    """
    # Determine settings
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()

    if structured is None:
        structured = os.getenv('STRUCTURED_LOGGING', 'false').lower() == 'true'

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root_logger.level)

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class LogContext:
    """
    Context manager for temporarily adding context to logs.

    Usage:
        logger = get_logger(__name__)

        with LogContext(logger, video_id="vid_123"):
            logger.info("Processing video")  # Automatically includes video_id
    """

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.

        Args:
            logger: Logger to add context to
            **context: Context fields to add
        """
        self.logger = logger
        self.context = context
        self.adapter = None
        self.original_logger = None

    def __enter__(self):
        """Enter context - wrap logger in adapter."""
        if isinstance(self.logger, StructuredAdapter):
            # Already an adapter, merge contexts
            merged_context = {**self.logger.extra, **self.context}
            self.adapter = StructuredAdapter(self.logger.logger, merged_context)
        else:
            self.adapter = StructuredAdapter(self.logger, self.context)

        return self.adapter

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original logger."""
        # Context is automatically discarded when exiting
        pass


# Convenience functions for common operations
def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_seconds: float,
    **extra_fields
):
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_seconds: Duration in seconds
        **extra_fields: Additional fields to log
    """
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_seconds": duration_seconds,
            "duration_ms": duration_seconds * 1000,
            **extra_fields
        }
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
):
    """
    Log an error with structured context.

    Args:
        logger: Logger instance
        error: The exception that occurred
        context: Optional context dictionary
        message: Optional custom message (defaults to error message)
    """
    log_message = message or str(error)

    extra = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        extra.update(context)

    logger.error(log_message, exc_info=error, extra=extra)


# Example usage
if __name__ == '__main__':
    # Example 1: Basic structured logging
    print("Example 1: Basic structured logging")
    logger = get_logger(__name__, force_structured=True)

    logger.info("Application started")
    logger.info("Processing video", extra={
        "video_id": "vid_123",
        "duration": 45.2,
        "character": "Butcher"
    })

    # Example 2: Using context
    print("\nExample 2: With context")
    logger_with_context = get_logger(__name__, context={"service": "animation"}, force_structured=True)
    logger_with_context.info("Animation started")

    # Example 3: Performance logging
    print("\nExample 3: Performance logging")
    import time
    start = time.time()
    time.sleep(0.1)
    duration = time.time() - start
    log_performance(logger, "test_operation", duration, status="success")

    # Example 4: Error logging
    print("\nExample 4: Error logging")
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, e, context={"video_id": "vid_123"})

    # Example 5: Using LogContext
    print("\nExample 5: Using LogContext")
    logger_plain = get_logger("test", force_structured=True)
    with LogContext(logger_plain, request_id="req_456", user="test_user"):
        logger_plain.info("Inside context")

    logger_plain.info("Outside context")
