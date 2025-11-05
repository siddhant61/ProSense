"""
Structured Logging Framework for ProSense

Provides centralized logging configuration with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output
- Colored console output
- JSON structured logging (optional)
- Performance timing decorators
- Context managers for operation logging
"""

import logging
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    use_colors: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for ProSense.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_to_console: Whether to output logs to console
        use_colors: Whether to use colored output (console only)
        format_string: Custom format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="prosense.log")
        >>> logger.info("Processing started")
    """
    # Create logger
    logger = logging.getLogger("prosense")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Default format string
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if use_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "prosense") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (default: "prosense")

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing EEG data")
    """
    return logging.getLogger(name)


# =============================================================================
# LOGGING DECORATORS
# =============================================================================

def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (uses default if None)

    Example:
        >>> @log_execution_time()
        ... def process_data(data):
        ...     # Processing code
        ...     pass
    """
    if logger is None:
        logger = get_logger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}...")

            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Completed {func.__name__} in {elapsed_time:.2f}s"
                )
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"Failed {func.__name__} after {elapsed_time:.2f}s: {e}"
                )
                raise

        return wrapper
    return decorator


def log_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments.

    Args:
        logger: Logger instance (uses default if None)
        level: Log level for the message

    Example:
        >>> @log_function_call()
        ... def load_data(file_path, modality):
        ...     pass
    """
    if logger is None:
        logger = get_logger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_method = getattr(logger, level.lower())
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            log_method(f"Calling {func.__name__}({signature})")

            try:
                result = func(*args, **kwargs)
                log_method(f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to log exceptions that occur in a function.

    Args:
        logger: Logger instance (uses default if None)
        reraise: Whether to re-raise the exception after logging

    Example:
        >>> @log_exceptions()
        ... def risky_operation():
        ...     # Code that might raise exceptions
        ...     pass
    """
    if logger is None:
        logger = get_logger()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}"
                )
                if reraise:
                    raise
                return None

        return wrapper
    return decorator


# =============================================================================
# LOGGING CONTEXT MANAGERS
# =============================================================================

class LoggingContext:
    """
    Context manager for logging operations.

    Example:
        >>> with LoggingContext("EEG Processing", logger):
        ...     # EEG processing code
        ...     pass
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        level: str = "INFO"
    ):
        """
        Initialize logging context.

        Args:
            operation_name: Name of the operation
            logger: Logger instance
            level: Log level
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Enter the context."""
        self.start_time = time.time()
        log_method = getattr(self.logger, self.level.lower())
        log_method(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        elapsed_time = time.time() - self.start_time

        if exc_type is None:
            log_method = getattr(self.logger, self.level.lower())
            log_method(
                f"Completed: {self.operation_name} ({elapsed_time:.2f}s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} ({elapsed_time:.2f}s) - "
                f"{exc_type.__name__}: {exc_val}"
            )

        # Don't suppress exceptions
        return False


class ProgressLogger:
    """
    Context manager for logging progress of iterative operations.

    Example:
        >>> with ProgressLogger("Processing files", total=100) as progress:
        ...     for i in range(100):
        ...         # Process file
        ...         progress.update(1)
    """

    def __init__(
        self,
        operation_name: str,
        total: int,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10
    ):
        """
        Initialize progress logger.

        Args:
            operation_name: Name of the operation
            total: Total number of items to process
            logger: Logger instance
            log_interval: Log progress every N percent
        """
        self.operation_name = operation_name
        self.total = total
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
        self.last_logged_percent = 0

    def __enter__(self):
        """Enter the context."""
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name} (0/{self.total})")
        return self

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items completed
        """
        self.current += n
        percent = int((self.current / self.total) * 100)

        if percent >= self.last_logged_percent + self.log_interval:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0

            self.logger.info(
                f"{self.operation_name}: {percent}% "
                f"({self.current}/{self.total}) - "
                f"Rate: {rate:.1f} items/s - "
                f"ETA: {eta:.1f}s"
            )
            self.last_logged_percent = percent

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        elapsed_time = time.time() - self.start_time
        rate = self.current / elapsed_time if elapsed_time > 0 else 0

        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name} - "
                f"{self.current}/{self.total} items in {elapsed_time:.2f}s "
                f"({rate:.1f} items/s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} at {self.current}/{self.total} - "
                f"{exc_type.__name__}: {exc_val}"
            )

        return False


# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize default logger
_default_logger = None


def initialize_logging_from_config(config_loader):
    """
    Initialize logging using configuration from config.yaml.

    Args:
        config_loader: ConfigLoader instance

    Example:
        >>> from config_loader import get_config
        >>> config = get_config()
        >>> initialize_logging_from_config(config)
    """
    global _default_logger

    log_level = config_loader.get("logging.level", "INFO")
    log_file = config_loader.get("logging.file", None)

    _default_logger = setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_console=True,
        use_colors=True
    )

    return _default_logger


# Example usage
if __name__ == "__main__":
    # Set up logging
    logger = setup_logging(level="DEBUG", log_file="prosense.log")

    # Test basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test decorator
    @log_execution_time()
    def example_function():
        time.sleep(0.1)
        return "Done"

    result = example_function()

    # Test context manager
    with LoggingContext("Test Operation", logger):
        time.sleep(0.1)

    # Test progress logger
    with ProgressLogger("Processing Items", total=50, logger=logger) as progress:
        for i in range(50):
            time.sleep(0.01)
            progress.update(1)
