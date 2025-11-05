"""
Error Handling Utilities for ProSense

Provides custom exceptions, error handling decorators, and utilities for
robust error management throughout the signal processing pipeline.
"""

import functools
import traceback
from typing import Callable, Optional, Type, Union, Tuple, Any
from logging_config import get_logger


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ProSenseError(Exception):
    """Base exception for all ProSense errors."""
    pass


class ConfigurationError(ProSenseError):
    """Raised when there's an error in configuration."""
    pass


class DataValidationError(ProSenseError):
    """Raised when data validation fails."""
    pass


class DataLoadingError(ProSenseError):
    """Raised when data loading fails."""
    pass


class SignalProcessingError(ProSenseError):
    """Raised when signal processing fails."""
    pass


class FeatureExtractionError(ProSenseError):
    """Raised when feature extraction fails."""
    pass


class SamplingRateError(ProSenseError):
    """Raised when sampling rate is invalid or inconsistent."""
    pass


class TimestampError(ProSenseError):
    """Raised when timestamp validation or processing fails."""
    pass


class FileFormatError(ProSenseError):
    """Raised when file format is invalid or unsupported."""
    pass


class InsufficientDataError(ProSenseError):
    """Raised when there's insufficient data for processing."""
    pass


# =============================================================================
# ERROR HANDLING DECORATORS
# =============================================================================

def handle_errors(
    error_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    logger_name: Optional[str] = None,
    reraise: bool = True,
    custom_message: Optional[str] = None
):
    """
    Decorator for comprehensive error handling.

    Args:
        error_types: Exception type(s) to catch
        default_return: Default return value if error occurs and not reraising
        logger_name: Logger name to use
        reraise: Whether to re-raise the exception after handling
        custom_message: Custom error message prefix

    Example:
        >>> @handle_errors(ValueError, default_return=None)
        ... def risky_function(x):
        ...     return int(x)
    """
    logger = get_logger(logger_name or "prosense")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Build error message
                if custom_message:
                    error_msg = f"{custom_message}: {type(e).__name__}: {e}"
                else:
                    error_msg = f"Error in {func.__name__}: {type(e).__name__}: {e}"

                logger.error(error_msg)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")

                if reraise:
                    raise
                else:
                    logger.warning(f"Returning default value: {default_return}")
                    return default_return

        return wrapper
    return decorator


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_name: Optional[str] = None
):
    """
    Decorator to retry function on error with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Exception types to retry on
        logger_name: Logger name to use

    Example:
        >>> @retry_on_error(max_attempts=3, delay=1.0)
        ... def unstable_function():
        ...     # Code that might fail temporarily
        ...     pass
    """
    import time

    logger = get_logger(logger_name or "prosense")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: "
                            f"{type(e).__name__}: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.

    Args:
        **validators: Keyword arguments mapping parameter names to validation functions

    Example:
        >>> def is_positive(x):
        ...     if x <= 0:
        ...         raise ValueError("Must be positive")
        ...     return True
        >>>
        >>> @validate_inputs(value=is_positive)
        ... def process_value(value):
        ...     return value * 2
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validator(value)
                    except Exception as e:
                        raise DataValidationError(
                            f"Validation failed for parameter '{param_name}': {e}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# ERROR CONTEXT MANAGERS
# =============================================================================

class ErrorContext:
    """
    Context manager for error handling with automatic logging.

    Example:
        >>> with ErrorContext("EEG Processing", reraise=True):
        ...     # Processing code that might fail
        ...     pass
    """

    def __init__(
        self,
        operation_name: str,
        logger_name: Optional[str] = None,
        reraise: bool = True,
        on_error: Optional[Callable] = None
    ):
        """
        Initialize error context.

        Args:
            operation_name: Name of the operation
            logger_name: Logger name to use
            reraise: Whether to re-raise exceptions
            on_error: Optional callback function to call on error
        """
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or "prosense")
        self.reraise = reraise
        self.on_error = on_error
        self.error_occurred = False
        self.exception = None

    def __enter__(self):
        """Enter the context."""
        self.logger.debug(f"Entering error context: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if exc_type is not None:
            self.error_occurred = True
            self.exception = exc_val

            # Log the error
            self.logger.error(
                f"Error in {self.operation_name}: {exc_type.__name__}: {exc_val}"
            )
            self.logger.debug(f"Traceback:\n{''.join(traceback.format_tb(exc_tb))}")

            # Call error callback if provided
            if self.on_error:
                try:
                    self.on_error(exc_type, exc_val, exc_tb)
                except Exception as callback_error:
                    self.logger.error(
                        f"Error in error callback: {callback_error}"
                    )

            # Suppress exception if not reraising
            return not self.reraise

        self.logger.debug(f"Exiting error context: {self.operation_name}")
        return False


# =============================================================================
# ERROR REPORTING
# =============================================================================

class ErrorReport:
    """
    Collects and formats error information for reporting.
    """

    def __init__(self):
        """Initialize error report."""
        self.errors = []

    def add_error(
        self,
        error_type: str,
        message: str,
        context: Optional[dict] = None,
        exception: Optional[Exception] = None
    ):
        """
        Add an error to the report.

        Args:
            error_type: Type/category of error
            message: Error message
            context: Additional context information
            exception: The exception object (if any)
        """
        error_entry = {
            'type': error_type,
            'message': message,
            'context': context or {},
            'exception': str(exception) if exception else None,
            'timestamp': None  # Could add datetime here
        }
        self.errors.append(error_entry)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0

    def get_summary(self) -> str:
        """
        Get a summary of all errors.

        Returns:
            Formatted error summary string
        """
        if not self.errors:
            return "No errors recorded."

        summary_lines = [f"Error Report: {len(self.errors)} error(s) found\n"]
        for i, error in enumerate(self.errors, 1):
            summary_lines.append(f"{i}. {error['type']}: {error['message']}")
            if error['context']:
                summary_lines.append(f"   Context: {error['context']}")

        return "\n".join(summary_lines)

    def clear(self):
        """Clear all recorded errors."""
        self.errors = []


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_not_none(value: Any, param_name: str = "value"):
    """Validate that a value is not None."""
    if value is None:
        raise DataValidationError(f"{param_name} cannot be None")
    return True


def validate_positive(value: Union[int, float], param_name: str = "value"):
    """Validate that a numeric value is positive."""
    if value <= 0:
        raise DataValidationError(f"{param_name} must be positive, got {value}")
    return True


def validate_in_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    param_name: str = "value"
):
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise DataValidationError(
            f"{param_name} must be between {min_val} and {max_val}, got {value}"
        )
    return True


def validate_file_exists(file_path: str):
    """Validate that a file exists."""
    from pathlib import Path
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise FileFormatError(f"Path is not a file: {file_path}")
    return True


# Example usage
if __name__ == "__main__":
    from logging_config import setup_logging

    # Set up logging
    setup_logging(level="DEBUG")

    # Test error handling decorator
    @handle_errors(ValueError, default_return=0)
    def divide(a, b):
        return a / b

    result = divide(10, 0)  # Will handle the exception
    print(f"Result: {result}")

    # Test retry decorator
    @retry_on_error(max_attempts=3, delay=0.5)
    def unstable_operation():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Temporary failure")
        return "Success"

    try:
        result = unstable_operation()
        print(f"Operation result: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")

    # Test error context
    with ErrorContext("Test Operation", reraise=False):
        raise ValueError("This error will be caught and logged")

    print("Continued execution after error context")

    # Test error report
    report = ErrorReport()
    report.add_error("DataError", "Invalid data format", {"file": "test.pkl"})
    report.add_error("ConfigError", "Missing parameter", {"param": "sampling_rate"})
    print(report.get_summary())
