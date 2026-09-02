"""
Structured Logging
------------------
Provides structured logging for all application events.
Logs important technical events while protecting sensitive data.
"""

import os
import logging
import json
from datetime import datetime
from app.utils.helpers import utcnow
from logging.handlers import RotatingFileHandler
from config.settings import DATA_DIR


# ── Log Levels ────────────────────────────────────────────────────────────────

class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ── Event Types ───────────────────────────────────────────────────────────────

class EventType:
    # Application events
    APP_START = "app.start"
    APP_STOP = "app.stop"

    # Camera events
    CAMERA_INIT = "camera.init"
    CAMERA_START = "camera.start"
    CAMERA_STOP = "camera.stop"
    CAMERA_ERROR = "camera.error"
    CAMERA_FRAME = "camera.frame"

    # Model events
    MODEL_LOAD = "model.load"
    MODEL_ERROR = "model.error"

    # Registration events
    REGISTRATION_START = "registration.start"
    REGISTRATION_COMPLETE = "registration.complete"
    REGISTRATION_FAILED = "registration.failed"

    # Recognition events
    RECOGNITION_ATTEMPT = "recognition.attempt"
    RECOGNITION_SUCCESS = "recognition.success"
    RECOGNITION_FAILED = "recognition.failed"
    RECOGNITION_UNKNOWN = "recognition.unknown"

    # Attendance events
    ATTENDANCE_MARK = "attendance.mark"
    ATTENDANCE_DUPLICATE = "attendance.duplicate"
    ATTENDANCE_CHECKOUT = "attendance.checkout"

    # User events
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    USER_LOGIN = "user.login"

    # Database events
    DB_QUERY = "db.query"
    DB_ERROR = "db.error"
    DB_CONNECT = "db.connect"

    # Security events
    SECURITY_VIOLATION = "security.violation"
    RATE_LIMIT = "security.rate_limit"


# ── Structured Logger ─────────────────────────────────────────────────────────

class StructuredLogger:
    """Structured logger for application events."""

    def __init__(self, name: str = "app"):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with file and console handlers."""
        # Console handler (always works)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

        # File handler (only on non-Vercel / writable filesystem)
        try:
            import os
            is_vercel = os.environ.get('VERCEL', '0') == '1'
            if not is_vercel:
                log_dir = DATA_DIR / "logs"
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / "app.log"
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.setLevel(logging.DEBUG)
        except Exception:
            pass  # Silently skip file logging on Vercel

    def log_event(self, event_type: str, message: str, **kwargs):
        """
        Log a structured event.

        Args:
            event_type: Type of event (use EventType constants)
            message: Human-readable message
            **kwargs: Additional context data
        """
        extra_data = {
            "event_type": event_type,
            "timestamp": utcnow().isoformat(),
            **kwargs
        }

        # Remove sensitive data
        extra_data = self._sanitize(extra_data)

        log_message = f"[{event_type}] {message}"
        if extra_data:
            log_message += f" | {json.dumps(extra_data, default=str)}"

        self.logger.info(log_message)

    def log_error(self, event_type: str, message: str, error: Exception = None, **kwargs):
        """Log an error event."""
        extra_data = {
            "event_type": event_type,
            "timestamp": utcnow().isoformat(),
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            **kwargs
        }

        extra_data = self._sanitize(extra_data)

        log_message = f"[{event_type}] {message}"
        if extra_data:
            log_message += f" | {json.dumps(extra_data, default=str)}"

        self.logger.error(log_message)

    def log_warning(self, event_type: str, message: str, **kwargs):
        """Log a warning event."""
        extra_data = {
            "event_type": event_type,
            "timestamp": utcnow().isoformat(),
            **kwargs
        }

        extra_data = self._sanitize(extra_data)

        log_message = f"[{event_type}] {message}"
        if extra_data:
            log_message += f" | {json.dumps(extra_data, default=str)}"

        self.logger.warning(log_message)

    def _sanitize(self, data: dict) -> dict:
        """Remove sensitive data from log entries."""
        sensitive_keys = [
            "password", "password_hash", "token", "secret",
            "face_encoding", "face_data", "biometric"
        ]

        sanitized = {}
        for key, value in data.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize(value)
            else:
                sanitized[key] = value

        return sanitized


# ── Application Logger Instance ───────────────────────────────────────────────

app_logger = StructuredLogger("face_attendance")


# ── Convenience Functions ────────────────────────────────────────────────────

def log_app_start(config: str = "development"):
    """Log application startup."""
    app_logger.log_event(EventType.APP_START, "Application started", config=config)


def log_camera_event(event: str, camera_index: int = 0, details: str = None):
    """Log camera events."""
    app_logger.log_event(f"camera.{event}", f"Camera {event}", camera_index=camera_index, details=details)


def log_model_load(model_name: str, success: bool, details: str = None):
    """Log model loading events."""
    if success:
        app_logger.log_event(EventType.MODEL_LOAD, f"Model loaded: {model_name}", details=details)
    else:
        app_logger.log_error(EventType.MODEL_ERROR, f"Model failed to load: {model_name}", details=details)


def log_registration(user_id: int, employee_id: str, success: bool, samples: int = 0):
    """Log registration events."""
    if success:
        app_logger.log_event(
            EventType.REGISTRATION_COMPLETE,
            f"Registration completed for {employee_id}",
            user_id=user_id, samples=samples
        )
    else:
        app_logger.log_error(
            EventType.REGISTRATION_FAILED,
            f"Registration failed for {employee_id}",
            user_id=user_id
        )


def log_recognition(user_id: int, employee_id: str, confidence: float, success: bool):
    """Log recognition events."""
    if success:
        app_logger.log_event(
            EventType.RECOGNITION_SUCCESS,
            f"Recognition successful: {employee_id}",
            user_id=user_id, confidence=confidence
        )
    else:
        app_logger.log_event(
            EventType.RECOGNITION_UNKNOWN,
            "Unknown face detected",
            confidence=confidence
        )


def log_attendance(user_id: int, employee_id: str, action: str, success: bool, reason: str = None):
    """Log attendance events."""
    event_type = f"attendance.{action}"
    if success:
        app_logger.log_event(
            event_type,
            f"Attendance {action} for {employee_id}",
            user_id=user_id
        )
    else:
        app_logger.log_warning(
            event_type,
            f"Attendance {action} failed for {employee_id}: {reason}",
            user_id=user_id, reason=reason
        )


def log_database_error(operation: str, error: Exception):
    """Log database errors."""
    app_logger.log_error(
        EventType.DB_ERROR,
        f"Database error during {operation}",
        error=error
    )


def log_security_event(event_type: str, details: str, ip: str = None):
    """Log security events."""
    app_logger.log_warning(
        f"security.{event_type}",
        details,
        ip=ip
    )
