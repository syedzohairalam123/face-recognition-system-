"""
Security Module
---------------
Security middleware, input validation, and protection utilities.
"""

import os
import re
import time
import logging
from functools import wraps
from collections import defaultdict
from flask import request, jsonify, g

logger = logging.getLogger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old requests
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= self.max_requests:
            return False

        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for a key."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        return max(0, self.max_requests - len(self._requests[key]))


# Global rate limiter instances
api_limiter = RateLimiter(max_requests=100, window_seconds=60)
upload_limiter = RateLimiter(max_requests=20, window_seconds=60)


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Decorator for rate limiting endpoints."""
    limiter = RateLimiter(max_requests, window_seconds)

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = f"{request.remote_addr}:{f.__name__}"
            if not limiter.is_allowed(key):
                return jsonify({
                    "success": False,
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": window_seconds
                }), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ── Input Validators ──────────────────────────────────────────────────────────

class InputValidator:
    """Validate and sanitize user input."""

    @staticmethod
    def validate_employee_id(employee_id: str) -> tuple:
        """Validate employee ID format."""
        if not employee_id:
            return False, "Employee ID is required"
        employee_id = employee_id.strip()
        if len(employee_id) < 2 or len(employee_id) > 50:
            return False, "Employee ID must be 2-50 characters"
        if not re.match(r'^[A-Za-z0-9\-_]+$', employee_id):
            return False, "Employee ID can only contain letters, numbers, hyphens, and underscores"
        return True, employee_id

    @staticmethod
    def validate_email(email: str) -> tuple:
        """Validate email format."""
        if not email:
            return False, "Email is required"
        email = email.strip().lower()
        pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "Invalid email format"
        if len(email) > 150:
            return False, "Email too long (max 150 characters)"
        return True, email

    @staticmethod
    def validate_name(name: str, field_name: str = "Name") -> tuple:
        """Validate name field."""
        if not name:
            return False, f"{field_name} is required"
        name = name.strip()
        if len(name) < 1 or len(name) > 100:
            return False, f"{field_name} must be 1-100 characters"
        if not re.match(r'^[a-zA-Z\s\-\'\.]+$', name):
            return False, f"{field_name} contains invalid characters"
        return True, name

    @staticmethod
    def validate_phone(phone: str) -> tuple:
        """Validate phone number (optional field)."""
        if not phone:
            return True, None  # Optional
        phone = phone.strip()
        if not re.match(r'^[\+]?[\d\s\-\(\)]{7,20}$', phone):
            return False, "Invalid phone number format"
        return True, phone

    @staticmethod
    def validate_department(department: str) -> tuple:
        """Validate department (optional field)."""
        if not department:
            return True, None  # Optional
        department = department.strip()
        if len(department) > 100:
            return False, "Department name too long (max 100 characters)"
        return True, department

    @staticmethod
    def validate_role(role: str) -> tuple:
        """Validate role field."""
        valid_roles = ["employee", "manager", "admin"]
        if role not in valid_roles:
            return False, f"Role must be one of: {', '.join(valid_roles)}"
        return True, role

    @staticmethod
    def validate_user_data(data: dict) -> tuple:
        """Validate complete user data dictionary."""
        errors = []

        # Required fields
        for field in ["employee_id", "first_name", "last_name", "email"]:
            if field not in data or not data[field]:
                errors.append(f"{field} is required")

        if errors:
            return False, errors

        # Validate each field
        validations = [
            InputValidator.validate_employee_id(data.get("employee_id", "")),
            InputValidator.validate_name(data.get("first_name", ""), "First name"),
            InputValidator.validate_name(data.get("last_name", ""), "Last name"),
            InputValidator.validate_email(data.get("email", "")),
            InputValidator.validate_phone(data.get("phone", "")),
            InputValidator.validate_department(data.get("department", "")),
            InputValidator.validate_role(data.get("role", "employee")),
        ]

        for valid, message in validations:
            if not valid:
                errors.append(message)

        if errors:
            return False, errors

        return True, "Valid"


# ── File Validators ───────────────────────────────────────────────────────────

class FileValidator:
    """Validate uploaded files."""

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
    MAX_IMAGE_DIMENSION = 4096  # Max width/height in pixels

    @staticmethod
    def validate_image_file(file) -> tuple:
        """Validate an uploaded image file."""
        if not file or not file.filename:
            return False, "No file provided"

        # Check extension
        ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else ""
        if ext not in FileValidator.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(FileValidator.ALLOWED_EXTENSIONS)}"

        # Check file size (read first chunk to estimate)
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > FileValidator.MAX_FILE_SIZE:
            return False, f"File too large. Max size: {FileValidator.MAX_FILE_SIZE // (1024*1024)}MB"

        if size == 0:
            return False, "Empty file"

        return True, "Valid"

    @staticmethod
    def validate_image_content(file) -> tuple:
        """Validate that file contains valid image data."""
        try:
            import cv2
            import numpy as np

            # Read file bytes
            file.seek(0)
            file_bytes = file.read()
            file.seek(0)

            # Decode image
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return False, "Invalid image data"

            # Check dimensions
            h, w = image.shape[:2]
            if w > FileValidator.MAX_IMAGE_DIMENSION or h > FileValidator.MAX_IMAGE_DIMENSION:
                return False, f"Image too large. Max dimension: {FileValidator.MAX_IMAGE_DIMENSION}px"

            if w < 100 or h < 100:
                return False, "Image too small. Min dimension: 100px"

            return True, "Valid"

        except Exception as e:
            return False, f"Error validating image: {str(e)}"


# ── Security Headers ──────────────────────────────────────────────────────────

def add_security_headers(response):
    """Add security headers to response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def init_security(app):
    """Initialize security features for the Flask app."""
    # Add security headers to all responses
    app.after_request(add_security_headers)

    # Log security events
    @app.before_request
    def log_request():
        g.request_start_time = time.time()

    @app.after_request
    def log_response(response):
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
            if duration > 5:  # Log slow requests
                logger.warning(f"Slow request: {request.path} took {duration:.2f}s")
        return response
