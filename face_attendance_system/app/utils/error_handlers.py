"""
Error Handlers
--------------
Comprehensive error handling for all application scenarios.
Provides user-friendly messages while keeping technical details in logs.
"""

import logging
import traceback
from functools import wraps
from flask import jsonify, render_template, request, current_app, session

logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, status_code: int = 500, user_message: str = None):
        self.message = message
        self.status_code = status_code
        self.user_message = user_message or message
        super().__init__(self.message)


class CameraError(AppError):
    """Camera-related errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=503, user_message=user_message or "Camera is not available")


class FaceDetectionError(AppError):
    """Face detection errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=422, user_message=user_message or "Could not detect face")


class RecognitionError(AppError):
    """Face recognition errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=422, user_message=user_message or "Recognition failed")


class DatabaseError(AppError):
    """Database-related errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=500, user_message=user_message or "Database error occurred")


class ValidationError(AppError):
    """Input validation errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=400, user_message=user_message or "Invalid input")


class NotFoundError(AppError):
    """Resource not found errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=404, user_message=user_message or "Resource not found")


class DuplicateError(AppError):
    """Duplicate registration/attendance errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=409, user_message=user_message or "Duplicate entry")


class ConfigurationError(AppError):
    """Configuration errors."""
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message, status_code=500, user_message=user_message or "Configuration error")


# ── Error Handler Registration ────────────────────────────────────────────────

def register_error_handlers(app):
    """Register all error handlers with the Flask app."""

    @app.errorhandler(400)
    def bad_request(error):
        logger.warning(f"Bad request: {request.path} - {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Bad request",
                "error": str(error)
            }), 400
        return render_template("errors/400.html", error=error), 400

    @app.errorhandler(401)
    def unauthorized(error):
        logger.warning(f"Unauthorized access: {request.path} (IP: {request.remote_addr})")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Authentication required. Please log in."
            }), 401
        return render_template("errors/403.html", error=error), 401

    @app.errorhandler(403)
    def forbidden(error):
        logger.warning(
            f"Forbidden access: {request.path} by "
            f"{request.remote_addr} (role: {session.get('user_role', 'anonymous')})"
        )
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "You do not have permission to perform this action"
            }), 403
        return render_template("errors/403.html", error=error), 403

    @app.errorhandler(404)
    def not_found(error):
        logger.info(f"Not found: {request.path}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Resource not found"
            }), 404
        return render_template("errors/404.html", error=error), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        logger.warning(f"Method not allowed: {request.method} {request.path}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Method not allowed"
            }), 405
        return render_template("errors/405.html", error=error), 405

    @app.errorhandler(409)
    def conflict(error):
        logger.warning(f"Conflict: {request.path} - {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Conflict - resource already exists"
            }), 409
        return render_template("errors/409.html", error=error), 409

    @app.errorhandler(413)
    def file_too_large(error):
        logger.warning(f"File too large: {request.path}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "File too large. Maximum size is 16MB"
            }), 413
        return render_template("errors/413.html", error=error), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        logger.warning(f"Rate limit exceeded: {request.remote_addr}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Rate limit exceeded. Please try again later"
            }), 429
        return render_template("errors/429.html", error=error), 429

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {request.path} - {str(error)}")
        logger.error(traceback.format_exc())
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Internal server error"
            }), 500
        return render_template("errors/500.html", error=error), 500

    @app.errorhandler(503)
    def service_unavailable(error):
        logger.error(f"Service unavailable: {request.path} - {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": "Service temporarily unavailable"
            }), 503
        return render_template("errors/503.html", error=error), 503

    # Handle custom AppError exceptions
    @app.errorhandler(AppError)
    def handle_app_error(error):
        logger.error(f"AppError: {error.message}")
        if request.path.startswith('/api/'):
            return jsonify({
                "success": False,
                "message": error.user_message
            }), error.status_code
        return render_template("errors/error.html", error=error), error.status_code


# ── Error Handling Decorator ──────────────────────────────────────────────────

def handle_errors(f):
    """Decorator to handle errors in route functions."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AppError as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            if request.path.startswith('/api/'):
                return jsonify({
                    "success": False,
                    "message": "An unexpected error occurred"
                }), 500
            return render_template("errors/500.html", error=e), 500
    return decorated_function


# ── Specific Error Handlers ──────────────────────────────────────────────────

def handle_camera_error(error_type: str, details: str = None):
    """Handle camera-specific errors with user-friendly messages."""
    messages = {
        "not_found": "No camera detected. Please connect a camera.",
        "permission_denied": "Camera permission denied. Please allow camera access in your browser.",
        "in_use": "Camera is being used by another application.",
        "init_failed": "Could not initialize camera. Please try again.",
        "frame_error": "Could not capture frame from camera.",
    }
    user_msg = messages.get(error_type, "Camera error occurred")
    logger.error(f"Camera error ({error_type}): {details}")
    raise CameraError(f"Camera {error_type}: {details}", user_msg)


def handle_face_error(error_type: str, details: str = None):
    """Handle face detection/recognition errors."""
    messages = {
        "no_face": "No face detected in the image. Please ensure your face is clearly visible.",
        "multiple_faces": "Multiple faces detected. Please ensure only one face is in the frame.",
        "poor_quality": "Image quality is too poor. Please improve lighting and try again.",
        "blurry": "Image is too blurry. Please hold still and try again.",
        "dark": "Image is too dark. Please improve lighting.",
        "bright": "Image is too bright. Please reduce lighting.",
        "small": "Face is too small. Please move closer to the camera.",
        "encoding_failed": "Could not process face. Please try again.",
        "recognition_failed": "Could not recognize face. Please register first.",
        "unknown": "Face not recognized. Please register your face.",
    }
    user_msg = messages.get(error_type, "Face processing error")
    logger.error(f"Face error ({error_type}): {details}")
    raise FaceDetectionError(f"Face {error_type}: {details}", user_msg)


def handle_database_error(error_type: str, details: str = None):
    """Handle database errors."""
    messages = {
        "connection": "Database connection failed. Please try again later.",
        "query": "Database query failed. Please try again.",
        "constraint": "Data constraint violation.",
        "timeout": "Database operation timed out. Please try again.",
    }
    user_msg = messages.get(error_type, "Database error occurred")
    logger.error(f"Database error ({error_type}): {details}")
    raise DatabaseError(f"Database {error_type}: {details}", user_msg)


def handle_validation_error(field: str, issue: str):
    """Handle validation errors."""
    message = f"Validation error for {field}: {issue}"
    logger.warning(message)
    raise ValidationError(message, f"Invalid {field}: {issue}")
