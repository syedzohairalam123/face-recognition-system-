"""
Helper Functions
----------------
Common utility functions used across the application.
"""

import os
from datetime import datetime
from functools import wraps
from flask import request
from config.settings import Config


def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.

    Args:
        filename: Name of the file to check

    Returns:
        True if the file extension is allowed
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object to string.

    Args:
        dt: Datetime object
        fmt: Format string

    Returns:
        Formatted datetime string
    """
    if dt is None:
        return "N/A"
    return dt.strftime(fmt)


def format_date(dt: datetime) -> str:
    """Format a datetime to date only."""
    return format_datetime(dt, "%Y-%m-%d")


def format_time(dt: datetime) -> str:
    """Format a datetime to time only."""
    return format_datetime(dt, "%H:%M:%S")


def format_duration(minutes: float) -> str:
    """
    Format duration in minutes to human-readable string.

    Args:
        minutes: Duration in minutes

    Returns:
        Formatted string like '2h 30m'
    """
    if minutes is None:
        return "N/A"

    hours = int(minutes // 60)
    mins = int(minutes % 60)

    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def get_client_ip() -> str:
    """Get the client's IP address from the request."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    return request.remote_addr or "unknown"


def save_uploaded_file(file, destination: str) -> str:
    """
    Save an uploaded file to disk.

    Args:
        file: Werkzeug FileStorage object
        destination: Directory to save to

    Returns:
        Path to the saved file
    """
    filename = file.filename
    # Sanitize filename
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    filepath = os.path.join(destination, filename)
    file.save(filepath)
    return filepath


def validate_email(email: str) -> bool:
    """
    Basic email validation.

    Args:
        email: Email address to validate

    Returns:
        True if email format is valid
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def get_status_color(status: str) -> str:
    """
    Get Bootstrap color class for attendance status.

    Args:
        status: Attendance status string

    Returns:
        Bootstrap color class name
    """
    colors = {
        "present": "success",
        "checked_out": "info",
        "absent": "danger",
        "late": "warning",
    }
    return colors.get(status, "secondary")
