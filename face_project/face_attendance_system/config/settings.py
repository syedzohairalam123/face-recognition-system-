"""
Application Configuration
-------------------------
Central configuration for the Face Recognition Attendance System.
Supports environment variables with sensible defaults.
"""

import os
from pathlib import Path

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FACE_DATA_DIR = DATA_DIR / "face_data"
UPLOAD_DIR = DATA_DIR / "uploads"

# Check if running on Vercel serverless (no persistent filesystem)
IS_VERCEL = os.environ.get("VERCEL", "0") == "1"

# Only create data directories if NOT on Vercel (Vercel has read-only filesystem)
if not IS_VERCEL:
    DATA_DIR.mkdir(exist_ok=True)
    FACE_DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


class Config:
    """Base configuration."""

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "face-attendance-dev-key-change-in-production")
    DEBUG = False
    TESTING = False

    # Database
    # On Vercel, use /tmp for SQLite (ephemeral but writable)
    # For production, set DATABASE_URL to a cloud database (e.g., PostgreSQL on Supabase/Neon)
    if IS_VERCEL:
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            "DATABASE_URL",
            "sqlite:////tmp/attendance.db"
        )
    else:
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{DATA_DIR / 'attendance.db'}"
        )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Face Recognition
    FACE_DETECTION_MODEL = os.environ.get("FACE_DETECTION_MODEL", "hog")  # hog or cnn
    FACE_RECOGNITION_TOLERANCE = float(os.environ.get("FACE_RECOGNITION_TOLERANCE", "0.5"))
    FACE_RECOGNITION_MODEL = os.environ.get("FACE_RECOGNITION_MODEL", "small")  # small or large
    FACE_MIN_FACE_SIZE = int(os.environ.get("FACE_MIN_FACE_SIZE", "40"))
    FACE_JITTERS = int(os.environ.get("FACE_JITTERS", "100"))

    # Decision Engine - Advanced Multi-Signal Decision System
    DECISION_MIN_FACE_SIMILARITY = float(os.environ.get("DECISION_MIN_FACE_SIMILARITY", "0.5"))
    DECISION_MIN_LIVENESS_SCORE = float(os.environ.get("DECISION_MIN_LIVENESS_SCORE", "0.6"))
    DECISION_MIN_FACE_QUALITY = float(os.environ.get("DECISION_MIN_FACE_QUALITY", "0.4"))
    DECISION_MIN_CANDIDATE_MARGIN = float(os.environ.get("DECISION_MIN_CANDIDATE_MARGIN", "0.1"))
    DECISION_HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("DECISION_HIGH_CONFIDENCE_THRESHOLD", "0.85"))
    DECISION_LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("DECISION_LOW_CONFIDENCE_THRESHOLD", "0.60"))
    DECISION_WEIGHT_SIMILARITY = float(os.environ.get("DECISION_WEIGHT_SIMILARITY", "0.35"))
    DECISION_WEIGHT_LIVENESS = float(os.environ.get("DECISION_WEIGHT_LIVENESS", "0.25"))
    DECISION_WEIGHT_QUALITY = float(os.environ.get("DECISION_WEIGHT_QUALITY", "0.15"))
    DECISION_WEIGHT_MARGIN = float(os.environ.get("DECISION_WEIGHT_MARGIN", "0.15"))
    DECISION_WEIGHT_DETECTION = float(os.environ.get("DECISION_WEIGHT_DETECTION", "0.10"))
    DECISION_REQUIRE_LIVENESS = os.environ.get("DECISION_REQUIRE_LIVENESS", "true").lower() == "true"
    DECISION_LIVENESS_MIN_CHECKS = int(os.environ.get("DECISION_LIVENESS_MIN_CHECKS", "3"))

    # Candidate Margin Analysis
    CANDIDATE_MARGIN_CLEAR_THRESHOLD = float(os.environ.get("CANDIDATE_MARGIN_CLEAR_THRESHOLD", "0.15"))

    # Attendance
    ATTENDANCE_WINDOW_MINUTES = int(os.environ.get("ATTENDANCE_WINDOW_MINUTES", "60"))
    ATTENDANCE_GRACE_PERIOD_SECONDS = int(os.environ.get("ATTENDANCE_GRACE_PERIOD_SECONDS", "30"))
    # Check-ins after this time (HH:MM) are marked as 'late'
    ATTENDANCE_LATE_AFTER = os.environ.get("ATTENDANCE_LATE_AFTER", "09:15")

    # Security / Auth
    API_KEY = os.environ.get("API_KEY")  # When set, mutating API endpoints require X-API-Key
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    # Camera (not available on Vercel serverless)
    CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
    CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
    CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))

    # Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # Notifications
    NOTIFICATIONS_ENABLED = os.environ.get("NOTIFICATIONS_ENABLED", "false").lower() == "true"
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "localhost")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "attendance@company.com")
    ADMIN_NOTIFICATION_EMAILS = os.environ.get("ADMIN_NOTIFICATION_EMAILS", "")

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE = DATA_DIR / "app.log"


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

# ── Module-level aliases for backward compatibility ─────────────────────────
# These allow importing directly: from config.settings import FACE_RECOGNITION_TOLERANCE
FACE_DETECTION_MODEL = Config.FACE_DETECTION_MODEL
FACE_RECOGNITION_TOLERANCE = Config.FACE_RECOGNITION_TOLERANCE
FACE_RECOGNITION_MODEL = Config.FACE_RECOGNITION_MODEL
FACE_MIN_FACE_SIZE = Config.FACE_MIN_FACE_SIZE
FACE_JITTERS = Config.FACE_JITTERS
ATTENDANCE_WINDOW_MINUTES = Config.ATTENDANCE_WINDOW_MINUTES
ATTENDANCE_LATE_AFTER = Config.ATTENDANCE_LATE_AFTER
CAMERA_INDEX = Config.CAMERA_INDEX
CAMERA_WIDTH = Config.CAMERA_WIDTH
CAMERA_HEIGHT = Config.CAMERA_HEIGHT
CAMERA_FPS = Config.CAMERA_FPS


def get_config():
    """Return the configuration class based on environment."""
    env = os.environ.get("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
