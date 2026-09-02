"""
Database Package
----------------
Provides the SQLAlchemy database instance and migration support.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        # Import models so they are registered with SQLAlchemy
        from app.models.user import User
        from app.models.attendance import Attendance
        from app.models.face_data import FaceData, CameraSource
        db.create_all()
