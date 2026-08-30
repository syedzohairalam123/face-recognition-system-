"""
Models Package
--------------
SQLAlchemy ORM models for the attendance system.

Tables:
    - User: Employee/user profiles
    - FaceData: Face embeddings and registration data
    - Attendance: Check-in/out records
    - CameraSource: Camera device tracking
"""

from app.models.user import User
from app.models.attendance import Attendance
from app.models.face_data import FaceData, CameraSource

__all__ = ["User", "Attendance", "FaceData", "CameraSource"]
