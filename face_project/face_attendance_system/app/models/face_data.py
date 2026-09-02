"""
Face Data Model
---------------
Stores face embeddings and face registration metadata.
Each user can have multiple face samples for better recognition accuracy.
"""

from datetime import datetime
from app.utils.helpers import utcnow
from app.database import db


class FaceData(db.Model):
    """Face data record linking a user to their face representations."""

    __tablename__ = "face_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # Face representation reference
    # Stores the file path to the .npy encoding file
    encoding_path = db.Column(db.String(500), nullable=False)

    # Optional: store a thumbnail of the face for UI display
    thumbnail_path = db.Column(db.String(500), nullable=True)

    # Metadata
    sample_count = db.Column(db.Integer, default=1, nullable=False)
    model_used = db.Column(db.String(50), default="small", nullable=False)
    encoding_dimension = db.Column(db.Integer, default=128, nullable=False)

    # Quality metrics
    avg_confidence = db.Column(db.Float, nullable=True)
    is_primary = db.Column(db.Boolean, default=False, nullable=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    user = db.relationship("User", backref=db.backref("face_data_records", lazy="dynamic", cascade="all, delete-orphan"))

    def to_dict(self):
        """Serialize face data to dictionary (does not expose encoding path in production)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "employee_id": self.user.employee_id if self.user else None,
            "sample_count": self.sample_count,
            "model_used": self.model_used,
            "encoding_dimension": self.encoding_dimension,
            "is_primary": self.is_primary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<FaceData user={self.user_id} samples={self.sample_count}>"


class CameraSource(db.Model):
    """Optional: Track different camera sources/devices."""

    __tablename__ = "camera_sources"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    camera_index = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "camera_index": self.camera_index,
            "is_active": self.is_active,
        }

    def __repr__(self):
        return f"<CameraSource {self.name}>"
