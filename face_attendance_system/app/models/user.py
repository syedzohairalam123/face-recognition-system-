"""
User Model
----------
Represents a registered employee/user in the system.
Stores personal information, authentication data, and face registration status.
"""

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.database import db


class User(db.Model):
    """Employee/User model for the face recognition attendance system."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    department = db.Column(db.String(100), nullable=True)
    role = db.Column(db.String(50), default="employee")  # employee, manager, admin
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # Authentication
    password_hash = db.Column(db.String(256), nullable=True)

    # Face registration status
    face_registered = db.Column(db.Boolean, default=False, nullable=False)
    face_data_path = db.Column(db.String(500), nullable=True)  # Legacy: file path

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    attendance_records = db.relationship(
        "Attendance", backref="user", lazy="dynamic", cascade="all, delete-orphan"
    )

    # ── Password Methods ──────────────────────────────────────────────────────

    def set_password(self, password: str):
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    # ── Name Properties ───────────────────────────────────────────────────────

    @property
    def full_name(self):
        """Return the user's full name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def display_name(self):
        """Return a short display name."""
        return f"{self.first_name} {self.last_name[0]}." if self.last_name else self.first_name

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self):
        """Serialize user to dictionary (never expose password_hash)."""
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "department": self.department,
            "role": self.role,
            "is_active": self.is_active,
            "face_registered": self.face_registered,
            "has_password": self.password_hash is not None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<User {self.employee_id}: {self.full_name}>"
