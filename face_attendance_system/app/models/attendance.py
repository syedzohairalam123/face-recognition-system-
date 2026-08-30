"""
Attendance Model
----------------
Stores attendance records with timestamps and metadata.
Supports duplicate prevention via configurable time windows.
Tracks camera source for audit purposes.
"""

from datetime import datetime
from app.database import db


class Attendance(db.Model):
    """Attendance record for a user on a specific date/time."""

    __tablename__ = "attendance"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # Date and time fields
    attendance_date = db.Column(db.Date, nullable=False, index=True)
    check_in_time = db.Column(db.DateTime, nullable=False)
    check_out_time = db.Column(db.DateTime, nullable=True)

    # Status: present, checked_out, late, absent
    status = db.Column(db.String(20), default="present", nullable=False)

    # Recognition metadata
    confidence_score = db.Column(db.Float, nullable=True)
    face_image_path = db.Column(db.String(500), nullable=True)

    # Camera/Source tracking (optional)
    camera_source = db.Column(db.String(100), nullable=True)

    # Additional info
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Indexes for efficient queries
    __table_args__ = (
        db.Index("idx_user_date", "user_id", "attendance_date"),
        db.Index("idx_date_status", "attendance_date", "status"),
    )

    def __init__(self, **kwargs):
        """Auto-set attendance_date from check_in_time if not provided."""
        super().__init__(**kwargs)
        if self.check_in_time and not self.attendance_date:
            self.attendance_date = self.check_in_time.date()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_checked_out(self):
        """Check if the user has checked out."""
        return self.check_out_time is not None

    @property
    def duration(self):
        """Calculate duration of attendance in minutes."""
        if self.check_out_time:
            delta = self.check_out_time - self.check_in_time
            return delta.total_seconds() / 60
        return None

    @property
    def date(self):
        """Return the date of attendance."""
        return self.attendance_date or (self.check_in_time.date() if self.check_in_time else None)

    @property
    def check_in_str(self):
        """Return formatted check-in time."""
        return self.check_in_time.strftime("%Y-%m-%d %H:%M:%S") if self.check_in_time else None

    @property
    def check_out_str(self):
        """Return formatted check-out time."""
        return self.check_out_time.strftime("%Y-%m-%d %H:%M:%S") if self.check_out_time else None

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self):
        """Serialize attendance record to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "employee_id": self.user.employee_id if self.user else None,
            "employee_name": self.user.full_name if self.user else None,
            "department": self.user.department if self.user else None,
            "attendance_date": self.attendance_date.isoformat() if self.attendance_date else None,
            "check_in_time": self.check_in_str,
            "check_out_time": self.check_out_str,
            "status": self.status,
            "confidence_score": round(self.confidence_score, 2) if self.confidence_score else None,
            "duration_minutes": round(self.duration, 1) if self.duration else None,
            "camera_source": self.camera_source,
            "notes": self.notes,
            "date": self.date.isoformat() if self.date else None,
        }

    def __repr__(self):
        return (
            f"<Attendance user={self.user_id} "
            f"date={self.attendance_date} status={self.status}>"
        )
