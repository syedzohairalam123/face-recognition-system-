"""
AttendanceService Tests
-----------------------
Unit tests for attendance operations.
"""

import pytest
from datetime import datetime, timedelta
from app.services.attendance_service import AttendanceService
from app.services.user_service import UserService


class TestMarkAttendance:
    """Tests for marking attendance."""

    def test_mark_attendance_success(self, app, db, sample_user):
        """Test successful attendance marking."""
        with app.app_context():
            service = AttendanceService()
            success, message = service.mark_attendance(
                user_id=sample_user.id,
                confidence_score=0.95,
            )

            assert success is True
            assert "success" in message.lower() or "marked" in message.lower()

    def test_mark_attendance_nonexistent_user(self, app, db):
        """Test marking attendance for non-existent user."""
        with app.app_context():
            service = AttendanceService()
            success, message = service.mark_attendance(user_id=9999)

            assert success is False
            assert "not found" in message.lower()

    def test_duplicate_prevention(self, app, db, sample_user):
        """Test that duplicate attendance is prevented within time window."""
        with app.app_context():
            service = AttendanceService()

            # First check-in
            success1, _ = service.mark_attendance(sample_user.id)
            assert success1 is True

            # Immediate duplicate should be blocked
            success2, message2 = service.mark_attendance(sample_user.id)
            assert success2 is False
            assert "already checked in" in message2.lower()


class TestCheckOut:
    """Tests for check-out."""

    def test_check_out_success(self, app, db, sample_user):
        """Test successful check-out."""
        from app.database import db as _db
        from app.models.attendance import Attendance
        from datetime import datetime, timedelta

        with app.app_context():
            # Create a record with check-in time 10 minutes ago
            # to satisfy the minimum duration policy
            record = Attendance(
                user_id=sample_user.id,
                check_in_time=datetime.utcnow() - timedelta(minutes=10),
                status="present",
            )
            _db.session.add(record)
            _db.session.commit()

            service = AttendanceService()
            success, message = service.check_out(sample_user.id)
            assert success is True
            assert "checked out" in message.lower()

    def test_check_out_no_active_checkin(self, app, db, sample_user):
        """Test check-out when no active check-in exists."""
        with app.app_context():
            service = AttendanceService()
            success, message = service.check_out(sample_user.id)

            assert success is False
            assert "no active" in message.lower()


class TestGetRecords:
    """Tests for record retrieval."""

    def test_get_today_records(self, app, db, sample_user):
        """Test getting today's records."""
        with app.app_context():
            service = AttendanceService()
            service.mark_attendance(sample_user.id)

            records = service.get_today_records()
            assert len(records) >= 1

    def test_get_user_records(self, app, db, sample_user):
        """Test getting records for a specific user."""
        with app.app_context():
            service = AttendanceService()
            service.mark_attendance(sample_user.id)

            records = service.get_user_records(sample_user.id)
            assert len(records) >= 1

    def test_get_records_by_date(self, app, db, sample_user):
        """Test getting records for a specific date."""
        with app.app_context():
            service = AttendanceService()
            service.mark_attendance(sample_user.id)

            today = datetime.utcnow().strftime("%Y-%m-%d")
            records = service.get_records_by_date(today)
            assert len(records) >= 1

    def test_get_records_invalid_date(self, app, db):
        """Test getting records with invalid date format."""
        with app.app_context():
            service = AttendanceService()
            records = service.get_records_by_date("not-a-date")
            assert len(records) == 0


class TestStats:
    """Tests for attendance statistics."""

    def test_get_attendance_stats(self, app, db, sample_user):
        """Test getting attendance statistics."""
        with app.app_context():
            service = AttendanceService()
            service.mark_attendance(sample_user.id)

            stats = service.get_attendance_stats()
            assert "total_checked_in" in stats
            assert "total_active_users" in stats
            assert "attendance_rate" in stats
            assert stats["total_checked_in"] >= 1

    def test_get_daily_report(self, app, db, sample_user):
        """Test getting daily report."""
        with app.app_context():
            service = AttendanceService()
            report = service.get_daily_report(days=7)

            assert len(report) == 7
            for day in report:
                assert "date" in day
                assert "count" in day
