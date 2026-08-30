"""
Attendance Service
------------------
Handles attendance recording, duplicate prevention, and reporting.
Implements time-window based duplicate checking.
"""

import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, List, Dict, Tuple

from app.database import db
from app.models.user import User
from app.models.attendance import Attendance
from config.settings import get_config

logger = logging.getLogger(__name__)


class AttendanceService:
    """Service layer for attendance operations."""

    def __init__(self):
        from app.services.attendance_policy import attendance_policy
        self.policy = attendance_policy
        config = get_config()
        self.window_minutes = config.ATTENDANCE_WINDOW_MINUTES
        self.late_after = self.policy.late_after

    def mark_attendance(
        self,
        user_id: int,
        confidence_score: float = None,
        face_image_path: str = None,
    ) -> Tuple[bool, str]:
        """
        Mark attendance for a user with duplicate prevention.

        Args:
            user_id: Database ID of the user
            confidence_score: Recognition confidence (0-1)
            face_image_path: Path to the captured face image

        Returns:
            Tuple of (success boolean, message)
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return False, "User not found"
            if not user.is_active:
                return False, "User account is inactive"

            now = datetime.utcnow()
            window_start = now - timedelta(minutes=self.window_minutes)

            # Check for duplicate within the time window (using policy)
            recent_attendance = Attendance.query.filter(
                Attendance.user_id == user_id,
                Attendance.check_in_time >= window_start,
            ).first()

            if recent_attendance:
                is_dup, dup_msg = self.policy.is_duplicate_check_in(
                    last_check_in_time=recent_attendance.check_in_time,
                    last_check_out_time=recent_attendance.check_out_time,
                    now=now,
                )
                if is_dup:
                    logger.info(
                        f"Duplicate attendance blocked for {user.employee_id}: {dup_msg}"
                    )
                    return False, (
                        f"Already checked in at {recent_attendance.check_in_str}. "
                        f"{dup_msg}"
                    )

            # Classify check-in using policy (present/late/rejected)
            status, reason = self.policy.classify_check_in(now)
            if status == "rejected":
                logger.info(f"Check-in rejected for {user.employee_id}: {reason}")
                return False, reason

            # Create new attendance record
            record = Attendance(
                user_id=user_id,
                check_in_time=now,
                status=status,
                confidence_score=confidence_score,
                face_image_path=face_image_path,
            )

            db.session.add(record)
            db.session.commit()

            logger.info(
                f"Attendance marked: {user.employee_id} at {record.check_in_str} "
                f"(confidence: {confidence_score:.2f})" if confidence_score
                else f"Attendance marked: {user.employee_id} at {record.check_in_str}"
            )

            # Send notifications
            self._send_attendance_notification(user, status, now, confidence_score)

            return True, f"Attendance marked for {user.full_name} at {record.check_in_time.strftime('%H:%M:%S')}"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error marking attendance for user {user_id}: {e}")
            return False, f"Error marking attendance: {str(e)}"

    def check_out(self, user_id: int) -> Tuple[bool, str]:
        """
        Check out a user (mark end of attendance).

        Args:
            user_id: Database ID of the user

        Returns:
            Tuple of (success boolean, message)
        """
        try:
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            record = Attendance.query.filter(
                Attendance.user_id == user_id,
                Attendance.check_in_time >= today_start,
                Attendance.check_out_time.is_(None),
            ).order_by(Attendance.check_in_time.desc()).first()

            if not record:
                return False, "No active check-in found for today"

            # Check eligibility using policy
            can_checkout, reason = self.policy.can_check_out(record.check_in_time, now)
            if not can_checkout:
                return False, reason

            record.check_out_time = now
            record.status = "checked_out"
            db.session.commit()

            duration = record.duration
            logger.info(
                f"Check-out: {record.user.employee_id} at {record.check_out_str} "
                f"(duration: {duration:.0f} min)"
            )
            return True, f"Checked out at {record.check_out_str}. Duration: {duration:.0f} minutes"

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error checking out user {user_id}: {e}")
            return False, f"Error checking out: {str(e)}"

    def _send_attendance_notification(self, user, status, check_in_time, confidence):
        """Send attendance notifications (confirmation + late alert)."""
        try:
            from app.services.notification_service import notification_service

            if not user.email:
                return

            # Always send confirmation
            notification_service.send_attendance_confirmation(
                user_email=user.email,
                user_name=user.full_name,
                employee_id=user.employee_id,
                action="check_in",
                timestamp=check_in_time,
                confidence=confidence,
            )

            # Send late alert if applicable
            if status == "late":
                notification_service.send_late_arrival_alert(
                    user_email=user.email,
                    user_name=user.full_name,
                    employee_id=user.employee_id,
                    check_in_time=check_in_time,
                    late_after=self.policy.late_after.strftime("%H:%M"),
                )

        except Exception as e:
            # Notification failures should not block attendance
            logger.warning(f"Failed to send notification: {e}")

    def get_today_records(self) -> List[Attendance]:
        """Get all attendance records for today."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return Attendance.query.filter(
            Attendance.check_in_time >= today
        ).order_by(Attendance.check_in_time.desc()).all()

    def get_records_by_date(self, date_str: str) -> List[Attendance]:
        """
        Get attendance records for a specific date.

        Args:
            date_str: Date string in YYYY-MM-DD format
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return []

        next_day = target_date + timedelta(days=1)
        return Attendance.query.filter(
            Attendance.check_in_time >= target_date,
            Attendance.check_in_time < next_day,
        ).order_by(Attendance.check_in_time.desc()).all()

    def get_user_records(self, user_id: int, limit: int = 50) -> List[Attendance]:
        """Get recent attendance records for a specific user."""
        return Attendance.query.filter_by(user_id=user_id).order_by(
            Attendance.check_in_time.desc()
        ).limit(limit).all()

    def get_records_by_date_range(self, date_from: str, date_to: str) -> List[Attendance]:
        """
        Get attendance records for a date range.

        Args:
            date_from: Start date string in YYYY-MM-DD format
            date_to: End date string in YYYY-MM-DD format
        """
        try:
            start_date = datetime.strptime(date_from, "%Y-%m-%d")
            end_date = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            logger.error(f"Invalid date format: {date_from} or {date_to}")
            return []

        return Attendance.query.filter(
            Attendance.check_in_time >= start_date,
            Attendance.check_in_time < end_date,
        ).order_by(Attendance.check_in_time.desc()).all()

    def get_attendance_stats(self, date_str: str = None) -> Dict:
        """
        Get attendance statistics for a given date or today.

        Returns:
            Dictionary with attendance statistics
        """
        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                target_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            target_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        next_day = target_date + timedelta(days=1)

        day_filter = (
            Attendance.check_in_time >= target_date,
            Attendance.check_in_time < next_day,
        )

        total_checked_in = Attendance.query.filter(*day_filter).count()

        total_present = Attendance.query.filter(
            *day_filter, Attendance.status == "present"
        ).count()

        total_late = Attendance.query.filter(
            *day_filter, Attendance.status == "late"
        ).count()

        total_checked_out = Attendance.query.filter(
            *day_filter, Attendance.check_out_time.isnot(None),
        ).count()

        total_active_users = User.query.filter_by(is_active=True).count()

        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "total_checked_in": total_checked_in,
            "total_present": total_present,
            "total_late": total_late,
            "total_checked_out": total_checked_out,
            "total_active_users": total_active_users,
            "attendance_rate": (
                round((total_checked_in / total_active_users * 100), 1)
                if total_active_users > 0 else 0
            ),
        }

    def get_user_attendance_summary(self, user_id: int) -> Dict:
        """
        Get an attendance summary for a user (used on the profile page).

        Returns:
            Dictionary with totals for this month and all time.
        """
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        def _count(start=None, status=None):
            filters = [Attendance.user_id == user_id]
            if start is not None:
                filters.append(Attendance.check_in_time >= start)
            if status is not None:
                filters.append(Attendance.status == status)
            return Attendance.query.filter(*filters).count()

        last_record = Attendance.query.filter_by(user_id=user_id).order_by(
            Attendance.check_in_time.desc()
        ).first()

        month_records = Attendance.query.filter(
            Attendance.user_id == user_id,
            Attendance.check_in_time >= month_start,
        ).count()

        month_late = Attendance.query.filter(
            Attendance.user_id == user_id,
            Attendance.check_in_time >= month_start,
            Attendance.status == "late",
        ).count()

        return {
            "total_days": _count(),
            "month_days": month_records,
            "month_late": month_late,
            "last_check_in": last_record.check_in_str if last_record else None,
        }

    def get_daily_report(self, days: int = 7) -> List[Dict]:
        """Get attendance summary for the last N days."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        report = []

        for i in range(days):
            day = today - timedelta(days=i)
            next_day = day + timedelta(days=1)

            count = Attendance.query.filter(
                Attendance.check_in_time >= day,
                Attendance.check_in_time < next_day,
            ).count()

            report.append({
                "date": day.strftime("%Y-%m-%d"),
                "day_name": day.strftime("%A"),
                "count": count,
            })

        return list(reversed(report))
