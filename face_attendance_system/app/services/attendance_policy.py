"""
Attendance Policy Service
-------------------------
Centralized business logic for attendance rules.
All attendance decisions flow through this service.

Rules implemented:
    1. Present / Late / Absent determination
    2. Duplicate recognition suppression (cooldown window)
    3. Allowed check-in period (time window)
    4. Grace period for late classification
    5. Check-out handling
    6. Auto-absent marking for users who never checked in

Architecture:
    Recognition → AttendancePolicy → Present / Late / Already Marked / Absent
"""

import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Tuple

from config.settings import get_config

logger = logging.getLogger(__name__)


class AttendancePolicy:
    """
    Centralized attendance rules engine.

    All attendance decisions should go through this class rather than
    being hardcoded in camera or recognition services.
    """

    def __init__(self):
        config = get_config()

        # Duplicate prevention
        self.duplicate_window_minutes = config.ATTENDANCE_WINDOW_MINUTES

        # Late classification
        self.late_after = self._parse_time(
            getattr(config, "ATTENDANCE_LATE_AFTER", "09:15")
        )

        # Grace period (seconds after late_after before marking as late)
        self.grace_period_seconds = int(
            getattr(config, "ATTENDANCE_GRACE_PERIOD_SECONDS", 30)
        )

        # Allowed check-in period
        self.check_in_start = self._parse_time(
            getattr(config, "ATTENDANCE_CHECK_IN_START", "06:00")
        )
        self.check_in_end = self._parse_time(
            getattr(config, "ATTENDANCE_CHECK_IN_END", "23:59")
        )

        # Recognition cooldown (seconds between re-recognition of same user)
        self.recognition_cooldown_seconds = int(
            getattr(config, "RECOGNITION_COOLDOWN_SECONDS", 60)
        )

        logger.info(
            f"AttendancePolicy initialized: "
            f"late_after={self.late_after}, "
            f"grace={self.grace_period_seconds}s, "
            f"window={self.duplicate_window_minutes}min, "
            f"cooldown={self.recognition_cooldown_seconds}s"
        )

    @staticmethod
    def _parse_time(value: str) -> dt_time:
        """Parse 'HH:MM' string into a time object."""
        try:
            hour, minute = value.split(":")
            return dt_time(int(hour), int(minute))
        except (ValueError, AttributeError):
            return dt_time(9, 15)

    # ── Core Decision Methods ─────────────────────────────────────────────

    def classify_check_in(
        self, check_in_time: datetime = None
    ) -> Tuple[str, str]:
        """
        Classify a check-in as 'present' or 'late'.

        Args:
            check_in_time: When the user is checking in (default: now)

        Returns:
            Tuple of (status, reason)
        """
        now = check_in_time or datetime.utcnow()
        current_time = now.time()

        # Check if within allowed check-in period
        if current_time < self.check_in_start:
            return "rejected", "Check-in period has not started yet"

        if current_time > self.check_in_end:
            return "rejected", "Check-in period has ended"

        # Apply grace period: if within grace_seconds after late_after,
        # still count as present
        late_cutoff = self._add_seconds_to_time(self.late_after, self.grace_period_seconds)

        if current_time > late_cutoff:
            return "late", f"Checked in after {self.late_after.strftime('%H:%M')} (with {self.grace_period_seconds}s grace)"

        return "present", "Checked in on time"

    def is_duplicate_check_in(
        self,
        last_check_in_time: Optional[datetime],
        last_check_out_time: Optional[datetime] = None,
        now: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Check if a new check-in would be a duplicate within the time window.

        Args:
            last_check_in_time: When the user last checked in
            last_check_out_time: When the user last checked out (if any)
            now: Current time (default: now)

        Returns:
            Tuple of (is_duplicate, message)
        """
        now = now or datetime.utcnow()

        if last_check_in_time is None:
            return False, "No previous check-in"

        # If user checked out, allow new check-in regardless of window
        if last_check_out_time is not None:
            return False, "User checked out, new check-in allowed"

        # Check if within duplicate window
        window_start = now - timedelta(minutes=self.duplicate_window_minutes)
        if last_check_in_time >= window_start:
            remaining = (last_check_in_time + timedelta(minutes=self.duplicate_window_minutes)) - now
            return True, (
                f"Already checked in. Next check-in allowed in "
                f"{int(remaining.total_seconds() // 60)} minutes"
            )

        return False, "Window elapsed, check-in allowed"

    def should_auto_checkout(
        self,
        check_in_time: datetime,
        now: datetime = None,
        max_hours: float = 12.0,
    ) -> Tuple[bool, str]:
        """
        Determine if a user should be auto-checked-out (e.g., end of day).

        Args:
            check_in_time: When the user checked in
            now: Current time
            max_hours: Maximum attendance duration before auto-checkout

        Returns:
            Tuple of (should_checkout, reason)
        """
        now = now or datetime.utcnow()
        duration = now - check_in_time

        if duration.total_seconds() > max_hours * 3600:
            return True, f"Auto-checkout after {max_hours} hours"

        return False, "Within normal hours"

    def can_check_out(
        self,
        check_in_time: datetime,
        now: datetime = None,
        min_minutes: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Check if a user is eligible for check-out.

        Args:
            check_in_time: When the user checked in
            now: Current time
            min_minutes: Minimum attendance duration before checkout allowed

        Returns:
            Tuple of (can_checkout, reason)
        """
        now = now or datetime.utcnow()
        duration_minutes = (now - check_in_time).total_seconds() / 60

        if duration_minutes < min_minutes:
            return False, f"Must wait {int(min_minutes - duration_minutes)} more minutes before checking out"

        return True, "Eligible for checkout"

    def get_absent_users(
        self,
        active_user_ids: list,
        checked_in_user_ids: list,
    ) -> list:
        """
        Determine which active users are absent today.

        Args:
            active_user_ids: List of all active user IDs
            checked_in_user_ids: List of user IDs who checked in today

        Returns:
            List of user IDs who are absent
        """
        checked_in_set = set(checked_in_user_ids)
        return [uid for uid in active_user_ids if uid not in checked_in_set]

    def get_cooldown_remaining(
        self,
        user_id: int,
        recognition_cooldown: dict,
        now: float = None,
    ) -> float:
        """
        Get remaining cooldown time for a user.

        Args:
            user_id: User ID to check
            recognition_cooldown: Dict of {user_id: last_recognition_timestamp}
            now: Current timestamp

        Returns:
            Remaining seconds (0 if cooldown expired)
        """
        import time
        now = now or time.time()
        last_time = recognition_cooldown.get(user_id, 0)
        elapsed = now - last_time
        remaining = self.recognition_cooldown_seconds - elapsed
        return max(0.0, remaining)

    def is_in_cooldown(
        self,
        user_id: int,
        recognition_cooldown: dict,
        now: float = None,
    ) -> Tuple[bool, str]:
        """
        Check if a user is in recognition cooldown.

        Returns:
            Tuple of (in_cooldown, message)
        """
        remaining = self.get_cooldown_remaining(user_id, recognition_cooldown, now)
        if remaining > 0:
            return True, f"Already recognized. Wait {int(remaining)}s"
        return False, "Cooldown expired"

    # ── Helper Methods ────────────────────────────────────────────────────

    @staticmethod
    def _add_seconds_to_time(t: dt_time, seconds: int) -> dt_time:
        """Add seconds to a time object, clamping to end of day."""
        dt = datetime.combine(datetime.today(), t) + timedelta(seconds=seconds)
        return dt.time()

    def get_policy_config(self) -> dict:
        """Return the current policy configuration for display/API."""
        return {
            "late_after": self.late_after.strftime("%H:%M"),
            "grace_period_seconds": self.grace_period_seconds,
            "duplicate_window_minutes": self.duplicate_window_minutes,
            "recognition_cooldown_seconds": self.recognition_cooldown_seconds,
            "check_in_start": self.check_in_start.strftime("%H:%M"),
            "check_in_end": self.check_in_end.strftime("%H:%M"),
        }


# Global policy instance
attendance_policy = AttendancePolicy()
