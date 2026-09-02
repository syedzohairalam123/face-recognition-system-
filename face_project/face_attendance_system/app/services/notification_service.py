"""
Notification Service
--------------------
Email notification system for attendance events.
All credentials are loaded from environment variables (never hardcoded).

Supported notifications:
    1. Attendance confirmation (check-in / check-out)
    2. Late arrival alert
    3. Admin notification (system events)
    4. Daily attendance summary

Configuration (environment variables):
    MAIL_SERVER   - SMTP server (default: localhost)
    MAIL_PORT     - SMTP port (default: 587)
    MAIL_USE_TLS  - Use TLS (default: True)
    MAIL_USERNAME - SMTP username
    MAIL_PASSWORD - SMTP password
    MAIL_DEFAULT_SENDER - Default sender email
    NOTIFICATIONS_ENABLED - Enable/disable notifications (default: False)
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date
from typing import List, Dict

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Email notification service.

    All credentials come from environment variables.
    Set NOTIFICATIONS_ENABLED=True to activate.
    """

    def __init__(self):
        self.enabled = os.environ.get("NOTIFICATIONS_ENABLED", "false").lower() == "true"
        self.smtp_server = os.environ.get("MAIL_SERVER", "localhost")
        self.smtp_port = int(os.environ.get("MAIL_PORT", "587"))
        self.use_tls = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
        self.username = os.environ.get("MAIL_USERNAME", "")
        self.password = os.environ.get("MAIL_PASSWORD", "")
        self.default_sender = os.environ.get("MAIL_DEFAULT_SENDER", "attendance@company.com")
        self.admin_emails = self._parse_admin_emails()

        if self.enabled and not self.username:
            logger.warning(
                "Notifications enabled but MAIL_USERNAME not set. "
                "Notifications will be logged but not sent."
            )

        logger.info(
            f"NotificationService initialized: "
            f"enabled={self.enabled}, server={self.smtp_server}:{self.smtp_port}"
        )

    @staticmethod
    def _parse_admin_emails() -> list:
        """Parse admin notification emails from env var (comma-separated)."""
        raw = os.environ.get("ADMIN_NOTIFICATION_EMAILS", "")
        if not raw:
            return []
        return [e.strip() for e in raw.split(",") if e.strip()]

    def _send_email(
        self,
        to_emails: List[str],
        subject: str,
        html_body: str,
        text_body: str = None,
    ) -> bool:
        """
        Send an email via SMTP.

        Returns True if sent successfully, False otherwise.
        """
        if not to_emails:
            logger.warning("No recipients for email, skipping")
            return False

        # Build message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.default_sender
        msg["To"] = ", ".join(to_emails)

        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        if not self.enabled:
            logger.info(f"[NOTIFICATION DISABLED] Would send: {subject} to {to_emails}")
            return True

        if not self.username:
            logger.warning(f"SMTP not configured. Would send: {subject}")
            return False

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.default_sender, to_emails, msg.as_string())

            logger.info(f"Email sent: {subject} to {to_emails}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check MAIL_USERNAME and MAIL_PASSWORD.")
            return False
        except smtplib.SMTPConnectError:
            logger.error(f"Could not connect to SMTP server {self.smtp_server}:{self.smtp_port}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    # ── Notification Types ─────────────────────────────────────────────────

    def send_attendance_confirmation(
        self,
        user_email: str,
        user_name: str,
        employee_id: str,
        action: str,  # "check_in" or "check_out"
        timestamp: datetime,
        confidence: float = None,
    ) -> bool:
        """Send attendance confirmation to the user."""
        action_label = "Checked In" if action == "check_in" else "Checked Out"
        time_str = timestamp.strftime("%I:%M %p")
        date_str = timestamp.strftime("%B %d, %Y")

        subject = f"✅ {action_label} Confirmation — {employee_id}"

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #4361ee, #3a0ca3); padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="color: white; margin: 0;">Face Attendance System</h2>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 0 0 10px 10px;">
                <h3 style="color: #2b2d42;">{action_label} Confirmed</h3>
                <p>Hello <strong>{user_name}</strong>,</p>
                <p>Your attendance has been recorded:</p>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 8px; color: #6c757d;">Employee ID</td><td style="padding: 8px; font-weight: bold;">{employee_id}</td></tr>
                    <tr><td style="padding: 8px; color: #6c757d;">Action</td><td style="padding: 8px; font-weight: bold;">{action_label}</td></tr>
                    <tr><td style="padding: 8px; color: #6c757d;">Time</td><td style="padding: 8px; font-weight: bold;">{time_str}</td></tr>
                    <tr><td style="padding: 8px; color: #6c757d;">Date</td><td style="padding: 8px; font-weight: bold;">{date_str}</td></tr>
                    {f'<tr><td style="padding: 8px; color: #6c757d;">Confidence</td><td style="padding: 8px; font-weight: bold;">{confidence*100:.1f}%</td></tr>' if confidence else ''}
                </table>
                <p style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                    This is an automated notification from the Face Attendance System.
                </p>
            </div>
        </div>
        """

        return self._send_email([user_email], subject, html)

    def send_late_arrival_alert(
        self,
        user_email: str,
        user_name: str,
        employee_id: str,
        check_in_time: datetime,
        late_after: str,
        admin_emails: List[str] = None,
    ) -> bool:
        """Send late arrival notification to user and optionally admins."""
        time_str = check_in_time.strftime("%I:%M %p")

        subject = f"⚠️ Late Arrival — {employee_id} ({time_str})"

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #ffd166, #ef476f); padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="color: white; margin: 0;">Late Arrival Alert</h2>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 0 0 10px 10px;">
                <p>Hello <strong>{user_name}</strong>,</p>
                <p>You checked in <strong>after {late_after}</strong> today.</p>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 8px; color: #6c757d;">Employee ID</td><td style="padding: 8px; font-weight: bold;">{employee_id}</td></tr>
                    <tr><td style="padding: 8px; color: #6c757d;">Check-in Time</td><td style="padding: 8px; font-weight: bold; color: #ef476f;">{time_str}</td></tr>
                    <tr><td style="padding: 8px; color: #6c757d;">Expected By</td><td style="padding: 8px; font-weight: bold;">{late_after}</td></tr>
                </table>
                <p style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                    This is an automated notification from the Face Attendance System.
                </p>
            </div>
        </div>
        """

        recipients = [user_email]
        if admin_emails:
            recipients.extend(admin_emails)

        return self._send_email(recipients, subject, html)

    def send_admin_notification(
        self,
        subject: str,
        message: str,
        details: Dict = None,
    ) -> bool:
        """Send a notification to admin users."""
        if not self.admin_emails:
            logger.warning("No admin emails configured for notifications")
            return False

        details_html = ""
        if details:
            details_html = "<table style='width: 100%; border-collapse: collapse;'>"
            for key, value in details.items():
                details_html += f"""
                <tr>
                    <td style="padding: 6px; color: #6c757d;">{key}</td>
                    <td style="padding: 6px; font-weight: bold;">{value}</td>
                </tr>"""
            details_html += "</table>"

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #118ab2, #073b4c); padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="color: white; margin: 0;">🔔 Admin Notification</h2>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 0 0 10px 10px;">
                <h3 style="color: #2b2d42;">{subject}</h3>
                <p>{message}</p>
                {details_html}
                <p style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                    Face Attendance System — Admin Notification
                </p>
            </div>
        </div>
        """

        return self._send_email(self.admin_emails, f"🔔 {subject}", html)

    def send_daily_summary(
        self,
        admin_emails: List[str] = None,
        stats: Dict = None,
        records: List[Dict] = None,
        summary_date: date = None,
    ) -> bool:
        """Send daily attendance summary to admins."""
        target_date = summary_date or date.today()
        date_str = target_date.strftime("%B %d, %Y")

        recipients = admin_emails or self.admin_emails
        if not recipients:
            logger.warning("No recipients for daily summary")
            return False

        stats = stats or {}
        records = records or []

        total_checked_in = stats.get("total_checked_in", 0)
        total_present = stats.get("total_present", 0)
        total_late = stats.get("total_late", 0)
        total_active = stats.get("total_active_users", 0)
        total_absent = total_active - total_checked_in
        rate = stats.get("attendance_rate", 0)

        records_html = ""
        if records:
            records_html = """
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="background: #e9ecef;">
                    <th style="padding: 8px; text-align: left;">Employee</th>
                    <th style="padding: 8px; text-align: left;">Status</th>
                    <th style="padding: 8px; text-align: left;">Check In</th>
                </tr>
            """
            for r in records[:20]:  # Limit to 20
                status_color = "#06d6a0" if r.get("status") == "present" else "#ffd166" if r.get("status") == "late" else "#6c757d"
                records_html += f"""
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #e9ecef;">{r.get('employee_name', '-')}</td>
                    <td style="padding: 6px; border-bottom: 1px solid #e9ecef; color: {status_color}; font-weight: bold;">{r.get('status', '-').replace('_', ' ').title()}</td>
                    <td style="padding: 6px; border-bottom: 1px solid #e9ecef;">{r.get('check_in_time', '-')}</td>
                </tr>
                """
            records_html += "</table>"
            if len(records) > 20:
                records_html += f"<p style='color: #6c757d; font-size: 12px;'>...and {len(records) - 20} more records</p>"

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #4361ee, #3a0ca3); padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="color: white; margin: 0;">📊 Daily Attendance Summary</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">{date_str}</p>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border-radius: 0 0 10px 10px;">
                <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #06d6a0;">
                        <div style="font-size: 24px; font-weight: bold; color: #06d6a0;">{total_present}</div>
                        <div style="color: #6c757d; font-size: 12px;">Present</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #ffd166;">
                        <div style="font-size: 24px; font-weight: bold; color: #ffd166;">{total_late}</div>
                        <div style="color: #6c757d; font-size: 12px;">Late</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #ef476f;">
                        <div style="font-size: 24px; font-weight: bold; color: #ef476f;">{total_absent}</div>
                        <div style="color: #6c757d; font-size: 12px;">Absent</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #4361ee;">
                        <div style="font-size: 24px; font-weight: bold; color: #4361ee;">{rate}%</div>
                        <div style="color: #6c757d; font-size: 12px;">Rate</div>
                    </div>
                </div>
                {records_html}
                <p style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                    Auto-generated daily summary from the Face Attendance System.
                </p>
            </div>
        </div>
        """

        return self._send_email(recipients, f"📊 Daily Attendance Summary — {date_str}", html)

    def get_config(self) -> dict:
        """Get notification configuration (safe for API, no secrets)."""
        return {
            "enabled": self.enabled,
            "smtp_configured": bool(self.username),
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "tls": self.use_tls,
            "admin_email_count": len(self.admin_emails),
            "default_sender": self.default_sender,
        }


# Global notification service instance
notification_service = NotificationService()
