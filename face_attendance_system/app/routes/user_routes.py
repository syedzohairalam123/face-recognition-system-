"""
User Routes
-----------
Routes for user/employee CRUD operations and face registration.
"""

import logging
import os
import uuid
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, jsonify, session,
)

from app.services.user_service import UserService
from app.utils.helpers import allowed_file, save_uploaded_file
from config.settings import UPLOAD_DIR, FACE_DATA_DIR

logger = logging.getLogger(__name__)
user_bp = Blueprint("user", __name__)


@user_bp.before_request
def require_auth():
    """
    Protect user-management routes: authentication required for all.
    Admin role required only for write actions (add, edit, delete, etc.).
    Read-only actions (view list, view user) available to all logged-in users.
    """
    if not session.get("user_id"):
        flash("Please log in to access this page.", "warning")
        return redirect(url_for("main.login", next=request.path))

    # Admin-only actions: POST methods (edit, delete, deactivate, register face, etc.)
    # GET requests to list/view are allowed for all logged-in users
    if request.method == "POST" and session.get("user_role") != "admin":
        return render_template("errors/403.html"), 403

    return None


@user_bp.route("/")
def list_users():
    """List all users with optional search and status filters."""
    search = request.args.get("search", "").strip()
    status_filter = request.args.get("status", "all")  # all | active | inactive | face_pending

    users = UserService.get_all_users(active_only=False)

    if search:
        users = UserService.search_users(search)

    if status_filter == "active":
        users = [u for u in users if u.is_active]
    elif status_filter == "inactive":
        users = [u for u in users if not u.is_active]
    elif status_filter == "face_pending":
        users = [u for u in users if u.is_active and not u.face_registered]

    stats = UserService.get_stats()
    return render_template(
        "users/list.html",
        users=users,
        stats=stats,
        search=search,
        status_filter=status_filter,
    )


@user_bp.route("/add", methods=["GET", "POST"])
def add_user():
    """Add a new user."""
    if request.method == "POST":
        data = {
            "employee_id": request.form.get("employee_id", "").strip(),
            "first_name": request.form.get("first_name", "").strip(),
            "last_name": request.form.get("last_name", "").strip(),
            "email": request.form.get("email", "").strip(),
            "phone": request.form.get("phone", "").strip() or None,
            "department": request.form.get("department", "").strip(),
            "role": request.form.get("role", "employee"),
        }

        user, message = UserService.create_user(data)

        if user:
            flash(message, "success")
            return redirect(url_for("user.list_users"))
        else:
            flash(message, "danger")

    return render_template("users/add.html")


@user_bp.route("/<int:user_id>")
def view_user(user_id):
    """View user details."""
    user = UserService.get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("user.list_users"))

    from app.services.attendance_service import AttendanceService
    attendance_service = AttendanceService()
    records = attendance_service.get_user_records(user_id, limit=20)
    summary = attendance_service.get_user_attendance_summary(user_id)

    return render_template(
        "users/view.html", user=user, records=records, summary=summary
    )


@user_bp.route("/<int:user_id>/edit", methods=["GET", "POST"])
def edit_user(user_id):
    """Edit user information."""
    user = UserService.get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("user.list_users"))

    if request.method == "POST":
        data = {
            "first_name": request.form.get("first_name", "").strip(),
            "last_name": request.form.get("last_name", "").strip(),
            "email": request.form.get("email", "").strip(),
            "phone": request.form.get("phone", "").strip() or None,
            "department": request.form.get("department", "").strip(),
            "role": request.form.get("role", "employee"),
            "is_active": request.form.get("is_active") == "on",
        }

        updated_user, message = UserService.update_user(user_id, data)

        if updated_user:
            flash(message, "success")
            return redirect(url_for("user.view_user", user_id=user_id))
        else:
            flash(message, "danger")

    return render_template("users/edit.html", user=user)


@user_bp.route("/<int:user_id>/deactivate", methods=["POST"])
def deactivate_user(user_id):
    """Soft-delete (deactivate) a user."""
    # Prevent an admin from deactivating their own account
    if session.get("user_id") == user_id:
        flash("You cannot deactivate your own account.", "danger")
        return redirect(url_for("user.list_users"))

    message = UserService.delete_user(user_id)
    flash(message, "info")
    return redirect(url_for("user.list_users"))


@user_bp.route("/<int:user_id>/reactivate", methods=["POST"])
def reactivate_user(user_id):
    """Reactivate a previously deactivated user."""
    user, message = UserService.activate_user(user_id)
    flash(message, "success" if user else "danger")
    return redirect(url_for("user.list_users"))


@user_bp.route("/<int:user_id>/delete", methods=["POST"])
def delete_user(user_id):
    """Permanently delete a user and all related data."""
    # Prevent an admin from deleting their own account
    if session.get("user_id") == user_id:
        flash("You cannot delete your own account while logged in.", "danger")
        return redirect(url_for("user.list_users"))

    success, message = UserService.hard_delete_user(user_id)
    flash(message, "success" if success else "danger")
    return redirect(url_for("user.list_users"))


@user_bp.route("/<int:user_id>/delete-face", methods=["POST"])
def delete_face(user_id):
    """Delete a user's face data."""
    user = UserService.get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("user.list_users"))

    try:
        # Delete face encoding from disk
        from app.vision.face_encoder import FaceEncoder
        encoder = FaceEncoder()
        encoder.delete_encoding(user_id)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Error deleting face encoding: {e}")

    # Delete FaceData records
    from app.models.face_data import FaceData
    FaceData.query.filter_by(user_id=user_id).delete()

    # Update user
    user.face_registered = False
    user.face_data_path = None
    from app.database import db
    db.session.commit()

    flash(f"Face data deleted for {user.full_name}", "info")
    return redirect(url_for("user.view_user", user_id=user_id))


@user_bp.route("/<int:user_id>/register-face", methods=["GET", "POST"])
def register_face(user_id):
    """Register a user's face for recognition."""
    user = UserService.get_user_by_id(user_id)
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("user.list_users"))

    if request.method == "POST":
        files = request.files.getlist("face_images")

        if not files or all(f.filename == "" for f in files):
            flash("Please upload at least one face image.", "danger")
            return render_template("users/register_face.html", user=user)

        saved_paths = []
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # Generate unique filename
                ext = file.filename.rsplit(".", 1)[1].lower()
                unique_name = f"{user.employee_id}_{uuid.uuid4().hex[:8]}.{ext}"
                filepath = os.path.join(str(UPLOAD_DIR), unique_name)
                file.save(filepath)
                saved_paths.append(filepath)

        if not saved_paths:
            flash("No valid image files uploaded. Allowed: PNG, JPG, JPEG.", "danger")
            return render_template("users/register_face.html", user=user)

        # Create face encoding
        try:
            from app.vision.face_encoder import FaceEncoder
            encoder = FaceEncoder()
            encoding_path = encoder.create_user_encoding_from_images(
                saved_paths, user.id, user.employee_id
            )

            if encoding_path:
                success, message = UserService.mark_face_registered(user.id, encoding_path)
                if success:
                    flash(f"Face registered successfully from {len(saved_paths)} image(s)!", "success")

                    # Reload recognizer
                    try:
                        from app.vision.face_recognizer import FaceRecognizer
                        recognizer = FaceRecognizer()
                        recognizer.reload_known_faces()
                    except Exception:
                        pass

                    return redirect(url_for("user.view_user", user_id=user_id))
                else:
                    flash(message, "danger")
            else:
                flash("Could not create face encoding. Ensure faces are clearly visible.", "danger")

        except ImportError as e:
            flash(f"Face recognition library not available: {e}", "danger")
        except Exception as e:
            flash(f"Error during face registration: {str(e)}", "danger")
            logger.error(f"Face registration error for user {user_id}: {e}")

    return render_template("users/register_face.html", user=user)
