"""
Decorators
----------
Custom decorators for route protection, authorization, and validation.
"""

from functools import wraps
from flask import flash, redirect, url_for, request, jsonify, session, render_template


def login_required(f):
    """
    Decorator to ensure a user is authenticated (has an active session).
    Unauthenticated users are redirected to the login page
    (or receive a 401 JSON response for API endpoints).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Authentication required"}), 401
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("main.login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator to restrict access to admin users.

    - Unauthenticated users are redirected to the login page.
    - Authenticated non-admin users receive a 403 forbidden page.
    - API requests receive JSON error responses.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")

        # Not logged in at all → send to login
        if not user_id:
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Authentication required"}), 401
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("main.login", next=request.path))

        # Logged in but not an admin → 403 Forbidden
        if session.get("user_role") != "admin":
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Admin access required"}), 403
            return render_template("errors/403.html"), 403

        return f(*args, **kwargs)
    return decorated_function


def face_registration_required(f):
    """
    Decorator to ensure the logged-in user has registered their face.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from app.services.user_service import UserService

        user_id = session.get("user_id")
        if not user_id:
            flash("Please log in first.", "warning")
            return redirect(url_for("main.login"))

        user = UserService.get_user_by_id(user_id)
        if user and not user.face_registered:
            flash("Please register your face first.", "warning")
            return redirect(url_for("user.register_face", user_id=user_id))
        return f(*args, **kwargs)
    return decorated_function


def api_key_required(f):
    """
    Decorator for API endpoints requiring API key authentication.
    Only enforced when API_KEY is configured; open when unset (dev mode).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from config.settings import Config
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")

        expected_key = getattr(Config, "API_KEY", None)
        if expected_key and api_key != expected_key:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function
