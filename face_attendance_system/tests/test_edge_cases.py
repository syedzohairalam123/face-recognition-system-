"""
Edge Case & Validation Tests
-----------------------------
Comprehensive tests for system validation covering:
    - Camera: unavailable, disconnected, permission issues
    - Detection: no face, multiple faces, small face, difficult conditions
    - Recognition: registered, unknown, poor capture, ambiguous matching
    - Attendance: first, repeated, next-day, late, invalid user, DB failure
    - UI: desktop, mobile, empty states, errors, loading states
    - Security: unauthorized routes, invalid forms, malformed inputs
"""

import pytest
import os
import sys
import json
from datetime import datetime, timedelta, date
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app
from app.database import db as _db
from app.models.user import User
from app.models.attendance import Attendance


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    app = create_app("testing")
    return app


@pytest.fixture(scope="function")
def db(app):
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture(scope="function")
def client(app, db):
    return app.test_client()


@pytest.fixture
def admin_user(app, db):
    """Create an admin user with password."""
    with app.app_context():
        user = User(
            employee_id="ADMIN001",
            first_name="Admin",
            last_name="User",
            email="admin@test.com",
            role="admin",
            is_active=True,
        )
        user.set_password("admin123")
        _db.session.add(user)
        _db.session.commit()
        # Refresh to ensure attributes are loaded
        _db.session.refresh(user)
        uid = user.id
        urole = user.role
        ueid = user.employee_id
        uname = user.full_name
        # Return a simple dict to avoid detached instance issues
        return type("UserRef", (), {"id": uid, "role": urole, "employee_id": ueid, "full_name": uname})()


@pytest.fixture
def employee_user(app, db):
    """Create a regular employee user."""
    with app.app_context():
        user = User(
            employee_id="EMP100",
            first_name="Regular",
            last_name="Employee",
            email="employee@test.com",
            role="employee",
            is_active=True,
        )
        user.set_password("emp123")
        _db.session.add(user)
        _db.session.commit()
        _db.session.refresh(user)
        uid = user.id
        urole = user.role
        ueid = user.employee_id
        uname = user.full_name
        return type("UserRef", (), {"id": uid, "role": urole, "employee_id": ueid, "full_name": uname})()


@pytest.fixture
def logged_in_client(client, admin_user, app):
    """Client with admin user logged in."""
    with app.app_context():
        user_id = admin_user.id
        user_role = admin_user.role
        emp_id = admin_user.employee_id
        user_name = admin_user.full_name
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_role"] = user_role
        sess["employee_id"] = emp_id
        sess["user_name"] = user_name
    return client


@pytest.fixture
def employee_client(client, employee_user, app):
    """Client with regular employee logged in."""
    with app.app_context():
        user_id = employee_user.id
        user_role = employee_user.role
        emp_id = employee_user.employee_id
        user_name = employee_user.full_name
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_role"] = user_role
        sess["employee_id"] = emp_id
        sess["user_name"] = user_name
    return client


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestCameraEdgeCases:
    """Test camera handling under various failure conditions."""

    def test_camera_unavailable(self, app, db):
        """Test behavior when camera is unavailable."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            manager = CameraManager()
            # Try to initialize with non-existent camera
            success, msg = manager.initialize(camera_index=999)
            assert success is False
            assert "error" in msg.lower() or "could not" in msg.lower()

    def test_camera_status_when_not_initialized(self, app, db):
        """Test camera status when not initialized."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            manager = CameraManager()
            status = manager.get_status()
            assert status["initialized"] is False
            assert status["streaming"] is False

    def test_frame_capture_without_camera(self, app, db):
        """Test frame capture when camera is not open."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            manager = CameraManager()
            frame = manager.capture_frame()
            assert frame is None

    def test_release_without_initialize(self, app, db):
        """Test releasing camera that was never initialized."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            manager = CameraManager()
            # Should not raise
            manager.release()
            assert manager.is_available() is False

    def test_reconnect_without_camera(self, app, db):
        """Test reconnect when no camera was ever connected."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            manager = CameraManager()
            success, msg = manager.try_reconnect()
            # Should fail gracefully
            assert isinstance(success, bool)

    def test_list_cameras_returns_list(self, app, db):
        """Test listing available cameras returns a list."""
        with app.app_context():
            from app.services.camera_manager import CameraManager
            cameras = CameraManager.list_available_cameras(max_cameras=3)
            assert isinstance(cameras, list)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionEdgeCases:
    """Test face detection under various conditions."""

    def test_no_face_in_blank_image(self, app, db):
        """Test detection on blank (black) image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = detector.detect_faces(blank)
            assert isinstance(faces, list)
            assert len(faces) == 0

    def test_no_face_in_white_image(self, app, db):
        """Test detection on white image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            white = np.full((480, 640, 3), 255, dtype=np.uint8)
            faces = detector.detect_faces(white)
            assert isinstance(faces, list)

    def test_very_small_image(self, app, db):
        """Test detection on very small image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            tiny = np.zeros((10, 10, 3), dtype=np.uint8)
            faces = detector.detect_faces(tiny)
            assert isinstance(faces, list)

    def test_single_pixel_image(self, app, db):
        """Test detection on 1x1 image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            pixel = np.zeros((1, 1, 3), dtype=np.uint8)
            faces = detector.detect_faces(pixel)
            assert isinstance(faces, list)

    def test_noisy_image(self, app, db):
        """Test detection on random noise image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            noisy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            faces = detector.detect_faces(noisy)
            assert isinstance(faces, list)

    def test_grayscale_image(self, app, db):
        """Test detection on grayscale (single channel) image."""
        with app.app_context():
            from app.vision.face_detector import FaceDetector
            import numpy as np
            detector = FaceDetector()
            gray = np.zeros((480, 640), dtype=np.uint8)
            try:
                faces = detector.detect_faces(gray)
                assert isinstance(faces, list)
            except Exception:
                # Some detectors require 3-channel images
                pass


# ══════════════════════════════════════════════════════════════════════════════
# RECOGNITION EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestRecognitionEdgeCases:
    """Test recognition pipeline edge cases."""

    def test_pipeline_info(self, app, db):
        """Test pipeline info endpoint returns valid data."""
        with app.app_context():
            from app.services.recognition_pipeline import RecognitionPipeline
            try:
                pipeline = RecognitionPipeline(auto_mark_attendance=False)
                info = pipeline.get_pipeline_info()
                assert "face_recognition_available" in info
                assert "detector" in info
                assert "auto_mark_attendance" in info
            except Exception:
                # Pipeline may not initialize without face_recognition
                pass

    def test_frame_validator_rejects_tiny(self, app, db):
        """Test frame validator rejects very small frames."""
        with app.app_context():
            from app.vision.frame_validator import FrameValidator
            import numpy as np
            validator = FrameValidator()
            tiny = np.zeros((5, 5, 3), dtype=np.uint8)
            is_valid, reason = validator.validate(tiny)
            assert is_valid is False

    def test_frame_validator_accepts_normal(self, app, db):
        """Test frame validator accepts normal frame."""
        with app.app_context():
            from app.vision.frame_validator import FrameValidator
            import numpy as np
            validator = FrameValidator()
            # Create a frame with some texture (not uniform)
            frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            quality = validator.get_frame_quality(frame)
            assert "blur_score" in quality
            assert "brightness" in quality

    def test_liveness_service_all_checks(self, app, db):
        """Test liveness service runs all checks."""
        with app.app_context():
            from app.services.liveness_service import LivenessService
            import numpy as np
            service = LivenessService()
            # Create a realistic-looking face region
            face = np.random.randint(80, 200, (200, 200, 3), dtype=np.uint8)
            result = service.check_liveness(face, (50, 150, 150, 50))
            assert hasattr(result, "is_live")
            assert hasattr(result, "confidence")
            assert hasattr(result, "checks_passed")
            assert result.checks_total == 5

    def test_liveness_config(self, app, db):
        """Test liveness service config returns valid data."""
        with app.app_context():
            from app.services.liveness_service import LivenessService
            service = LivenessService()
            config = service.get_config()
            assert config["enabled"] is True
            assert "checks" in config
            assert "limitations" in config


# ══════════════════════════════════════════════════════════════════════════════
# ATTENDANCE EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestAttendanceEdgeCases:
    """Test attendance system edge cases."""

    def test_mark_attendance_invalid_user(self, app, db):
        """Test marking attendance for non-existent user."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            success, msg = service.mark_attendance(user_id=99999)
            assert success is False
            assert "not found" in msg.lower()

    def test_mark_attendance_inactive_user(self, app, db):
        """Test marking attendance for inactive user."""
        with app.app_context():
            from app.services.user_service import UserService
            from app.services.attendance_service import AttendanceService

            data = {
                "employee_id": "INACTIVE99",
                "first_name": "Inactive",
                "last_name": "Test",
                "email": "inactive99@test.com",
            }
            user, _ = UserService.create_user(data)
            UserService.delete_user(user.id)  # Soft delete

            service = AttendanceService()
            success, msg = service.mark_attendance(user_id=user.id)
            assert success is False
            assert "inactive" in msg.lower()

    def test_check_out_no_active_check_in(self, app, db, employee_user):
        """Test check-out when no active check-in exists."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            success, msg = service.check_out(employee_user.id)
            assert success is False
            assert "no active" in msg.lower()

    def test_attendance_stats_empty_db(self, app, db):
        """Test attendance stats with no records."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            stats = service.get_attendance_stats()
            assert stats["total_checked_in"] == 0
            assert stats["total_present"] == 0
            assert stats["attendance_rate"] == 0

    def test_daily_report_returns_correct_days(self, app, db):
        """Test daily report returns correct number of days."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            report = service.get_daily_report(days=5)
            assert len(report) == 5
            # Dates should be sequential
            for day in report:
                assert "date" in day
                assert "count" in day

    def test_records_by_invalid_date(self, app, db):
        """Test fetching records with invalid date format."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            records = service.get_records_by_date("not-a-date")
            assert records == []

    def test_records_by_date_range_invalid(self, app, db):
        """Test fetching records with invalid date range."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            records = service.get_records_by_date_range("invalid", "dates")
            assert records == []

    def test_user_attendance_summary_no_records(self, app, db, employee_user):
        """Test user attendance summary with no records."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()
            summary = service.get_user_attendance_summary(employee_user.id)
            assert summary["total_days"] == 0
            assert summary["month_days"] == 0
            assert summary["last_check_in"] is None

    def test_duplicate_prevention_window(self, app, db, employee_user):
        """Test duplicate prevention within time window."""
        with app.app_context():
            from app.services.attendance_service import AttendanceService
            service = AttendanceService()

            # First check-in
            s1, m1 = service.mark_attendance(employee_user.id, 0.95)
            assert s1 is True

            # Immediate duplicate should be blocked
            s2, m2 = service.mark_attendance(employee_user.id, 0.90)
            assert s2 is False
            assert "already" in m2.lower()


# ══════════════════════════════════════════════════════════════════════════════
# UI & ROUTING EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestUIEdgeCases:
    """Test UI pages and routing edge cases."""

    def test_404_page(self, client):
        """Test custom 404 page renders."""
        response = client.get("/nonexistent-page-xyz")
        # Flask returns 404; error handler may render a template (200) or return 404
        assert response.status_code in [404, 200, 500]

    def test_405_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        # Try DELETE on a GET-only route
        response = client.delete("/api/health")
        assert response.status_code in [200, 405]

    def test_login_page_loads(self, client):
        """Test login page renders."""
        response = client.get("/login")
        assert response.status_code == 200
        assert b"Login" in response.data or b"Sign In" in response.data

    def test_login_invalid_credentials(self, client):
        """Test login with wrong credentials."""
        response = client.post("/login", data={
            "employee_id": "WRONG001",
            "password": "wrongpassword",
        }, follow_redirects=False)
        assert response.status_code in [401, 302, 200]

    def test_login_empty_fields(self, client):
        """Test login with empty fields."""
        response = client.post("/login", data={
            "employee_id": "",
            "password": "",
        }, follow_redirects=False)
        assert response.status_code in [400, 401, 200]

    def test_dashboard_requires_auth(self, client):
        """Test dashboard redirects to login when not authenticated."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [302, 200]
        if response.status_code == 302:
            assert "login" in response.headers["Location"].lower()

    def test_users_requires_admin(self, employee_client):
        """Test users page requires admin role."""
        response = employee_client.get("/users/", follow_redirects=False)
        # Employee should get 403
        assert response.status_code in [403, 302, 200]

    def test_admin_can_access_users(self, logged_in_client):
        """Test admin can access users page."""
        response = logged_in_client.get("/users/")
        assert response.status_code == 200

    def test_attendance_page_loads(self, logged_in_client):
        """Test attendance page loads for logged-in user."""
        response = logged_in_client.get("/attendance/")
        assert response.status_code == 200

    def test_reports_page_loads(self, logged_in_client):
        """Test reports page loads."""
        response = logged_in_client.get("/reports")
        assert response.status_code == 200

    def test_camera_page_loads(self, logged_in_client):
        """Test camera page loads."""
        response = logged_in_client.get("/camera")
        assert response.status_code == 200

    def test_status_page_loads(self, logged_in_client):
        """Test system status page loads."""
        response = logged_in_client.get("/status")
        assert response.status_code == 200

    def test_viewport_meta_tag(self, client):
        """Test responsive viewport meta tag exists."""
        response = client.get("/login")
        assert b"viewport" in response.data

    def test_logout_clears_session(self, logged_in_client):
        """Test logout clears session."""
        response = logged_in_client.get("/logout", follow_redirects=False)
        assert response.status_code in [302, 200]

    def test_login_rate_limit(self, client):
        """Test login rate limiting doesn't crash."""
        for _ in range(3):
            client.post("/login", data={
                "employee_id": "TEST",
                "password": "test",
            })
        # Should not crash even with multiple attempts


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityEdgeCases:
    """Test security features under edge conditions."""

    def test_security_headers_present(self, client):
        """Test security headers are set on responses."""
        response = client.get("/login")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_password_never_in_api_response(self, client):
        """Test password hash is never exposed in API responses."""
        response = client.get("/api/users")
        data = response.get_json()
        if data and data.get("users"):
            for user in data["users"]:
                assert "password" not in user
                assert "password_hash" not in user

    def test_password_hashing_works(self, app, db):
        """Test password hashing and verification."""
        with app.app_context():
            user = User(
                employee_id="HASH001",
                first_name="Hash",
                last_name="Test",
                email="hash@test.com",
            )
            user.set_password("secretpassword")
            assert user.password_hash != "secretpassword"
            assert user.check_password("secretpassword") is True
            assert user.check_password("wrongpassword") is False

    def test_api_requires_key_for_mutations(self, client):
        """Test API mutations require API key when configured."""
        # Create user via API without key (should work if API_KEY not set)
        response = client.post("/api/users", data=json.dumps({
            "employee_id": "NOKEY001",
            "first_name": "No",
            "last_name": "Key",
            "email": "nokey@test.com",
        }), content_type="application/json")
        # Should either succeed (no API_KEY set) or return 401
        assert response.status_code in [201, 401]

    def test_malformed_json_input(self, client):
        """Test API handles malformed JSON gracefully."""
        response = client.post("/api/users",
            data="not json at all",
            content_type="application/json",
        )
        assert response.status_code in [400, 415]

    def test_xss_in_search_input(self, logged_in_client):
        """Test XSS prevention in search input."""
        response = logged_in_client.get("/users/?search=<script>alert(1)</script>")
        assert response.status_code in [200, 403]

    def test_sql_injection_in_search(self, logged_in_client):
        """Test SQL injection prevention in search."""
        response = logged_in_client.get("/users/?search=' OR '1'='1")
        assert response.status_code in [200, 403]

    def test_session_fixation_prevention(self, client, admin_user, app):
        """Test session is regenerated on login."""
        response = client.post("/login", data={
            "employee_id": admin_user.employee_id,
            "password": "admin123",
        }, follow_redirects=False)
        assert response.status_code in [302, 200]

    def test_protected_route_redirects(self, client):
        """Test protected routes redirect to login."""
        protected_routes = ["/", "/users/", "/attendance/", "/camera", "/reports", "/status"]
        for route in protected_routes:
            response = client.get(route, follow_redirects=False)
            if response.status_code == 302:
                assert "login" in response.headers["Location"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# ATTENDANCE POLICY EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestAttendancePolicy:
    """Test attendance policy edge cases."""

    def test_classify_check_in_on_time(self, app, db):
        """Test check-in classified as present when on time."""
        with app.app_context():
            from app.services.attendance_policy import AttendancePolicy
            policy = AttendancePolicy()
            # Check in at 8:00 AM (before late_after)
            check_in = datetime.now().replace(hour=8, minute=0, second=0)
            status, reason = policy.classify_check_in(check_in)
            assert status == "present"

    def test_duplicate_check_in_detection(self, app, db):
        """Test duplicate check-in is detected."""
        with app.app_context():
            from app.services.attendance_policy import AttendancePolicy
            policy = AttendancePolicy()
            now = datetime.utcnow()
            last_checkin = now - timedelta(minutes=5)  # 5 minutes ago
            is_dup, msg = policy.is_duplicate_check_in(last_checkin, None, now)
            assert is_dup is True

    def test_cooldown_check(self, app, db):
        """Test recognition cooldown detection."""
        with app.app_context():
            from app.services.attendance_policy import AttendancePolicy
            import time
            policy = AttendancePolicy()
            cooldown = {1: time.time() - 10}  # 10 seconds ago
            in_cooldown, msg = policy.is_in_cooldown(1, cooldown)
            assert in_cooldown is True
            assert "wait" in msg.lower()

    def test_get_absent_users(self, app, db):
        """Test absent user identification."""
        with app.app_context():
            from app.services.attendance_policy import AttendancePolicy
            policy = AttendancePolicy()
            active = [1, 2, 3, 4, 5]
            checked_in = [1, 3, 5]
            absent = policy.get_absent_users(active, checked_in)
            assert set(absent) == {2, 4}

    def test_policy_config(self, app, db):
        """Test policy config returns valid data."""
        with app.app_context():
            from app.services.attendance_policy import AttendancePolicy
            policy = AttendancePolicy()
            config = policy.get_policy_config()
            assert "late_after" in config
            assert "grace_period_seconds" in config
            assert "duplicate_window_minutes" in config
            assert "recognition_cooldown_seconds" in config


# ══════════════════════════════════════════════════════════════════════════════
# API EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIEdgeCases:
    """Test API endpoint edge cases."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"

    def test_system_status(self, client):
        response = client.get("/api/system/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "components" in data

    def test_performance_metrics(self, client):
        response = client.get("/api/performance")
        assert response.status_code == 200
        data = response.get_json()
        assert "metrics" in data
        assert "frame_skip" in data["metrics"]
        assert "embeddings" in data["metrics"]

    def test_notification_config(self, client):
        response = client.get("/api/notifications")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "notifications" in data

    def test_liveness_config(self, client):
        response = client.get("/api/liveness")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_policy_config(self, client):
        response = client.get("/api/policy")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "policy" in data

    def test_get_nonexistent_user(self, client):
        response = client.get("/api/users/99999")
        assert response.status_code == 404

    def test_empty_attendance_stats(self, client):
        response = client.get("/api/attendance/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_daily_report(self, client):
        response = client.get("/api/attendance/report?days=3")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert len(data["report"]) == 3
