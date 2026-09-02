"""
API Routes
----------
RESTful API endpoints for external integrations.
"""

import logging
from datetime import datetime
from app.utils.helpers import utcnow
from flask import Blueprint, request, jsonify

from app.database import db
from app.services.user_service import UserService
from app.services.attendance_service import AttendanceService
from app.utils.decorators import api_key_required

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


@api_bp.route("/users", methods=["GET"])
def api_list_users():
    """API: List all users."""
    active_only = request.args.get("active", "true").lower() == "true"
    users = UserService.get_all_users(active_only=active_only)
    return jsonify({
        "success": True,
        "count": len(users),
        "users": [u.to_dict() for u in users],
    })


@api_bp.route("/users/<int:user_id>", methods=["GET"])
def api_get_user(user_id):
    """API: Get a specific user."""
    user = UserService.get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404
    return jsonify({"success": True, "user": user.to_dict()})


@api_bp.route("/users", methods=["POST"])
@api_key_required
def api_create_user():
    """API: Create a new user."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    user, message = UserService.create_user(data)
    if user:
        return jsonify({"success": True, "message": message, "user": user.to_dict()}), 201
    return jsonify({"success": False, "message": message}), 400


@api_bp.route("/users/<int:user_id>", methods=["PUT"])
@api_key_required
def api_update_user(user_id):
    """API: Update a user."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    user, message = UserService.update_user(user_id, data)
    if user:
        return jsonify({"success": True, "message": message, "user": user.to_dict()})
    return jsonify({"success": False, "message": message}), 400


@api_bp.route("/attendance", methods=["GET"])
def api_attendance_today():
    """API: Get today's attendance records."""
    service = AttendanceService()
    date_str = request.args.get("date")

    if date_str:
        records = service.get_records_by_date(date_str)
    else:
        records = service.get_today_records()

    return jsonify({
        "success": True,
        "count": len(records),
        "records": [r.to_dict() for r in records],
    })


@api_bp.route("/attendance/mark", methods=["POST"])
@api_key_required
def api_mark_attendance():
    """API: Mark attendance for a user."""
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"success": False, "message": "user_id is required"}), 400

    service = AttendanceService()
    success, message = service.mark_attendance(
        user_id=data["user_id"],
        confidence_score=data.get("confidence_score"),
    )

    status_code = 200 if success else 400
    return jsonify({"success": success, "message": message}), status_code


@api_bp.route("/attendance/check-out", methods=["POST"])
@api_key_required
def api_check_out():
    """API: Check out a user."""
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"success": False, "message": "user_id is required"}), 400

    service = AttendanceService()
    success, message = service.check_out(data["user_id"])

    status_code = 200 if success else 400
    return jsonify({"success": success, "message": message}), status_code


@api_bp.route("/attendance/stats", methods=["GET"])
def api_attendance_stats():
    """API: Get attendance statistics."""
    date_str = request.args.get("date")
    service = AttendanceService()
    stats = service.get_attendance_stats(date_str)
    return jsonify({"success": True, "stats": stats})


@api_bp.route("/attendance/report", methods=["GET"])
def api_daily_report():
    """API: Get daily attendance report."""
    days = request.args.get("days", 7, type=int)
    service = AttendanceService()
    report = service.get_daily_report(days)
    return jsonify({"success": True, "report": report})


@api_bp.route("/system/status", methods=["GET"])
def api_system_status():
    """
    API: Real-time system component status.
    Checks database connectivity, recognition engine availability,
    and camera accessibility. Used by the dashboard status section.
    """
    status = {"database": {}, "recognition_engine": {}, "camera": {}}

    # ── Database ──────────────────────────────────────────────────────────
    try:
        from sqlalchemy import text
        db.session.execute(text("SELECT 1"))
        status["database"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        status["database"] = {"status": "error", "message": "Connection failed"}

    # ── Recognition Engine ────────────────────────────────────────────────
    try:
        import face_recognition  # noqa: F401
        from app.vision.face_encoder import FaceEncoder
        encoder = FaceEncoder()
        known = len(encoder.get_known_encodings()) if hasattr(encoder, "get_known_encodings") else None
        msg = "Ready" + (f" ({known} known faces)" if known is not None else "")
        status["recognition_engine"] = {"status": "ok", "message": msg}
    except ImportError:
        status["recognition_engine"] = {
            "status": "warning",
            "message": "face_recognition library not installed",
        }
    except Exception as e:
        logger.warning(f"Recognition engine check failed: {e}")
        status["recognition_engine"] = {"status": "warning", "message": "Initialization issue"}

    # ── Camera ────────────────────────────────────────────────────────────
    try:
        from config.settings import Config
        status["camera"] = {
            "status": "ok",
            "message": f"Configured (index {Config.CAMERA_INDEX})",
            "index": Config.CAMERA_INDEX,
        }
    except Exception as e:
        status["camera"] = {"status": "warning", "message": str(e)}

    overall = "ok"
    for comp in status.values():
        if comp.get("status") == "error":
            overall = "degraded"
            break
        if comp.get("status") == "warning":
            overall = "warning"

    return jsonify({
        "success": True,
        "overall": overall,
        "components": status,
        "timestamp": utcnow().isoformat(),
    })


@api_bp.route("/health", methods=["GET"])
def api_health():
    """API: Health check endpoint."""
    from app.services.performance_monitor import performance_monitor
    return jsonify({
        "status": "healthy",
        "service": "Face Recognition Attendance System",
        "performance": performance_monitor.get_health_status(),
    })


@api_bp.route("/policy", methods=["GET"])
def api_attendance_policy():
    """API: Get attendance policy configuration."""
    from app.services.attendance_policy import attendance_policy
    return jsonify({
        "success": True,
        "policy": attendance_policy.get_policy_config(),
    })


@api_bp.route("/liveness", methods=["GET"])
def api_liveness_config():
    """API: Get liveness detection configuration and status."""
    from app.services.liveness_service import liveness_service
    return jsonify({
        "success": True,
        "liveness": liveness_service.get_config(),
    })


@api_bp.route("/notifications", methods=["GET"])
def api_notification_config():
    """API: Get notification service configuration."""
    from app.services.notification_service import notification_service
    return jsonify({
        "success": True,
        "notifications": notification_service.get_config(),
    })


@api_bp.route("/notifications/daily-summary", methods=["POST"])
@api_key_required
def api_send_daily_summary():
    """API: Trigger daily attendance summary email."""
    from app.services.notification_service import notification_service
    from app.services.attendance_service import AttendanceService

    service = AttendanceService()
    stats = service.get_attendance_stats()
    records = service.get_today_records()

    records_data = [r.to_dict() for r in records]

    success = notification_service.send_daily_summary(
        stats=stats,
        records=records_data,
    )

    return jsonify({
        "success": success,
        "message": "Daily summary sent" if success else "Failed to send summary",
    })


@api_bp.route("/performance", methods=["GET"])
def api_performance():
    """API: Get performance metrics."""
    from app.services.performance_monitor import performance_monitor
    return jsonify({
        "success": True,
        "metrics": performance_monitor.get_metrics(),
        "health": performance_monitor.get_health_status(),
    })


@api_bp.route("/performance/frame-skip", methods=["POST"])
@api_key_required
def api_set_frame_skip():
    """API: Configure frame skip interval."""
    data = request.get_json()
    if not data or "interval" not in data:
        return jsonify({"success": False, "message": "interval is required"}), 400

    interval = int(data["interval"])
    if interval < 1 or interval > 30:
        return jsonify({"success": False, "message": "interval must be 1-30"}), 400

    from app.services.performance_monitor import performance_monitor
    performance_monitor.set_skip_interval(interval)
    return jsonify({"success": True, "message": f"Frame skip interval set to {interval}"})


@api_bp.route("/pipeline", methods=["GET"])
def api_pipeline_info():
    """API: Get recognition pipeline configuration."""
    try:
        from app.services.recognition_pipeline import RecognitionPipeline
        pipeline = RecognitionPipeline()
        return jsonify({"success": True, "pipeline": pipeline.get_pipeline_info()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@api_bp.route("/recognize", methods=["POST"])
@api_key_required
def api_recognize():
    """API: Recognize face from uploaded image."""
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"success": False, "message": "No image provided"}), 400

    import os, uuid
    from config.settings import UPLOAD_DIR

    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "jpg"
    unique_name = f"api_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(str(UPLOAD_DIR), unique_name)
    file.save(filepath)

    try:
        from app.services.recognition_pipeline import RecognitionPipeline
        pipeline = RecognitionPipeline(auto_mark_attendance=True)
        results = pipeline.process_image_file(filepath, mark_attendance=True)

        if not results:
            return jsonify({"success": False, "message": "No faces detected"}), 404

        result = results[0]
        return jsonify(result.to_dict())

    except Exception as e:
        logger.error(f"API recognition error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@api_bp.route("/decision-engine/policy", methods=["GET"])
def api_decision_engine_policy():
    """API: Get current decision engine policy configuration."""
    from app.services.decision_engine import decision_engine
    return jsonify({
        "success": True,
        "policy": decision_engine.get_policy().to_dict(),
    })


@api_bp.route("/decision-engine/policy", methods=["PUT"])
@api_key_required
def api_update_decision_engine_policy():
    """API: Update decision engine policy parameters."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    from app.services.decision_engine import decision_engine
    
    # Validate and update policy parameters
    valid_params = {
        'min_face_similarity', 'min_liveness_score', 'min_face_quality',
        'min_candidate_margin', 'high_confidence_threshold', 
        'low_confidence_threshold', 'weight_similarity', 'weight_liveness',
        'weight_quality', 'weight_margin', 'weight_detection',
        'require_liveness', 'liveness_min_checks'
    }
    
    # Filter to only valid parameters
    update_params = {k: v for k, v in data.items() if k in valid_params}
    
    if not update_params:
        return jsonify({"success": False, "message": "No valid parameters provided"}), 400
    
    try:
        decision_engine.update_policy(**update_params)
        return jsonify({
            "success": True,
            "message": "Decision engine policy updated",
            "policy": decision_engine.get_policy().to_dict(),
        })
    except Exception as e:
        logger.error(f"Error updating decision engine policy: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@api_bp.route("/decision-engine/status", methods=["GET"])
def api_decision_engine_status():
    """API: Get decision engine status and statistics."""
    from app.services.decision_engine import decision_engine
    from app.services.candidate_margin import candidate_margin_analyzer
    
    return jsonify({
        "success": True,
        "decision_engine": {
            "policy": decision_engine.get_policy().to_dict(),
            "available": True,
        },
        "candidate_margin": {
            "threshold": candidate_margin_analyzer.clear_margin_threshold,
            "available": True,
        },
    })


@api_bp.route("/decision-engine/evaluate", methods=["POST"])
def api_evaluate_decision():
    """API: Evaluate a decision with custom signals (for testing/analysis)."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No signals provided"}), 400
    
    from app.services.decision_engine import decision_engine, DecisionSignals
    
    try:
        signals = DecisionSignals(
            face_similarity=data.get('face_similarity', 0.0),
            liveness_score=data.get('liveness_score', 0.0),
            face_quality=data.get('face_quality', 0.0),
            candidate_margin=data.get('candidate_margin', 0.0),
            detection_confidence=data.get('detection_confidence', 0.0),
            embedding_stability=data.get('embedding_stability', 1.0),
            raw_distance=data.get('raw_distance', 0.0),
            raw_top_score=data.get('raw_top_score', 0.0),
            raw_second_score=data.get('raw_second_score', 0.0),
            liveness_checks_passed=data.get('liveness_checks_passed', 0),
            liveness_checks_total=data.get('liveness_checks_total', 0),
        )
        
        user_id = data.get('user_id', -1)
        result = decision_engine.make_decision(signals, user_id)
        
        return jsonify({
            "success": True,
            "result": result.to_dict(),
        })
    except Exception as e:
        logger.error(f"Error evaluating decision: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@api_bp.route("/candidate-margin/threshold", methods=["PUT"])
@api_key_required
def api_update_margin_threshold():
    """API: Update candidate margin analyzer threshold."""
    data = request.get_json()
    if not data or "threshold" not in data:
        return jsonify({"success": False, "message": "threshold is required"}), 400
    
    threshold = float(data["threshold"])
    if threshold < 0.0 or threshold > 1.0:
        return jsonify({"success": False, "message": "threshold must be between 0.0 and 1.0"}), 400
    
    from app.services.candidate_margin import candidate_margin_analyzer
    candidate_margin_analyzer.update_threshold(threshold)
    
    return jsonify({
        "success": True,
        "message": f"Candidate margin threshold updated to {threshold}",
        "threshold": threshold,
    })
