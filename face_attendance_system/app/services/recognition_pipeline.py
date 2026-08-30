"""
Recognition Pipeline (Optimized)
---------------------------------
Unified service that orchestrates the complete face recognition flow
with performance monitoring and optimizations.

Performance Optimizations:
    - Frame resizing for faster processing
    - Cached face encodings (loaded once)
    - Efficient numpy operations
    - Minimal database queries during live recognition
    - Skip unnecessary processing steps

Pipeline Flow:
    Camera Frame -> Frame Validation -> Face Detection -> Face Alignment
    -> Face Embedding -> Similarity Matching -> Identity Decision
    -> Attendance Validation -> Attendance Record
"""

import logging
import time
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from app.vision.frame_validator import FrameValidator
from app.vision.face_detector import FaceDetector
from app.vision.face_aligner import FaceAligner

try:
    from app.vision.face_encoder import FaceEncoder
    from app.vision.face_recognizer import FaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from app.services.attendance_service import AttendanceService
from app.services.performance_monitor import performance_monitor, FrameMetrics
from app.services.liveness_service import liveness_service, LivenessResult
from app.services.decision_engine import (
    decision_engine,
    DecisionSignals,
    DecisionResult,
    DecisionType,
    UncertaintyState,
)
from app.services.candidate_margin import candidate_margin_analyzer, MarginResult
from config.settings import (
    FACE_DETECTION_MODEL,
    FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MODEL,
    FACE_JITTERS,
    FACE_MIN_FACE_SIZE,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

# Performance constants
RECOGNITION_COOLDOWN = 60  # seconds between recognitions for same user
FRAME_RESIZE_WIDTH = 640  # resize frame for faster processing
SKIP_FRAMES = 2  # process every Nth frame for live stream


class RecognitionResult:
    """Result of a single face recognition attempt."""

    def __init__(
        self,
        success: bool,
        user_id: int = -1,
        employee_id: str = "Unknown",
        full_name: str = "",
        confidence: float = 0.0,
        attendance_marked: bool = False,
        attendance_message: str = "",
        frame_quality: dict = None,
        processing_time_ms: float = 0.0,
        metrics: dict = None,
        decision_result: DecisionResult = None,
        margin_result: MarginResult = None,
    ):
        self.success = success
        self.user_id = user_id
        self.employee_id = employee_id
        self.full_name = full_name
        self.confidence = confidence
        self.attendance_marked = attendance_marked
        self.attendance_message = attendance_message
        self.frame_quality = frame_quality or {}
        self.processing_time_ms = processing_time_ms
        self.metrics = metrics or {}
        self.decision_result = decision_result
        self.margin_result = margin_result

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "user_id": self.user_id,
            "employee_id": self.employee_id,
            "full_name": self.full_name,
            "confidence": self.confidence,
            "attendance_marked": self.attendance_marked,
            "attendance_message": self.attendance_message,
            "frame_quality": self.frame_quality,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "metrics": self.metrics,
        }
        
        # Add decision engine results if available
        if self.decision_result:
            result["decision"] = self.decision_result.to_dict()
        
        # Add margin analysis if available
        if self.margin_result:
            result["margin_analysis"] = self.margin_result.to_dict()
        
        return result


class RecognitionPipeline:
    """
    Complete face recognition pipeline with performance optimizations.

    Optimizations:
    - Frame resizing for faster detection
    - Cached face encodings (loaded once, reused)
    - Efficient numpy distance calculations
    - Recognition cooldown to prevent rapid re-recognition
    - Minimal database queries
    """

    def __init__(self, auto_mark_attendance: bool = True):
        """
        Initialize the recognition pipeline.

        Args:
            auto_mark_attendance: If True, automatically mark attendance on successful recognition
        """
        self.auto_mark_attendance = auto_mark_attendance

        # Initialize pipeline components
        self.frame_validator = FrameValidator()
        self.detector = FaceDetector(
            model=FACE_DETECTION_MODEL,
            min_face_size=FACE_MIN_FACE_SIZE,
        )
        self.aligner = FaceAligner()
        self.encoder = None
        self.recognizer = None

        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.encoder = FaceEncoder(
                    num_jitters=FACE_JITTERS,
                    model=FACE_RECOGNITION_MODEL,
                )
                self.recognizer = FaceRecognizer(
                    tolerance=FACE_RECOGNITION_TOLERANCE,
                    model=FACE_RECOGNITION_MODEL,
                )
            except ImportError:
                logger.warning("face_recognition library not available - recognition disabled")
        else:
            logger.warning("face_recognition library not available - recognition disabled")

        self.attendance_service = AttendanceService()

        # Performance: cache user names to avoid DB queries
        self._user_name_cache: Dict[int, str] = {}

        # Performance: recognition cooldown
        self._recognition_cooldown: Dict[int, float] = {}

        logger.info("RecognitionPipeline initialized (optimized)")

    def process_frame(self, frame: np.ndarray, mark_attendance: bool = None) -> List[RecognitionResult]:
        """
        Process a camera frame through the complete recognition pipeline.

        Performance Optimizations:
        - Resizes frame for faster detection
        - Uses cached encodings
        - Skips unnecessary processing

        Pipeline Steps:
            1. Frame Validation
            2. Face Detection
            3. Face Alignment
            4. Face Embedding
            5. Similarity Matching
            6. Identity Decision
            7. Attendance Validation
            8. Attendance Record
        """
        overall_start = time.time()
        metrics = FrameMetrics()

        results = []
        should_mark = mark_attendance if mark_attendance is not None else self.auto_mark_attendance

        # Get original dimensions
        h, w = frame.shape[:2]
        metrics.frame_width = w
        metrics.frame_height = h

        # ── Step 1: Frame Validation ──────────────────────────────────────
        t0 = time.time()
        is_valid, reason = self.frame_validator.validate(frame)
        frame_quality = self.frame_validator.get_frame_quality(frame)
        metrics.validation_time_ms = (time.time() - t0) * 1000

        if not is_valid:
            logger.warning(f"Frame rejected: {reason}")
            metrics.total_time_ms = (time.time() - overall_start) * 1000
            performance_monitor.record_frame(metrics)
            return [RecognitionResult(
                success=False,
                frame_quality=frame_quality,
                processing_time_ms=metrics.total_time_ms,
                metrics=metrics.to_dict(),
                decision_result=None,
                margin_result=None,
            )]

        # ── Step 2: Face Detection (with frame resize optimization) ───────
        t0 = time.time()

        # Resize frame for faster detection if too large
        if w > FRAME_RESIZE_WIDTH:
            scale = FRAME_RESIZE_WIDTH / w
            resized = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))
            face_locations = self.detector.detect_faces(resized)
            # Scale back to original coordinates
            face_locations = [
                (int(t/scale), int(r/scale), int(b/scale), int(l/scale))
                for t, r, b, l in face_locations
            ]
        else:
            face_locations = self.detector.detect_faces(frame)

        metrics.detection_time_ms = (time.time() - t0) * 1000

        if not face_locations:
            logger.debug("No faces detected in frame")
            metrics.total_time_ms = (time.time() - overall_start) * 1000
            performance_monitor.record_frame(metrics)
            return [RecognitionResult(
                success=False,
                frame_quality=frame_quality,
                processing_time_ms=metrics.total_time_ms,
                metrics=metrics.to_dict(),
                decision_result=None,
                margin_result=None,
            )]

        metrics.faces_detected = len(face_locations)
        logger.debug(f"Detected {len(face_locations)} face(s)")

        for face_location in face_locations:
            # ── Step 3: Face Crop & Alignment ─────────────────────────────
            t0 = time.time()
            aligned_face = self.aligner.align(frame, face_location)

            if aligned_face is None:
                aligned_face = self._fallback_crop(frame, face_location)
            metrics.alignment_time_ms = (time.time() - t0) * 1000

            if aligned_face is None:
                continue

            # ── Step 4: Face Embedding ────────────────────────────────────
            if self.encoder is None:
                logger.warning("Face encoder not available")
                continue

            t0 = time.time()
            encoding = self.encoder.encode_face(aligned_face)
            metrics.embedding_time_ms = (time.time() - t0) * 1000

            if encoding is None:
                logger.warning("Could not generate face encoding")
                continue

            # ── Step 5 & 6: Similarity Matching & Identity Decision ───────
            t0 = time.time()
            user_id, employee_id, confidence, decision_result, margin_result = self._match_identity(encoding)
            metrics.matching_time_ms = (time.time() - t0) * 1000

            # Get full name from cache (avoid DB query)
            full_name = self._get_user_name(user_id)

            # ── Step 6.5: Liveness / Anti-Spoofing Check ─────────────────
            liveness_result = None
            if user_id > 0 and should_mark:
                t0 = time.time()
                liveness_result = liveness_service.check_liveness(frame, face_location)
                matching_time_extra = (time.time() - t0) * 1000
                metrics.matching_time_ms += matching_time_extra

                # Update decision signals with liveness and quality
                if decision_result:
                    decision_result.signals.liveness_score = liveness_result.confidence
                    decision_result.signals.liveness_checks_passed = liveness_result.checks_passed
                    decision_result.signals.liveness_checks_total = liveness_result.checks_total
                    decision_result.signals.face_quality = frame_quality.get('overall_quality', 0.5)

                if not liveness_result.is_live:
                    logger.warning(
                        f"Liveness check FAILED for {employee_id}: "
                        f"{liveness_result.checks_passed}/{liveness_result.checks_total} checks"
                    )
                    user_id = -1  # Treat as unrecognized
                    attendance_message = (
                        f"Liveness check failed "
                        f"({liveness_result.checks_passed}/{liveness_result.checks_total} checks)"
                    )
            elif decision_result:
                # Even if not marking attendance, update signals with quality
                decision_result.signals.face_quality = frame_quality.get('overall_quality', 0.5)

            # ── Step 7 & 8: Attendance Validation & Record ────────────────
            attendance_marked = False
            attendance_message_final = attendance_message if attendance_message else ""

            if should_mark and user_id > 0:
                # Check cooldown
                now = time.time()
                last_time = self._recognition_cooldown.get(user_id, 0)
                if now - last_time >= RECOGNITION_COOLDOWN:
                    t0 = time.time()
                    attendance_marked, attendance_message_final = self._mark_attendance(
                        user_id, confidence
                    )
                    metrics.attendance_time_ms = (time.time() - t0) * 1000
                    if attendance_marked:
                        self._recognition_cooldown[user_id] = now
                else:
                    remaining = RECOGNITION_COOLDOWN - (now - last_time)
                    attendance_message_final = f"Already recognized. Wait {int(remaining)}s"

            # Merge any pending message
            if not attendance_message_final and attendance_message:
                attendance_message_final = attendance_message

            if user_id > 0:
                metrics.faces_recognized += 1

            # Add liveness info to metrics
            if liveness_result:
                metrics_dict = metrics.to_dict()
                metrics_dict["liveness"] = liveness_result.to_dict()
            else:
                metrics_dict = metrics.to_dict()

            result = RecognitionResult(
                success=user_id > 0,
                user_id=user_id,
                employee_id=employee_id,
                full_name=full_name,
                confidence=confidence,
                attendance_marked=attendance_marked,
                attendance_message=attendance_message_final,
                frame_quality=frame_quality,
                processing_time_ms=(time.time() - overall_start) * 1000,
                metrics=metrics_dict,
                decision_result=decision_result,
                margin_result=margin_result,
            )
            results.append(result)

        # Record performance metrics
        metrics.total_time_ms = (time.time() - overall_start) * 1000
        performance_monitor.record_frame(metrics)

        return results

    def _get_user_name(self, user_id: int) -> str:
        """Get user name from cache or database."""
        if user_id <= 0:
            return ""

        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]

        # Cache miss - query database
        try:
            from app.models.user import User
            user = User.query.get(user_id)
            if user:
                self._user_name_cache[user_id] = user.full_name
                return user.full_name
        except Exception as e:
            logger.error(f"Error fetching user name: {e}")

        return ""

    def process_image_file(self, file_path: str, mark_attendance: bool = None) -> List[RecognitionResult]:
        """Process an image file through the pipeline."""
        frame = cv2.imread(file_path)
        if frame is None:
            logger.error(f"Could not read image: {file_path}")
            return [RecognitionResult(
                success=False,
                attendance_message=f"Could not read image",
            )]

        return self.process_frame(frame, mark_attendance=mark_attendance)

    def process_image_bytes(self, image_bytes: bytes, mark_attendance: bool = None) -> List[RecognitionResult]:
        """Process image bytes through the pipeline."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return [RecognitionResult(
                success=False,
                attendance_message="Could not decode image bytes",
            )]

        return self.process_frame(frame, mark_attendance=mark_attendance)

    def recognize_and_mark(self, file_path: str) -> Tuple[bool, str, Optional[RecognitionResult]]:
        """Convenience method: recognize face and mark attendance."""
        results = self.process_image_file(file_path, mark_attendance=True)

        if not results:
            return False, "No faces detected", None

        result = results[0]

        if not result.success:
            return False, "Face not recognized", result

        if result.attendance_marked:
            return True, result.attendance_message, result
        else:
            return False, result.attendance_message, result

    def _match_identity(self, encoding: np.ndarray) -> Tuple[int, str, float, DecisionResult, MarginResult]:
        """Match face encoding against known faces with advanced decision engine."""
        if self.recognizer is None:
            return -1, "Unknown", 0.0, None, None

        if not self.recognizer._loaded:
            self.recognizer.load_known_faces()

        known_encodings = self.recognizer._known_encodings
        known_user_ids = self.recognizer._known_user_ids
        known_employee_ids = self.recognizer._known_employee_ids

        if not known_encodings:
            return -1, "Unknown", 0.0, None, None

        # Efficient numpy distance calculation
        distances = np.linalg.norm(
            np.array(known_encodings) - encoding, axis=1
        )

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        best_similarity = 1.0 - best_distance

        # Candidate margin analysis
        margin_result = candidate_margin_analyzer.analyze_margin(
            distances=distances,
            user_ids=known_user_ids,
            employee_ids=known_employee_ids,
            tolerance=self.recognizer.tolerance,
        )

        # Build decision signals
        signals = DecisionSignals(
            face_similarity=best_similarity,
            liveness_score=0.0,  # Will be filled in later
            face_quality=0.0,  # Will be filled in later
            candidate_margin=candidate_margin_analyzer.get_margin_signal(margin_result),
            detection_confidence=0.8,  # Default detection confidence
            raw_distance=best_distance,
            raw_top_score=margin_result.top_score,
            raw_second_score=margin_result.second_score,
        )

        # Use decision engine to make decision
        decision_result = decision_engine.make_decision(
            signals=signals,
            user_id=known_user_ids[best_idx] if best_distance <= self.recognizer.tolerance else -1,
        )

        # Return based on decision engine result
        if decision_result.decision == DecisionType.ACCEPT:
            return (
                known_user_ids[best_idx],
                known_employee_ids[best_idx],
                round(best_similarity, 4),
                decision_result,
                margin_result,
            )
        else:
            return -1, "Unknown", 0.0, decision_result, margin_result

    def _mark_attendance(self, user_id: int, confidence: float) -> Tuple[bool, str]:
        """Mark attendance for an identified user."""
        try:
            return self.attendance_service.mark_attendance(
                user_id=user_id,
                confidence_score=confidence,
            )
        except Exception as e:
            logger.error(f"Error marking attendance for user {user_id}: {e}")
            return False, f"Error marking attendance: {str(e)}"

    def _fallback_crop(self, frame: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """Fallback: simple crop without alignment."""
        try:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]
            pad_x = int((right - left) * 0.2)
            pad_y = int((bottom - top) * 0.2)
            x1 = max(0, left - pad_x)
            y1 = max(0, top - pad_y)
            x2 = min(w, right + pad_x)
            y2 = min(h, bottom + pad_y)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                return cv2.resize(face, (150, 150), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        return None

    def get_pipeline_info(self) -> Dict:
        """Get pipeline configuration and performance status."""
        info = {
            "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
            "frame_validator": {
                "min_width": self.frame_validator.min_width,
                "min_height": self.frame_validator.min_height,
                "blur_threshold": self.frame_validator.blur_threshold,
            },
            "detector": {
                "model": self.detector.model,
                "min_face_size": self.detector.min_face_size,
            },
            "liveness": liveness_service.get_config(),
            "auto_mark_attendance": self.auto_mark_attendance,
            "performance": performance_monitor.get_metrics(),
            "health": performance_monitor.get_health_status(),
        }

        if self.encoder:
            info["encoder"] = {
                "jitters": self.encoder.num_jitters,
                "model": self.encoder.model,
            }

        if self.recognizer:
            info["recognizer"] = {
                "tolerance": self.recognizer.tolerance,
                "known_faces": len(self.recognizer._known_encodings),
            }

        return info
