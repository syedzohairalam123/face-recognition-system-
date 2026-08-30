"""
Camera Service
--------------
Handles live camera streaming and real-time face recognition.
Uses Flask-SocketIO for WebSocket communication.
"""

import logging
import time
import threading
from typing import Optional

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
from config.settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    FACE_DETECTION_MODEL,
    FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MODEL,
    FACE_JITTERS,
    FACE_MIN_FACE_SIZE,
)

logger = logging.getLogger(__name__)


class CameraService:
    """Manages camera capture and real-time face recognition."""

    def __init__(self):
        self.frame_validator = FrameValidator()
        self.detector = FaceDetector(
            model=FACE_DETECTION_MODEL,
            min_face_size=FACE_MIN_FACE_SIZE,
        )
        self.aligner = FaceAligner()
        self.recognizer = None

        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = FaceRecognizer(
                    tolerance=FACE_RECOGNITION_TOLERANCE,
                    model=FACE_RECOGNITION_MODEL,
                )
            except ImportError:
                logger.warning("face_recognition library not available - camera recognition disabled")
        else:
            logger.warning("face_recognition library not available - camera recognition disabled")

        self.attendance_service = AttendanceService()

        self._camera = None
        self._is_streaming = False
        self._stream_thread = None
        self._last_frame = None
        self._last_results = []
        self._recognition_cooldown = {}  # user_id -> last_recognition_time
        self._cooldown_seconds = 60  # Don't re-recognize same user within 60 seconds

    def start_camera(self, camera_index: int = None) -> bool:
        """
        Open camera device.

        Args:
            camera_index: Camera device index (default from config)

        Returns:
            True if camera started successfully
        """
        try:
            idx = camera_index if camera_index is not None else CAMERA_INDEX
            self._camera = cv2.VideoCapture(idx)

            if not self._camera.isOpened():
                logger.error(f"Could not open camera {idx}")
                return False

            # Set camera properties
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self._camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

            logger.info(f"Camera {idx} started ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps)")
            return True

        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False

    def stop_camera(self):
        """Stop camera capture."""
        self._is_streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        if self._camera:
            self._camera.release()
            self._camera = None
        logger.info("Camera stopped")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if not self._camera or not self._camera.isOpened():
            return None

        ret, frame = self._camera.read()
        if not ret:
            return None

        return frame

    def recognize_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the recognition pipeline.

        Returns:
            Dictionary with recognition results
        """
        import time as time_module
        start = time_module.time()

        # Step 1: Validate frame
        is_valid, reason = self.frame_validator.validate(frame)
        if not is_valid:
            return {
                "valid_frame": False,
                "reason": reason,
                "faces": [],
                "processing_time_ms": round((time_module.time() - start) * 1000, 2),
            }

        # Step 2: Detect faces
        face_locations = self.detector.detect_faces(frame)
        if not face_locations:
            return {
                "valid_frame": True,
                "faces": [],
                "processing_time_ms": round((time_module.time() - start) * 1000, 2),
            }

        # Step 3: Process each face
        faces = []
        for location in face_locations:
            face_result = self._process_single_face(frame, location)
            if face_result:
                faces.append(face_result)

        return {
            "valid_frame": True,
            "faces": faces,
            "face_count": len(face_locations),
            "processing_time_ms": round((time_module.time() - start) * 1000, 2),
        }

    def _process_single_face(self, frame: np.ndarray, face_location: tuple) -> Optional[dict]:
        """Process a single detected face through alignment, encoding, and matching."""
        try:
            # Align face
            aligned = self.aligner.align(frame, face_location)
            if aligned is None:
                # Fallback to simple crop
                top, right, bottom, left = face_location
                h, w = frame.shape[:2]
                pad = int((right - left) * 0.2)
                x1, y1 = max(0, left - pad), max(0, top - pad)
                x2, y2 = min(w, right + pad), min(h, bottom + pad)
                aligned = cv2.resize(frame[y1:y2, x1:x2], (150, 150))

            if self.recognizer is None or self.recognizer.encoder is None:
                return None

            # Encode face
            encoding = self.recognizer.encoder.encode_face(aligned)
            if encoding is None:
                return None

            # Match against known faces
            if not self.recognizer._loaded:
                self.recognizer.load_known_faces()

            user_id, employee_id, confidence = self._match_face(encoding)

            # Apply cooldown to prevent rapid re-recognition
            now = time.time()
            if user_id > 0:
                last_time = self._recognition_cooldown.get(user_id, 0)
                if now - last_time < self._cooldown_seconds:
                    return {
                        "user_id": user_id,
                        "employee_id": employee_id,
                        "confidence": confidence,
                        "face_location": face_location,
                        "cooldown_active": True,
                    }
                self._recognition_cooldown[user_id] = now

            # Mark attendance
            attendance_marked = False
            attendance_message = ""
            if user_id > 0:
                attendance_marked, attendance_message = self.attendance_service.mark_attendance(
                    user_id=user_id,
                    confidence_score=confidence,
                )

            # Get full name
            full_name = ""
            if user_id > 0:
                from app.models.user import User
                user = User.query.get(user_id)
                if user:
                    full_name = user.full_name

            return {
                "user_id": user_id,
                "employee_id": employee_id,
                "full_name": full_name,
                "confidence": confidence,
                "face_location": face_location,
                "attendance_marked": attendance_marked,
                "attendance_message": attendance_message,
                "cooldown_active": False,
            }

        except Exception as e:
            logger.error(f"Error processing face: {e}")
            return None

    def _match_face(self, encoding: np.ndarray) -> tuple:
        """Match encoding against known faces. Returns (user_id, employee_id, confidence)."""
        if self.recognizer is None:
            return -1, "Unknown", 0.0

        known_encodings = self.recognizer._known_encodings
        known_user_ids = self.recognizer._known_user_ids
        known_employee_ids = self.recognizer._known_employee_ids

        if not known_encodings:
            return -1, "Unknown", 0.0

        distances = np.linalg.norm(np.array(known_encodings) - encoding, axis=1)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= self.recognizer.tolerance:
            confidence = 1.0 - best_distance
            return known_user_ids[best_idx], known_employee_ids[best_idx], round(confidence, 4)

        return -1, "Unknown", 0.0

    def get_annotated_frame(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Draw recognition results on frame.

        Args:
            frame: Original frame
            results: Recognition results from recognize_frame()

        Returns:
            Annotated frame with bounding boxes and labels
        """
        annotated = frame.copy()

        for face in results.get("faces", []):
            top, right, bottom, left = face["face_location"]

            # Choose color based on recognition
            if face.get("user_id", -1) > 0:
                if face.get("attendance_marked"):
                    color = (0, 255, 0)  # Green - attendance marked
                    label = f"{face.get('full_name', face.get('employee_id', ''))}"
                    status = "CHECKED IN"
                elif face.get("cooldown_active"):
                    color = (0, 255, 255)  # Yellow - cooldown
                    label = f"{face.get('full_name', '')}"
                    status = "ALREADY RECOGNIZED"
                else:
                    color = (255, 165, 0)  # Orange - recognized but not marked
                    label = f"{face.get('employee_id', '')}"
                    status = face.get("attendance_message", "")
            else:
                color = (0, 0, 255)  # Red - unknown
                label = "Unknown"
                status = ""

            # Draw bounding box
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(annotated, (left, top - text_h - 10), (left + text_w, top), color, -1)
            cv2.putText(annotated, label, (left, top - 5), font, font_scale, (255, 255, 255), thickness)

            # Draw status below box
            if status:
                cv2.putText(annotated, status, (left, bottom + 20), font, 0.4, color, 1)

        # Draw frame info
        info = f"Faces: {results.get('face_count', 0)} | Time: {results.get('processing_time_ms', 0):.0f}ms"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def is_camera_open(self) -> bool:
        """Check if camera is currently open."""
        return self._camera is not None and self._camera.isOpened()

    def get_camera_info(self) -> dict:
        """Get current camera information."""
        if not self._camera or not self._camera.isOpened():
            return {"open": False}

        return {
            "open": True,
            "width": int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._camera.get(cv2.CAP_PROP_FPS)),
        }
