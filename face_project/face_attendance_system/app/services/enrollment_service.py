"""
Enrollment Service
------------------
Complete face enrollment workflow for Phase 08.

Workflow:
    Select user
    -> Open camera
    -> Detect face
    -> Validate exactly one face
    -> Capture samples
    -> Align face
    -> Generate embeddings
    -> Validate embeddings
    -> Store face representation
    -> Confirm successful enrollment
"""

import logging
import time
import os
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from app.vision.face_detector import FaceDetector
from app.vision.face_aligner import FaceAligner
from app.vision.frame_validator import FrameValidator

try:
    from app.vision.face_encoder import FaceEncoder
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from app.database import db
from app.models.user import User
from app.models.face_data import FaceData
from config.settings import FACE_DATA_DIR, FACE_DETECTION_MODEL, FACE_MIN_FACE_SIZE, FACE_JITTERS, FACE_RECOGNITION_MODEL

logger = logging.getLogger(__name__)


class EnrollmentStatus:
    """Status constants for enrollment process."""
    NOT_STARTED = "not_started"
    CAMERA_READY = "camera_ready"
    DETECTING = "detecting"
    FACE_FOUND = "face_found"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    VALIDATING = "validating"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class EnrollmentService:
    """
    Complete face enrollment service.
    
    Handles the entire workflow from camera capture to face storage.
    """

    def __init__(self):
        """Initialize enrollment service."""
        self.detector = FaceDetector(
            model=FACE_DETECTION_MODEL,
            min_face_size=FACE_MIN_FACE_SIZE,
        )
        self.aligner = FaceAligner()
        self.validator = FrameValidator()
        
        self.encoder = None
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.encoder = FaceEncoder(
                    num_jitters=FACE_JITTERS,
                    model=FACE_RECOGNITION_MODEL,
                )
            except ImportError:
                logger.warning("face_recognition not available - enrollment disabled")
        
        # Enrollment state
        self._current_user_id: Optional[int] = None
        self._status: str = EnrollmentStatus.NOT_STARTED
        self._captured_samples: List[np.ndarray] = []
        self._target_samples: int = 5  # Number of samples to capture
        self._progress: float = 0.0
        
        logger.info("EnrollmentService initialized")

    def start_enrollment(self, user_id: int) -> Tuple[bool, str]:
        """
        Start enrollment process for a user.
        
        Args:
            user_id: Database user ID
            
        Returns:
            Tuple of (success, message)
        """
        try:
            user = db.session.get(User, user_id)
            if not user:
                return False, "User not found"
            
            if not user.is_active:
                return False, "User account is inactive"
            
            self._current_user_id = user_id
            self._captured_samples = []
            self._progress = 0.0
            self._status = EnrollmentStatus.CAMERA_READY
            
            logger.info(f"Enrollment started for user: {user.employee_id}")
            return True, f"Enrollment started for {user.full_name}"
            
        except Exception as e:
            logger.error(f"Error starting enrollment: {e}")
            return False, f"Error: {str(e)}"

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a camera frame for enrollment.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary with processing results
        """
        if self._current_user_id is None:
            return {"status": "error", "message": "No enrollment in progress"}
        
        result = {
            "status": self._status,
            "faces": [],
            "face_count": 0,
            "samples_captured": len(self._captured_samples),
            "target_samples": self._target_samples,
            "progress": self._progress,
        }
        
        # Step 1: Validate frame
        is_valid, reason = self.validator.validate(frame)
        if not is_valid:
            result["status"] = EnrollmentStatus.FAILED
            result["message"] = reason
            return result
        
        # Step 2: Detect faces
        faces = self.detector.detect_faces(frame)
        result["face_count"] = len(faces)
        result["faces"] = [{"location": f} for f in faces]
        
        # Step 3: Validate exactly one face
        if len(faces) == 0:
            result["status"] = EnrollmentStatus.DETECTING
            result["message"] = "No face detected. Please position your face in the camera."
            return result
        
        if len(faces) > 1:
            result["status"] = EnrollmentStatus.DETECTING
            result["message"] = "Multiple faces detected. Please ensure only one face is visible."
            return result
        
        # Step 4: Face found
        face_location = faces[0]
        result["status"] = EnrollmentStatus.FACE_FOUND
        result["message"] = f"Face detected! Samples: {len(self._captured_samples)}/{self._target_samples}"
        
        return result

    def capture_sample(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Capture a face sample from the frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (success, message)
        """
        if self._current_user_id is None:
            return False, "No enrollment in progress"
        
        if len(self._captured_samples) >= self._target_samples:
            return False, f"Already captured {self._target_samples} samples"
        
        try:
            # Detect face
            faces = self.detector.detect_faces(frame)
            
            if len(faces) == 0:
                return False, "No face detected"
            
            if len(faces) > 1:
                return False, "Multiple faces detected"
            
            # Align face
            face_location = faces[0]
            aligned = self.aligner.align(frame, face_location)
            
            if aligned is None:
                # Fallback to simple crop
                top, right, bottom, left = face_location
                h, w = frame.shape[:2]
                pad = int((right - left) * 0.2)
                x1, y1 = max(0, left - pad), max(0, top - pad)
                x2, y2 = min(w, right + pad), min(h, bottom + pad)
                aligned = cv2.resize(frame[y1:y2, x1:x2], (150, 150))
            
            if aligned is None or aligned.size == 0:
                return False, "Could not align face"
            
            # Store sample
            self._captured_samples.append(aligned)
            self._progress = len(self._captured_samples) / self._target_samples
            self._status = EnrollmentStatus.CAPTURING
            
            count = len(self._captured_samples)
            logger.info(f"Sample {count}/{self._target_samples} captured")
            
            if count >= self._target_samples:
                return True, f"All {count} samples captured! Processing..."
            else:
                return True, f"Sample {count}/{self._target_samples} captured"
                
        except Exception as e:
            logger.error(f"Error capturing sample: {e}")
            return False, f"Capture error: {str(e)}"

    def complete_enrollment(self) -> Tuple[bool, str, Optional[Dict]]:
        """
        Complete the enrollment process by generating and storing embeddings.
        
        Returns:
            Tuple of (success, message, enrollment_data)
        """
        if self._current_user_id is None:
            return False, "No enrollment in progress", None
        
        if len(self._captured_samples) < 3:
            return False, f"Not enough samples ({len(self._captured_samples)}/3 minimum)", None
        
        if self.encoder is None:
            return False, "Face recognition library not available", None
        
        try:
            self._status = EnrollmentStatus.PROCESSING
            user = db.session.get(User, self._current_user_id)
            
            if not user:
                return False, "User not found", None
            
            # Generate encodings for all samples
            encodings = []
            for sample in self._captured_samples:
                encoding = self.encoder.encode_face(sample)
                if encoding is not None:
                    encodings.append(encoding)
            
            if len(encodings) < 3:
                return False, f"Could not encode enough faces ({len(encodings)}/3 minimum)", None
            
            # Validate embeddings (check they're similar)
            self._status = EnrollmentStatus.VALIDATING
            if not self._validate_embeddings(encodings):
                return False, "Face samples are too different. Please try again with consistent positioning.", None
            
            # Average encodings for better accuracy
            avg_encoding = np.mean(encodings, axis=0)
            avg_encoding = avg_encoding / np.linalg.norm(avg_encoding)
            
            # Save encoding to disk
            self._status = EnrollmentStatus.STORING
            encoding_path = self.encoder.save_encoding(
                avg_encoding, user.id, user.employee_id
            )
            
            # Store in database
            face_data = FaceData(
                user_id=user.id,
                encoding_path=encoding_path,
                sample_count=len(encodings),
                model_used=self.encoder.model,
                encoding_dimension=128,
                is_primary=True,
            )
            db.session.add(face_data)
            
            # Update user's face registration status
            user.face_registered = True
            user.face_data_path = encoding_path
            db.session.commit()
            
            self._status = EnrollmentStatus.COMPLETED
            
            enrollment_data = {
                "user_id": user.id,
                "employee_id": user.employee_id,
                "full_name": user.full_name,
                "samples_captured": len(self._captured_samples),
                "encodings_generated": len(encodings),
                "encoding_path": encoding_path,
            }
            
            logger.info(f"Enrollment completed for {user.employee_id}: {len(encodings)} encodings")
            
            # Reload recognizer
            try:
                from app.vision.face_recognizer import FaceRecognizer
                recognizer = FaceRecognizer()
                recognizer.reload_known_faces()
            except Exception:
                pass
            
            return True, f"Face registered successfully for {user.full_name}!", enrollment_data
            
        except Exception as e:
            self._status = EnrollmentStatus.FAILED
            logger.error(f"Error completing enrollment: {e}")
            return False, f"Enrollment error: {str(e)}", None

    def _validate_embeddings(self, encodings: List[np.ndarray]) -> bool:
        """
        Validate that encodings are consistent (same person).
        
        Args:
            encodings: List of 128-dim encodings
            
        Returns:
            True if embeddings are consistent
        """
        if len(encodings) < 2:
            return True
        
        # Calculate pairwise distances
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                dist = np.linalg.norm(encodings[i] - encodings[j])
                if dist > 0.6:  # Threshold for same person
                    logger.warning(f"Embedding mismatch: distance={dist:.4f}")
                    return False
        
        return True

    def cancel_enrollment(self):
        """Cancel current enrollment process."""
        self._current_user_id = None
        self._captured_samples = []
        self._progress = 0.0
        self._status = EnrollmentStatus.NOT_STARTED
        logger.info("Enrollment cancelled")

    def get_status(self) -> Dict:
        """Get current enrollment status."""
        return {
            "user_id": self._current_user_id,
            "status": self._status,
            "samples_captured": len(self._captured_samples),
            "target_samples": self._target_samples,
            "progress": round(self._progress * 100, 1),
        }

    def re_enroll(self, user_id: int) -> Tuple[bool, str]:
        """
        Re-enroll a user (replace existing face data).
        
        Args:
            user_id: Database user ID
            
        Returns:
            Tuple of (success, message)
        """
        try:
            user = db.session.get(User, user_id)
            if not user:
                return False, "User not found"
            
            # Delete old face data if exists
            if self.encoder:
                self.encoder.delete_encoding(user_id)
            
            # Delete old FaceData records
            FaceData.query.filter_by(user_id=user_id).delete()
            
            # Update user
            user.face_registered = False
            user.face_data_path = None
            db.session.commit()
            
            # Start new enrollment
            return self.start_enrollment(user_id)
            
        except Exception as e:
            logger.error(f"Error re-enrolling user {user_id}: {e}")
            return False, f"Re-enrollment error: {str(e)}"


# Global enrollment service instance
enrollment_service = EnrollmentService()
