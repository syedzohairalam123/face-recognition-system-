"""
Face Recognizer
---------------
Compares face encodings to identify known users.
Implements face matching with configurable tolerance thresholds.
"""

import logging
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from app.vision.face_detector import FaceDetector
from app.vision.face_encoder import FaceEncoder
from config.settings import (
    FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MODEL,
    FACE_JITTERS,
    FACE_MIN_FACE_SIZE,
)

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Identify users by comparing face encodings."""

    def __init__(self, tolerance: float = None, model: str = None):
        """
        Initialize the face recognizer.

        Args:
            tolerance: Maximum distance for a match (lower = stricter, default 0.5)
            model: Encoding model ('small' or 'large')
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition library not available - recognition disabled")
            self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE
            self.model = model or FACE_RECOGNITION_MODEL
            self.detector = FaceDetector(
                model="hog",
                min_face_size=FACE_MIN_FACE_SIZE,
            )
            self.encoder = FaceEncoder(
                num_jitters=FACE_JITTERS,
                model=self.model,
            )
            self._known_encodings: List[np.ndarray] = []
            self._known_user_ids: List[int] = []
            self._known_employee_ids: List[str] = []
            self._loaded = False
            return

        self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE
        self.model = model or FACE_RECOGNITION_MODEL

        self.detector = FaceDetector(
            model="hog",
            min_face_size=FACE_MIN_FACE_SIZE,
        )
        self.encoder = FaceEncoder(
            num_jitters=FACE_JITTERS,
            model=self.model,
        )

        # Known faces database (loaded from disk)
        self._known_encodings: List[np.ndarray] = []
        self._known_user_ids: List[int] = []
        self._known_employee_ids: List[str] = []

        self._loaded = False
        logger.info(
            f"FaceRecognizer initialized (tolerance={self.tolerance}, model={self.model})"
        )

    def load_known_faces(self) -> int:
        """
        Load all known face encodings from disk.

        Returns:
            Number of known faces loaded
        """
        self._known_encodings, self._known_user_ids, self._known_employee_ids = (
            self.encoder.load_all_encodings()
        )
        self._loaded = True

        count = len(self._known_encodings)
        logger.info(f"Loaded {count} known face(s) for recognition")
        return count

    def reload_known_faces(self) -> int:
        """Force reload all known faces from disk."""
        return self.load_known_faces()

    def recognize_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize all faces in a frame.

        Args:
            frame: BGR image from camera

        Returns:
            List of recognition results, each containing:
            - user_id: Database user ID (or -1 if unknown)
            - employee_id: Employee identifier (or 'Unknown')
            - confidence: Recognition confidence (0-1)
            - face_location: (top, right, bottom, left)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition library not available - cannot recognize")
            return []

        if not self._loaded:
            self.load_known_faces()

        results = []

        # Detect faces
        face_locations = self.detector.detect_faces(frame)

        if not face_locations:
            return results

        # Encode all detected faces
        rgb_frame = __import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=face_locations,
            num_jitters=1,  # Fast encoding for real-time
            model=self.model,
        )

        for encoding, location in zip(encodings, face_locations):
            result = self._match_face(encoding, location)
            results.append(result)

        logger.debug(f"Recognized {len(results)} faces in frame")
        return results

    def _match_face(self, encoding: np.ndarray, face_location: Tuple) -> Dict:
        """
        Match a face encoding against known faces.

        Args:
            encoding: 128-dimensional face encoding
            face_location: Bounding box tuple

        Returns:
            Recognition result dictionary
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "user_id": -1,
                "employee_id": "Unknown",
                "confidence": 0.0,
                "face_location": face_location,
            }

        if len(self._known_encodings) == 0:
            return {
                "user_id": -1,
                "employee_id": "Unknown",
                "confidence": 0.0,
                "face_location": face_location,
            }

        # Compare against all known faces
        distances = face_recognition.face_distance(self._known_encodings, encoding)

        if len(distances) == 0:
            return {
                "user_id": -1,
                "employee_id": "Unknown",
                "confidence": 0.0,
                "face_location": face_location,
            }

        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]

        if best_distance <= self.tolerance:
            # Match found
            confidence = 1.0 - best_distance  # Convert distance to confidence
            return {
                "user_id": self._known_user_ids[best_match_index],
                "employee_id": self._known_employee_ids[best_match_index],
                "confidence": round(confidence, 4),
                "face_location": face_location,
            }
        else:
            # No match
            return {
                "user_id": -1,
                "employee_id": "Unknown",
                "confidence": 0.0,
                "face_location": face_location,
            }

    def identify_user(self, image: np.ndarray) -> Tuple[Optional[int], str, float]:
        """
        Identify a single user from an image.

        Args:
            image: BGR image containing a face

        Returns:
            Tuple of (user_id, employee_id, confidence)
            user_id is -1 if unknown
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return -1, "Face recognition disabled", 0.0

        if not self._loaded:
            self.load_known_faces()

        # Detect largest face
        face_location = self.detector.detect_largest_face(image)
        if face_location is None:
            return -1, "No face detected", 0.0

        # Encode the face
        encoding = self.encoder.encode_face(image, face_location)
        if encoding is None:
            return -1, "Could not encode face", 0.0

        # Match
        result = self._match_face(encoding, face_location)
        return result["user_id"], result["employee_id"], result["confidence"]

    def verify_user(self, image: np.ndarray, user_id: int) -> Tuple[bool, float]:
        """
        Verify if an image matches a specific user.

        Args:
            image: BGR image containing a face
            user_id: Expected user ID

        Returns:
            Tuple of (is_match, confidence)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, 0.0

        if not self._loaded:
            self.load_known_faces()

        # Find the encoding for this user
        target_indices = [
            i for i, uid in enumerate(self._known_user_ids) if uid == user_id
        ]

        if not target_indices:
            return False, 0.0

        # Detect and encode face from image
        face_location = self.detector.detect_largest_face(image)
        if face_location is None:
            return False, 0.0

        encoding = self.encoder.encode_face(image, face_location)
        if encoding is None:
            return False, 0.0

        # Compare against user's known encoding(s)
        target_encodings = [self._known_encodings[i] for i in target_indices]
        distances = face_recognition.face_distance(target_encodings, encoding)

        best_distance = min(distances)
        confidence = 1.0 - best_distance

        is_match = best_distance <= self.tolerance
        logger.info(
            f"Verification for user {user_id}: "
            f"{'MATCH' if is_match else 'NO MATCH'} "
            f"(distance={best_distance:.4f}, confidence={confidence:.4f})"
        )

        return is_match, round(confidence, 4)

    def get_stats(self) -> Dict:
        """Get recognizer statistics."""
        return {
            "known_faces": len(self._known_encodings),
            "tolerance": self.tolerance,
            "model": self.model,
            "loaded": self._loaded,
        }
