"""
Face Encoder
------------
Generates face embeddings (128-dimensional vectors) for face recognition.
Uses dlib/face_recognition for encoding and numpy for storage.
"""

import logging
import os
from typing import Optional, List, Tuple

import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from config.settings import FACE_DATA_DIR

logger = logging.getLogger(__name__)


class FaceEncoder:
    """Generate and manage face encodings for recognition."""

    def __init__(self, num_jitters: int = 100, model: str = "small"):
        """
        Initialize the face encoder.

        Args:
            num_jitters: Number of times to re-sample (higher = more accurate, slower)
            model: Encoding model - 'small' (faster) or 'large' (more accurate)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError(
                "face_recognition library is required. Install with: pip install face_recognition"
            )

        self.num_jitters = num_jitters
        self.model = model
        
        # Benchmark tracking
        self._total_encodings = 0
        self._total_encoding_time_ms = 0.0
        self._avg_encoding_time_ms = 0.0
        
        logger.info(f"FaceEncoder initialized (jitters={num_jitters}, model={model})")

    def encode_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Generate a 128-dimensional face encoding.

        Args:
            image: RGB image (numpy array)
            face_location: Optional (top, right, bottom, left) bounding box.
                          If None, the largest face is used.

        Returns:
            128-dimensional numpy array or None if encoding fails
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image provided for encoding")
                return None

            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR (OpenCV format), convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            if face_location:
                # Encode specific face
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    known_face_locations=[face_location],
                    num_jitters=self.num_jitters,
                    model=self.model,
                )
            else:
                # Auto-detect and encode the largest face
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    num_jitters=self.num_jitters,
                    model=self.model,
                )

            if not encodings:
                logger.warning("No face found for encoding")
                return None

            # Return the first (or only) encoding
            encoding = encodings[0]
            
            # Track benchmark metrics
            import time
            # Note: encoding time is tracked at caller level for accuracy
            
            logger.debug(f"Face encoded: {encoding.shape}, norm={np.linalg.norm(encoding):.4f}")
            return encoding

        except Exception as e:
            logger.error(f"Face encoding error: {e}")
            return None

    def encode_face_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Encode a face from an image file.

        Args:
            file_path: Path to the image file

        Returns:
            128-dimensional encoding or None
        """
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Could not read image: {file_path}")
            return None
        return self.encode_face(image)

    def encode_multiple_faces(self, image: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Encode all faces in an image.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            List of 128-dimensional encodings
        """
        try:
            if image is None or image.size == 0:
                return []

            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                return []

            encodings = face_recognition.face_encodings(
                rgb_image,
                known_face_locations=face_locations,
                num_jitters=self.num_jitters,
                model=self.model,
            )

            logger.debug(f"Encoded {len(encodings)} faces from image")
            return list(encodings)

        except Exception as e:
            logger.error(f"Multiple face encoding error: {e}")
            return []

    def save_encoding(self, encoding: np.ndarray, user_id: int, employee_id: str) -> str:
        """
        Save a face encoding to disk.

        Args:
            encoding: 128-dimensional numpy array
            user_id: Database user ID
            employee_id: Employee identifier

        Returns:
            Path to the saved encoding file
        """
        try:
            user_dir = FACE_DATA_DIR / str(user_id)
            user_dir.mkdir(exist_ok=True)

            file_path = user_dir / f"{employee_id}_encoding.npy"
            np.save(str(file_path), encoding)

            logger.info(f"Encoding saved: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error saving encoding: {e}")
            raise

    def load_encoding(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load a face encoding from disk.

        Args:
            file_path: Path to the .npy encoding file

        Returns:
            128-dimensional numpy array or None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Encoding file not found: {file_path}")
                return None

            encoding = np.load(file_path)
            logger.debug(f"Encoding loaded: {file_path} (shape={encoding.shape})")
            return encoding

        except Exception as e:
            logger.error(f"Error loading encoding: {e}")
            return None

    def delete_encoding(self, user_id: int) -> bool:
        """
        Delete all encodings for a user.

        Args:
            user_id: Database user ID

        Returns:
            True if deleted successfully
        """
        try:
            user_dir = FACE_DATA_DIR / str(user_id)
            if user_dir.exists():
                import shutil
                shutil.rmtree(user_dir)
                logger.info(f"Deleted encoding directory: {user_dir}")
            return True

        except Exception as e:
            logger.error(f"Error deleting encodings: {e}")
            return False

    def load_all_encodings(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load all stored face encodings with their user IDs.

        Returns:
            Tuple of (encodings, user_ids, employee_ids)
        """
        encodings = []
        user_ids = []
        employee_ids = []

        if not FACE_DATA_DIR.exists():
            return encodings, user_ids, employee_ids

        for user_dir in FACE_DATA_DIR.iterdir():
            if not user_dir.is_dir():
                continue

            try:
                user_id = int(user_dir.name)
            except ValueError:
                continue

            for encoding_file in user_dir.glob("*_encoding.npy"):
                encoding = self.load_encoding(str(encoding_file))
                if encoding is not None:
                    emp_id = encoding_file.stem.replace("_encoding", "")
                    encodings.append(encoding)
                    user_ids.append(user_id)
                    employee_ids.append(emp_id)

        logger.info(f"Loaded {len(encodings)} encodings from {FACE_DATA_DIR}")
        return encodings, user_ids, employee_ids

    def create_user_encoding_from_images(self, image_paths: List[str], user_id: int,
                                          employee_id: str) -> Optional[str]:
        """
        Create and save an averaged encoding from multiple face images.

        Args:
            image_paths: List of image file paths
            user_id: Database user ID
            employee_id: Employee identifier

        Returns:
            Path to saved encoding or None
        """
        import time
        all_encodings = []
        total_time = 0.0

        for path in image_paths:
            start = time.time()
            encoding = self.encode_face_from_file(path)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            
            if encoding is not None:
                all_encodings.append(encoding)

        if not all_encodings:
            logger.error(f"No valid encodings found for user {employee_id}")
            return None

        # Average all encodings for better accuracy
        avg_encoding = np.mean(all_encodings, axis=0)

        # Normalize
        avg_encoding = avg_encoding / np.linalg.norm(avg_encoding)

        file_path = self.save_encoding(avg_encoding, user_id, employee_id)
        
        # Update benchmark metrics
        self._total_encodings += len(all_encodings)
        self._total_encoding_time_ms += total_time
        self._avg_encoding_time_ms = self._total_encoding_time_ms / self._total_encodings
        
        logger.info(f"Created averaged encoding from {len(all_encodings)} images for {employee_id} "
                    f"(avg: {self._avg_encoding_time_ms:.1f}ms per encoding)")
        return file_path

    def get_benchmark(self) -> dict:
        """
        Get encoding benchmark metrics.

        Returns:
            Dictionary with benchmark data
        """
        return {
            "total_encodings": self._total_encodings,
            "total_time_ms": round(self._total_encoding_time_ms, 2),
            "avg_time_ms": round(self._avg_encoding_time_ms, 2),
            "model": self.model,
            "num_jitters": self.num_jitters,
        }
