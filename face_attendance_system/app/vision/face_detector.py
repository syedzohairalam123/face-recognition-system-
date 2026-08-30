"""
Face Detector
-------------
Detects faces in images and video frames using OpenCV and face_recognition.
Supports both HOG and CNN detection models.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect and extract faces from images and video frames."""

    def __init__(self, model: str = "hog", min_face_size: int = 40):
        """
        Initialize the face detector.

        Args:
            model: Detection model - 'hog' (CPU, faster) or 'cnn' (GPU, more accurate)
            min_face_size: Minimum face size in pixels
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning(
                "face_recognition library not installed. "
                "Install with: pip install face_recognition"
            )

        self.model = model
        self.min_face_size = min_face_size
        self._face_cascade = None

        # Fallback: OpenCV's Haar cascade if face_recognition not available
        if not FACE_RECOGNITION_AVAILABLE:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Using OpenCV Haar cascade for face detection (fallback)")
            except AttributeError:
                logger.warning("OpenCV cascade not available in headless mode")
                self._face_cascade = None

        logger.info(f"FaceDetector initialized (model={model}, min_size={min_face_size})")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: BGR image (OpenCV format) or RGB numpy array

        Returns:
            List of (top, right, bottom, left) bounding boxes
        """
        if image is None or image.size == 0:
            return []

        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        if FACE_RECOGNITION_AVAILABLE:
            return self._detect_with_face_recognition(rgb_image)
        else:
            return self._detect_with_opencv(image)

    def _detect_with_face_recognition(self, rgb_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        try:
            face_locations = face_recognition.face_locations(
                rgb_image,
                model=self.model,
                number_of_times_to_upsample=1,
            )
            # Filter by minimum size
            valid_faces = []
            for top, right, bottom, left in face_locations:
                width = right - left
                height = bottom - top
                if width >= self.min_face_size and height >= self.min_face_size:
                    valid_faces.append((top, right, bottom, left))

            logger.debug(f"Detected {len(valid_faces)} faces (filtered from {len(face_locations)})")
            return valid_faces

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def _detect_with_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascade (fallback)."""
        if self._face_cascade is None:
            logger.error("No face detection backend available")
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )

            # Convert (x, y, w, h) to (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))

            logger.debug(f"OpenCV detected {len(face_locations)} faces")
            return face_locations

        except Exception as e:
            logger.error(f"OpenCV face detection error: {e}")
            return []

    def detect_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image

        Returns:
            Bounding box of the largest face, or None
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Return the face with the largest area
        return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))

    def extract_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int],
                     padding: int = 20) -> Optional[np.ndarray]:
        """
        Extract a face region from an image with optional padding.

        Args:
            image: Input image
            face_location: (top, right, bottom, left) bounding box
            padding: Extra pixels around the face

        Returns:
            Cropped face image or None
        """
        try:
            top, right, bottom, left = face_location
            h, w = image.shape[:2]

            # Apply padding with bounds checking
            top_pad = max(0, top - padding)
            left_pad = max(0, left - padding)
            bottom_pad = min(h, bottom + padding)
            right_pad = min(w, right + padding)

            face_image = image[top_pad:bottom_pad, left_pad:right_pad]

            if face_image.size == 0:
                return None

            return face_image

        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None

    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.

        Args:
            image: Input image
            faces: List of face bounding boxes
            color: BGR color for the boxes
            thickness: Line thickness

        Returns:
            Image with drawn face boxes
        """
        annotated = image.copy()
        for top, right, bottom, left in faces:
            cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
        return annotated

    def detect_from_file(self, file_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image file.

        Args:
            file_path: Path to the image file

        Returns:
            List of face bounding boxes
        """
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Could not read image: {file_path}")
            return []
        return self.detect_faces(image)

    def detect_from_camera(self, frame: np.ndarray = None, camera_index: int = 0) -> Tuple[Optional[np.ndarray], List]:
        """
        Capture a frame from camera and detect faces.

        Args:
            frame: Optional pre-captured frame
            camera_index: Camera device index

        Returns:
            Tuple of (annotated_frame, face_locations)
        """
        if frame is None:
            cap = cv2.VideoCapture(camera_index)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Could not read from camera")
                return None, []

        faces = self.detect_faces(frame)
        annotated = self.draw_faces(frame, faces)
        return annotated, faces
