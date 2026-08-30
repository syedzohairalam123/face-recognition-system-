"""
Face Aligner
------------
Aligns detected faces to a standard orientation using eye positions.
Normalizes rotation so faces are always upright for better recognition accuracy.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Standard face dimensions for aligned output
STANDARD_FACE_WIDTH = 150
STANDARD_FACE_HEIGHT = 150

# Reference eye positions (left_eye, right_eye) for alignment
# These are approximate positions in a standard 150x150 aligned face
LEFT_EYE_POS = (45, 65)
RIGHT_EYE_POS = (105, 65)


class FaceAligner:
    """Align faces to a standard orientation for consistent recognition."""

    def __init__(self, target_width: int = STANDARD_FACE_WIDTH, target_height: int = STANDARD_FACE_HEIGHT):
        """
        Initialize face aligner.

        Args:
            target_width: Width of the aligned face output
            target_height: Height of the aligned face output
        """
        self.target_width = target_width
        self.target_height = target_height

    def align(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Align a detected face to standard orientation.

        Args:
            image: BGR image
            face_location: (top, right, bottom, left) face bounding box

        Returns:
            Aligned face image or None if alignment fails
        """
        try:
            # Get facial landmarks for eye positions
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            landmarks = face_recognition.face_landmarks(rgb_image, [face_location])

            if not landmarks or not landmarks[0]:
                # No landmarks found - do simple crop without alignment
                return self._simple_crop(image, face_location)

            face_landmarks = landmarks[0]

            # Get eye positions
            left_eye = face_landmarks.get("left_eye", [])
            right_eye = face_landmarks.get("right_eye", [])

            if not left_eye or not right_eye:
                return self._simple_crop(image, face_location)

            # Calculate eye centers
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)

            # Calculate angle between eyes
            dx = right_center[0] - left_center[0]
            dy = right_center[1] - left_center[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # Calculate distance between eyes
            eye_dist = np.sqrt(dx**2 + dy**2)

            # Calculate center point between eyes
            eye_center = ((left_center[0] + right_center[0]) / 2,
                          (left_center[1] + right_center[1]) / 2)

            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

            # Calculate new bounding dimensions
            h, w = image.shape[:2]
            cos_a = np.abs(M[0, 0])
            sin_a = np.abs(M[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)

            # Adjust translation
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            # Apply rotation
            rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

            # Calculate crop region centered on the eyes
            crop_center_x = int(eye_center[0] + (new_w - w) / 2)
            crop_center_y = int(eye_center[1] + (new_h - h) / 2)

            # Scale eye distance to target face width
            scale = self.target_width / (eye_dist * 2.5)
            crop_w = int(self.target_width / scale)
            crop_h = int(self.target_height / scale)

            x1 = max(0, crop_center_x - crop_w // 2)
            y1 = max(0, crop_center_y - crop_h // 2)
            x2 = min(new_w, x1 + crop_w)
            y2 = min(new_h, y1 + crop_h)

            aligned = rotated[y1:y2, x1:x2]

            # Resize to standard dimensions
            if aligned.size > 0:
                aligned = cv2.resize(aligned, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_AREA)
                return aligned

            return self._simple_crop(image, face_location)

        except Exception as e:
            logger.warning(f"Alignment failed, using simple crop: {e}")
            return self._simple_crop(image, face_location)

    def _simple_crop(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Simple face crop without alignment (fallback).

        Args:
            image: Input image
            face_location: (top, right, bottom, left) bounding box

        Returns:
            Cropped and resized face or None
        """
        try:
            top, right, bottom, left = face_location
            h, w = image.shape[:2]

            # Add some padding
            pad_x = int((right - left) * 0.2)
            pad_y = int((bottom - top) * 0.2)

            x1 = max(0, left - pad_x)
            y1 = max(0, top - pad_y)
            x2 = min(w, right + pad_x)
            y2 = min(h, bottom + pad_y)

            face = image[y1:y2, x1:x2]

            if face.size > 0:
                return cv2.resize(face, (self.target_width, self.target_height),
                                  interpolation=cv2.INTER_AREA)
            return None

        except Exception as e:
            logger.error(f"Simple crop failed: {e}")
            return None

    def align_from_eyes(self, image: np.ndarray,
                        left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Align face using explicit eye positions.

        Args:
            image: Input image
            left_eye: (x, y) of left eye center
            right_eye: (x, y) of right eye center

        Returns:
            Aligned face image or None
        """
        try:
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))

            eye_center = ((left_eye[0] + right_eye[0]) / 2,
                          (left_eye[1] + right_eye[1]) / 2)

            eye_dist = np.sqrt(dx**2 + dy**2)

            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            h, w = image.shape[:2]
            cos_a = np.abs(M[0, 0])
            sin_a = np.abs(M[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

            crop_center_x = int(eye_center[0] + (new_w - w) / 2)
            crop_center_y = int(eye_center[1] + (new_h - h) / 2)

            scale = self.target_width / (eye_dist * 2.5)
            crop_w = int(self.target_width / scale)
            crop_h = int(self.target_height / scale)

            x1 = max(0, crop_center_x - crop_w // 2)
            y1 = max(0, crop_center_y - crop_h // 2)
            x2 = min(new_w, x1 + crop_w)
            y2 = min(new_h, y1 + crop_h)

            aligned = rotated[y1:y2, x1:x2]

            if aligned.size > 0:
                return cv2.resize(aligned, (self.target_width, self.target_height),
                                  interpolation=cv2.INTER_AREA)
            return None

        except Exception as e:
            logger.error(f"Eye-based alignment failed: {e}")
            return None
