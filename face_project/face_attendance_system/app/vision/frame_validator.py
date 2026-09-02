"""
Frame Validator
--------------
Validates camera frames for quality before face processing.
Checks for blur, brightness, and minimum resolution.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameValidator:
    """Validate frame quality for reliable face recognition."""

    def __init__(
        self,
        min_width: int = 320,
        min_height: int = 240,
        blur_threshold: float = 100.0,
        min_brightness: int = 40,
        max_brightness: int = 220,
    ):
        """
        Initialize frame validator.

        Args:
            min_width: Minimum frame width in pixels
            min_height: Minimum frame height in pixels
            blur_threshold: Laplacian variance threshold (lower = blurrier)
            min_brightness: Minimum average brightness (0-255)
            max_brightness: Maximum average brightness (0-255)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def validate(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Run all validation checks on a frame.

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if frame is None or frame.size == 0:
            return False, "Empty or None frame"

        # Check resolution
        valid, reason = self._check_resolution(frame)
        if not valid:
            return False, reason

        # Check blur
        valid, reason = self._check_blur(frame)
        if not valid:
            return False, reason

        # Check brightness
        valid, reason = self._check_brightness(frame)
        if not valid:
            return False, reason

        return True, "Frame is valid"

    def _check_resolution(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Check if frame meets minimum resolution requirements."""
        h, w = frame.shape[:2]
        if w < self.min_width or h < self.min_height:
            return False, f"Frame too small: {w}x{h} (need {self.min_width}x{self.min_height})"
        return True, ""

    def _check_blur(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Detect blur using Laplacian variance.
        Low variance = blurry image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < self.blur_threshold:
            return False, f"Frame too blurry (score: {laplacian_var:.1f}, need >= {self.blur_threshold})"
        return True, ""

    def _check_brightness(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Check if frame brightness is within acceptable range."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        if avg_brightness < self.min_brightness:
            return False, f"Frame too dark (brightness: {avg_brightness:.1f}, need >= {self.min_brightness})"
        if avg_brightness > self.max_brightness:
            return False, f"Frame too bright (brightness: {avg_brightness:.1f}, need <= {self.max_brightness})"
        return True, ""

    def get_frame_quality(self, frame: np.ndarray) -> dict:
        """
        Get detailed frame quality metrics.

        Returns:
            Dictionary with quality scores
        """
        if frame is None or frame.size == 0:
            return {"valid": False, "error": "Empty frame"}

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        avg_brightness = float(np.mean(gray))

        is_valid, reason = self.validate(frame)

        return {
            "valid": is_valid,
            "reason": reason,
            "width": w,
            "height": h,
            "blur_score": round(laplacian_var, 2),
            "brightness": round(avg_brightness, 1),
        }
