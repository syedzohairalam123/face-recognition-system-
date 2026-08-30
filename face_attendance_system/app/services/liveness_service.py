"""
Liveness / Anti-Spoofing Service
--------------------------------
Basic liveness detection to reduce simple presentation attacks.
Distinguishes live persons from photos, screens, and masks.

Approach:
    1. Texture Analysis (Laplacian variance) — detects screen/print artifacts
    2. Face Size Consistency — rejects faces that are suspiciously flat/uniform
    3. Brightness Analysis — detects screen glow vs natural lighting
    4. Edge Analysis — real faces have richer edge patterns than photos
    5. Color Distribution — photos often have unnatural color distributions

Limitations (documented):
    - This is a BASIC liveness check, not a full anti-spoofing system
    - Cannot defeat high-quality printed photos or video replay attacks
    - Cannot detect 3D masks
    - For production use, consider a dedicated liveness model (e.g., FaceLivenessDetection)
    - The current approach reduces simple attacks but does not eliminate them

Architecture:
    Face Recognition (pass) → Liveness Check → Final Decision
    If liveness check fails, attendance is NOT marked even if face is recognized.
"""

import logging
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LivenessResult:
    """Result of a liveness check."""
    is_live: bool
    confidence: float  # 0.0 to 1.0
    checks_passed: int
    checks_total: int
    details: dict
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_live": self.is_live,
            "confidence": round(self.confidence, 3),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "details": self.details,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class LivenessService:
    """
    Basic liveness detection service.

    Performs multiple lightweight checks to determine if a face
    belongs to a live person or a presentation attack (photo, screen).

    Checks:
        1. Texture analysis (Laplacian variance)
        2. Face region brightness
        3. Edge density
        4. Color distribution uniformity
        5. Face size ratio (anti-flat detection)

    Threshold: A face must pass at least 3 out of 5 checks to be considered live.
    """

    # Minimum checks required out of total checks
    MIN_CHECKS_PASSED = 3
    TOTAL_CHECKS = 5

    def __init__(self, min_checks: int = None):
        """
        Initialize liveness service.

        Args:
            min_checks: Minimum number of checks to pass (default: 3)
        """
        self.min_checks = min_checks or self.MIN_CHECKS_PASSED
        logger.info(
            f"LivenessService initialized (min_checks={self.min_checks}/{self.TOTAL_CHECKS})"
        )

    def check_liveness(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
    ) -> LivenessResult:
        """
        Perform liveness detection on a detected face.

        Args:
            frame: Full camera frame (BGR)
            face_location: (top, right, bottom, left) face bounding box

        Returns:
            LivenessResult with is_live, confidence, and details
        """
        start_time = time.time()
        details = {}
        checks_passed = 0

        # Extract face region
        top, right, bottom, left = face_location
        h, w = frame.shape[:2]

        # Add padding for better analysis
        pad_x = int((right - left) * 0.15)
        pad_y = int((bottom - top) * 0.15)
        x1 = max(0, left - pad_x)
        y1 = max(0, top - pad_y)
        x2 = min(w, right + pad_x)
        y2 = min(h, bottom + pad_y)

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                checks_passed=0,
                checks_total=self.TOTAL_CHECKS,
                details={"error": "Face region too small for analysis"},
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # ── Check 1: Texture Analysis (Laplacian Variance) ──────────────
        texture_score, texture_detail = self._check_texture(face_region)
        details["texture"] = texture_detail
        if texture_score:
            checks_passed += 1

        # ── Check 2: Brightness Analysis ────────────────────────────────
        brightness_score, brightness_detail = self._check_brightness(face_region)
        details["brightness"] = brightness_detail
        if brightness_score:
            checks_passed += 1

        # ── Check 3: Edge Density ───────────────────────────────────────
        edge_score, edge_detail = self._check_edge_density(face_region)
        details["edges"] = edge_detail
        if edge_score:
            checks_passed += 1

        # ── Check 4: Color Distribution ─────────────────────────────────
        color_score, color_detail = self._check_color_distribution(face_region)
        details["color"] = color_detail
        if color_score:
            checks_passed += 1

        # ── Check 5: Face Region Variance ───────────────────────────────
        variance_score, variance_detail = self._check_region_variance(face_region)
        details["variance"] = variance_detail
        if variance_score:
            checks_passed += 1

        # Final decision
        is_live = checks_passed >= self.min_checks
        confidence = checks_passed / self.TOTAL_CHECKS

        processing_ms = (time.time() - start_time) * 1000

        result = LivenessResult(
            is_live=is_live,
            confidence=confidence,
            checks_passed=checks_passed,
            checks_total=self.TOTAL_CHECKS,
            details=details,
            processing_time_ms=processing_ms,
        )

        logger.debug(
            f"Liveness check: {'LIVE' if is_live else 'SPOOF'} "
            f"({checks_passed}/{self.TOTAL_CHECKS} checks, "
            f"confidence={confidence:.2f}, {processing_ms:.1f}ms)"
        )

        return result

    # ── Individual Checks ─────────────────────────────────────────────────

    def _check_texture(self, face_region: np.ndarray) -> Tuple[bool, dict]:
        """
        Texture analysis using Laplacian variance.
        Real faces have moderate texture; photos/screens are either
        too smooth (printed) or too sharp/regular (screen pixels).
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Real faces: variance typically 100-2000
        # Too low = blurry photo/print; Too high = screen pixels
        is_pass = 80 < variance < 3000

        return is_pass, {
            "laplacian_variance": round(variance, 2),
            "threshold": "80-3000",
            "passed": is_pass,
        }

    def _check_brightness(self, face_region: np.ndarray) -> Tuple[bool, dict]:
        """
        Brightness analysis.
        Real faces have natural lighting variation.
        Screens often have uniform, overly bright illumination.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Real faces: mean 60-200, std > 15 (natural variation)
        # Screens: often very bright (>200) or uniform (low std)
        is_mean_ok = 40 < mean_brightness < 220
        is_std_ok = std_brightness > 12
        is_pass = is_mean_ok and is_std_ok

        return is_pass, {
            "mean_brightness": round(mean_brightness, 2),
            "std_brightness": round(std_brightness, 2),
            "threshold_mean": "40-220",
            "threshold_std": ">12",
            "passed": is_pass,
        }

    def _check_edge_density(self, face_region: np.ndarray) -> Tuple[bool, dict]:
        """
        Edge density analysis.
        Real faces have natural edge patterns (eyes, nose, mouth contours).
        Photos may have reduced edge information; screens may have artificial edges.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Real faces: edge density typically 0.05-0.25
        is_pass = 0.03 < edge_density < 0.35

        return is_pass, {
            "edge_density": round(edge_density, 4),
            "threshold": "0.03-0.35",
            "passed": is_pass,
        }

    def _check_color_distribution(self, face_region: np.ndarray) -> Tuple[bool, dict]:
        """
        Color distribution analysis.
        Real faces have natural skin-tone color distribution.
        Photos/screens may have unnatural color shifts or uniform saturation.
        """
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Check saturation distribution
        mean_sat = np.mean(saturation)
        std_sat = np.std(saturation)

        # Real faces: moderate saturation with variation
        # Screens: often over-saturated or uniform
        is_pass = 20 < mean_sat < 180 and std_sat > 8

        return is_pass, {
            "mean_saturation": round(mean_sat, 2),
            "std_saturation": round(std_sat, 2),
            "threshold_mean": "20-180",
            "threshold_std": ">8",
            "passed": is_pass,
        }

    def _check_region_variance(self, face_region: np.ndarray) -> Tuple[bool, dict]:
        """
        Region variance check.
        Divides face into grid cells and checks that different regions
        have different intensities (real faces have shadows, highlights).
        Flat/uniform regions suggest a photo or screen.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Divide into 3x3 grid
        cell_h = h // 3
        cell_w = w // 3
        cell_means = []

        for i in range(3):
            for j in range(3):
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                if cell.size > 0:
                    cell_means.append(np.mean(cell))

        if len(cell_means) < 2:
            return False, {"error": "Could not divide face into grid"}

        # Check variation between cells
        cells_array = np.array(cell_means)
        inter_cell_std = np.std(cells_array)

        # Real faces have shadow/highlight variation across regions
        is_pass = inter_cell_std > 5

        return is_pass, {
            "inter_region_std": round(inter_cell_std, 2),
            "threshold": ">5",
            "grid_means": [round(m, 1) for m in cell_means],
            "passed": is_pass,
        }

    # ── Utility ───────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Get current liveness configuration."""
        return {
            "enabled": True,
            "min_checks_required": self.min_checks,
            "total_checks": self.TOTAL_CHECKS,
            "checks": [
                "texture_analysis",
                "brightness_analysis",
                "edge_density",
                "color_distribution",
                "region_variance",
            ],
            "limitations": [
                "Basic liveness check — not a full anti-spoofing system",
                "Cannot defeat high-quality printed photos",
                "Cannot detect video replay attacks on high-res screens",
                "Cannot detect 3D masks",
                "For production, consider a dedicated liveness model",
            ],
        }


# Global liveness service instance
liveness_service = LivenessService()
