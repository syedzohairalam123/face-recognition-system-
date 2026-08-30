"""
Camera Manager
--------------
Pure camera management service - NOT coupled to recognition or attendance.
Handles camera initialization, streaming, frame capture, and error handling.

This module is the camera subsystem required by Phase 04.
Recognition and attendance are handled separately.
"""

import logging
import time
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Pure camera management service.
    
    Responsibilities:
    - Camera initialization and connection
    - Live video streaming
    - Frame capture
    - Camera error detection
    - Camera release
    - Reconnect handling
    - Camera status indicator
    - FPS calculation
    - Clean shutdown
    """

    def __init__(self):
        """Initialize camera manager."""
        self._camera: Optional[cv2.VideoCapture] = None
        self._camera_index: int = CAMERA_INDEX
        self._is_streaming: bool = False
        self._stream_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # FPS tracking
        self._frame_count: int = 0
        self._fps_start_time: float = time.time()
        self._current_fps: float = 0.0

        # Status tracking
        self._last_error: Optional[str] = None
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 3

        logger.info("CameraManager initialized")

    # ── Camera Lifecycle ──────────────────────────────────────────────────────

    def initialize(self, camera_index: int = None) -> Tuple[bool, str]:
        """
        Initialize and open camera device.

        Args:
            camera_index: Camera device index (default from config)

        Returns:
            Tuple of (success, message)
        """
        try:
            idx = camera_index if camera_index is not None else self._camera_index

            # Release any existing camera first
            self.release()

            self._camera = cv2.VideoCapture(idx)

            if not self._camera.isOpened():
                self._last_error = f"Could not open camera {idx}"
                logger.error(self._last_error)
                return False, f"Could not open camera {idx}. Check if camera is connected."

            # Set camera properties
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self._camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

            # Verify camera is actually working by reading a test frame
            ret, test_frame = self._camera.read()
            if not ret or test_frame is None:
                self.release()
                self._last_error = f"Camera {idx} opened but cannot read frames"
                logger.error(self._last_error)
                return False, f"Camera {idx} opened but cannot read frames"

            self._camera_index = idx
            self._last_error = None
            self._reconnect_attempts = 0

            # Reset FPS tracking
            self._frame_count = 0
            self._fps_start_time = time.time()
            self._current_fps = 0.0

            logger.info(f"Camera {idx} initialized successfully ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps)")
            return True, f"Camera {idx} initialized"

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error initializing camera: {e}")
            return False, f"Camera initialization failed: {str(e)}"

    def release(self):
        """Release camera resources."""
        self._is_streaming = False

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)

        if self._camera:
            try:
                self._camera.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self._camera = None

        logger.info("Camera released")

    def shutdown(self):
        """Clean shutdown of camera system."""
        logger.info("Shutting down camera system...")
        self._is_streaming = False
        self.release()
        logger.info("Camera system shutdown complete")

    # ── Frame Capture ─────────────────────────────────────────────────────────

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns:
            BGR numpy array or None if capture failed
        """
        with self._lock:
            if not self._camera or not self._camera.isOpened():
                return None

            ret, frame = self._camera.read()

            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                self._last_error = "Frame capture failed"
                return None

            # Update FPS tracking
            self._frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.time()

            return frame

    def capture_and_validate(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Capture a frame and validate it.

        Returns:
            Tuple of (frame or None, status message)
        """
        frame = self.capture_frame()

        if frame is None:
            return None, "Could not capture frame"

        # Basic validation
        h, w = frame.shape[:2]
        if w < 100 or h < 100:
            return None, f"Frame too small: {w}x{h}"

        return frame, "Frame captured successfully"

    # ── Live Streaming ────────────────────────────────────────────────────────

    def start_streaming(self, callback=None):
        """
        Start continuous frame streaming in background thread.

        Args:
            callback: Optional function to call with each frame
        """
        if self._is_streaming:
            logger.warning("Already streaming")
            return

        if not self._camera or not self._camera.isOpened():
            logger.error("Cannot start streaming: camera not initialized")
            return

        self._is_streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=True
        )
        self._stream_thread.start()
        logger.info("Streaming started")

    def stop_streaming(self):
        """Stop continuous frame streaming."""
        self._is_streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        logger.info("Streaming stopped")

    def _stream_loop(self, callback=None):
        """Internal streaming loop."""
        while self._is_streaming:
            frame = self.capture_frame()
            if frame is not None and callback:
                try:
                    callback(frame)
                except Exception as e:
                    logger.error(f"Stream callback error: {e}")
            time.sleep(0.01)  # Small delay to prevent CPU spinning

    # ── Reconnect Handling ────────────────────────────────────────────────────

    def try_reconnect(self) -> Tuple[bool, str]:
        """
        Attempt to reconnect to camera after failure.

        Returns:
            Tuple of (success, message)
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            return False, f"Max reconnect attempts ({self._max_reconnect_attempts}) reached"

        self._reconnect_attempts += 1
        logger.info(f"Reconnect attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")

        # Release current camera
        self.release()

        # Wait before reconnecting
        time.sleep(1.0)

        # Try to initialize again
        success, message = self.initialize(self._camera_index)

        if success:
            self._reconnect_attempts = 0
            logger.info("Reconnect successful")
        else:
            logger.warning(f"Reconnect failed: {message}")

        return success, message

    # ── Status & Info ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Get comprehensive camera status.

        Returns:
            Dictionary with camera status information
        """
        is_open = self._camera is not None and self._camera.isOpened()

        status = {
            "initialized": is_open,
            "streaming": self._is_streaming,
            "camera_index": self._camera_index,
            "fps": round(self._current_fps, 1),
            "last_error": self._last_error,
            "reconnect_attempts": self._reconnect_attempts,
        }

        if is_open:
            try:
                status["width"] = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                status["height"] = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                status["actual_fps"] = int(self._camera.get(cv2.CAP_PROP_FPS))
            except Exception:
                pass

        return status

    def get_fps(self) -> float:
        """Get current FPS."""
        return round(self._current_fps, 1)

    def is_available(self) -> bool:
        """Check if camera is available and working."""
        return self._camera is not None and self._camera.isOpened()

    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error

    # ── Utility Methods ───────────────────────────────────────────────────────

    def draw_status_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw camera status overlay on frame.

        Args:
            frame: Input frame

        Returns:
            Frame with status overlay
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # FPS display
        fps_text = f"FPS: {self._current_fps:.1f}"
        cv2.putText(overlay, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Resolution display
        res_text = f"Resolution: {w}x{h}"
        cv2.putText(overlay, res_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicator
        if self._is_streaming:
            # Recording dot
            cv2.circle(overlay, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(overlay, "LIVE", (w - 80, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return overlay

    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> list:
        """
        List available camera devices.

        Args:
            max_cameras: Maximum number of cameras to check

        Returns:
            List of available camera indices
        """
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available


# Global camera manager instance
camera_manager = CameraManager()
