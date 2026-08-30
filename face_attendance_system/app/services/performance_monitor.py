"""
Performance Monitor
-------------------
Tracks and reports system performance metrics.
Monitors FPS, detection time, recognition time, and total processing latency.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for a single frame processing."""
    timestamp: float = 0.0
    validation_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    alignment_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    matching_time_ms: float = 0.0
    attendance_time_ms: float = 0.0
    total_time_ms: float = 0.0
    faces_detected: int = 0
    faces_recognized: int = 0
    frame_width: int = 0
    frame_height: int = 0

    def to_dict(self) -> dict:
        return {
            "validation_ms": round(self.validation_time_ms, 2),
            "detection_ms": round(self.detection_time_ms, 2),
            "alignment_ms": round(self.alignment_time_ms, 2),
            "embedding_ms": round(self.embedding_time_ms, 2),
            "matching_ms": round(self.matching_time_ms, 2),
            "attendance_ms": round(self.attendance_time_ms, 2),
            "total_ms": round(self.total_time_ms, 2),
            "faces_detected": self.faces_detected,
            "faces_recognized": self.faces_recognized,
            "resolution": f"{self.frame_width}x{self.frame_height}",
        }


class PerformanceMonitor:
    """Track and report system performance."""

    def __init__(self, history_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            history_size: Number of recent frames to keep in history
        """
        self.history_size = history_size
        self._frame_times = deque(maxlen=history_size)
        self._detection_times = deque(maxlen=history_size)
        self._recognition_times = deque(maxlen=history_size)
        self._total_times = deque(maxlen=history_size)
        self._faces_detected_count = deque(maxlen=history_size)
        self._faces_recognized_count = deque(maxlen=history_size)

        # DB latency tracking
        self._db_query_times = deque(maxlen=history_size)
        self._db_query_count = 0

        # Frame skip tracking
        self._frames_skipped = 0
        self._frames_processed = 0
        self._skip_interval = 2  # process every Nth frame (configurable)

        # Embedding storage metrics
        self._embedding_count = 0
        self._embedding_size_bytes = 0

        self._last_frame_time = time.time()
        self._frame_count = 0
        self._start_time = time.time()

        logger.info(f"PerformanceMonitor initialized (history={history_size})")

    def record_frame(self, metrics: FrameMetrics):
        """Record metrics for a processed frame."""
        now = time.time()

        # Calculate FPS from frame intervals
        if self._last_frame_time > 0:
            frame_time = now - self._last_frame_time
            self._frame_times.append(frame_time)

        self._last_frame_time = now
        self._frame_count += 1

        # Record timing metrics
        self._detection_times.append(metrics.detection_time_ms)
        self._recognition_times.append(metrics.embedding_time_ms + metrics.matching_time_ms)
        self._total_times.append(metrics.total_time_ms)
        self._faces_detected_count.append(metrics.faces_detected)
        self._faces_recognized_count.append(metrics.faces_recognized)

    def get_fps(self) -> float:
        """Calculate current FPS (frames per second)."""
        if len(self._frame_times) < 2:
            return 0.0

        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        if avg_frame_time <= 0:
            return 0.0

        return 1.0 / avg_frame_time

    def get_average_detection_time(self) -> float:
        """Get average detection time in milliseconds."""
        if not self._detection_times:
            return 0.0
        return sum(self._detection_times) / len(self._detection_times)

    def get_average_recognition_time(self) -> float:
        """Get average recognition time in milliseconds."""
        if not self._recognition_times:
            return 0.0
        return sum(self._recognition_times) / len(self._recognition_times)

    def get_average_total_time(self) -> float:
        """Get average total processing time in milliseconds."""
        if not self._total_times:
            return 0.0
        return sum(self._total_times) / len(self._total_times)

    def get_detection_rate(self) -> float:
        """Get average faces detected per frame."""
        if not self._faces_detected_count:
            return 0.0
        return sum(self._faces_detected_count) / len(self._faces_detected_count)

    def get_recognition_rate(self) -> float:
        """Get average faces recognized per frame."""
        if not self._faces_recognized_count:
            return 0.0
        return sum(self._faces_recognized_count) / len(self._faces_recognized_count)

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time

    def get_total_frames_processed(self) -> int:
        """Get total number of frames processed."""
        return self._frame_count

    # ── DB Latency Tracking ───────────────────────────────────────────────

    def record_db_query(self, duration_ms: float):
        """Record a database query duration."""
        self._db_query_times.append(duration_ms)
        self._db_query_count += 1

    def get_average_db_latency(self) -> float:
        """Get average DB query latency in milliseconds."""
        if not self._db_query_times:
            return 0.0
        return sum(self._db_query_times) / len(self._db_query_times)

    def get_db_query_count(self) -> int:
        """Get total number of DB queries tracked."""
        return self._db_query_count

    # ── Frame Skip Tracking ───────────────────────────────────────────────

    def set_skip_interval(self, interval: int):
        """Set the frame skip interval (process every Nth frame)."""
        self._skip_interval = max(1, interval)
        logger.info(f"Frame skip interval set to {self._skip_interval}")

    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed (frame skipping)."""
        self._frames_processed += 1
        if self._frames_processed % self._skip_interval == 0:
            return True
        self._frames_skipped += 1
        return False

    def get_frame_skip_stats(self) -> Dict:
        """Get frame skipping statistics."""
        total = self._frames_processed + self._frames_skipped
        return {
            "skip_interval": self._skip_interval,
            "frames_processed": self._frames_processed,
            "frames_skipped": self._frames_skipped,
            "skip_ratio": round(self._frames_skipped / total * 100, 1) if total > 0 else 0,
        }

    # ── Embedding Metrics ─────────────────────────────────────────────────

    def record_embedding(self, size_bytes: int = 0):
        """Record an embedding storage event."""
        self._embedding_count += 1
        self._embedding_size_bytes += size_bytes

    def get_embedding_stats(self) -> Dict:
        """Get embedding storage statistics."""
        return {
            "total_embeddings": self._embedding_count,
            "total_size_bytes": self._embedding_size_bytes,
            "total_size_kb": round(self._embedding_size_bytes / 1024, 1),
        }

    # ── Comprehensive Metrics ─────────────────────────────────────────────

    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        return {
            "fps": round(self.get_fps(), 1),
            "avg_detection_ms": round(self.get_average_detection_time(), 1),
            "avg_recognition_ms": round(self.get_average_recognition_time(), 1),
            "avg_total_ms": round(self.get_average_total_time(), 1),
            "avg_db_latency_ms": round(self.get_average_db_latency(), 1),
            "db_query_count": self.get_db_query_count(),
            "avg_faces_detected": round(self.get_detection_rate(), 1),
            "avg_faces_recognized": round(self.get_recognition_rate(), 1),
            "total_frames": self._frame_count,
            "uptime_seconds": round(self.get_uptime(), 0),
            "history_size": len(self._frame_times),
            "frame_skip": self.get_frame_skip_stats(),
            "embeddings": self.get_embedding_stats(),
        }

    def get_health_status(self) -> Dict:
        """Get system health status based on performance."""
        fps = self.get_fps()
        avg_time = self.get_average_total_time()

        if fps >= 15 and avg_time < 100:
            status = "excellent"
            message = "System performing optimally"
        elif fps >= 10 and avg_time < 200:
            status = "good"
            message = "System performing well"
        elif fps >= 5 and avg_time < 500:
            status = "fair"
            message = "System performing adequately"
        else:
            status = "poor"
            message = "System performance degraded"

        return {
            "status": status,
            "message": message,
            "fps": round(fps, 1),
            "avg_processing_ms": round(avg_time, 1),
        }

    def reset(self):
        """Reset all metrics."""
        self._frame_times.clear()
        self._detection_times.clear()
        self._recognition_times.clear()
        self._total_times.clear()
        self._faces_detected_count.clear()
        self._faces_recognized_count.clear()
        self._db_query_times.clear()
        self._db_query_count = 0
        self._frames_skipped = 0
        self._frames_processed = 0
        self._embedding_count = 0
        self._embedding_size_bytes = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._start_time = time.time()
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
