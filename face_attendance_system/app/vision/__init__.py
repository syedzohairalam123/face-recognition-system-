"""
Vision Package
--------------
Computer vision services for face detection, embedding, and recognition.
Uses OpenCV for detection and dlib/face_recognition for embeddings.

Pipeline Components:
    - FrameValidator: Validates frame quality (blur, brightness)
    - FaceDetector: Detects faces in images
    - FaceAligner: Aligns faces to standard orientation
    - FaceEncoder: Generates 128-dim face embeddings
    - FaceRecognizer: Matches embeddings to known users
"""

from app.vision.frame_validator import FrameValidator
from app.vision.face_detector import FaceDetector
from app.vision.face_aligner import FaceAligner

try:
    from app.vision.face_encoder import FaceEncoder
    from app.vision.face_recognizer import FaceRecognizer
except ImportError:
    FaceEncoder = None
    FaceRecognizer = None

__all__ = ["FrameValidator", "FaceDetector", "FaceAligner", "FaceEncoder", "FaceRecognizer"]
