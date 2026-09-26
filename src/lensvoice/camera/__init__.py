"""Camera sources for LensVoice."""

from lensvoice.camera.mock_camera import MockCamera
from lensvoice.camera.webcam import WebcamCamera

__all__ = ["MockCamera", "WebcamCamera"]