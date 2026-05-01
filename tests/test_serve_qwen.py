"""Tests for QwenAnnotator. GPU tests are skipped unless pytest -m gpu is used."""
import pytest


def test_import_does_not_load_model():
    """Importing QwenAnnotator must NOT load the model (no GPU required)."""
    from src.annotation.serve_qwen import QwenAnnotator
    ann = QwenAnnotator()
    assert ann._llm is None


@pytest.mark.gpu
def test_smoke_inference(tmp_path):
    """Load the model and run inference on a synthetic 2-second clip.

    Skipped unless run with: pytest -m gpu
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python not installed")

    from src.annotation.serve_qwen import QwenAnnotator

    # Create a minimal 2-second test video (10 frames at 5 fps)
    video_path = str(tmp_path / "test_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 5, (64, 64))
    for _ in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    ann = QwenAnnotator()
    result = ann.annotate_episode(video_path, "Describe what the robot is doing.")
    assert isinstance(result, str)
    assert len(result) > 0
