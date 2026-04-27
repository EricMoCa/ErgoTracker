import numpy as np
from loguru import logger


class VisualOdometry:
    """
    Estimates per-frame camera rotation from a video using OpenCV optical flow.
    Used as fallback when SimpleVO / DPVO are not installed.
    """

    def __init__(self, method: str = "opencv_fallback"):
        self.method = method

    def estimate(self, video_path: str) -> list[np.ndarray]:
        """
        Returns a list of 3x3 rotation matrices, one per frame.
        On failure returns an empty list (caller must handle gracefully).
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available — returning empty camera rotations")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        rotations: list[np.ndarray] = []
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        rotations.append(np.eye(3, dtype=np.float32))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            if pts_prev is None or len(pts_prev) < 5:
                rotations.append(np.eye(3, dtype=np.float32))
                prev_gray = gray
                continue

            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
            good_prev = pts_prev[status.ravel() == 1]
            good_curr = pts_curr[status.ravel() == 1]

            if len(good_prev) >= 5:
                H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
                if H is not None:
                    R = H[:3, :3].astype(np.float32)
                    rotations.append(R)
                else:
                    rotations.append(np.eye(3, dtype=np.float32))
            else:
                rotations.append(np.eye(3, dtype=np.float32))

            prev_gray = gray

        cap.release()
        return rotations
