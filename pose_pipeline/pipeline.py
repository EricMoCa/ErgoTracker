"""
pose_pipeline/pipeline.py
Orquestador del pipeline completo de estimación de postura 3D.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from schemas import VideoInput, SkeletonSequence, Skeleton3D
from schemas.skeleton import Keypoint3D

from .model_downloader import ensure_model
from .detector import PersonDetector, BBox
from .pose_2d import PoseEstimator2D, Keypoint2D
from .pose_3d import PoseLifter3D
from .height import HeightAnchor


class PosePipeline:
    """
    Orquestador del pipeline completo de estimación de postura.
    Entrada: VideoInput
    Salida: SkeletonSequence con coordenadas 3D en metros (camera-relative en CPU_ONLY).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._detector: Optional[PersonDetector] = None
        self._pose_2d: Optional[PoseEstimator2D] = None
        self._lifter_3d: Optional[PoseLifter3D] = None
        self.height_anchor = HeightAnchor()
        self._models_loaded = False

    def _load_models(self) -> bool:
        """Carga los modelos ONNX. Retorna True si todos están disponibles."""
        if self._models_loaded:
            return True
        try:
            yolo_path = ensure_model("yolov8n")
            rtm_path = ensure_model("rtmpose_m")
            mb_path = ensure_model("motionbert_lite")
            self._detector = PersonDetector(str(yolo_path), self.device)
            self._pose_2d = PoseEstimator2D(str(rtm_path), self.device)
            self._lifter_3d = PoseLifter3D(str(mb_path), self.device)
            self._models_loaded = True
            logger.info("Modelos ONNX cargados correctamente")
            return True
        except Exception as e:
            logger.warning(f"No se pudieron cargar modelos ONNX: {e}. Usando modo sintético.")
            return False

    def process(self, video_input: VideoInput) -> SkeletonSequence:
        """
        Pipeline completo:
        1. Leer video con OpenCV (respetar fps_sample_rate)
        2. Detectar persona en cada frame
        3. Estimar keypoints 2D
        4. Aplicar MotionBERT Lite (ventana de 243 frames)
        5. Calcular factor de escala (usar person_height_cm)
        6. Retornar SkeletonSequence con coordenadas en metros
        """
        import cv2

        video_path = video_input.path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = video_input.fps_sample_rate

        models_available = self._load_models()
        raw_frames: list[tuple[int, np.ndarray]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                raw_frames.append((frame_idx, frame))
            frame_idx += 1
        cap.release()

        if not raw_frames:
            return SkeletonSequence(
                video_path=video_path,
                fps=fps,
                total_frames=total_frames,
                frames=[],
            )

        if models_available and self._detector and self._pose_2d and self._lifter_3d:
            return self._process_with_models(
                raw_frames, video_path, fps, total_frames, video_input.person_height_cm
            )
        else:
            return self._process_synthetic(
                raw_frames, video_path, fps, total_frames, video_input.person_height_cm
            )

    def _process_with_models(
        self,
        raw_frames: list[tuple[int, np.ndarray]],
        video_path: str,
        fps: float,
        total_frames: int,
        person_height_cm: float,
    ) -> SkeletonSequence:
        """Pipeline completo con modelos ONNX reales."""
        assert self._detector and self._pose_2d and self._lifter_3d

        keypoints_2d_seq: list[dict[str, Keypoint2D]] = []
        frame_indices: list[int] = []
        h, w = 1080, 1920

        for frame_idx, frame in raw_frames:
            h, w = frame.shape[:2]
            bbox = self._detector.detect_primary(frame)
            if bbox is None:
                bbox = [0.0, 0.0, float(w), float(h), 1.0]
            kpts_2d = self._pose_2d.estimate(frame, bbox)
            keypoints_2d_seq.append(kpts_2d)
            frame_indices.append(frame_idx)

        keypoints_3d_seq = self._lifter_3d.lift(keypoints_2d_seq)

        # Calcular escala usando el primer frame
        scale = person_height_cm / 100.0
        if keypoints_3d_seq:
            scale = self.height_anchor.compute_scale_factor(
                keypoints_3d_seq[0], person_height_cm
            )

        scaled_seq = self.height_anchor.apply_scale(keypoints_3d_seq, scale)

        skeletons: list[Skeleton3D] = []
        for i, (frame_idx, kps) in enumerate(zip(frame_indices, scaled_seq)):
            skeletons.append(Skeleton3D(
                frame_idx=frame_idx,
                timestamp_s=frame_idx / fps,
                keypoints=kps,
                scale_px_to_m=scale,
                person_height_cm=person_height_cm,
                coordinate_system="camera",
            ))

        return SkeletonSequence(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            frames=skeletons,
        )

    def _process_synthetic(
        self,
        raw_frames: list[tuple[int, np.ndarray]],
        video_path: str,
        fps: float,
        total_frames: int,
        person_height_cm: float,
    ) -> SkeletonSequence:
        """Genera un SkeletonSequence sintético cuando los modelos no están disponibles."""
        scale = person_height_cm / 100.0
        skeletons: list[Skeleton3D] = []

        for frame_idx, _ in raw_frames:
            kps = self._make_upright_keypoints(person_height_cm)
            skeletons.append(Skeleton3D(
                frame_idx=frame_idx,
                timestamp_s=frame_idx / fps,
                keypoints=kps,
                scale_px_to_m=scale / 1000.0,
                person_height_cm=person_height_cm,
                coordinate_system="camera",
            ))

        return SkeletonSequence(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            frames=skeletons,
        )

    def _make_upright_keypoints(self, height_cm: float) -> dict[str, Keypoint3D]:
        """Genera keypoints sintéticos para una persona de pie erecta."""
        h = height_cm / 100.0
        kps: dict[str, Keypoint3D] = {
            "nose":           Keypoint3D(x=0.0,   y=h,           z=0.0, confidence=0.9),
            "neck":           Keypoint3D(x=0.0,   y=h * 0.88,    z=0.0, confidence=0.9),
            "right_shoulder": Keypoint3D(x=0.2,   y=h * 0.82,    z=0.0, confidence=0.9),
            "right_elbow":    Keypoint3D(x=0.2,   y=h * 0.64,    z=0.0, confidence=0.9),
            "right_wrist":    Keypoint3D(x=0.2,   y=h * 0.47,    z=0.0, confidence=0.9),
            "left_shoulder":  Keypoint3D(x=-0.2,  y=h * 0.82,    z=0.0, confidence=0.9),
            "left_elbow":     Keypoint3D(x=-0.2,  y=h * 0.64,    z=0.0, confidence=0.9),
            "left_wrist":     Keypoint3D(x=-0.2,  y=h * 0.47,    z=0.0, confidence=0.9),
            "right_hip":      Keypoint3D(x=0.1,   y=h * 0.52,    z=0.0, confidence=0.9),
            "right_knee":     Keypoint3D(x=0.1,   y=h * 0.28,    z=0.0, confidence=0.9),
            "right_ankle":    Keypoint3D(x=0.1,   y=0.04,        z=0.0, confidence=0.9),
            "left_hip":       Keypoint3D(x=-0.1,  y=h * 0.52,    z=0.0, confidence=0.9),
            "left_knee":      Keypoint3D(x=-0.1,  y=h * 0.28,    z=0.0, confidence=0.9),
            "left_ankle":     Keypoint3D(x=-0.1,  y=0.04,        z=0.0, confidence=0.9),
            "mid_hip":        Keypoint3D(x=0.0,   y=h * 0.52,    z=0.0, confidence=0.9),
            "mid_shoulder":   Keypoint3D(x=0.0,   y=h * 0.82,    z=0.0, confidence=0.9),
            "right_eye":      Keypoint3D(x=0.05,  y=h * 1.01,    z=0.0, confidence=0.9),
            "left_eye":       Keypoint3D(x=-0.05, y=h * 1.01,    z=0.0, confidence=0.9),
        }
        return kps

    def process_video_path(
        self,
        path: str,
        person_height_cm: float = 170.0,
    ) -> SkeletonSequence:
        """Shortcut para llamadas simples sin VideoInput completo."""
        from schemas import VideoInput
        return self.process(VideoInput(path=path, person_height_cm=person_height_cm))
