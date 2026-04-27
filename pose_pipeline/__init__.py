# pose_pipeline/__init__.py
# Task A — Estimación de postura 2D/3D
# Depende de: schemas
#
# Implementar en este orden:
#   1. model_downloader.py  → descarga lazy de modelos ONNX
#   2. detector.py          → PersonDetector (YOLOv8n ONNX)
#   3. pose_2d.py           → PoseEstimator2D (RTMPose-m ONNX)
#   4. pose_3d.py           → PoseLifter3D (MotionBERT Lite ONNX)
#   5. height.py            → HeightAnchor
#   6. pipeline.py          → PosePipeline (orquestador)
#
# Ver CLAUDE.md para la implementación detallada.

from .pipeline import PosePipeline

__all__ = ["PosePipeline"]
