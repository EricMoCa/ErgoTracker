import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from schemas import VideoInput, AnalysisReport, ReportSummary, RiskLevel, ProcessingMode
from pose_pipeline import PosePipeline
from ergo_engine import ErgoEngine
from reports import ReportGenerator
from reports.video_annotator import VideoAnnotator
from loguru import logger


class AnalysisOrchestrator:
    """
    Orchestrates the full ergonomic analysis pipeline:
    1. PosePipeline / AdvancedPosePipeline  → SkeletonSequence
    2. ErgoEngine                           → list[FrameErgonomicScore]
    3. ReportGenerator                      → PDF saved to disk
    4. VideoAnnotator                       → annotated MP4
    5. Skeleton JSON                        → saved for 3D viewer

    When processing_mode=GPU_ENHANCED, uses AdvancedPosePipeline which auto-selects
    among GVHMR / WHAM / TRAM / HumanMM based on VideoProfile and hardware.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._cpu_pipeline = PosePipeline(device="cpu")
        self._gpu_pipeline = None   # lazy — only if gpu_enhanced requested
        self.report_generator = ReportGenerator()

    def _get_pipeline(self, video_input: VideoInput):
        if video_input.processing_mode == ProcessingMode.GPU_ENHANCED:
            if self._gpu_pipeline is None:
                try:
                    from advanced_pipeline import AdvancedPosePipeline
                    self._gpu_pipeline = AdvancedPosePipeline(device="cuda")
                    logger.info("Using AdvancedPosePipeline (GPU Enhanced)")
                except Exception as e:
                    logger.warning(f"AdvancedPosePipeline unavailable ({e}), falling back to CPU")
                    return self._cpu_pipeline
            return self._gpu_pipeline
        return self._cpu_pipeline

    def run(
        self,
        video_input: VideoInput,
        output_dir: str = "./output/reports",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> AnalysisReport:
        def report(pct: float, stage: str):
            logger.info(f"[{pct:.0f}%] {stage}")
            if progress_callback:
                progress_callback(pct, stage)

        report(5, "Preparando video...")
        logger.info(f"Starting analysis: {video_input.path} "
                    f"[mode={video_input.processing_mode}, engine={video_input.preferred_engine}]")

        report(10, "Estimando postura 3D...")
        pipeline = self._get_pipeline(video_input)
        skeleton_seq = pipeline.process(video_input)

        report(60, "Análisis ergonómico...")
        ergo_engine = ErgoEngine(
            methods=video_input.ergo_methods,
            rules_json_path=video_input.rules_profile_path,
        )
        frame_scores = ergo_engine.analyze(skeleton_seq)

        report(75, "Construyendo reporte...")
        analysis_report = self._build_report(video_input, skeleton_seq, frame_scores)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report(80, "Generando PDF...")
        pdf_path = out / f"{analysis_report.id}.pdf"
        self.report_generator.generate(analysis_report, str(pdf_path))

        report(88, "Generando video anotado...")
        video_out = out / f"{analysis_report.id}_annotated.mp4"
        try:
            VideoAnnotator().generate(
                video_path=video_input.path,
                skeleton_seq=skeleton_seq,
                frame_scores=frame_scores,
                output_path=str(video_out),
            )
        except Exception as e:
            logger.warning(f"Annotated video not generated (non-critical): {e}")

        report(96, "Guardando datos 3D...")
        skeleton_path = out / f"{analysis_report.id}_skeleton.json"
        try:
            self._save_skeleton_json(skeleton_seq, skeleton_path)
        except Exception as e:
            logger.warning(f"Skeleton JSON not saved (non-critical): {e}")

        report(100, "Completado")
        logger.success(f"Analysis complete. Report: {pdf_path}")
        return analysis_report

    def _save_skeleton_json(self, skeleton_seq, path: Path) -> None:
        """Save compact skeleton frames for the 3D viewer."""
        frames = []
        for skel in skeleton_seq.frames:
            kps = {}
            for name, kp in skel.keypoints.items():
                if kp.confidence > 0.05:
                    kps[name] = {"x": round(kp.x, 4), "y": round(kp.y, 4),
                                 "z": round(kp.z, 4), "confidence": round(kp.confidence, 3)}
            frames.append({"frame_idx": skel.frame_idx, "keypoints": kps})
        payload = {
            "fps": float(getattr(skeleton_seq, "fps", 0.0) or 0.0),
            "total_frames": int(getattr(skeleton_seq, "total_frames", 0) or 0),
            "coordinate_system": getattr(skeleton_seq.frames[0], "coordinate_system", "camera")
            if skeleton_seq.frames
            else "camera",
            "frames": frames,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")))

    def _build_report(self, video_input, skeleton_seq, frame_scores) -> AnalysisReport:
        high_risk_frames = sum(
            1 for f in frame_scores
            if f.overall_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        )
        max_reba = max((f.reba.total for f in frame_scores if f.reba), default=None)

        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            video_path=video_input.path,
            duration_s=len(skeleton_seq.frames) / max(skeleton_seq.fps, 1.0),
            total_frames=skeleton_seq.total_frames,
            analyzed_frames=len(frame_scores),
            person_height_cm=video_input.person_height_cm,
            methods_used=video_input.ergo_methods,
            frame_scores=frame_scores,
            summary=ReportSummary(
                max_reba_score=max_reba,
                pct_frames_high_risk=high_risk_frames / max(len(frame_scores), 1),
            ),
        )
