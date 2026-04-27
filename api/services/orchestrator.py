import uuid
from datetime import datetime
from pathlib import Path
from schemas import VideoInput, AnalysisReport, ReportSummary, RiskLevel
from pose_pipeline import PosePipeline
from ergo_engine import ErgoEngine
from reports import ReportGenerator
from loguru import logger


class AnalysisOrchestrator:
    """
    Orchestrates the full ergonomic analysis pipeline:
    1. PosePipeline     → SkeletonSequence
    2. ErgoEngine       → list[FrameErgonomicScore]
    3. ReportGenerator  → PDF saved to disk

    GPU note: the LLM rule extraction step (llm_rules/) must complete and release
    VRAM before calling this orchestrator when ProcessingMode.GPU_ENHANCED is used.
    The orchestrator does NOT call RuleExtractor directly.
    """

    def __init__(self, device: str = "cpu"):
        self.pose_pipeline = PosePipeline(device=device)
        self.report_generator = ReportGenerator()

    def run(self, video_input: VideoInput, output_dir: str = "./output/reports") -> AnalysisReport:
        logger.info(f"Starting analysis: {video_input.path}")

        logger.info("Phase 1: estimating 3D pose...")
        skeleton_seq = self.pose_pipeline.process(video_input)

        logger.info("Phase 2: ergonomic analysis...")
        ergo_engine = ErgoEngine(
            methods=video_input.ergo_methods,
            rules_json_path=video_input.rules_profile_path,
        )
        frame_scores = ergo_engine.analyze(skeleton_seq)

        report = self._build_report(video_input, skeleton_seq, frame_scores)

        pdf_path = Path(output_dir) / f"{report.id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_generator.generate(report, str(pdf_path))
        logger.success(f"Analysis complete. Report: {pdf_path}")

        return report

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
