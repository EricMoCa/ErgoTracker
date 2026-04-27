import numpy as np
from schemas import SkeletonSequence, Skeleton3D, Keypoint3D
from loguru import logger

_LOW_CONFIDENCE_THRESHOLD = 0.4


class STRIDERefinement:
    """
    Refines poses in frames with severe occlusion (WACV 2025 approach).

    v1: linear temporal interpolation between valid (high-confidence) adjacent frames.
    v2: Test-Time Training over a human motion prior (requires STRIDE model).
    """

    def refine(self, skeleton_seq: SkeletonSequence) -> SkeletonSequence:
        occluded_indices = self._detect_occluded_frames(skeleton_seq)
        if not occluded_indices:
            return skeleton_seq
        logger.debug(f"STRIDE: interpolating {len(occluded_indices)} occluded frames")
        return self._interpolate_occluded(skeleton_seq, occluded_indices)

    def _detect_occluded_frames(self, skeleton_seq: SkeletonSequence) -> list[int]:
        occluded = []
        for i, frame in enumerate(skeleton_seq.frames):
            confidences = [kp.confidence for kp in frame.keypoints.values()]
            if confidences and np.mean(confidences) < _LOW_CONFIDENCE_THRESHOLD:
                occluded.append(i)
        return occluded

    def _interpolate_occluded(
        self,
        skeleton_seq: SkeletonSequence,
        occluded_indices: set,
    ) -> SkeletonSequence:
        frames = list(skeleton_seq.frames)
        occluded_set = set(occluded_indices)
        n = len(frames)

        for idx in occluded_indices:
            prev_idx = next((j for j in range(idx - 1, -1, -1) if j not in occluded_set), None)
            next_idx = next((j for j in range(idx + 1, n) if j not in occluded_set), None)

            if prev_idx is None and next_idx is None:
                continue
            if prev_idx is None:
                source = frames[next_idx]
            elif next_idx is None:
                source = frames[prev_idx]
            else:
                t = (idx - prev_idx) / (next_idx - prev_idx)
                source = self._lerp_skeleton(frames[prev_idx], frames[next_idx], t, frames[idx])
                frames[idx] = source
                continue

            frames[idx] = Skeleton3D(
                frame_idx=frames[idx].frame_idx,
                timestamp_s=frames[idx].timestamp_s,
                keypoints=source.keypoints,
                scale_px_to_m=frames[idx].scale_px_to_m,
                person_height_cm=frames[idx].person_height_cm,
                coordinate_system=frames[idx].coordinate_system,
            )

        return SkeletonSequence(
            video_path=skeleton_seq.video_path,
            fps=skeleton_seq.fps,
            total_frames=skeleton_seq.total_frames,
            frames=frames,
        )

    def _lerp_skeleton(
        self,
        a: Skeleton3D,
        b: Skeleton3D,
        t: float,
        template: Skeleton3D,
    ) -> Skeleton3D:
        interpolated: dict[str, Keypoint3D] = {}
        for name in a.keypoints:
            if name not in b.keypoints:
                interpolated[name] = a.keypoints[name]
                continue
            ka, kb = a.keypoints[name], b.keypoints[name]
            interpolated[name] = Keypoint3D(
                x=ka.x + t * (kb.x - ka.x),
                y=ka.y + t * (kb.y - ka.y),
                z=ka.z + t * (kb.z - ka.z),
                confidence=ka.confidence + t * (kb.confidence - ka.confidence),
            )
        return Skeleton3D(
            frame_idx=template.frame_idx,
            timestamp_s=template.timestamp_s,
            keypoints=interpolated,
            scale_px_to_m=template.scale_px_to_m,
            person_height_cm=template.person_height_cm,
            coordinate_system=template.coordinate_system,
        )
