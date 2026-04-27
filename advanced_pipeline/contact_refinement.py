import numpy as np
from schemas import Keypoint3D, Skeleton3D, SkeletonSequence
from loguru import logger

FOOT_JOINTS = [
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
_CONTACT_VELOCITY_THRESHOLD_M = 0.02

# Joints whose Y coordinate must stay >= floor when a foot is in contact.
# Ordered from lowest (feet) to highest to preserve relative distances.
_ALL_JOINTS_ORDERED = [
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    "left_knee", "right_knee",
    "left_hip", "right_hip", "mid_hip",
    "left_shoulder", "right_shoulder", "mid_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "neck", "nose", "left_eye", "right_eye",
]


class ContactRefinement:
    """
    Detects foot-ground contact and flags frames where a foot is in contact.
    Applies a floor constraint as a post-process on any SkeletonSequence.

    v1: detection by keypoint velocity; floor constraint is a passthrough stub.
    v2: integrate WHAM contact classifier when available.
    """

    def refine(self, skeleton_seq: SkeletonSequence) -> SkeletonSequence:
        contact_labels = self._detect_contact(skeleton_seq)
        logger.debug(f"Contact frames: {sum(contact_labels)}/{len(contact_labels)}")
        return self._apply_floor_constraint(skeleton_seq, contact_labels)

    def _detect_contact(self, skeleton_seq: SkeletonSequence) -> list[bool]:
        frames = skeleton_seq.frames
        contact = [False] * len(frames)

        for i in range(1, len(frames)):
            prev, curr = frames[i - 1], frames[i]
            for joint in FOOT_JOINTS:
                if joint not in prev.keypoints or joint not in curr.keypoints:
                    continue
                p = prev.keypoints[joint]
                c = curr.keypoints[joint]
                velocity = float(np.sqrt(
                    (c.x - p.x) ** 2 + (c.y - p.y) ** 2 + (c.z - p.z) ** 2
                ))
                if velocity < _CONTACT_VELOCITY_THRESHOLD_M:
                    contact[i] = True
                    break

        return contact

    def _apply_floor_constraint(
        self,
        skeleton_seq: SkeletonSequence,
        contact_labels: list[bool],
    ) -> SkeletonSequence:
        """
        Pin foot joints to the ground plane during contact frames to eliminate
        foot sliding.

        Algorithm:
        1. Compute the ground plane Y level as the minimum foot-joint Y across
           all contact frames (robustly, using the 5th percentile to ignore noise).
        2. For each contact frame, if any foot joint is below the floor Y,
           shift the entire skeleton up by the penetration amount.
        3. For non-contact frames leave the skeleton untouched.
        """
        frames = list(skeleton_seq.frames)
        if not any(contact_labels):
            return skeleton_seq

        # Collect foot-joint Y values in contact frames to estimate floor level
        foot_ys: list[float] = []
        for i, frame in enumerate(frames):
            if not contact_labels[i]:
                continue
            for jn in FOOT_JOINTS:
                if jn in frame.keypoints and frame.keypoints[jn].confidence > 0.3:
                    foot_ys.append(frame.keypoints[jn].y)

        if not foot_ys:
            return skeleton_seq

        floor_y = float(np.percentile(foot_ys, 5))
        logger.debug(f"ContactRefinement floor_y = {floor_y:.4f} m")

        corrected_frames: list[Skeleton3D] = []
        for i, frame in enumerate(frames):
            if not contact_labels[i]:
                corrected_frames.append(frame)
                continue

            # Find the lowest foot joint in this frame
            min_foot_y = min(
                (frame.keypoints[jn].y for jn in FOOT_JOINTS
                 if jn in frame.keypoints and frame.keypoints[jn].confidence > 0.3),
                default=floor_y,
            )
            penetration = floor_y - min_foot_y  # positive → foot is below floor

            if abs(penetration) < 1e-4:
                corrected_frames.append(frame)
                continue

            # Shift all keypoints vertically by penetration
            new_kps: dict[str, Keypoint3D] = {}
            for name, kp in frame.keypoints.items():
                new_kps[name] = Keypoint3D(
                    x=kp.x,
                    y=kp.y + penetration,
                    z=kp.z,
                    confidence=kp.confidence,
                    occluded=kp.occluded,
                )

            corrected_frames.append(
                Skeleton3D(
                    frame_idx=frame.frame_idx,
                    timestamp_s=frame.timestamp_s,
                    keypoints=new_kps,
                    scale_px_to_m=frame.scale_px_to_m,
                    person_height_cm=frame.person_height_cm,
                    coordinate_system=frame.coordinate_system,
                )
            )

        return SkeletonSequence(
            video_path=skeleton_seq.video_path,
            fps=skeleton_seq.fps,
            total_frames=skeleton_seq.total_frames,
            frames=corrected_frames,
        )

    def get_contact_events(self, skeleton_seq: SkeletonSequence) -> list[dict]:
        """
        Returns contact events as a list of dicts with start_frame, end_frame, foot.
        Useful to distinguish static vs. dynamic posture in REBA/OWAS.
        """
        contact = self._detect_contact(skeleton_seq)
        events: list[dict] = []
        in_contact = False
        start = 0

        for i, c in enumerate(contact):
            if c and not in_contact:
                in_contact = True
                start = i
            elif not c and in_contact:
                in_contact = False
                events.append({"start_frame": start, "end_frame": i - 1, "foot": "unknown"})

        if in_contact:
            events.append({"start_frame": start, "end_frame": len(contact) - 1, "foot": "unknown"})

        return events
