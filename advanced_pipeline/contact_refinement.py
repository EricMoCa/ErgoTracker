import numpy as np
from schemas import SkeletonSequence
from loguru import logger

FOOT_JOINTS = [
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
_CONTACT_VELOCITY_THRESHOLD_M = 0.02


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
        # v1 stub: return unmodified — floor pinning not yet implemented
        return skeleton_seq

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
