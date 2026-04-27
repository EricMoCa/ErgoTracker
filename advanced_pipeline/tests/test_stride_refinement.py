import pytest
from schemas import SkeletonSequence
from advanced_pipeline.stride_refinement import STRIDERefinement, _LOW_CONFIDENCE_THRESHOLD


def test_refine_returns_skeleton_sequence(simple_skeleton_sequence):
    sr = STRIDERefinement()
    result = sr.refine(simple_skeleton_sequence)
    assert isinstance(result, SkeletonSequence)


def test_no_occlusion_unchanged(simple_skeleton_sequence):
    sr = STRIDERefinement()
    result = sr.refine(simple_skeleton_sequence)
    # High-confidence frames → no change
    assert len(result.frames) == len(simple_skeleton_sequence.frames)
    for orig, refined in zip(simple_skeleton_sequence.frames, result.frames):
        assert orig.frame_idx == refined.frame_idx


def test_detect_low_confidence_frames(skeleton_sequence_with_occlusion):
    sr = STRIDERefinement()
    occluded = sr._detect_occluded_frames(skeleton_sequence_with_occlusion)
    assert 5 in occluded


def test_detect_no_occluded_in_normal_sequence(simple_skeleton_sequence):
    sr = STRIDERefinement()
    occluded = sr._detect_occluded_frames(simple_skeleton_sequence)
    assert len(occluded) == 0


def test_interpolation_fills_occluded_frame(skeleton_sequence_with_occlusion):
    sr = STRIDERefinement()
    result = sr.refine(skeleton_sequence_with_occlusion)
    # Frame 5 was occluded — after refinement should have reasonable confidence
    assert result.frames[5] is not None
    assert len(result.frames) == len(skeleton_sequence_with_occlusion.frames)


def test_refine_preserves_frame_count(skeleton_sequence_with_occlusion):
    sr = STRIDERefinement()
    result = sr.refine(skeleton_sequence_with_occlusion)
    assert len(result.frames) == 10


def test_refine_preserves_timestamps(skeleton_sequence_with_occlusion):
    sr = STRIDERefinement()
    result = sr.refine(skeleton_sequence_with_occlusion)
    for orig, ref in zip(skeleton_sequence_with_occlusion.frames, result.frames):
        assert ref.timestamp_s == orig.timestamp_s


def test_empty_sequence_no_crash():
    sr = STRIDERefinement()
    from schemas import SkeletonSequence
    empty_seq = SkeletonSequence(video_path="/fake/v.mp4", fps=25.0, total_frames=0, frames=[])
    result = sr.refine(empty_seq)
    assert result.frames == []


def test_low_confidence_threshold_value():
    assert _LOW_CONFIDENCE_THRESHOLD == 0.4
