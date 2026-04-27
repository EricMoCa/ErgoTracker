import pytest
from schemas import SkeletonSequence
from advanced_pipeline.contact_refinement import ContactRefinement


def test_refine_returns_skeleton_sequence(simple_skeleton_sequence):
    cr = ContactRefinement()
    result = cr.refine(simple_skeleton_sequence)
    assert isinstance(result, SkeletonSequence)


def test_refine_same_frame_count(simple_skeleton_sequence):
    cr = ContactRefinement()
    result = cr.refine(simple_skeleton_sequence)
    assert len(result.frames) == len(simple_skeleton_sequence.frames)


def test_static_foot_detected_as_contact(skeleton_sequence_static_feet):
    cr = ContactRefinement()
    contact = cr._detect_contact(skeleton_sequence_static_feet)
    # All frames after the first should have foot contact (no movement)
    assert any(contact[1:])


def test_moving_foot_not_contact(skeleton_sequence_moving_feet):
    cr = ContactRefinement()
    contact = cr._detect_contact(skeleton_sequence_moving_feet)
    # Ankles moving 0.5m/frame → velocity >> threshold → no contact
    assert not any(contact[1:])


def test_detect_contact_first_frame_always_false(simple_skeleton_sequence):
    cr = ContactRefinement()
    contact = cr._detect_contact(simple_skeleton_sequence)
    assert contact[0] is False


def test_get_contact_events_returns_list(simple_skeleton_sequence):
    cr = ContactRefinement()
    events = cr.get_contact_events(simple_skeleton_sequence)
    assert isinstance(events, list)


def test_get_contact_events_structure(skeleton_sequence_static_feet):
    cr = ContactRefinement()
    events = cr.get_contact_events(skeleton_sequence_static_feet)
    for event in events:
        assert "start_frame" in event
        assert "end_frame" in event
        assert "foot" in event
        assert event["start_frame"] <= event["end_frame"]


def test_empty_sequence_no_crash():
    cr = ContactRefinement()
    from schemas import SkeletonSequence
    empty_seq = SkeletonSequence(video_path="/fake/v.mp4", fps=25.0, total_frames=0, frames=[])
    result = cr.refine(empty_seq)
    assert result.frames == []


def test_contact_labels_length_matches_frames(simple_skeleton_sequence):
    cr = ContactRefinement()
    contact = cr._detect_contact(simple_skeleton_sequence)
    assert len(contact) == len(simple_skeleton_sequence.frames)
