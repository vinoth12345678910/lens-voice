from lensvoice.config.settings import TrackingConfig
from lensvoice.models.schemas import BBox, DetectedObject
from lensvoice.tracking.bytetrack import ByteTrackTracker


def make_detection(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: BBox,
    timestamp: float,
) -> DetectedObject:
    return DetectedObject(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        timestamp=timestamp,
    )


def test_bytetrack_assigns_new_track_id():
    tracker = ByteTrackTracker(TrackingConfig())

    detection = make_detection(
        0,
        "person",
        0.9,
        BBox(10, 20, 50, 80),
        1.0,
    )

    output = tracker.update([detection])

    assert output[0].track_id == 1
    assert list(tracker.tracks.keys()) == [1]


def test_bytetrack_keeps_same_id_for_small_movement():
    tracker = ByteTrackTracker(TrackingConfig())

    first = make_detection(
        0,
        "person",
        0.9,
        BBox(10, 20, 50, 80),
        1.0,
    )

    second = make_detection(
        0,
        "person",
        0.9,
        BBox(15, 20, 55, 80),
        2.0,
    )

    first_output = tracker.update([first])
    second_output = tracker.update([second])

    assert first_output[0].track_id == 1
    assert second_output[0].track_id == 1
    assert second_output[0].track_id == first_output[0].track_id


def test_bytetrack_handles_multiple_objects():
    tracker = ByteTrackTracker(TrackingConfig())

    person = make_detection(
        0,
        "person",
        0.9,
        BBox(10, 20, 50, 80),
        1.0,
    )

    car = make_detection(
        1,
        "car",
        0.9,
        BBox(200, 200, 300, 300),
        1.0,
    )

    output = tracker.update([person, car])

    assert output[0].track_id == 1
    assert output[1].track_id == 2
    assert list(tracker.tracks.keys()) == [1, 2]


def test_bytetrack_records_trajectory():
    tracker = ByteTrackTracker(TrackingConfig())

    first = make_detection(
        0,
        "person",
        0.9,
        BBox(10, 20, 50, 80),
        1.0,
    )

    second = make_detection(
        0,
        "person",
        0.9,
        BBox(15, 20, 55, 80),
        2.0,
    )

    tracker.update([first])
    output = tracker.update([second])

    track_id = output[0].track_id
    trajectory = tracker.tracks[track_id]

    assert len(trajectory.bboxes) == 2
    assert trajectory.last_seen == 2.0
    assert trajectory.frames_missing == 0


def test_bytetrack_counts_missing_frames():
    tracker = ByteTrackTracker(TrackingConfig())

    detection = make_detection(
        0,
        "person",
        0.9,
        BBox(10, 20, 50, 80),
        1.0,
    )

    tracker.update([detection])

    assert tracker.tracks[1].frames_missing == 0

    tracker.update([])

    assert tracker.tracks[1].frames_missing == 1

    tracker.update([])

    assert tracker.tracks[1].frames_missing == 2