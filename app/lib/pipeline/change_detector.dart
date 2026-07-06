class ChangeDetector {
  static const int positionBucketSize = 3;
  Map<String, Map<String, String>> _lastSpokenState = {};

  String _bucketPosition(List<double> bbox, int frameWidth) {
    final cx = (bbox[0] + bbox[2]) / 2;
    final bucketWidth = frameWidth / positionBucketSize;
    final idx = (cx / bucketWidth).floor();
    const labels = ['left', 'center', 'right'];
    return labels[idx.clamp(0, labels.length - 1)];
  }

  String _bucketDistance(double area, int frameArea) {
    final ratio = area / frameArea;
    if (ratio > 0.15) return 'near';
    if (ratio > 0.04) return 'medium';
    return 'far';
  }

  List<Map<String, dynamic>> check(
    List<dynamic> infoObjects,
    int frameWidth,
    int frameHeight,
  ) {
    final frameArea = frameWidth * frameHeight;
    final changes = <Map<String, dynamic>>[];
    final currentState = <String, Map<String, String>>{};

    for (final obj in infoObjects) {
      final cls = obj.cls as String;
      final position = _bucketPosition(obj.bbox as List<double>, frameWidth);
      final distance = _bucketDistance(
          (obj.area as double).toDouble(), frameArea);

      currentState[cls] = {'position': position, 'distance': distance};

      final prev = _lastSpokenState[cls];
      if (prev == null) {
        changes.add({
          'cls': cls,
          'position': position,
          'distance': distance,
          'bbox': obj.bbox,
          'urgency': 'INFO',
          'motion': obj.motion,
          'area': obj.area,
        });
      } else if (prev['position'] != position ||
          prev['distance'] != distance) {
        changes.add({
          'cls': cls,
          'position': position,
          'distance': distance,
          'bbox': obj.bbox,
          'urgency': 'INFO',
          'motion': obj.motion,
          'area': obj.area,
        });
      }
    }

    _lastSpokenState = currentState;
    return changes;
  }

  void reset() {
    _lastSpokenState = {};
  }
}
