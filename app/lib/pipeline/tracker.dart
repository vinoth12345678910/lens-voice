import 'dart:collection';
import 'dart:math';

class TrackedObject {
  List<double> bbox;
  String cls;
  List<double> centroid;
  List<double> areaHistory;
  int missed;
  String motion;

  TrackedObject({
    required this.bbox,
    required this.cls,
    required this.centroid,
    required this.areaHistory,
    this.missed = 0,
    this.motion = 'new',
  });
}

class Detection {
  final List<double> bbox;
  final String cls;

  Detection({required this.bbox, required this.cls});
}

class Tracker {
  static const double iouThreshold = 0.3;
  static const double centroidDistThreshold = 60.0;
  static const int maxMissedFrames = 5;
  static const double growthApproaching = 0.08;
  static const int maxAreaHistory = 5;

  int _nextId = 0;
  final LinkedHashMap<int, TrackedObject> objects = LinkedHashMap();
  final int _maxMissedFrames;

  Tracker({int? maxMissedFrames})
      : _maxMissedFrames = maxMissedFrames ?? Tracker.maxMissedFrames;

  List<double> _centroid(List<double> bbox) {
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
  }

  double _bboxArea(List<double> bbox) {
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]);
  }

  double _iou(List<double> boxA, List<double> boxB) {
    final xA = max(boxA[0], boxB[0]);
    final yA = max(boxA[1], boxB[1]);
    final xB = min(boxA[2], boxB[2]);
    final yB = min(boxA[3], boxB[3]);
    final inter = max(0.0, xB - xA) * max(0.0, yB - yA);
    final union = _bboxArea(boxA) + _bboxArea(boxB) - inter;
    return union > 0 ? inter / union : 0;
  }

  Map<int, TrackedObject> update(List<Detection> detections) {
    final unmatched = List.generate(detections.length, (i) => i);

    for (final objId in objects.keys.toList()) {
      final obj = objects[objId]!;
      int? bestMatch;
      double bestIou = -1;
      final objCentroid = obj.centroid;

      for (final i in unmatched) {
        final det = detections[i];
        if (det.cls != obj.cls) continue;

        final iou = _iou(obj.bbox, det.bbox);
        final detCentroid = _centroid(det.bbox);
        final dist = sqrt(
          pow(objCentroid[0] - detCentroid[0], 2) +
              pow(objCentroid[1] - detCentroid[1], 2),
        );
        final qualifies =
            iou > iouThreshold || dist < centroidDistThreshold;

        if (qualifies && iou > bestIou) {
          bestIou = iou;
          bestMatch = i;
        }
      }

      if (bestMatch != null) {
        final det = detections[bestMatch];
        final areaPrev = _bboxArea(obj.bbox);
        final areaNow = _bboxArea(det.bbox);

        obj.bbox = det.bbox;
        obj.centroid = _centroid(det.bbox);
        obj.missed = 0;
        obj.areaHistory.add(areaNow);
        if (obj.areaHistory.length > maxAreaHistory) {
          obj.areaHistory.removeAt(0);
        }

        final growth =
            areaPrev > 0 ? (areaNow - areaPrev) / areaPrev : 0;
        obj.motion = growth > growthApproaching
            ? 'approaching'
            : growth < -growthApproaching
                ? 'receding'
                : 'static';

        unmatched.remove(bestMatch);
      } else {
        obj.missed += 1;
      }
    }

    objects.removeWhere((_, obj) => obj.missed > _maxMissedFrames);

    for (final i in unmatched) {
      final det = detections[i];
      objects[_nextId] = TrackedObject(
        bbox: det.bbox,
        cls: det.cls,
        centroid: _centroid(det.bbox),
        areaHistory: [_bboxArea(det.bbox)],
        missed: 0,
        motion: 'new',
      );
      _nextId++;
    }

    return objects;
  }
}
