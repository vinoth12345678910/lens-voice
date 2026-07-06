class UrgencyResult {
  final int id;
  final String cls;
  final String motion;
  final double area;
  final List<double> bbox;
  final String urgency;

  UrgencyResult({
    required this.id,
    required this.cls,
    required this.motion,
    required this.area,
    required this.bbox,
    required this.urgency,
  });
}

class UrgencyClassifier {
  static final Set<String> hazardVehicleClasses = {
    'car', 'bus', 'truck', 'motorcycle', 'autorickshaw',
    'bicycle', 'trailer', 'caravan', 'train', 'vehicle fallback',
  };

  static final Set<String> hazardPersonClasses = {'person', 'rider'};

  static const double closeAreaThreshold = 0.15 * (512 * 512);

  static List<UrgencyResult> classify(Map<int, dynamic> trackedObjects) {
    final results = <UrgencyResult>[];

    for (final entry in trackedObjects.entries) {
      final objId = entry.key;
      final obj = entry.value;
      final cls = obj.cls as String;
      final motion = obj.motion as String;
      final areaHistory = obj.areaHistory as List<double>;
      final area = areaHistory.isNotEmpty ? areaHistory.last : 0.0;
      final bbox = obj.bbox as List<double>;

      String urgency = 'INFO';

      if (hazardVehicleClasses.contains(cls) && motion == 'approaching') {
        urgency = 'HAZARD';
      } else if (hazardPersonClasses.contains(cls) &&
          motion == 'approaching' &&
          area > closeAreaThreshold) {
        urgency = 'HAZARD';
      } else if (hazardVehicleClasses.contains(cls) &&
          area > closeAreaThreshold * 1.5) {
        urgency = 'HAZARD';
      }

      results.add(UrgencyResult(
        id: objId,
        cls: cls,
        motion: motion,
        area: area,
        bbox: bbox,
        urgency: urgency,
      ));
    }

    return results;
  }
}
