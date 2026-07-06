const Map<String, String> friendlyNames = {
  'car': 'a car',
  'bus': 'a bus',
  'truck': 'a truck',
  'motorcycle': 'a motorcycle',
  'autorickshaw': 'an auto-rickshaw',
  'bicycle': 'a bicycle',
  'person': 'a person',
  'rider': 'a rider',
  'animal': 'an animal',
  'traffic light': 'a traffic light',
  'traffic sign': 'a traffic sign',
  'trailer': 'a trailer',
  'caravan': 'a caravan',
  'train': 'a train',
  'vehicle fallback': 'a vehicle',
};

String generateDescription(Map<String, dynamic> obj) {
  final cls = obj['cls'] as String? ?? '';
  final urgency = obj['urgency'] as String? ?? 'INFO';
  final motion = obj['motion'] as String? ?? 'static';
  final position = obj['position'] as String? ?? 'ahead';
  final distance = obj['distance'] as String? ?? 'medium';

  final subject = friendlyNames[cls] ?? cls;

  if (urgency == 'HAZARD') {
    if (motion == 'approaching') {
      return 'Careful, $subject is approaching from your $position.';
    }
    return 'Careful, $subject is close on your $position.';
  }

  const distancePhrases = {
    'near': 'close by',
    'medium': 'nearby',
    'far': 'further away',
  };
  final phrase = distancePhrases[distance] ?? 'nearby';
  return 'There is $subject $phrase on your $position.';
}
