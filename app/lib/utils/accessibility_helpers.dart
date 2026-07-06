import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void triggerHapticHazard() {
  HapticFeedback.heavyImpact();
  Future.delayed(const Duration(milliseconds: 200), () {
    HapticFeedback.heavyImpact();
  });
}

void triggerHapticInfo() {
  HapticFeedback.lightImpact();
}

Widget accessibleButton({
  required String label,
  required String hint,
  required VoidCallback onPressed,
  required Widget child,
  double minSize = 48.0,
}) {
  return Semantics(
    label: label,
    hint: hint,
    button: true,
    child: ConstrainedBox(
      constraints: BoxConstraints(minWidth: minSize, minHeight: minSize),
      child: GestureDetector(
        onTap: onPressed,
        child: child,
      ),
    ),
  );
}
