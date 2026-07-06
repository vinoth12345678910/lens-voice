import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

enum HazardLevel { none, info, hazard }

class HazardOverlay extends StatelessWidget {
  final HazardLevel level;
  final String message;
  final String translatedMessage;

  const HazardOverlay({
    super.key,
    this.level = HazardLevel.none,
    this.message = '',
    this.translatedMessage = '',
  });

  @override
  Widget build(BuildContext context) {
    if (level == HazardLevel.none) return const SizedBox.shrink();

    final (bgColor, borderColor, label) = level == HazardLevel.hazard
        ? (Colors.red.withOpacity(0.85), Colors.redAccent, 'HAZARD')
        : (Colors.blue.withOpacity(0.85), Colors.blueAccent, 'INFO');

    return Semantics(
      label: '$label: $message',
      liveRegion: true,
      child: Positioned(
        top: 100,
        left: 16,
        right: 16,
        child: Material(
          elevation: 8,
          borderRadius: BorderRadius.circular(16),
          color: bgColor,
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: borderColor, width: 2),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    Icon(
                      level == HazardLevel.hazard ? Icons.warning : Icons.info,
                      color: Colors.white,
                      size: 28,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      label,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  translatedMessage.isNotEmpty ? translatedMessage : message,
                  style: const TextStyle(color: Colors.white, fontSize: 18),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
