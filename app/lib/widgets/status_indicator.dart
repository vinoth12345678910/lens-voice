import 'package:flutter/material.dart';

enum PipelineStatus { idle, modelLoading, modelReady, listening, error }

class StatusIndicator extends StatelessWidget {
  final PipelineStatus status;
  final String? statusText;

  const StatusIndicator({
    super.key,
    this.status = PipelineStatus.idle,
    this.statusText,
  });

  @override
  Widget build(BuildContext context) {
    final (label, color, icon) = _statusInfo;

    return Semantics(
      label: 'Status: $label',
      liveRegion: true,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.9),
          borderRadius: BorderRadius.circular(24),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(
              statusText ?? label,
              style: const TextStyle(
                  color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }

  (String, Color, IconData) get _statusInfo {
    switch (status) {
      case PipelineStatus.idle:
        return ('Ready', Colors.grey, Icons.circle_outlined);
      case PipelineStatus.modelLoading:
        return ('Loading Model', Colors.orange, Icons.hourglass_top);
      case PipelineStatus.modelReady:
        return ('Model Ready', Colors.green, Icons.check_circle);
      case PipelineStatus.listening:
        return ('Listening', Colors.green, Icons.visibility);
      case PipelineStatus.error:
        return ('Error', Colors.red, Icons.error);
    }
  }
}
