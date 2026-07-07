import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../pipeline/tracker.dart';

enum ModelLoadState { notLoaded, loading, ready, failed }

class DetectionService {
  Interpreter? _interpreter;
  final ValueNotifier<ModelLoadState> loadState =
      ValueNotifier(ModelLoadState.notLoaded);

  static const int _inputSize = 320;
  static const double _confThreshold = 0.25;
  static const double _nmsThreshold = 0.5;

  bool get isLoaded => loadState.value == ModelLoadState.ready;

  Future<void> loadModel() async {
    if (loadState.value == ModelLoadState.ready) return;
    if (loadState.value == ModelLoadState.loading) return;

    loadState.value = ModelLoadState.loading;

    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        'models/best.tflite',
        options: options,
      );
      loadState.value = ModelLoadState.ready;
    } catch (e) {
      debugPrint('Failed to load TFLite model: $e');
      loadState.value = ModelLoadState.failed;
    }
  }

  List<Detection>? runInference(Uint8List jpegBytes) {
    if (!isLoaded || _interpreter == null) return null;

    try {
      final image = img.decodeImage(jpegBytes);
      if (image == null) return null;

      final resized = img.copyResize(image, width: _inputSize, height: _inputSize);
      final input = _imageToFloat32ListNCHW(resized);

      final outputShape = [1, 19, 2100];
      final output = List<double>.filled(1 * 19 * 2100, 0.0).reshape(outputShape);
      _interpreter!.run(input, output);

      final typedOutput = output as List<List<List<double>>>;
      return _parseOutput(typedOutput);
    } catch (e) {
      debugPrint('Inference error: $e');
      return null;
    }
  }

  Float32List _imageToFloat32ListNCHW(img.Image image) {
    final input = Float32List(1 * 3 * _inputSize * _inputSize);
    int cIndex = 0;
    final rOffset = 0 * _inputSize * _inputSize;
    final gOffset = 1 * _inputSize * _inputSize;
    final bOffset = 2 * _inputSize * _inputSize;

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = image.getPixel(x, y);
        input[rOffset + cIndex] = pixel.r / 255.0;
        input[gOffset + cIndex] = pixel.g / 255.0;
        input[bOffset + cIndex] = pixel.b / 255.0;
        cIndex++;
      }
    }
    return input;
  }

  List<Detection> _parseOutput(List<List<List<double>>> output) {
    final raw = output[0];
    final numPredictions = raw[0].length;

    final boxes = <_Box>[];

    for (int i = 0; i < numPredictions; i++) {
      final cx = raw[0][i];
      final cy = raw[1][i];
      final w = raw[2][i];
      final h = raw[3][i];

      double bestScore = 0;
      int bestClass = -1;
      for (int c = 0; c < 15; c++) {
        final score = raw[4 + c][i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < _confThreshold || bestClass < 0) continue;

      final x1 = ((cx - w / 2) * _inputSize).clamp(0, _inputSize - 1).toDouble();
      final y1 = ((cy - h / 2) * _inputSize).clamp(0, _inputSize - 1).toDouble();
      final x2 = ((cx + w / 2) * _inputSize).clamp(0, _inputSize - 1).toDouble();
      final y2 = ((cy + h / 2) * _inputSize).clamp(0, _inputSize - 1).toDouble();

      boxes.add(_Box(
        x1: x1, y1: y1, x2: x2, y2: y2,
        score: bestScore, classId: bestClass,
      ));
    }

    boxes.sort((a, b) => b.score.compareTo(a.score));

    final kept = <_Box>[];
    for (final box in boxes) {
      bool suppressed = false;
      for (final keep in kept) {
        if (_iou(box, keep) > _nmsThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept.add(box);
      }
    }

    return kept.map((b) {
      final clsName = _getClassName(b.classId);
      return Detection(
        bbox: [b.x1, b.y1, b.x2, b.y2],
        cls: clsName ?? 'unknown',
      );
    }).toList();
  }

  double _iou(_Box a, _Box b) {
    final xA = max(a.x1, b.x1);
    final yA = max(a.y1, b.y1);
    final xB = min(a.x2, b.x2);
    final yB = min(a.y2, b.y2);
    final inter = max(0.0, xB - xA) * max(0.0, yB - yA);
    final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    final union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
  }

  String? _getClassName(int id) {
    const classNames = [
      'animal', 'autorickshaw', 'bicycle', 'bus', 'car',
      'caravan', 'motorcycle', 'person', 'rider', 'traffic light',
      'traffic sign', 'trailer', 'train', 'truck', 'vehicle fallback',
    ];
    if (id >= 0 && id < classNames.length) return classNames[id];
    return null;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    loadState.value = ModelLoadState.notLoaded;
  }
}

class _Box {
  final double x1, y1, x2, y2, score;
  final int classId;
  _Box({
    required this.x1, required this.y1,
    required this.x2, required this.y2,
    required this.score, required this.classId,
  });
}
