import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../pipeline/tracker.dart';

enum ModelLoadState { notLoaded, downloading, loading, ready, failed }

class DetectionService {
  Interpreter? _interpreter;
  final ValueNotifier<ModelLoadState> loadState =
      ValueNotifier(ModelLoadState.notLoaded);

  static const int _inputSize = 320;
  static const double _confThreshold = 0.1;
  static const double _nmsThreshold = 0.5;

  bool get isLoaded => loadState.value == ModelLoadState.ready;

  Future<void> loadModel(String filePath) async {
    if (loadState.value == ModelLoadState.ready) return;
    if (loadState.value == ModelLoadState.loading) return;

    loadState.value = ModelLoadState.loading;

    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = Interpreter.fromFile(
        File(filePath),
        options: options,
      );
      loadState.value = ModelLoadState.ready;
    } catch (e) {
      debugPrint('Failed to load TFLite model: $e');
      loadState.value = ModelLoadState.failed;
    }
  }

  List<Detection>? runInference(CameraImage cameraImage) {
    if (!isLoaded || _interpreter == null) return null;

    try {
      final input = _cameraImageToNCHW(cameraImage);
      if (input == null) return null;

      final output = List<double>.filled(1 * 300 * 6, 0.0).reshape([1, 300, 6]);
      _interpreter!.run(input, output);

      return _parseOutput(output as List<List<List<double>>>);
    } catch (e) {
      debugPrint('Inference error: $e');
      return null;
    }
  }

  Float32List? _cameraImageToNCHW(CameraImage cameraImage) {
    final format = cameraImage.format.group;

    if (format == ImageFormatGroup.jpeg) {
      final image = img.decodeImage(cameraImage.planes[0].bytes);
      if (image == null) return null;
      final resized = img.copyResize(image, width: _inputSize, height: _inputSize);
      return _imageToFloat32ListNCHW(resized);
    }

    if (format == ImageFormatGroup.nv21 || format == ImageFormatGroup.yuv420) {
      return _yuvToNCHW(cameraImage);
    }

    if (cameraImage.planes.length >= 2) {
      return _yuvToNCHW(cameraImage);
    }

    return null;
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

  Float32List _yuvToNCHW(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final planes = image.planes;
    final yPlane = planes[0].bytes;
    final isNv21 = planes.length == 2;

    final output = Float32List(1 * 3 * _inputSize * _inputSize);
    final rOffset = 0 * _inputSize * _inputSize;
    final gOffset = 1 * _inputSize * _inputSize;
    final bOffset = 2 * _inputSize * _inputSize;

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final srcX = x * width ~/ _inputSize;
        final srcY = y * height ~/ _inputSize;

        final yIdx = srcY * width + srcX;
        final uvW = width >> 1;
        final uvIdx = (srcY >> 1) * uvW + (srcX >> 1);

        final Y = yPlane[yIdx].toDouble();

        double U, V;
        if (isNv21) {
          final vuPlane = planes[1].bytes;
          V = vuPlane[uvIdx * 2].toDouble();
          U = vuPlane[uvIdx * 2 + 1].toDouble();
        } else {
          V = planes[2].bytes[uvIdx].toDouble();
          U = planes[1].bytes[uvIdx].toDouble();
        }

        final r = _clampByte(Y + 1.402 * (V - 128));
        final g = _clampByte(Y - 0.344136 * (U - 128) - 0.714136 * (V - 128));
        final b = _clampByte(Y + 1.772 * (U - 128));

        final cIndex = y * _inputSize + x;
        output[rOffset + cIndex] = r / 255.0;
        output[gOffset + cIndex] = g / 255.0;
        output[bOffset + cIndex] = b / 255.0;
      }
    }
    return output;
  }

  int _clampByte(double v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v.round();
  }

  List<Detection> _parseOutput(List<List<List<double>>> output) {
    final raw = output[0];
    final boxes = <_Box>[];

    for (int i = 0; i < 300; i++) {
      final x1 = raw[i][0].clamp(0, _inputSize - 1).toDouble();
      final y1 = raw[i][1].clamp(0, _inputSize - 1).toDouble();
      final x2 = raw[i][2].clamp(0, _inputSize - 1).toDouble();
      final y2 = raw[i][3].clamp(0, _inputSize - 1).toDouble();
      final conf = raw[i][4];
      final clsId = raw[i][5].round();

      if (conf < _confThreshold) continue;

      boxes.add(_Box(
        x1: x1, y1: y1, x2: x2, y2: y2,
        score: conf, classId: clsId,
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
