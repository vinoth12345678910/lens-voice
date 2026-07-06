import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../pipeline/tracker.dart';

class DetectionService {
  Interpreter? _interpreter;
  bool _loaded = false;
  bool _loading = false;
  Completer<void>? _loadCompleter;

  static const int _inputSize = 512;
  static const double _confThreshold = 0.25;

  bool get isLoaded => _loaded;

  Future<void> loadModel() async {
    if (_loaded) return;
    if (_loading) return _loadCompleter!.future;
    _loading = true;
    _loadCompleter = Completer<void>();

    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        'models/best.tflite',
        options: options,
      );
      _loaded = true;
      debugPrint('TFLite model loaded successfully');
    } catch (e) {
      debugPrint('Failed to load TFLite model: $e');
    } finally {
      _loading = false;
      _loadCompleter?.complete();
    }
  }

  List<Detection>? runInference(Uint8List jpegBytes) {
    if (!_loaded || _interpreter == null) return null;

    try {
      final image = img.decodeImage(jpegBytes);
      if (image == null) return null;

      final resized = img.copyResize(image, width: _inputSize, height: _inputSize);
      final input = _imageToFloat32List(resized);

      final outputShape = [1, 300, 6];
      final output = List<double>.filled(1 * 300 * 6, 0.0).reshape(outputShape);
      _interpreter!.run(input, output);

      final typedOutput = output as List<List<List<double>>>;
      return _parseOutput(typedOutput);
    } catch (e) {
      debugPrint('Inference error: $e');
      return null;
    }
  }

  Float32List _imageToFloat32List(img.Image image) {
    final input = Float32List(1 * _inputSize * _inputSize * 3);
    int index = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = image.getPixel(x, y);
        input[index++] = pixel.r / 255.0;
        input[index++] = pixel.g / 255.0;
        input[index++] = pixel.b / 255.0;
      }
    }
    return input;
  }

  List<Detection> _parseOutput(List<List<List<double>>> output) {
    final detections = <Detection>[];
    final rawDetections = output[0];

    for (final det in rawDetections) {
      final x1 = det[0];
      final y1 = det[1];
      final x2 = det[2];
      final y2 = det[3];
      final conf = det[4];
      final clsId = det[5].round();

      if (conf < _confThreshold) continue;

      final bbox = [x1, y1, x2, y2];
      // bbox values are normalized 0-1, convert to pixel coordinates
      final pixelBbox = [
        bbox[0] * _inputSize,
        bbox[1] * _inputSize,
        bbox[2] * _inputSize,
        bbox[3] * _inputSize,
      ];

      final clsName = _getClassName(clsId);
      if (clsName == null) continue;

      detections.add(Detection(bbox: pixelBbox, cls: clsName));
    }

    return detections;
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
    _loaded = false;
  }
}
