import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

typedef FrameCallback = void Function(Uint8List jpegBytes);

class CameraService {
  CameraController? _controller;
  bool _isStreaming = false;
  FrameCallback? onFrame;
  bool _isPaused = false;
  Timer? _throttleTimer;
  static const Duration _minInterval = Duration(milliseconds: 500);
  bool _imageStreamActive = false;

  bool get isStreaming => _isStreaming;
  bool get isPaused => _isPaused;
  CameraController? get controller => _controller;

  Future<bool> initialize({CameraDescription? camera}) async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return false;

      final cam = camera ?? cameras.first;
      _controller = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      return true;
    } catch (e) {
      debugPrint('Camera init error: $e');
      return false;
    }
  }

  Future<void> startStreaming() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isStreaming) return;

    _isStreaming = true;
    _isPaused = false;
    _imageStreamActive = true;

    await _controller!.startImageStream(_onImage);
  }

  void _onImage(CameraImage image) {
    if (_isPaused) return;
    if (_throttleTimer != null && _throttleTimer!.isActive) return;

    _throttleTimer = Timer(_minInterval, () {});

    try {
      final jpegBytes = _convertToJpeg(image);
      if (jpegBytes != null && onFrame != null) {
        onFrame!(jpegBytes);
      }
    } catch (e) {
      debugPrint('Frame conversion error: $e');
    }
  }

  Uint8List? _convertToJpeg(CameraImage image) {
    // CameraImage from camera plugin is already JPEG if format is jpeg
    // If not, we'd need to encode. For now return the planes data.
    if (image.format.group == ImageFormatGroup.jpeg) {
      return image.planes[0].bytes;
    }
    return null;
  }

  void pauseStreaming() {
    _isPaused = true;
  }

  void resumeStreaming() {
    _isPaused = false;
  }

  Future<void> stopStreaming() async {
    _isStreaming = false;
    _isPaused = false;
    _imageStreamActive = false;
    _throttleTimer?.cancel();
  }

  Future<void> dispose() async {
    await stopStreaming();
    await _controller?.dispose();
    _controller = null;
  }
}
