import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;

class ModelDownloadService {
  static const String _modelUrl =
      'https://huggingface.co/Vinothanand06/lensvoice-model/resolve/main/best.tflite';
  static const String _modelFileName = 'best.tflite';

  final ValueNotifier<double> downloadProgress = ValueNotifier(0.0);

  Future<String> getModelPath() async {
    final dir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${dir.path}/models');
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return '${modelDir.path}/$_modelFileName';
  }

  Future<bool> isModelCached() async {
    final path = await getModelPath();
    return File(path).exists();
  }

  Future<String> ensureModel() async {
    final path = await getModelPath();
    if (await File(path).exists()) {
      return path;
    }
    return _download(path);
  }

  Future<String> _download(String path) async {
    downloadProgress.value = 0.0;

    final client = http.Client();
    final request = http.Request('GET', Uri.parse(_modelUrl));
    final response = await client.send(request);

    if (response.statusCode != 200) {
      throw HttpException('Download failed: HTTP ${response.statusCode}');
    }

    final contentLength = response.contentLength ?? 0;
    final file = File(path);
    final sink = file.openWrite();
    int bytesReceived = 0;

    await for (final chunk in response.stream) {
      sink.add(chunk);
      bytesReceived += chunk.length;
      if (contentLength > 0) {
        downloadProgress.value = bytesReceived / contentLength;
      }
    }

    await sink.close();
    downloadProgress.value = 1.0;
    return path;
  }

  Future<void> clearCache() async {
    final path = await getModelPath();
    final file = File(path);
    if (await file.exists()) {
      await file.delete();
    }
  }

  void dispose() {
    downloadProgress.dispose();
  }
}
