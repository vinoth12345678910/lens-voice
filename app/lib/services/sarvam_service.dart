import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class SarvamService {
  final String apiKey;
  static const String _translateUrl = 'https://api.sarvam.ai/translate';
  static const String _ttsUrl = 'https://api.sarvam.ai/text-to-speech';
  static const Duration _timeout = Duration(seconds: 10);

  SarvamService({required this.apiKey});

  Map<String, String> get _headers => {
        'api-subscription-key': apiKey,
        'Content-Type': 'application/json',
      };

  Future<String> translate(String text, String targetLanguage) async {
    if (targetLanguage == 'en-IN') return text;

    try {
      final response = await http
          .post(
            Uri.parse(_translateUrl),
            headers: _headers,
            body: jsonEncode({
              'input': text,
              'source_language_code': 'en-IN',
              'target_language_code': targetLanguage,
            }),
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return data['translated_text'] as String? ?? text;
      } else {
        debugPrint('Translation failed (${response.statusCode}): ${response.body}');
        return text;
      }
    } catch (e) {
      debugPrint('Translation error: $e — falling back to English');
      return text;
    }
  }

  Future<Uint8List?> synthesize(String text, String targetLanguage, String speaker) async {
    try {
      final response = await http
          .post(
            Uri.parse(_ttsUrl),
            headers: _headers,
            body: jsonEncode({
              'inputs': [text],
              'target_language_code': targetLanguage,
              'speaker': speaker,
              'model': 'bulbul:v3',
            }),
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final audios = data['audios'] as List<dynamic>?;
        if (audios != null && audios.isNotEmpty) {
          return base64Decode(audios[0] as String);
        }
        return null;
      } else {
        debugPrint('TTS failed (${response.statusCode}): ${response.body}');
        return null;
      }
    } catch (e) {
      debugPrint('TTS error: $e');
      return null;
    }
  }

  Future<SarvamResult> synthesizeWithTranslation({
    required String text,
    required String targetLanguage,
    required String speaker,
  }) async {
    final translated = await translate(text, targetLanguage);
    final audioBytes = await synthesize(translated, targetLanguage, speaker);
    return SarvamResult(
      translatedText: translated,
      audioBytes: audioBytes,
    );
  }
}

class SarvamResult {
  final String translatedText;
  final Uint8List? audioBytes;

  SarvamResult({required this.translatedText, this.audioBytes});
}
