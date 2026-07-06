import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';

enum AnnouncementUrgency { hazard, info }

class Announcement {
  final String textEn;
  final String textTranslated;
  final Uint8List? audioBytes;
  final AnnouncementUrgency urgency;

  Announcement({
    required this.textEn,
    required this.textTranslated,
    this.audioBytes,
    this.urgency = AnnouncementUrgency.info,
  });
}

class AudioService {
  final AudioPlayer _player = AudioPlayer();
  final List<Announcement> _queue = [];
  bool _isPlaying = false;
  Announcement? _currentAnnouncement;

  AudioService() {
    _player.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        _isPlaying = false;
        _currentAnnouncement = null;
        _playNext();
      }
    });
  }

  void dispose() {
    _player.dispose();
  }

  void enqueue(Announcement announcement) {
    if (!_isPlaying) {
      _play(announcement);
    } else if (announcement.urgency == AnnouncementUrgency.hazard) {
      _player.stop();
      _queue.insert(0, announcement);
      _playNext();
    } else {
      _queue.add(announcement);
    }
  }

  void skip() {
    _player.stop();
    _isPlaying = false;
    _currentAnnouncement = null;
    _playNext();
  }

  AudioPlayer get player => _player;

  Announcement? get currentAnnouncement => _currentAnnouncement;

  void _play(Announcement announcement) async {
    try {
      _isPlaying = true;
      _currentAnnouncement = announcement;
      if (announcement.audioBytes != null && announcement.audioBytes!.isNotEmpty) {
        final dir = await getTemporaryDirectory();
        final file = File(
          '${dir.path}/lv_${DateTime.now().millisecondsSinceEpoch}.wav',
        );
        await file.writeAsBytes(announcement.audioBytes!);
        final source = AudioSource.file(file.path);
        await _player.setAudioSource(source);
        await _player.play();
        file.delete().catchError((_) {});
      }
    } catch (e) {
      debugPrint('Audio playback error: $e');
      _isPlaying = false;
      _currentAnnouncement = null;
      _playNext();
    }
  }

  void _playNext() {
    if (_queue.isNotEmpty) {
      final next = _queue.removeAt(0);
      _play(next);
    }
  }
}
