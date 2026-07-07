import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:just_audio/just_audio.dart';
import '../services/audio_service.dart' as audio_svc;
import '../services/camera_service.dart';
import '../services/detection_service.dart';
import '../services/sarvam_service.dart';
import '../services/preferences_service.dart';
import '../pipeline/tracker.dart';
import '../pipeline/urgency_classifier.dart';
import '../pipeline/change_detector.dart';
import '../pipeline/description_generator.dart';
import '../utils/accessibility_helpers.dart';
import '../widgets/status_indicator.dart';
import '../widgets/hazard_overlay.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  final audio_svc.AudioService _audioService = audio_svc.AudioService();
  final CameraService _cameraService = CameraService();
  final DetectionService _detectionService = DetectionService();
  final FlutterTts _tts = FlutterTts();
  final Tracker _tracker = Tracker();
  final ChangeDetector _changeDetector = ChangeDetector();

  SarvamService? _sarvamService;

  HazardLevel _hazardLevel = HazardLevel.none;
  String _currentMessage = '';
  String _currentTranslatedMessage = '';
  bool _isListening = false;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  bool _wasPausedByBackground = false;
  bool _isLoadingModel = false;

  StreamSubscription? _playerStateSub;
  VoidCallback? _modelLoadListener;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  Future<void> _init() async {
    await _initCamera();
    _initSarvam();
    _watchAudioState();
    _speakReady();
  }

  Future<void> _initCamera() async {
    final ok = await _cameraService.initialize();
    if (mounted) setState(() => _isCameraReady = ok);
  }

  void _initSarvam() {
    const apiKey = String.fromEnvironment('SARVAM_API_KEY');
    if (apiKey.isNotEmpty) {
      _sarvamService = SarvamService(apiKey: apiKey);
    }
  }

  void _watchAudioState() {
    _playerStateSub = _audioService.player.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        if (mounted && _hazardLevel != HazardLevel.none) {
          setState(() => _hazardLevel = HazardLevel.none);
        }
      }
    });
  }

  Future<void> _speak(String text) async {
    try {
      await _tts.setLanguage('en-US');
      await _tts.speak(text);
    } catch (_) {}
  }

  void _speakReady() {
    _speak('Camera ready. Double tap to start listening.');
  }

  Future<void> _toggleListening() async {
    if (!_isListening) {
      if (_sarvamService == null) {
        _speak('Sarvam API key not configured. Set it with dart define at build time.');
        return;
      }

      final currentState = _detectionService.loadState.value;
      if (currentState == ModelLoadState.notLoaded ||
          currentState == ModelLoadState.failed) {
        setState(() => _isLoadingModel = true);
        _speak('Loading, please wait.');

        await _detectionService.loadModel();
        if (!mounted) return;

        setState(() => _isLoadingModel = false);

        if (_detectionService.loadState.value == ModelLoadState.failed) {
          _speak('Could not load the AI model. Please restart the app.');
          return;
        }

        _speak('Ready.');
      } else if (currentState == ModelLoadState.loading) {
        _speak('Model is still loading. Please wait.');
        return;
      }

      await _cameraService.startStreaming();
      _cameraService.onFrame = _onFrame;

      setState(() => _isListening = true);
      _speak('Listening');
    } else {
      await _cameraService.stopStreaming();
      setState(() {
        _isListening = false;
        _hazardLevel = HazardLevel.none;
      });
      _tracker.objects.clear();
      _changeDetector.reset();
      _speak('Stopped');
    }
  }

  void _onFrame(Uint8List jpegBytes) {
    if (_isProcessing) return;
    _isProcessing = true;

    final detections = _detectionService.runInference(jpegBytes);
    if (detections == null || detections.isEmpty) {
      _isProcessing = false;
      return;
    }

    try {
      final tracked = _tracker.update(detections);
      final urgencyResults = UrgencyClassifier.classify(tracked);

      final hazards =
          urgencyResults.where((r) => r.urgency == 'HAZARD').toList();
      final infoObjects =
          urgencyResults.where((r) => r.urgency == 'INFO').toList();

      if (hazards.isNotEmpty) {
        _handleHazard(hazards.first);
      } else {
        final changes =
            _changeDetector.check(infoObjects, 320, 320);
        if (changes.isNotEmpty) {
          _handleInfo(changes.first);
        }
      }
    } catch (e) {
      debugPrint('Pipeline error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  void _handleHazard(UrgencyResult hazard) {
    final obj = {
      'cls': hazard.cls,
      'urgency': 'HAZARD',
      'motion': hazard.motion,
      'position': 'ahead',
      'distance': 'near',
    };
    final sentence = generateDescription(obj);

    if (mounted) {
      setState(() {
        _hazardLevel = HazardLevel.hazard;
        _currentMessage = sentence;
      });
    }
    triggerHapticHazard();

    _synthesizeAndPlay(sentence, audio_svc.AnnouncementUrgency.hazard);

    Future.delayed(const Duration(seconds: 4), () {
      if (mounted) setState(() => _hazardLevel = HazardLevel.none);
    });
  }

  void _handleInfo(Map<String, dynamic> change) {
    change['urgency'] = 'INFO';
    change['position'] = change['position'] ?? 'ahead';
    change['distance'] = change['distance'] ?? 'medium';
    final sentence = generateDescription(change);

    if (mounted) {
      setState(() {
        _hazardLevel = HazardLevel.info;
        _currentMessage = sentence;
        _currentTranslatedMessage = '';
      });
    }
    triggerHapticInfo();

    _synthesizeAndPlay(sentence, audio_svc.AnnouncementUrgency.info);

    Future.delayed(const Duration(seconds: 4), () {
      if (mounted) setState(() => _hazardLevel = HazardLevel.none);
    });
  }

  Future<void> _synthesizeAndPlay(
    String sentence,
    audio_svc.AnnouncementUrgency urgency,
  ) async {
    if (_sarvamService == null) return;

    try {
      final lang = await PreferencesService.getLanguage();
      final speaker = await PreferencesService.getSpeaker();
      final result =
          await _sarvamService!.synthesizeWithTranslation(
        text: sentence,
        targetLanguage: lang,
        speaker: speaker,
      );

      if (mounted && result.translatedText != sentence) {
        setState(() => _currentTranslatedMessage = result.translatedText);
      }

      _audioService.enqueue(audio_svc.Announcement(
        textEn: sentence,
        textTranslated: result.translatedText,
        audioBytes: result.audioBytes,
        urgency: urgency,
      ));
    } catch (e) {
      debugPrint('Sarvam error: $e');
    }
  }

  void _skipSpeech() {
    _audioService.skip();
  }

  void _openSettings() {
    Navigator.of(context).pushNamed('/settings');
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      if (_isListening) {
        _wasPausedByBackground = true;
        _cameraService.pauseStreaming();
        _speak('Paused');
      }
    } else if (state == AppLifecycleState.resumed) {
      if (_wasPausedByBackground && _isListening) {
        _wasPausedByBackground = false;
        _cameraService.resumeStreaming();
        _speak('Resumed');
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _playerStateSub?.cancel();
    _cameraService.dispose();
    _audioService.dispose();
    _detectionService.dispose();
    _tts.stop();
    _modelLoadListener?.call();
    super.dispose();
  }

  Widget _buildModelStatus() {
    final state = _detectionService.loadState.value;
    if (state == ModelLoadState.loading || _isLoadingModel) {
      return const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 14, height: 14,
            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.orange),
          ),
          SizedBox(width: 8),
          Text(
            'Loading AI Model...',
            style: TextStyle(color: Colors.orange, fontSize: 14, fontWeight: FontWeight.w600),
          ),
        ],
      );
    }
    if (state == ModelLoadState.ready) {
      return const Text(
        'Model Ready',
        style: TextStyle(color: Colors.greenAccent, fontSize: 14, fontWeight: FontWeight.w600),
      );
    }
    if (state == ModelLoadState.failed) {
      return const Text(
        'Model Failed',
        style: TextStyle(color: Colors.redAccent, fontSize: 14, fontWeight: FontWeight.w600),
      );
    }
    return const Text(
      'Tap to Start',
      style: TextStyle(color: Colors.white54, fontSize: 14, fontWeight: FontWeight.w600),
    );
  }

  @override
  Widget build(BuildContext context) {
    final modelState = _detectionService.loadState.value;
    final startDisabled = _isLoadingModel || modelState == ModelLoadState.loading;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Semantics(
        label: 'Camera view',
        child: Stack(
          children: [
            if (_isCameraReady && _cameraService.controller != null)
              CameraPreview(_cameraService.controller!)
            else
              const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.camera_alt, color: Colors.white38, size: 64),
                    SizedBox(height: 16),
                    Text(
                      'Camera not available',
                      style: TextStyle(color: Colors.white38, fontSize: 18),
                    ),
                  ],
                ),
              ),

            if (_hazardLevel != HazardLevel.none)
              HazardOverlay(
                level: _hazardLevel,
                message: _currentMessage,
                translatedMessage: _currentTranslatedMessage,
              ),

            Positioned(
              top: 48,
              left: 16,
              right: 16,
              child: Row(
                children: [
                  _buildModelStatus(),
                  const Spacer(),
                  Semantics(
                    label: 'Settings',
                    button: true,
                    child: IconButton(
                      icon:
                          const Icon(Icons.settings, color: Colors.white, size: 28),
                      onPressed: _openSettings,
                      style: IconButton.styleFrom(
                        backgroundColor: Colors.black38,
                        minimumSize: const Size(48, 48),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            if (_isLoadingModel || modelState == ModelLoadState.loading)
              Center(
                child: Semantics(
                  label: 'Loading AI model, please wait',
                  liveRegion: true,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                    decoration: BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 24, height: 24,
                          child: CircularProgressIndicator(
                            strokeWidth: 3, color: Colors.orangeAccent,
                          ),
                        ),
                        SizedBox(width: 16),
                        Text(
                          'Loading AI Model...\nPlease wait',
                          style: TextStyle(color: Colors.white, fontSize: 16),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

            Positioned(
              bottom: 60,
              left: 0,
              right: 0,
              child: Column(
                children: [
                  if (_isListening)
                    Semantics(
                      label: 'Skip current speech',
                      button: true,
                      child: TextButton(
                        onPressed: _skipSpeech,
                        child: const Text(
                          'Skip speech',
                          style: TextStyle(color: Colors.white54, fontSize: 16),
                        ),
                      ),
                    ),
                  const SizedBox(height: 8),
                  Semantics(
                    label: startDisabled
                        ? 'Loading model'
                        : _isListening
                            ? 'Stop listening'
                            : 'Start listening',
                    hint: 'Double tap to toggle',
                    button: true,
                    child: GestureDetector(
                      onDoubleTap: startDisabled ? null : _toggleListening,
                      onTap: startDisabled ? null : _toggleListening,
                      child: Container(
                        width: 88,
                        height: 88,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: startDisabled
                              ? Colors.grey
                              : _isListening
                                  ? Colors.red
                                  : Colors.white,
                          border: Border.all(color: Colors.white, width: 3),
                          boxShadow: [
                            BoxShadow(
                              color: startDisabled
                                  ? Colors.grey.withOpacity(0.2)
                                  : _isListening
                                      ? Colors.red.withOpacity(0.4)
                                      : Colors.white.withOpacity(0.2),
                              blurRadius: 20,
                              spreadRadius: 4,
                            ),
                          ],
                        ),
                        child: Icon(
                          _isListening ? Icons.stop : Icons.play_arrow,
                          color: startDisabled || _isListening
                              ? Colors.white
                              : Colors.black,
                          size: 44,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Semantics(
                    label: _isListening ? 'Listening' : 'Tap to start',
                    child: Text(
                      startDisabled
                          ? 'Loading...'
                          : _isListening
                              ? 'Tap to stop'
                              : 'Tap to start',
                      style: const TextStyle(color: Colors.white54, fontSize: 14),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
