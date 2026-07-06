import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../services/preferences_service.dart';
import '../services/sarvam_service.dart';
import '../utils/accessibility_helpers.dart';
import 'language_select_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final FlutterTts _tts = FlutterTts();
  String _currentLanguage = 'en-IN';
  bool _testing = false;
  String? _testResult;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final lang = await PreferencesService.getLanguage();
    setState(() => _currentLanguage = lang);
  }

  Future<void> _testSarvamConnection() async {
    setState(() {
      _testing = true;
      _testResult = null;
    });

    const apiKey = String.fromEnvironment('SARVAM_API_KEY');
    if (apiKey.isEmpty) {
      setState(() {
        _testResult = 'API key not set. Use --dart-define=SARVAM_API_KEY=xxx';
        _testing = false;
      });
      return;
    }

    try {
      final service = SarvamService(apiKey: apiKey);
      final result = await service.synthesizeWithTranslation(
        text: 'Hello from LensVoice',
        targetLanguage: 'ta-IN',
        speaker: 'priya',
      );

      setState(() => _testResult = 'Sarvam OK');
      await _tts.setLanguage('en-US');
      await _tts.speak('Sarvam connection successful');
    } catch (e) {
      setState(() => _testResult = 'Connection failed');
      await _tts.setLanguage('en-US');
      await _tts.speak(
          'Connection failed. Check your internet connection and API key.');
    } finally {
      setState(() => _testing = false);
    }
  }

  Future<void> _changeLanguage() async {
    final result = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => const LanguageSelectScreen(showSkip: false),
      ),
    );
    if (result == true) {
      _loadSettings();
    }
  }

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: Semantics(
        label: 'Settings',
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _sectionHeader('Sarvam AI Connection'),
            const SizedBox(height: 8),
            Semantics(
              label: _testResult != null
                  ? 'Test result: $_testResult'
                  : 'Test Sarvam connection',
              button: true,
              child: SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton(
                  onPressed: _testing ? null : _testSarvamConnection,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _testResult == 'Sarvam OK'
                        ? Colors.green
                        : Colors.white,
                    foregroundColor: _testResult == 'Sarvam OK'
                        ? Colors.white
                        : Colors.black,
                  ),
                  child: Text(
                    _testing
                        ? 'Testing...'
                        : _testResult ?? 'Test Sarvam Connection',
                    style: const TextStyle(fontSize: 18),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Semantics(
              label: 'API key note',
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text(
                  'API key is set at build time via --dart-define=SARVAM_API_KEY=xxx. '
                  'It cannot be changed at runtime.',
                  style: TextStyle(color: Colors.white54, fontSize: 13),
                ),
              ),
            ),

            const SizedBox(height: 32),
            _sectionHeader('Language & Voice'),
            const SizedBox(height: 8),
            Semantics(
              label: 'Selected language',
              child: ListTile(
                title: Text(
                  'Language: ${_languageDisplayName(_currentLanguage)}',
                  style: const TextStyle(color: Colors.white, fontSize: 18),
                ),
                trailing:
                    const Icon(Icons.chevron_right, color: Colors.white54),
                onTap: _changeLanguage,
              ),
            ),

            const SizedBox(height: 32),
            _sectionHeader('About'),
            const SizedBox(height: 8),
            Semantics(
              label: 'Battery and data usage information',
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Battery & Data Usage',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Continuous camera use and AI inference will use significant '
                    'battery. Sarvam API calls require an internet connection.',
                      style: TextStyle(color: Colors.white54, fontSize: 14),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            const Center(
              child: Text(
                'LensVoice v1.0.0',
                style: TextStyle(color: Colors.white24, fontSize: 14),
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _sectionHeader(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Colors.white70,
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 1,
      ),
    );
  }

  String _languageDisplayName(String code) {
    switch (code) {
      case 'en-IN':
        return 'English';
      case 'ta-IN':
        return 'தமிழ் (Tamil)';
      case 'hi-IN':
        return 'हिन्दी (Hindi)';
      default:
        return code;
    }
  }
}
