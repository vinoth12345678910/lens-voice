import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../services/preferences_service.dart';
import '../utils/accessibility_helpers.dart';

class Language {
  final String code;
  final String displayName;
  final String nativeName;
  final String ttsSample;

  const Language({
    required this.code,
    required this.displayName,
    required this.nativeName,
    required this.ttsSample,
  });
}

const List<Language> languages = [
  Language(
    code: 'en-IN',
    displayName: 'English',
    nativeName: 'English',
    ttsSample: 'English selected',
  ),
  Language(
    code: 'ta-IN',
    displayName: 'Tamil',
    nativeName: 'தமிழ்',
    ttsSample: 'தமிழ் தேர்ந்தெடுக்கப்பட்டது',
  ),
  Language(
    code: 'hi-IN',
    displayName: 'Hindi',
    nativeName: 'हिन्दी',
    ttsSample: 'हिन्दी चयनित',
  ),
];

class LanguageSelectScreen extends StatefulWidget {
  final bool showSkip;

  const LanguageSelectScreen({super.key, this.showSkip = true});

  @override
  State<LanguageSelectScreen> createState() => _LanguageSelectScreenState();
}

class _LanguageSelectScreenState extends State<LanguageSelectScreen> {
  final FlutterTts _tts = FlutterTts();
  String? _selectedCode;

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }

  Future<void> _selectLanguage(Language lang) async {
    setState(() => _selectedCode = lang.code);
    await PreferencesService.setLanguage(lang.code);
    await _tts.setLanguage(lang.code);
    await _tts.speak(lang.ttsSample);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Semantics(
        label: 'Choose your language',
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 60),
              const Text(
                'Choose your language',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 48),
              ...languages.map((lang) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Semantics(
                      label: lang.displayName,
                      hint: 'Select ${lang.displayName} language',
                      button: true,
                      child: SizedBox(
                        width: double.infinity,
                        height: 80,
                        child: ElevatedButton(
                          onPressed: () => _selectLanguage(lang),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: _selectedCode == lang.code
                                ? Colors.white
                                : Colors.white.withOpacity(0.15),
                            foregroundColor: _selectedCode == lang.code
                                ? Colors.black
                                : Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          child: Text(
                            lang.nativeName,
                            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
                          ),
                        ),
                      ),
                    ),
                  )),
              const Spacer(),
              if (widget.showSkip)
                Semantics(
                  label: 'Continue with selected language',
                  button: true,
                  child: SizedBox(
                    width: double.infinity,
                    height: 64,
                    child: ElevatedButton(
                      onPressed: _selectedCode != null
                          ? () => Navigator.of(context).pop(true)
                          : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.black,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(32),
                        ),
                      ),
                      child: const Text(
                        'Continue',
                        style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}
