import 'package:flutter/material.dart';
import 'services/preferences_service.dart';
import 'screens/onboarding_screen.dart';
import 'screens/language_select_screen.dart';
import 'screens/camera_screen.dart';
import 'screens/settings_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const LensVoiceApp());
}

class LensVoiceApp extends StatelessWidget {
  const LensVoiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LensVoice',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const AppEntryPoint(),
      routes: {
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}

class AppEntryPoint extends StatefulWidget {
  const AppEntryPoint({super.key});

  @override
  State<AppEntryPoint> createState() => _AppEntryPointState();
}

class _AppEntryPointState extends State<AppEntryPoint> {
  bool _loading = true;
  bool _onboardingComplete = false;
  bool _languageSelected = false;

  @override
  void initState() {
    super.initState();
    _checkFirstLaunch();
  }

  Future<void> _checkFirstLaunch() async {
    final onboarding = await PreferencesService.isOnboardingComplete();
    final lang = await PreferencesService.getLanguage();
    setState(() {
      _onboardingComplete = onboarding;
      _languageSelected = lang != 'en-IN' || onboarding;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (!_onboardingComplete) {
      return OnboardingScreen(onComplete: _onOnboardingComplete);
    }

    if (!_languageSelected) {
      return LanguageSelectScreen(
        showSkip: true,
      );
    }

    return const CameraScreen();
  }

  void _onOnboardingComplete() async {
    await PreferencesService.setOnboardingComplete();
    setState(() => _onboardingComplete = true);
  }
}
