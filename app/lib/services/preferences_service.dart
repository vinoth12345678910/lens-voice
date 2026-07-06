import 'package:shared_preferences/shared_preferences.dart';

class PreferencesService {
  static const _keyBackendHost = 'backend_host';
  static const _keyBackendPort = 'backend_port';
  static const _keyLanguage = 'language';
  static const _keySpeaker = 'speaker';
  static const _keyOnboardingComplete = 'onboarding_complete';

  static Future<SharedPreferences> get _prefs => SharedPreferences.getInstance();

  static Future<String> getBackendHost() async {
    final prefs = await _prefs;
    return prefs.getString(_keyBackendHost) ?? '';
  }

  static Future<void> setBackendHost(String host) async {
    final prefs = await _prefs;
    await prefs.setString(_keyBackendHost, host);
  }

  static Future<String> getBackendPort() async {
    final prefs = await _prefs;
    return prefs.getString(_keyBackendPort) ?? '8000';
  }

  static Future<void> setBackendPort(String port) async {
    final prefs = await _prefs;
    await prefs.setString(_keyBackendPort, port);
  }

  static Future<String> getBackendUrl() async {
    final host = await getBackendHost();
    final port = await getBackendPort();
    if (host.isEmpty) return '';
    return 'ws://$host:$port/stream';
  }

  static Future<String> getLanguage() async {
    final prefs = await _prefs;
    return prefs.getString(_keyLanguage) ?? 'en-IN';
  }

  static Future<void> setLanguage(String lang) async {
    final prefs = await _prefs;
    await prefs.setString(_keyLanguage, lang);
  }

  static Future<String> getSpeaker() async {
    final prefs = await _prefs;
    return prefs.getString(_keySpeaker) ?? 'priya';
  }

  static Future<void> setSpeaker(String speaker) async {
    final prefs = await _prefs;
    await prefs.setString(_keySpeaker, speaker);
  }

  static Future<bool> isOnboardingComplete() async {
    final prefs = await _prefs;
    return prefs.getBool(_keyOnboardingComplete) ?? false;
  }

  static Future<void> setOnboardingComplete() async {
    final prefs = await _prefs;
    await prefs.setBool(_keyOnboardingComplete, true);
  }
}
