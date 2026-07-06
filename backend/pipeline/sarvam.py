import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY not found in .env")
SARVAM_API_KEY = SARVAM_API_KEY.strip()
HEADERS = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}

TARGET_LANGUAGE_MAP = {
    "en-IN": "en-IN",
    "ta-IN": "ta-IN",
    "hi-IN": "hi-IN",
    "en": "en-IN",
    "ta": "ta-IN",
    "hi": "hi-IN",
}

SPEAKER_MAP = {
    "en-IN": {"male": "meera", "female": "priya"},
    "ta-IN": {"male": "meera", "female": "priya"},
    "hi-IN": {"male": "meera", "female": "priya"},
}


def translate(text, target_language):
    if target_language == "en-IN":
        return text
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers=HEADERS,
            json={
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": target_language,
            },
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()["translated_text"]
        else:
            print(f"Translation failed ({r.status_code}): {r.text} — falling back to English")
            return text
    except Exception as e:
        print(f"Translation error: {e} — falling back to English")
        return text


def text_to_speech(text, target_language, speaker):
    try:
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=HEADERS,
            json={
                "inputs": [text],
                "target_language_code": target_language,
                "speaker": speaker,
                "model": "bulbul:v3",
            },
            timeout=10,
        )
        if r.status_code == 200:
            audio_bytes = base64.b64decode(r.json()["audios"][0])
            return audio_bytes
        else:
            print(f"TTS failed ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def synthesize(text_english, target_language, speaker):
    translated = translate(text_english, target_language)
    audio_bytes = text_to_speech(translated, target_language, speaker)
    return translated, audio_bytes
