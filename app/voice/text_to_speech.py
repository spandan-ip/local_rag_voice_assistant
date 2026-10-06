import os
import logging

import pyttsx3

from app.config import TTS_RATE, TTS_OUTPUT_DIR

logger = logging.getLogger(__name__)


class TextToSpeechError(Exception):
    pass


def synthesize_speech(text: str, output_filename: str = "answer.wav") -> str:
    """Convert text to a spoken .wav file and return its path.

    Kept as a standalone function with one clear job (text in, audio file
    path out) so the underlying engine (currently pyttsx3) can be swapped
    for something else later (e.g. Coqui TTS, Piper) without touching
    any code that calls this function.
    """
    if not text or not text.strip():
        raise TextToSpeechError("Cannot synthesize empty text.")

    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except Exception as exc:
        raise TextToSpeechError(f"Text-to-speech failed: {exc}") from exc

    if not os.path.exists(output_path):
        raise TextToSpeechError("TTS did not produce an audio file.")

    logger.info("Synthesized speech to '%s'", output_path)
    return output_path
