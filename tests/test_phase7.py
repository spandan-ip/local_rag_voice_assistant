import os
import tempfile

import pytest

from app.voice.speech_to_text import transcribe_audio, SpeechToTextError
from app.voice.text_to_speech import synthesize_speech, TextToSpeechError


def test_transcribe_missing_file_raises_clear_error():
    with pytest.raises(SpeechToTextError, match="not found"):
        transcribe_audio("audio/does_not_exist.wav")


def test_transcribe_empty_file_raises_clear_error():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(SpeechToTextError, match="empty"):
            transcribe_audio(path)
    finally:
        os.remove(path)


def test_synthesize_empty_text_raises_clear_error():
    with pytest.raises(TextToSpeechError, match="empty"):
        synthesize_speech("")

    with pytest.raises(TextToSpeechError, match="empty"):
        synthesize_speech("   ")
