import os
import logging

from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)

_model_cache: WhisperModel | None = None


class SpeechToTextError(Exception):
    pass


def _get_model() -> WhisperModel:
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        except Exception as exc:
            raise SpeechToTextError(f"Failed to load Whisper model '{WHISPER_MODEL}': {exc}") from exc
    return _model_cache


def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise SpeechToTextError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        raise SpeechToTextError(f"Audio file is empty: {audio_path}")

    model = _get_model()

    try:
        segments, _info = model.transcribe(audio_path)
        text = " ".join(segment.text.strip() for segment in segments)
    except Exception as exc:
        raise SpeechToTextError(f"Speech recognition failed: {exc}") from exc

    if not text.strip():
        raise SpeechToTextError("No speech detected in audio.")

    logger.info("Transcribed audio (%d chars)", len(text))
    return text.strip()
