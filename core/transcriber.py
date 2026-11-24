import os
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

_model = None


def load_model() -> WhisperModel:
    global _model
    if _model is None:
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        compute_type = "float16" if use_cuda else "int8"

        print(f"Loading faster-whisper model: {WHISPER_MODEL} ({device}, {compute_type}) ...")
        _model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        print("faster-whisper model loaded.")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False, language: str = None) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"

    segments, _ = model.transcribe(
        chunk_path,
        task=task,
        language=language if not translate else None,
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def transcribe_all(chunks: list, translate: bool = False, language: str = None) -> str:
    transcript_segments = []

    # Pre-load model once before loop
    load_model()

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, translate=translate, language=language)
        if text:
            transcript_segments.append(text.strip())

    print("Transcription completed.")
    return " ".join(transcript_segments)