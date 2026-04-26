import glob
import os
import sys

# Necessary as we need to ensure the CUDA libraries are in LD_LIBRARY_PATH before loading PyTorch, which faster-whisper does internally.
# This is especially an issue in virtual environments where the CUDA libs may not be in the default system path.
def adjust_and_relaunch():
    if os.environ.get("_TRANSCRIBE_REEXEC"):
        return

    _cuda_libs = glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cublas/lib"))
    if not _cuda_libs:
        return

    _lib = _cuda_libs[0]
    os.environ["LD_LIBRARY_PATH"] = _lib + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    os.environ["_TRANSCRIBE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

adjust_and_relaunch()

import argparse
import json
import logging
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from huggingface_hub import try_to_load_from_cache

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 1600 samples per chunk

SILENCE_RMS_THRESHOLD = 0.01
SILENCE_CHUNKS_TO_FLUSH = 8   # 800ms of silence triggers transcription
MIN_SPEECH_CHUNKS = 4          # ignore bursts shorter than 400ms

OUTPUT_FORMATS = ("plain", "annotated", "jsonl")

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", metavar="FILE", help="input audio file")
    source.add_argument("-s", "--stream", action="store_true", help="stream from microphone in real time")
    parser.add_argument("-o", "--output", metavar="FILE", help="output file (default: stdout)")
    parser.add_argument("-f", "--output-format", choices=OUTPUT_FORMATS, default="plain", help="output format (default: plain)")
    parser.add_argument("-l", "--language", help="language code (e.g. en, pt); auto-detect if omitted")
    parser.add_argument("-p", "--prompt", help="initial prompt text, or @path to read from file")
    parser.add_argument("-t", "--translate", action="store_true", help="translate audio to English (uses Whisper's built-in translation)")
    parser.add_argument("-T", "--temperature", type=float, help="sampling temperature (default: faster-whisper default)")
    parser.add_argument("-d", "--device", choices=("auto", "cpu", "cuda"), default="auto", help="compute device (default: auto)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def resolve_prompt(prompt: str | None) -> str | None:
    if prompt and prompt.startswith("@"):
        with open(prompt[1:], encoding="utf-8") as f:
            return f.read()
    return prompt


def build_transcribe_kwargs(args) -> dict:
    kwargs: dict = {
        "language": args.language,
        "task": "translate" if args.translate else "transcribe",
        "initial_prompt": resolve_prompt(args.prompt),
        "vad_filter": True,
        "beam_size": 5,
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def format_segment(s: Segment, fmt: str, language: str | None = None, language_probability: float | None = None, session_offset: float = 0.0, chunk_duration: float | None = None) -> str:
    text = s.text.strip()
    if fmt == "plain":
        return text
    start = session_offset + s.start
    end = session_offset + s.end
    if fmt == "annotated":
        ts = f"[{start:.1f}s-{end:.1f}s]"
        lang = f"[{language}:{language_probability:.0%}]" if language else ""
        conf = f"[conf:{s.avg_logprob:.2f}]"
        return f"{lang:<8} {conf:<11} {ts:<13} {text}"
    # jsonl
    obj: dict = {
        "timing": {
            "start": round(start, 2),
            "end": round(end, 2),
            "chunk_duration": round(chunk_duration, 2) if chunk_duration is not None else None,
        },
        "speech": {
            "language": language,
            "language_probability": round(language_probability, 2) if language_probability is not None else None,
            "no_speech_prob": round(s.no_speech_prob, 2),
        },
        "transcription": {
            "confidence": round(s.avg_logprob, 2),
            "text": text,
        },
    }
    return json.dumps(obj, ensure_ascii=False)


def load_model(device: str) -> WhisperModel:
    model_id = "large-v3"
    log.info("loading model: %s on %s", model_id, device)
    cached = try_to_load_from_cache("Systran/faster-whisper-large-v3", "model.bin")
    if cached is None:
        log.warning("Downloading faster-whisper large-v3 model from Hugging Face (~3GB, first run only)...")
    return WhisperModel(model_id, device=device, compute_type="auto")


def transcribe_file(model: WhisperModel, audio: str, output: str | None, fmt: str, kwargs: dict) -> None:
    log.info("transcribing %s", audio)
    segments, info = model.transcribe(audio, **kwargs)
    log.debug("detected language: %s (%.0f%%)", info.language, info.language_probability * 100)
    f = open(output, "w", encoding="utf-8") if output else sys.stdout
    try:
        for s in segments:
            log.debug("[%.1fs-%.1fs] %s", s.start, s.end, s.text.strip())
            f.write(format_segment(s, fmt, language=info.language, language_probability=info.language_probability, chunk_duration=s.end - s.start) + "\n")
    finally:
        if output:
            f.close()
            log.info("written to %s", output)


def transcribe_stream(model: WhisperModel, output: str | None, fmt: str, kwargs: dict) -> None:
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    session_start = time.monotonic()
    out_file = open(output, "w", encoding="utf-8") if output else sys.stdout

    def audio_callback(indata, frames, ts, status):
        if status:
            log.debug("sounddevice: %s", status)
        audio_q.put(indata[:, 0].copy())

    def flush(speech_chunks: list[np.ndarray], t_start: float) -> None:
        audio = np.concatenate(speech_chunks).astype(np.float32)
        chunk_duration = len(audio) / SAMPLE_RATE
        kwargs["vad_filter"] = False
        segments, info = model.transcribe(audio, **kwargs)
        kwargs["vad_filter"] = True
        for s in segments:
            if not s.text.strip():
                continue
            line = format_segment(s, fmt, language=info.language, language_probability=info.language_probability, session_offset=t_start, chunk_duration=chunk_duration)
            print(line, file=out_file, flush=True)

    speech_buf: list[np.ndarray] = []
    silence_count = 0
    speech_start = 0.0

    def process_chunk(chunk: np.ndarray) -> None:
        nonlocal speech_buf, silence_count, speech_start
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms >= SILENCE_RMS_THRESHOLD:
            if not speech_buf:
                speech_start = time.monotonic() - session_start
            speech_buf.append(chunk)
            silence_count = 0
        elif speech_buf:
            speech_buf.append(chunk)
            silence_count += 1
            if silence_count >= SILENCE_CHUNKS_TO_FLUSH:
                if len(speech_buf) >= MIN_SPEECH_CHUNKS + SILENCE_CHUNKS_TO_FLUSH:
                    flush(speech_buf, speech_start)
                speech_buf = []
                silence_count = 0

    if kwargs['task'] == "translate":
        print("Translating... Press Ctrl+C to stop", file=sys.stderr)
    elif kwargs['task'] == "transcribe":
        print("Transcribing... Press Ctrl+C to stop", file=sys.stderr)
    else:
        raise ValueError(f"unexpected task: {kwargs['task']}")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK_SAMPLES, callback=audio_callback):
            while True:
                process_chunk(audio_q.get())

    except KeyboardInterrupt:
        if len(speech_buf) >= MIN_SPEECH_CHUNKS:
            flush(speech_buf, speech_start)
    finally:
        if output:
            out_file.close()
            log.info("written to %s", output)


def main():
    args = parse_args()

    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    if args.stream:
        print("Initializing...", file=sys.stderr)

    model = load_model(args.device)

    kwargs = build_transcribe_kwargs(args)

    if args.stream:
        transcribe_stream(model, args.output, args.output_format, kwargs)
    else:
        transcribe_file(model, args.input, args.output, args.output_format, kwargs)
