import argparse
import logging
import queue
import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 1600 samples per chunk

SILENCE_RMS_THRESHOLD = 0.01
SILENCE_CHUNKS_TO_FLUSH = 8   # 800ms of silence triggers transcription
MIN_SPEECH_CHUNKS = 4          # ignore bursts shorter than 400ms

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="input audio file (omit with --stream)")
    parser.add_argument("output", nargs="?", help="output markdown file")
    parser.add_argument("-s", "--stream", action="store_true", help="stream from microphone in real time")
    parser.add_argument("-l", "--language", help="language code (e.g. en, pt); auto-detect if omitted")
    parser.add_argument("-p", "--prompt", help="initial prompt text, or @path to read from file")
    parser.add_argument("-t", "--temperature", type=float, help="sampling temperature (default: faster-whisper default)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if not args.stream and not args.audio:
        parser.error("audio is required unless --stream is set")
    if not args.stream and not args.output:
        parser.error("output is required unless --stream is set")

    return args


def resolve_prompt(prompt: str | None) -> str | None:
    if prompt and prompt.startswith("@"):
        with open(prompt[1:], encoding="utf-8") as f:
            return f.read()
    return prompt


def build_transcribe_kwargs(args) -> dict:
    kwargs: dict = {
        "language": args.language,
        "initial_prompt": resolve_prompt(args.prompt),
        "vad_filter": True,
        "beam_size": 5,
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def transcribe_file(model: WhisperModel, audio: str, output: str, kwargs: dict) -> None:
    log.info("transcribing %s", audio)
    segments, info = model.transcribe(audio, **kwargs)
    log.debug("detected language: %s (%.0f%%)", info.language, info.language_probability * 100)
    with open(output, "w", encoding="utf-8") as f:
        for s in segments:
            log.debug("[%.1fs-%.1fs] %s", s.start, s.end, s.text.strip())
            f.write(f"- [{s.start:.1f}s-{s.end:.1f}s] {s.text.strip()}\n")
    log.info("written to %s", output)


def transcribe_stream(model: WhisperModel, output: str | None, kwargs: dict) -> None:
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    session_start = time.monotonic()

    def audio_callback(indata, frames, ts, status):
        if status:
            log.debug("sounddevice: %s", status)
        audio_q.put(indata[:, 0].copy())

    def flush(speech_chunks: list[np.ndarray], t_start: float) -> None:
        audio = np.concatenate(speech_chunks).astype(np.float32)
        kwargs["vad_filter"] = False
        segments, _ = model.transcribe(audio, **kwargs)
        kwargs["vad_filter"] = True
        t_end = time.monotonic() - session_start
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            line = f"- [{t_start:.1f}s-{t_end:.1f}s] {text}"
            print(line, flush=True)
            if output:
                with open(output, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

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

    print("Listening... (Ctrl+C to stop)", file=sys.stderr)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK_SAMPLES, callback=audio_callback):
            while True:
                process_chunk(audio_q.get())

    except KeyboardInterrupt:
        if len(speech_buf) >= MIN_SPEECH_CHUNKS:
            flush(speech_buf, speech_start)
        print("\nStopped.", file=sys.stderr)


def main():
    args = parse_args()

    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    log.info("loading model")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    kwargs = build_transcribe_kwargs(args)

    if args.stream:
        transcribe_stream(model, args.output, kwargs)
    else:
        transcribe_file(model, args.audio, args.output, kwargs)


if __name__ == "__main__":
    main()
