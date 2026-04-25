# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Local audio transcription tool using faster-whisper (CTranslate2-based Whisper). Transcribes audio to a timestamped markdown bullet list using GPU acceleration.

## Usage

```bash
./run.sh -i <audio_file> [-o output.md] [-v] [-vv]
./run.sh --stream [-o output.md]
```

`run.sh` sets `LD_LIBRARY_PATH` to the local CUDA libs in `.venv` before invoking `uv run main.py`.
Output defaults to stdout when `-o` is omitted.

## Architecture

Single-file script (`main.py`):
- `large-v3` model on CUDA/float16, VAD filtering enabled
- Args: positional `audio`, `output`; `-v` = INFO, `-vv` = DEBUG
- Output format: `- [<start>s-<end>s] <text>`

## Features

### Batch transcription
Transcribes a pre-recorded audio file. Usage: `./run.sh -i <audio> [-o output.md]`. Output defaults to stdout.

### Streaming transcription (mic)
Real-time transcription from microphone using RMS-based VAD. Accumulates audio while speech is detected, flushes to faster-whisper on silence. Prints each segment to stdout; optionally writes to a file with `-o`. Usage: `uv run main.py --stream [-o output.md]`. Tuning constants at top of `main.py`: `SILENCE_RMS_THRESHOLD`, `SILENCE_CHUNKS_TO_FLUSH`, `MIN_SPEECH_CHUNKS`.

## Pending work

See `TODO.md` for known improvements and future work. When implementing new major features, add a description under `## Features` above and remove the corresponding TODO entry.
