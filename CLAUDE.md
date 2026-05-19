# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Local audio transcription tool using faster-whisper (CTranslate2-based Whisper). Supports batch file transcription and real-time mic streaming, with configurable output formats and GPU acceleration.

## Usage

```bash
uv run transcribe -i <audio_file> [-o output.txt] [-f plain|annotated|jsonl] [-l lang] [-t] [-v]
uv run transcribe -s [-o output.txt] [-f plain|annotated|jsonl] [-l lang] [-t] [-v]
```

Output defaults to stdout when `-o` is omitted.

Key flags:
- `-i`/`--input`: input audio file (mutually exclusive with `-s`)
- `-s`/`--stream`: stream from microphone (mutually exclusive with `-i`)
- `-o`/`--output`: output file (default: stdout)
- `-f`/`--output-format`: `plain` (default), `annotated`, `jsonl`, or `markdowntable`
- `-l`/`--language`: pin language code (e.g. `en`, `pt`); auto-detect if omitted
- `-t`/`--translate`: translate to English using Whisper's built-in translation task
- `-T`/`--temperature`: sampling temperature
- `-p`/`--prompt`: initial prompt string or `@filepath`
- `-d`/`--device`: `auto` (default), `cuda`, or `cpu`
- `-v`/`-vv`: logging verbosity

## Architecture

`transcribe/cli.py` (single module):
- `large-v3` model; device selectable via `-d` (`auto`/`cuda`/`cpu`), `compute_type="auto"`
- Output formats: `plain` (text only), `annotated` (fixed-width columns: timestamp, lang, confidence, text), `markdowntable` (same columns as a GitHub-flavored markdown table with header), `jsonl` (structured with `timing`, `speech`, `transcription` subobjects)

## Features

### Batch transcription
Transcribes a pre-recorded audio file. Usage: `uv run transcribe -i <audio> [-o output.txt]`. Output defaults to stdout.

### Streaming transcription (mic)
Real-time transcription from microphone using RMS-based VAD. Accumulates audio while speech is detected, flushes to faster-whisper on silence. Prints each segment to stdout; optionally writes to a file with `-o`. Usage: `uv run transcribe -s [-o output.txt]`. Tuning constants at top of `transcribe/cli.py`: `SILENCE_RMS_THRESHOLD`, `SILENCE_CHUNKS_TO_FLUSH`, `MIN_SPEECH_CHUNKS`.

## Pending work

See `TODO.md` for known improvements and future work. When implementing new major features, add a description under `## Features` above and remove the corresponding TODO entry.
