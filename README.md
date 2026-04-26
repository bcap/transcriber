# transcriber

Local audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3). Runs entirely on your machine — no cloud, no API keys.

## Requirements

- [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU with CUDA support (optional but recommended; falls back to CPU)

## Running

The first run will download the Whisper large-v3 model (~3GB) from Hugging Face. It will work without an account, but creating one and setting `HF_TOKEN` enables faster downloads:

```bash
export HF_TOKEN=your_token_here
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### One-shot via uvx (no install, no clone)

```bash
uvx --from git+https://github.com/bcap/transcriber transcribe -i audio.mp3
```

### Persistent install via uv tool

```bash
uv tool install git+https://github.com/bcap/transcriber
transcribe -i audio.mp3
```

### From a local clone

```bash
git clone git@github.com:bcap/transcriber.git
cd transcriber
uv run transcribe -i audio.mp3
```

## Usage

### Transcribe a file

```bash
transcribe -i audio.mp3
transcribe -i audio.mp3 -o output.txt
```

### Stream from microphone

```bash
transcribe -s
transcribe -s -o session.txt
```

### Output formats

```bash
transcribe -i audio.mp3 -f plain       # plain text (default)
transcribe -i audio.mp3 -f annotated   # timestamped, with language and confidence
transcribe -i audio.mp3 -f jsonl       # one JSON object per segment
```

**plain**
```
okay this is a simple transcriber test
```

**annotated**
```
[en:99%]  [conf:-0.18]   [0.8s-7.7s]    okay this is a simple transcriber test
```

**jsonl**
```json
{"timing": {"start": 0.8, "end": 7.7, "chunk_duration": 6.9}, "speech": {"language": "en", "language_probability": 0.99, "no_speech_prob": 0.01}, "transcription": {"confidence": -0.18, "text": "okay this is a simple transcriber test"}}
```

## All flags

| Flag | Description |
|------|-------------|
| `-i`, `--input FILE` | Input audio file (mutually exclusive with `-s`) |
| `-s`, `--stream` | Stream from microphone (mutually exclusive with `-i`) |
| `-o`, `--output FILE` | Output file (default: stdout) |
| `-f`, `--output-format` | `plain` (default), `annotated`, or `jsonl` |
| `-l`, `--language` | Pin language code, e.g. `en`, `pt` (default: auto-detect) |
| `-t`, `--translate` | Translate to English using Whisper's built-in translation |
| `-T`, `--temperature` | Sampling temperature |
| `-p`, `--prompt` | Initial prompt text, or `@filepath` to read from file |
| `-d`, `--device` | `auto` (default), `cuda`, or `cpu` |
| `-v` / `-vv` | Increase logging verbosity |
