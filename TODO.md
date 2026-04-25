# TODO

## Streaming improvements

- **Persistent language detection**: currently language is detected per-chunk independently. Better approach: detect language on the first chunk (or first few seconds), then lock it in for the session to avoid per-chunk re-detection overhead and inconsistency.

- **True real-time transcription**: move away from silence-triggered chunking toward a continuous transcription loop that emits partial results and corrects/refines them as more audio arrives. Reference: whisper-streaming (github.com/ufal/whisper_streaming) uses a local-agreement algorithm to emit stable tokens while buffering unstable trailing context.
