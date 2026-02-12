# Gemini Audio Pilot v1

Standalone timestamp probe for testing Gemini on call-segment extraction from audio.

This pilot is intentionally separate from the WavLM call extractor and does not clip audio.

## What it does

- Accepts local audio files and/or `s3://bucket/key` audio URIs.
- Runs Gemini per input across one or more models.
- Saves:
  - raw model text response,
  - optional parsed candidate call segments (`start_s`, `end_s`) if enabled.

Default mode is raw-output-first (`--no-parse-timestamps`), so you can evaluate Gemini quality before relying on parser logic.

## Environment

Set one of:

- `GEMINI_API_KEY` (preferred), or
- `GOOGLE_API_KEY`.

The runner also loads a local `.env` file (best effort) if shell vars are missing.

## Default models

- `gemini-3-flash-preview`
- `gemini-2.0-flash`

Override with repeatable `--model`.

## Commands

Dry-run (no Gemini calls):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3 \
  --audio-s3-uri s3://rezora-whisperx-us-east-1-864981718771/audio/pretraining/example.mp3 \
  --dry-run
```

Live run (local audio, default two models):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3
```

Live run with optional timestamp parsing enabled:

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3 \
  --parse-timestamps
```

Live run (S3 audio + explicit model set):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-s3-uri s3://rezora-whisperx-us-east-1-864981718771/audio/pretraining/example.mp3 \
  --model gemini-3-flash-preview \
  --model gemini-2.0-flash
```

## Output layout

Default output root:

`artifacts/gemini_pilot/run_<UTC_TS>/`

Key files:

- `run_manifest.json`
- `run_summary.json`
- `outputs/<source_id>/<model>.raw.txt`
- `outputs/<source_id>/<model>.result.json`
- `outputs/<source_id>/<model>.error.json` (only on failures)
