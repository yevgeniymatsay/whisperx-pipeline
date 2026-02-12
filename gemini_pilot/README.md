# Gemini Audio Pilot v1

Standalone timestamp probe for testing Gemini on call-segment extraction from audio.

This pilot is intentionally separate from the WavLM call extractor and does not clip audio.

## What it does

- Accepts local audio files and/or `s3://bucket/key` audio URIs.
- Runs Gemini per input across one or more models.
- Saves:
  - raw model text response (JSON text),
  - minimal request/response metadata (no parsing in v1).

This pilot uses the Gemini Files API (`files.upload`) and then references the uploaded file in `generate_content`.

## Environment

Set one of:

- `GEMINI_API_KEY` (preferred), or
- `GOOGLE_API_KEY`.

The runner also loads a local `.env` file (best effort) if shell vars are missing.

Recommended: put Gemini-only keys in `gemini_pilot/.env` (gitignored) to keep it separate from other pipeline env.
You can start from `gemini_pilot/.env.example`.

## Default models

- `gemini-2.5-flash`
- `gemini-2.0-flash`
- `gemini-2.5-pro`
- `gemini-3-pro-preview`
- `gemini-3-flash-preview`

Override with repeatable `--model`.

## Request config (frozen for comparisons)

- Structured output via `response_schema` + `response_mime_type=application/json`
- `temperature=0`

## Commands

Dry-run (no Gemini calls):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3 \
  --audio-s3-uri s3://rezora-whisperx-us-east-1-864981718771/audio/pretraining/example.mp3 \
  --dry-run
```

Live run (local audio, default models):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3
```

Live run (local audio, run the 5 default models in parallel, print full JSON):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-path /absolute/path/to/sample.mp3 \
  --model-concurrency 5 \
  --print-max-chars 0
```

Live run (S3 audio + explicit model set):

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/gemini_pilot_extract_call_timestamps.py \
  --audio-s3-uri s3://rezora-whisperx-us-east-1-864981718771/audio/pretraining/example.mp3 \
  --model gemini-3-flash-preview \
  --model gemini-2.0-flash
```

If you want to limit terminal printing, use:

`--print-max-chars 6000`

## Local-dir sweep + eval (10 files × 5 models)

If you have exactly 10 `.m4a` files already downloaded locally:

```bash
python /Users/yevgeniymatsay/whisperx-pipeline/scripts/run_gemini_local_dir_sweep_and_eval.py \
  --audio-dir "/Volumes/Yevgeniy's Drive/tmp_gemini_audio"
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
