# Speaker Labeler UI Design

## Overview

A local web app for labeling WhisperX speaker diarization output. Users assign roles (agent/user/narrator) to speakers in cold call transcripts, with synchronized audio playback.

**Stack:** FastAPI backend + React frontend + wavesurfer.js

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend (Python)                               │
│  - Serves calls from S3 (spk_turns.json)               │
│  - Streams audio files from S3                          │
│  - Saves labels to CSV                                  │
│  - Tracks progress (which calls labeled)                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  React Frontend                                         │
│  - Waveform display (wavesurfer.js)                    │
│  - Transcript with speaker turns                        │
│  - Role assignment buttons                              │
│  - Linear queue navigation                              │
└─────────────────────────────────────────────────────────┘
```

**Data flow:**
1. Backend loads call queue from S3 prefix (all `spk_turns.json` files)
2. Frontend requests current call → backend returns transcript + audio URL
3. User labels speakers → frontend sends `{speaker_id: role}` to backend
4. Backend appends to `speaker_labels.csv` and marks call complete
5. Frontend requests next call

## Main Interface

```
┌─────────────────────────────────────────────────────────┐
│  HEADER BAR                                             │
│  ← Prev    Call 12 of 847    [Skip] [Save & Next →]    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SPEAKER ASSIGNMENT (bulk)                              │
│                                                         │
│  SPEAKER_00: [🟢 Agent] [🔵 User] [⚪ Narrator]         │
│  SPEAKER_01: [🟢 Agent] [🔵 User] [⚪ Narrator]         │
│  SPEAKER_02: [🟢 Agent] [🔵 User] [⚪ Narrator]         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  WAVEFORM + TRANSPORT                                   │
│  ┌───────────────────────────────────────────────────┐ │
│  │ ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁        │ │
│  └───────────────────────────────────────────────────┘ │
│  [▶ Play]  0:32 / 2:15   [🔊 Volume]                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  TRANSCRIPT (scrollable)                                │
│                                                         │
│  ▶ SPEAKER_00 [0:00-0:08] 🟢                           │
│    "Hi, is this John? This is Mike calling from..."     │
│                                                         │
│    SPEAKER_01 [0:08-0:10] 🔵                           │
│    "Yeah, who's this?"                                  │
│                                                         │
│    SPEAKER_00 [0:10-0:25] 🟢                           │
│    "I'm calling about your listing on Oak Street..."    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Interactions

**Audio playback (three modes):**
- Click any turn → plays that audio segment, waveform cursor jumps there
- Click play → continuous playback, current turn highlights as it plays
- Drag waveform → seeks audio, corresponding turn highlights

**Role assignment:**
- Bulk buttons at top → assigns role to ALL turns for that speaker
- Turn-level color dot → click to override if diarization was wrong for that turn

**Navigation:**
- Linear queue: Prev / Next buttons
- Skip button for problematic calls
- Progress indicator shows position in queue

## API Endpoints

```
GET  /api/queue              → List of all calls with status
GET  /api/queue/current      → Current call to label (next unlabeled)
GET  /api/calls/{call_id}    → Full call data (turns, audio URL)
POST /api/calls/{call_id}/labels  → Save speaker role assignments
POST /api/calls/{call_id}/skip    → Skip this call (mark for later)
GET  /api/audio/{video_id}/{call_id}  → Stream audio segment
GET  /api/progress           → Stats (labeled, skipped, remaining)
```

**Call Response (GET /api/calls/{call_id}):**
```json
{
  "call_id": "call_001",
  "video_id": "abc123",
  "audio_url": "/api/audio/abc123/call_001",
  "duration_s": 135.2,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "turns": [
    {"spk": "SPEAKER_00", "t0": 0.0, "t1": 8.2, "text": "Hi, is this..."},
    {"spk": "SPEAKER_01", "t0": 8.2, "t1": 10.1, "text": "Yeah, who's this?"}
  ]
}
```

**Label Submission (POST /api/calls/{call_id}/labels):**
```json
{
  "speaker_roles": {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "user",
    "SPEAKER_02": "narrator"
  }
}
```

## Audio Handling

**Approach:** Pre-extract audio segments for each call.

Audio extraction step (one-time, before labeling):
```bash
python -m scripts.whisperx_pipeline.extract_call_audio runs/
# Creates: s3://bucket/runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3
```

Benefits:
- Faster loading (small files)
- Accurate waveforms (wavesurfer sees the full segment)
- Simpler caching

## Project Structure

```
speaker-labeler/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── s3_client.py         # S3 audio/data access
│   ├── labels.py            # CSV read/write
│   └── requirements.txt     # fastapi, boto3, uvicorn
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main app
│   │   ├── components/
│   │   │   ├── Waveform.tsx      # wavesurfer.js wrapper
│   │   │   ├── Transcript.tsx    # Turn list with highlighting
│   │   │   ├── SpeakerPanel.tsx  # Bulk role assignment
│   │   │   └── NavBar.tsx        # Prev/Next/Progress
│   │   └── hooks/
│   │       └── useAudioSync.ts   # Sync waveform ↔ transcript
│   ├── package.json
│   └── vite.config.ts
├── labeler_progress.json    # Tracks labeled/skipped calls
└── speaker_labels.csv       # Output (feeds into training)
```

## Startup

```bash
# Terminal 1: Backend
cd speaker-labeler/backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd speaker-labeler/frontend
npm run dev   # → http://localhost:5173
```

Or single command:
```bash
./speaker-labeler/start.sh   # Launches both, opens browser
```

## Output Format

Labels save to `speaker_labels.csv`:
```csv
video_id,call_id,speaker_id,role,labeled_at
abc123,call_001,SPEAKER_00,agent,2026-01-28T10:30:00
abc123,call_001,SPEAKER_01,user,2026-01-28T10:30:00
abc123,call_001,SPEAKER_02,narrator,2026-01-28T10:30:00
```

This CSV is consumed directly by `train_role_model_v2.py`.
