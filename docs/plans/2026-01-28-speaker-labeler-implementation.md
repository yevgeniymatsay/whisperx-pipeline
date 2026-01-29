# Speaker Labeler UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local web app for labeling WhisperX speaker diarization output with synchronized audio playback.

**Architecture:** FastAPI backend serves call data and audio from S3, saves labels to CSV. React frontend with wavesurfer.js for waveform display and audio-transcript synchronization. Linear queue workflow for labeling calls one-by-one.

**Tech Stack:** FastAPI, boto3, uvicorn (backend); React, TypeScript, Vite, wavesurfer.js, Tailwind CSS (frontend)

---

## Task 1: Backend Project Setup

**Files:**
- Create: `speaker-labeler/backend/main.py`
- Create: `speaker-labeler/backend/requirements.txt`

**Step 1: Create backend directory and requirements**

```bash
mkdir -p speaker-labeler/backend
```

**Step 2: Create requirements.txt**

Create `speaker-labeler/backend/requirements.txt`:
```
fastapi==0.109.0
uvicorn==0.27.0
boto3==1.34.0
python-multipart==0.0.6
```

**Step 3: Create minimal FastAPI app**

Create `speaker-labeler/backend/main.py`:
```python
"""Speaker Labeler API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Speaker Labeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
```

**Step 4: Verify backend starts**

Run:
```bash
cd speaker-labeler/backend && pip install -r requirements.txt && uvicorn main:app --port 8000
```

Open http://localhost:8000/api/health - should return `{"status": "ok"}`

**Step 5: Commit**

```bash
git add speaker-labeler/backend/
git commit -m "feat(speaker-labeler): add backend scaffolding"
```

---

## Task 2: Backend S3 Client

**Files:**
- Create: `speaker-labeler/backend/s3_client.py`
- Create: `speaker-labeler/backend/config.py`

**Step 1: Create config**

Create `speaker-labeler/backend/config.py`:
```python
"""Configuration for speaker labeler."""
import os

S3_BUCKET = os.environ.get("S3_BUCKET", "rezora-whisperx-us-east-1-864981718771")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", "runs/")
```

**Step 2: Create S3 client**

Create `speaker-labeler/backend/s3_client.py`:
```python
"""S3 client for loading calls and audio."""
import json
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from config import S3_BUCKET, AWS_REGION, S3_PREFIX


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def list_calls(prefix: str = S3_PREFIX) -> List[Dict[str, Any]]:
    """List all calls under the given prefix."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    calls = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                # Parse path: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
                parts = key.split("/")
                if len(parts) >= 5:
                    calls.append({
                        "s3_key": key,
                        "video_id": parts[1],
                        "run_id": parts[2],
                        "call_id": parts[4],
                    })
    return calls


def get_call_data(s3_key: str) -> Optional[Dict[str, Any]]:
    """Load call data from S3."""
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        data = json.loads(response["Body"].read())
        return data
    except ClientError:
        return None


def get_audio_url(video_id: str, call_id: str, run_id: str) -> Optional[str]:
    """Generate presigned URL for call audio."""
    s3 = get_s3_client()
    # Audio key: runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3
    audio_key = f"runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3"

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": audio_key},
            ExpiresIn=3600,
        )
        return url
    except ClientError:
        return None
```

**Step 3: Commit**

```bash
git add speaker-labeler/backend/config.py speaker-labeler/backend/s3_client.py
git commit -m "feat(speaker-labeler): add S3 client for calls and audio"
```

---

## Task 3: Backend Labels Handler

**Files:**
- Create: `speaker-labeler/backend/labels.py`

**Step 1: Create labels handler**

Create `speaker-labeler/backend/labels.py`:
```python
"""Labels CSV handling."""
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, List, Any

LABELS_FILE = Path("speaker_labels.csv")
PROGRESS_FILE = Path("labeler_progress.json")


def load_progress() -> Dict[str, Any]:
    """Load progress tracking."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"labeled": [], "skipped": [], "current_index": 0}


def save_progress(progress: Dict[str, Any]) -> None:
    """Save progress tracking."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_labeled_call_ids() -> Set[str]:
    """Get set of already labeled call IDs."""
    progress = load_progress()
    return set(progress.get("labeled", []))


def get_skipped_call_ids() -> Set[str]:
    """Get set of skipped call IDs."""
    progress = load_progress()
    return set(progress.get("skipped", []))


def save_labels(
    video_id: str,
    call_id: str,
    speaker_roles: Dict[str, str],
) -> None:
    """Save labels for a call and update progress."""
    # Append to CSV
    file_exists = LABELS_FILE.exists()
    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "speaker_id", "role", "labeled_at"])

        timestamp = datetime.now().isoformat()
        for speaker_id, role in speaker_roles.items():
            writer.writerow([video_id, call_id, speaker_id, role, timestamp])

    # Update progress
    progress = load_progress()
    if call_id not in progress["labeled"]:
        progress["labeled"].append(call_id)
    progress["current_index"] = progress.get("current_index", 0) + 1
    save_progress(progress)


def skip_call(call_id: str) -> None:
    """Mark a call as skipped."""
    progress = load_progress()
    if call_id not in progress["skipped"]:
        progress["skipped"].append(call_id)
    progress["current_index"] = progress.get("current_index", 0) + 1
    save_progress(progress)


def get_current_index() -> int:
    """Get current position in queue."""
    progress = load_progress()
    return progress.get("current_index", 0)


def set_current_index(index: int) -> None:
    """Set current position in queue."""
    progress = load_progress()
    progress["current_index"] = index
    save_progress(progress)
```

**Step 2: Commit**

```bash
git add speaker-labeler/backend/labels.py
git commit -m "feat(speaker-labeler): add labels CSV handler"
```

---

## Task 4: Backend API Routes

**Files:**
- Modify: `speaker-labeler/backend/main.py`

**Step 1: Add API routes**

Replace `speaker-labeler/backend/main.py` with:
```python
"""Speaker Labeler API."""
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import s3_client
import labels

app = FastAPI(title="Speaker Labeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache calls list
_calls_cache: List[Dict[str, Any]] = []


def get_calls() -> List[Dict[str, Any]]:
    """Get cached calls list."""
    global _calls_cache
    if not _calls_cache:
        _calls_cache = s3_client.list_calls()
    return _calls_cache


class LabelRequest(BaseModel):
    speaker_roles: Dict[str, str]


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/progress")
def get_progress():
    """Get labeling progress."""
    calls = get_calls()
    labeled = labels.get_labeled_call_ids()
    skipped = labels.get_skipped_call_ids()
    current_index = labels.get_current_index()

    return {
        "total": len(calls),
        "labeled": len(labeled),
        "skipped": len(skipped),
        "remaining": len(calls) - len(labeled) - len(skipped),
        "current_index": current_index,
    }


@app.get("/api/queue")
def get_queue():
    """Get all calls with their status."""
    calls = get_calls()
    labeled = labels.get_labeled_call_ids()
    skipped = labels.get_skipped_call_ids()

    result = []
    for call in calls:
        call_id = call["call_id"]
        status = "labeled" if call_id in labeled else ("skipped" if call_id in skipped else "pending")
        result.append({**call, "status": status})

    return result


@app.get("/api/queue/current")
def get_current_call():
    """Get current call to label."""
    calls = get_calls()
    current_index = labels.get_current_index()

    if current_index >= len(calls):
        return {"done": True, "message": "All calls labeled"}

    call = calls[current_index]
    return {
        "done": False,
        "index": current_index,
        "total": len(calls),
        **call,
    }


@app.get("/api/calls/{call_id}")
def get_call(call_id: str):
    """Get full call data."""
    calls = get_calls()

    # Find call in list
    call_info = None
    for c in calls:
        if c["call_id"] == call_id:
            call_info = c
            break

    if not call_info:
        raise HTTPException(status_code=404, detail="Call not found")

    # Load call data from S3
    data = s3_client.get_call_data(call_info["s3_key"])
    if not data:
        raise HTTPException(status_code=404, detail="Call data not found in S3")

    # Get audio URL
    audio_url = s3_client.get_audio_url(
        call_info["video_id"],
        call_info["call_id"],
        call_info["run_id"],
    )

    # Extract unique speakers
    turns = data.get("turns", [])
    speakers = list(dict.fromkeys(t.get("spk") for t in turns if t.get("spk")))

    # Calculate duration
    duration_s = 0
    if turns:
        duration_s = max(t.get("t1_abs", t.get("t1", 0)) for t in turns)

    return {
        "call_id": call_id,
        "video_id": call_info["video_id"],
        "audio_url": audio_url,
        "duration_s": duration_s,
        "speakers": speakers,
        "turns": turns,
    }


@app.post("/api/calls/{call_id}/labels")
def save_call_labels(call_id: str, request: LabelRequest):
    """Save labels for a call."""
    calls = get_calls()

    # Find call to get video_id
    call_info = None
    for c in calls:
        if c["call_id"] == call_id:
            call_info = c
            break

    if not call_info:
        raise HTTPException(status_code=404, detail="Call not found")

    labels.save_labels(
        call_info["video_id"],
        call_id,
        request.speaker_roles,
    )

    return {"success": True}


@app.post("/api/calls/{call_id}/skip")
def skip_call(call_id: str):
    """Skip a call."""
    labels.skip_call(call_id)
    return {"success": True}


@app.post("/api/navigate/{direction}")
def navigate(direction: str):
    """Navigate prev/next."""
    calls = get_calls()
    current = labels.get_current_index()

    if direction == "next":
        new_index = min(current + 1, len(calls) - 1)
    elif direction == "prev":
        new_index = max(current - 1, 0)
    else:
        raise HTTPException(status_code=400, detail="Invalid direction")

    labels.set_current_index(new_index)
    return {"index": new_index}
```

**Step 2: Verify API works**

Run:
```bash
cd speaker-labeler/backend && uvicorn main:app --reload --port 8000
```

Test endpoints:
- http://localhost:8000/api/health → `{"status": "ok"}`
- http://localhost:8000/api/progress → shows progress stats

**Step 3: Commit**

```bash
git add speaker-labeler/backend/main.py
git commit -m "feat(speaker-labeler): add API routes for queue and labels"
```

---

## Task 5: Frontend Project Setup

**Files:**
- Create: `speaker-labeler/frontend/` (Vite project)

**Step 1: Create Vite React project**

```bash
cd speaker-labeler && npm create vite@latest frontend -- --template react-ts
cd frontend && npm install
```

**Step 2: Install dependencies**

```bash
cd speaker-labeler/frontend && npm install wavesurfer.js @tanstack/react-query axios tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Step 3: Configure Tailwind**

Update `speaker-labeler/frontend/tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Step 4: Add Tailwind to CSS**

Replace `speaker-labeler/frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 5: Create API client**

Create `speaker-labeler/frontend/src/api.ts`:
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export interface Turn {
  spk: string;
  t0_abs?: number;
  t1_abs?: number;
  t0?: number;
  t1?: number;
  text: string;
}

export interface CallData {
  call_id: string;
  video_id: string;
  audio_url: string | null;
  duration_s: number;
  speakers: string[];
  turns: Turn[];
}

export interface QueueItem {
  call_id: string;
  video_id: string;
  run_id: string;
  s3_key: string;
  status: 'pending' | 'labeled' | 'skipped';
}

export interface Progress {
  total: number;
  labeled: number;
  skipped: number;
  remaining: number;
  current_index: number;
}

export interface CurrentCall {
  done: boolean;
  message?: string;
  index?: number;
  total?: number;
  call_id?: string;
  video_id?: string;
}

export const getProgress = () => api.get<Progress>('/api/progress').then(r => r.data);
export const getCurrentCall = () => api.get<CurrentCall>('/api/queue/current').then(r => r.data);
export const getCallData = (callId: string) => api.get<CallData>(`/api/calls/${callId}`).then(r => r.data);
export const saveLabels = (callId: string, speakerRoles: Record<string, string>) =>
  api.post(`/api/calls/${callId}/labels`, { speaker_roles: speakerRoles });
export const skipCall = (callId: string) => api.post(`/api/calls/${callId}/skip`);
export const navigate = (direction: 'prev' | 'next') => api.post(`/api/navigate/${direction}`);

export default api;
```

**Step 6: Verify frontend starts**

```bash
cd speaker-labeler/frontend && npm run dev
```

Open http://localhost:5173 - should show Vite default page.

**Step 7: Commit**

```bash
git add speaker-labeler/frontend/
git commit -m "feat(speaker-labeler): add frontend scaffolding with Vite + React"
```

---

## Task 6: Frontend Waveform Component

**Files:**
- Create: `speaker-labeler/frontend/src/components/Waveform.tsx`

**Step 1: Create Waveform component**

Create `speaker-labeler/frontend/src/components/Waveform.tsx`:
```typescript
import { useEffect, useRef, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';

interface WaveformProps {
  audioUrl: string | null;
  onTimeUpdate?: (time: number) => void;
  onReady?: (duration: number) => void;
  seekTo?: number;
}

export function Waveform({ audioUrl, onTimeUpdate, onReady, seekTo }: WaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const isPlayingRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4F46E5',
      progressColor: '#818CF8',
      cursorColor: '#1F2937',
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    });

    ws.load(audioUrl);

    ws.on('ready', () => {
      onReady?.(ws.getDuration());
    });

    ws.on('audioprocess', () => {
      onTimeUpdate?.(ws.getCurrentTime());
    });

    ws.on('seeking', () => {
      onTimeUpdate?.(ws.getCurrentTime());
    });

    ws.on('play', () => {
      isPlayingRef.current = true;
    });

    ws.on('pause', () => {
      isPlayingRef.current = false;
    });

    wavesurferRef.current = ws;

    return () => {
      ws.destroy();
    };
  }, [audioUrl, onTimeUpdate, onReady]);

  useEffect(() => {
    if (wavesurferRef.current && seekTo !== undefined) {
      const duration = wavesurferRef.current.getDuration();
      if (duration > 0) {
        wavesurferRef.current.seekTo(seekTo / duration);
      }
    }
  }, [seekTo]);

  const togglePlay = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-100 rounded-lg p-4">
      <div ref={containerRef} className="mb-4" />
      <div className="flex items-center gap-4">
        <button
          onClick={togglePlay}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
        >
          {isPlayingRef.current ? '⏸ Pause' : '▶ Play'}
        </button>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add speaker-labeler/frontend/src/components/Waveform.tsx
git commit -m "feat(speaker-labeler): add Waveform component with wavesurfer.js"
```

---

## Task 7: Frontend Transcript Component

**Files:**
- Create: `speaker-labeler/frontend/src/components/Transcript.tsx`

**Step 1: Create Transcript component**

Create `speaker-labeler/frontend/src/components/Transcript.tsx`:
```typescript
import { useEffect, useRef } from 'react';
import { Turn } from '../api';

interface TranscriptProps {
  turns: Turn[];
  currentTime: number;
  speakerRoles: Record<string, string>;
  onTurnClick: (turn: Turn) => void;
  onTurnRoleChange?: (turnIndex: number, role: string) => void;
}

const ROLE_COLORS: Record<string, string> = {
  agent: 'bg-green-100 border-green-400',
  user: 'bg-blue-100 border-blue-400',
  narrator: 'bg-gray-100 border-gray-400',
  '': 'bg-white border-gray-200',
};

const ROLE_DOTS: Record<string, string> = {
  agent: '🟢',
  user: '🔵',
  narrator: '⚪',
  '': '⚫',
};

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function Transcript({ turns, currentTime, speakerRoles, onTurnClick }: TranscriptProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  // Find current turn based on time
  const currentTurnIndex = turns.findIndex((turn) => {
    const t0 = turn.t0_abs ?? turn.t0 ?? 0;
    const t1 = turn.t1_abs ?? turn.t1 ?? 0;
    return currentTime >= t0 && currentTime < t1;
  });

  // Auto-scroll to active turn
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentTurnIndex]);

  return (
    <div ref={containerRef} className="h-96 overflow-y-auto space-y-2 p-2">
      {turns.map((turn, index) => {
        const t0 = turn.t0_abs ?? turn.t0 ?? 0;
        const t1 = turn.t1_abs ?? turn.t1 ?? 0;
        const role = speakerRoles[turn.spk] || '';
        const isActive = index === currentTurnIndex;
        const colorClass = ROLE_COLORS[role];
        const dot = ROLE_DOTS[role];

        return (
          <div
            key={index}
            ref={isActive ? activeRef : null}
            onClick={() => onTurnClick(turn)}
            className={`p-3 rounded-lg border-2 cursor-pointer transition ${colorClass} ${
              isActive ? 'ring-2 ring-indigo-500' : ''
            } hover:shadow-md`}
          >
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
              <span className="font-mono">{turn.spk}</span>
              <span className="text-xs">[{formatTime(t0)} - {formatTime(t1)}]</span>
              <span>{dot}</span>
            </div>
            <p className="text-gray-900">{turn.text}</p>
          </div>
        );
      })}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add speaker-labeler/frontend/src/components/Transcript.tsx
git commit -m "feat(speaker-labeler): add Transcript component with highlighting"
```

---

## Task 8: Frontend Speaker Panel Component

**Files:**
- Create: `speaker-labeler/frontend/src/components/SpeakerPanel.tsx`

**Step 1: Create SpeakerPanel component**

Create `speaker-labeler/frontend/src/components/SpeakerPanel.tsx`:
```typescript
interface SpeakerPanelProps {
  speakers: string[];
  speakerRoles: Record<string, string>;
  onRoleChange: (speaker: string, role: string) => void;
}

const ROLES = [
  { id: 'agent', label: 'Agent', color: 'bg-green-500 hover:bg-green-600' },
  { id: 'user', label: 'User', color: 'bg-blue-500 hover:bg-blue-600' },
  { id: 'narrator', label: 'Narrator', color: 'bg-gray-500 hover:bg-gray-600' },
];

export function SpeakerPanel({ speakers, speakerRoles, onRoleChange }: SpeakerPanelProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Assign Speaker Roles</h3>
      <div className="space-y-3">
        {speakers.map((speaker) => {
          const currentRole = speakerRoles[speaker] || '';
          return (
            <div key={speaker} className="flex items-center gap-3">
              <span className="font-mono text-sm w-28 text-gray-600">{speaker}:</span>
              <div className="flex gap-2">
                {ROLES.map((role) => {
                  const isSelected = currentRole === role.id;
                  return (
                    <button
                      key={role.id}
                      onClick={() => onRoleChange(speaker, role.id)}
                      className={`px-3 py-1 rounded text-white text-sm transition ${role.color} ${
                        isSelected ? 'ring-2 ring-offset-2 ring-gray-900' : 'opacity-60'
                      }`}
                    >
                      {role.id === 'agent' && '🟢'}
                      {role.id === 'user' && '🔵'}
                      {role.id === 'narrator' && '⚪'}
                      {role.label}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add speaker-labeler/frontend/src/components/SpeakerPanel.tsx
git commit -m "feat(speaker-labeler): add SpeakerPanel for bulk role assignment"
```

---

## Task 9: Frontend NavBar Component

**Files:**
- Create: `speaker-labeler/frontend/src/components/NavBar.tsx`

**Step 1: Create NavBar component**

Create `speaker-labeler/frontend/src/components/NavBar.tsx`:
```typescript
interface NavBarProps {
  currentIndex: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  onSkip: () => void;
  onSave: () => void;
  canSave: boolean;
}

export function NavBar({ currentIndex, total, onPrev, onNext, onSkip, onSave, canSave }: NavBarProps) {
  return (
    <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
      <button
        onClick={onPrev}
        disabled={currentIndex === 0}
        className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
      >
        ← Prev
      </button>

      <div className="text-center">
        <span className="text-lg font-semibold">
          Call {currentIndex + 1} of {total}
        </span>
      </div>

      <div className="flex gap-2">
        <button
          onClick={onSkip}
          className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg border border-gray-300"
        >
          Skip
        </button>
        <button
          onClick={onSave}
          disabled={!canSave}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Save & Next →
        </button>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add speaker-labeler/frontend/src/components/NavBar.tsx
git commit -m "feat(speaker-labeler): add NavBar for queue navigation"
```

---

## Task 10: Frontend Main App

**Files:**
- Modify: `speaker-labeler/frontend/src/App.tsx`

**Step 1: Create main App**

Replace `speaker-labeler/frontend/src/App.tsx`:
```typescript
import { useState, useEffect, useCallback } from 'react';
import { Waveform } from './components/Waveform';
import { Transcript } from './components/Transcript';
import { SpeakerPanel } from './components/SpeakerPanel';
import { NavBar } from './components/NavBar';
import { getCurrentCall, getCallData, saveLabels, skipCall, navigate, Turn, CallData } from './api';

function App() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [total, setTotal] = useState(0);
  const [callId, setCallId] = useState<string | null>(null);
  const [callData, setCallData] = useState<CallData | null>(null);
  const [speakerRoles, setSpeakerRoles] = useState<Record<string, string>>({});
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [done, setDone] = useState(false);

  const loadCurrentCall = useCallback(async () => {
    setLoading(true);
    try {
      const current = await getCurrentCall();
      if (current.done) {
        setDone(true);
        setLoading(false);
        return;
      }

      setCurrentIndex(current.index ?? 0);
      setTotal(current.total ?? 0);
      setCallId(current.call_id ?? null);

      if (current.call_id) {
        const data = await getCallData(current.call_id);
        setCallData(data);
        setSpeakerRoles({});
      }
    } catch (error) {
      console.error('Failed to load call:', error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadCurrentCall();
  }, [loadCurrentCall]);

  const handleRoleChange = (speaker: string, role: string) => {
    setSpeakerRoles((prev) => ({ ...prev, [speaker]: role }));
  };

  const handleTurnClick = (turn: Turn) => {
    const t0 = turn.t0_abs ?? turn.t0 ?? 0;
    setSeekTo(t0);
  };

  const handleSave = async () => {
    if (!callId) return;
    await saveLabels(callId, speakerRoles);
    await navigate('next');
    loadCurrentCall();
  };

  const handleSkip = async () => {
    if (!callId) return;
    await skipCall(callId);
    await navigate('next');
    loadCurrentCall();
  };

  const handlePrev = async () => {
    await navigate('prev');
    loadCurrentCall();
  };

  const handleNext = async () => {
    await navigate('next');
    loadCurrentCall();
  };

  const canSave = callData?.speakers.every((s) => speakerRoles[s]) ?? false;

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    );
  }

  if (done) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🎉</div>
          <div className="text-2xl font-semibold text-gray-800">All calls labeled!</div>
          <div className="text-gray-600 mt-2">Great work. Labels saved to speaker_labels.csv</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <NavBar
        currentIndex={currentIndex}
        total={total}
        onPrev={handlePrev}
        onNext={handleNext}
        onSkip={handleSkip}
        onSave={handleSave}
        canSave={canSave}
      />

      <div className="flex-1 p-4 space-y-4 max-w-5xl mx-auto w-full">
        {callData && (
          <>
            <div className="text-sm text-gray-500">
              Call ID: {callData.call_id} | Video: {callData.video_id}
            </div>

            <SpeakerPanel
              speakers={callData.speakers}
              speakerRoles={speakerRoles}
              onRoleChange={handleRoleChange}
            />

            <Waveform
              audioUrl={callData.audio_url}
              onTimeUpdate={setCurrentTime}
              seekTo={seekTo}
            />

            <Transcript
              turns={callData.turns}
              currentTime={currentTime}
              speakerRoles={speakerRoles}
              onTurnClick={handleTurnClick}
            />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
```

**Step 2: Clean up default files**

```bash
rm speaker-labeler/frontend/src/App.css
```

**Step 3: Verify app works**

Start backend:
```bash
cd speaker-labeler/backend && uvicorn main:app --reload --port 8000
```

Start frontend:
```bash
cd speaker-labeler/frontend && npm run dev
```

Open http://localhost:5173 - should show the labeler UI (will show "Loading..." then error if no S3 data).

**Step 4: Commit**

```bash
git add speaker-labeler/frontend/src/App.tsx
git rm speaker-labeler/frontend/src/App.css 2>/dev/null || true
git commit -m "feat(speaker-labeler): add main App with full labeling flow"
```

---

## Task 11: Startup Script

**Files:**
- Create: `speaker-labeler/start.sh`

**Step 1: Create startup script**

Create `speaker-labeler/start.sh`:
```bash
#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Starting Speaker Labeler..."

# Start backend
echo "Starting backend on port 8000..."
cd backend
pip install -q -r requirements.txt
uvicorn main:app --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 2

# Start frontend
echo "Starting frontend on port 5173..."
cd frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!
cd ..

# Open browser
sleep 3
if command -v open &> /dev/null; then
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
fi

echo ""
echo "Speaker Labeler running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop..."

# Handle shutdown
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
```

**Step 2: Make executable**

```bash
chmod +x speaker-labeler/start.sh
```

**Step 3: Commit**

```bash
git add speaker-labeler/start.sh
git commit -m "feat(speaker-labeler): add startup script"
```

---

## Task 12: Audio Extraction Script

**Files:**
- Create: `scripts/whisperx_pipeline/extract_call_audio.py`

**Step 1: Create audio extraction script**

Create `scripts/whisperx_pipeline/extract_call_audio.py`:
```python
"""Extract audio segments for each call from full video audio."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import boto3

from .config import S3_BUCKET, AWS_REGION


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
) -> bool:
    """Extract audio segment using ffmpeg."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_s),
        "-t", str(duration),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio segments for calls from S3"
    )
    parser.add_argument(
        "s3_prefix",
        help="S3 prefix to scan (e.g., 'runs/' or 'runs/VIDEO_ID/')"
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without doing it"
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    # Find all spk_turns.json files
    calls_to_process = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                parts = key.split("/")
                if len(parts) >= 5:
                    calls_to_process.append({
                        "spk_turns_key": key,
                        "video_id": parts[1],
                        "run_id": parts[2],
                        "call_id": parts[4],
                    })

    print(f"Found {len(calls_to_process)} calls to process")

    if args.dry_run:
        for call in calls_to_process[:5]:
            print(f"  Would extract: {call['video_id']}/{call['call_id']}")
        if len(calls_to_process) > 5:
            print(f"  ... and {len(calls_to_process) - 5} more")
        return

    # Group by video to avoid re-downloading audio
    by_video = {}
    for call in calls_to_process:
        vid = call["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(call)

    extracted = 0
    errors = 0

    for video_id, video_calls in by_video.items():
        print(f"\nProcessing video: {video_id} ({len(video_calls)} calls)")

        # Download full audio
        audio_key = f"audio/{video_id}.mp3"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_audio = Path(tmpdir) / "full.mp3"

            try:
                s3.download_file(args.bucket, audio_key, str(local_audio))
            except Exception as e:
                print(f"  ERROR: Could not download {audio_key}: {e}")
                errors += len(video_calls)
                continue

            for call in video_calls:
                # Load spk_turns.json to get timestamps
                try:
                    response = s3.get_object(Bucket=args.bucket, Key=call["spk_turns_key"])
                    data = json.loads(response["Body"].read())
                except Exception as e:
                    print(f"  ERROR loading {call['call_id']}: {e}")
                    errors += 1
                    continue

                turns = data.get("turns", [])
                if not turns:
                    print(f"  SKIP {call['call_id']}: no turns")
                    continue

                # Calculate start/end from turns
                start_s = min(t.get("t0_abs", t.get("t0", 0)) for t in turns)
                end_s = max(t.get("t1_abs", t.get("t1", 0)) for t in turns)

                # Add small padding
                start_s = max(0, start_s - 0.5)
                end_s = end_s + 0.5

                # Extract segment
                local_segment = Path(tmpdir) / f"{call['call_id']}.mp3"
                if not extract_audio_segment(str(local_audio), str(local_segment), start_s, end_s):
                    print(f"  ERROR extracting {call['call_id']}")
                    errors += 1
                    continue

                # Upload to S3
                output_key = f"runs/{video_id}/{call['run_id']}/calls/{call['call_id']}/audio.mp3"
                try:
                    s3.upload_file(str(local_segment), args.bucket, output_key)
                    print(f"  Extracted: {call['call_id']} ({end_s - start_s:.1f}s)")
                    extracted += 1
                except Exception as e:
                    print(f"  ERROR uploading {call['call_id']}: {e}")
                    errors += 1

    print(f"\nDone! Extracted: {extracted}, Errors: {errors}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/whisperx_pipeline/extract_call_audio.py
git commit -m "feat(whisperx): add audio extraction script for labeler"
```

---

## Task 13: Final Integration Test

**Files:**
- None (manual testing)

**Step 1: Extract audio for test calls**

```bash
python -m scripts.whisperx_pipeline.extract_call_audio runs/ --dry-run
```

If looks good:
```bash
python -m scripts.whisperx_pipeline.extract_call_audio runs/
```

**Step 2: Start the labeler**

```bash
./speaker-labeler/start.sh
```

**Step 3: Verify full flow**

1. Browser opens to http://localhost:5173
2. See first call with speakers listed
3. Click speaker role buttons (Agent/User/Narrator)
4. Click a turn → audio plays that segment
5. Click Play → continuous playback with highlighting
6. Drag waveform → seeks audio
7. Click "Save & Next" → labels saved, next call loads
8. Check `speaker_labels.csv` has the labels

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(speaker-labeler): complete labeling UI implementation"
```

---

## Summary

The speaker labeler is now complete with:
- FastAPI backend serving calls from S3
- React frontend with wavesurfer.js
- Bulk speaker role assignment
- Click-to-play turns
- Continuous playback with highlighting
- Waveform scrubbing
- Linear queue navigation
- CSV output for training pipeline
