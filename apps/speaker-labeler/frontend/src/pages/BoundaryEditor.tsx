import { useState, useEffect, useRef, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import {
  getVideos,
  getVideoBoundaries,
  saveVideoBoundaries,
  getVideoAudioUrl,
  getVideosProgress,
  getNextUnlabeled,
  exportLabels,
  type VideoInfo,
  type AutoBoundary,
  type VideosProgress,
} from '../api';

interface Boundary {
  id: string;
  start_s: number;
  end_s: number;
  type: 'auto' | 'corrected';
}

// Skip interval configuration for time navigation buttons
const SKIP_INTERVALS = [
  { seconds: -300, label: '-5m' },
  { seconds: -60, label: '-60s' },
  { seconds: -25, label: '-25s' },
  { seconds: -5, label: '-5s' },
  { seconds: -0.25, label: '-0.25s' },
  { seconds: 0.25, label: '+0.25s' },
  { seconds: 5, label: '+5s' },
  { seconds: 25, label: '+25s' },
  { seconds: 60, label: '+60s' },
  { seconds: 300, label: '+5m' },
];

// Playback speed options
const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];

export function BoundaryEditor() {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [autoBoundaries, setAutoBoundaries] = useState<AutoBoundary[]>([]);
  const [correctedBoundaries, setCorrectedBoundaries] = useState<Boundary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [saving, setSaving] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showAuto, setShowAuto] = useState(true);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(50); // pixels per second
  const [progress, setProgress] = useState<VideosProgress | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const initialBoundariesRef = useRef<string>('');

  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null);

  // Load videos list and progress
  useEffect(() => {
    Promise.all([getVideos(), getVideosProgress()])
      .then(([videosData, progressData]) => {
        setVideos(videosData);
        setProgress(progressData);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Refresh progress when videos change
  const refreshProgress = useCallback(() => {
    getVideosProgress().then(setProgress).catch(console.error);
  }, []);

  // Track unsaved changes
  useEffect(() => {
    const current = JSON.stringify(correctedBoundaries.map(b => ({ s: b.start_s, e: b.end_s })));
    setHasUnsavedChanges(current !== initialBoundariesRef.current);
  }, [correctedBoundaries]);

  // Store initial state when boundaries are loaded
  useEffect(() => {
    initialBoundariesRef.current = JSON.stringify(correctedBoundaries.map(b => ({ s: b.start_s, e: b.end_s })));
    setHasUnsavedChanges(false);
  }, [selectedVideoId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Warn on page unload
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasUnsavedChanges]);

  // Load video boundaries when selected
  useEffect(() => {
    if (!selectedVideoId) return;

    setLoadingAudio(true);
    getVideoBoundaries(selectedVideoId)
      .then((boundaries) => {
        setAutoBoundaries(boundaries.auto_boundaries);
        // Convert corrected boundaries to our format
        const corrected = boundaries.corrected_boundaries.map((b, i) => ({
          id: `corrected-${i}`,
          start_s: b.start_s,
          end_s: b.end_s,
          type: 'corrected' as const,
        }));
        setCorrectedBoundaries(corrected.length > 0 ? corrected : []);
        // Use streaming endpoint (synchronous URL)
        setAudioUrl(getVideoAudioUrl(selectedVideoId));
      })
      .catch(console.error)
      .finally(() => setLoadingAudio(false));
  }, [selectedVideoId]);

  // Initialize WaveSurfer when audio URL changes
  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    // Cleanup previous instance
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

    // Create WaveSurfer without URL first (avoids StrictMode abort race)
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#6366F1',
      progressColor: '#818CF8',
      cursorColor: '#1F2937',
      height: 128,
      barWidth: 2,
      barGap: 1,
      plugins: [regions],
    });

    // AbortController to cancel in-flight requests on cleanup
    const abortController = new AbortController();

    // Fetch audio using streaming reader (response.blob() fails with large streaming responses)
    const loadAudio = async () => {
      try {
        const response = await fetch(audioUrl, { signal: abortController.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        if (!response.body) throw new Error('No response body');

        // Read chunks manually - works around browser streaming issues
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (abortController.signal.aborted) {
            reader.cancel();
            return;
          }
          chunks.push(value);
        }

        if (abortController.signal.aborted) return;

        const blob = new Blob(chunks, { type: 'audio/mpeg' });
        await ws.loadBlob(blob);
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') return;
        console.error('Failed to load audio:', err);
      }
    };

    // Start loading immediately (AbortController handles cleanup)
    loadAudio();

    ws.on('ready', () => {
      setDuration(ws.getDuration());
      // Sync playback rate when WaveSurfer is ready (maintains rate when switching videos)
      ws.setPlaybackRate(playbackRate);
    });

    ws.on('error', (err) => {
      console.error('WaveSurfer error:', {
        name: err?.name,
        message: err?.message,
        stack: err?.stack?.split('\n').slice(0, 5).join('\n')
      });
    });

    ws.on('timeupdate', () => {
      setCurrentTime(ws.getCurrentTime());
    });

    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));

    // Handle region clicks for selection
    regions.on('region-clicked', (region, e) => {
      e.stopPropagation();
      // Only select corrected (editable) regions
      if (region.id.startsWith('corrected-')) {
        setSelectedRegionId(region.id);
      }
    });

    // Handle region updates from drag/resize
    regions.on('region-updated', (region) => {
      if (region.id.startsWith('corrected-')) {
        setCorrectedBoundaries((prev) =>
          prev.map((b) =>
            b.id === region.id ? { ...b, start_s: region.start, end_s: region.end } : b
          )
        );
      }
    });

    wavesurferRef.current = ws;

    return () => {
      abortController.abort();
      ws.destroy();
    };
  }, [audioUrl]);

  // Update regions when boundaries change
  useEffect(() => {
    if (!regionsRef.current || !wavesurferRef.current) return;

    // Clear existing regions
    regionsRef.current.clearRegions();

    // Add auto boundaries (red)
    if (showAuto) {
      autoBoundaries.forEach((b, i) => {
        regionsRef.current?.addRegion({
          id: `auto-${i}`,
          start: b.start_s,
          end: b.end_s,
          color: 'rgba(239, 68, 68, 0.2)',
          drag: false,
          resize: false,
        });
      });
    }

    // Add corrected boundaries (green, brighter if selected)
    correctedBoundaries.forEach((b) => {
      const isSelected = b.id === selectedRegionId;
      regionsRef.current?.addRegion({
        id: b.id,
        start: b.start_s,
        end: b.end_s,
        color: isSelected ? 'rgba(34, 197, 94, 0.5)' : 'rgba(34, 197, 94, 0.3)',
        drag: true,
        resize: true,
      });
    });
  }, [autoBoundaries, correctedBoundaries, showAuto, selectedRegionId]);

  // Apply zoom when level changes
  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.zoom(zoomLevel);
    }
  }, [zoomLevel]);

  const togglePlay = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  // Unified skip function - positive for forward, negative for backward
  const skip = useCallback((seconds: number) => {
    if (!wavesurferRef.current) return;
    const current = wavesurferRef.current.getCurrentTime();
    const duration = wavesurferRef.current.getDuration();
    const newTime = Math.max(0, Math.min(current + seconds, duration));
    wavesurferRef.current.setTime(newTime);
  }, []);

  // Keyboard shortcut wrappers (keep 5s for [ ] keys)
  const skipForward = useCallback(() => skip(5), [skip]);
  const skipBackward = useCallback(() => skip(-5), [skip]);

  const handleZoomIn = useCallback(() => {
    setZoomLevel((prev) => Math.min(prev * 1.5, 500));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomLevel((prev) => Math.max(prev / 1.5, 10));
  }, []);

  const handlePlaybackRateChange = useCallback((rate: number) => {
    setPlaybackRate(rate);
    if (wavesurferRef.current) {
      wavesurferRef.current.setPlaybackRate(rate);
    }
  }, []);

  // Nudge selected region edge
  const nudgeSelectedRegion = useCallback((edge: 'start' | 'end', delta: number) => {
    if (!selectedRegionId) return;
    setCorrectedBoundaries((prev) =>
      prev.map((b) => {
        if (b.id !== selectedRegionId) return b;
        if (edge === 'start') {
          const newStart = Math.max(0, b.start_s + delta);
          return { ...b, start_s: Math.min(newStart, b.end_s - 0.1) };
        } else {
          const newEnd = Math.min(duration, b.end_s + delta);
          return { ...b, end_s: Math.max(newEnd, b.start_s + 0.1) };
        }
      })
    );
  }, [selectedRegionId, duration]);

  const deleteSelectedBoundary = useCallback(() => {
    if (!selectedRegionId) return;
    setCorrectedBoundaries((prev) => prev.filter((b) => b.id !== selectedRegionId));
    setSelectedRegionId(null);
  }, [selectedRegionId]);

  const formatTimeMs = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  const addBoundaryAtPlayhead = useCallback(() => {
    if (!wavesurferRef.current) return;
    const time = wavesurferRef.current.getCurrentTime();
    const dur = wavesurferRef.current.getDuration();

    // Use functional update to avoid stale closure issues
    setCorrectedBoundaries((prev) => {
      // Find if we're inside an existing corrected boundary
      const existingIdx = prev.findIndex(
        (b) => time >= b.start_s && time <= b.end_s
      );

      if (existingIdx >= 0) {
        // Split the existing boundary at playhead
        const existing = prev[existingIdx];
        const newBoundaries = [...prev];
        newBoundaries[existingIdx] = { ...existing, end_s: time };
        newBoundaries.splice(existingIdx + 1, 0, {
          id: `corrected-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          start_s: time,
          end_s: existing.end_s,
          type: 'corrected',
        });
        return newBoundaries;
      } else {
        // Add new boundary from playhead to end (or next boundary)
        const nextBoundary = prev.find((b) => b.start_s > time);
        const endTime = nextBoundary ? nextBoundary.start_s : dur;

        const newBoundary: Boundary = {
          id: `corrected-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          start_s: time,
          end_s: endTime,
          type: 'corrected',
        };
        return [...prev, newBoundary].sort((a, b) => a.start_s - b.start_s);
      }
    });
  }, []); // No dependency on correctedBoundaries - functional update reads fresh state

  const addBoundaryEndingAtPlayhead = useCallback(() => {
    if (!wavesurferRef.current) return;
    const time = wavesurferRef.current.getCurrentTime();

    // Use functional update to avoid stale closure issues
    setCorrectedBoundaries((prev) => {
      // Find if we're inside an existing corrected boundary
      const existingIdx = prev.findIndex(
        (b) => time >= b.start_s && time <= b.end_s
      );

      if (existingIdx >= 0) {
        // Split the existing boundary at playhead (same as B)
        const existing = prev[existingIdx];
        const newBoundaries = [...prev];
        newBoundaries[existingIdx] = { ...existing, end_s: time };
        newBoundaries.splice(existingIdx + 1, 0, {
          id: `corrected-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          start_s: time,
          end_s: existing.end_s,
          type: 'corrected',
        });
        return newBoundaries;
      } else {
        // Add new boundary from previous boundary end (or 0) to playhead
        const prevBoundary = [...prev]
          .filter((b) => b.end_s <= time)
          .sort((a, b) => b.end_s - a.end_s)[0];
        const startTime = prevBoundary ? prevBoundary.end_s : 0;

        const newBoundary: Boundary = {
          id: `corrected-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          start_s: startTime,
          end_s: time,
          type: 'corrected',
        };
        return [...prev, newBoundary].sort((a, b) => a.start_s - b.start_s);
      }
    });
  }, []); // No dependency - functional update reads fresh state

  const clearAllCorrected = useCallback(() => {
    setCorrectedBoundaries([]);
  }, []);

  const initFromAuto = useCallback(() => {
    // Initialize corrected boundaries from auto-detected ones
    const corrected = autoBoundaries.map((b, i) => ({
      id: `corrected-${i}`,
      start_s: b.start_s,
      end_s: b.end_s,
      type: 'corrected' as const,
    }));
    setCorrectedBoundaries(corrected);
  }, [autoBoundaries]);

  const deleteBoundary = useCallback((id: string) => {
    setCorrectedBoundaries((prev) => prev.filter((b) => b.id !== id));
  }, []);

  // Check if corrected boundaries match auto boundaries (no real edits made)
  const boundariesMatchAuto = useCallback(() => {
    if (correctedBoundaries.length !== autoBoundaries.length) return false;

    const sortedCorrected = [...correctedBoundaries].sort((a, b) => a.start_s - b.start_s);
    const sortedAuto = [...autoBoundaries].sort((a, b) => a.start_s - b.start_s);

    // Check if all boundaries match within 0.1s tolerance
    for (let i = 0; i < sortedCorrected.length; i++) {
      const diff_start = Math.abs(sortedCorrected[i].start_s - sortedAuto[i].start_s);
      const diff_end = Math.abs(sortedCorrected[i].end_s - sortedAuto[i].end_s);
      if (diff_start > 0.1 || diff_end > 0.1) {
        return false;
      }
    }
    return true;
  }, [correctedBoundaries, autoBoundaries]);

  const handleSave = useCallback(async () => {
    if (!selectedVideoId) return;

    // Warn if boundaries match auto-detected (user may have forgotten to edit)
    if (boundariesMatchAuto()) {
      const confirmed = confirm(
        'These boundaries match the auto-detected ones.\n\n' +
        'Are you sure you want to save without any manual corrections?\n\n' +
        '(This is typically a mistake - you may have clicked "Copy from Auto" without editing)'
      );
      if (!confirmed) return;
    }

    setSaving(true);
    try {
      const boundariesData = correctedBoundaries.map((b) => ({
        start_s: b.start_s,
        end_s: b.end_s,
      }));
      await saveVideoBoundaries(selectedVideoId, boundariesData);
      // Update video list to show correction status
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === selectedVideoId ? { ...v, has_corrections: true } : v
        )
      );
      alert('Boundaries saved!');
      initialBoundariesRef.current = JSON.stringify(boundariesData.map(b => ({ s: b.start_s, e: b.end_s })));
      setHasUnsavedChanges(false);
      refreshProgress();
    } catch (error: unknown) {
      console.error('Save failed:', error);
      // Check if it's a validation error
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as { response?: { data?: { detail?: { errors?: string[] } } } };
        const errors = axiosError.response?.data?.detail?.errors;
        if (errors && Array.isArray(errors)) {
          alert('Validation errors:\n' + errors.join('\n'));
          return;
        }
      }
      alert('Failed to save boundaries');
    } finally {
      setSaving(false);
    }
  }, [selectedVideoId, correctedBoundaries, refreshProgress]);

  // Find gaps (unlabeled regions) in the video
  const findGaps = useCallback(() => {
    if (duration === 0) return [];

    interface Gap {
      start_s: number;
      end_s: number;
      duration_s: number;
    }

    if (correctedBoundaries.length === 0) {
      return [{ start_s: 0, end_s: duration, duration_s: duration }];
    }

    const sorted = [...correctedBoundaries].sort((a, b) => a.start_s - b.start_s);
    const gaps: Gap[] = [];

    // Gap before first boundary
    if (sorted[0].start_s > 5) {
      gaps.push({ start_s: 0, end_s: sorted[0].start_s, duration_s: sorted[0].start_s });
    }

    // Gaps between boundaries
    for (let i = 0; i < sorted.length - 1; i++) {
      const gapStart = sorted[i].end_s;
      const gapEnd = sorted[i + 1].start_s;
      const gapDuration = gapEnd - gapStart;
      if (gapDuration > 5) {
        gaps.push({ start_s: gapStart, end_s: gapEnd, duration_s: gapDuration });
      }
    }

    // Gap after last boundary
    const last = sorted[sorted.length - 1];
    if (duration - last.end_s > 5) {
      gaps.push({ start_s: last.end_s, end_s: duration, duration_s: duration - last.end_s });
    }

    return gaps;
  }, [correctedBoundaries, duration]);

  const jumpToNextGap = useCallback(() => {
    const gaps = findGaps();
    if (gaps.length === 0) {
      alert('No unlabeled gaps found!');
      return;
    }

    const currentPos = wavesurferRef.current?.getCurrentTime() ?? 0;

    // Find next gap after current position
    const nextGap = gaps.find(g => g.start_s > currentPos + 1);
    if (nextGap) {
      wavesurferRef.current?.setTime(nextGap.start_s);
    } else {
      // Wrap around to first gap
      wavesurferRef.current?.setTime(gaps[0].start_s);
    }
  }, [findGaps]);

  const goToNextUnlabeled = useCallback(async () => {
    if (hasUnsavedChanges) {
      if (!confirm('You have unsaved changes. Discard and switch videos?')) {
        return;
      }
    }
    try {
      const result = await getNextUnlabeled();
      if (result.found && result.video_id) {
        setSelectedVideoId(result.video_id);
      } else {
        alert('All videos have been labeled!');
      }
    } catch (error) {
      console.error('Failed to get next unlabeled:', error);
    }
  }, [hasUnsavedChanges]);

  const handleExportLabels = useCallback(async () => {
    try {
      const labels = await exportLabels();
      const blob = new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'labels.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export labels');
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'b':
        case 'B':
          e.preventDefault();
          if (e.shiftKey) {
            addBoundaryEndingAtPlayhead();
          } else {
            addBoundaryAtPlayhead();
          }
          break;
        case 'd':
        case 'D':
        case 'Delete':
        case 'Backspace':
          if (selectedRegionId) {
            e.preventDefault();
            deleteSelectedBoundary();
          }
          break;
        case '[':
          e.preventDefault();
          skipBackward();
          break;
        case ']':
          e.preventDefault();
          skipForward();
          break;
        case '+':
        case '=':
          e.preventDefault();
          handleZoomIn();
          break;
        case '-':
        case '_':
          e.preventDefault();
          handleZoomOut();
          break;
        case 'ArrowLeft':
          if (selectedRegionId) {
            e.preventDefault();
            const delta = e.shiftKey ? -1.0 : -0.1;
            nudgeSelectedRegion(e.altKey ? 'end' : 'start', delta);
          }
          break;
        case 'ArrowRight':
          if (selectedRegionId) {
            e.preventDefault();
            const delta = e.shiftKey ? 1.0 : 0.1;
            nudgeSelectedRegion(e.altKey ? 'end' : 'start', delta);
          }
          break;
        case 'Escape':
          setSelectedRegionId(null);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay, addBoundaryAtPlayhead, addBoundaryEndingAtPlayhead, deleteSelectedBoundary, skipBackward, skipForward, handleZoomIn, handleZoomOut, nudgeSelectedRegion, selectedRegionId]);

  // Navigate to next/prev video
  const currentVideoIndex = videos.findIndex((v) => v.video_id === selectedVideoId);

  const handleVideoSelect = useCallback((videoId: string) => {
    if (hasUnsavedChanges) {
      if (!confirm('You have unsaved changes. Discard and switch videos?')) {
        return;
      }
    }
    setSelectedVideoId(videoId);
  }, [hasUnsavedChanges]);

  const goToNextVideo = () => {
    if (currentVideoIndex < videos.length - 1) {
      handleVideoSelect(videos[currentVideoIndex + 1].video_id);
    }
  };
  const goToPrevVideo = () => {
    if (currentVideoIndex > 0) {
      handleVideoSelect(videos[currentVideoIndex - 1].video_id);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading videos...</div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex">
      {/* Video list sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 overflow-y-auto flex flex-col">
        {/* Progress section */}
        <div className="p-4 border-b border-gray-200 bg-green-50">
          <h2 className="font-semibold text-gray-800">Progress</h2>
          {progress && (
            <div className="mt-2 text-sm">
              <div className="flex justify-between">
                <span>Labeled:</span>
                <span className="font-semibold">{progress.labeled_videos} / {progress.total_videos}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all"
                  style={{ width: `${progress.completion_percentage}%` }}
                />
              </div>
              <div className="text-center mt-1 text-green-700 font-medium">
                {progress.completion_percentage}% complete
              </div>
            </div>
          )}
          <div className="mt-3 space-y-2">
            <button
              onClick={goToNextUnlabeled}
              className="w-full px-3 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
            >
              Next Unlabeled Video
            </button>
            <button
              onClick={handleExportLabels}
              className="w-full px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
            >
              Export labels.json
            </button>
          </div>
        </div>
        <div className="p-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-800">Videos ({videos.length})</h2>
        </div>
        <div className="divide-y divide-gray-100">
          {videos.map((video) => (
            <button
              key={video.video_id}
              onClick={() => handleVideoSelect(video.video_id)}
              className={`w-full text-left p-3 hover:bg-gray-50 ${
                selectedVideoId === video.video_id ? 'bg-indigo-50 border-l-4 border-indigo-500' : ''
              }`}
            >
              <div className="font-mono text-sm truncate">{video.video_id}</div>
              <div className="text-xs text-gray-500 mt-1">
                {video.call_count} auto calls
                {video.has_corrections && (
                  <span className="ml-2 text-green-600">Corrected</span>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main editor area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {!selectedVideoId ? (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            Select a video to edit boundaries
          </div>
        ) : loadingAudio ? (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            Loading audio...
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={goToPrevVideo}
                  disabled={currentVideoIndex <= 0}
                  className="px-3 py-1 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                >
                  ← Prev
                </button>
                <span className="font-semibold">
                  Video {currentVideoIndex + 1} of {videos.length}
                </span>
                <button
                  onClick={goToNextVideo}
                  disabled={currentVideoIndex >= videos.length - 1}
                  className="px-3 py-1 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                >
                  Next →
                </button>
              </div>
              <div className="text-sm text-gray-500">
                <span className="font-mono">{selectedVideoId}</span>
              </div>
            </div>

            {/* Labeling guidance */}
            <details className="bg-blue-50 border-b border-blue-100">
              <summary className="px-4 py-2 cursor-pointer text-blue-800 font-medium hover:bg-blue-100">
                Labeling Instructions (click to expand)
              </summary>
              <div className="px-4 py-3 text-sm text-blue-700 space-y-2">
                <p><strong>IN_CALL starts at:</strong> First ring, dial tone, "calling..." UI sound, or first remote voice</p>
                <p><strong>IN_CALL ends at:</strong> Hangup sound, last remote voice, or call ends and host returns to narration</p>
                <p className="pt-2 border-t border-blue-200">
                  <strong>Tip:</strong> If host says "Alright, next call..." but no actual call audio yet, it's still OUT_OF_CALL until dialing/ringing begins.
                </p>
              </div>
            </details>

            {/* Waveform */}
            <div className="p-4 bg-white border-b border-gray-200">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  Auto: <span className="text-red-500">{autoBoundaries.length} boundaries</span>
                  {' | '}
                  Corrected: <span className="text-green-600">{correctedBoundaries.length} boundaries</span>
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={showAuto}
                    onChange={(e) => setShowAuto(e.target.checked)}
                  />
                  Show auto boundaries
                </label>
              </div>
              <div ref={containerRef} className="bg-gray-100 rounded-lg overflow-x-auto" />
              {/* Playback controls - compact row */}
              <div className="mt-4 flex items-center gap-1.5 flex-wrap">
                {/* Left skip buttons (backward) */}
                {SKIP_INTERVALS.slice(0, 5).map(({ seconds, label }) => (
                  <button
                    key={label}
                    onClick={() => skip(seconds)}
                    className="px-2 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
                    title={`Skip ${label}`}
                  >
                    {label}
                  </button>
                ))}

                {/* Play button - prominent */}
                <button
                  onClick={togglePlay}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium mx-1"
                  title="Play/Pause (Space)"
                >
                  {isPlaying ? '⏸' : '▶'}
                </button>

                {/* Right skip buttons (forward) */}
                {SKIP_INTERVALS.slice(5).map(({ seconds, label }) => (
                  <button
                    key={label}
                    onClick={() => skip(seconds)}
                    className="px-2 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
                    title={`Skip ${label}`}
                  >
                    {label}
                  </button>
                ))}

                {/* Time display */}
                <span className="font-mono text-sm text-gray-800 bg-gray-100 px-2 py-1 rounded ml-2">
                  {formatTimeMs(currentTime)} / {formatTimeMs(duration)}
                </span>

                {/* Spacer */}
                <div className="flex-1" />

                {/* Playback speed control */}
                <div className="flex items-center gap-1">
                  <span className="text-sm text-gray-600">Speed:</span>
                  <select
                    value={playbackRate}
                    onChange={(e) => handlePlaybackRateChange(Number(e.target.value))}
                    className="px-2 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm cursor-pointer border-0"
                  >
                    {PLAYBACK_RATES.map((rate) => (
                      <option key={rate} value={rate}>
                        {rate}x
                      </option>
                    ))}
                  </select>
                </div>

                {/* Zoom controls */}
                <div className="flex items-center gap-1 ml-2">
                  <button
                    onClick={handleZoomOut}
                    className="px-2 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
                    title="Zoom out (-)"
                  >
                    -
                  </button>
                  <span className="text-xs text-gray-600 w-14 text-center">
                    {Math.round(zoomLevel)}px/s
                  </span>
                  <button
                    onClick={handleZoomIn}
                    className="px-2 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm"
                    title="Zoom in (+)"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="p-4 bg-white border-b border-gray-200 flex flex-wrap gap-2">
              <button
                onClick={addBoundaryAtPlayhead}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                title="Add boundary starting at playhead (B)"
              >
                + Start at Playhead
              </button>
              <button
                onClick={addBoundaryEndingAtPlayhead}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                title="Add boundary ending at playhead (Shift+B)"
              >
                + End at Playhead
              </button>
              <button
                onClick={initFromAuto}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Copy from Auto
              </button>
              <button
                onClick={clearAllCorrected}
                className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200"
              >
                Clear All Corrected
              </button>
              <button
                onClick={jumpToNextGap}
                className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600"
                title="Jump to next unlabeled gap (>5s)"
              >
                Jump to Gap
              </button>
              <div className="flex-1" />
              <button
                onClick={handleSave}
                disabled={saving}
                className={`px-6 py-2 text-white rounded-lg disabled:opacity-50 ${
                  hasUnsavedChanges
                    ? 'bg-orange-500 hover:bg-orange-600'
                    : 'bg-indigo-600 hover:bg-indigo-700'
                }`}
              >
                {saving ? 'Saving...' : hasUnsavedChanges ? 'Save Changes *' : 'Save Corrected Boundaries'}
              </button>
            </div>

            {/* Keyboard hints */}
            <div className="px-4 py-2 bg-blue-50 border-b border-blue-100 text-xs text-blue-700">
              <span className="font-medium">Shortcuts:</span>{' '}
              <kbd className="px-1 bg-blue-100 rounded">Space</kbd> Play/Pause{' '}
              <kbd className="px-1 bg-blue-100 rounded">B</kbd> Start boundary{' '}
              <kbd className="px-1 bg-blue-100 rounded">Shift+B</kbd> End boundary{' '}
              <kbd className="px-1 bg-blue-100 rounded">[ ]</kbd> Skip 5s{' '}
              <kbd className="px-1 bg-blue-100 rounded">+ -</kbd> Zoom{' '}
              {selectedRegionId && (
                <>
                  <span className="ml-2 text-green-700">|</span>{' '}
                  <kbd className="px-1 bg-green-100 rounded">←→</kbd> Nudge start{' '}
                  <kbd className="px-1 bg-green-100 rounded">Alt+←→</kbd> Nudge end{' '}
                  <kbd className="px-1 bg-green-100 rounded">Shift</kbd> ±1s{' '}
                  <kbd className="px-1 bg-green-100 rounded">D</kbd> Delete{' '}
                  <kbd className="px-1 bg-green-100 rounded">Esc</kbd> Deselect
                </>
              )}
            </div>

            {/* Gap indicators */}
            {findGaps().length > 0 && (
              <div className="px-4 py-2 bg-yellow-50 border-b border-yellow-100">
                <div className="text-sm font-medium text-yellow-800 mb-2">
                  Unlabeled Gaps ({findGaps().length})
                </div>
                <div className="flex flex-wrap gap-2">
                  {findGaps().map((gap, i) => (
                    <button
                      key={`gap-${i}`}
                      onClick={() => wavesurferRef.current?.setTime(gap.start_s)}
                      className="px-2 py-1 text-xs bg-yellow-200 text-yellow-800 rounded hover:bg-yellow-300"
                    >
                      {formatTimeMs(gap.start_s)} ({Math.round(gap.duration_s)}s)
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Boundary list */}
            <div className="flex-1 p-4 overflow-y-auto">
              <h3 className="font-semibold text-gray-800 mb-3">
                Corrected Boundaries ({correctedBoundaries.length})
                {selectedRegionId && (
                  <span className="ml-2 text-sm font-normal text-green-600">
                    (selected: {correctedBoundaries.findIndex(b => b.id === selectedRegionId) + 1})
                  </span>
                )}
              </h3>
              {correctedBoundaries.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  No corrected boundaries yet. Add boundaries using the button above
                  or copy from auto-detected.
                </div>
              ) : (
                <div className="space-y-2">
                  {correctedBoundaries.map((boundary, index) => (
                    <div
                      key={boundary.id}
                      onClick={() => setSelectedRegionId(boundary.id)}
                      className={`rounded-lg p-3 border flex items-center gap-4 cursor-pointer transition-colors ${
                        selectedRegionId === boundary.id
                          ? 'bg-green-50 border-green-400 ring-2 ring-green-200'
                          : 'bg-white border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      <span className="font-semibold text-gray-700">Call {index + 1}</span>
                      <span className="font-mono text-sm text-gray-600">
                        {formatTimeMs(boundary.start_s)} - {formatTimeMs(boundary.end_s)}
                      </span>
                      <span className="text-sm text-gray-500">
                        ({(boundary.end_s - boundary.start_s).toFixed(1)}s)
                      </span>
                      <div className="flex-1" />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          wavesurferRef.current?.setTime(boundary.start_s);
                          wavesurferRef.current?.play();
                        }}
                        className="px-3 py-1 text-indigo-600 hover:bg-indigo-50 rounded"
                      >
                        ▶ Preview
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteBoundary(boundary.id);
                        }}
                        className="px-3 py-1 text-red-600 hover:bg-red-50 rounded"
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
