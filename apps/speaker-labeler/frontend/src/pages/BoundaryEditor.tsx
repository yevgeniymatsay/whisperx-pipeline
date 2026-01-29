import { useState, useEffect, useRef, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import {
  getVideos,
  getVideoBoundaries,
  saveVideoBoundaries,
  getVideoAudioUrl,
  type VideoInfo,
  type AutoBoundary,
} from '../api';

interface Boundary {
  id: string;
  start_s: number;
  end_s: number;
  type: 'auto' | 'corrected';
}

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
  const [showAuto, setShowAuto] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null);

  // Load videos list
  useEffect(() => {
    getVideos()
      .then(setVideos)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

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

    // Create audio element first - this handles streaming better than fetch
    const audio = new Audio(audioUrl);

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#6366F1',
      progressColor: '#818CF8',
      cursorColor: '#1F2937',
      height: 128,
      barWidth: 2,
      barGap: 1,
      plugins: [regions],
      media: audio,
    });

    // Handle audio metadata loaded
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration);
    });

    ws.on('ready', () => {
      setDuration(ws.getDuration());
    });

    ws.on('timeupdate', () => {
      setCurrentTime(ws.getCurrentTime());
    });

    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));

    wavesurferRef.current = ws;

    return () => {
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

    // Add corrected boundaries (green)
    correctedBoundaries.forEach((b) => {
      regionsRef.current?.addRegion({
        id: b.id,
        start: b.start_s,
        end: b.end_s,
        color: 'rgba(34, 197, 94, 0.3)',
        drag: true,
        resize: true,
      });
    });
  }, [autoBoundaries, correctedBoundaries, showAuto]);

  const togglePlay = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const addBoundaryAtPlayhead = useCallback(() => {
    if (!wavesurferRef.current) return;
    const time = wavesurferRef.current.getCurrentTime();
    const dur = wavesurferRef.current.getDuration();

    // Find if we're inside an existing corrected boundary
    const existingIdx = correctedBoundaries.findIndex(
      (b) => time >= b.start_s && time <= b.end_s
    );

    if (existingIdx >= 0) {
      // Split the existing boundary at playhead
      const existing = correctedBoundaries[existingIdx];
      const newBoundaries = [...correctedBoundaries];
      newBoundaries[existingIdx] = { ...existing, end_s: time };
      newBoundaries.splice(existingIdx + 1, 0, {
        id: `corrected-${Date.now()}`,
        start_s: time,
        end_s: existing.end_s,
        type: 'corrected',
      });
      setCorrectedBoundaries(newBoundaries);
    } else {
      // Add new boundary from playhead to end (or next boundary)
      const nextBoundary = correctedBoundaries.find((b) => b.start_s > time);
      const endTime = nextBoundary ? nextBoundary.start_s : dur;

      const newBoundary: Boundary = {
        id: `corrected-${Date.now()}`,
        start_s: time,
        end_s: endTime,
        type: 'corrected',
      };
      setCorrectedBoundaries((prev) => [...prev, newBoundary].sort((a, b) => a.start_s - b.start_s));
    }
  }, [correctedBoundaries]);

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

  const handleSave = useCallback(async () => {
    if (!selectedVideoId) return;
    setSaving(true);
    try {
      const boundaries = correctedBoundaries.map((b) => ({
        start_s: b.start_s,
        end_s: b.end_s,
      }));
      await saveVideoBoundaries(selectedVideoId, boundaries);
      // Update video list to show correction status
      setVideos((prev) =>
        prev.map((v) =>
          v.video_id === selectedVideoId ? { ...v, has_corrections: true } : v
        )
      );
      alert('Boundaries saved!');
    } catch (error) {
      console.error('Save failed:', error);
      alert('Failed to save boundaries');
    } finally {
      setSaving(false);
    }
  }, [selectedVideoId, correctedBoundaries]);

  // Navigate to next/prev video
  const currentVideoIndex = videos.findIndex((v) => v.video_id === selectedVideoId);
  const goToNextVideo = () => {
    if (currentVideoIndex < videos.length - 1) {
      setSelectedVideoId(videos[currentVideoIndex + 1].video_id);
    }
  };
  const goToPrevVideo = () => {
    if (currentVideoIndex > 0) {
      setSelectedVideoId(videos[currentVideoIndex - 1].video_id);
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
      <div className="w-64 bg-white border-r border-gray-200 overflow-y-auto">
        <div className="p-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-800">Videos ({videos.length})</h2>
          <div className="text-sm text-gray-500 mt-1">
            {videos.filter((v) => v.has_corrections).length} corrected
          </div>
        </div>
        <div className="divide-y divide-gray-100">
          {videos.map((video) => (
            <button
              key={video.video_id}
              onClick={() => setSelectedVideoId(video.video_id)}
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
              <div ref={containerRef} className="bg-gray-100 rounded-lg" />
              <div className="mt-4 flex items-center gap-4">
                <button
                  onClick={togglePlay}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                >
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
                <span className="font-mono text-gray-600">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="p-4 bg-white border-b border-gray-200 flex flex-wrap gap-2">
              <button
                onClick={addBoundaryAtPlayhead}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                + Add Boundary at Playhead
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
              <div className="flex-1" />
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Corrected Boundaries'}
              </button>
            </div>

            {/* Boundary list */}
            <div className="flex-1 p-4 overflow-y-auto">
              <h3 className="font-semibold text-gray-800 mb-3">
                Corrected Boundaries ({correctedBoundaries.length})
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
                      className="bg-white rounded-lg p-3 border border-gray-200 flex items-center gap-4"
                    >
                      <span className="font-semibold text-gray-700">Call {index + 1}</span>
                      <span className="font-mono text-sm text-gray-600">
                        {formatTime(boundary.start_s)} - {formatTime(boundary.end_s)}
                      </span>
                      <span className="text-sm text-gray-500">
                        ({Math.round(boundary.end_s - boundary.start_s)}s)
                      </span>
                      <div className="flex-1" />
                      <button
                        onClick={() => {
                          wavesurferRef.current?.setTime(boundary.start_s);
                          wavesurferRef.current?.play();
                        }}
                        className="px-3 py-1 text-indigo-600 hover:bg-indigo-50 rounded"
                      >
                        ▶ Preview
                      </button>
                      <button
                        onClick={() => deleteBoundary(boundary.id)}
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
