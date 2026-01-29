import { useEffect, useRef, useCallback, useState } from 'react';
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
  const mountedRef = useRef(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mountedRef.current = true;

    if (!containerRef.current || !audioUrl) return;

    setError(null);

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

    // Handle the load promise to catch AbortError during Strict Mode cleanup
    ws.load(audioUrl).catch((err) => {
      // Ignore AbortError - expected during React Strict Mode double-invoke
      if (err?.name === 'AbortError') return;
      // Real errors will be handled by the 'error' event
    });

    ws.on('ready', () => {
      if (!mountedRef.current) return;
      const dur = ws.getDuration();
      setDuration(dur);
      onReady?.(dur);
    });

    // v7 uses 'timeupdate' instead of 'audioprocess'
    ws.on('timeupdate', () => {
      if (!mountedRef.current) return;
      const time = ws.getCurrentTime();
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });

    ws.on('seeking', () => {
      if (!mountedRef.current) return;
      const time = ws.getCurrentTime();
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });

    ws.on('play', () => {
      if (!mountedRef.current) return;
      setIsPlaying(true);
    });

    ws.on('pause', () => {
      if (!mountedRef.current) return;
      setIsPlaying(false);
    });

    ws.on('error', (err) => {
      // Ignore AbortError - happens during React Strict Mode cleanup
      if (err?.name === 'AbortError') return;
      if (!mountedRef.current) return;
      console.error('Waveform error:', err);
      setError('Failed to load audio. The file may be missing or inaccessible.');
    });

    wavesurferRef.current = ws;

    return () => {
      mountedRef.current = false;
      wavesurferRef.current = null;
      // Destroy WaveSurfer - the load().catch() above handles any AbortError
      ws.destroy();
    };
  }, [audioUrl, onTimeUpdate, onReady]);

  useEffect(() => {
    if (wavesurferRef.current && seekTo !== undefined) {
      const dur = wavesurferRef.current.getDuration();
      if (dur > 0) {
        wavesurferRef.current.seekTo(seekTo / dur);
        wavesurferRef.current.play();
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

  if (!audioUrl) {
    return (
      <div className="bg-gray-100 rounded-lg p-4 text-center text-gray-500">
        No audio available
      </div>
    );
  }

  return (
    <div className="bg-gray-100 rounded-lg p-4">
      {error && (
        <div className="bg-red-100 text-red-700 p-3 rounded-lg mb-4">
          {error}
        </div>
      )}
      <div ref={containerRef} className="mb-4" />
      <div className="flex items-center gap-4">
        <button
          onClick={togglePlay}
          disabled={!!error}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <span className="text-gray-600 font-mono">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}
