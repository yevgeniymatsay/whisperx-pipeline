import { useState, useEffect, useCallback } from 'react';
import { Waveform } from '../components/Waveform';
import { Transcript } from '../components/Transcript';
import { SpeakerPanel } from '../components/SpeakerPanel';
import { NavBar } from '../components/NavBar';
import { getCurrentCall, getCallData, saveLabels, skipCall, navigate } from '../api';
import type { Turn, CallData } from '../api';

export function SpeakerLabeler() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [total, setTotal] = useState(0);
  const [callId, setCallId] = useState<string | null>(null);
  const [callData, setCallData] = useState<CallData | null>(null);
  const [speakerRoles, setSpeakerRoles] = useState<Record<string, string>>({});
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
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
        setSeekTo(undefined);
        setCurrentTime(0);
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
    const t0_abs = turn.t0_abs ?? turn.t0 ?? 0;
    const clipStart = callData?.start_time_s ?? 0;
    const relativeTime = Math.max(0, t0_abs - clipStart);
    setSeekTo(undefined);
    setTimeout(() => setSeekTo(relativeTime), 0);
  };

  const handleSave = async () => {
    if (!callId || saving) return;
    setSaving(true);
    try {
      await saveLabels(callId, speakerRoles);
      await navigate('next');
      await loadCurrentCall();
    } finally {
      setSaving(false);
    }
  };

  const handleSkip = async () => {
    if (!callId || saving) return;
    setSaving(true);
    try {
      await skipCall(callId);
      await navigate('next');
      await loadCurrentCall();
    } finally {
      setSaving(false);
    }
  };

  const handlePrev = async () => {
    await navigate('prev');
    await loadCurrentCall();
  };

  const canSave = !saving && ((callData?.speakers.length ?? 0) > 0) && (callData?.speakers.every((s) => speakerRoles[s]) ?? false);

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
    <div className="flex-1 flex flex-col">
      <NavBar
        currentIndex={currentIndex}
        total={total}
        onPrev={handlePrev}
        onSkip={handleSkip}
        onSave={handleSave}
        canSave={canSave}
        saving={saving}
      />

      <div className="flex-1 p-4 space-y-4 max-w-5xl mx-auto w-full">
        {callData && (
          <>
            <div className="text-sm text-gray-500">
              Call ID: <span className="font-mono">{callData.call_id}</span> | Video: <span className="font-mono">{callData.video_id}</span>
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
