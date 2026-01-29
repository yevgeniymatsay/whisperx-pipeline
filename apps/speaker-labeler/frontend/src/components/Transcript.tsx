import { useEffect, useRef } from 'react';
import type { Turn } from '../api';

interface TranscriptProps {
  turns: Turn[];
  currentTime: number;
  speakerRoles: Record<string, string>;
  onTurnClick: (turn: Turn) => void;
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
    <div ref={containerRef} className="h-96 overflow-y-auto space-y-2 p-2 bg-white rounded-lg border border-gray-200">
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
              <span className="font-mono font-semibold">{turn.spk}</span>
              <span className="text-xs text-gray-400">[{formatTime(t0)} - {formatTime(t1)}]</span>
              <span>{dot}</span>
            </div>
            <p className="text-gray-900">{turn.text}</p>
          </div>
        );
      })}
    </div>
  );
}
