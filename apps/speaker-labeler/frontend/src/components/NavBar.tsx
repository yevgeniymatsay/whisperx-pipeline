interface NavBarProps {
  currentIndex: number;
  total: number;
  onPrev: () => void;
  onSkip: () => void;
  onSave: () => void;
  canSave: boolean;
  saving?: boolean;
}

export function NavBar({ currentIndex, total, onPrev, onSkip, onSave, canSave, saving }: NavBarProps) {
  return (
    <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
      <button
        onClick={onPrev}
        disabled={currentIndex === 0 || saving}
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
          disabled={saving}
          className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg border border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Skip
        </button>
        <button
          onClick={onSave}
          disabled={!canSave}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? 'Saving...' : 'Save & Next →'}
        </button>
      </div>
    </div>
  );
}
