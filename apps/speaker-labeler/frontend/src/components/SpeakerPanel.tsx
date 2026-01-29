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
                      {role.id === 'agent' && '🟢 '}
                      {role.id === 'user' && '🔵 '}
                      {role.id === 'narrator' && '⚪ '}
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
