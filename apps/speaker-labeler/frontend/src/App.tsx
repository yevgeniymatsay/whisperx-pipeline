import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { SpeakerLabeler } from './pages/SpeakerLabeler';
import { BoundaryEditor } from './pages/BoundaryEditor';

function NavTabs() {
  const location = useLocation();

  return (
    <div className="bg-indigo-800 text-white px-4">
      <div className="flex gap-1">
        <Link
          to="/boundary-editor"
          className={`px-4 py-3 text-sm font-medium ${
            location.pathname === '/boundary-editor' || location.pathname === '/'
              ? 'bg-indigo-900 border-b-2 border-white'
              : 'hover:bg-indigo-700'
          }`}
        >
          Boundary Editor
        </Link>
        <Link
          to="/speaker-labeler"
          className={`px-4 py-3 text-sm font-medium ${
            location.pathname === '/speaker-labeler'
              ? 'bg-indigo-900 border-b-2 border-white'
              : 'hover:bg-indigo-700'
          }`}
        >
          Speaker Labeler
        </Link>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <NavTabs />
        <Routes>
          <Route path="/" element={<BoundaryEditor />} />
          <Route path="/boundary-editor" element={<BoundaryEditor />} />
          <Route path="/speaker-labeler" element={<SpeakerLabeler />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
