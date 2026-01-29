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
