#!/usr/bin/env bash
# start.sh — Start both API and Streamlit for local development
set -e

echo "🚀 Starting DataPrep Pro..."
echo ""

# Start FastAPI backend
echo "📡 Starting API server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait a moment for the API to start
sleep 2

# Start Streamlit frontend
echo "🖥️  Starting Streamlit UI on port 8501..."
streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0 &
APP_PID=$!

echo ""
echo "✅ DataPrep Pro is running!"
echo "   API:  http://localhost:8000"
echo "   UI:   http://localhost:8501"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services."

# Handle shutdown
trap "kill $API_PID $APP_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for both processes
wait
