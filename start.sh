#!/bin/bash

# Terminate background processes on exit
trap "kill 0" EXIT

echo "============================================="
echo "   Starting DeepThinker Multi-Agent Hub...   "
echo "============================================="

# Sourcing compiler variables if Intel environment exists
if [ -f "/opt/intel/oneapi/compiler/latest/env/vars.sh" ]; then
    echo "Sourcing Intel oneAPI Compiler environment variables..."
    source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1
elif [ -f "/opt/intel/oneapi/compiler/2026.0/env/vars.sh" ]; then
    echo "Sourcing Intel oneAPI 2026.0 Compiler environment variables..."
    source /opt/intel/oneapi/compiler/2026.0/env/vars.sh >/dev/null 2>&1
fi

if [ -f "/opt/intel/oneapi/mkl/2026.0/env/vars.sh" ]; then
    echo "Sourcing Intel MKL environment variables..."
    source /opt/intel/oneapi/mkl/2026.0/env/vars.sh >/dev/null 2>&1
fi

# 1. Start backend server
echo "Launching FastAPI Backend..."
venv/bin/python backend/app.py &
BACKEND_PID=$!

# Wait a second for backend to boot
sleep 2

# 2. Start frontend dev server
echo "Launching React/Vite Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "============================================="
echo "🟢 Both servers are running!"
echo "👉 Web UI URL: http://localhost:5173"
echo "👉 Backend API URL: http://127.0.0.1:8000"
echo "👉 Monitor downloads, GPU caching, and RAM in the dashboard!"
echo "Press Ctrl+C to stop both servers."
echo "============================================="

# Keep script running
wait
