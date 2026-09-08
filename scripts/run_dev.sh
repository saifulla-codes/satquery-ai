#!/bin/bash
# SatQuery AI — Development Startup Script
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=================================================="
echo "SatQuery AI 🛰️ — Starting Local M2 Engine"
echo "SIH Problem Statement: SIH26167 | ISRO"
echo "=================================================="

# Ensure Python Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Setting up Python Virtual Environment..."
    python3 -m venv --system-site-packages .venv
fi

# Build Frontend if dist doesn't exist
if [ ! -d "frontend/dist" ]; then
    echo "Building Space-Tech Frontend..."
    if ! command -v npm &> /dev/null; then
        if [ -s "$HOME/.nvm/nvm.sh" ]; then
            . "$HOME/.nvm/nvm.sh"
        fi
    fi
    cd frontend && npm run build && cd ..
fi

# Run Integration Test Suite
echo "Running Validation Test Suite..."
PYTHONPATH="$DIR" "$DIR/.venv/bin/python" "$DIR/tests/test_api.py"

echo "=================================================="
echo "SatQuery AI is starting on: http://localhost:8000"
echo "Open http://localhost:8000 in your browser."
echo "=================================================="

PYTHONPATH="$DIR" "$DIR/.venv/bin/python" "$DIR/backend/main.py"
