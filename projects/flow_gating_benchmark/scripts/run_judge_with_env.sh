#!/bin/bash
# Wrapper script to run judge phase with proper environment variables
set -e

# Export all env vars from .env
export $(grep -v '^#' /home/jb/claude-projects/.env | grep -v '^$' | xargs)

# Verify API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not set"
    exit 1
fi

echo "GOOGLE_API_KEY is set"
echo "Running judge phase..."

cd /home/jb/claude-projects/projects/flow_gating_benchmark

# Run the judge phase with all arguments passed through
python scripts/run_modular_pipeline.py "$@"
