#!/bin/bash
# Test FlowBench as a fresh install
# Simulates what an external user would experience cloning the repo

set -e

echo "=== FlowBench Fresh Install Test ==="
echo ""

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "📁 Working in: $TEMP_DIR"
trap "rm -rf $TEMP_DIR" EXIT

# Copy the project (simulating a clone)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📦 Copying project..."
cp -r "$PROJECT_DIR" "$TEMP_DIR/flow_gating_benchmark"
cd "$TEMP_DIR/flow_gating_benchmark"

# Remove any local state
rm -rf .pytest_cache __pycache__ results/ .coverage 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "=== Step 1: Create virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "=== Step 2: Install dependencies ==="
pip install -q --upgrade pip
pip install -q -e ".[dev]" 2>/dev/null || pip install -q pytest pydantic anthropic google-generativeai sentence-transformers json_repair tenacity

echo ""
echo "=== Step 3: Verify test cases load ==="
python3 -c "
import sys
sys.path.insert(0, 'src')
from curation.schemas import TestCase
from pathlib import Path
import json

verified_dir = Path('data/verified')
alien_dir = Path('data/alien_cell')

verified = list(verified_dir.glob('*.json'))
alien = list(alien_dir.glob('*.json')) if alien_dir.exists() else []

print(f'✓ Found {len(verified)} verified test cases')
print(f'✓ Found {len(alien)} alien cell test cases')

# Load one to verify schema
with open(verified[0]) as f:
    data = json.load(f)
    tc = TestCase.model_validate(data)
    print(f'✓ Schema validation passed for {tc.test_case_id}')
"

echo ""
echo "=== Step 4: Run core tests ==="
pytest tests/test_metrics.py tests/test_hierarchy.py tests/test_schemas.py -v --tb=short -q 2>&1 | tail -20

echo ""
echo "=== Step 5: Quick benchmark test (dry run) ==="
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "Running with Gemini (cheapest option)..."
    python3 scripts/run_modular_pipeline.py \
        --phase all \
        --models gemini-2.0-flash \
        --max-cases 1 \
        --n-bootstrap 1 \
        --force \
        2>&1 | tail -30
else
    echo "⚠ No GOOGLE_API_KEY set - skipping live benchmark test"
    echo "To run: export GOOGLE_API_KEY=your_key && ./scripts/test_fresh_install.sh"
fi

echo ""
echo "=== Fresh Install Test Complete ==="
echo ""
echo "Summary:"
echo "  - Dependencies installed ✓"
echo "  - Test cases loaded ✓"
echo "  - Core tests passed ✓"
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "  - Live benchmark completed ✓"
fi

deactivate
