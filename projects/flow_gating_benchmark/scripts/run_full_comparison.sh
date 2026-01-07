#!/bin/bash
#
# Run full benchmark comparison across all conditions
#
# Usage: ./scripts/run_full_comparison.sh
#
# Estimated time: ~2 hours
# Estimated cost: ~$8-10
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================================"
echo "FLOW GATING BENCHMARK - FULL COMPARISON"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ -f "../../.env" ]; then
        export $(grep -v '^#' ../../.env | xargs)
    fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

# Results directory
RESULTS_DIR="results/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================================
# Phase 1: Baseline
# ============================================================
echo "------------------------------------------------------------"
echo "Phase 1: Baseline (CoT, Standard Context)"
echo "------------------------------------------------------------"
python run_benchmark.py --strategy cot --context standard 2>&1 | tee "$RESULTS_DIR/01_baseline_cot.log"

# ============================================================
# Phase 2: MCP Ablation
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "Phase 2: MCP Ablation"
echo "------------------------------------------------------------"
python run_benchmark.py --ablation mcp --strategy cot 2>&1 | tee "$RESULTS_DIR/02_mcp_cot.log"
python run_benchmark.py --ablation mcp --strategy direct 2>&1 | tee "$RESULTS_DIR/03_mcp_direct.log"

# ============================================================
# Phase 3: Skills Ablation
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "Phase 3: Skills Ablation"
echo "------------------------------------------------------------"
python run_benchmark.py --ablation skills --strategy cot 2>&1 | tee "$RESULTS_DIR/04_skills_cot.log"
python run_benchmark.py --ablation skills --strategy direct 2>&1 | tee "$RESULTS_DIR/05_skills_direct.log"

# ============================================================
# Phase 4: Strategy Comparison
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "Phase 4: Strategy Comparison"
echo "------------------------------------------------------------"
python run_benchmark.py --strategy direct --context standard 2>&1 | tee "$RESULTS_DIR/06_direct_baseline.log"

# ============================================================
# Phase 5: Context Level Comparison
# ============================================================
echo ""
echo "------------------------------------------------------------"
echo "Phase 5: Context Level Comparison"
echo "------------------------------------------------------------"
python run_benchmark.py --context minimal --strategy cot 2>&1 | tee "$RESULTS_DIR/07_minimal_context.log"
python run_benchmark.py --context rich --strategy cot 2>&1 | tee "$RESULTS_DIR/08_rich_context.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "COMPARISON COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Extract summary metrics from each log
echo "Summary of Results:"
echo "-------------------"
for log in "$RESULTS_DIR"/*.log; do
    name=$(basename "$log" .log)
    f1=$(grep "Hierarchy F1:" "$log" | tail -1 | awk '{print $3}')
    crit=$(grep "Critical Gate Recall:" "$log" | tail -1 | awk '{print $4}')
    echo "$name: F1=$f1, Critical=$crit"
done

echo ""
echo "See individual log files for detailed results."
