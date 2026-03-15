#!/usr/bin/env bash
#
# Run notebook-agent on the Beat the Average Game problem.
#
# Usage:
#   bash examples/beat_the_average_game/run.sh
#   bash examples/beat_the_average_game/run.sh --max-rounds 10
#   bash examples/beat_the_average_game/run.sh --max-sessions 3 --rounds-per-session 10
#   bash examples/beat_the_average_game/run.sh --verbose
#
# Requires:
#   - GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable set
#   - notebook-agent installed: pip install -e ".[dev]"
#   - numpy installed in the kernel environment
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default args (can be overridden via CLI)
MAX_ROUNDS="${MAX_ROUNDS:-30}"
MODEL="${MODEL:-gemini-2.5-pro}"

# Check API key — accept GEMINI_API_KEY or GOOGLE_API_KEY
GEMINI_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
if [[ -z "$GEMINI_KEY" ]]; then
    echo "Error: GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set."
    echo "  export GEMINI_API_KEY=AI..."
    exit 1
fi

echo "=== Beat the Average Game ==="
echo "Problem: maximize P[X1+X2+X3 < 2*X4], Xi iid ~ mu"
echo "Model:   ${MODEL}"
echo "Rounds:  ${MAX_ROUNDS}"
echo "Dir:     ${SCRIPT_DIR}"
echo ""

cd "$SCRIPT_DIR"

exec python -m notebook_agent \
    --config-file config.json \
    --task "Find a discrete probability distribution mu on [0, infinity) that maximizes P[X1+X2+X3 < 2*X4] where X1,...,X4 are iid from mu.

You have two evaluator tools:
- evaluate_distribution(support=[...], probs=[...]): Score a candidate distribution. Returns exact P[X1+X2+X3 < 2X4].
- check_score(): Check current best score and known benchmarks.

The distribution is specified as a discrete measure: support points (non-negative reals) and probability weights (normalized automatically).

Known benchmarks:
- AlphaEvolve reported: 0.389
- Human best (Bellec-Fritz): 0.400695

Your goal: find a distribution that achieves a score as high as possible. Try different approaches:
1. Start with simple distributions (uniform, geometric, etc.) to build intuition
2. Analyze what makes X4 large relative to X1+X2+X3 — the support should have mass on large values
3. Try optimization: gradient-free search, evolutionary strategies, or analytical insights
4. Iterate and refine the best distributions found

Max support size: 256 atoms." \
    --notebook notebook.ipynb \
    --model "$MODEL" \
    --provider openai_compatible \
    --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
    --api-key "$GEMINI_KEY" \
    --thinking-budget 0 \
    --working-dir "$SCRIPT_DIR" \
    --evaluator-module evaluator_adapter.py \
    --evaluator-path evaluator.py \
    --evaluator-path evaluator_adapter.py \
    --checkpoint-dir .checkpoints \
    --max-rounds "$MAX_ROUNDS" \
    "$@"
