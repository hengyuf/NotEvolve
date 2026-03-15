"""Evaluator adapter for notebook-agent.

Wraps the core math from evaluator.py without requiring openevolve.
The notebook-agent kernel can call `evaluate_distribution(support, probs)`
to score a candidate distribution.

This file is in the PROTECTED evaluator directory — the LLM cannot read it,
but it can call the registered evaluator tools.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core evaluation logic (copied from evaluator.py, no openevolve dependency)
# ---------------------------------------------------------------------------

MAX_SUPPORT_SIZE = 256
MERGE_TOL = 1e-16
ROUND_DIGITS = 15


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("Probability vector has non-positive total mass")
    return probs / total


def _canonicalize_distribution(
    support_raw, probs_raw
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support_raw, dtype=np.float64).ravel()
    probs = np.asarray(probs_raw, dtype=np.float64).ravel()

    if support.size == 0 or probs.size == 0:
        raise ValueError("Empty support or probability list")
    if support.size != probs.size:
        raise ValueError("support and probs must have the same length")
    if support.size > MAX_SUPPORT_SIZE:
        raise ValueError(
            f"Support size {support.size} exceeds MAX_SUPPORT_SIZE={MAX_SUPPORT_SIZE}"
        )
    if np.any(~np.isfinite(support)) or np.any(~np.isfinite(probs)):
        raise ValueError("support/probs contain non-finite values")
    if np.any(support < 0.0):
        raise ValueError("support points must be non-negative")

    order = np.argsort(support)
    support = support[order]
    probs = probs[order]

    merged_support = [float(support[0])]
    merged_probs = [float(probs[0])]
    for x, p in zip(support[1:], probs[1:]):
        if abs(float(x) - merged_support[-1]) <= MERGE_TOL:
            merged_probs[-1] += float(p)
        else:
            merged_support.append(float(x))
            merged_probs.append(float(p))

    support = np.asarray(merged_support, dtype=np.float64)
    probs = _normalize_probs(np.asarray(merged_probs, dtype=np.float64))

    keep = probs > 0.0
    support = support[keep]
    probs = probs[keep]

    if support.size < 2:
        raise ValueError("Need at least 2 support atoms with positive probability")

    return support, probs


def exact_strict_probability(support: np.ndarray, probs: np.ndarray) -> float:
    """Exact P[X1+X2+X3 < 2*X4] via weighted pair-sum CDF queries."""
    pair_sums = (support[:, None] + support[None, :]).ravel()
    pair_weights = (probs[:, None] * probs[None, :]).ravel()

    order = np.argsort(pair_sums)
    sorted_sums = pair_sums[order]
    cdf_weights = np.cumsum(pair_weights[order])

    thresholds = 2.0 * support[:, None] - support[None, :]
    positions = np.searchsorted(sorted_sums, thresholds.ravel(), side="left")
    pair_lt = np.where(positions > 0, cdf_weights[positions - 1], 0.0).reshape(
        len(support), len(support)
    )

    triple_lt = np.sum(probs[None, :] * pair_lt, axis=1)
    return float(np.dot(probs, triple_lt))


def exact_equal_probability(support: np.ndarray, probs: np.ndarray) -> float:
    """Exact P[X1+X2+X3 = 2*X4] with rounded key aggregation."""
    pair_sums = (support[:, None] + support[None, :]).ravel()
    pair_weights = (probs[:, None] * probs[None, :]).ravel()

    mass_by_sum: dict[float, float] = {}
    for s, w in zip(pair_sums, pair_weights):
        key = round(float(s), ROUND_DIGITS)
        mass_by_sum[key] = mass_by_sum.get(key, 0.0) + float(w)

    eq = 0.0
    for l_idx, x_l in enumerate(support):
        target_base = 2.0 * float(x_l)
        for i_idx, x_i in enumerate(support):
            target = round(target_base - float(x_i), ROUND_DIGITS)
            eq += float(probs[l_idx] * probs[i_idx] * mass_by_sum.get(target, 0.0))

    return float(eq)


# ---------------------------------------------------------------------------
# Public API: called by notebook-agent evaluator tools
# ---------------------------------------------------------------------------

# Track the best score seen so far
_best_score: float = 0.0
_best_support: list[float] = []
_best_probs: list[float] = []


def evaluate_distribution(support: list[float], probs: list[float]) -> str:
    """Evaluate a candidate discrete distribution.

    Args:
        support: List of non-negative support points.
        probs: List of probability weights (will be normalized).

    Returns:
        String with evaluation results including score.
    """
    global _best_score, _best_support, _best_probs

    try:
        s, p = _canonicalize_distribution(support, probs)
        strict = exact_strict_probability(s, p)
        equal = exact_equal_probability(s, p)
        non_strict = strict + equal
        lifted = strict / (1.0 - equal) if equal < 1.0 else 0.0
        entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))

        is_new_best = strict > _best_score
        if is_new_best:
            _best_score = strict
            _best_support = support
            _best_probs = probs

        return (
            f"Score (P[X1+X2+X3 < 2X4]): {strict:.10f}\n"
            f"  Equal probability:        {equal:.10f}\n"
            f"  Non-strict probability:   {non_strict:.10f}\n"
            f"  Lifted ratio:             {lifted:.10f}\n"
            f"  Support size:             {len(s)}\n"
            f"  Entropy:                  {entropy:.6f}\n"
            f"  {'*** NEW BEST ***' if is_new_best else ''}\n"
            f"  Current best score:       {_best_score:.10f}"
        )

    except (ValueError, RuntimeError) as e:
        return f"Evaluation error: {e}"


def check_score() -> str:
    """Check the current best score and distribution.

    Returns:
        String with current best score and distribution details.
    """
    if _best_score == 0.0:
        return (
            "No valid distribution evaluated yet.\n"
            "Known benchmarks:\n"
            "  AlphaEvolve reported: 0.389\n"
            "  Human best (Bellec-Fritz): 0.400695"
        )

    return (
        f"Current best score: {_best_score:.10f}\n"
        f"  Support size: {len(_best_support)}\n"
        f"  Support: {_best_support[:10]}{'...' if len(_best_support) > 10 else ''}\n"
        f"  Probs:   {_best_probs[:10]}{'...' if len(_best_probs) > 10 else ''}\n"
        f"\n"
        f"Known benchmarks:\n"
        f"  AlphaEvolve reported: 0.389\n"
        f"  Human best (Bellec-Fritz): 0.400695"
    )
