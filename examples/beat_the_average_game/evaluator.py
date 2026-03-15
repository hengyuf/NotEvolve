"""
Evaluator for Beat the average game with discrete probability search space.

Expected program output:
    {
      "support": [x_0, ..., x_{m-1}],
      "probs":   [c_0, ..., c_{m-1}]
    }

where c_i >= 0, sum c_i = 1, x_i >= 0.

Primary metric:
    combined_score = exact strict probability
        P[X1 + X2 + X3 < 2X4], Xi iid~mu.
"""

import importlib.util
import time
import traceback

import numpy as np

from openevolve.evaluation_result import EvaluationResult


MAX_SUPPORT_SIZE = 256
MERGE_TOL = 1e-16
ROUND_DIGITS = 15


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load program from {program_path}")
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if total <= 0.0:
        raise RuntimeError("Probability vector has non-positive total mass")
    return probs / total


def _canonicalize_distribution(support_raw, probs_raw) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support_raw, dtype=np.float64).ravel()
    probs = np.asarray(probs_raw, dtype=np.float64).ravel()

    if support.size == 0 or probs.size == 0:
        raise RuntimeError("Empty support or probability list")
    if support.size != probs.size:
        raise RuntimeError("support and probs must have the same length")
    if support.size > MAX_SUPPORT_SIZE:
        raise RuntimeError(
            f"Support size {support.size} exceeds MAX_SUPPORT_SIZE={MAX_SUPPORT_SIZE}"
        )

    if np.any(~np.isfinite(support)) or np.any(~np.isfinite(probs)):
        raise RuntimeError("support/probs contain non-finite values")
    if np.any(support < 0.0):
        raise RuntimeError("support points must be non-negative")

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
        raise RuntimeError("Need at least 2 support atoms with positive probability")

    return support, probs


def exact_strict_probability(support: np.ndarray, probs: np.ndarray) -> float:
    """Exact strict probability via weighted pair-sum CDF queries."""
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
    """Exact equality probability P[X1+X2+X3 = 2X4] with rounded key aggregation."""
    pair_sums = (support[:, None] + support[None, :]).ravel()
    pair_weights = (probs[:, None] * probs[None, :]).ravel()

    mass_by_sum = {}
    for s, w in zip(pair_sums, pair_weights):
        key = round(float(s), ROUND_DIGITS)
        mass_by_sum[key] = mass_by_sum.get(key, 0.0) + float(w)

    eq = 0.0
    for l, x_l in enumerate(support):
        target_base = 2.0 * float(x_l)
        for i, x_i in enumerate(support):
            target = round(target_base - float(x_i), ROUND_DIGITS)
            eq += float(probs[l] * probs[i] * mass_by_sum.get(target, 0.0))

    return float(eq)


def _extract_distribution(program_path: str) -> tuple[np.ndarray, np.ndarray]:
    program = _load_program(program_path)

    if not hasattr(program, "run_construction"):
        raise RuntimeError("Missing run_construction()")

    result = program.run_construction()
    if not isinstance(result, dict):
        raise RuntimeError("run_construction() must return a dict")

    if "support" not in result or "probs" not in result:
        raise RuntimeError("run_construction() must return keys: support, probs")

    return _canonicalize_distribution(result["support"], result["probs"])


def evaluate(program_path: str) -> EvaluationResult:
    try:
        start = time.time()

        support, probs = _extract_distribution(program_path)
        strict = exact_strict_probability(support, probs)
        equal = exact_equal_probability(support, probs)
        non_strict = strict + equal

        lifted = strict / (1.0 - equal) if equal < 1.0 else 0.0
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-300))))
        mass_at_zero = float(np.sum(probs[np.abs(support) <= MERGE_TOL]))

        eval_time = time.time() - start

        return EvaluationResult(
            metrics={
                "combined_score": float(strict),
                "strict_probability": float(strict),
                "equal_probability": float(equal),
                "non_strict_probability": float(non_strict),
                "lifted_ratio": float(lifted),
                "support_size": float(len(support)),
                "mass_at_zero": float(mass_at_zero),
                "entropy": float(entropy),
                "validity": 1.0,
                "eval_time": float(eval_time),
            },
            artifacts={
                "interpretation": (
                    "combined_score is exact P[X1+X2+X3 < 2X4] for the returned "
                    "discrete iid distribution"
                )
            },
        )

    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "strict_probability": 0.0,
                "equal_probability": 0.0,
                "non_strict_probability": 0.0,
                "lifted_ratio": 0.0,
                "support_size": 0.0,
                "mass_at_zero": 0.0,
                "entropy": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
            },
            artifacts={
                "stderr": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    try:
        _extract_distribution(program_path)
        return EvaluationResult(
            metrics={"combined_score": 0.5},
            artifacts={"stage": "validation_passed"},
        )
    except Exception as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"stderr": str(exc)},
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation."""
    return evaluate(program_path)
