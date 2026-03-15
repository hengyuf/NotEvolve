# Beat the Average Game

AlphaEvolve Problem 39: maximize

\[
C = \sup_{\mu} \Pr[X_1 + X_2 + X_3 < 2X_4], \quad X_1,\dots,X_4 \stackrel{iid}{\sim} \mu,
\]

where \(\mu\) is a probability measure on \([0,\infty)\).

## Source Mapping

- Problem page: `alphaevolve_repository_of_problems/problems/39.html`
- In this repository's `paper.pdf`, this appears as **Problem 6.39** under "Beat the average game".
- The problem HTML currently references "Section 6.22" (older numbering).

## Corrected Search Space (Discrete Probabilities)

This implementation now follows the discrete-measure formulation described in the paper discussion:

\[
\mu = \sum_{i=1}^{m} c_i\,\delta_{x_i}, \quad c_i \ge 0,\; \sum_i c_i = 1,
\]

and optimizes the exact objective

\[
\Pr[X_1 + X_2 + X_3 < 2X_4].
\]

The evaluator computes this exactly for returned discrete laws using convolution-style pair-sum CDF queries.

## Current Art (Cited)

- Human lower bound (cited): **0.400695** (Bellec-Fritz, arXiv:2412.15179).
- AlphaEvolve-reported lower bound: **0.389** (paper discussion).

This runnable example provides a fully discrete-probability OpenEvolve setup; exact original search artifacts for the reported 0.389/0.400695 constructions are not included in the local repository.

## Files

- `initial_program.py`: Evolvable discrete-probability search (`search_for_best_distribution`).
- `evaluator.py`: Trusted exact scoring for discrete iid measures.
- `best_program.py`: Hardcoded strongest local discrete vector in this folder.
- `best_known_program.py`: Structured baseline discrete vector.
- `best_program_info.json`: Metric definitions and directions.
- `config_phase_1.yaml`: Exploration config.
- `config_phase_2.yaml`: Refinement config.
- `config.yaml`: Single-phase default config.
- `run.sh`: Phase 1/Phase 2 launcher.
- `requirements.txt`: Minimal dependency list.

## Running

```bash
# Phase 1 (default)
bash tao_examples/beat_the_average_game/run.sh

# Phase 2 (starts from latest checkpoint best_program.py)
bash tao_examples/beat_the_average_game/run.sh --phase2

# Quick smoke test
bash tao_examples/beat_the_average_game/run.sh --iterations 5
```

## References

- AlphaEvolve Problem 39 page: https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/39.html
- AlphaEvolve paper: https://arxiv.org/abs/2511.02864
- Bellec-Fritz paper: https://arxiv.org/pdf/2412.15179
