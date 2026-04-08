"""
optimizers.py — Three Optimization Strategies for PJT MARS
===========================================================
Implements the lexicographic constraint: quality (Ra + 2σ ≤ Ra_target) is a
HARD constraint; MRR is maximized only among feasible solutions.

Three strategies:
  1. Naive        — Exhaustive filter-then-maximize over the discrete grid
  2. CP-SAT       — Exact discrete optimizer (OR-Tools, integer-scaled)
  3. NSGA-II      — Multi-objective Pareto optimizer (pymoo, continuous domain)
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from surrogate_model import BootstrapEnsemble, predict
from candidate_space import build_candidate_space, GRID_CONFIG, compute_MRR


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class OptimizationResult:
    """
    Unified result for all three optimizers.

    Attributes
    ----------
    method        : str — "Naive", "CP-SAT", or "NSGA-II"
    best_Vc       : float | None — None if no feasible solution found
    best_fz       : float | None
    best_ap       : float | None
    best_MRR      : float | None
    best_Ra_mean  : float | None
    best_Ra_sigma : float | None
    feasible      : bool — True if Ra_mean + 2*Ra_sigma <= Ra_target
    n_feasible    : int — number of candidates satisfying quality constraint
    solve_time_s  : float — wall-clock seconds
    pareto_front  : pd.DataFrame | None — populated only for NSGA-II
    """
    method: str = ""
    best_Vc: Optional[float] = None
    best_fz: Optional[float] = None
    best_ap: Optional[float] = None
    best_MRR: Optional[float] = None
    best_Ra_mean: Optional[float] = None
    best_Ra_sigma: Optional[float] = None
    feasible: bool = False
    n_feasible: int = 0
    solve_time_s: float = 0.0
    pareto_front: Optional[pd.DataFrame] = None


# ── Optimizer 1: Naive Exhaustive Search ──────────────────────────────────────
def optimize_naive(
    ensemble: BootstrapEnsemble,
    VB: float,
    Ra_target: float,
    grid_config: Optional[dict] = None,
) -> OptimizationResult:
    """
    Exhaustive filter-then-maximize optimizer.

    Algorithm:
    1. Build discrete candidate space
    2. Predict Ra_mean, Ra_sigma for ALL candidates (one vectorized call)
    3. Compute Ra_upper = Ra_mean + 2 * Ra_sigma
    4. Filter candidates where Ra_upper <= Ra_target
    5. Among feasible candidates, pick the one with highest MRR

    Time complexity: O(n_candidates) — always terminates.
    """
    t0 = time.perf_counter()
    grid_config = grid_config or GRID_CONFIG

    candidates = build_candidate_space(VB, grid_config)
    Ra_mean, Ra_sigma = predict(ensemble, candidates[["Vc", "fz", "ap", "VB"]])

    candidates = candidates.copy()
    candidates["Ra_mean"] = Ra_mean
    candidates["Ra_sigma"] = Ra_sigma
    candidates["Ra_upper"] = Ra_mean + 2.0 * Ra_sigma

    feasible = candidates[candidates["Ra_upper"] <= Ra_target]
    n_feasible = len(feasible)

    if n_feasible == 0:
        return OptimizationResult(
            method="Naive",
            feasible=False,
            n_feasible=0,
            solve_time_s=time.perf_counter() - t0,
        )

    best = feasible.sort_values("MRR", ascending=False).iloc[0]

    return OptimizationResult(
        method="Naive",
        best_Vc=float(best["Vc"]),
        best_fz=float(best["fz"]),
        best_ap=float(best["ap"]),
        best_MRR=float(best["MRR"]),
        best_Ra_mean=float(best["Ra_mean"]),
        best_Ra_sigma=float(best["Ra_sigma"]),
        feasible=True,
        n_feasible=n_feasible,
        solve_time_s=time.perf_counter() - t0,
    )


# ── Optimizer 2: CP-SAT Exact Discrete Optimizer ─────────────────────────────
def optimize_cpsat(
    ensemble: BootstrapEnsemble,
    VB: float,
    Ra_target: float,
    grid_config: Optional[dict] = None,
    scale: int = 1000,
    time_limit_s: float = 10.0,
) -> OptimizationResult:
    """
    Exact discrete optimizer using OR-Tools CP-SAT solver.

    Integer encoding (scale=1000):
    - All float values × scale → rounded integer
    - MRR_int[i] = round(MRR_float[i] * scale)  — max ≈ 270,000 (safe for int32)
    - Ra_upper_int[i] = round(Ra_upper[i] * scale)
    - Ra_target_int = round(Ra_target * scale)

    IMPORTANT: MRR_int uses the float MRR directly, NOT Vc_int * fz_int * ap_int
    (the latter would overflow int32).

    Model:
    - Binary x[i] ∈ {0,1} for each candidate
    - Constraint: sum(x) == 1 (select exactly one)
    - Quality: model.Add(Ra_upper_int[i] <= Ra_target_int).OnlyEnforceIf(x[i])
    - Objective: maximize sum(x[i] * MRR_int[i])
    """
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    grid_config = grid_config or GRID_CONFIG

    # Pre-compute all predictions on the candidate space
    candidates = build_candidate_space(VB, grid_config)
    Ra_mean, Ra_sigma = predict(ensemble, candidates[["Vc", "fz", "ap", "VB"]])
    Ra_upper = Ra_mean + 2.0 * Ra_sigma

    n = len(candidates)
    MRR_vals = candidates["MRR"].values

    # Integer encoding — scale float values, NOT multiply integer coordinates
    Ra_upper_int = np.round(Ra_upper * scale).astype(int)
    MRR_int = np.round(MRR_vals * scale).astype(int)
    Ra_target_int = int(round(Ra_target * scale))

    # Build CP-SAT model
    model = cp_model.CpModel()

    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    # Exactly one candidate must be selected
    model.AddExactlyOne(x)

    # Quality constraint: only enforce Ra_upper <= Ra_target for the selected candidate
    for i in range(n):
        model.Add(Ra_upper_int[i] <= Ra_target_int).OnlyEnforceIf(x[i])

    # Objective: maximize MRR
    model.Maximize(sum(x[i] * int(MRR_int[i]) for i in range(n)))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)

    elapsed = time.perf_counter() - t0

    if solver.StatusName(status) not in ("OPTIMAL", "FEASIBLE"):
        return OptimizationResult(
            method="CP-SAT",
            feasible=False,
            n_feasible=0,
            solve_time_s=elapsed,
        )

    # Extract selected candidate
    selected = None
    for i in range(n):
        if solver.Value(x[i]) == 1:
            selected = i
            break

    if selected is None:
        return OptimizationResult(
            method="CP-SAT",
            feasible=False,
            n_feasible=0,
            solve_time_s=elapsed,
        )

    row = candidates.iloc[selected]
    n_feasible = int(np.sum(Ra_upper <= Ra_target))

    return OptimizationResult(
        method="CP-SAT",
        best_Vc=float(row["Vc"]),
        best_fz=float(row["fz"]),
        best_ap=float(row["ap"]),
        best_MRR=float(row["MRR"]),
        best_Ra_mean=float(Ra_mean[selected]),
        best_Ra_sigma=float(Ra_sigma[selected]),
        feasible=True,
        n_feasible=n_feasible,
        solve_time_s=elapsed,
    )


# ── Optimizer 3: NSGA-II Multi-Objective (pymoo 0.6.x) ───────────────────────
def optimize_nsgaii(
    ensemble: BootstrapEnsemble,
    VB: float,
    Ra_target: float,
    grid_config: Optional[dict] = None,
    pop_size: int = 50,
    n_gen: int = 100,
    seed: int = 42,
) -> OptimizationResult:
    """
    Multi-objective Pareto optimization using pymoo NSGA-II.

    Operates on the CONTINUOUS cutting parameter domain (not the discrete grid).
    VB is fixed as a problem constant (not a decision variable).

    Objectives (both minimized by pymoo convention):
        f1 = Ra_mean     (minimize Ra)
        f2 = -MRR        (maximize MRR ≡ minimize -MRR)

    Decision variables: x = [Vc, fz, ap]
        xl = [50.0, 0.05, 0.5]
        xu = [300.0, 0.30, 3.0]

    Post-processing:
    - Extract Pareto front via result.opt.get("X") and result.opt.get("F")
    - Recompute Ra_sigma for each Pareto point
    - Filter: Ra_mean + 2*Ra_sigma <= Ra_target
    - Best = feasible Pareto point with highest MRR
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    t0 = time.perf_counter()

    class MillingProblem(ElementwiseProblem):
        def __init__(self, _ensemble, _VB):
            super().__init__(
                n_var=3,
                n_obj=2,
                xl=np.array([50.0, 0.05, 0.5]),
                xu=np.array([300.0, 0.30, 3.0]),
            )
            self._ensemble = _ensemble
            self._VB = _VB

        def _evaluate(self, x, out, *args, **kwargs):
            Vc, fz, ap = x[0], x[1], x[2]
            X_pred = pd.DataFrame({
                "Vc": [Vc], "fz": [fz], "ap": [ap], "VB": [self._VB]
            })
            Ra_mean, Ra_sigma = predict(self._ensemble, X_pred)
            MRR = Vc * fz * ap
            out["F"] = [float(Ra_mean[0]), -float(MRR)]

    problem = MillingProblem(ensemble, VB)
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )

    elapsed = time.perf_counter() - t0

    # Extract Pareto front (pymoo 0.6.x: result.opt, not result.X)
    try:
        opt_X = result.opt.get("X")
        opt_F = result.opt.get("F")
    except Exception:
        opt_X = result.X
        opt_F = result.F

    if opt_X is None or len(opt_X) == 0:
        return OptimizationResult(
            method="NSGA-II",
            feasible=False,
            n_feasible=0,
            solve_time_s=elapsed,
        )

    # Reconstruct Ra_sigma for each Pareto point
    pareto_rows = []
    for i in range(len(opt_X)):
        Vc, fz, ap = opt_X[i]
        X_pred = pd.DataFrame({"Vc": [Vc], "fz": [fz], "ap": [ap], "VB": [VB]})
        Ra_mean_i, Ra_sigma_i = predict(ensemble, X_pred)
        MRR_i = Vc * fz * ap
        Ra_upper_i = float(Ra_mean_i[0]) + 2.0 * float(Ra_sigma_i[0])
        pareto_rows.append({
            "Vc": Vc,
            "fz": fz,
            "ap": ap,
            "Ra_mean": float(Ra_mean_i[0]),
            "Ra_sigma": float(Ra_sigma_i[0]),
            "Ra_upper": Ra_upper_i,
            "MRR": MRR_i,
            "feasible": Ra_upper_i <= Ra_target,
        })

    pareto_df = pd.DataFrame(pareto_rows)

    feasible_df = pareto_df[pareto_df["feasible"]]
    n_feasible = len(feasible_df)

    if n_feasible == 0:
        return OptimizationResult(
            method="NSGA-II",
            feasible=False,
            n_feasible=0,
            solve_time_s=elapsed,
            pareto_front=pareto_df,
        )

    best = feasible_df.sort_values("MRR", ascending=False).iloc[0]

    return OptimizationResult(
        method="NSGA-II",
        best_Vc=float(best["Vc"]),
        best_fz=float(best["fz"]),
        best_ap=float(best["ap"]),
        best_MRR=float(best["MRR"]),
        best_Ra_mean=float(best["Ra_mean"]),
        best_Ra_sigma=float(best["Ra_sigma"]),
        feasible=True,
        n_feasible=n_feasible,
        solve_time_s=elapsed,
        pareto_front=pareto_df,
    )


# ── Run all three ─────────────────────────────────────────────────────────────
def run_all_optimizers(
    ensemble: BootstrapEnsemble,
    VB: float,
    Ra_target: float,
    grid_config: Optional[dict] = None,
) -> dict:
    """
    Run all three optimizers and return results keyed by method name.

    Returns
    -------
    dict: {"Naive": OptimizationResult, "CP-SAT": OptimizationResult, "NSGA-II": OptimizationResult}
    """
    return {
        "Naive":   optimize_naive(ensemble, VB, Ra_target, grid_config),
        "CP-SAT":  optimize_cpsat(ensemble, VB, Ra_target, grid_config),
        "NSGA-II": optimize_nsgaii(ensemble, VB, Ra_target, grid_config),
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import load_all_datasets
    from surrogate_model import train_ensemble
    from pathlib import Path

    print("PJT MARS — Optimizer Test")
    print("=" * 50)

    df = load_all_datasets(Path("."))
    ensemble, _, _ = train_ensemble(df, n_bootstrap=20,
                                    gbr_params={"n_estimators": 200, "learning_rate": 0.05,
                                                "max_depth": 4, "subsample": 0.8,
                                                "min_samples_leaf": 5, "random_state": 0},
                                    verbose=False)

    VB, Ra_target = 0.1, 0.5
    print(f"\nRunning all optimizers: VB={VB}, Ra_target={Ra_target} µm\n")

    results = run_all_optimizers(ensemble, VB, Ra_target)

    for name, res in results.items():
        print(f"[{name}]")
        if res.feasible:
            print(f"  Vc={res.best_Vc:.1f}  fz={res.best_fz:.3f}  ap={res.best_ap:.2f}")
            print(f"  MRR={res.best_MRR:.4f}  Ra={res.best_Ra_mean:.4f}±{res.best_Ra_sigma:.4f}")
            print(f"  Ra_upper={res.best_Ra_mean + 2*res.best_Ra_sigma:.4f} ≤ {Ra_target}")
        else:
            print(f"  No feasible solution found.")
        print(f"  Time: {res.solve_time_s:.3f}s | Feasible candidates: {res.n_feasible}\n")

    # Consistency check: Naive and CP-SAT should agree on MRR
    if results["Naive"].feasible and results["CP-SAT"].feasible:
        diff = abs(results["Naive"].best_MRR - results["CP-SAT"].best_MRR)
        print(f"Naive vs CP-SAT MRR diff: {diff:.4f} ({'PASS' if diff < 0.01 else 'CHECK'})")
