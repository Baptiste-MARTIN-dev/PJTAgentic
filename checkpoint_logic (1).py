"""
checkpoint_logic.py — Tool Condition Decision Engine for PJT MARS
=================================================================
Implements the lexicographic action rules based on flank wear VB and
solution feasibility. Quality constraint (Ra + 2σ ≤ Ra_target) is primary;
MRR is secondary.

Actions and thresholds:
  CONTINUE : VB < 0.1 mm AND feasible solution exists
  DERATE   : 0.1 ≤ VB < 0.2 mm  OR  (VB < 0.1 AND no feasible solution)
  INSPECT  : 0.2 ≤ VB < 0.3 mm AND feasible
  REPLACE  : VB ≥ 0.3 mm  OR  (0.2 ≤ VB < 0.3 AND infeasible)
"""

from dataclasses import dataclass
from typing import Optional

from surrogate_model import BootstrapEnsemble
from optimizers import OptimizationResult, optimize_naive


# ── Wear thresholds (single source of truth) ──────────────────────────────────
VB_DERATE  = 0.1   # mm — begin derating above this
VB_INSPECT = 0.2   # mm — inspection required above this
VB_REPLACE = 0.3   # mm — tool replacement mandatory


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class CheckpointDecision:
    """
    Result of the checkpoint evaluation.

    Attributes
    ----------
    action            : str — "CONTINUE", "DERATE", "INSPECT", or "REPLACE"
    VB                : float — input wear level [mm]
    feasible          : bool — whether any solution met Ra_target
    recommended_params: dict | None — {Vc, fz, ap, MRR, Ra_mean, Ra_sigma}
                        None when action is REPLACE or no feasible solution
    message           : str — human-readable explanation for the UI
    color             : str — Streamlit semantic color: "green"/"orange"/"red"
    n_feasible        : int — number of feasible candidates found
    """
    action: str
    VB: float
    feasible: bool
    recommended_params: Optional[dict]
    message: str
    color: str
    n_feasible: int = 0


# ── Main decision function ────────────────────────────────────────────────────
def evaluate_checkpoint(
    VB: float,
    ensemble: BootstrapEnsemble,
    Ra_target: float,
    Vc: Optional[float] = None,
    fz: Optional[float] = None,
    ap: Optional[float] = None,
) -> CheckpointDecision:
    """
    Evaluate tool condition and recommend action.

    Decision logic (lexicographic — quality constraint is primary):

        VB ≥ 0.3                    → REPLACE  (mandatory, regardless of quality)
        0.2 ≤ VB < 0.3, feasible    → INSPECT  (high wear, but still operable)
        0.2 ≤ VB < 0.3, infeasible  → REPLACE  (worn and cannot meet Ra_target)
        0.1 ≤ VB < 0.2              → DERATE   (moderate wear, reduce parameters)
        VB < 0.1, feasible           → CONTINUE (fresh tool, proceed normally)
        VB < 0.1, infeasible         → DERATE   (Ra_target too tight for current setup)

    Parameters
    ----------
    VB        : float, current flank wear [mm]
    ensemble  : trained BootstrapEnsemble
    Ra_target : float, surface roughness target [µm]
    Vc, fz, ap: optional current operating parameters (used only in message string)

    Returns
    -------
    CheckpointDecision dataclass
    """
    # Run naive optimizer to determine feasibility efficiently
    opt_result: OptimizationResult = optimize_naive(ensemble, VB, Ra_target)
    feasible = opt_result.feasible
    n_feasible = opt_result.n_feasible

    # Build recommended params dict from optimizer result
    if feasible:
        recommended = {
            "Vc": opt_result.best_Vc,
            "fz": opt_result.best_fz,
            "ap": opt_result.best_ap,
            "MRR": opt_result.best_MRR,
            "Ra_mean": opt_result.best_Ra_mean,
            "Ra_sigma": opt_result.best_Ra_sigma,
            "Ra_upper": opt_result.best_Ra_mean + 2.0 * opt_result.best_Ra_sigma,
        }
    else:
        recommended = None

    # Current params string for messages
    current_str = ""
    if Vc is not None and fz is not None and ap is not None:
        current_str = f" (current: Vc={Vc:.0f}, fz={fz:.3f}, ap={ap:.2f})"

    # ── Decision tree ─────────────────────────────────────────────────────────
    if VB >= VB_REPLACE:
        return CheckpointDecision(
            action="REPLACE",
            VB=VB,
            feasible=feasible,
            recommended_params=None,
            message=(
                f"Flank wear VB={VB:.2f} mm exceeds replacement threshold "
                f"({VB_REPLACE} mm). Tool must be replaced immediately."
                f"{current_str}"
            ),
            color="red",
            n_feasible=n_feasible,
        )

    if VB >= VB_INSPECT:  # 0.2 ≤ VB < 0.3
        if feasible:
            return CheckpointDecision(
                action="INSPECT",
                VB=VB,
                feasible=True,
                recommended_params=recommended,
                message=(
                    f"Flank wear VB={VB:.2f} mm is in the inspection zone "
                    f"({VB_INSPECT}–{VB_REPLACE} mm). A feasible solution exists "
                    f"({n_feasible} candidates). Schedule inspection soon and use "
                    f"recommended parameters.{current_str}"
                ),
                color="orange",
                n_feasible=n_feasible,
            )
        else:
            return CheckpointDecision(
                action="REPLACE",
                VB=VB,
                feasible=False,
                recommended_params=None,
                message=(
                    f"Flank wear VB={VB:.2f} mm is in the inspection zone but "
                    f"NO feasible solution meets Ra_target={Ra_target} µm. "
                    f"Replace tool immediately.{current_str}"
                ),
                color="red",
                n_feasible=0,
            )

    if VB >= VB_DERATE:  # 0.1 ≤ VB < 0.2
        return CheckpointDecision(
            action="DERATE",
            VB=VB,
            feasible=feasible,
            recommended_params=recommended,
            message=(
                f"Flank wear VB={VB:.2f} mm exceeds the derate threshold ({VB_DERATE} mm). "
                f"Reduce cutting parameters to maintain surface quality. "
                f"{'Recommended parameters available.' if feasible else f'WARNING: No feasible solution at Ra_target={Ra_target} µm — consider relaxing target.'}"
                f"{current_str}"
            ),
            color="orange",
            n_feasible=n_feasible,
        )

    # VB < 0.1 — fresh tool
    if feasible:
        return CheckpointDecision(
            action="CONTINUE",
            VB=VB,
            feasible=True,
            recommended_params=recommended,
            message=(
                f"Flank wear VB={VB:.2f} mm is within acceptable limits. "
                f"Continue machining with optimized parameters "
                f"(MRR = {recommended['MRR']:.3f}, Ra ≤ {recommended['Ra_upper']:.4f} µm).{current_str}"
            ),
            color="green",
            n_feasible=n_feasible,
        )
    else:
        return CheckpointDecision(
            action="DERATE",
            VB=VB,
            feasible=False,
            recommended_params=None,
            message=(
                f"Flank wear VB={VB:.2f} mm is low, but no feasible solution meets "
                f"Ra_target={Ra_target} µm at any combination in the candidate space. "
                f"Consider relaxing the Ra target or changing the tool grade.{current_str}"
            ),
            color="orange",
            n_feasible=0,
        )


def get_action_color_map() -> dict:
    """Return Streamlit semantic color mapping for all actions."""
    return {
        "CONTINUE": "green",
        "DERATE":   "orange",
        "INSPECT":  "orange",
        "REPLACE":  "red",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import load_all_datasets
    from surrogate_model import train_ensemble
    from pathlib import Path

    print("PJT MARS — Checkpoint Logic Test")
    print("=" * 50)

    df = load_all_datasets(Path("."))
    ensemble, _, _ = train_ensemble(df, n_bootstrap=20,
                                    gbr_params={"n_estimators": 200, "learning_rate": 0.05,
                                                "max_depth": 4, "subsample": 0.8,
                                                "min_samples_leaf": 5, "random_state": 0},
                                    verbose=False)

    test_cases = [
        (0.0,  0.5, "Fresh tool, reasonable target"),
        (0.05, 0.5, "Low wear, reasonable target"),
        (0.15, 0.5, "Moderate wear"),
        (0.25, 0.5, "High wear"),
        (0.30, 0.5, "At replacement threshold"),
        (0.0,  0.01, "Fresh tool, impossible target"),
    ]

    print(f"\n{'VB':>5}  {'Ra_tgt':>7}  {'Action':>10}  {'Feasible':>9}  Description")
    print("-" * 70)
    for VB, Ra_target, desc in test_cases:
        decision = evaluate_checkpoint(VB, ensemble, Ra_target)
        print(f"{VB:>5.2f}  {Ra_target:>7.3f}  {decision.action:>10}  "
              f"{str(decision.feasible):>9}  {desc}")
