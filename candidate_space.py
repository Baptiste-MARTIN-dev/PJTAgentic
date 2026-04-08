"""
candidate_space.py — Discrete Search Grid for PJT MARS
=======================================================
Generates the Cartesian product of cutting parameter values used by all
three optimizers (Naive, CP-SAT, NSGA-II).

Grid size (default): 11 × 6 × 6 = 396 candidates per VB level.
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Optional


# ── Default grid resolution ───────────────────────────────────────────────────
GRID_CONFIG = {
    "Vc_steps": np.arange(50, 301, 25),       # 11 values: 50, 75, ..., 300 m/min
    "fz_steps": np.arange(0.05, 0.31, 0.05),  # 6 values:  0.05, 0.10, ..., 0.30 mm/tooth
    "ap_steps": np.arange(0.5, 3.01, 0.5),    # 6 values:  0.5, 1.0, ..., 3.0 mm
}
# Total: 11 × 6 × 6 = 396 candidates


def compute_MRR(Vc, fz, ap, n_edges: int = 2):
    """
    Compute proportional Material Removal Rate.

    The absolute formula for a 20mm cutter:
        MRR_abs [cm³/min] = Vc * fz * ap * n_edges * (D * π / 1000)
                          = Vc * fz * ap * 2 * (20π/1000)
                          ≈ Vc * fz * ap * 0.12566

    For ranking purposes the constant factor cancels, so we use:
        MRR = Vc * fz * ap   [m·mm²/min, proportional]

    Parameters
    ----------
    Vc      : cutting speed [m/min]
    fz      : feed per tooth [mm/tooth]
    ap      : axial depth [mm]
    n_edges : number of cutting edges (default 2, not used in proportional form)

    Returns
    -------
    MRR : proportional MRR value
    """
    return np.asarray(Vc) * np.asarray(fz) * np.asarray(ap)


def build_candidate_space(
    VB: float,
    grid_config: Optional[dict] = None,
    custom_Vc: Optional[np.ndarray] = None,
    custom_fz: Optional[np.ndarray] = None,
    custom_ap: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build the full Cartesian product candidate space as a DataFrame.

    Parameters
    ----------
    VB          : float, current flank wear [mm] — added as a constant column
    grid_config : dict with keys Vc_steps, fz_steps, ap_steps
                  (uses GRID_CONFIG if None)
    custom_Vc   : optional override array for Vc values
    custom_fz   : optional override array for fz values
    custom_ap   : optional override array for ap values

    Returns
    -------
    pd.DataFrame with columns [Vc, fz, ap, VB, MRR]
    Sorted by MRR descending (highest MRR first — greedy order for Naive optimizer).
    """
    if grid_config is None:
        grid_config = GRID_CONFIG

    Vc_vals = custom_Vc if custom_Vc is not None else grid_config["Vc_steps"]
    fz_vals = custom_fz if custom_fz is not None else grid_config["fz_steps"]
    ap_vals = custom_ap if custom_ap is not None else grid_config["ap_steps"]

    rows = list(product(Vc_vals, fz_vals, ap_vals))
    df = pd.DataFrame(rows, columns=["Vc", "fz", "ap"])
    df["VB"] = float(VB)
    df["MRR"] = compute_MRR(df["Vc"].values, df["fz"].values, df["ap"].values)

    return df.sort_values("MRR", ascending=False).reset_index(drop=True)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PJT MARS — Candidate Space")
    print("=" * 50)

    df = build_candidate_space(VB=0.1)
    print(f"Grid size    : {len(df)} candidates")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nTop 5 by MRR (VB=0.1):")
    print(df.head())
    print(f"\nMRR range: [{df['MRR'].min():.4f}, {df['MRR'].max():.4f}]")
    print(f"Max MRR point: Vc={df.iloc[0]['Vc']}, fz={df.iloc[0]['fz']}, ap={df.iloc[0]['ap']}")
