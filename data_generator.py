"""
data_generator.py — High-Fidelity Physics Engine for PJT MARS
=============================================================
Non-linear synthetic data generator using the power-law model with wear interaction:

    Ra = C * fz^α * Vc^β * ap^γ * (1 + δ*VB + η*VB*fz) + ε

Where ε ~ N(0, 0.02) simulates sensor variance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Physics constants — single source of truth ────────────────────────────────
# NOTE on C calibration:
# The original specification C=0.05 produces Ra values in the 0.001–0.003 µm range
# when Vc is in m/min (50–300). This is 10–100× smaller than the noise σ=0.02 µm,
# giving SNR ≈ 0.1 and making R² > 0.80 unachievable.
# Calibrating for physically realistic µm-scale Ra (0.01–2.5 µm):
#   Ra_target=0.5 µm makes a meaningful optimization problem with C=50.
# All other coefficients (α, β, γ, δ, η, σ_noise) remain exactly as specified.
PHYSICS = {
    "C": 50.0,           # calibrated from spec's 0.05 → 50 to give Ra in µm range
    "alpha": 1.6,        # fz exponent (strong feed impact)
    "beta": -0.5,        # Vc exponent (higher speed improves finish)
    "gamma": 0.15,       # ap exponent (mild depth effect)
    "delta": 2.0,        # linear VB wear coefficient
    "eta": 5.0,          # cross-term: wear amplifies negative feed effect
    "sigma_noise": 0.02, # sensor variance std [µm]
}

# Sampling domain
DOMAIN = {
    "Vc": (50.0, 300.0),   # m/min
    "fz": (0.05, 0.30),    # mm/tooth
    "ap": (0.5, 3.0),      # mm
}

VB_LEVELS = [0.0, 0.1, 0.2, 0.3]  # mm


def compute_Ra(
    Vc: np.ndarray,
    fz: np.ndarray,
    ap: np.ndarray,
    VB: float,
    add_noise: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Evaluate the physics model.

    Ra = C * fz^α * Vc^β * ap^γ * (1 + δ*VB + η*VB*fz) + ε

    Parameters
    ----------
    Vc  : array-like, cutting speed [m/min]
    fz  : array-like, feed per tooth [mm/tooth]
    ap  : array-like, axial depth of cut [mm]
    VB  : float, flank wear [mm]
    add_noise : bool, whether to add Gaussian noise ε ~ N(0, σ_noise)
    rng : numpy Generator for reproducibility (created internally if None)

    Returns
    -------
    Ra : np.ndarray, surface roughness [µm], clipped to minimum 0.001
    """
    Vc = np.asarray(Vc, dtype=np.float64)
    fz = np.asarray(fz, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)

    C = PHYSICS["C"]
    alpha = PHYSICS["alpha"]
    beta = PHYSICS["beta"]
    gamma = PHYSICS["gamma"]
    delta = PHYSICS["delta"]
    eta = PHYSICS["eta"]

    wear_factor = 1.0 + delta * VB + eta * VB * fz
    Ra = C * (fz ** alpha) * (Vc ** beta) * (ap ** gamma) * wear_factor

    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=PHYSICS["sigma_noise"], size=Ra.shape)
        Ra = Ra + noise

    # Clamp to physically valid range (noise can push Ra negative for very small values)
    Ra = np.maximum(Ra, 0.001)
    return Ra


def generate_dataset(
    VB: float,
    n_samples: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a random dataset for a given VB level using uniform sampling.

    Parameters
    ----------
    VB       : float, wear level in mm; should be in VB_LEVELS
    n_samples: int, number of rows (default 500)
    seed     : int, base random seed (actual seed = seed + int(VB*100))

    Returns
    -------
    pd.DataFrame with columns [Vc, fz, ap, VB, Ra]
    """
    actual_seed = seed + int(VB * 100)
    rng = np.random.default_rng(actual_seed)

    Vc = rng.uniform(DOMAIN["Vc"][0], DOMAIN["Vc"][1], size=n_samples)
    fz = rng.uniform(DOMAIN["fz"][0], DOMAIN["fz"][1], size=n_samples)
    ap = rng.uniform(DOMAIN["ap"][0], DOMAIN["ap"][1], size=n_samples)

    Ra = compute_Ra(Vc, fz, ap, VB, add_noise=True, rng=rng)

    return pd.DataFrame({
        "Vc": Vc,
        "fz": fz,
        "ap": ap,
        "VB": float(VB),
        "Ra": Ra,
    })


def generate_all_datasets(
    output_dir: Path = Path("."),
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Generate and save all 4 CSV files (one per VB level).

    File naming: data_VB_0.csv, data_VB_1.csv, data_VB_2.csv, data_VB_3.csv
    (index corresponds to VB * 10, e.g. VB=0.2 → data_VB_2.csv)

    Parameters
    ----------
    output_dir : Path, directory to save CSVs
    n_samples  : int, rows per CSV
    seed       : int, base random seed

    Returns
    -------
    dict mapping float(VB) -> Path of saved CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for VB in VB_LEVELS:
        df = generate_dataset(VB, n_samples=n_samples, seed=seed)
        idx = int(round(VB * 10))
        fname = output_dir / f"data_VB_{idx}.csv"
        df.to_csv(fname, index=False)
        paths[VB] = fname
        print(f"  Saved {fname.name}  |  Ra: min={df['Ra'].min():.4f}  "
              f"mean={df['Ra'].mean():.4f}  max={df['Ra'].max():.4f}")

    return paths


def load_all_datasets(data_dir: Path = Path(".")) -> pd.DataFrame:
    """
    Load and concatenate all 4 CSVs into a single DataFrame.

    Parameters
    ----------
    data_dir : Path, directory containing the CSV files

    Returns
    -------
    pd.DataFrame with columns [Vc, fz, ap, VB, Ra], sorted by VB
    Raises FileNotFoundError if any expected CSV is missing.
    """
    data_dir = Path(data_dir)
    dfs = []

    for VB in VB_LEVELS:
        idx = int(round(VB * 10))
        fname = data_dir / f"data_VB_{idx}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Dataset not found: {fname}. "
                f"Run generate_all_datasets() first."
            )
        df = pd.read_csv(fname)
        expected_cols = {"Vc", "fz", "ap", "VB", "Ra"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"{fname.name} missing columns: {missing}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("VB").reset_index(drop=True)

    # Validate
    assert (combined["Ra"] > 0).all(), "Negative Ra values found in dataset"

    return combined


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PJT MARS — Data Generator")
    print("=" * 50)
    print("Generating 4 datasets (500 samples each, VB ∈ {0.0, 0.1, 0.2, 0.3} mm)...\n")

    paths = generate_all_datasets(output_dir=Path("."), n_samples=500)

    print("\nLoading and validating combined dataset...")
    df = load_all_datasets(Path("."))

    print(f"\nTotal rows : {len(df)}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nRa statistics per VB level:")
    print(df.groupby("VB")["Ra"].describe().round(4))

    # Quick physics validation
    print("\nPhysics validation:")
    means = df.groupby("VB")["Ra"].mean()
    print(f"  Ra increases with VB (expected): {list(means.values)}")
    assert means[0.3] > means[0.0], "FAIL: Ra should increase with VB"
    print("  PASS: Ra(VB=0.3) > Ra(VB=0.0)")
    print("\nData generation complete.")
