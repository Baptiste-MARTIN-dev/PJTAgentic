"""
surrogate_model.py — Bootstrap Ensemble Surrogate for PJT MARS
==============================================================
GradientBoostingRegressor with 50-run Bootstrap Ensemble for Ra prediction
with uncertainty quantification (Ra_mean ± Ra_sigma).

Target: R² > 0.80 on hold-out test set.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ── Hyperparameters ───────────────────────────────────────────────────────────
DEFAULT_GBR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "random_state": 0,   # overridden per bootstrap run
}

FEATURE_NAMES = ["Vc", "fz", "ap", "VB"]


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class BootstrapEnsemble:
    """
    Container for the bootstrap ensemble of GradientBoostingRegressor models.

    Attributes
    ----------
    models        : list of trained GradientBoostingRegressor (len = n_bootstrap)
    feature_names : list[str], must be ["Vc", "fz", "ap", "VB"]
    n_bootstrap   : int, number of bootstrap models
    gbr_params    : dict, hyperparameters shared by all models
    train_r2      : float, R² on training split
    test_r2       : float, R² on hold-out test split
    """
    models: list = field(default_factory=list)
    feature_names: list = field(default_factory=lambda: FEATURE_NAMES.copy())
    n_bootstrap: int = 50
    gbr_params: dict = field(default_factory=dict)
    train_r2: float = float("nan")
    test_r2: float = float("nan")


# ── Training ──────────────────────────────────────────────────────────────────
def train_ensemble(
    df: pd.DataFrame,
    n_bootstrap: int = 50,
    test_size: float = 0.20,
    gbr_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Train the bootstrap ensemble on the combined dataset.

    Procedure:
    1. Hold-out split BEFORE bootstrapping (test set never seen during training)
    2. For each of n_bootstrap runs: resample train_df with replacement, fit GBR
    3. Compute ensemble mean prediction on both splits → compute R²
    4. Raise ValueError if test_r2 < 0.80

    Parameters
    ----------
    df          : pd.DataFrame from load_all_datasets(), columns [Vc,fz,ap,VB,Ra]
    n_bootstrap : int, number of bootstrap models
    test_size   : float, fraction for hold-out evaluation
    gbr_params  : dict, GBR hyperparameters (uses DEFAULT_GBR_PARAMS if None)
    seed        : int, base random seed
    verbose     : bool, print training progress

    Returns
    -------
    (BootstrapEnsemble, train_r2, test_r2)
    """
    if gbr_params is None:
        gbr_params = DEFAULT_GBR_PARAMS.copy()

    X = df[FEATURE_NAMES].values
    y = df["Ra"].values

    # 1. Hold-out split — done once, before any bootstrapping
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if verbose:
        print(f"Training split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # 2. Bootstrap training
    models = []
    train_preds = np.zeros((n_bootstrap, len(X_train)))
    test_preds = np.zeros((n_bootstrap, len(X_test)))

    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed + i)
        indices = rng.integers(0, len(X_train), size=len(X_train))
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        params = {**gbr_params, "random_state": seed + i}
        model = GradientBoostingRegressor(**params)
        model.fit(X_boot, y_boot)
        models.append(model)

        train_preds[i] = model.predict(X_train)
        test_preds[i] = model.predict(X_test)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap} complete")

    # 3. Compute R² on ensemble MEAN (not average of individual R² scores)
    Ra_mean_train = train_preds.mean(axis=0)
    Ra_mean_test = test_preds.mean(axis=0)

    train_r2 = r2_score(y_train, Ra_mean_train)
    test_r2 = r2_score(y_test, Ra_mean_test)

    if verbose:
        print(f"\nTrain R² : {train_r2:.4f}")
        print(f"Test  R² : {test_r2:.4f}")

    # 4. Enforce quality gate
    if test_r2 < 0.80:
        raise ValueError(
            f"Test R² = {test_r2:.4f} is below the required threshold of 0.80. "
            f"Try increasing n_estimators (current: {gbr_params.get('n_estimators', 300)}) "
            f"or n_bootstrap."
        )

    ensemble = BootstrapEnsemble(
        models=models,
        feature_names=FEATURE_NAMES.copy(),
        n_bootstrap=n_bootstrap,
        gbr_params=gbr_params,
        train_r2=train_r2,
        test_r2=test_r2,
    )

    return ensemble, train_r2, test_r2


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(
    ensemble: BootstrapEnsemble,
    X: pd.DataFrame,
) -> tuple:
    """
    Predict Ra with uncertainty from the bootstrap ensemble.

    Parameters
    ----------
    ensemble : trained BootstrapEnsemble
    X        : pd.DataFrame with columns ["Vc", "fz", "ap", "VB"]

    Returns
    -------
    (Ra_mean, Ra_sigma) : each np.ndarray of shape (n_samples,)
    Ra_mean  = mean of n_bootstrap individual predictions
    Ra_sigma = std  of n_bootstrap individual predictions
    """
    X_arr = X[ensemble.feature_names].values
    preds = np.stack([m.predict(X_arr) for m in ensemble.models], axis=0)
    Ra_mean = preds.mean(axis=0)
    Ra_sigma = preds.std(axis=0)
    return Ra_mean, Ra_sigma


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(ensemble: BootstrapEnsemble, path: Path) -> None:
    """Serialize ensemble with joblib. Recommended extension: .joblib"""
    joblib.dump(ensemble, path)
    print(f"Model saved → {path}")


def load_model(path: Path) -> BootstrapEnsemble:
    """Deserialize ensemble from a .joblib file."""
    ensemble = joblib.load(path)
    if not isinstance(ensemble, BootstrapEnsemble):
        raise TypeError(f"Loaded object is not a BootstrapEnsemble: {type(ensemble)}")
    return ensemble


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import load_all_datasets

    print("PJT MARS — Surrogate Model Training")
    print("=" * 50)

    df = load_all_datasets(Path("."))
    print(f"Loaded {len(df)} samples from 4 VB levels.\n")

    ensemble, tr2, te2 = train_ensemble(df, n_bootstrap=50, verbose=True)

    print(f"\nFinal Results:")
    print(f"  Train R² : {tr2:.4f}")
    print(f"  Test  R² : {te2:.4f}")
    print(f"  Status   : {'PASS' if te2 >= 0.80 else 'FAIL'} (threshold: 0.80)")

    # Quick prediction sanity check
    X_sample = pd.DataFrame({
        "Vc": [150.0, 300.0],
        "fz": [0.10, 0.30],
        "ap": [1.0, 3.0],
        "VB": [0.0, 0.3],
    })
    Ra_mean, Ra_sigma = predict(ensemble, X_sample)
    print(f"\nPrediction check (fresh tool vs worn tool):")
    for i, (mu, sigma) in enumerate(zip(Ra_mean, Ra_sigma)):
        print(f"  Sample {i+1}: Ra = {mu:.4f} ± {2*sigma:.4f} µm (95% CI)")

    save_model(ensemble, Path("ensemble_model.joblib"))
