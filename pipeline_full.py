#!/usr/bin/env python3
"""
mega_variant_pipeline_full.py

Consolidated variant filtering pipeline that merges the repeated logic from the
current project scripts into one reusable module.

Included functionality
----------------------
- VCF annotation extraction
- Truth-set labeling
- Feature selection and cleaning
- Missingness audit with configurable drop threshold
- Fold-aware (leakage-free) column-mean imputation
- Stratified cross-validation
- GM, BGM, LR, RF, LGB, and Bayesian-optimized LGB support
- Fold-level and averaged evaluation summaries
- Final model training
- Tranche analysis
- Per-variant classification output
- Unique TP/TN analysis across models
- Bootstrap confidence intervals for metrics (B=2000, percentile method)

Scientific rigor notes
----------------------
- Imputation is computed exclusively on the training partition of each fold
  and applied to the validation partition, preventing information leakage.
- Features with missingness exceeding MISSINGNESS_THRESHOLD (default 30%) are
  dropped prior to cross-validation and logged in missingness_audit.csv.
- All models share identical stratified splits (random_state fixed) for
  strict comparability.
- Bootstrap 95% CIs are reported for AUC, precision, recall, F1, and accuracy.

Notes
-----
- This file is designed to be publication-friendly and GitLab-ready.
- Paths are provided through CLI arguments, not hard-coded.
- LightGBM and scikit-optimize are optional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pysam
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False
    LGBMClassifier = None

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    BAYES_OPT_AVAILABLE = True
except Exception:
    BAYES_OPT_AVAILABLE = False
    BayesSearchCV = None
    Integer = Real = None


# =============================================================================
# Global scientific constants
# =============================================================================

# Features with missingness rate strictly above this threshold are excluded
# prior to cross-validation. Value of 0.30 means >30% missing → dropped.
MISSINGNESS_THRESHOLD: float = 0.30


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    input_vcf: str
    truth_vcf: str
    output_dir: str
    desired_features: List[str] = field(
        default_factory=lambda: [
            "MQ", "QD", "FS", "SOR", "MQRankSum", "ReadPosRankSum", "BaseQRankSum", "DP"
        ]
    )
    label_column: str = "TruthLabel"
    random_state: int = 42
    n_splits: int = 5
    missingness_threshold: float = MISSINGNESS_THRESHOLD
    gm_components: int = 5

    # -------------------------------------------------------------------------
    # GMM hyperparameters
    # GMM is fitted via the standard Expectation-Maximisation (EM) algorithm,
    # which alternates between two closed-form steps. Training stops as soon
    # as the per-iteration improvement in the observed log-likelihood falls
    # below gm_tol, or when gm_max_iter iterations are reached — whichever
    # comes first. gm_max_iter is therefore a safety ceiling, not the primary
    # stopping criterion.
    # -------------------------------------------------------------------------
    gm_max_iter: int = 500
    gm_tol: float = 1e-3  # Convergence tolerance for GMM (log-likelihood improvement)

    # -------------------------------------------------------------------------
    # BGM hyperparameters
    # BGM is fitted via variational Bayesian EM, which must additionally update
    # the posterior over the Dirichlet weight prior at every iteration. This
    # extra variational update makes each iteration more expensive and slows
    # overall convergence relative to standard EM, which is why a higher
    # iteration ceiling is used (1000 vs 500 for GMM). As with GMM, training
    # stops early if the per-iteration improvement in the evidence lower bound
    # (ELBO) falls below bgm_tol.
    # -------------------------------------------------------------------------
    bgm_max_iter: int = 1000
    bgm_tol: float = 1e-3  # Convergence tolerance for BGM (ELBO improvement)

    # -------------------------------------------------------------------------
    # BGM Dirichlet weight-concentration prior
    # A value much smaller than 1 is sparsity-inducing: it places most prior
    # mass on distributions where only a few components carry substantial
    # weight. Components with weak data support will have their mixing
    # coefficients shrunk toward zero, effectively reducing the active model
    # complexity and preventing the retention of uninformative ("phantom")
    # components.
    # -------------------------------------------------------------------------
    bgm_weight_concentration_prior: float = 1e-2

    bootstrap_iterations: int = 2000
    tranche_sensitivities: List[float] = field(default_factory=lambda: [1.0, 0.999, 0.99, 0.90])
    bayes_cv_splits: int = 3
    bayes_n_iter: int = 10
    lgb_n_estimators: int = 100
    rf_n_estimators: int = 50
    rf_max_depth: int = 10
    caller_name: Optional[str] = None
    dataset_name: Optional[str] = None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


# =============================================================================
# Project setup
# =============================================================================


DEFAULT_SUBDIRS = {
    "tables": "tables",
    "models": "models",
    "plots": "plots",
    "logs": "logs",
    "variants": "variants",
}


def initialize_project_environment(config: PipelineConfig) -> Dict[str, Path]:
    base = config.output_path
    base.mkdir(parents=True, exist_ok=True)

    paths = {name: base / subdir for name, subdir in DEFAULT_SUBDIRS.items()}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    log_file = paths["logs"] / "pipeline.log"

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    with open(base / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    logging.info("Initialized output environment at %s", base)
    return {**paths, "base": base, "log_file": log_file}


# =============================================================================
# VCF extraction and labeling
# =============================================================================


def extract_annotations_from_vcf(vcf_path: str, desired_features: Sequence[str]) -> pd.DataFrame:
    """Extract annotations from a VCF while tolerating missing INFO tags."""
    logging.info("Reading VCF: %s", vcf_path)
    vcf = pysam.VariantFile(vcf_path)

    available_info = set(vcf.header.info.keys())
    usable_features = [k for k in desired_features if k in available_info]
    missing_features = [k for k in desired_features if k not in available_info]

    if missing_features:
        logging.warning("Missing INFO tags in header; filling with NaN: %s", missing_features)

    records: List[Dict[str, Any]] = []
    for record in vcf:
        row: Dict[str, Any] = {
            "CHROM": record.chrom,
            "POS": record.pos,
            "REF": str(record.ref),
            "ALT": str(record.alts[0]) if record.alts else "N",
        }

        for key in usable_features:
            value = record.info.get(key, None)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            try:
                row[key] = float(value) if value is not None else np.nan
            except (TypeError, ValueError):
                row[key] = np.nan

        for key in missing_features:
            row[key] = np.nan

        records.append(row)

    df = pd.DataFrame(records)
    all_nan = [c for c in desired_features if c in df.columns and df[c].isna().all()]
    if all_nan:
        logging.warning("Columns that are entirely NaN: %s", all_nan)

    logging.info("Extracted %d variants", len(df))
    return df


def normalize_chromosome_name(chrom: Any) -> str:
    return str(chrom).replace("chr", "")


def extract_truth_positions(truth_vcf_path: str) -> set[Tuple[str, int]]:
    logging.info("Extracting truth positions from: %s", truth_vcf_path)
    truth_vcf = pysam.VariantFile(truth_vcf_path)
    truth_set: set[Tuple[str, int]] = set()
    for rec in truth_vcf:
        truth_set.add((normalize_chromosome_name(rec.chrom), int(rec.pos)))
    logging.info("Loaded %d truth positions", len(truth_set))
    return truth_set


def label_variants_against_truth(
    annotations_df: pd.DataFrame,
    truth_lookup: set[Tuple[str, int]],
    label_column: str,
) -> pd.DataFrame:
    logging.info("Labeling variants against truth set")
    df = annotations_df.copy()
    df["CHROM_NORM"] = df["CHROM"].map(normalize_chromosome_name)
    df[label_column] = [
        1 if (chrom, int(pos)) in truth_lookup else 0
        for chrom, pos in zip(df["CHROM_NORM"], df["POS"])
    ]
    return df.drop(columns=["CHROM_NORM"], errors="ignore")


# =============================================================================
# Feature preparation
# =============================================================================


def select_usable_features(df: pd.DataFrame, desired_features: Sequence[str]) -> List[str]:
    """
    Retain features that exist in the DataFrame and have at least one
    non-NaN value. Entirely absent or all-NaN columns are dropped here;
    the missingness threshold (>30%) is applied separately via
    audit_missingness().
    """
    usable = [c for c in desired_features if c in df.columns and df[c].notna().any()]
    dropped = [c for c in desired_features if c not in usable]
    logging.info("Initially selected features: %s", usable)
    if dropped:
        logging.warning("Dropped unavailable or entirely-NaN features: %s", dropped)
    if not usable:
        raise ValueError("No usable features were found.")
    return usable


def audit_missingness(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold: float = MISSINGNESS_THRESHOLD,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Compute per-feature missingness rates and drop features whose missing
    fraction exceeds `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled variant DataFrame (full dataset, before splitting).
    feature_columns : sequence of str
        Candidate feature names to audit.
    threshold : float
        Maximum tolerated missing fraction (default 0.30 → 30%).

    Returns
    -------
    retained : list of str
        Feature names that pass the threshold.
    audit_df : pd.DataFrame
        Audit table with columns: feature, n_total, n_missing,
        missing_pct, retained.
    """
    n_total = len(df)
    rows = []
    retained: List[str] = []

    for col in feature_columns:
        n_missing = int(df[col].isna().sum())
        rate = n_missing / n_total if n_total > 0 else 0.0
        keep = rate <= threshold
        rows.append(
            {
                "feature": col,
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_pct": round(rate * 100, 2),
                "retained": keep,
            }
        )
        if keep:
            retained.append(col)

    audit_df = pd.DataFrame(rows)
    dropped = audit_df[~audit_df["retained"]]["feature"].tolist()

    logging.info(
        "Missingness audit (threshold=%.0f%%):\n%s",
        threshold * 100,
        audit_df.to_string(index=False),
    )
    if dropped:
        logging.warning(
            "Excluded %d feature(s) exceeding %.0f%% missingness threshold: %s",
            len(dropped),
            threshold * 100,
            dropped,
        )
    if not retained:
        raise ValueError(
            f"All features were dropped by the missingness threshold ({threshold:.0%}). "
            "Consider raising --missingness_threshold."
        )

    logging.info("Retained features after missingness audit: %s", retained)
    return retained, audit_df


def clean_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    return_clean_df: bool = False,
) -> "np.ndarray | Tuple[pd.DataFrame, np.ndarray]":
    """
    Replace infinite values with NaN and convert to a float32 NumPy array.

    IMPORTANT — imputation is intentionally NOT performed here.
    Column-mean imputation is applied fold-by-fold inside
    cross_validation_evaluation() via fold_aware_impute() to prevent
    information leakage from the validation partition into the training
    imputation means.
    """
    cleaned = df.copy()
    cleaned[list(feature_columns)] = cleaned[list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    X = cleaned[list(feature_columns)].to_numpy(dtype=np.float32)
    return (cleaned, X) if return_clean_df else X


def fold_aware_impute(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leakage-free column-mean imputation.

    Column means are estimated exclusively from `X_train`. Those means
    are then applied to fill NaN values in both `X_train` and `X_val`.
    This ensures no statistical information from the validation partition
    influences the imputed values, in accordance with rigorous
    cross-validation methodology.

    If a column is entirely NaN in the training partition (edge case),
    the fallback imputation value is 0.0, and a warning is emitted.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training feature matrix, may contain NaN.
    X_val : np.ndarray, shape (n_val, n_features)
        Validation feature matrix, may contain NaN.

    Returns
    -------
    X_train_imp : np.ndarray
        Imputed training matrix.
    X_val_imp : np.ndarray
        Imputed validation matrix using training means only.
    """
    train_means = np.nanmean(X_train, axis=0)

    # Handle columns entirely NaN in training (edge case)
    all_nan_cols = np.where(np.isnan(train_means))[0]
    if len(all_nan_cols) > 0:
        logging.warning(
            "Column(s) %s are entirely NaN in this training fold; imputing with 0.0.",
            all_nan_cols.tolist(),
        )
        train_means = np.where(np.isnan(train_means), 0.0, train_means)

    def _apply_imputation(X: np.ndarray, means: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            if nan_mask.any():
                X_out[nan_mask, j] = means[j]
        return X_out

    X_train_imp = _apply_imputation(X_train, train_means)
    X_val_imp = _apply_imputation(X_val, train_means)
    return X_train_imp, X_val_imp


def impute_full_dataset(X: np.ndarray) -> np.ndarray:
    """
    Column-mean imputation on the full dataset.

    Used only when training final models on the complete labeled set
    (outside cross-validation). No leakage concern applies here as
    there is no held-out split.
    """
    means = np.nanmean(X, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        if nan_mask.any():
            X_out[nan_mask, j] = means[j]
    return X_out


def extract_target_vector(df: pd.DataFrame, label_column: str) -> np.ndarray:
    return df[label_column].to_numpy(dtype=int)


# =============================================================================
# Shared metrics and thresholds
# =============================================================================


def compute_vqslod(model_good: Any, model_bad: Any, X: np.ndarray) -> np.ndarray:
    return model_good.score_samples(X) - model_bad.score_samples(X)


def metric_from_counts(tp: int, fp: int, fn: int, tn: Optional[int] = None) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan
    accuracy = (
        (tp + (tn or 0)) / (tp + fp + fn + (tn or 0))
        if tn is not None and (tp + fp + fn + tn) > 0
        else np.nan
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def choose_threshold_from_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str = "youden",
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    if strategy != "youden":
        raise ValueError(f"Unsupported threshold strategy: {strategy}")
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_strategy: str = "youden",
) -> Dict[str, Any]:
    threshold = choose_threshold_from_roc(y_true, scores, strategy=threshold_strategy)
    predictions = (scores > threshold).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "optimal_threshold": float(threshold),
        "predictions": predictions,
    }
    return metrics


def find_threshold_at_fixed_sensitivity(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_sensitivity: float,
) -> float:
    positive_scores = np.sort(scores[y_true == 1])[::-1]
    if len(positive_scores) == 0:
        return float("nan")
    k = int(np.ceil(target_sensitivity * len(positive_scores))) - 1
    k = max(k, 0)
    return float(positive_scores[k])


def derive_status_labels(y_true: np.ndarray, y_pred: np.ndarray) -> List[str]:
    status: List[str] = []
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            status.append("TP")
        elif truth == 0 and pred == 1:
            status.append("FP")
        elif truth == 1 and pred == 0:
            status.append("FN")
        elif truth == 0 and pred == 0:
            status.append("TN")
        else:
            status.append("UNKNOWN")
    return status


# =============================================================================
# Model helpers
# =============================================================================


def build_model_registry(config: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {
        "GM": {"type": "mixture"},
        "BGM": {"type": "mixture"},
        "LR": {"type": "classifier"},
        "RF": {"type": "classifier"},
    }
    if LGBM_AVAILABLE:
        registry["LGB"] = {"type": "classifier"}
    if LGBM_AVAILABLE and BAYES_OPT_AVAILABLE:
        registry["LGB_Bayes"] = {"type": "classifier"}
    return registry


def bayesian_optimize_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    cv_splits: int = 3,
    n_iter: int = 10,
):
    if not LGBM_AVAILABLE or not BAYES_OPT_AVAILABLE:
        raise RuntimeError("Bayesian LightGBM optimization requires lightgbm and scikit-optimize.")

    estimator = LGBMClassifier(random_state=random_state, n_jobs=-1)
    search_spaces = {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 12),
        "learning_rate": Real(1e-3, 1e-1, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
    }
    optimizer = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        scoring="f1",
        n_iter=n_iter,
        cv=cv_splits,
        verbose=0,
        refit=True,
        random_state=random_state,
    )
    optimizer.fit(X, y)
    logging.info("Bayesian optimization best params: %s", optimizer.best_params_)
    return optimizer.best_estimator_


def fit_mixture_models(
    config: PipelineConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Fit GMM and BGM density estimators separately on positive (y=1) and
    negative (y=0) training variants.

    Both models are iterative and share the same stopping logic:
      - Training halts when the per-iteration improvement in the
        log-likelihood lower bound falls below the configured tolerance
        (gm_tol / bgm_tol), OR when the maximum iteration ceiling is
        reached — whichever comes first.
      - The iteration ceilings differ (GMM: gm_max_iter=500,
        BGM: bgm_max_iter=1000) because BGM uses variational Bayesian EM,
        which must additionally update the posterior over the Dirichlet
        weight prior at every step, making each iteration more expensive
        and overall convergence slower than the standard EM used for GMM.

    The BGM Dirichlet weight-concentration prior (bgm_weight_concentration_prior,
    default 1e-2) is sparsity-inducing: values << 1 shrink the mixing
    weights of weakly supported components toward zero, preventing the
    model from retaining uninformative ("phantom") components.
    """
    X_good = X_train[y_train == 1]
    X_bad = X_train[y_train == 0]
    if len(X_good) == 0 or len(X_bad) == 0:
        raise ValueError("Both positive and negative classes are required for mixture models.")

    # -- GMM: standard EM, converges in fewer iterations ---------------------
    gm_good = GaussianMixture(
        n_components=config.gm_components,
        max_iter=config.gm_max_iter,
        tol=config.gm_tol,
        random_state=config.random_state,
    ).fit(X_good)
    gm_bad = GaussianMixture(
        n_components=config.gm_components,
        max_iter=config.gm_max_iter,
        tol=config.gm_tol,
        random_state=config.random_state,
    ).fit(X_bad)

    # -- BGM: variational Bayesian EM, higher iteration ceiling warranted ----
    bgm_good = BayesianGaussianMixture(
        n_components=config.gm_components,
        max_iter=config.bgm_max_iter,
        tol=config.bgm_tol,
        weight_concentration_prior=config.bgm_weight_concentration_prior,
        random_state=config.random_state,
    ).fit(X_good)
    bgm_bad = BayesianGaussianMixture(
        n_components=config.gm_components,
        max_iter=config.bgm_max_iter,
        tol=config.bgm_tol,
        weight_concentration_prior=config.bgm_weight_concentration_prior,
        random_state=config.random_state,
    ).fit(X_bad)

    return {"GM": (gm_good, gm_bad), "BGM": (bgm_good, bgm_bad)}


def fit_classifier_models(
    config: PipelineConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "LR": LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
        ).fit(X_train, y_train),
        "RF": RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            max_features="sqrt",
            n_jobs=-1,
            random_state=config.random_state,
        ).fit(X_train, y_train),
    }

    if LGBM_AVAILABLE:
        models["LGB"] = LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            random_state=config.random_state,
        ).fit(X_train, y_train)

    if LGBM_AVAILABLE and BAYES_OPT_AVAILABLE:
        models["LGB_Bayes"] = bayesian_optimize_lightgbm(
            X_train,
            y_train,
            random_state=config.random_state,
            cv_splits=config.bayes_cv_splits,
            n_iter=config.bayes_n_iter,
        )

    return models


def train_models_on_training_data(
    config: PipelineConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit all models on a pre-imputed, pre-split training partition.
    X_train must already be imputed (via fold_aware_impute or
    impute_full_dataset) before calling this function.
    """
    models: Dict[str, Any] = {}
    models.update(fit_mixture_models(config, X_train, y_train))
    models.update(fit_classifier_models(config, X_train, y_train))
    return models


def score_model(model_name: str, model_obj: Any, X: np.ndarray) -> np.ndarray:
    if model_name in {"GM", "BGM"}:
        return compute_vqslod(model_obj[0], model_obj[1], X)
    return model_obj.predict_proba(X)[:, 1]


# =============================================================================
# Cross-validation
# =============================================================================


def cross_validation_evaluation(
    config: PipelineConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Stratified k-fold cross-validation with fold-aware imputation.

    For each fold:
      1. Split indices into train / validation.
      2. Apply fold_aware_impute() — means computed on training partition
         only, then applied to both partitions.
      3. Fit all models on the imputed training data.
      4. Evaluate on the imputed validation data.

    This design guarantees that no information from the validation
    partition leaks into preprocessing, consistent with rigorous
    cross-validation methodology.
    """
    logging.info(
        "Starting %d-fold stratified CV (random_state=%d)",
        config.n_splits,
        config.random_state,
    )
    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_records: List[Dict[str, Any]] = []
    metrics_all: Dict[str, List[Dict[str, Any]]] = {
        k: [] for k in build_model_registry(config).keys()
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        logging.info("Processing fold %d / %d", fold_idx, config.n_splits)

        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # ------------------------------------------------------------------ #
        # Fold-aware imputation: means derived from training partition only.  #
        # ------------------------------------------------------------------ #
        X_train, X_val = fold_aware_impute(X_train_raw, X_val_raw)

        fitted_models = train_models_on_training_data(config, X_train, y_train)

        for model_name, model_obj in fitted_models.items():
            scores = score_model(model_name, model_obj, X_val)
            metrics = compute_binary_metrics(y_val, scores, threshold_strategy="youden")
            record = {
                "fold": fold_idx,
                "model": model_name,
                "caller": config.caller_name,
                "dataset": config.dataset_name,
                "auc": metrics["auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "optimal_threshold": metrics["optimal_threshold"],
            }
            fold_records.append(record)
            metrics_all[model_name].append(record)
            logging.info(
                "Fold %d | %-12s | AUC=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  Acc=%.4f  Thr=%.4f",
                fold_idx,
                model_name,
                metrics["auc"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["accuracy"],
                metrics["optimal_threshold"],
            )

    fold_df = pd.DataFrame(fold_records)

    summary_records: List[Dict[str, Any]] = []
    for model_name, rows in metrics_all.items():
        if not rows:
            continue
        rdf = pd.DataFrame(rows)
        summary_records.append(
            {
                "caller": config.caller_name,
                "dataset": config.dataset_name,
                "model": model_name,
                "average_auc": rdf["auc"].mean(),
                "std_auc": rdf["auc"].std(),
                "average_precision": rdf["precision"].mean(),
                "std_precision": rdf["precision"].std(),
                "average_recall": rdf["recall"].mean(),
                "std_recall": rdf["recall"].std(),
                "average_f1": rdf["f1"].mean(),
                "std_f1": rdf["f1"].std(),
                "average_accuracy": rdf["accuracy"].mean(),
                "std_accuracy": rdf["accuracy"].std(),
                "average_optimal_threshold": rdf["optimal_threshold"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    return fold_df, summary_df, metrics_all


# =============================================================================
# Final models and persistence
# =============================================================================


def train_final_models(
    config: PipelineConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
) -> Dict[str, Any]:
    """
    Train final models on the complete dataset.

    Full-dataset column-mean imputation (impute_full_dataset) is applied
    here. No leakage concern applies because there is no held-out split.
    """
    logging.info("Training final models on full dataset (with full-dataset imputation)")
    X_imputed = impute_full_dataset(X_all)
    return train_models_on_training_data(config, X_imputed, y_all)


def save_models(models: Dict[str, Any], models_dir: Path) -> Dict[str, str]:
    model_paths: Dict[str, str] = {}
    for model_name, model_obj in models.items():
        if model_name in {"GM", "BGM"}:
            good_path = models_dir / f"{model_name.lower()}_good.pkl"
            bad_path = models_dir / f"{model_name.lower()}_bad.pkl"
            joblib.dump(model_obj[0], good_path)
            joblib.dump(model_obj[1], bad_path)
            model_paths[f"{model_name}_good"] = str(good_path)
            model_paths[f"{model_name}_bad"] = str(bad_path)
        else:
            model_path = models_dir / f"{model_name.lower()}.pkl"
            joblib.dump(model_obj, model_path)
            model_paths[model_name] = str(model_path)
    logging.info("Saved trained models")
    return model_paths


# =============================================================================
# Tranche analysis and plots
# =============================================================================


def run_tranche_analysis(
    config: PipelineConfig,
    models: Dict[str, Any],
    X_all: np.ndarray,
    y_all: np.ndarray,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model_name, model_obj in models.items():
        scores = score_model(model_name, model_obj, X_all)
        for sens in config.tranche_sensitivities:
            threshold = find_threshold_at_fixed_sensitivity(y_all, scores, sens)
            preds = (scores >= threshold).astype(int)
            precision = precision_score(y_all, preds, zero_division=0)
            recall = recall_score(y_all, preds, zero_division=0)
            f1 = f1_score(y_all, preds, zero_division=0)
            fdr = 1.0 - precision
            records.append(
                {
                    "caller": config.caller_name,
                    "dataset": config.dataset_name,
                    "model": model_name,
                    "sensitivity": sens,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fdr": fdr,
                }
            )
    return pd.DataFrame(records)


def plot_tranche_metrics(tranche_df: pd.DataFrame, plots_dir: Path) -> List[str]:
    saved: List[str] = []
    for metric in ["precision", "f1", "fdr"]:
        plt.figure(figsize=(8, 5))
        for model_name, g in tranche_df.groupby("model", dropna=False):
            gg = g.sort_values("sensitivity")
            plt.plot(gg["sensitivity"], gg[metric], marker="o", label=model_name)
        plt.xlabel("Sensitivity")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} at fixed sensitivity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = plots_dir / f"{metric}_vs_sensitivity.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        saved.append(str(out))
    return saved


def plot_roc_curves(
    models: Dict[str, Any],
    X_all: np.ndarray,
    y_all: np.ndarray,
    plots_dir: Path,
) -> str:
    plt.figure(figsize=(7, 6))
    for model_name, model_obj in models.items():
        scores = score_model(model_name, model_obj, X_all)
        fpr, tpr, _ = roc_curve(y_all, scores)
        auc = roc_auc_score(y_all, scores)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves on full dataset")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "roc_curves_full_data.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return str(out)


# =============================================================================
# Per-variant classification and unique correctness
# =============================================================================


def classify_variant_type(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "SNP"
    if len(ref) != len(alt):
        return "INDEL"
    return "OTHER"


def classify_variants_for_all_models(
    config: PipelineConfig,
    variants_df: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    models: Dict[str, Any],
    target_sensitivity: float = 0.99,
) -> pd.DataFrame:
    detailed_records: List[pd.DataFrame] = []
    base_cols = ["CHROM", "POS", "REF", "ALT", config.label_column]
    df = variants_df.copy()
    df["VariantType"] = [classify_variant_type(r, a) for r, a in zip(df["REF"], df["ALT"])]

    for model_name, model_obj in models.items():
        scores = score_model(model_name, model_obj, X_all)
        threshold = find_threshold_at_fixed_sensitivity(y_all, scores, target_sensitivity)
        preds = (scores >= threshold).astype(int)
        status = derive_status_labels(y_all, preds)

        temp = df[base_cols + ["VariantType"]].copy()
        temp["caller"] = config.caller_name
        temp["dataset"] = config.dataset_name
        temp["model"] = model_name
        temp["score"] = scores
        temp["threshold"] = threshold
        temp["prediction"] = preds
        temp["status"] = status
        temp["variant_id"] = (
            temp["CHROM"].astype(str)
            + ":"
            + temp["POS"].astype(str)
            + ":"
            + temp["REF"].astype(str)
            + ">"
            + temp["ALT"].astype(str)
        )
        detailed_records.append(temp)

    return pd.concat(detailed_records, ignore_index=True)


def summarize_variant_classification(classification_df: pd.DataFrame) -> pd.DataFrame:
    return (
        classification_df
        .groupby(
            ["caller", "dataset", "model", "status", "VariantType"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )


def analyze_unique_correct_variants(
    classification_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivoted_status = classification_df.pivot_table(
        index=["variant_id", "CHROM", "POS", "REF", "ALT", "VariantType", "TruthLabel"],
        columns="model",
        values="status",
        aggfunc="first",
    ).reset_index()

    meta_cols = ["variant_id", "CHROM", "POS", "REF", "ALT", "VariantType", "TruthLabel"]
    model_cols = [c for c in pivoted_status.columns if c not in meta_cols]

    long_records: List[pd.DataFrame] = []
    summary_records: List[Dict[str, Any]] = []

    for model in model_cols:
        others = [c for c in model_cols if c != model]

        unique_tp_mask = pivoted_status[model].eq("TP")
        unique_tn_mask = pivoted_status[model].eq("TN")
        for other in others:
            unique_tp_mask &= ~pivoted_status[other].eq("TP")
            unique_tn_mask &= ~pivoted_status[other].eq("TN")

        tp_df = pivoted_status.loc[unique_tp_mask, meta_cols].copy()
        tp_df["model"] = model
        tp_df["which"] = "unique_tp"

        tn_df = pivoted_status.loc[unique_tn_mask, meta_cols].copy()
        tn_df["model"] = model
        tn_df["which"] = "unique_tn"

        long_records.extend([tp_df, tn_df])
        summary_records.append(
            {
                "model": model,
                "unique_tp_count": len(tp_df),
                "unique_tn_count": len(tn_df),
            }
        )

    unique_long = pd.concat(long_records, ignore_index=True)
    unique_summary = pd.DataFrame(summary_records)
    return unique_long, unique_summary


# =============================================================================
# Bootstrap confidence intervals
# =============================================================================


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols = []
    for c in out.columns:
        nc = str(c).strip().replace(" ", "_").replace("-", "_").lower()
        while "__" in nc:
            nc = nc.replace("__", "_")
        new_cols.append(nc)
    out.columns = new_cols

    aliases = {
        "average_precision": "precision",
        "avg_precision": "precision",
        "mean_precision": "precision",
        "average_recall": "recall",
        "avg_recall": "recall",
        "mean_recall": "recall",
        "average_f1": "f1",
        "avg_f1": "f1",
        "mean_f1": "f1",
        "average_accuracy": "accuracy",
        "avg_accuracy": "accuracy",
        "mean_accuracy": "accuracy",
        "average_auc": "auc",
        "avg_auc": "auc",
        "mean_auc": "auc",
        "optimal_threshold": "threshold",
        "average_optimal_threshold": "threshold",
    }
    for src, tgt in aliases.items():
        if src in out.columns and tgt not in out.columns:
            out[tgt] = out[src]
    return out


def read_table_smart(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return _normalize_columns(pd.read_csv(path, sep="\t"))
    if ext == ".csv":
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            if len(df.columns) > 1:
                return _normalize_columns(df)
        except Exception:
            pass
        for sep in [";", "|", "\t", ","]:
            try:
                df = pd.read_csv(path, sep=sep)
                if len(df.columns) > 1:
                    return _normalize_columns(df)
            except Exception:
                continue
        return _normalize_columns(pd.read_csv(path))
    try:
        return _normalize_columns(pd.read_excel(path))
    except Exception:
        return _normalize_columns(pd.read_excel(path, engine="openpyxl"))


def _find_col_case_insensitive(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def safe_groupby(df: pd.DataFrame, group_cols: Sequence[str]):
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Grouping column '{col}' not found. Columns: {list(df.columns)}")
    return df.groupby(list(group_cols), dropna=False)


def bootstrap_confidence_intervals_from_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    n_boot: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Compute 95% bootstrap confidence intervals (percentile method) for
    each metric within each group defined by group_cols.

    Resampling strategy:
      - If fold-level counts (TP/FP/FN) are present: resample folds with
        replacement and aggregate counts per bootstrap replicate.
      - If variant-level status labels are present: resample variants with
        replacement and recompute confusion-matrix metrics per replicate.
      - Otherwise: resample rows (fold-level metric values) with
        replacement and take the column mean per replicate.

    Parameters
    ----------
    df : pd.DataFrame
        Input table (fold-level metrics or variant-level classifications).
    group_cols : sequence of str
        Columns to group by (e.g. ['model', 'caller']).
    metric_cols : sequence of str
        Metric names to bootstrap (e.g. ['auc', 'f1', 'precision']).
    n_boot : int
        Number of bootstrap replicates (default 2000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: group keys + ['metric', 'mean', 'ci95_lo', 'ci95_hi'].
    """
    df = _normalize_columns(df)
    rng = np.random.RandomState(seed)

    tp_col = _find_col_case_insensitive(df, "tp", "true_positive", "true_positives")
    fp_col = _find_col_case_insensitive(df, "fp", "false_positive", "false_positives")
    fn_col = _find_col_case_insensitive(df, "fn", "false_negative", "false_negatives")
    tn_col = _find_col_case_insensitive(df, "tn", "true_negative", "true_negatives")
    fold_col = _find_col_case_insensitive(df, "fold", "cv_fold")
    status_col = _find_col_case_insensitive(df, "status")

    have_counts = all(c is not None for c in [tp_col, fp_col, fn_col])
    have_fold = fold_col is not None
    have_variant = "variant_id" in df.columns or "locus" in df.columns

    need_metrics = not set(metric_cols).issubset(df.columns)
    if need_metrics and have_counts:
        def _row_metrics(row: pd.Series) -> pd.Series:
            tp = row[tp_col]
            fp = row[fp_col]
            fn = row[fn_col]
            tn = row[tn_col] if tn_col and tn_col in row else np.nan
            metrics = metric_from_counts(tp, fp, fn, tn if not pd.isna(tn) else None)
            return pd.Series(metrics)

        met = df.apply(_row_metrics, axis=1)
        for c in ["precision", "recall", "f1", "accuracy"]:
            if c not in df.columns and c in met.columns:
                df[c] = met[c]

    out_rows: List[Dict[str, Any]] = []

    for keys, g in safe_groupby(df, group_cols):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        context = dict(zip(group_cols, keys_tuple))

        if have_counts and have_fold and fold_col in g.columns:
            folds = g[fold_col].dropna().unique()
            boot_metrics: List[Dict[str, float]] = []
            for _ in range(n_boot):
                if len(folds) == 0:
                    break
                sel = rng.choice(folds, size=len(folds), replace=True)
                gg = pd.concat([g[g[fold_col] == f] for f in sel], ignore_index=True)
                tp = int(gg[tp_col].sum())
                fp = int(gg[fp_col].sum())
                fn = int(gg[fn_col].sum())
                tn = int(gg[tn_col].sum()) if tn_col and tn_col in gg.columns else None
                m = metric_from_counts(tp, fp, fn, tn)
                if "auc" in gg.columns:
                    m["auc"] = float(gg["auc"].mean())
                boot_metrics.append(m)

        elif have_variant and status_col and status_col in g.columns:
            statuses = g[status_col].astype(str).to_numpy()
            idx = np.arange(len(statuses))
            boot_metrics = []
            for _ in range(n_boot):
                sel = rng.choice(idx, size=len(idx), replace=True)
                sampled = statuses[sel]
                tp = int(np.sum(sampled == "TP"))
                fp = int(np.sum(sampled == "FP"))
                fn = int(np.sum(sampled == "FN"))
                tn = int(np.sum(sampled == "TN"))
                boot_metrics.append(metric_from_counts(tp, fp, fn, tn))

        else:
            present_metrics = [m for m in metric_cols if m in g.columns]
            if not present_metrics:
                raise ValueError(
                    f"No requested metric columns available for bootstrap in group {context}. "
                    f"Requested: {metric_cols}; columns: {list(g.columns)}"
                )
            values = g[present_metrics].to_numpy()
            n = len(values)
            boot_metrics = []
            for _ in range(n_boot):
                sel = rng.choice(np.arange(n), size=n, replace=True)
                boot_metrics.append(
                    dict(zip(present_metrics, np.nanmean(values[sel], axis=0)))
                )

        if not boot_metrics:
            continue

        bm = pd.DataFrame(boot_metrics)
        for metric in bm.columns:
            lo, hi = np.nanpercentile(bm[metric], [2.5, 97.5])
            out_rows.append(
                {
                    **context,
                    "metric": metric,
                    "mean": float(np.nanmean(bm[metric])),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
            )

    return pd.DataFrame(out_rows)


# =============================================================================
# Pipeline orchestration
# =============================================================================


def save_dataframe(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    logging.info("Saved %s", path)
    return str(path)


def run_full_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    End-to-end pipeline with full scientific rigor controls:

    1. Extract VCF annotations and label against truth set.
    2. Select initially usable features (non-empty columns).
    3. Audit missingness: drop features exceeding config.missingness_threshold.
    4. Convert to float32 NumPy array (inf → NaN; no imputation yet).
    5. Cross-validate with fold-aware imputation (no leakage).
    6. Train final models with full-dataset imputation.
    7. Tranche analysis, ROC plots, variant classification, bootstrap CIs.
    """
    started = time.time()
    paths = initialize_project_environment(config)

    # -- Data extraction and labeling ----------------------------------------
    annotations = extract_annotations_from_vcf(config.input_vcf, config.desired_features)
    truth_positions = extract_truth_positions(config.truth_vcf)
    labeled = label_variants_against_truth(annotations, truth_positions, config.label_column)

    # -- Feature selection and missingness audit ------------------------------
    feature_columns = select_usable_features(labeled, config.desired_features)
    feature_columns, missingness_audit = audit_missingness(
        labeled,
        feature_columns,
        threshold=config.missingness_threshold,
    )

    outputs: Dict[str, Any] = {"features": feature_columns}
    outputs["missingness_audit_csv"] = save_dataframe(
        missingness_audit,
        paths["tables"] / "missingness_audit.csv",
    )

    # -- Build feature matrix (inf→NaN; imputation deferred to fold level) ---
    labeled_clean, X_all = clean_feature_matrix(labeled, feature_columns, return_clean_df=True)
    y_all = extract_target_vector(labeled_clean, config.label_column)

    outputs["annotations_labeled_csv"] = save_dataframe(
        labeled_clean,
        paths["tables"] / "annotations_labeled.csv",
    )

    logging.info(
        "Dataset summary: %d variants | %d positives (%.1f%%) | %d negatives (%.1f%%)",
        len(y_all),
        int(y_all.sum()),
        100.0 * y_all.mean(),
        int((y_all == 0).sum()),
        100.0 * (1 - y_all.mean()),
    )

    # -- Cross-validation (fold-aware imputation inside) ----------------------
    fold_df, summary_df, _ = cross_validation_evaluation(config, X_all, y_all)
    outputs["cv_fold_metrics_csv"] = save_dataframe(
        fold_df,
        paths["tables"] / "cv_fold_metrics.csv",
    )
    outputs["cv_summary_csv"] = save_dataframe(
        summary_df,
        paths["tables"] / "cv_evaluation_summary.csv",
    )

    # -- Final models (full-dataset imputation) --------------------------------
    final_models = train_final_models(config, X_all, y_all)
    outputs["model_paths"] = save_models(final_models, paths["models"])

    # Need imputed X for downstream scoring
    X_all_imputed = impute_full_dataset(X_all)

    # -- Tranche analysis and plots -------------------------------------------
    tranche_df = run_tranche_analysis(config, final_models, X_all_imputed, y_all)
    outputs["tranche_metrics_csv"] = save_dataframe(
        tranche_df,
        paths["tables"] / "tranche_metrics.csv",
    )
    outputs["tranche_plots"] = plot_tranche_metrics(tranche_df, paths["plots"])
    outputs["roc_plot"] = plot_roc_curves(
        final_models, X_all_imputed, y_all, paths["plots"]
    )

    # -- Per-variant classification -------------------------------------------
    classification_df = classify_variants_for_all_models(
        config, labeled_clean, X_all_imputed, y_all, final_models
    )
    outputs["variant_classification_csv"] = save_dataframe(
        classification_df,
        paths["variants"] / "variant_classification_by_model.csv",
    )

    classification_summary = summarize_variant_classification(classification_df)
    outputs["variant_classification_summary_csv"] = save_dataframe(
        classification_summary,
        paths["variants"] / "variant_classification_summary.csv",
    )

    # -- Unique TP/TN analysis ------------------------------------------------
    unique_long, unique_summary = analyze_unique_correct_variants(classification_df)
    outputs["unique_tp_tn_long_csv"] = save_dataframe(
        unique_long,
        paths["variants"] / "unique_tp_tn_long.csv",
    )
    outputs["unique_tp_tn_summary_csv"] = save_dataframe(
        unique_summary,
        paths["variants"] / "unique_tp_tn_summary.csv",
    )

    # -- Bootstrap confidence intervals (B=2000, percentile method) -----------
    bootstrap_df = bootstrap_confidence_intervals_from_table(
        fold_df,
        group_cols=[c for c in ["caller", "dataset", "model"] if c in fold_df.columns],
        metric_cols=["precision", "recall", "f1", "auc", "accuracy"],
        n_boot=config.bootstrap_iterations,
        seed=config.random_state,
    )
    outputs["bootstrap_cis_csv"] = save_dataframe(
        bootstrap_df,
        paths["tables"] / "bootstrap_cis.csv",
    )

    elapsed = time.time() - started
    outputs["elapsed_seconds"] = elapsed
    logging.info("Pipeline completed successfully in %.2f seconds", elapsed)
    return outputs


# =============================================================================
# Optional McNemar utility
# =============================================================================


def mcnemar_test_from_status(
    a_status: np.ndarray,
    b_status: np.ndarray,
) -> Tuple[float, float, int, int]:
    a_correct = np.isin(a_status, ["TP", "TN"])
    b_correct = np.isin(b_status, ["TP", "TN"])
    b01 = int(np.sum((a_correct == True) & (b_correct == False)))
    b10 = int(np.sum((a_correct == False) & (b_correct == True)))
    if (b01 + b10) == 0:
        return 0.0, 1.0, b01, b10
    chi2_stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value), b01, b10


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidated variant filtering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_vcf", required=True, help="Input VCF path")
    parser.add_argument("--truth_vcf", required=True, help="Truth VCF path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--caller_name", default=None, help="Caller label for outputs")
    parser.add_argument("--dataset_name", default=None, help="Dataset label for outputs")
    parser.add_argument(
        "--desired_features",
        nargs="+",
        default=["MQ", "QD", "FS", "SOR", "MQRankSum", "ReadPosRankSum", "BaseQRankSum", "DP"],
        help="Feature names to use",
    )
    parser.add_argument("--label_column", default="TruthLabel")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--missingness_threshold",
        type=float,
        default=MISSINGNESS_THRESHOLD,
        help=(
            "Maximum fraction of missing values allowed per feature before "
            "it is excluded (default: 0.30 → 30%%)."
        ),
    )
    parser.add_argument("--gm_components", type=int, default=5)
    parser.add_argument(
        "--gm_max_iter",
        type=int,
        default=500,
        help="Maximum EM iterations for GMM (training stops earlier if gm_tol is reached).",
    )
    parser.add_argument(
        "--gm_tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for GMM (log-likelihood improvement per iteration).",
    )
    parser.add_argument(
        "--bgm_max_iter",
        type=int,
        default=1000,
        help=(
            "Maximum variational EM iterations for BGM. Higher than GMM because "
            "variational Bayesian inference converges more slowly than standard EM."
        ),
    )
    parser.add_argument(
        "--bgm_tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for BGM (ELBO improvement per iteration).",
    )
    parser.add_argument(
        "--bgm_weight_concentration_prior",
        type=float,
        default=1e-2,
        help=(
            "Dirichlet weight-concentration prior for BGM. Values << 1 are "
            "sparsity-inducing: weakly supported components shrink toward zero."
        ),
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument(
        "--tranche_sensitivities",
        nargs="+",
        type=float,
        default=[1.0, 0.999, 0.99, 0.90],
    )
    parser.add_argument("--bayes_cv_splits", type=int, default=3)
    parser.add_argument("--bayes_n_iter", type=int, default=10)
    parser.add_argument("--lgb_n_estimators", type=int, default=100)
    parser.add_argument("--rf_n_estimators", type=int, default=50)
    parser.add_argument("--rf_max_depth", type=int, default=10)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_vcf=args.input_vcf,
        truth_vcf=args.truth_vcf,
        output_dir=args.output_dir,
        desired_features=list(args.desired_features),
        label_column=args.label_column,
        random_state=args.random_state,
        n_splits=args.n_splits,
        missingness_threshold=args.missingness_threshold,
        gm_components=args.gm_components,
        gm_max_iter=args.gm_max_iter,
        gm_tol=args.gm_tol,
        bgm_max_iter=args.bgm_max_iter,
        bgm_tol=args.bgm_tol,
        bgm_weight_concentration_prior=args.bgm_weight_concentration_prior,
        bootstrap_iterations=args.bootstrap_iterations,
        tranche_sensitivities=list(args.tranche_sensitivities),
        bayes_cv_splits=args.bayes_cv_splits,
        bayes_n_iter=args.bayes_n_iter,
        lgb_n_estimators=args.lgb_n_estimators,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        caller_name=args.caller_name,
        dataset_name=args.dataset_name,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
