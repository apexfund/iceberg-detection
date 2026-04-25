"""
LightGBM-based iceberg detector training and degradation analysis.

Trains two identical models:
  - Model A on L3 features (full order-book information)
  - Model B on L2 features (aggregated depth only)

Then quantifies the information loss due to aggregation:
  degradation = (AUC_L3 - AUC_L2) / AUC_L3

Reported overall and per-regime, so analysts can see which market conditions
make L2-based detection hardest.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    lgb = None

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, classification_report,
    )
    from sklearn.model_selection import train_test_split
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from training.config import ModelConfig

logger = logging.getLogger(__name__)

# Columns that are metadata, not features
_META_COLS = {"run_id", "regime", "window_start", "label"}


def _split_by_run(df: pd.DataFrame, train_frac: float, val_frac: float,
                  seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at the run_id level so that no window from the same simulation run
    appears in both train and test sets.  This prevents temporal leakage.
    """
    run_ids = df["run_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)

    n = len(run_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(run_ids[:n_train])
    val_ids = set(run_ids[n_train:n_train + n_val])
    test_ids = set(run_ids[n_train + n_val:])

    return (
        df[df["run_id"].isin(train_ids)],
        df[df["run_id"].isin(val_ids)],
        df[df["run_id"].isin(test_ids)],
    )


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS]


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ModelConfig,
    feature_set_name: str = "",
) -> "lgb.Booster":
    assert _LGB_AVAILABLE, "lightgbm is not installed"

    feat_cols = _feature_cols(train_df)
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df["label"].values

    pos_weight = cfg.scale_pos_weight
    lgb_train = lgb.Dataset(X_train, label=y_train,
                             weight=np.where(y_train == 1, pos_weight, 1.0),
                             feature_name=feat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "feature_fraction": cfg.colsample_bytree,
        "bagging_fraction": cfg.subsample,
        "bagging_freq": 5,
        "min_child_samples": cfg.min_child_samples,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "n_jobs": cfg.n_jobs,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    logger.info(f"Training LightGBM [{feature_set_name}] on {len(X_train):,} samples, "
                f"{len(feat_cols)} features …")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=cfg.n_estimators,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    logger.info(f"  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(
    model: "lgb.Booster",
    test_df: pd.DataFrame,
    feature_set_name: str = "",
    regime_col: str = "regime",
) -> Dict:
    feat_cols = _feature_cols(test_df)
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df["label"].values
    probs = model.predict(X_test)

    results = {}
    results["auc"] = float(roc_auc_score(y_test, probs))
    results["avg_precision"] = float(average_precision_score(y_test, probs))
    results["positive_rate"] = float(y_test.mean())
    results["n_samples"] = int(len(y_test))
    results["feature_set"] = feature_set_name

    # Per-regime breakdown
    regime_results = {}
    for regime, grp in test_df.groupby(regime_col):
        idx = grp.index
        y_r = y_test[test_df.index.get_indexer(idx)]
        p_r = probs[test_df.index.get_indexer(idx)]
        if y_r.sum() == 0 or (1 - y_r).sum() == 0:
            continue
        regime_results[regime] = {
            "auc": float(roc_auc_score(y_r, p_r)),
            "avg_precision": float(average_precision_score(y_r, p_r)),
            "positive_rate": float(y_r.mean()),
            "n": int(len(y_r)),
        }
    results["by_regime"] = regime_results

    logger.info(
        f"[{feature_set_name}] AUC={results['auc']:.4f}  "
        f"AP={results['avg_precision']:.4f}  n={results['n_samples']:,}"
    )
    return results


def compute_degradation(l3_results: Dict, l2_results: Dict) -> Dict:
    """
    Compute degradation metrics comparing L3 vs L2 models.

    Degradation = (AUC_L3 - AUC_L2) / AUC_L3
    Expressed as percentage; higher = more information lost in aggregation.
    """
    auc_l3 = l3_results["auc"]
    auc_l2 = l2_results["auc"]

    overall_deg = (auc_l3 - auc_l2) / max(auc_l3, 1e-6)
    ap_deg = (l3_results["avg_precision"] - l2_results["avg_precision"]) / max(
        l3_results["avg_precision"], 1e-6)

    regime_deg = {}
    for regime in l3_results.get("by_regime", {}):
        if regime in l2_results.get("by_regime", {}):
            a3 = l3_results["by_regime"][regime]["auc"]
            a2 = l2_results["by_regime"][regime]["auc"]
            regime_deg[regime] = {
                "auc_l3": a3,
                "auc_l2": a2,
                "degradation_pct": float((a3 - a2) / max(a3, 1e-6) * 100),
            }

    return {
        "auc_l3": auc_l3,
        "auc_l2": auc_l2,
        "overall_degradation_pct": float(overall_deg * 100),
        "avg_precision_degradation_pct": float(ap_deg * 100),
        "by_regime": regime_deg,
        "interpretation": (
            f"L2 aggregation causes {overall_deg*100:.1f}% relative AUC degradation. "
            f"L3 AUC={auc_l3:.4f} vs L2 AUC={auc_l2:.4f}."
        ),
    }


def run_full_experiment(
    data_dir: str,
    cfg: Optional[ModelConfig] = None,
) -> Dict:
    """
    Load feature matrices, train L3, L2, and Hybrid models, return report.
    """
    if not _LGB_AVAILABLE:
        raise ImportError("lightgbm is required: pip install lightgbm")
    if not _SKLEARN:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    if cfg is None:
        cfg = ModelConfig()

    data_dir = Path(data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrices …")
    l3_df = pd.read_parquet(data_dir / "l3_features.parquet")
    l2_df = pd.read_parquet(data_dir / "l2_features.parquet")
    hybrid_df = pd.read_parquet(data_dir / "hybrid_features.parquet")

    logger.info(f"L3/L2/Hybrid rows: {len(l3_df):,}")
    logger.info(f"Label positive rate: {l3_df['label'].mean():.3f}")

    # Split by run_id to avoid data leakage
    l3_train, l3_val, l3_test = _split_by_run(l3_df, cfg.train_frac, cfg.val_frac)
    l2_train, l2_val, l2_test = _split_by_run(l2_df, cfg.train_frac, cfg.val_frac)
    hy_train, hy_val, hy_test = _split_by_run(hybrid_df, cfg.train_frac, cfg.val_frac)

    # Train models
    model_l3 = train_lgbm(l3_train, l3_val, cfg, feature_set_name="L3")
    model_l2 = train_lgbm(l2_train, l2_val, cfg, feature_set_name="L2")
    model_hy = train_lgbm(hy_train, hy_val, cfg, feature_set_name="Hybrid")

    # Save models
    model_l3.save_model(str(out_dir / "model_l3.txt"))
    model_l2.save_model(str(out_dir / "model_l2.txt"))
    model_hy.save_model(str(out_dir / "model_hybrid.txt"))

    # Evaluate on held-out test set
    l3_results = evaluate_model(model_l3, l3_test, feature_set_name="L3")
    l2_results = evaluate_model(model_l2, l2_test, feature_set_name="L2")
    hy_results = evaluate_model(model_hy, hy_test, feature_set_name="Hybrid")

    degradation = compute_degradation(l3_results, l2_results)
    # Also compare Hybrid to L3
    hybrid_gain = (hy_results["auc"] - l3_results["auc"]) / max(l3_results["auc"], 1e-6)

    # Feature importance
    hy_importance = pd.DataFrame({
        "feature": _feature_cols(hy_train),
        "importance": model_hy.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    hy_importance.to_csv(out_dir / "hybrid_feature_importance.csv", index=False)

    report = {
        "l3_eval": l3_results,
        "l2_eval": l2_results,
        "hybrid_eval": hy_results,
        "degradation": degradation,
        "hybrid_vs_l3_gain_pct": float(hybrid_gain * 100),
        "top_hybrid_features": hy_importance.head(15).to_dict(orient="records"),
    }

    with open(out_dir / "degradation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT REPORT")
    logger.info("=" * 60)
    logger.info(f"  AUC L3:     {l3_results['auc']:.4f}")
    logger.info(f"  AUC L2:     {l2_results['auc']:.4f}")
    logger.info(f"  AUC Hybrid: {hy_results['auc']:.4f}")
    logger.info("-" * 60)
    logger.info(f"  L2 Degradation: {degradation['overall_degradation_pct']:.1f}%")
    logger.info(f"  Hybrid Gain:    {hybrid_gain*100:+.1f}%")
    logger.info("=" * 60)

    return report

