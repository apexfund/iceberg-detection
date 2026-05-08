import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training.config import ModelConfig
from training.feature_extractor import get_cnn_sequences
from training.model_cnn import train_cnn_model

logger = logging.getLogger(__name__)


def evaluate_cnn(model, X_eval, y_eval, device="cpu", batch_size=512):
    from torch.utils.data import DataLoader, TensorDataset

    if len(X_eval) == 0:
        raise ValueError("Evaluation set is empty.")

    model.eval()
    ds = TensorDataset(torch.tensor(X_eval, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for (x,) in loader:
            batch_probs = model(x.to(device)).squeeze(-1).cpu().numpy()
            probs.append(np.atleast_1d(batch_probs))

    probs = np.concatenate(probs) if probs else np.empty((0,), dtype=np.float32)
    if len(np.unique(y_eval)) < 2:
        auc = 0.5
        ap = float(np.mean(y_eval))
    else:
        auc = roc_auc_score(y_eval, probs)
        ap = average_precision_score(y_eval, probs)
    return {"auc": float(auc), "avg_precision": float(ap), "n_samples": int(len(y_eval))}


def split_run_ids(
    run_ids: Sequence[str],
    label_idx: pd.DataFrame,
    cfg: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_ids = np.array(run_ids, copy=True)
    if len(run_ids) < 3:
        raise ValueError("At least 3 runs are required to create train/val/test splits.")

    if cfg.train_frac <= 0 or cfg.val_frac <= 0 or cfg.train_frac + cfg.val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1.")

    run_labels = (
        label_idx.groupby("run_id")["label"]
        .max()
        .reindex(run_ids)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    stratify = run_labels if len(np.unique(run_labels)) > 1 else None

    try:
        train_ids, temp_ids, train_labels, temp_labels = train_test_split(
            run_ids,
            run_labels,
            train_size=cfg.train_frac,
            random_state=cfg.split_seed,
            stratify=stratify,
        )
        val_share = cfg.val_frac / (1.0 - cfg.train_frac)
        temp_stratify = temp_labels if len(np.unique(temp_labels)) > 1 else None
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_share,
            random_state=cfg.split_seed,
            stratify=temp_stratify,
        )
    except ValueError:
        rng = np.random.default_rng(cfg.split_seed)
        rng.shuffle(run_ids)

        n_runs = len(run_ids)
        n_train = max(1, int(n_runs * cfg.train_frac))
        n_val = max(1, int(n_runs * cfg.val_frac))

        if n_train + n_val >= n_runs:
            n_val = max(1, min(n_val, n_runs - 2))
            n_train = max(1, min(n_train, n_runs - n_val - 1))

        train_ids = run_ids[:n_train]
        val_ids = run_ids[n_train:n_train + n_val]
        test_ids = run_ids[n_train + n_val:]

    if len(test_ids) == 0:
        test_ids = val_ids[-1:]
        val_ids = val_ids[:-1]

    if len(val_ids) == 0:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]

    if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
        raise ValueError("Unable to form non-empty train/val/test splits.")

    return train_ids, val_ids, test_ids


def load_sequence_set(
    ids: Sequence[str],
    data_dir: Path,
    label_idx: pd.DataFrame,
    mode: str,
    seq_len: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    channels = 9 if mode == "L3" else 4
    empty_X = np.empty((0, channels, seq_len), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)

    if len(ids) == 0:
        return empty_X, empty_y

    if n_jobs == 1:
        res = [get_cnn_sequences(rid, data_dir, label_idx, mode=mode, seq_len=seq_len) for rid in ids]
    else:
        try:
            res = Parallel(n_jobs=n_jobs)(
                delayed(get_cnn_sequences)(rid, data_dir, label_idx, mode=mode, seq_len=seq_len)
                for rid in ids
            )
        except PermissionError:
            logger.warning("Falling back to serial sequence extraction because process workers are unavailable.")
            res = [get_cnn_sequences(rid, data_dir, label_idx, mode=mode, seq_len=seq_len) for rid in ids]
    non_empty = [(X, y) for X, y in res if len(X) > 0]
    if not non_empty:
        return empty_X, empty_y

    X = np.concatenate([X for X, _ in non_empty], axis=0)
    y = np.concatenate([y for _, y in non_empty], axis=0)
    return X, y


def balance_binary_training_set(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Training set must contain both positive and negative labels.")

    rng = np.random.default_rng(seed)
    keep_size = min(len(pos_idx), len(neg_idx))
    pos_keep = rng.choice(pos_idx, keep_size, replace=False)
    neg_keep = rng.choice(neg_idx, keep_size, replace=False)
    keep = np.sort(np.concatenate([pos_keep, neg_keep]))
    return X[keep], y[keep]


def run_full_experiment(
    data_dir: str,
    cfg: Optional[ModelConfig] = None,
) -> Dict:
    """Load data, extract sequences, and train L3 and L2 CNN models."""
    if cfg is None:
        cfg = ModelConfig()

    data_dir = Path(data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading label index and runs ...")
    label_idx = pd.read_parquet(data_dir / "label_index.parquet")
    run_ids = label_idx["run_id"].unique()
    train_ids, val_ids, test_ids = split_run_ids(run_ids, label_idx, cfg)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(
        "Using device: %s | split sizes: train=%d val=%d test=%d",
        device,
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    results = {}

    for mode in ["L2", "L3"]:
        logger.info("--- Processing %s ---", mode)

        X_train, y_train = load_sequence_set(
            train_ids, data_dir, label_idx, mode=mode, seq_len=cfg.seq_len, n_jobs=cfg.n_jobs
        )
        X_val, y_val = load_sequence_set(
            val_ids, data_dir, label_idx, mode=mode, seq_len=cfg.seq_len, n_jobs=cfg.n_jobs
        )
        X_test, y_test = load_sequence_set(
            test_ids, data_dir, label_idx, mode=mode, seq_len=cfg.seq_len, n_jobs=cfg.n_jobs
        )

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError(f"{mode} produced an empty split; cannot train/evaluate.")

        X_train_balanced, y_train_balanced = balance_binary_training_set(
            X_train, y_train, seed=cfg.split_seed
        )

        logger.info(
            "%s samples: train=%d raw / %d balanced | val=%d | test=%d",
            mode,
            len(X_train),
            len(X_train_balanced),
            len(X_val),
            len(X_test),
        )

        model, history = train_cnn_model(
            X_train_balanced,
            y_train_balanced,
            X_val,
            y_val,
            in_channels=X_train_balanced.shape[1],
            seq_len=X_train_balanced.shape[2],
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.learning_rate,
            device=device,
        )

        val_eval = evaluate_cnn(model, X_val, y_val, device=device, batch_size=cfg.batch_size)
        test_eval = evaluate_cnn(model, X_test, y_test, device=device, batch_size=cfg.batch_size)
        logger.info(
            "%s result: val_auc=%.4f | test_auc=%.4f",
            mode,
            val_eval["auc"],
            test_eval["auc"],
        )

        results[mode] = {
            "val_eval": val_eval,
            "test_eval": test_eval,
            "training_history": history,
            "sample_counts": {
                "train_raw": int(len(X_train)),
                "train_balanced": int(len(X_train_balanced)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
        }

        torch.save(model.state_dict(), out_dir / f"model_{mode}.pth")

    auc_l3 = results["L3"]["test_eval"]["auc"]
    auc_l2 = results["L2"]["test_eval"]["auc"]
    degradation = (auc_l3 - auc_l2) / max(auc_l3, 1e-6)

    report = {
        "split_counts": {
            "train_runs": int(len(train_ids)),
            "val_runs": int(len(val_ids)),
            "test_runs": int(len(test_ids)),
        },
        "l3_eval": results["L3"]["test_eval"],
        "l2_eval": results["L2"]["test_eval"],
        "l3_val_eval": results["L3"]["val_eval"],
        "l2_val_eval": results["L2"]["val_eval"],
        "l3_training_history": results["L3"]["training_history"],
        "l2_training_history": results["L2"]["training_history"],
        "l3_sample_counts": results["L3"]["sample_counts"],
        "l2_sample_counts": results["L2"]["sample_counts"],
        "overall_degradation_pct": float(degradation * 100),
        "interpretation": (
            f"L3 test AUC={auc_l3:.4f} vs L2 test AUC={auc_l2:.4f}. "
            f"Degradation={degradation * 100:.1f}%"
        ),
    }

    with open(out_dir / "degradation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT REPORT (CNN)")
    logger.info("=" * 60)
    logger.info("  Test AUC L3:     %.4f", auc_l3)
    logger.info("  Test AUC L2:     %.4f", auc_l2)
    logger.info("  Degradation:     %.1f%%", degradation * 100)
    logger.info("=" * 60)

    return report
