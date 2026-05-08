import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

from training.config import ModelConfig
from training.model_cnn import train_cnn_model
from training.feature_extractor import get_cnn_sequences

logger = logging.getLogger(__name__)

def evaluate_cnn(model, X_test, y_test, device="cpu"):
    model.eval()
    X_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_torch).squeeze().cpu().numpy()
    
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    return {"auc": float(auc), "avg_precision": float(ap), "n_samples": len(y_test)}

def run_full_experiment(
    data_dir: str,
    cfg: Optional[ModelConfig] = None,
) -> Dict:
    """
    Load data, extract sequences, and train L3 and L2 CNN models.
    """
    if cfg is None:
        cfg = ModelConfig()

    data_dir = Path(data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading label index and runs …")
    label_idx = pd.read_parquet(data_dir / "label_index.parquet")
    run_ids = label_idx["run_id"].unique()
    
    # Shuffle and split runs
    np.random.default_rng(42).shuffle(run_ids)
    n_train = int(len(run_ids) * cfg.train_frac)
    train_ids = run_ids[:n_train]
    test_ids = run_ids[n_train:]

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    results = {}
    
    for mode in ["L2", "L3"]:
        logger.info(f"--- Processing {mode} ---")
        
        # Parallel extraction of sequences
        def load_set(ids):
            res = Parallel(n_jobs=-1)(
                delayed(get_cnn_sequences)(rid, data_dir, label_idx, mode=mode)
                for rid in ids
            )
            X = np.concatenate([r[0] for r in res if len(r[0]) > 0])
            y = np.concatenate([r[1] for r in res if len(r[1]) > 0])
            return X, y

        X_train, y_train = load_set(train_ids)
        X_test, y_test = load_set(test_ids)
        
        # Balanced undersampling
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        
        if n_pos > n_neg:
            pos_idx = np.random.choice(pos_idx, n_neg, replace=False)
        elif n_neg > n_pos:
            neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
            
        keep = np.sort(np.concatenate([pos_idx, neg_idx]))
        X_train, y_train = X_train[keep], y_train[keep]

        # Cap sizes for fast iteration across all hardware
        MAX_TRAIN = 10000
        MAX_TEST = 5000
        if len(X_train) > MAX_TRAIN:
            idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        if len(X_test) > MAX_TEST:
            idx = np.random.choice(len(X_test), MAX_TEST, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]


        logger.info(f"{mode} Samples: {len(X_train)} train (balanced), {len(X_test)} test")
        
        in_channels = X_train.shape[1]
        seq_len = X_train.shape[2]
        
        model = train_cnn_model(
            X_train, y_train, X_test, y_test,
            in_channels=in_channels, seq_len=seq_len,
            epochs=cfg.epochs, batch_size=cfg.batch_size, device=device
        )
        
        eval_res = evaluate_cnn(model, X_test, y_test, device=device)
        logger.info(f"{mode} Result: AUC={eval_res['auc']:.4f}")
        results[mode] = eval_res
        
        # Save model
        torch.save(model.state_dict(), out_dir / f"model_{mode}.pth")

    # Degradation
    auc_l3 = results["L3"]["auc"]
    auc_l2 = results["L2"]["auc"]
    degradation = (auc_l3 - auc_l2) / max(auc_l3, 1e-6)
    
    report = {
        "l3_eval": results["L3"],
        "l2_eval": results["L2"],
        "overall_degradation_pct": float(degradation * 100),
        "interpretation": f"L3 AUC={auc_l3:.4f} vs L2 AUC={auc_l2:.4f}. Degradation={degradation*100:.1f}%"
    }

    with open(out_dir / "degradation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT REPORT (CNN)")
    logger.info("=" * 60)
    logger.info(f"  AUC L3:     {auc_l3:.4f}")
    logger.info(f"  AUC L2:     {auc_l2:.4f}")
    logger.info(f"  Degradation: {degradation*100:.1f}%")
    logger.info("=" * 60)

    return report
