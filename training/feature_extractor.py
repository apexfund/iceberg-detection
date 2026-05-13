"""Window-based feature extraction for L3 and L2 data."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────────────

def build_label_index(
    gt_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
    window_s: float,
    step_s: float,
) -> pd.DataFrame:
    TICK = 0.02
    records = []

    for run_id, run_snaps in snaps_df.groupby("run_id"):
        run_snaps = run_snaps.sort_values("timestamp")
        run_gt = gt_df[gt_df["run_id"] == run_id]
        regime = run_snaps["regime"].iloc[0]

        snap_times = run_snaps["timestamp"].values
        best_bid_arr = run_snaps["best_bid"].fillna(-9999).values
        best_ask_arr = run_snaps["best_ask"].fillna(9999).values
        S = len(snap_times)

        snap_active = np.zeros(S, dtype=bool)
        if not run_gt.empty:
            end_times = run_gt["end_time"].fillna(np.inf).values
            start_times = run_gt["start_time"].values
            prices = run_gt["price"].values
            sides = run_gt["side"].values

            alive = (
                (start_times[:, None] <= snap_times[None, :]) &
                (end_times[:, None] > snap_times[None, :])
            )
            buy_mask = (sides == "BUY")[:, None]
            near_bid = np.abs(prices[:, None] - best_bid_arr[None, :]) <= TICK
            near_ask = np.abs(prices[:, None] - best_ask_arr[None, :]) <= TICK
            at_best = alive & ((buy_mask & near_bid) | (~buy_mask & near_ask))
            snap_active = at_best.any(axis=0)

        t_min, t_max = snap_times[0], snap_times[-1]
        windows = np.arange(t_min, t_max - window_s + step_s * 0.5, step_s)

        for w_start in windows:
            w_end = w_start + window_s
            mask = (snap_times >= w_start) & (snap_times < w_end)
            if not mask.any(): continue
            label = 1 if snap_active[mask].mean() >= 0.4 else 0
            records.append({
                "run_id": run_id, "regime": regime,
                "window_start": float(w_start), "window_end": float(w_end),
                "label": label,
            })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction (Sequence-based for CNN)
# ─────────────────────────────────────────────────────────────────────────────

def get_cnn_sequences(
    rid: str,
    data_dir: Path,
    label_idx: pd.DataFrame,
    mode: str = "L3",
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    filters = [("run_id", "==", rid)]
    run_label = label_idx[label_idx["run_id"] == rid]
    snaps = pd.read_parquet(data_dir / "l2_snapshots.parquet", filters=filters).sort_values("timestamp")
    
    if mode == "L3":
        orders = pd.read_parquet(data_dir / "l3_orders.parquet", filters=filters)
        trades = pd.read_parquet(data_dir / "l3_trades.parquet", filters=filters)
        side_map = {"BUY": 1.0, "SELL": -1.0}
        
        ord_s = orders.copy()
        ord_s["side"] = orders["side"].map(side_map).fillna(0.0)
        ord_s["type"] = 1.0
        trd_s = trades.copy()
        trd_s["side"] = trades["aggressor_side"].map(side_map).fillna(0.0)
        trd_s["type"] = 2.0
        
        events = pd.concat([ord_s, trd_s]).sort_values("timestamp")
        
        # Bin L3 events into the same 0.05s snapshot intervals to provide uniform time-series to the CNN
        events = pd.merge_asof(events, snaps[["timestamp", "best_bid", "best_ask", "bid_qty_1", "ask_qty_1"]], on="timestamp", direction="forward")
        
        def calc_modal_freq(x):
            if len(x) == 0: return 0.0
            return x.value_counts().max() / len(x)
            
        agg_events = events.groupby("timestamp").agg(
            event_count=("type", "count"),
            trade_count=("type", lambda x: (x == 2.0).sum()),
            qty_sum=("quantity", "sum"),
            qty_max=("quantity", "max"),
            modal_freq=("quantity", calc_modal_freq)
        ).reset_index()
        
        l3_df = pd.merge(snaps[["timestamp", "best_bid", "best_ask", "bid_qty_1", "ask_qty_1"]], agg_events, on="timestamp", how="left").fillna(0.0)
        
        # Features: [best_bid, best_ask, bid_qty_1, ask_qty_1, event_count, trade_count, qty_sum, qty_max, modal_freq] (9 channels)
        run_data = l3_df[["timestamp", "best_bid", "best_ask", "bid_qty_1", "ask_qty_1", "event_count", "trade_count", "qty_sum", "qty_max", "modal_freq"]].values.astype(np.float32)
    else:
        # L2 features: [best_bid, best_ask, bid_qty_1, ask_qty_1]
        run_data = snaps[["timestamp", "best_bid", "best_ask", "bid_qty_1", "ask_qty_1"]].fillna(0).values.astype(np.float32)

    X, y = [], []
    for _, lrow in run_label.iterrows():
        t = lrow["window_start"]
        idx = np.searchsorted(run_data[:, 0], t)
        if idx < seq_len: continue
        
        seq = run_data[idx-seq_len:idx, 1:].copy()
        
        if mode == "L3":
            # Normalize constructed L2 data inside L3
            mid = (seq[:, 0] + seq[:, 1]) / 2.0
            seq[:, 0] = (seq[:, 0] - mid) / 0.1
            seq[:, 1] = (seq[:, 1] - mid) / 0.1
            seq[:, 2] = np.log1p(seq[:, 2]) / 6.0
            seq[:, 3] = np.log1p(seq[:, 3]) / 6.0
            # Normalize LightGBM microstructural aggregates
            seq[:, 4] = np.log1p(seq[:, 4]) / 4.0 # event_count
            seq[:, 5] = np.log1p(seq[:, 5]) / 4.0 # trade_count
            seq[:, 6] = np.log1p(seq[:, 6]) / 8.0 # qty_sum
            seq[:, 7] = np.log1p(seq[:, 7]) / 6.0 # qty_max
            # seq[:, 8] (modal_freq) is already bounded [0, 1]
        else:
            # L2 normalization
            mid = (seq[:, 0] + seq[:, 1]) / 2.0
            seq[:, 0] = (seq[:, 0] - mid) / 0.1
            seq[:, 1] = (seq[:, 1] - mid) / 0.1
            seq[:, 2] = np.log1p(seq[:, 2]) / 6.0
            seq[:, 3] = np.log1p(seq[:, 3]) / 6.0

        if not np.isfinite(seq).all(): continue
        X.append(seq.T)
        y.append(lrow["label"])
        
    if not X:
        if mode == "L3":
            return np.empty((0, 9, seq_len), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.empty((0, 4, seq_len), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def build_feature_matrices(data_dir: str, window_s: float = 0.3, step_s: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    snaps_df = pd.read_parquet(data_dir / "l2_snapshots.parquet")
    gt_df = pd.read_parquet(data_dir / "iceberg_gt.parquet")
    label_idx = build_label_index(gt_df, snaps_df, window_s, step_s)
    label_idx.to_parquet(data_dir / "label_index.parquet", index=False)
    return label_idx, label_idx, label_idx
