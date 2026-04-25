"""
Window-based feature extraction for L3 and L2 data.

Produces aligned feature matrices where each row covers a fixed time window
[t, t + window_s].  The SAME label is attached to both the L3 and L2 row for
that window — the only difference between the two matrices is the information
available to compute features.

Label definition:
  label = 1  if any iceberg chain was active at best bid OR best ask at any
              point during the window.
  label = 0  otherwise.

L3 features exploit order-level information (order IDs, individual sizes,
fill patterns, repeated-quantity fingerprints).

L2 features use only aggregated depth snapshots — no order counts, no IDs,
no exact fill timestamps.  Leakage prevention is structural: the L2 feature
function receives only the snapshots DataFrame, never the orders DataFrame.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
    """
    For every (run_id, window_start) pair, compute:
      - label: 1 if an iceberg was active at best bid/ask during [t, t+window_s]
      - regime, run_id, window_start, window_end

    Vectorised per-run using numpy broadcasting.
    An iceberg is "active at best" if its price is within 2 ticks of best bid
    (BUY) or best ask (SELL) at any snapshot in the window.
    """
    TICK = 0.02  # 2-tick tolerance
    records = []

    for run_id, run_snaps in snaps_df.groupby("run_id"):
        run_snaps = run_snaps.sort_values("timestamp")
        run_gt = gt_df[gt_df["run_id"] == run_id]
        regime = run_snaps["regime"].iloc[0]

        snap_times = run_snaps["timestamp"].values          # (S,)
        best_bid_arr = run_snaps["best_bid"].fillna(-9999).values   # (S,)
        best_ask_arr = run_snaps["best_ask"].fillna(9999).values    # (S,)
        S = len(snap_times)

        # ── Vectorised active-at-best flag per snapshot ───────────────────
        snap_active = np.zeros(S, dtype=bool)

        if not run_gt.empty:
            end_times = run_gt["end_time"].fillna(np.inf).values     # (C,)
            start_times = run_gt["start_time"].values                # (C,)
            prices = run_gt["price"].values                          # (C,)
            sides = run_gt["side"].values                            # (C,)

            # active[c, s] = chain c is alive at snapshot s
            # shape: (C, S)
            alive = (
                (start_times[:, None] <= snap_times[None, :]) &
                (end_times[:, None] > snap_times[None, :])
            )

            # price proximity: buy iceberg near best_bid, sell near best_ask
            buy_mask = (sides == "BUY")[:, None]      # (C, 1)
            sell_mask = ~buy_mask

            near_bid = np.abs(prices[:, None] - best_bid_arr[None, :]) <= TICK
            near_ask = np.abs(prices[:, None] - best_ask_arr[None, :]) <= TICK

            at_best = alive & ((buy_mask & near_bid) | (sell_mask & near_ask))
            # snap_active[s] = any chain is at-best at snapshot s
            snap_active = at_best.any(axis=0)   # (S,)

        # ── Assign each snapshot to all windows that cover it ──────────────
        t_min = snap_times[0]
        t_max = snap_times[-1]
        windows = np.arange(t_min, t_max - window_s + step_s * 0.5, step_s)

        for w_start in windows:
            w_end = w_start + window_s
            mask = (snap_times >= w_start) & (snap_times < w_end)
            if not mask.any():
                continue
            
            # Stricter label: Iceberg must be at-best for at least 40% of snapshots in window
            label = 1 if snap_active[mask].mean() >= 0.4 else 0
            
            records.append({
                "run_id": run_id,
                "regime": regime,
                "window_start": float(w_start),
                "window_end": float(w_end),
                "label": label,
            })

    return pd.DataFrame(records)



# ─────────────────────────────────────────────────────────────────────────────
# L3 feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _l3_window_features(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    w_start: float,
    w_end: float,
) -> Dict:
    o = orders[(orders["timestamp"] >= w_start) & (orders["timestamp"] < w_end)]
    t = trades[(trades["timestamp"] >= w_start) & (trades["timestamp"] < w_end)]

    feat: Dict = {}

    # ── Volume and flow ──────────────────────────────────────────────────────
    buy_o = o[o["side"] == "BUY"]
    sell_o = o[o["side"] == "SELL"]
    feat["n_orders"] = len(o)
    feat["n_trades"] = len(t)
    feat["buy_vol"] = float(buy_o["quantity"].sum())
    feat["sell_vol"] = float(sell_o["quantity"].sum())
    total_vol = feat["buy_vol"] + feat["sell_vol"] + 1e-6
    feat["net_flow_norm"] = (feat["buy_vol"] - feat["sell_vol"]) / total_vol
    feat["trade_vol"] = float(t["quantity"].sum())
    feat["trade_buy_vol"] = float(t[t["aggressor_side"] == "BUY"]["quantity"].sum()) if len(t) > 0 else 0.0
    feat["tick_imbalance"] = (feat["trade_buy_vol"] / (feat["trade_vol"] + 1e-6)) - 0.5

    # ── Order size distribution ──────────────────────────────────────────────
    if len(o) > 1:
        sizes = o["quantity"].values.astype(float)
        feat["avg_order_size"] = float(sizes.mean())
        feat["std_order_size"] = float(sizes.std())
        feat["max_order_size"] = float(sizes.max())
        feat["order_size_cv"] = float(sizes.std() / (sizes.mean() + 1e-6))
        # Fraction of sizes that appear ≥2 times (iceberg repeated visible qty)
        counts = pd.Series(sizes.astype(int)).value_counts()
        feat["repeat_size_frac"] = float((counts >= 2).sum() / len(counts))
        feat["modal_size_freq"] = float(counts.iloc[0] / len(o))
    else:
        feat.update({k: 0.0 for k in [
            "avg_order_size", "std_order_size", "max_order_size",
            "order_size_cv", "repeat_size_frac", "modal_size_freq",
        ]})

    # ── Fill behaviour ───────────────────────────────────────────────────────
    if len(o) > 0:
        fill_rates = o["filled_qty"] / (o["quantity"] + 1e-6)
        feat["fill_rate_mean"] = float(fill_rates.mean())
        feat["fill_rate_std"] = float(fill_rates.std()) if len(o) > 1 else 0.0
        fully_filled = (o["filled_qty"] >= o["quantity"]).sum()
        feat["fully_filled_frac"] = float(fully_filled / len(o))
    else:
        feat.update({"fill_rate_mean": 0.0, "fill_rate_std": 0.0, "fully_filled_frac": 0.0})

    # ── Trade size distribution ──────────────────────────────────────────────
    if len(t) > 1:
        tsizes = t["quantity"].values.astype(float)
        feat["avg_trade_size"] = float(tsizes.mean())
        feat["std_trade_size"] = float(tsizes.std())
        t_counts = pd.Series(tsizes.astype(int)).value_counts()
        feat["repeat_trade_size_frac"] = float((t_counts >= 2).sum() / len(t_counts))
        # Regularity: low CV of trade size suggests iceberg absorbing equal chunks
        feat["trade_size_cv"] = float(tsizes.std() / (tsizes.mean() + 1e-6))
    else:
        feat.update({k: 0.0 for k in [
            "avg_trade_size", "std_trade_size",
            "repeat_trade_size_frac", "trade_size_cv",
        ]})

    # ── Price dynamics ───────────────────────────────────────────────────────
    if len(t) > 1:
        feat["price_range"] = float(t["price"].max() - t["price"].min())
        feat["price_std"] = float(t["price"].std())
    else:
        feat["price_range"] = 0.0
        feat["price_std"] = 0.0

    return feat


def _extract_l3_for_run(
    run_id: str,
    run_label: pd.DataFrame,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
) -> List[Dict]:
    o_ts = orders["timestamp"].values if len(orders) > 0 else np.array([])
    t_ts = trades["timestamp"].values if len(trades) > 0 else np.array([])
    rows = []
    for _, lrow in run_label.iterrows():
        ws, we = lrow["window_start"], lrow["window_end"]
        o_lo, o_hi = np.searchsorted(o_ts, [ws, we])
        t_lo, t_hi = np.searchsorted(t_ts, [ws, we])
        feat = _l3_window_features(orders.iloc[o_lo:o_hi], trades.iloc[t_lo:t_hi], ws, we)
        feat["run_id"] = run_id
        feat["regime"] = lrow["regime"]
        feat["window_start"] = ws
        feat["label"] = lrow["label"]
        rows.append(feat)
    return rows


def extract_l3_features(
    orders_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    label_index: pd.DataFrame,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute L3 features for every window in label_index.
    Parallelized across runs.
    """
    grouped_orders = {rid: grp.sort_values("timestamp")
                      for rid, grp in orders_df.groupby("run_id")}
    grouped_trades = {rid: grp.sort_values("timestamp")
                      for rid, grp in trades_df.groupby("run_id")}

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_l3_for_run)(
            run_id,
            run_label,
            grouped_orders.get(run_id, pd.DataFrame()),
            grouped_trades.get(run_id, pd.DataFrame()),
        )
        for run_id, run_label in label_index.groupby("run_id")
    )

    return pd.DataFrame([row for rows in results for row in rows])


# ─────────────────────────────────────────────────────────────────────────────
# L2 feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _l2_window_features(snaps: pd.DataFrame) -> Dict:
    """
    Compute features from a sequence of L2 snapshots covering one window.
    Strictly no order-level info — only aggregated depth and price data.
    """
    feat: Dict = {}
    n = len(snaps)
    feat["n_snapshots"] = n
    if n == 0:
        return feat

    # ── Spread and price ─────────────────────────────────────────────────────
    spread = snaps["spread"].dropna()
    feat["spread_mean"] = float(spread.mean()) if len(spread) > 0 else float("nan")
    feat["spread_std"] = float(spread.std()) if len(spread) > 1 else 0.0

    mid = snaps["mid_price"].dropna()
    if len(mid) > 1:
        feat["mid_price_change"] = float(mid.iloc[-1] - mid.iloc[0])
        feat["mid_price_std"] = float(mid.std())
        feat["mid_price_range"] = float(mid.max() - mid.min())
    else:
        feat.update({"mid_price_change": 0.0, "mid_price_std": 0.0, "mid_price_range": 0.0})

    # ── Best-level depth oscillation (primary iceberg signal at L2) ──────────
    for side in ("bid", "ask"):
        qty1 = snaps[f"{side}_qty_1"].dropna()
        if len(qty1) > 1:
            feat[f"{side}1_qty_mean"] = float(qty1.mean())
            feat[f"{side}1_qty_std"] = float(qty1.std())
            feat[f"{side}1_qty_range"] = float(qty1.max() - qty1.min())
            feat[f"{side}1_qty_cv"] = float(qty1.std() / (qty1.mean() + 1e-6))
            # Recovery count: transitions low→high (depth replenishment after trade)
            diffs = np.diff(qty1.values)
            recoveries = int((diffs > qty1.mean() * 0.3).sum())
            feat[f"{side}1_recovery_count"] = recoveries
            # Fraction of snapshots where depth is above 80th percentile (persistence)
            p80 = np.percentile(qty1, 80)
            feat[f"{side}1_persistent_frac"] = float((qty1 >= p80 * 0.9).mean())
        else:
            v = float(qty1.iloc[0]) if len(qty1) == 1 else float("nan")
            feat.update({
                f"{side}1_qty_mean": v, f"{side}1_qty_std": 0.0,
                f"{side}1_qty_range": 0.0, f"{side}1_qty_cv": 0.0,
                f"{side}1_recovery_count": 0, f"{side}1_persistent_frac": 1.0,
            })

    # ── Depth imbalance ──────────────────────────────────────────────────────
    b1 = snaps["bid_qty_1"].fillna(0)
    a1 = snaps["ask_qty_1"].fillna(0)
    imb = (b1 - a1) / (b1 + a1 + 1e-6)
    feat["depth_imbalance_mean"] = float(imb.mean())
    feat["depth_imbalance_std"] = float(imb.std()) if len(imb) > 1 else 0.0

    # ── Multi-level depth ratios ─────────────────────────────────────────────
    for side in ("bid", "ask"):
        q1 = snaps[f"{side}_qty_1"].fillna(0)
        q2 = snaps[f"{side}_qty_2"].fillna(0)
        ratio = q1 / (q2 + 1e-6)
        feat[f"{side}_level_ratio_mean"] = float(ratio.mean())

    # ── Total book depth ─────────────────────────────────────────────────────
    for side in ("bid", "ask"):
        total = sum(snaps[f"{side}_qty_{i}"].fillna(0) for i in range(1, 6))
        feat[f"total_{side}_depth_mean"] = float(total.mean())

    total_bid = sum(snaps[f"bid_qty_{i}"].fillna(0) for i in range(1, 6))
    total_ask = sum(snaps[f"ask_qty_{i}"].fillna(0) for i in range(1, 6))
    book_imb = (total_bid - total_ask) / (total_bid + total_ask + 1e-6)
    feat["book_depth_imbalance"] = float(book_imb.mean())

    # ── Price level changes (best bid/ask price stability) ───────────────────
    if n > 1:
        bb_changes = (snaps["best_bid"].diff().dropna() != 0).sum()
        ba_changes = (snaps["best_ask"].diff().dropna() != 0).sum()
        feat["best_bid_change_rate"] = float(bb_changes / n)
        feat["best_ask_change_rate"] = float(ba_changes / n)
    else:
        feat["best_bid_change_rate"] = 0.0
        feat["best_ask_change_rate"] = 0.0

    return feat


def _extract_l2_for_run(
    run_id: str,
    run_label: pd.DataFrame,
    snaps: pd.DataFrame,
) -> List[Dict]:
    s_ts = snaps["timestamp"].values if len(snaps) > 0 else np.array([])
    rows = []
    for _, lrow in run_label.iterrows():
        ws, we = lrow["window_start"], lrow["window_end"]
        s_lo, s_hi = np.searchsorted(s_ts, [ws, we])
        feat = _l2_window_features(snaps.iloc[s_lo:s_hi])
        feat["run_id"] = run_id
        feat["regime"] = lrow["regime"]
        feat["window_start"] = ws
        feat["label"] = lrow["label"]
        rows.append(feat)
    return rows


def extract_l2_features(
    snaps_df: pd.DataFrame,
    label_index: pd.DataFrame,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute L2 features for every window in label_index.
    Only receives snaps_df — never orders/trades — to prevent leakage.
    Parallelized across runs.
    """
    grouped_snaps = {rid: grp.sort_values("timestamp")
                      for rid, grp in snaps_df.groupby("run_id")}

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_l2_for_run)(
            run_id,
            run_label,
            grouped_snaps.get(run_id, pd.DataFrame()),
        )
        for run_id, run_label in label_index.groupby("run_id")
    )

    return pd.DataFrame([row for rows in results for row in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_features(l3_df: pd.DataFrame, l2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine L3 and L2 features, adding regime indicators.
    """
    # Join on metadata columns
    meta_cols = ["run_id", "regime", "window_start", "label"]
    
    # Drop overlapping meta columns from L2 before joining
    l2_cols_to_drop = [c for c in meta_cols if c in l2_df.columns]
    
    # Prefix L2 features to avoid name collisions (though they should be unique)
    l2_feats_only = l2_df.drop(columns=l2_cols_to_drop)
    
    # Combine
    hybrid = pd.concat([l3_df, l2_feats_only], axis=1)
    
    # Add one-hot regime encoded columns
    regime_dummies = pd.get_dummies(hybrid["regime"], prefix="regime")
    hybrid = pd.concat([hybrid, regime_dummies], axis=1)
    
    return hybrid


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrices(
    data_dir: str,
    window_s: float = 0.3,
    step_s: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load parquet files from data_dir, build label index, extract L3, L2,
    and Hybrid feature matrices.
    """
    data_dir = Path(data_dir)
    logger.info("Loading parquet files …")
    orders_df = pd.read_parquet(data_dir / "l3_orders.parquet")
    trades_df = pd.read_parquet(data_dir / "l3_trades.parquet")
    snaps_df = pd.read_parquet(data_dir / "l2_snapshots.parquet")
    gt_df = pd.read_parquet(data_dir / "iceberg_gt.parquet")

    logger.info(
        f"Loaded: {len(orders_df):,} orders | {len(trades_df):,} trades | "
        f"{len(snaps_df):,} snapshots | {len(gt_df):,} GT rows"
    )

    logger.info("Building label index …")
    label_idx = build_label_index(gt_df, snaps_df, window_s, step_s)
    logger.info(
        f"Label index: {len(label_idx):,} windows | "
        f"positive rate = {label_idx['label'].mean():.3f}"
    )

    logger.info("Extracting L3 features …")
    l3_feats = extract_l3_features(orders_df, trades_df, label_idx)

    logger.info("Extracting L2 features …")
    l2_feats = extract_l2_features(snaps_df, label_idx)
    
    logger.info("Building Hybrid features …")
    hybrid_feats = build_hybrid_features(l3_feats, l2_feats)

    l3_feats.to_parquet(data_dir / "l3_features.parquet", index=False)
    l2_feats.to_parquet(data_dir / "l2_features.parquet", index=False)
    hybrid_feats.to_parquet(data_dir / "hybrid_features.parquet", index=False)
    
    logger.info(f"Features saved to {data_dir}")

    return l3_feats, l2_feats, hybrid_feats, rid, grp in snaps_df.groupby("run_id")

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_l2_for_run)(
            run_id,
            run_label,
            grouped_snaps.get(run_id, pd.DataFrame()),
        )
        for run_id, run_label in label_index.groupby("run_id")
    )

    return pd.DataFrame([row for rows in results for row in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrices(
    data_dir: str,
    window_s: float = 1.0,
    step_s: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load parquet files from data_dir, build label index, extract L3 and L2
    feature matrices.

    Returns:
        (l3_features_df, l2_features_df)  – aligned on run_id + window_start
    """
    data_dir = Path(data_dir)
    logger.info("Loading parquet files …")
    orders_df = pd.read_parquet(data_dir / "l3_orders.parquet")
    trades_df = pd.read_parquet(data_dir / "l3_trades.parquet")
    snaps_df = pd.read_parquet(data_dir / "l2_snapshots.parquet")
    gt_df = pd.read_parquet(data_dir / "iceberg_gt.parquet")

    logger.info(
        f"Loaded: {len(orders_df):,} orders | {len(trades_df):,} trades | "
        f"{len(snaps_df):,} snapshots | {len(gt_df):,} GT rows"
    )

    logger.info("Building label index …")
    label_idx = build_label_index(gt_df, snaps_df, window_s, step_s)
    logger.info(
        f"Label index: {len(label_idx):,} windows | "
        f"positive rate = {label_idx['label'].mean():.3f}"
    )

    logger.info("Extracting L3 features …")
    l3_feats = extract_l3_features(orders_df, trades_df, label_idx)

    logger.info("Extracting L2 features …")
    l2_feats = extract_l2_features(snaps_df, label_idx)

    l3_feats.to_parquet(data_dir / "l3_features.parquet", index=False)
    l2_feats.to_parquet(data_dir / "l2_features.parquet", index=False)
    logger.info(f"Features saved to {data_dir}")

    return l3_feats, l2_feats
