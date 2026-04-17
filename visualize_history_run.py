"""
Visualize the simulated market run from orderbook_history_run.

Plot behavior:
1) Draw synthetic mid path for full duration.
2) Reveal buy/sell orders in 5-second steps from left to right.
3) Mark iceberg orders as X markers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orderbook_history_run import HistoryRunConfig, run_history
from core.order import OrderSide


def build_cfg(args: argparse.Namespace) -> HistoryRunConfig:
    output = Path(args.output) if args.output else Path("histories") / f"viz_run_{int(args.duration)}s.json"
    return HistoryRunConfig(
        duration=args.duration,
        mid=args.mid,
        tick_size=args.tick_size,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
        icebergs=args.icebergs,
        iceberg_peak=args.iceberg_peak,
        iceberg_visible=args.iceberg_visible,
        price_std=args.price_std,
        cluster=max(0.0, min(1.0, args.cluster)),
        lp_per_level=args.lp_per_level,
        imbalance=args.imbalance,
        min_spread=args.min_spread,
        base_orders_per_sec=args.base_orders_per_sec,
        quantity_min=args.quantity_min,
        quantity_max=args.quantity_max,
        path_max_deviation=args.path_max_deviation,
        agent_half_spread=args.agent_half_spread,
        cluster_near=args.cluster_near,
        cluster_ticks=args.cluster_ticks,
        output=output,
    )


def split_orders(sim, include_lp: bool):
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    ice_buy_x, ice_buy_y, ice_sell_x, ice_sell_y = [], [], [], []

    for order in sim.order_history:
        if not hasattr(order, "side") or order.side is None:
            continue
        price = getattr(order, "price", None)
        if price is None:
            continue
        if (not include_lp) and getattr(order, "trader_id", "") == "liquidity_provider":
            continue

        t = float(order.timestamp)
        is_iceberg = str(getattr(order, "order_id", "")).startswith("ICEBERG_")

        if order.side == OrderSide.BUY:
            if is_iceberg:
                ice_buy_x.append(t)
                ice_buy_y.append(price)
            else:
                buy_x.append(t)
                buy_y.append(price)
        elif order.side == OrderSide.SELL:
            if is_iceberg:
                ice_sell_x.append(t)
                ice_sell_y.append(price)
            else:
                sell_x.append(t)
                sell_y.append(price)

    return {
        "buy": (buy_x, buy_y),
        "sell": (sell_x, sell_y),
        "ice_buy": (ice_buy_x, ice_buy_y),
        "ice_sell": (ice_sell_x, ice_sell_y),
    }


def upto(xs, ys, t):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if x <= t:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def visualize(args: argparse.Namespace):
    cfg = build_cfg(args)
    sim, payload = run_history(cfg)

    path = payload.get("synthetic_mid_path", [])
    path_t = [p["t"] for p in path]
    path_p = [p["price"] for p in path]

    orders = split_orders(sim, include_lp=args.include_lp)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(path_t, path_p, color="black", linewidth=2.0, label="Synthetic mid path")

    buy_line, = ax.plot([], [], "o", color="green", markersize=3, linestyle="None", alpha=0.7, label="Buy orders")
    sell_line, = ax.plot([], [], "o", color="red", markersize=3, linestyle="None", alpha=0.7, label="Sell orders")
    ice_buy_line, = ax.plot([], [], "x", color="green", markersize=8, linestyle="None", mew=2, label="Iceberg BUY")
    ice_sell_line, = ax.plot([], [], "x", color="red", markersize=8, linestyle="None", mew=2, label="Iceberg SELL")

    step = max(1.0, args.step_seconds)
    cur = step
    while cur <= cfg.duration + 1e-9:
        bx, by = upto(*orders["buy"], cur)
        sx, sy = upto(*orders["sell"], cur)
        ibx, iby = upto(*orders["ice_buy"], cur)
        isx, isy = upto(*orders["ice_sell"], cur)

        buy_line.set_data(bx, by)
        sell_line.set_data(sx, sy)
        ice_buy_line.set_data(ibx, iby)
        ice_sell_line.set_data(isx, isy)

        ax.set_title(f"Simulated Market Over Time (revealed through t={cur:.0f}s)")
        plt.pause(max(0.0, args.pause_seconds))
        cur += step

    # Draw 5-second separators for readability of time blocks.
    mark = 5.0
    while mark < cfg.duration:
        ax.axvline(mark, color="gray", linestyle="--", linewidth=0.6, alpha=0.25)
        mark += 5.0

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Price level")
    ax.set_xlim(0.0, cfg.duration)

    y_all = []
    y_all.extend(path_p)
    y_all.extend(orders["buy"][1])
    y_all.extend(orders["sell"][1])
    y_all.extend(orders["ice_buy"][1])
    y_all.extend(orders["ice_sell"][1])
    if y_all:
        lo = min(y_all)
        hi = max(y_all)
        pad = max(0.02, 0.1 * (hi - lo if hi > lo else 1.0))
        ax.set_ylim(lo - pad, hi + pad)

    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    if args.save_png:
        out = Path(args.save_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        print(f"Saved visualization to {out}")

    if not args.no_show:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize synthetic market path + order placements over time")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--mid", type=float, default=100.0)
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--snapshot-interval", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--icebergs", type=int, default=1)
    p.add_argument("--iceberg-peak", type=int, default=15_000)
    p.add_argument("--iceberg-visible", type=int, default=25)
    p.add_argument("--price-std", type=float, default=0.05)
    p.add_argument("--cluster", type=float, default=0.35)
    p.add_argument("--lp-per-level", type=int, default=80_000)
    p.add_argument("--imbalance", type=float, default=0.0)
    p.add_argument("--min-spread", type=float, default=0.30)
    p.add_argument("--base-orders-per-sec", type=float, default=5.0)
    p.add_argument("--quantity-min", type=int, default=10)
    p.add_argument("--quantity-max", type=int, default=100)
    p.add_argument("--path-max-deviation", type=float, default=0.25)
    p.add_argument("--agent-half-spread", type=float, default=0.02)
    p.add_argument("--cluster-near", type=float, default=0.06)
    p.add_argument("--cluster-ticks", type=int, default=6)
    p.add_argument("--output", type=str, default=None, help="Run-history JSON path from simulator")

    p.add_argument("--step-seconds", type=float, default=5.0, help="Reveal interval for dots")
    p.add_argument("--pause-seconds", type=float, default=0.6, help="Pause between reveal steps")
    p.add_argument("--include-lp", action="store_true", help="Include liquidity provider seed orders in dots")
    p.add_argument("--save-png", type=str, default=None, help="Optional output figure path")
    p.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    return p.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
