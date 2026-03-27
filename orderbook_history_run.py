"""
Parameterized order book simulation: runs the market like new_run.py but exposes
CLI controls and exports order book snapshot history to JSON.

Example:
  python orderbook_history_run.py --duration 60 --icebergs 2 --cluster 0.4 \\
    --lp-per-level 80000 --imbalance 0.2 --min-spread 0.35 --output histories/out.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from core import (
    MarketSimulator,
    SimulationConfig,
    EventType,
    PeriodicEvent,
)
from core.order import LimitOrder, OrderSide
from agents import BuyAgent, SellAgent
from agents.iceberg import inject_iceberg_buy


def json_sanitize(obj: Any) -> Any:
    """Make snapshot / nested structures JSON-serializable."""
    if isinstance(obj, tuple):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, list):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 10)
    return obj


def build_ladder(mid: float, min_spread: float, tick_size: float) -> tuple[list[float], list[float]]:
    """
    Bid and ask price levels spanning at least min_spread total (± half_spread from mid).
    Bids: [mid - half, ...]; asks: [mid + tick, ..., mid + half].
    """
    half = max(min_spread / 2.0, tick_size)
    step = max(tick_size, 0.05)
    n = max(2, int(half / step))
    bid_prices = []
    ask_prices = []
    for i in range(1, n + 1):
        b = mid - i * step
        a = mid + i * step
        if b > 0:
            bid_prices.append(round(b / tick_size) * tick_size)
        ask_prices.append(round(a / tick_size) * tick_size)
    return sorted(set(bid_prices), reverse=True), sorted(set(ask_prices))


@dataclass
class HistoryRunConfig:
    duration: float
    mid: float
    tick_size: float
    snapshot_interval: float
    seed: int
    icebergs: int
    iceberg_peak: int
    iceberg_visible: int
    price_std: float
    cluster: float
    lp_per_level: int
    imbalance: float
    min_spread: float
    base_orders_per_sec: float
    quantity_min: int
    quantity_max: int
    output: Path


def seed_liquidity_walls(sim: MarketSimulator, bid_prices: list[float], ask_prices: list[float], qty: int) -> int:
    """Large passive limits at ladder levels (non-iceberg, hard to deplete)."""
    n = 0
    for i, p in enumerate(bid_prices):
        o = LimitOrder(
            order_id=f"LP_B_{i:04d}",
            timestamp=sim.current_time,
            trader_id="liquidity_provider",
            side=OrderSide.BUY,
            price=p,
            quantity=qty,
        )
        sim.submit_order(o, delay=0.0)
        n += 1
    for j, p in enumerate(ask_prices):
        o = LimitOrder(
            order_id=f"LP_A_{j:04d}",
            timestamp=sim.current_time,
            trader_id="liquidity_provider",
            side=OrderSide.SELL,
            price=p,
            quantity=qty,
        )
        sim.submit_order(o, delay=0.0)
        n += 1
    return n


def run_history(cfg: HistoryRunConfig) -> tuple[MarketSimulator, dict]:
    random.seed(cfg.seed)

    bid_ladder, ask_ladder = build_ladder(cfg.mid, cfg.min_spread, cfg.tick_size)
    if not bid_ladder or not ask_ladder:
        raise ValueError("Empty ladder; check mid and min_spread")

    sim_config = SimulationConfig(
        initial_mid_price=cfg.mid,
        tick_size=cfg.tick_size,
        start_time=0.0,
        end_time=cfg.duration,
        snapshot_interval=cfg.snapshot_interval,
        random_seed=cfg.seed,
    )
    sim = MarketSimulator(sim_config)

    def seed_lp_callback():
        seed_liquidity_walls(sim, bid_ladder, ask_ladder, cfg.lp_per_level)

    sim.event_queue.schedule(timestamp=0.0, event_type=EventType.AGENT_ACTION, callback=seed_lp_callback)

    buy_agent = BuyAgent(
        agent_id="buy_agent",
        target_price=max(cfg.mid - 0.10, cfg.tick_size),
        price_std=cfg.price_std,
        quantity_min=cfg.quantity_min,
        quantity_max=cfg.quantity_max,
    )
    sell_agent = SellAgent(
        agent_id="sell_agent",
        target_price=cfg.mid + 0.10,
        price_std=cfg.price_std,
        quantity_min=cfg.quantity_min,
        quantity_max=cfg.quantity_max,
    )

    imb = max(-0.95, min(0.95, cfg.imbalance))
    base = cfg.base_orders_per_sec
    buy_hz = base * (1.0 + imb)
    sell_hz = base * (1.0 - imb)
    buy_interval = 1.0 / max(0.05, buy_hz)
    sell_interval = 1.0 / max(0.05, sell_hz)

    def buy_agent_tick():
        if sim.current_time >= cfg.duration:
            return
        if random.random() < cfg.cluster:
            for _ in range(random.randint(1, 3)):
                p = random.choice(bid_ladder)
                p = sim.order_book.round_price(p)
                q = random.randint(buy_agent.quantity_min, buy_agent.quantity_max)
                o = LimitOrder(
                    order_id=buy_agent._next_order_id(),
                    timestamp=sim.current_time,
                    trader_id=buy_agent.agent_id,
                    side=OrderSide.BUY,
                    price=p,
                    quantity=q,
                )
                sim.submit_order(o, delay=0.0)
        else:
            buy_agent.submit_orders(sim, current_time=sim.current_time)

    def sell_agent_tick():
        if sim.current_time >= cfg.duration:
            return
        if random.random() < cfg.cluster:
            for _ in range(random.randint(1, 3)):
                p = random.choice(ask_ladder)
                p = sim.order_book.round_price(p)
                q = random.randint(sell_agent.quantity_min, sell_agent.quantity_max)
                o = LimitOrder(
                    order_id=sell_agent._next_order_id(),
                    timestamp=sim.current_time,
                    trader_id=sell_agent.agent_id,
                    side=OrderSide.SELL,
                    price=p,
                    quantity=q,
                )
                sim.submit_order(o, delay=0.0)
        else:
            sell_agent.submit_orders(sim, current_time=sim.current_time)

    PeriodicEvent(
        event_queue=sim.event_queue,
        interval=buy_interval,
        event_type=EventType.AGENT_ACTION,
        callback=buy_agent_tick,
        start_time=buy_interval,
    )
    PeriodicEvent(
        event_queue=sim.event_queue,
        interval=sell_interval,
        event_type=EventType.AGENT_ACTION,
        callback=sell_agent_tick,
        start_time=sell_interval,
    )

    if cfg.icebergs > 0:
        for k in range(cfg.icebergs):
            inject_time = cfg.duration * (k + 1) / (cfg.icebergs + 1)
            order_id = f"ICEBERG_BUY_{k}"

            def make_inject(ki=k, oid=order_id):
                def _inject():
                    offset = 0.02 + ki * 0.02
                    inject_iceberg_buy(
                        sim,
                        oid,
                        current_time=sim.current_time,
                        target_price=cfg.mid - offset,
                        peak_quantity=cfg.iceberg_peak,
                        visible_quantity=cfg.iceberg_visible,
                    )

                return _inject

            sim.event_queue.schedule(
                timestamp=inject_time,
                event_type=EventType.AGENT_ACTION,
                callback=make_inject(),
            )

    sim.run(until_time=cfg.duration)

    meta = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    meta["output"] = str(cfg.output)

    payload = {
        "meta": meta,
        "ladder": {"bid_prices": bid_ladder, "ask_prices": ask_ladder},
        "snapshots": json_sanitize(sim.snapshots),
        "stats": {f.name: getattr(sim.stats, f.name) for f in fields(sim.stats)},
        "final_book": json_sanitize(sim.order_book.snapshot()),
    }
    return sim, payload


def parse_args() -> HistoryRunConfig:
    p = argparse.ArgumentParser(description="Run LOB sim and export order book history JSON")
    p.add_argument("--duration", type=float, default=60.0, help="Simulation length (seconds)")
    p.add_argument("--mid", type=float, default=100.0, help="Reference mid price")
    p.add_argument("--tick-size", type=float, default=0.01, help="Book tick size")
    p.add_argument("--snapshot-interval", type=float, default=1.0, help="Seconds between snapshots")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--icebergs", type=int, default=1, help="Number of buy iceberg orders")
    p.add_argument("--iceberg-peak", type=int, default=15_000, help="Iceberg peak quantity")
    p.add_argument("--iceberg-visible", type=int, default=25, help="Iceberg visible tip")
    p.add_argument(
        "--price-std",
        type=float,
        default=0.05,
        help="Gaussian std for agent limit prices around side anchors (market variability)",
    )
    p.add_argument(
        "--cluster",
        type=float,
        default=0.0,
        help="[0,1] fraction of agent events that place at random ladder prices (trade clustering, non-iceberg)",
    )
    p.add_argument(
        "--lp-per-level",
        type=int,
        default=80_000,
        help="Passive LP size at each ladder level (volume without quick depletion, non-iceberg)",
    )
    p.add_argument(
        "--imbalance",
        type=float,
        default=0.0,
        help="Order skew in [-0.95, 0.95]: positive = more buy flow",
    )
    p.add_argument(
        "--min-spread",
        type=float,
        default=0.30,
        help="Minimum $ span of bid/ask ladders from mid (>=0.30 for multi-level book)",
    )
    p.add_argument(
        "--base-orders-per-sec",
        type=float,
        default=5.0,
        help="Baseline order rate per side before imbalance scaling",
    )
    p.add_argument("--quantity-min", type=int, default=10)
    p.add_argument("--quantity-max", type=int, default=100)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: histories/orderbook_<duration>s.json)",
    )
    a = p.parse_args()
    out = a.output
    if out is None:
        root = Path(__file__).resolve().parent
        out = root / "histories" / f"orderbook_{int(a.duration)}s_seed{a.seed}.json"
    return HistoryRunConfig(
        duration=a.duration,
        mid=a.mid,
        tick_size=a.tick_size,
        snapshot_interval=a.snapshot_interval,
        seed=a.seed,
        icebergs=a.icebergs,
        iceberg_peak=a.iceberg_peak,
        iceberg_visible=a.iceberg_visible,
        price_std=a.price_std,
        cluster=max(0.0, min(1.0, a.cluster)),
        lp_per_level=a.lp_per_level,
        imbalance=a.imbalance,
        min_spread=a.min_spread,
        base_orders_per_sec=a.base_orders_per_sec,
        quantity_min=a.quantity_min,
        quantity_max=a.quantity_max,
        output=out,
    )


def main():
    cfg = parse_args()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    _sim, payload = run_history(cfg)
    with open(cfg.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(payload['snapshots'])} snapshots to {cfg.output}")
    print(f"Stats: orders={payload['stats'].get('total_orders')} trades={payload['stats'].get('total_trades')} volume={payload['stats'].get('total_volume')}")


if __name__ == "__main__":
    main()
