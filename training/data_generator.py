"""
Multi-regime synthetic order book data generator.

Produces paired L3 (full order stream) and L2 (aggregated snapshots) datasets
from the same simulation runs, guaranteeing that the only variable between
the two data representations is information level — not market conditions.

Architecture:
  SyntheticPriceProcess  – regime-driven GBM/OU mid-price path
  IcebergOrchestrator    – manages iceberg lifecycle with stochastic refresh delays
  RegimeSimRunner        – single simulation run → (L3 orders, L3 trades, L2 snapshots, ground truth)
  DatasetBuilder         – runs all regimes, saves parquet files
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Allow imports from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.order import NaiveIcebergOrder, OrderSide
from core.market_simulator import MarketSimulator, SimulationConfig
from core.event_queue import EventType, PeriodicEvent
from agents.agent import BuyAgent, SellAgent
from training.config import GenerationConfig, IcebergConfig, RegimeConfig, REGIMES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Price Process
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticPriceProcess:
    """
    Generates regime-specific mid-price dynamics using Euler-Maruyama
    discretisation of either GBM (trending/volatile) or OU (mean-reverting).
    """

    def __init__(self, initial_price: float, regime: RegimeConfig,
                 rng: np.random.RandomState, tick_size: float = 0.01):
        self.price = initial_price
        self.regime = regime
        self.rng = rng
        self.tick_size = tick_size
        self._long_run = initial_price  # OU equilibrium level

    def step(self, dt: float) -> float:
        r = self.regime
        noise = r.volatility * np.sqrt(dt) * self.rng.randn()
        # Mean-reversion pull (normalised log-return toward long run)
        pull = r.mean_reversion * np.log(self._long_run / max(self.price, 1e-6)) * dt
        # Drift + diffusion + OU pull on log-price
        log_return = r.drift * dt + pull + noise
        # Occasional jump in volatile regime
        if r.name == "volatile" and self.rng.random() < 0.003:
            log_return += self.rng.choice([-1, 1]) * self.rng.uniform(0.002, 0.008)
        self.price = max(self.price * np.exp(log_return), self.tick_size * 2)
        return round(self.price / self.tick_size) * self.tick_size


# ─────────────────────────────────────────────────────────────────────────────
# Iceberg Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IcebergChain:
    """Tracks the full lifecycle of one logical iceberg order."""
    chain_id: str
    side: str               # "BUY" or "SELL"
    price: float
    total_hidden: int
    visible_qty: int
    refresh_delay_s: float
    start_time: float
    remaining_hidden: int   # hidden qty not yet submitted as tips
    end_time: Optional[float] = None
    n_refills: int = 0
    current_tip_id: Optional[str] = None


class IcebergOrchestrator:
    """
    Manages iceberg orders with stochastic refresh delays.

    Strategy:
    - Each iceberg is modelled as a sequence of single-tip NaiveIcebergOrder
      submissions (peak_quantity == visible_quantity, so no built-in auto-refill).
    - When a tip is fully filled (detected via on_order callback), a new tip is
      scheduled after a stochastic delay drawn from the chain's refresh_delay_s.
    - Immediate-refresh icebergs use delay ≈ 0, so the gap is sub-millisecond.
    - Jitter (±10 % of delay) prevents trivially-detectable deterministic cadence.

    Label leakage prevention:
    - chain_id / parent linkage never appears in L2 features.
    - Exact fill timestamps are stored only in ground_truth, not in L2 snapshots.
    """

    def __init__(self, sim: MarketSimulator, rng: np.random.RandomState,
                 cfg: IcebergConfig):
        self.sim = sim
        self.rng = rng
        self.cfg = cfg
        self._chains: Dict[str, IcebergChain] = {}       # chain_id → chain
        self._tip_to_chain: Dict[str, str] = {}          # order_id → chain_id
        self._counter = 0
        sim.on_order(self._on_order)

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:09d}"

    def inject(self, side: str, price: float, total_hidden: int,
               visible_qty: int, refresh_delay_s: float) -> str:
        chain_id = self._new_id("CHAIN")
        chain = IcebergChain(
            chain_id=chain_id,
            side=side,
            price=price,
            total_hidden=total_hidden,
            visible_qty=visible_qty,
            refresh_delay_s=refresh_delay_s,
            start_time=self.sim.current_time,
            remaining_hidden=total_hidden - visible_qty,
        )
        self._chains[chain_id] = chain
        self._submit_tip(chain, visible_qty)
        return chain_id

    def _submit_tip(self, chain: IcebergChain, tip_qty: int):
        oid = self._new_id("ICE")
        rounded_price = self.sim.order_book.round_price(chain.price)
        order_side = OrderSide.BUY if chain.side == "BUY" else OrderSide.SELL
        # peak_quantity == visible_quantity → needs_refill stays False → no auto-refill
        order = NaiveIcebergOrder(
            order_id=oid,
            timestamp=self.sim.current_time,
            trader_id="iceberg_agent",
            side=order_side,
            price=rounded_price,
            peak_quantity=tip_qty,
            visible_quantity=tip_qty,
            quantity=tip_qty,
            min_visible_quantity=tip_qty,
            max_visible_quantity=tip_qty,
            randomize_refill=False,
        )
        chain.current_tip_id = oid
        self._tip_to_chain[oid] = chain.chain_id
        self.sim.submit_order(order)

    def _on_order(self, order, result):
        """Detect consumed tips; schedule next tip with stochastic delay."""
        oid = order.order_id
        if oid not in self._tip_to_chain:
            return
        chain_id = self._tip_to_chain[oid]
        chain = self._chains.get(chain_id)
        if chain is None:
            return

        if result.is_filled:
            chain.n_refills += 1
            del self._tip_to_chain[oid]

            if chain.remaining_hidden > 0:
                next_qty = min(chain.visible_qty, chain.remaining_hidden)
                chain.remaining_hidden -= next_qty

                delay = chain.refresh_delay_s
                if delay > 1e-4:
                    jitter = self.rng.uniform(-0.10, 0.10) * delay
                    delay = max(1e-4, delay + jitter)

                refill_at = self.sim.current_time + delay

                # Closure captures chain reference and next_qty
                def _do_refill(c=chain, q=next_qty):
                    if c.chain_id in self._chains:
                        self._submit_tip(c, q)

                self.sim.event_queue.schedule(
                    timestamp=refill_at,
                    event_type=EventType.AGENT_ACTION,
                    callback=_do_refill,
                )
            else:
                chain.end_time = self.sim.current_time
                del self._chains[chain_id]

    def ground_truth_df(self, run_id: str, regime: str) -> pd.DataFrame:
        """Return one row per iceberg chain with timing/size metadata."""
        rows = []
        # Collect completed chains from internal log (end_time set) +
        # chains still active at sim end (end_time = None)
        seen = set()
        for chain in self._chains.values():
            rows.append({
                "run_id": run_id, "regime": regime,
                "chain_id": chain.chain_id, "side": chain.side,
                "price": chain.price, "total_hidden": chain.total_hidden,
                "visible_qty": chain.visible_qty,
                "refresh_delay_s": chain.refresh_delay_s,
                "start_time": chain.start_time, "end_time": chain.end_time,
                "n_refills": chain.n_refills,
            })
            seen.add(chain.chain_id)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Single-run simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimRunResult:
    run_id: str
    regime_name: str
    l3_orders: pd.DataFrame
    l3_trades: pd.DataFrame
    l2_snapshots: pd.DataFrame
    iceberg_gt: pd.DataFrame


class RegimeSimRunner:
    """
    Executes one simulation run for a given market regime and packages output.

    Agent configuration:
      - n_agents buy agents + n_agents sell agents, where n_agents is chosen
        so that the combined tick rate approximates regime.order_rate_per_sec.
      - Each agent tick submits 1–4 orders with price = mid ± Gaussian noise.
    """

    TICK_INTERVAL = 0.05   # 20 ticks/sec per agent group

    def __init__(self, run_id: str, regime: RegimeConfig,
                 cfg: GenerationConfig, seed: int):
        self.run_id = run_id
        self.regime = regime
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def run(self) -> SimRunResult:
        sim_cfg = SimulationConfig(
            initial_mid_price=100.0,
            tick_size=0.01,
            start_time=0.0,
            end_time=self.cfg.sim_duration,
            snapshot_interval=self.cfg.l2_snapshot_interval_s,
            random_seed=self.seed,
        )
        sim = MarketSimulator(sim_cfg)
        price_proc = SyntheticPriceProcess(100.0, self.regime, self.rng)
        orchestrator = IcebergOrchestrator(sim, self.rng, self.cfg.iceberg)

        # Number of agent pairs to hit target order rate at TICK_INTERVAL
        orders_per_tick_avg = 2.5   # expected 1–4
        n_pairs = max(1, round(
            self.regime.order_rate_per_sec * self.TICK_INTERVAL / (2 * orders_per_tick_avg)
        ))
        buy_agents = [BuyAgent(f"BA{i}", target_price=100.0, price_std=0.08,
                               quantity_min=5, quantity_max=200)
                      for i in range(n_pairs)]
        sell_agents = [SellAgent(f"SA{i}", target_price=100.0, price_std=0.08,
                                 quantity_min=5, quantity_max=200)
                       for i in range(n_pairs)]

        # Iceberg injection probability per tick
        ice_cfg = self.cfg.iceberg
        prev_mid = self.rng.uniform(ice_cfg.prevalence_min, ice_cfg.prevalence_max)
        # Scale to per-tick rate: prevalence means % of "large order events"
        # Large orders arrive at ~0.5% of order rate → scale probability accordingly
        ice_prob_per_tick = prev_mid * 0.05

        def agent_tick():
            cur_time = sim.current_time
            mid = price_proc.step(self.TICK_INTERVAL)
            half_spread = self.regime.spread_ticks * sim_cfg.tick_size / 2.0

            for a in buy_agents:
                a.target_price = mid - half_spread
            for a in sell_agents:
                a.target_price = mid + half_spread

            n_each = int(self.rng.randint(1, 5))
            for a in buy_agents:
                a.submit_orders(sim, orders_per_tick=n_each, current_time=cur_time)
            for a in sell_agents:
                a.submit_orders(sim, orders_per_tick=n_each, current_time=cur_time)

            # Probabilistic iceberg injection near best bid/ask
            if self.rng.random() < ice_prob_per_tick:
                self._inject_iceberg(sim, price_proc, orchestrator, mid, sim_cfg.tick_size)

        PeriodicEvent(
            event_queue=sim.event_queue,
            interval=self.TICK_INTERVAL,
            event_type=EventType.AGENT_ACTION,
            callback=agent_tick,
            start_time=self.TICK_INTERVAL,
        )

        sim.run()
        return self._package(sim, orchestrator)

    def _inject_iceberg(self, sim, price_proc, orch, mid, tick_size):
        ice_cfg = self.cfg.iceberg
        side = "BUY" if self.rng.random() < 0.5 else "SELL"
        snap = sim.order_book.snapshot()

        if side == "BUY":
            anchor = snap.get("best_bid") or (mid - tick_size * 2)
            price = anchor - self.rng.randint(0, 4) * tick_size
        else:
            anchor = snap.get("best_ask") or (mid + tick_size * 2)
            price = anchor + self.rng.randint(0, 4) * tick_size

        visible_qty = int(self.rng.randint(ice_cfg.visible_qty_min, ice_cfg.visible_qty_max))
        hidden_mult = self.rng.uniform(ice_cfg.hidden_mult_min, ice_cfg.hidden_mult_max)
        total_hidden = max(visible_qty + 1, int(visible_qty * hidden_mult))

        if self.rng.random() < ice_cfg.immediate_refresh_fraction:
            delay_s = 0.0
        else:
            delay_ms = self.rng.uniform(ice_cfg.refresh_delay_min_ms, ice_cfg.refresh_delay_max_ms)
            delay_s = delay_ms / 1000.0

        orch.inject(side, price, total_hidden, visible_qty, delay_s)

    def _package(self, sim: MarketSimulator, orch: IcebergOrchestrator) -> SimRunResult:
        iceberg_order_ids: set = set(orch._tip_to_chain.keys())
        # Also include already-removed tips (tracked via gt rows)

        # L3 orders
        order_rows = []
        for o in sim.order_history:
            order_rows.append({
                "run_id": self.run_id,
                "regime": self.regime.name,
                "timestamp": o.timestamp,
                "order_id": o.order_id,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "price": getattr(o, "price", float("nan")),
                "quantity": o.quantity,
                "filled_qty": o.filled_quantity,
                "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                "is_iceberg": o.order_id.startswith("ICE_"),
                "order_type": type(o).__name__,
            })

        # L3 trades
        trade_rows = []
        for t in sim.trade_history:
            trade_rows.append({
                "run_id": self.run_id,
                "regime": self.regime.name,
                "timestamp": t.timestamp,
                "trade_id": t.trade_id,
                "price": t.price,
                "quantity": t.quantity,
                "buy_order_id": t.buy_order_id,
                "sell_order_id": t.sell_order_id,
                "aggressor_side": t.aggressor_side.value if hasattr(t.aggressor_side, "value") else str(t.aggressor_side),
                "buy_is_iceberg": t.buy_order_id.startswith("ICE_"),
                "sell_is_iceberg": t.sell_order_id.startswith("ICE_"),
            })

        # L2 snapshots – only aggregated depth; no order-level info that could
        # leak iceberg identity (no exact fill timestamps, no order counts, no IDs)
        snap_rows = []
        for snap in sim.snapshots:
            row = {
                "run_id": self.run_id,
                "regime": self.regime.name,
                "timestamp": snap["timestamp"],
                "best_bid": snap.get("best_bid"),
                "best_ask": snap.get("best_ask"),
                "mid_price": snap.get("mid_price"),
                "spread": snap.get("spread"),
            }
            for i, level in enumerate(snap.get("bids", [])[:5]):
                row[f"bid_price_{i+1}"] = level[0]
                row[f"bid_qty_{i+1}"] = level[1]
            for i, level in enumerate(snap.get("asks", [])[:5]):
                row[f"ask_price_{i+1}"] = level[0]
                row[f"ask_qty_{i+1}"] = level[1]
            # Pad missing levels
            for i in range(5):
                for pfx in ("bid", "ask"):
                    for sfx in ("price", "qty"):
                        k = f"{pfx}_{sfx}_{i+1}"
                        row.setdefault(k, float("nan"))
            snap_rows.append(row)

        gt_df = orch.ground_truth_df(self.run_id, self.regime.name)

        return SimRunResult(
            run_id=self.run_id,
            regime_name=self.regime.name,
            l3_orders=pd.DataFrame(order_rows),
            l3_trades=pd.DataFrame(trade_rows),
            l2_snapshots=pd.DataFrame(snap_rows),
            iceberg_gt=gt_df,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Runs all regime × run combinations, assembles and saves parquet datasets.

    Output files (in cfg.output_dir):
      l3_orders.parquet      – full order stream with is_iceberg labels
      l3_trades.parquet      – full trade stream with iceberg flags
      l2_snapshots.parquet   – aggregated book depth (no order-level info)
      iceberg_gt.parquet     – ground truth iceberg chain timing
    """

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def build(self) -> Dict:
        n_regimes = len(self.cfg.regimes)
        total_runs = n_regimes * self.cfg.runs_per_regime
        logger.info(
            f"Starting dataset generation: {total_runs} runs "
            f"({self.cfg.runs_per_regime} × {n_regimes} regimes), "
            f"{self.cfg.sim_duration}s each"
        )

        all_orders, all_trades, all_snaps, all_gt = [], [], [], []
        run_idx = 0

        for regime in self.cfg.regimes:
            for run_num in range(self.cfg.runs_per_regime):
                seed = self.cfg.seed_base + run_idx * 1000
                run_id = f"{regime.name}_{run_num:04d}"
                logger.info(f"  [{run_idx+1}/{total_runs}] {run_id} seed={seed}")
                t0 = time.perf_counter()

                runner = RegimeSimRunner(run_id, regime, self.cfg, seed)
                result = runner.run()

                all_orders.append(result.l3_orders)
                all_trades.append(result.l3_trades)
                all_snaps.append(result.l2_snapshots)
                all_gt.append(result.iceberg_gt)

                elapsed = time.perf_counter() - t0
                n_ord = len(result.l3_orders)
                n_trd = len(result.l3_trades)
                n_ice = len(result.iceberg_gt)
                logger.info(
                    f"    → {n_ord:>7,} orders  {n_trd:>6,} trades  "
                    f"{n_ice:>4} icebergs  ({elapsed:.1f}s)"
                )
                run_idx += 1

        logger.info("Concatenating and saving …")
        orders_df = pd.concat(all_orders, ignore_index=True)
        trades_df = pd.concat(all_trades, ignore_index=True)
        snaps_df = pd.concat(all_snaps, ignore_index=True)
        gt_df = pd.concat(all_gt, ignore_index=True)

        orders_df.to_parquet(self.out / "l3_orders.parquet")
        trades_df.to_parquet(self.out / "l3_trades.parquet")
        snaps_df.to_parquet(self.out / "l2_snapshots.parquet")
        gt_df.to_parquet(self.out / "iceberg_gt.parquet")

        stats = {
            "total_orders": len(orders_df),
            "total_trades": len(trades_df),
            "total_snapshots": len(snaps_df),
            "total_iceberg_chains": len(gt_df),
            "iceberg_order_fraction": orders_df["is_iceberg"].mean(),
        }
        logger.info(
            f"Done. {stats['total_orders']:,} orders | "
            f"{stats['total_trades']:,} trades | "
            f"{stats['total_snapshots']:,} snapshots | "
            f"{stats['total_iceberg_chains']:,} iceberg chains"
        )
        return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from training.config import GenerationConfig
    cfg = GenerationConfig()
    builder = DatasetBuilder(cfg)
    builder.build()
