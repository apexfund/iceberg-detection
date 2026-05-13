"""Multi-regime synthetic order book data generator."""

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Allow imports from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.order import NaiveIcebergOrder, OrderSide
from core.market_simulator import MarketSimulator, SimulationConfig
from core.event_queue import EventType, PeriodicEvent
from training.agents import BuyAgent, SellAgent
from training.config import GenerationConfig, IcebergConfig, RegimeConfig

logger = logging.getLogger(__name__)

class SyntheticPriceProcess:
    def __init__(self, initial_price: float, regime: RegimeConfig, rng: np.random.RandomState, tick_size: float = 0.01):
        self.price = initial_price
        self.regime = regime
        self.rng = rng
        self.tick_size = tick_size
        self._long_run = initial_price

    def step(self, dt: float) -> float:
        r = self.regime
        noise = r.volatility * np.sqrt(dt) * self.rng.randn()
        pull = r.mean_reversion * np.log(self._long_run / max(self.price, 1e-6)) * dt
        log_return = r.drift * dt + pull + noise
        if r.name == "volatile" and self.rng.random() < 0.003:
            log_return += self.rng.choice([-1, 1]) * self.rng.uniform(0.002, 0.008)
        self.price = max(self.price * np.exp(log_return), self.tick_size * 2)
        return round(self.price / self.tick_size) * self.tick_size

@dataclass
class IcebergChain:
    chain_id: str
    side: str
    price: float
    total_hidden: int
    visible_qty: int
    refresh_delay_s: float
    start_time: float
    remaining_hidden: int
    end_time: Optional[float] = None
    n_refills: int = 0
    current_tip_id: Optional[str] = None

class IcebergOrchestrator:
    def __init__(self, sim: MarketSimulator, rng: np.random.RandomState, cfg: IcebergConfig):
        self.sim = sim
        self.rng = rng
        self.cfg = cfg
        self._chains: Dict[str, IcebergChain] = {}
        self._completed_chains: List[IcebergChain] = []
        self._tip_to_chain: Dict[str, str] = {}
        self._tip_orders: Dict[str, NaiveIcebergOrder] = {}
        self._counter = 0
        sim.on_trade(self._on_trade)

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:09d}"

    def inject(self, side: str, price: float, total_hidden: int, visible_qty: int, refresh_delay_s: float) -> str:
        chain_id = self._new_id("CHAIN")
        chain = IcebergChain(
            chain_id=chain_id, side=side, price=price, total_hidden=total_hidden,
            visible_qty=visible_qty, refresh_delay_s=refresh_delay_s,
            start_time=self.sim.current_time, remaining_hidden=total_hidden - visible_qty,
        )
        self._chains[chain_id] = chain
        self._submit_tip(chain, visible_qty, is_refill=False)
        return chain_id

    def _submit_tip(self, chain: IcebergChain, tip_qty: int, is_refill: bool):
        oid = self._new_id("ICE")
        rounded_price = self.sim.order_book.round_price(chain.price)
        order_side = OrderSide.BUY if chain.side == "BUY" else OrderSide.SELL
        order = NaiveIcebergOrder(
            order_id=oid, timestamp=self.sim.current_time, trader_id="iceberg_agent",
            side=order_side, price=rounded_price, peak_quantity=tip_qty,
            visible_quantity=tip_qty, quantity=tip_qty, min_visible_quantity=tip_qty,
            max_visible_quantity=tip_qty, randomize_refill=False,
        )
        chain.current_tip_id = oid
        if is_refill:
            chain.n_refills += 1
        self._tip_to_chain[oid] = chain.chain_id
        self._tip_orders[oid] = order
        self.sim.submit_order(order)

    def _on_trade(self, trade):
        for oid in (trade.buy_order_id, trade.sell_order_id):
            if oid not in self._tip_to_chain:
                continue
            order = self._tip_orders.get(oid)
            if order is not None and order.is_filled:
                self._handle_filled_tip(oid)

    def _handle_filled_tip(self, oid: str):
        chain_id = self._tip_to_chain.get(oid)
        if chain_id is None:
            return

        chain = self._chains.get(chain_id)
        if chain is None:
            return

        self._tip_to_chain.pop(oid, None)
        self._tip_orders.pop(oid, None)
        if chain.remaining_hidden > 0:
            next_qty = min(chain.visible_qty, chain.remaining_hidden)
            chain.remaining_hidden -= next_qty
            delay = chain.refresh_delay_s
            if delay > 1e-4:
                delay = max(1e-4, delay + self.rng.uniform(-0.1, 0.1) * delay)

            def _do_refill(c=chain, q=next_qty):
                if c.chain_id in self._chains:
                    self._submit_tip(c, q, is_refill=True)

            self.sim.event_queue.schedule(self.sim.current_time + delay, EventType.AGENT_ACTION, _do_refill)
        else:
            chain.end_time = self.sim.current_time
            self._completed_chains.append(chain)
            del self._chains[chain_id]

    def ground_truth_df(self, run_id: str, regime: str) -> pd.DataFrame:
        rows = []
        for chain in [*self._completed_chains, *self._chains.values()]:
            rows.append({
                "run_id": run_id, "regime": regime, "chain_id": chain.chain_id,
                "side": chain.side, "price": chain.price, "total_hidden": chain.total_hidden,
                "visible_qty": chain.visible_qty, "refresh_delay_s": chain.refresh_delay_s,
                "start_time": chain.start_time, "end_time": chain.end_time, "n_refills": chain.n_refills,
            })
        if not rows:
            return pd.DataFrame(columns=[
                "run_id", "regime", "chain_id", "side", "price", "total_hidden",
                "visible_qty", "refresh_delay_s", "start_time", "end_time", "n_refills",
            ])
        return pd.DataFrame(rows).sort_values(["start_time", "chain_id"]).reset_index(drop=True)

@dataclass
class SimRunResult:
    run_id: str
    regime_name: str
    l3_orders: pd.DataFrame
    l3_trades: pd.DataFrame
    l2_snapshots: pd.DataFrame
    iceberg_gt: pd.DataFrame

class RegimeSimRunner:
    TICK_INTERVAL = 0.05
    def __init__(self, run_id: str, regime: RegimeConfig, cfg: GenerationConfig, seed: int):
        self.run_id = run_id
        self.regime = regime
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def run(self) -> SimRunResult:
        sim_cfg = SimulationConfig(
            initial_mid_price=100.0, tick_size=0.01, start_time=0.0,
            end_time=self.cfg.sim_duration, snapshot_interval=self.cfg.l2_snapshot_interval_s,
            random_seed=self.seed,
        )
        sim = MarketSimulator(sim_cfg)
        price_proc = SyntheticPriceProcess(100.0, self.regime, self.rng)
        orchestrator = IcebergOrchestrator(sim, self.rng, self.cfg.iceberg)

        # Increase agent aggression: more orders per tick, larger sizes
        n_pairs = max(1, round(self.regime.order_rate_per_sec * self.TICK_INTERVAL / 3.0))
        buy_agents = [BuyAgent(f"BA{i}", target_price=100.0, price_std=0.04, quantity_min=10, quantity_max=500) for i in range(n_pairs)]
        sell_agents = [SellAgent(f"SA{i}", target_price=100.0, price_std=0.04, quantity_min=10, quantity_max=500) for i in range(n_pairs)]

        ice_prob = self.cfg.iceberg.prevalence_min * 0.10 # Increased injection
        def agent_tick():
            mid = price_proc.step(self.TICK_INTERVAL)
            half_spread = self.regime.spread_ticks * 0.01 / 2.0
            for a in buy_agents: a.target_price = mid - half_spread
            for a in sell_agents: a.target_price = mid + half_spread
            
            # Submit multiple orders per tick to create noise and pressure
            for a in buy_agents + sell_agents:
                a.submit_orders(sim, orders_per_tick=self.rng.randint(2, 6), current_time=sim.current_time)
                
            if self.rng.random() < ice_prob:
                self._inject_iceberg(sim, mid, orchestrator)

        PeriodicEvent(sim.event_queue, self.TICK_INTERVAL, EventType.AGENT_ACTION, agent_tick)
        sim.run()
        return self._package(sim, orchestrator)

    def _inject_iceberg(self, sim, mid, orch):
        ice_cfg = self.cfg.iceberg
        side = "BUY" if self.rng.random() < 0.5 else "SELL"
        # Always inject AT the best bid/ask to ensure it gets hit
        snap = sim.order_book.snapshot()
        if side == "BUY":
            price = snap.get("best_bid") or (mid - 0.01)
        else:
            price = snap.get("best_ask") or (mid + 0.01)
            
        visible_qty = self.rng.randint(ice_cfg.visible_qty_min, ice_cfg.visible_qty_max)
        hidden_mult = self.rng.uniform(ice_cfg.hidden_mult_min, ice_cfg.hidden_mult_max)
        total_hidden = int(visible_qty * hidden_mult)
        
        delay_s = 0.0
        if self.rng.random() > ice_cfg.immediate_refresh_fraction:
            delay_ms = self.rng.uniform(ice_cfg.refresh_delay_min_ms, ice_cfg.refresh_delay_max_ms)
            delay_s = delay_ms / 1000.0
            
        orch.inject(side, price, total_hidden, visible_qty, delay_s)

    def _package(self, sim, orch) -> SimRunResult:
        def top_level_quantity(snapshot: Dict, depth_key: str) -> int:
            depth = snapshot.get(depth_key) or []
            return int(depth[0][1]) if depth else 0

        orders = pd.DataFrame([{
            "run_id": self.run_id, "regime": self.regime.name, "timestamp": o.timestamp,
            "order_id": o.order_id, "side": str(o.side.value if hasattr(o.side, "value") else o.side), 
            "price": getattr(o, "price", 0.0), "quantity": o.quantity, "filled_qty": o.filled_quantity,
        } for o in sim.order_history])
        trades = pd.DataFrame([{
            "run_id": self.run_id, "regime": self.regime.name, "timestamp": t.timestamp,
            "trade_id": t.trade_id, "price": t.price, "quantity": t.quantity,
            "buy_order_id": t.buy_order_id, "sell_order_id": t.sell_order_id,
            "aggressor_side": str(t.aggressor_side.value if hasattr(t.aggressor_side, "value") else t.aggressor_side),
        } for t in sim.trade_history])
        snaps = pd.DataFrame([{
            "run_id": self.run_id, "regime": self.regime.name, "timestamp": s["timestamp"],
            "best_bid": s.get("best_bid"), "best_ask": s.get("best_ask"),
            "bid_qty_1": top_level_quantity(s, "bid_depth"),
            "ask_qty_1": top_level_quantity(s, "ask_depth"),
        } for s in sim.snapshots])
        return SimRunResult(self.run_id, self.regime.name, orders, trades, snaps, orch.ground_truth_df(self.run_id, self.regime.name))

class DatasetBuilder:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def build(self) -> Dict:
        jobs = []
        for regime in self.cfg.regimes:
            for i in range(self.cfg.runs_per_regime):
                jobs.append((f"{regime.name}_{i:04d}", regime, self.cfg, self.cfg.seed_base + len(jobs) * 1000))

        if self.cfg.n_jobs == 1:
            results = [RegimeSimRunner(*job).run() for job in jobs]
        else:
            try:
                results = Parallel(n_jobs=self.cfg.n_jobs, prefer="processes")(
                    delayed(RegimeSimRunner(*job).run)() for job in jobs
                )
            except PermissionError:
                logger.warning("Falling back to serial data generation because process workers are unavailable.")
                results = [RegimeSimRunner(*job).run() for job in jobs]
        
        orders = pd.concat([r.l3_orders for r in results], ignore_index=True)
        trades = pd.concat([r.l3_trades for r in results], ignore_index=True)
        snaps = pd.concat([r.l2_snapshots for r in results], ignore_index=True)
        gt = pd.concat([r.iceberg_gt for r in results], ignore_index=True)
        
        orders.to_parquet(self.out / "l3_orders.parquet", index=False)
        trades.to_parquet(self.out / "l3_trades.parquet", index=False)
        snaps.to_parquet(self.out / "l2_snapshots.parquet", index=False)
        gt.to_parquet(self.out / "iceberg_gt.parquet", index=False)
        
        return {"total_orders": len(orders), "total_trades": len(trades), "total_snapshots": len(snaps), "total_iceberg_chains": len(gt)}
