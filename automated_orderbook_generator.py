#!/usr/bin/env python3
"""
Single-file synthetic limit order book episode generator for iceberg vs non-iceberg datasets.
v5 — Two-phase balanced-final-labels mode: generate iceberg-labeled and noniceberg-labeled
     episodes independently via accept/reject until exact quotas are met.
     Full audit, coverage, splits, positive support, hard negatives from v4.

Outputs (under --output-dir):
  {split}/iceberg/*.csv, {split}/noniceberg/*.csv,
  metadata/episodes_metadata.csv, metadata/dataset_summary.json,
  metadata/dataset_audit.json, debug_optional/*.json (optional), Dataset.zip (optional).

Dependencies: Python 3.10+ standard library only.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "Dataset"
MEANINGFUL_ABSORB_VOL = 120
MEANINGFUL_REPLENISH = 1
MEANINGFUL_ACTIVE_SEC = 0.4
ROLLING_WINDOW_SEC = 5.0
_NAN = float("nan")

DIFFICULTY_EASY_POS = "easy_positive"
DIFFICULTY_MED_POS = "medium_positive"
DIFFICULTY_HARD_POS = "hard_positive"
DIFFICULTY_EASY_NEG = "easy_negative"
DIFFICULTY_HARD_NEG = "hard_negative"

_IMBALANCE_WARN_RATIO = 3.0
_MAX_PHASE_ATTEMPTS_FACTOR = 20  # per-quota-episode max retry multiplier


def _poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _safe_stdev(xs: Sequence[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Generation controls
# ---------------------------------------------------------------------------

@dataclass
class GenerationControls:
    target_class_balance: Optional[float] = None
    target_hard_negative_freq: Optional[float] = None
    target_fake_iceberg_freq: Optional[float] = None
    target_multi_iceberg_freq: Optional[float] = None
    target_regime_shift_freq: Optional[float] = None
    positive_support_prob: float = 0.80
    positive_support_strength: float = 0.5
    # v5: balanced final labels
    balanced_final_labels: bool = False
    target_iceberg_count: Optional[int] = None
    target_noniceberg_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Orders & iceberg runtime
# ---------------------------------------------------------------------------

@dataclass
class LimitOrder:
    order_id: str
    side: Side
    price: float
    quantity: int
    timestamp: float
    trader_id: str = "anon"
    agent_type: str = "generic"
    filled: int = 0
    canceled: bool = False
    iceberg_parent_id: Optional[str] = None
    is_iceberg_slice: bool = False

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled

    @property
    def is_done(self) -> bool:
        return self.remaining <= 0 or self.canceled


@dataclass
class IcebergRuntime:
    iceberg_id: str
    side: Side
    price: float
    peak_qty: int
    executed_qty: int
    display_min: int
    display_max: int
    refill_delay_lo: float
    refill_delay_hi: float
    active_slice_id: Optional[str]
    is_true: bool = True
    replenishment_count: int = 0
    volume_absorbed: int = 0
    first_post_time: Optional[float] = None
    last_event_time: Optional[float] = None
    slices_posted: int = 0
    price_jitter_prob: float = 0.0
    price_jitter_ticks: int = 1
    tick_size: float = 0.01
    skip_refill_prob: float = 0.0
    long_pause_prob: float = 0.0
    long_pause_lo: float = 1.0
    long_pause_hi: float = 5.0

    @property
    def remaining_parent(self) -> int:
        return max(0, self.peak_qty - self.executed_qty)

    def meaningful(self) -> bool:
        if self.peak_qty <= 0:
            return False
        if self.executed_qty == 0 and self.slices_posted == 0:
            return False
        active_t = 0.0
        if self.first_post_time is not None and self.last_event_time is not None:
            active_t = self.last_event_time - self.first_post_time
        return (
            self.replenishment_count >= MEANINGFUL_REPLENISH
            or self.volume_absorbed >= MEANINGFUL_ABSORB_VOL
            or active_t >= MEANINGFUL_ACTIVE_SEC
        )


# ---------------------------------------------------------------------------
# Price level & order book
# ---------------------------------------------------------------------------

class PriceLevel:
    __slots__ = ("price", "orders", "idmap", "total_qty")

    def __init__(self, price: float) -> None:
        self.price = price
        self.orders: Deque[LimitOrder] = deque()
        self.idmap: Dict[str, LimitOrder] = {}
        self.total_qty = 0

    def add(self, o: LimitOrder) -> None:
        self.orders.append(o)
        self.idmap[o.order_id] = o
        self.total_qty += o.remaining

    def remove(self, oid: str) -> Optional[LimitOrder]:
        o = self.idmap.pop(oid, None)
        if o is None:
            return None
        self.orders.remove(o)
        self.total_qty -= o.remaining
        return o

    def match(self, max_qty: int) -> List[Tuple[LimitOrder, int]]:
        out: List[Tuple[LimitOrder, int]] = []
        rem = max_qty
        while rem > 0 and self.orders:
            o = self.orders[0]
            take = min(rem, o.remaining)
            o.filled += take
            self.total_qty -= take
            rem -= take
            out.append((o, take))
            if o.is_done:
                self.orders.popleft()
                self.idmap.pop(o.order_id, None)
        return out


class OrderBook:
    def __init__(self, tick_size: float) -> None:
        self.tick_size = tick_size
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        self.all_orders: Dict[str, LimitOrder] = {}

    def round_p(self, p: float) -> float:
        return round(round(p / self.tick_size) * self.tick_size, 10)

    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return round((bb + ba) / 2.0, 10)

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return round(ba - bb, 10)

    def depth_side(self, side: Side, n: int) -> List[Tuple[float, int]]:
        if side == Side.BUY:
            prices = sorted(self.bids.keys(), reverse=True)[:n]
            return [(p, self.bids[p].total_qty) for p in prices]
        prices = sorted(self.asks.keys())[:n]
        return [(p, self.asks[p].total_qty) for p in prices]

    def total_depth(self, side: Side) -> int:
        return sum(lv.total_qty for lv in (self.bids.values() if side == Side.BUY else self.asks.values()))

    def add_limit(self, o: LimitOrder) -> None:
        p = self.round_p(o.price)
        o.price = p
        book = self.bids if o.side == Side.BUY else self.asks
        if p not in book:
            book[p] = PriceLevel(p)
        book[p].add(o)
        self.all_orders[o.order_id] = o

    def cancel(self, oid: str) -> Optional[LimitOrder]:
        o = self.all_orders.get(oid)
        if o is None or o.canceled:
            return None
        o.canceled = True
        book = self.bids if o.side == Side.BUY else self.asks
        lv = book.get(o.price)
        if lv:
            lv.remove(oid)
            if lv.total_qty == 0 and len(lv.orders) == 0:
                del book[o.price]
        self.all_orders.pop(oid, None)
        return o

    def _match_side(self, side_book: Dict[float, PriceLevel], pick_best, limit_price: float,
                    max_qty: int, check_fn) -> List[Tuple[LimitOrder, int, float]]:
        matches: List[Tuple[LimitOrder, int, float]] = []
        rem = max_qty
        while rem > 0:
            bp = pick_best()
            if bp is None or not check_fn(bp, limit_price):
                break
            lv = side_book[bp]
            for o, q in lv.match(rem):
                matches.append((o, q, bp))
                rem -= q
                if o.is_done:
                    self.all_orders.pop(o.order_id, None)
            if bp in side_book and side_book[bp].total_qty == 0 and len(side_book[bp].orders) == 0:
                del side_book[bp]
        return matches

    def match_aggressive_buy(self, limit_price: float, max_qty: int) -> List[Tuple[LimitOrder, int, float]]:
        return self._match_side(self.asks, lambda: self.best_ask, limit_price, max_qty,
                                lambda bp, lp: bp <= lp + 1e-9)

    def match_aggressive_sell(self, limit_price: float, max_qty: int) -> List[Tuple[LimitOrder, int, float]]:
        return self._match_side(self.bids, lambda: self.best_bid, limit_price, max_qty,
                                lambda bp, lp: lp >= float("inf") or bp >= lp - 1e-9)


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------

class EventQueue:
    __slots__ = ("_h", "_seq", "t")

    def __init__(self) -> None:
        self._h: List[Tuple[float, int, Callable[[], None]]] = []
        self._seq = 0
        self.t = 0.0

    def schedule(self, timestamp: float, fn: Callable[[], None]) -> None:
        self._seq += 1
        heapq.heappush(self._h, (timestamp, self._seq, fn))

    def run_until(self, end: float) -> None:
        while self._h and self._h[0][0] <= end:
            ts, _, fn = heapq.heappop(self._h)
            self.t = ts
            fn()


# ---------------------------------------------------------------------------
# Rolling activity
# ---------------------------------------------------------------------------

@dataclass
class ActivityEvent:
    t: float
    kind: str
    side: str
    qty: int
    price: float


class RollingActivity:
    def __init__(self, window: float) -> None:
        self.window = window
        self.events: Deque[ActivityEvent] = deque()

    def push(self, ev: ActivityEvent) -> None:
        self.events.append(ev)
        self._trim(ev.t)

    def _trim(self, now: float) -> None:
        cut = now - self.window
        while self.events and self.events[0].t < cut:
            self.events.popleft()

    def snapshot_stats(self, now: float) -> Dict[str, float]:
        self._trim(now)
        buy_v = sell_v = adds = cancels = trades = 0
        for e in self.events:
            if e.kind == "trade":
                trades += 1
                if e.side == Side.BUY.value:
                    buy_v += e.qty
                else:
                    sell_v += e.qty
            elif e.kind == "add":
                adds += 1
            elif e.kind == "cancel":
                cancels += 1
        tot_v = buy_v + sell_v
        return {
            "recent_traded_volume": float(tot_v),
            "recent_buy_volume": float(buy_v),
            "recent_sell_volume": float(sell_v),
            "recent_trade_count": float(trades),
            "recent_add_count": float(adds),
            "recent_cancel_count": float(cancels),
            "order_flow_imbalance": float((buy_v - sell_v) / (tot_v + 1e-9)),
        }


# ---------------------------------------------------------------------------
# Regime shifts
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    volatility_mult: float = 1.0
    imbalance: float = 0.0
    burstiness: float = 0.0
    price_std: float = 0.1
    cancel_intensity: float = 0.2


def _plan_regime_shifts(rng: random.Random, duration: float,
                        base_imb: float, base_burst: float,
                        base_pstd: float, base_cancel: float,
                        *, force_shifts: Optional[int] = None) -> List[Tuple[float, RegimeState]]:
    n = force_shifts if force_shifts is not None else rng.choices([0, 1, 2, 3, 4], weights=[.20, .30, .25, .15, .10])[0]
    if n == 0:
        return []
    times = sorted(rng.uniform(duration * .10, duration * .90) for _ in range(n))
    out: List[Tuple[float, RegimeState]] = []
    for t in times:
        vm = rng.choice([0.4, 0.6, 1.0, 1.5, 2.0, 3.0])
        out.append((t, RegimeState(
            volatility_mult=vm,
            imbalance=_clamp(base_imb + rng.uniform(-.6, .6), -.95, .95),
            burstiness=_clamp(base_burst + rng.uniform(-.3, .3), 0.0, 1.0),
            price_std=max(.01, base_pstd * vm),
            cancel_intensity=_clamp(base_cancel + rng.uniform(-.2, .2), 0.0, 1.0),
        )))
    return out


# ---------------------------------------------------------------------------
# Simulator core
# ---------------------------------------------------------------------------

@dataclass
class SimStats:
    total_orders: int = 0
    total_trades: int = 0
    total_volume: int = 0
    total_cancels: int = 0


class MarketSim:
    def __init__(self, rng: random.Random, tick_size: float, end_time: float,
                 iceberg_truth: Dict[str, IcebergRuntime], activity: RollingActivity) -> None:
        self.rng = rng
        self.book = OrderBook(tick_size)
        self.eq = EventQueue()
        self.end_time = end_time
        self.icebergs = iceberg_truth
        self.activity = activity
        self.stats = SimStats()
        self._oid = 0
        self.current_time = 0.0

    def _nid(self) -> str:
        self._oid += 1
        return f"o{self._oid}"

    def submit_limit(self, o: LimitOrder) -> None:
        o.timestamp = self.current_time
        chunk = (self.book.match_aggressive_buy(o.price, o.remaining) if o.side == Side.BUY
                 else self.book.match_aggressive_sell(o.price, o.remaining))
        for rest, q, px in chunk:
            o.filled += q
            self._on_trade(o, rest, q, px)
        if o.remaining > 0:
            self.book.add_limit(o)
            self.stats.total_orders += 1
            self.activity.push(ActivityEvent(self.current_time, "add", o.side.value, o.remaining, o.price))

    def _on_trade(self, aggressor: LimitOrder, resting: LimitOrder, qty: int, price: float) -> None:
        self.stats.total_trades += 1
        self.stats.total_volume += qty
        self.activity.push(ActivityEvent(self.current_time, "trade", aggressor.side.value, qty, price))
        pid = resting.iceberg_parent_id
        if pid and pid in self.icebergs:
            st = self.icebergs[pid]
            st.executed_qty += qty
            st.volume_absorbed += qty
            st.last_event_time = self.current_time
            if st.first_post_time is None:
                st.first_post_time = self.current_time
            if resting.is_iceberg_slice and resting.is_done:
                st.active_slice_id = None
                if st.remaining_parent > 0:
                    self._schedule_refill(st)

    def _schedule_refill(self, st: IcebergRuntime) -> None:
        if self.rng.random() < st.skip_refill_prob:
            d = self.rng.uniform(st.refill_delay_hi, st.refill_delay_hi * 3.0)
        elif self.rng.random() < st.long_pause_prob:
            d = self.rng.uniform(st.long_pause_lo, st.long_pause_hi)
        else:
            d = self.rng.uniform(st.refill_delay_lo, st.refill_delay_hi)
        self.eq.schedule(self.current_time + d, lambda s=st: self._refill(s))

    def _refill(self, st: IcebergRuntime) -> None:
        self.current_time = self.eq.t
        if st.remaining_parent <= 0 or st.active_slice_id is not None:
            return
        bv = self.rng.randint(st.display_min, st.display_max)
        if self.rng.random() < .15:
            bv = max(1, int(bv * self.rng.uniform(.3, .7)))
        elif self.rng.random() < .10:
            bv = int(bv * self.rng.uniform(1.2, 1.8))
        q = min(bv, st.remaining_parent)
        if q <= 0:
            return
        st.slices_posted += 1
        if st.slices_posted > 1:
            st.replenishment_count += 1
        price = st.price
        if self.rng.random() < st.price_jitter_prob:
            jt = self.rng.randint(-st.price_jitter_ticks, st.price_jitter_ticks)
            price = self.book.round_p(price + jt * st.tick_size)
            if price <= 0:
                price = st.price
            st.price = price
        oid = self._nid()
        o = LimitOrder(oid, st.side, price, q, self.current_time, f"ib_{st.iceberg_id}",
                       "iceberg_true", iceberg_parent_id=st.iceberg_id, is_iceberg_slice=True)
        st.active_slice_id = oid
        self.submit_limit(o)

    def market_order(self, side: Side, qty: int, agent_type: str = "market") -> None:
        agg = side
        chunk = (self.book.match_aggressive_buy(float("inf"), qty) if side == Side.BUY
                 else self.book.match_aggressive_sell(float("inf"), qty))
        for rest, q, px in chunk:
            pseudo = LimitOrder(self._nid(), agg, px, q, self.current_time, "mkt", agent_type)
            pseudo.filled = q
            self._on_trade(pseudo, rest, q, px)

    def cancel_random_visible(self) -> bool:
        cands = [o for o in self.book.all_orders.values() if not o.is_iceberg_slice]
        if not cands:
            return False
        o = self.rng.choice(cands)
        self.book.cancel(o.order_id)
        self.stats.total_cancels += 1
        self.activity.push(ActivityEvent(self.current_time, "cancel", o.side.value, o.remaining, o.price))
        return True


# ---------------------------------------------------------------------------
# Ladder & seeding
# ---------------------------------------------------------------------------

def build_ladder(mid: float, min_spread: float, tick: float, n_levels: int) -> Tuple[List[float], List[float]]:
    half = max(min_spread / 2.0, tick)
    step = max(tick, 0.01)
    k = max(2, min(n_levels, int(half / step) + 2))
    bids: List[float] = []
    asks: List[float] = []
    for i in range(1, k + 1):
        b = mid - i * step
        a = mid + i * step
        if b > 0:
            bids.append(round(b / tick) * tick)
        asks.append(round(a / tick) * tick)
    return sorted(set(bids), reverse=True), sorted(set(asks))


def _liquidity_profile(rng: random.Random, n: int) -> Tuple[List[float], str]:
    style = rng.choices(["flat", "decay_linear", "decay_exp", "humped", "random", "inverted"],
                        weights=[.20, .25, .20, .15, .10, .10])[0]
    if style == "flat":
        m = [1.0] * n
    elif style == "decay_linear":
        m = [max(.1, 1.0 - i * .8 / max(1, n - 1)) for i in range(n)]
    elif style == "decay_exp":
        r = rng.uniform(.15, .5)
        m = [math.exp(-r * i) for i in range(n)]
    elif style == "humped":
        pk = rng.randint(1, max(1, n // 2))
        m = [max(.1, 1.0 - abs(i - pk) * .3) for i in range(n)]
    elif style == "inverted":
        m = [max(.15, .2 + i * .8 / max(1, n - 1)) for i in range(n)]
    else:
        m = [rng.uniform(.2, 1.5) for _ in range(n)]
    return m, style


def seed_liquidity(sim: MarketSim, bids: List[float], asks: List[float],
                   base_qty: int, rng: random.Random, *, granular_prob: float = .35) -> str:
    pb, style = _liquidity_profile(rng, len(bids))
    pa, _ = _liquidity_profile(rng, len(asks))
    granular = rng.random() < granular_prob

    def _post(price: float, side: Side, mult: float) -> None:
        total = max(1, int(base_qty * mult))
        if granular:
            no = rng.randint(3, 12)
            for _ in range(no):
                sz = max(1, total // no + rng.randint(-total // (no * 3 + 1), total // (no * 3 + 1)))
                sim.submit_limit(LimitOrder(sim._nid(), side, price, sz, sim.current_time, "lp", "liquidity_provider"))
        else:
            sim.submit_limit(LimitOrder(sim._nid(), side, price, total, sim.current_time, "lp", "liquidity_provider"))

    for i, p in enumerate(bids):
        _post(p, Side.BUY, pb[i] if i < len(pb) else .5)
    for j, p in enumerate(asks):
        _post(p, Side.SELL, pa[j] if j < len(pa) else .5)
    return style


# ---------------------------------------------------------------------------
# Market maker
# ---------------------------------------------------------------------------

@dataclass
class MarketMakerState:
    trader_id: str
    half_spread_ticks: int
    qty_lo: int
    qty_hi: int
    update_lo: float
    update_hi: float
    bid_id: Optional[str] = None
    ask_id: Optional[str] = None


def _mm_tick(sim: MarketSim, mm: MarketMakerState, dur: float) -> None:
    sim.current_time = sim.eq.t
    if sim.current_time > dur:
        return
    for oid in (mm.bid_id, mm.ask_id):
        if oid and oid in sim.book.all_orders:
            sim.book.cancel(oid)
            sim.stats.total_cancels += 1
    mm.bid_id = mm.ask_id = None
    mid = sim.book.mid
    if mid is None:
        sim.eq.schedule(sim.current_time + sim.rng.uniform(mm.update_lo, mm.update_hi),
                        lambda: _mm_tick(sim, mm, dur))
        return
    tk = sim.book.tick_size
    bp = sim.book.round_p(mid - mm.half_spread_ticks * tk)
    ap = sim.book.round_p(mid + mm.half_spread_ticks * tk)
    if bp > 0:
        bo = LimitOrder(sim._nid(), Side.BUY, bp, sim.rng.randint(mm.qty_lo, mm.qty_hi),
                        sim.current_time, mm.trader_id, "market_maker")
        sim.submit_limit(bo)
        mm.bid_id = bo.order_id
    ao = LimitOrder(sim._nid(), Side.SELL, ap, sim.rng.randint(mm.qty_lo, mm.qty_hi),
                    sim.current_time, mm.trader_id, "market_maker")
    sim.submit_limit(ao)
    mm.ask_id = ao.order_id
    sim.eq.schedule(sim.current_time + sim.rng.uniform(mm.update_lo, mm.update_hi),
                    lambda: _mm_tick(sim, mm, dur))


# ---------------------------------------------------------------------------
# Fake iceberg
# ---------------------------------------------------------------------------

@dataclass
class FakeIcebergState:
    trader_id: str
    side: Side
    base_price: float
    qty_lo: int
    qty_hi: int
    refill_lo: float
    refill_hi: float
    lifetime: float
    start_time: float
    active_oid: Optional[str] = None
    reposts: int = 0
    jitter_ticks: int = 1
    jitter_prob: float = .12
    tick_size: float = .01


def _fake_ice_post(sim: MarketSim, fs: FakeIcebergState) -> None:
    sim.current_time = sim.eq.t
    if sim.current_time - fs.start_time > fs.lifetime or sim.current_time > sim.end_time:
        return
    if fs.active_oid and fs.active_oid in sim.book.all_orders:
        sim.book.cancel(fs.active_oid)
        sim.stats.total_cancels += 1
        sim.activity.push(ActivityEvent(sim.current_time, "cancel", fs.side.value, 0, fs.base_price))
    fs.active_oid = None
    price = fs.base_price
    if sim.rng.random() < fs.jitter_prob:
        jt = sim.rng.randint(-fs.jitter_ticks, fs.jitter_ticks)
        price = sim.book.round_p(price + jt * fs.tick_size)
        if price <= 0:
            price = fs.base_price
    o = LimitOrder(sim._nid(), fs.side, price, sim.rng.randint(fs.qty_lo, fs.qty_hi),
                   sim.current_time, fs.trader_id, "fake_iceberg")
    sim.submit_limit(o)
    fs.active_oid = o.order_id
    fs.reposts += 1
    sim.eq.schedule(sim.current_time + sim.rng.uniform(fs.refill_lo, fs.refill_hi),
                    lambda: _fake_ice_post(sim, fs))


# ---------------------------------------------------------------------------
# Positive support
# ---------------------------------------------------------------------------

@dataclass
class _SupportPlan:
    iceberg_side: Side
    iceberg_price: float
    inject_time: float
    strength: float
    tick_size: float


def _schedule_support_flow(sim: MarketSim, plan: _SupportPlan, dur: float, rng: random.Random) -> None:
    agg = Side.SELL if plan.iceberg_side == Side.BUY else Side.BUY
    s = plan.strength

    # pre-arrival depth
    pre_w = rng.uniform(.5, 3.0) * (.5 + s)
    pre_s = max(.01, plan.inject_time - pre_w)
    for _ in range(max(1, int(rng.uniform(3, 10) * (.3 + s)))):
        t = rng.uniform(pre_s, plan.inject_time)
        def _pre(t_=t):
            sim.current_time = sim.eq.t
            pr = sim.book.round_p(plan.iceberg_price + rng.choice([-1, 0, 1, 2]) * plan.tick_size)
            sim.submit_limit(LimitOrder(sim._nid(), plan.iceberg_side, pr,
                                        rng.randint(15, int(80 * (.5 + s))),
                                        sim.current_time, f"flow_{rng.randint(0,999)}", "liquidity_provider"))
        sim.eq.schedule(t, _pre)

    # aggressive bursts
    post_w = rng.uniform(2.0, 8.0) * (.5 + s)
    post_end = min(dur, plan.inject_time + post_w)
    for _ in range(max(2, int(rng.uniform(4, 18) * (.3 + s)))):
        t = rng.uniform(plan.inject_time + .01, post_end)
        def _burst(t_=t):
            sim.current_time = sim.eq.t
            qty = rng.randint(10, int(120 * (.4 + s)))
            mode = rng.choices(["market", "cross", "near"], weights=[.45, .30, .25])[0]
            if mode == "market":
                sim.market_order(agg, qty, "liquidity_taker")
            elif mode == "cross":
                if agg == Side.BUY:
                    ref = sim.book.best_ask or plan.iceberg_price
                    pr = ref + rng.randint(0, 3) * plan.tick_size
                else:
                    ref = sim.book.best_bid or plan.iceberg_price
                    pr = ref - rng.randint(0, 3) * plan.tick_size
                sim.submit_limit(LimitOrder(sim._nid(), agg, max(plan.tick_size, sim.book.round_p(pr)),
                                            qty, sim.current_time, f"flow_{rng.randint(0,999)}", "directional"))
            else:
                if agg == Side.BUY:
                    ref = sim.book.best_bid or plan.iceberg_price
                    pr = ref + rng.randint(0, 2) * plan.tick_size
                else:
                    ref = sim.book.best_ask or plan.iceberg_price
                    pr = ref - rng.randint(0, 2) * plan.tick_size
                sim.submit_limit(LimitOrder(sim._nid(), agg, max(plan.tick_size, sim.book.round_p(pr)),
                                            qty, sim.current_time, f"flow_{rng.randint(0,999)}", "directional"))
        sim.eq.schedule(t, _burst)

    # trickle
    trickle_end = min(dur, plan.inject_time + post_w * rng.uniform(1.5, 3.0))
    for _ in range(max(1, int(rng.uniform(2, 8) * s))):
        t = rng.uniform(post_end, trickle_end) if trickle_end > post_end else post_end
        def _trickle(t_=t):
            sim.current_time = sim.eq.t
            sim.market_order(agg, rng.randint(5, int(60 * (.3 + s))), "noise")
        sim.eq.schedule(t, _trickle)


# ---------------------------------------------------------------------------
# Episode params & truth
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTruth:
    icebergs: Dict[str, IcebergRuntime]
    episode_label_iceberg: bool
    buy_iceberg_count: int
    sell_iceberg_count: int
    meaningful_iceberg_count: int


@dataclass
class EpisodeParams:
    episode_id: str
    seed: int
    duration: float
    mid0: float
    tick_size: float
    min_spread: float
    snapshot_interval: float
    depth_book: int
    max_depth_csv: int
    lp_per_level: int
    base_order_rate: float
    imbalance: float
    cluster_prob: float
    price_std: float
    volatility: str
    cancel_intensity: float
    burstiness: float
    trend_strength: float
    regime_tags: List[str]
    inject_true_icebergs: bool
    num_icebergs: int
    iceberg_sides: List[Side]
    iceberg_peak_lo: int
    iceberg_peak_hi: int
    iceberg_vis_lo: int
    iceberg_vis_hi: int
    iceberg_refill_lo: float
    iceberg_refill_hi: float
    hard_negative_mix: Dict[str, float]
    iceberg_price_jitter_prob: float = 0.0
    iceberg_price_jitter_ticks: int = 1
    iceberg_skip_refill_prob: float = 0.0
    iceberg_long_pause_prob: float = 0.0
    iceberg_long_pause_lo: float = 1.0
    iceberg_long_pause_hi: float = 5.0
    num_market_makers: int = 0
    num_fake_icebergs: int = 0
    granular_liq_prob: float = .35
    regime_shifts: List[Tuple[float, RegimeState]] = field(default_factory=list)
    iceberg_timing_style: str = "none"
    iceberg_placement_style: str = "none"
    liquidity_profile_style: str = "unknown"
    difficulty_bucket: str = "unknown"
    imbalance_regime: str = "neutral"
    burstiness_regime: str = "normal"
    had_positive_support: bool = False
    positive_support_strength: float = 0.0
    # v5
    generation_phase: str = "unknown"   # "iceberg_phase" | "noniceberg_phase" | "legacy"
    candidate_attempt_index: int = 0
    saved_episode_index: int = 0


# ---------------------------------------------------------------------------
# Difficulty & regime classifiers
# ---------------------------------------------------------------------------

def _classify_difficulty(ep: EpisodeParams, truth: EpisodeTruth) -> str:
    if truth.episode_label_iceberg:
        sc = 0
        n_m = truth.meaningful_iceberg_count
        tr = sum(st.replenishment_count for st in truth.icebergs.values() if st.is_true and st.meaningful())
        if n_m >= 2: sc += 2
        if tr >= 5: sc += 2
        elif tr >= 2: sc += 1
        if ep.num_fake_icebergs > 0: sc -= 2
        if ep.regime_shifts: sc -= 1
        if ep.volatility == "noisy": sc -= 1
        if ep.num_market_makers >= 2: sc -= 1
        if sc >= 3: return DIFFICULTY_EASY_POS
        if sc >= 1: return DIFFICULTY_MED_POS
        return DIFFICULTY_HARD_POS
    else:
        c = 0
        if ep.num_fake_icebergs >= 2: c += 2
        elif ep.num_fake_icebergs >= 1: c += 1
        hn = ep.hard_negative_mix
        if hn.get("multi_refill", 0) > .15: c += 1
        if hn.get("sliced_visible", 0) > .15: c += 1
        if hn.get("churn", 0) > .20: c += 1
        if len(ep.regime_shifts) >= 2: c += 1
        return DIFFICULTY_HARD_NEG if c >= 3 else DIFFICULTY_EASY_NEG


def _classify_imbalance(v: float) -> str:
    if v < -.4: return "strong_sell"
    if v < -.15: return "mild_sell"
    if v <= .15: return "neutral"
    if v <= .4: return "mild_buy"
    return "strong_buy"


def _classify_burstiness(v: float) -> str:
    if v < .15: return "calm"
    if v < .4: return "normal"
    return "bursty"


# ---------------------------------------------------------------------------
# Episode sampling  (phase-aware)
# ---------------------------------------------------------------------------

def _sample_episode_config(
    ep_index: int, base_seed: int, rng: random.Random, *,
    max_depth_levels: int, snap_bounds: Tuple[float, float],
    min_rows: int, max_rows: int,
    controls: GenerationControls,
    phase: str,  # "iceberg_phase" | "noniceberg_phase" | "legacy"
    candidate_attempt: int,
    saved_index: int,
) -> EpisodeParams:
    ctrl = controls
    seed = (base_seed ^ (ep_index * 1_000_003) ^ rng.randint(0, 2**20)) & 0xFFFFFFFF
    e = random.Random(seed)

    duration_base = e.choice([20., 40., 60., 90., 120., 180., 240.])
    mid0 = round(e.uniform(15., 300.), 2)
    tick_size = e.choice([.01, .02, .05])
    min_spread = e.uniform(.2, 1.5)
    snap_lo, snap_hi = snap_bounds
    snapshot_interval = round(e.uniform(snap_lo, snap_hi), 4)
    target_rows = e.randint(min_rows, max_rows)
    duration = max(duration_base, target_rows * snapshot_interval * .95)
    depth_book = e.randint(4, max(5, max_depth_levels + 4))
    lp = int(_loguniform(e, 3000., 90000.))
    base_rate = _loguniform(e, .8, 22.)
    imbalance = e.uniform(-.88, .88)
    cluster = e.betavariate(2., 5.)
    if e.random() < .12:
        cluster = e.uniform(.45, .95)

    vol = e.choice(["calm", "normal", "noisy"])
    px_std = {"calm": e.uniform(.02, .07), "normal": e.uniform(.05, .14), "noisy": e.uniform(.1, .25)}[vol]
    cancel_i = e.betavariate(2., 6.)
    burst = e.betavariate(2., 5.)
    trend = e.uniform(-1., 1.)

    # --- phase-specific injection bias ---
    if phase == "iceberg_phase":
        # strongly bias toward injecting icebergs
        inject = e.random() < 0.95
    elif phase == "noniceberg_phase":
        # strongly bias toward NOT injecting
        inject = e.random() < 0.05
    else:
        # legacy mode — use target_class_balance if set
        rate = ctrl.target_class_balance if ctrl.target_class_balance is not None else 0.45
        inject = e.random() < rate

    n_ice = 0
    if inject:
        n_ice = e.choices([1, 2, 3, 4], weights=[.45, .3, .15, .1])[0]
        if ctrl.target_multi_iceberg_freq is not None and n_ice == 1:
            if e.random() < ctrl.target_multi_iceberg_freq:
                n_ice = e.choices([2, 3, 4], weights=[.5, .3, .2])[0]

    sides = [Side.BUY if e.random() < .5 else Side.SELL for _ in range(n_ice)]
    peak_lo = int(_loguniform(e, 600., 12000.))
    peak_hi = max(peak_lo + 50, int(_loguniform(e, 4000., 60000.)))
    vis_lo = e.randint(5, 80)
    vis_hi = max(vis_lo + 1, e.randint(vis_lo, 220))
    ref_lo = e.uniform(.008, .12)
    ref_hi = max(ref_lo + .02, e.uniform(.06, 1.5))

    pj_prob = e.uniform(0., .25)
    pj_ticks = e.randint(1, 3)
    skip_prob = e.uniform(0., .12)
    lp_prob = e.uniform(0., .20)
    lp_lo = ref_hi * e.uniform(1.5, 3.)
    lp_hi = lp_lo * e.uniform(1.5, 4.)

    hard_neg: Dict[str, float] = {
        "large_visible": e.uniform(.05, .35),
        "sliced_visible": e.uniform(.05, .3),
        "churn": e.uniform(.05, .35),
        "multi_refill": e.uniform(.05, .28),
        "noise": e.uniform(.15, .5),
        "directional": e.uniform(.2, .55),
    }

    # --- hard negative boosting in noniceberg phase ---
    if phase == "noniceberg_phase":
        if ctrl.target_hard_negative_freq is not None and e.random() < ctrl.target_hard_negative_freq:
            hard_neg["multi_refill"] = max(hard_neg["multi_refill"], e.uniform(.20, .35))
            hard_neg["sliced_visible"] = max(hard_neg["sliced_visible"], e.uniform(.18, .32))
            hard_neg["churn"] = max(hard_neg["churn"], e.uniform(.22, .40))

    n_mm = e.choices([0, 1, 2, 3], weights=[.15, .40, .30, .15])[0]

    # fake icebergs: more in noniceberg phase
    if phase == "noniceberg_phase" or (not inject):
        n_fake = e.choices([0, 1, 2, 3, 4], weights=[.08, .22, .30, .22, .18])[0]
        if ctrl.target_fake_iceberg_freq is not None and n_fake == 0:
            if e.random() < ctrl.target_fake_iceberg_freq:
                n_fake = e.randint(1, 3)
    else:
        n_fake = e.choices([0, 1, 2], weights=[.50, .30, .20])[0]

    gran = e.uniform(.15, .60)

    force_shifts: Optional[int] = None
    if ctrl.target_regime_shift_freq is not None:
        if e.random() < ctrl.target_regime_shift_freq:
            force_shifts = e.randint(1, 3)
    shifts = _plan_regime_shifts(e, duration, imbalance, burst, px_std, cancel_i, force_shifts=force_shifts)

    if inject and n_ice > 0:
        ice_timing = e.choices(["uniform_random", "clustered", "early_bias", "late_bias"],
                               weights=[.35, .30, .20, .15])[0]
        ice_place = e.choices(["at_bbo", "near_bbo", "deeper", "inside_spread", "mixed"],
                              weights=[.20, .25, .20, .10, .25])[0]
    else:
        ice_timing = "none"
        ice_place = "none"

    had_support = False
    support_str = 0.0
    if inject and n_ice > 0 and e.random() < ctrl.positive_support_prob:
        had_support = True
        support_str = _clamp(ctrl.positive_support_strength * e.uniform(.5, 1.5), .05, 1.0)

    tags = [vol, f"cx_{cluster:.2f}", f"imb_{imbalance:+.2f}"]
    if not inject: tags.append("no_true_iceberg")
    if n_fake > 0: tags.append(f"fake_ice_{n_fake}")
    if shifts: tags.append(f"shifts_{len(shifts)}")
    if had_support: tags.append(f"support_{support_str:.2f}")

    return EpisodeParams(
        episode_id=f"ep_{saved_index:06d}",
        seed=seed, duration=duration, mid0=mid0, tick_size=tick_size,
        min_spread=min_spread, snapshot_interval=snapshot_interval,
        depth_book=depth_book, max_depth_csv=max_depth_levels,
        lp_per_level=lp, base_order_rate=base_rate, imbalance=imbalance,
        cluster_prob=cluster, price_std=px_std, volatility=vol,
        cancel_intensity=cancel_i, burstiness=burst, trend_strength=trend,
        regime_tags=tags,
        inject_true_icebergs=inject and n_ice > 0, num_icebergs=n_ice,
        iceberg_sides=sides, iceberg_peak_lo=peak_lo, iceberg_peak_hi=peak_hi,
        iceberg_vis_lo=vis_lo, iceberg_vis_hi=vis_hi,
        iceberg_refill_lo=ref_lo, iceberg_refill_hi=ref_hi,
        hard_negative_mix=hard_neg,
        iceberg_price_jitter_prob=pj_prob, iceberg_price_jitter_ticks=pj_ticks,
        iceberg_skip_refill_prob=skip_prob,
        iceberg_long_pause_prob=lp_prob, iceberg_long_pause_lo=lp_lo, iceberg_long_pause_hi=lp_hi,
        num_market_makers=n_mm, num_fake_icebergs=n_fake,
        granular_liq_prob=gran, regime_shifts=shifts,
        iceberg_timing_style=ice_timing, iceberg_placement_style=ice_place,
        liquidity_profile_style="unknown", difficulty_bucket="unknown",
        imbalance_regime=_classify_imbalance(imbalance),
        burstiness_regime=_classify_burstiness(burst),
        had_positive_support=had_support, positive_support_strength=support_str,
        generation_phase=phase,
        candidate_attempt_index=candidate_attempt,
        saved_episode_index=saved_index,
    )


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(ep: EpisodeParams) -> Tuple[List[Dict[str, Any]], EpisodeTruth, Dict[str, Any]]:
    rng = random.Random(ep.seed)
    activity = RollingActivity(ROLLING_WINDOW_SEC)
    ice_truth: Dict[str, IcebergRuntime] = {}
    sim = MarketSim(rng, ep.tick_size, ep.duration, ice_truth, activity)
    sim.eq.t = sim.current_time = 0.0

    bid_ladder, ask_ladder = build_ladder(ep.mid0, ep.min_spread, ep.tick_size, ep.depth_book)
    if not bid_ladder or not ask_ladder:
        raise RuntimeError("empty ladder")

    snapshots: List[Dict[str, Any]] = []
    snap_idx = 0
    regime = RegimeState(1.0, ep.imbalance, ep.burstiness, ep.price_std, ep.cancel_intensity)
    pending_shifts = list(ep.regime_shifts)
    churn_anchor = ep.mid0
    liq_style = "unknown"
    support_dir_bias = 0.0

    def _apply_shifts(now: float) -> None:
        nonlocal pending_shifts
        while pending_shifts and pending_shifts[0][0] <= now:
            _, nr = pending_shifts.pop(0)
            regime.volatility_mult = nr.volatility_mult
            regime.imbalance = nr.imbalance
            regime.burstiness = nr.burstiness
            regime.price_std = nr.price_std
            regime.cancel_intensity = nr.cancel_intensity

    def seed_fn() -> None:
        nonlocal liq_style
        sim.current_time = sim.eq.t
        liq_style = seed_liquidity(sim, bid_ladder, ask_ladder, ep.lp_per_level, rng,
                                   granular_prob=ep.granular_liq_prob)

    def _pick_ice_price(side: Side, hint: str) -> float:
        bb, ba = sim.book.best_bid, sim.book.best_ask
        if bb is None or ba is None:
            return sim.book.round_p(ep.mid0 + (-.05 if side == Side.BUY else .05))
        tk = ep.tick_size
        h = hint if hint != "mixed" else rng.choice(["at_bbo", "near_bbo", "deeper", "inside_spread"])
        if h == "at_bbo":
            return bb if side == Side.BUY else ba
        if h == "near_bbo":
            ot = rng.randint(1, 4)
            return sim.book.round_p((bb - ot * tk) if side == Side.BUY else (ba + ot * tk))
        if h == "deeper":
            ot = rng.randint(4, 12)
            return max(tk, sim.book.round_p((bb - ot * tk) if side == Side.BUY else (ba + ot * tk)))
        mid = (bb + ba) / 2.
        if side == Side.BUY:
            return sim.book.round_p(rng.uniform(bb, mid))
        return sim.book.round_p(rng.uniform(mid, ba))

    # injection times
    inj_times: List[float] = []
    if ep.inject_true_icebergs and ep.num_icebergs > 0:
        irng = random.Random(ep.seed + 777)
        mlo, mhi = ep.duration * .05, ep.duration * .90
        ts = ep.iceberg_timing_style
        if ts == "clustered":
            ctr = irng.uniform(mlo + ep.duration * .1, mhi - ep.duration * .1)
            sp = ep.duration * irng.uniform(.02, .15)
            inj_times = sorted(_clamp(irng.gauss(ctr, sp), mlo, mhi) for _ in range(ep.num_icebergs))
        elif ts == "early_bias":
            inj_times = sorted(irng.betavariate(2., 5.) * (mhi - mlo) + mlo for _ in range(ep.num_icebergs))
        elif ts == "late_bias":
            inj_times = sorted(irng.betavariate(5., 2.) * (mhi - mlo) + mlo for _ in range(ep.num_icebergs))
        else:
            inj_times = sorted(irng.uniform(mlo, mhi) for _ in range(ep.num_icebergs))

    def inject_iceberg(k: int) -> Callable[[], None]:
        def _() -> None:
            sim.current_time = sim.eq.t
            if k >= ep.num_icebergs:
                return
            side = ep.iceberg_sides[k]
            peak = rng.randint(ep.iceberg_peak_lo, ep.iceberg_peak_hi)
            oid = f"ice_{ep.episode_id}_{k}"
            price = _pick_ice_price(side, ep.iceberg_placement_style)
            st = IcebergRuntime(
                oid, side, price, peak, 0, max(1, ep.iceberg_vis_lo),
                max(ep.iceberg_vis_lo + 1, ep.iceberg_vis_hi),
                ep.iceberg_refill_lo, ep.iceberg_refill_hi, None, True,
                price_jitter_prob=ep.iceberg_price_jitter_prob,
                price_jitter_ticks=ep.iceberg_price_jitter_ticks,
                tick_size=ep.tick_size,
                skip_refill_prob=ep.iceberg_skip_refill_prob,
                long_pause_prob=ep.iceberg_long_pause_prob,
                long_pause_lo=ep.iceberg_long_pause_lo,
                long_pause_hi=ep.iceberg_long_pause_hi,
            )
            ice_truth[oid] = st
            vis = rng.randint(st.display_min, st.display_max)
            q0 = min(vis, st.remaining_parent)
            st.slices_posted = 1
            o0 = LimitOrder(sim._nid(), side, price, q0, sim.current_time,
                            f"ib_{oid}", "iceberg_true",
                            iceberg_parent_id=oid, is_iceberg_slice=True)
            st.active_slice_id = o0.order_id
            if st.first_post_time is None:
                st.first_post_time = sim.current_time
            sim.submit_limit(o0)
        return _

    def take_snapshot() -> None:
        nonlocal snap_idx
        sim.current_time = sim.eq.t
        _apply_shifts(sim.current_time)
        bb, ba = sim.book.best_bid, sim.book.best_ask
        mid, sp = sim.book.mid, sim.book.spread
        row: Dict[str, Any] = {
            "episode_id": ep.episode_id, "class_label": "", "timestamp": sim.current_time,
            "snapshot_idx": snap_idx,
            "best_bid": bb if bb is not None else _NAN,
            "best_ask": ba if ba is not None else _NAN,
            "mid_price": mid if mid is not None else _NAN,
            "spread": sp if sp is not None else _NAN,
        }
        db = sim.book.depth_side(Side.BUY, ep.max_depth_csv)
        da = sim.book.depth_side(Side.SELL, ep.max_depth_csv)
        for i in range(ep.max_depth_csv):
            bi = db[i] if i < len(db) else (_NAN, 0)
            ai = da[i] if i < len(da) else (_NAN, 0)
            row[f"bid_price_{i+1}"] = bi[0] if i < len(db) else _NAN
            row[f"bid_size_{i+1}"] = bi[1]
            row[f"ask_price_{i+1}"] = ai[0] if i < len(da) else _NAN
            row[f"ask_size_{i+1}"] = ai[1]
        row["total_bid_depth"] = sim.book.total_depth(Side.BUY)
        row["total_ask_depth"] = sim.book.total_depth(Side.SELL)
        row.update(activity.snapshot_stats(sim.current_time))
        snapshots.append(row)
        snap_idx += 1

    def agent_tick() -> None:
        sim.current_time = sim.eq.t
        _apply_shifts(sim.current_time)
        hm = ep.hard_negative_mix
        mk = list(hm.keys())
        mw = [hm[k] for k in mk]
        if rng.random() < regime.cancel_intensity * .35:
            sim.cancel_random_visible()
        nb = rng.randint(1, 3) if rng.random() < regime.burstiness * .4 else 1
        for _ in range(nb):
            if rng.random() < ep.cluster_prob:
                if rng.random() < .5:
                    p, sd = rng.choice(bid_ladder), Side.BUY
                else:
                    p, sd = rng.choice(ask_ladder), Side.SELL
                sim.submit_limit(LimitOrder(sim._nid(), sd, sim.book.round_p(p),
                                            rng.randint(5, max(6, ep.iceberg_vis_hi)),
                                            sim.current_time, "cluster", "cluster_trader"))
                continue
            mode = rng.choices(mk, weights=mw, k=1)[0]
            if mode == "large_visible":
                sd = Side.BUY if rng.random() < .5 else Side.SELL
                ref = sim.book.best_bid if sd == Side.BUY else sim.book.best_ask
                pr = (ref or ep.mid0) + (-ep.tick_size * rng.randint(0, 3) if sd == Side.BUY
                                          else ep.tick_size * rng.randint(0, 3))
                sim.submit_limit(LimitOrder(sim._nid(), sd, pr, rng.randint(2000, 25000),
                                            sim.current_time, "large", "large_visible"))
            elif mode == "sliced_visible":
                sd = Side.BUY if rng.random() < .5 else Side.SELL
                lad = bid_ladder if sd == Side.BUY else ask_ladder
                sim.submit_limit(LimitOrder(sim._nid(), sd, sim.book.round_p(rng.choice(lad)),
                                            rng.randint(20, 200), sim.current_time, "slice_a", "sliced_visible"))
            elif mode == "churn":
                sim.cancel_random_visible()
                sd = Side.BUY if rng.random() < .5 else Side.SELL
                sim.submit_limit(LimitOrder(sim._nid(), sd,
                    sim.book.round_p(churn_anchor + rng.choice([-1, 0, 1]) * ep.tick_size),
                    rng.randint(10, 120), sim.current_time, "churn", "churn_trader"))
            elif mode == "multi_refill":
                pr = sim.book.round_p(rng.choice(bid_ladder if rng.random() < .5 else ask_ladder))
                sd = Side.BUY if rng.random() < .5 else Side.SELL
                for tr in ("m1", "m2", "m3"):
                    sim.submit_limit(LimitOrder(sim._nid(), sd, pr, rng.randint(5, 60),
                                                sim.current_time, tr, "multi_refill_illusion"))
            elif mode == "noise":
                sd = Side.BUY if rng.random() < .5 else Side.SELL
                sim.submit_limit(LimitOrder(sim._nid(), sd,
                    sim.book.round_p(ep.mid0 + rng.uniform(-.5, .5) * 15 * ep.tick_size),
                    rng.randint(5, 150), sim.current_time, "noise", "noise"))
            else:
                if rng.random() < .55:
                    bias = regime.imbalance * .15 + support_dir_bias
                    sd = Side.BUY if rng.random() < .5 + bias else Side.SELL
                    pr = rng.gauss(ep.mid0 + (-regime.price_std * 2 if sd == Side.BUY
                                               else regime.price_std * 2), regime.price_std)
                    sim.submit_limit(LimitOrder(sim._nid(), sd, max(ep.tick_size, sim.book.round_p(pr)),
                                                rng.randint(8, 180), sim.current_time, "dir", "directional"))
                else:
                    sim.market_order(Side.BUY if rng.random() < .5 else Side.SELL,
                                     rng.randint(10, 400), "directional_market")

    # schedule
    sim.eq.schedule(0., seed_fn)
    t = 0.
    while t <= ep.duration:
        sim.eq.schedule(t, take_snapshot)
        t += ep.snapshot_interval

    rng_ag = random.Random(ep.seed + 404)
    lam = max(1., ep.base_order_rate * ep.duration * .85)
    n_ag = _poisson(lam, rng_ag)
    hi_t = max(.02, ep.duration - .001)
    for tt in sorted(rng_ag.uniform(.001, hi_t) for _ in range(min(n_ag, 25000))):
        sim.eq.schedule(tt, agent_tick)

    for k, it in enumerate(inj_times):
        sim.eq.schedule(it, inject_iceberg(k))
        if ep.had_positive_support:
            plan = _SupportPlan(ep.iceberg_sides[k], ep.mid0, it,
                                ep.positive_support_strength, ep.tick_size)
            _schedule_support_flow(sim, plan, ep.duration, random.Random(ep.seed + 5000 + k))
            per_bias = ep.positive_support_strength * .08
            if ep.iceberg_sides[k] == Side.BUY:
                support_dir_bias -= per_bias
            else:
                support_dir_bias += per_bias
    support_dir_bias = _clamp(support_dir_bias, -.25, .25)

    for mi in range(ep.num_market_makers):
        mr = random.Random(ep.seed + 900 + mi)
        mm = MarketMakerState(f"mm_{mi}", mr.randint(1, 4), mr.randint(10, 80),
                              mr.randint(80, 400), mr.uniform(.05, .5), mr.uniform(.5, 3.))
        sim.eq.schedule(mr.uniform(.01, ep.duration * .15), lambda m=mm: _mm_tick(sim, m, ep.duration))

    for fi in range(ep.num_fake_icebergs):
        fr = random.Random(ep.seed + 1200 + fi)
        fs_side = Side.BUY if fr.random() < .5 else Side.SELL
        off = fr.randint(0, 6) * ep.tick_size
        fp = sim.book.round_p((ep.mid0 - off) if fs_side == Side.BUY else (ep.mid0 + off))
        if fp <= 0:
            fp = ep.tick_size
        fs = FakeIcebergState(
            f"fakeice_{fi}", fs_side, fp, max(1, ep.iceberg_vis_lo),
            max(ep.iceberg_vis_lo + 1, ep.iceberg_vis_hi),
            ep.iceberg_refill_lo * fr.uniform(.6, 1.5),
            ep.iceberg_refill_hi * fr.uniform(.8, 2.),
            fr.uniform(ep.duration * .15, ep.duration * .85),
            fr.uniform(.01, ep.duration * .4),
            jitter_ticks=fr.randint(0, 2), jitter_prob=fr.uniform(0., .20),
            tick_size=ep.tick_size,
        )
        sim.eq.schedule(fs.start_time, lambda f=fs: _fake_ice_post(sim, f))

    sim.eq.run_until(ep.duration)

    ep = replace(ep, liquidity_profile_style=liq_style)

    meaningful = [st for st in ice_truth.values() if st.is_true and st.meaningful()]
    buy_ic = sum(1 for st in ice_truth.values() if st.side == Side.BUY)
    sell_ic = sum(1 for st in ice_truth.values() if st.side == Side.SELL)
    is_ice = len(meaningful) > 0 and ep.inject_true_icebergs
    label = "iceberg" if is_ice else "noniceberg"
    for row in snapshots:
        row["class_label"] = label

    truth = EpisodeTruth(ice_truth, is_ice, buy_ic, sell_ic, len(meaningful))
    diff = _classify_difficulty(ep, truth)
    ep = replace(ep, difficulty_bucket=diff)

    spreads = [r["spread"] for r in snapshots
               if isinstance(r.get("spread"), (int, float)) and r["spread"] == r["spread"]]
    depths = [r["total_bid_depth"] + r["total_ask_depth"] for r in snapshots]

    mx: Dict[str, Any] = {
        "num_trades": sim.stats.total_trades, "num_orders": sim.stats.total_orders,
        "total_volume": sim.stats.total_volume, "num_cancels": sim.stats.total_cancels,
        "num_rows": len(snapshots), "num_market_makers": ep.num_market_makers,
        "num_fake_icebergs": ep.num_fake_icebergs,
        "num_regime_shifts": len(ep.regime_shifts),
        "difficulty_bucket": diff,
        "iceberg_timing_style": ep.iceberg_timing_style,
        "iceberg_placement_style": ep.iceberg_placement_style,
        "liquidity_profile_style": ep.liquidity_profile_style,
        "imbalance_regime": ep.imbalance_regime,
        "burstiness_regime": ep.burstiness_regime,
        "had_positive_support": ep.had_positive_support,
        "positive_support_strength": round(ep.positive_support_strength, 4),
        "iceberg_became_meaningful": is_ice,
        "mean_spread": _safe_mean(spreads), "mean_depth": _safe_mean(depths),
        "generation_phase": ep.generation_phase,
        "candidate_attempt_index": ep.candidate_attempt_index,
        "saved_episode_index": ep.saved_episode_index,
    }
    return snapshots, truth, mx


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class AuditAccumulator:
    class_counts: Counter = field(default_factory=Counter)
    difficulty_counts: Counter = field(default_factory=Counter)
    difficulty_by_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    volatility_by_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    imbalance_regime_by_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    burstiness_regime_by_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    buy_ice: int = 0
    sell_ice: int = 0
    fake_by_class: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    mm_by_class: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    shift_by_class: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    timing_counts: Counter = field(default_factory=Counter)
    placement_counts: Counter = field(default_factory=Counter)
    liq_counts: Counter = field(default_factory=Counter)
    spreads: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    depths: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    rows: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    volumes: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    support_counts: Counter = field(default_factory=Counter)
    meaningful_pairs: List[Tuple[bool, bool]] = field(default_factory=list)

    def ingest(self, label: str, ep: EpisodeParams, truth: EpisodeTruth, mx: Dict[str, Any]) -> None:
        self.class_counts[label] += 1
        d = mx["difficulty_bucket"]
        self.difficulty_counts[d] += 1
        self.difficulty_by_class[label][d] += 1
        self.volatility_by_class[label][ep.volatility] += 1
        self.imbalance_regime_by_class[label][ep.imbalance_regime] += 1
        self.burstiness_regime_by_class[label][ep.burstiness_regime] += 1
        self.buy_ice += truth.buy_iceberg_count
        self.sell_ice += truth.sell_iceberg_count
        self.fake_by_class[label].append(ep.num_fake_icebergs)
        self.mm_by_class[label].append(ep.num_market_makers)
        self.shift_by_class[label].append(len(ep.regime_shifts))
        self.timing_counts[mx["iceberg_timing_style"]] += 1
        self.placement_counts[mx["iceberg_placement_style"]] += 1
        self.liq_counts[mx["liquidity_profile_style"]] += 1
        self.spreads[label].append(mx["mean_spread"])
        self.depths[label].append(mx["mean_depth"])
        self.rows[label].append(mx["num_rows"])
        self.volumes[label].append(mx["total_volume"])
        self.support_counts["supported" if ep.had_positive_support else "unsupported"] += 1
        self.meaningful_pairs.append((ep.inject_true_icebergs, mx["iceberg_became_meaningful"]))

    def _cd(self, c: Counter) -> Dict[str, int]:
        return dict(sorted(c.items()))

    def _regime_warnings(self, rbc: Dict[str, Counter], name: str) -> List[str]:
        w: List[str] = []
        cls = list(rbc.keys())
        if len(cls) < 2:
            return w
        ks: Set[str] = set()
        for c in rbc.values():
            ks.update(c.keys())
        for k in sorted(ks):
            vals = [rbc[cl].get(k, 0) for cl in cls]
            mn, mx = min(vals), max(vals)
            if mn == 0 and mx > 0:
                w.append(f"{name}='{k}' in only one class ({mx} vs 0)")
            elif mn > 0 and mx / mn > _IMBALANCE_WARN_RATIO:
                w.append(f"{name}='{k}' ratio {mx}/{mn}={mx/mn:.1f}x")
        return w

    def build_report(self) -> Dict[str, Any]:
        w: List[str] = []
        vals = list(self.class_counts.values())
        if len(vals) >= 2 and min(vals) > 0 and max(vals) / min(vals) > _IMBALANCE_WARN_RATIO:
            w.append(f"Class imbalance: {dict(self.class_counts)}, ratio {max(vals)/min(vals):.1f}x")
        w.extend(self._regime_warnings(self.volatility_by_class, "volatility"))
        w.extend(self._regime_warnings(self.imbalance_regime_by_class, "imbalance_regime"))
        w.extend(self._regime_warnings(self.burstiness_regime_by_class, "burstiness_regime"))

        def _ds(bc, cast=float):
            return {cl: {"mean": round(_safe_mean([cast(v) for v in vs]), 4),
                         "std": round(_safe_stdev([cast(v) for v in vs]), 4), "n": len(vs)}
                    for cl, vs in bc.items()}

        for fn, bc in [("mean_spread", self.spreads), ("mean_depth", self.depths),
                       ("num_rows", self.rows), ("total_volume", self.volumes)]:
            ms = {cl: _safe_mean([float(v) for v in vs]) for cl, vs in bc.items() if vs}
            if len(ms) >= 2:
                mn, mx = min(ms.values()), max(ms.values())
                if mn > 0 and mx / mn > 2.5:
                    w.append(f"{fn} mean >2.5x across classes: {dict((k,round(v,2)) for k,v in ms.items())}")

        inj = sum(1 for i, _ in self.meaningful_pairs if i)
        moi = sum(1 for i, m in self.meaningful_pairs if i and m)
        mr = moi / max(1, inj)
        if inj > 10 and mr < .50:
            w.append(f"Low meaningfulness: {moi}/{inj} = {mr:.1%}")

        return {
            "class_counts": self._cd(self.class_counts),
            "difficulty_counts": self._cd(self.difficulty_counts),
            "difficulty_by_class": {c: self._cd(v) for c, v in self.difficulty_by_class.items()},
            "volatility_by_class": {c: self._cd(v) for c, v in self.volatility_by_class.items()},
            "imbalance_regime_by_class": {c: self._cd(v) for c, v in self.imbalance_regime_by_class.items()},
            "burstiness_regime_by_class": {c: self._cd(v) for c, v in self.burstiness_regime_by_class.items()},
            "buy_vs_sell_iceberg": {"buy": self.buy_ice, "sell": self.sell_ice},
            "fake_iceberg_by_class": {c: {"mean": round(_safe_mean(v), 2),
                                           "with_any": sum(1 for x in v if x > 0)} for c, v in self.fake_by_class.items()},
            "mm_by_class": {c: {"mean": round(_safe_mean(v), 2)} for c, v in self.mm_by_class.items()},
            "regime_shift_by_class": {c: {"mean": round(_safe_mean(v), 2),
                                           "with_any": sum(1 for x in v if x > 0)} for c, v in self.shift_by_class.items()},
            "timing_style_counts": self._cd(self.timing_counts),
            "placement_style_counts": self._cd(self.placement_counts),
            "liq_profile_counts": self._cd(self.liq_counts),
            "spread_stats": _ds(self.spreads),
            "depth_stats": _ds(self.depths),
            "rows_stats": _ds(self.rows, int),
            "volume_stats": _ds(self.volumes, int),
            "support_counts": self._cd(self.support_counts),
            "meaningfulness": {"injected": inj, "meaningful": moi, "rate": round(mr, 4)},
            "warnings": w,
        }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path, use_splits: bool) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    if use_splits:
        for sp in ("train", "val", "test"):
            for lb in ("iceberg", "noniceberg"):
                (root / sp / lb).mkdir(parents=True, exist_ok=True)
    else:
        for lb in ("iceberg", "noniceberg"):
            (root / lb).mkdir(parents=True, exist_ok=True)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fns = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and v != v else v) for k, v in r.items()})


def _is_held_out(ep: EpisodeParams, hv: Set[str], hi: Set[str], hb: Set[str]) -> bool:
    return ep.volatility in hv or ep.imbalance_regime in hi or ep.burstiness_regime in hb


# ---------------------------------------------------------------------------
# Two-phase generation engine (v5)
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    target: int = 0
    saved: int = 0
    attempts: int = 0

    @property
    def save_rate(self) -> float:
        return self.saved / max(1, self.attempts)


def _run_phase(
    phase_name: str,         # "iceberg_phase" | "noniceberg_phase"
    accept_label: str,       # "iceberg" | "noniceberg"
    target_count: int,
    rng: random.Random,
    base_seed: int,
    ep_counter: List[int],   # mutable [next_ep_index]
    saved_counter: List[int], # mutable [next_saved_index]
    controls: GenerationControls,
    snap_bounds: Tuple[float, float],
    fixed_snap: Optional[float],
    max_depth: int,
    min_rows: int,
    max_rows: int,
) -> Tuple[List[Tuple[EpisodeParams, List[Dict[str, Any]], EpisodeTruth, Dict[str, Any]]], PhaseStats]:
    """Generate episodes until `target_count` with final label == accept_label are collected."""
    results: List[Tuple[EpisodeParams, List[Dict[str, Any]], EpisodeTruth, Dict[str, Any]]] = []
    stats = PhaseStats(target=target_count)
    max_attempts = target_count * _MAX_PHASE_ATTEMPTS_FACTOR

    while stats.saved < target_count and stats.attempts < max_attempts:
        stats.attempts += 1
        eidx = ep_counter[0]
        ep_counter[0] += 1
        sidx = saved_counter[0]  # tentative

        ep = _sample_episode_config(
            eidx, base_seed, rng,
            max_depth_levels=max_depth,
            snap_bounds=snap_bounds,
            min_rows=min_rows, max_rows=max_rows,
            controls=controls,
            phase=phase_name,
            candidate_attempt=stats.attempts,
            saved_index=sidx,
        )
        if fixed_snap is not None:
            ep = replace(ep, snapshot_interval=fixed_snap)

        rows, truth, mx = run_episode(ep)
        final_label = "iceberg" if truth.episode_label_iceberg else "noniceberg"

        if final_label != accept_label:
            # discard — label doesn't match this phase's target
            continue

        # accepted
        # fix episode_id to reflect actual saved index
        ep = replace(ep, episode_id=f"ep_{sidx:06d}", saved_episode_index=sidx)
        for r in rows:
            r["episode_id"] = ep.episode_id
        mx["saved_episode_index"] = sidx
        mx["candidate_attempt_index"] = stats.attempts

        results.append((ep, rows, truth, mx))
        stats.saved += 1
        saved_counter[0] += 1

        if stats.saved % 200 == 0 or stats.saved == 1:
            print(f"    [{phase_name}] saved {stats.saved}/{target_count}  "
                  f"attempts={stats.attempts}  rate={stats.save_rate:.1%}")

    return results, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic LOB dataset generator v5")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--max-depth-levels", type=int, default=10)
    p.add_argument("--snapshot-interval", type=float, default=None)
    p.add_argument("--min-rows-per-episode", type=int, default=40)
    p.add_argument("--max-rows-per-episode", type=int, default=800)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")

    g = p.add_argument_group("generation controls")
    g.add_argument("--target-hard-negative-freq", type=float, default=None)
    g.add_argument("--target-fake-iceberg-freq", type=float, default=None)
    g.add_argument("--target-multi-iceberg-freq", type=float, default=None)
    g.add_argument("--target-regime-shift-freq", type=float, default=None)

    v4 = p.add_argument_group("positive support")
    v4.add_argument("--positive-support-prob", type=float, default=0.80)
    v4.add_argument("--positive-support-strength", type=float, default=0.5)

    v5 = p.add_argument_group("balanced final labels (v5)")
    v5.add_argument("--balanced-final-labels", action="store_true",
                    help="Two-phase generation: accept/reject until exact label counts are met")
    v5.add_argument("--target-iceberg-count", type=int, default=None,
                    help="Exact number of iceberg-labeled episodes (default: num_episodes // 2)")
    v5.add_argument("--target-noniceberg-count", type=int, default=None,
                    help="Exact number of noniceberg-labeled episodes (default: num_episodes - target_iceberg)")

    s = p.add_argument_group("splits")
    s.add_argument("--splits", action="store_true")
    s.add_argument("--train-frac", type=float, default=.70)
    s.add_argument("--val-frac", type=float, default=.15)
    s.add_argument("--test-frac", type=float, default=.15)

    h = p.add_argument_group("held-out regimes")
    h.add_argument("--holdout-volatility", nargs="*", default=None, choices=["calm", "normal", "noisy"])
    h.add_argument("--holdout-imbalance", nargs="*", default=None,
                   choices=["strong_sell", "mild_sell", "neutral", "mild_buy", "strong_buy"])
    h.add_argument("--holdout-burstiness", nargs="*", default=None, choices=["calm", "normal", "bursty"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    elif root.exists() and not args.overwrite:
        raise SystemExit(f"Output dir {root} exists. Pass --overwrite.")
    root.mkdir(parents=True)

    hv = set(args.holdout_volatility or [])
    hi = set(args.holdout_imbalance or [])
    hb = set(args.holdout_burstiness or [])
    use_splits = args.splits or bool(hv or hi or hb)
    _ensure_dirs(root, use_splits)
    debug_dir = root / "debug_optional"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    snap_bounds = (0.2, 2.5)
    if args.snapshot_interval is not None:
        snap_bounds = (args.snapshot_interval, args.snapshot_interval)

    controls = GenerationControls(
        target_hard_negative_freq=args.target_hard_negative_freq,
        target_fake_iceberg_freq=args.target_fake_iceberg_freq,
        target_multi_iceberg_freq=args.target_multi_iceberg_freq,
        target_regime_shift_freq=args.target_regime_shift_freq,
        positive_support_prob=args.positive_support_prob,
        positive_support_strength=args.positive_support_strength,
        balanced_final_labels=args.balanced_final_labels,
    )

    N = args.num_episodes
    all_results: List[Tuple[EpisodeParams, List[Dict[str, Any]], EpisodeTruth, Dict[str, Any]]] = []

    if args.balanced_final_labels:
        # --- v5 two-phase mode ---
        t_ice = args.target_iceberg_count if args.target_iceberg_count is not None else N // 2
        t_non = args.target_noniceberg_count if args.target_noniceberg_count is not None else N - t_ice

        print(f"Balanced-final-labels mode: target_iceberg={t_ice}  target_noniceberg={t_non}")

        ep_counter = [0]
        saved_counter = [0]
        rng1 = random.Random(args.seed)
        rng2 = random.Random(args.seed + 99991)

        # Phase 1: iceberg
        print(f"\n  Phase 1: generating {t_ice} iceberg-labeled episodes...")
        ice_results, ice_stats = _run_phase(
            "iceberg_phase", "iceberg", t_ice, rng1, args.seed,
            ep_counter, saved_counter, controls, snap_bounds,
            args.snapshot_interval, args.max_depth_levels,
            args.min_rows_per_episode, args.max_rows_per_episode,
        )
        all_results.extend(ice_results)
        print(f"    Phase 1 done: saved={ice_stats.saved}/{ice_stats.target}  "
              f"attempts={ice_stats.attempts}  rate={ice_stats.save_rate:.1%}")

        # Phase 2: noniceberg
        print(f"\n  Phase 2: generating {t_non} noniceberg-labeled episodes...")
        non_results, non_stats = _run_phase(
            "noniceberg_phase", "noniceberg", t_non, rng2, args.seed + 50000,
            ep_counter, saved_counter, controls, snap_bounds,
            args.snapshot_interval, args.max_depth_levels,
            args.min_rows_per_episode, args.max_rows_per_episode,
        )
        all_results.extend(non_results)
        print(f"    Phase 2 done: saved={non_stats.saved}/{non_stats.target}  "
              f"attempts={non_stats.attempts}  rate={non_stats.save_rate:.1%}")

        phase_report = {
            "iceberg_phase": {"target": ice_stats.target, "saved": ice_stats.saved,
                              "attempts": ice_stats.attempts, "save_rate": round(ice_stats.save_rate, 4)},
            "noniceberg_phase": {"target": non_stats.target, "saved": non_stats.saved,
                                 "attempts": non_stats.attempts, "save_rate": round(non_stats.save_rate, 4)},
        }
    else:
        # --- legacy single-pass mode ---
        rng_m = random.Random(args.seed)
        ep_counter = [0]
        saved_counter = [0]
        max_att = N * (_MAX_PHASE_ATTEMPTS_FACTOR + 1)
        att = 0
        while len(all_results) < N and att < max_att:
            att += 1
            eidx = ep_counter[0]; ep_counter[0] += 1
            sidx = saved_counter[0]
            ep = _sample_episode_config(
                eidx, args.seed, rng_m,
                max_depth_levels=args.max_depth_levels,
                snap_bounds=snap_bounds, min_rows=args.min_rows_per_episode,
                max_rows=args.max_rows_per_episode, controls=controls,
                phase="legacy", candidate_attempt=att, saved_index=sidx,
            )
            if args.snapshot_interval is not None:
                ep = replace(ep, snapshot_interval=args.snapshot_interval)
            rows, truth, mx = run_episode(ep)
            ep = replace(ep, episode_id=f"ep_{sidx:06d}", saved_episode_index=sidx)
            for r in rows:
                r["episode_id"] = ep.episode_id
            mx["saved_episode_index"] = sidx
            mx["candidate_attempt_index"] = att
            all_results.append((ep, rows, truth, mx))
            saved_counter[0] += 1
            if len(all_results) % 500 == 0 or len(all_results) == 1:
                print(f"  ... {len(all_results)}/{N}")
        phase_report = {"legacy": {"total": len(all_results), "attempts": att}}

    # --- count & assign splits ---
    counts: Dict[str, int] = {"iceberg": 0, "noniceberg": 0}
    split_rng = random.Random(args.seed + 31337)
    auditors: Dict[str, AuditAccumulator] = defaultdict(AuditAccumulator)
    meta_rows: List[Dict[str, Any]] = []

    for ep, rows, truth, mx in all_results:
        label = "iceberg" if truth.episode_label_iceberg else "noniceberg"
        counts[label] += 1
        fname = f"{ep.episode_id}.csv"

        if use_splits:
            if _is_held_out(ep, hv, hi, hb):
                split = "test"
            else:
                r = split_rng.random()
                if r < args.train_frac:
                    split = "train"
                elif r < args.train_frac + args.val_frac:
                    split = "val"
                else:
                    split = "test"
            out = root / split / label / fname
        else:
            split = "all"
            out = root / label / fname

        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out)
        auditors[split].ingest(label, ep, truth, mx)

        if args.debug:
            dbg: Dict[str, Any] = {
                "episode_label_iceberg": truth.episode_label_iceberg,
                "buy_iceberg_count": truth.buy_iceberg_count,
                "sell_iceberg_count": truth.sell_iceberg_count,
                "meaningful_iceberg_count": truth.meaningful_iceberg_count,
                "icebergs": {iid: {"side": st.side.value, "price": st.price,
                                    "peak_qty": st.peak_qty, "executed_qty": st.executed_qty,
                                    "replenishment_count": st.replenishment_count,
                                    "volume_absorbed": st.volume_absorbed,
                                    "meaningful": st.meaningful(), "slices_posted": st.slices_posted}
                             for iid, st in truth.icebergs.items()},
                "episode_id": ep.episode_id, "split": split,
                "difficulty_bucket": mx["difficulty_bucket"],
                "generation_phase": mx["generation_phase"],
                "candidate_attempt_index": mx["candidate_attempt_index"],
                "saved_episode_index": mx["saved_episode_index"],
                "had_positive_support": ep.had_positive_support,
                "positive_support_strength": round(ep.positive_support_strength, 4),
                "iceberg_became_meaningful": mx["iceberg_became_meaningful"],
                "num_market_makers": ep.num_market_makers,
                "num_fake_icebergs": ep.num_fake_icebergs,
                "num_regime_shifts": len(ep.regime_shifts),
                "iceberg_timing_style": ep.iceberg_timing_style,
                "iceberg_placement_style": ep.iceberg_placement_style,
                "liquidity_profile_style": ep.liquidity_profile_style,
                "imbalance_regime": ep.imbalance_regime,
                "burstiness_regime": ep.burstiness_regime,
            }
            with open(debug_dir / f"{ep.episode_id}.json", "w", encoding="utf-8") as f:
                json.dump(dbg, f, indent=2)

        meta_rows.append({
            "episode_id": ep.episode_id, "filename": fname, "class_label": label,
            "split": split, "generation_phase": mx["generation_phase"],
            "candidate_attempt_index": mx["candidate_attempt_index"],
            "saved_episode_index": mx["saved_episode_index"],
            "seed": ep.seed, "duration": ep.duration,
            "initial_mid_price": ep.mid0, "tick_size": ep.tick_size,
            "min_spread": ep.min_spread, "depth_levels": ep.depth_book,
            "base_order_rate": ep.base_order_rate,
            "volatility_regime": ep.volatility,
            "imbalance_regime": ep.imbalance_regime,
            "burstiness_regime": ep.burstiness_regime,
            "cancellation_regime": round(ep.cancel_intensity, 6),
            "imbalance": ep.imbalance, "burstiness": round(ep.burstiness, 6),
            "trend_regime": round(ep.trend_strength, 6),
            "cluster_regime": round(ep.cluster_prob, 6),
            "iceberg_count": len(truth.icebergs),
            "buy_iceberg_count": truth.buy_iceberg_count,
            "sell_iceberg_count": truth.sell_iceberg_count,
            "num_market_makers": mx["num_market_makers"],
            "num_fake_icebergs": mx["num_fake_icebergs"],
            "num_regime_shifts": mx["num_regime_shifts"],
            "iceberg_timing_style": mx["iceberg_timing_style"],
            "iceberg_placement_style": mx["iceberg_placement_style"],
            "liquidity_profile_style": mx["liquidity_profile_style"],
            "difficulty_bucket": mx["difficulty_bucket"],
            "had_positive_support": mx["had_positive_support"],
            "positive_support_strength": mx["positive_support_strength"],
            "iceberg_became_meaningful": mx["iceberg_became_meaningful"],
            "hard_negative_types": json.dumps(ep.hard_negative_mix),
            "num_rows": mx["num_rows"],
            "num_events_approx": mx["num_orders"] + mx["num_trades"] + mx["num_cancels"],
            "num_trades": mx["num_trades"],
            "total_volume": mx["total_volume"],
            "snapshot_interval": ep.snapshot_interval,
            "mean_spread": round(mx["mean_spread"], 6),
            "mean_depth": round(mx["mean_depth"], 2),
        })

    # metadata CSV
    meta_path = root / "metadata" / "episodes_metadata.csv"
    if meta_rows:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            for r in meta_rows:
                w.writerow(r)

    # audit
    audit: Dict[str, Any] = {}
    for sn, aud in sorted(auditors.items()):
        audit[sn] = aud.build_report()
    ga = AuditAccumulator()
    for ep, rows, truth, mx in all_results:
        ga.ingest("iceberg" if truth.episode_label_iceberg else "noniceberg", ep, truth, mx)
    audit["global"] = ga.build_report()
    audit["phase_stats"] = phase_report
    with open(root / "metadata" / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # summary
    n_final = len(all_results)
    summary: Dict[str, Any] = {
        "num_episodes": n_final, "seed": args.seed, "output_dir": str(root),
        "counts_by_label": counts,
        "iceberg_fraction": round(counts["iceberg"] / max(1, n_final), 4),
        "balanced_final_labels": args.balanced_final_labels,
        "phase_stats": phase_report,
        "use_splits": use_splits,
        "generation_controls": {
            "positive_support_prob": controls.positive_support_prob,
            "positive_support_strength": controls.positive_support_strength,
            "target_hard_negative_freq": controls.target_hard_negative_freq,
            "target_fake_iceberg_freq": controls.target_fake_iceberg_freq,
            "target_multi_iceberg_freq": controls.target_multi_iceberg_freq,
            "target_regime_shift_freq": controls.target_regime_shift_freq,
        },
        "held_out_regimes": {
            "volatility": sorted(hv) if hv else None,
            "imbalance": sorted(hi) if hi else None,
            "burstiness": sorted(hb) if hb else None,
        },
        "split_counts": {sp: dict(a.class_counts) for sp, a in auditors.items()},
    }
    with open(root / "metadata" / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # console
    frac = counts["iceberg"] / max(1, n_final)
    print(f"\nDone. {n_final} episodes under {root}")
    print(f"  iceberg: {counts['iceberg']}  noniceberg: {counts['noniceberg']}  fraction: {frac:.4f}")
    if args.balanced_final_labels:
        for pn, ps in phase_report.items():
            print(f"  [{pn}] saved={ps['saved']}/{ps['target']}  "
                  f"attempts={ps['attempts']}  rate={ps['save_rate']:.1%}")
    if use_splits:
        for sp, aud in sorted(auditors.items()):
            print(f"  [{sp}] {dict(aud.class_counts)}")
    mr = audit["global"].get("meaningfulness", {})
    print(f"  meaningfulness: {mr.get('meaningful',0)}/{mr.get('injected',0)} = {mr.get('rate',0):.1%}")
    gw = audit["global"].get("warnings", [])
    if gw:
        print(f"\n  AUDIT WARNINGS ({len(gw)}):")
        for w in gw:
            print(f"    ⚠ {w}")
    else:
        print("  No audit warnings.")

    if args.zip_output:
        zp = root.parent / f"{root.name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in root.rglob("*"):
                if fp.is_file():
                    zf.write(fp, arcname=str(fp.relative_to(root.parent)))
        print(f"  zip: {zp}")


if __name__ == "__main__":
    main()