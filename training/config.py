"""
Configuration dataclasses for data generation, feature extraction, and model training.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IcebergConfig:
    # Visible tip as fraction of total hidden size
    display_pct_min: float = 0.01    # 1% visible
    display_pct_max: float = 0.05    # 5% visible
    # True hidden size as multiple of display
    hidden_mult_min: float = 5.0     # 5x display
    hidden_mult_max: float = 40.0    # 40x display
    # Stochastic refresh delay range
    refresh_delay_min_ms: float = 5.0
    refresh_delay_max_ms: float = 500.0
    # Fraction of "large order" slots that are icebergs
    prevalence_min: float = 0.02
    prevalence_max: float = 0.08
    # Visible tip size range (shares)
    visible_qty_min: int = 10
    visible_qty_max: int = 150
    # Fraction of icebergs that use immediate refresh (vs stochastic delay)
    immediate_refresh_fraction: float = 0.35


@dataclass
class RegimeConfig:
    name: str
    # GBM / OU volatility per unit time (σ per second)
    volatility: float
    # Price drift per second (log-scale)
    drift: float
    # Mean-reversion strength (0 = none, 1 = strong)
    mean_reversion: float
    # Target order submissions per second across all agents
    order_rate_per_sec: float
    # Initial spread in ticks
    spread_ticks: int = 3


REGIMES: List[RegimeConfig] = [
    RegimeConfig("trending_up",    volatility=0.0004, drift=+0.0003, mean_reversion=0.0, order_rate_per_sec=80,  spread_ticks=3),
    RegimeConfig("trending_down",  volatility=0.0004, drift=-0.0003, mean_reversion=0.0, order_rate_per_sec=80,  spread_ticks=3),
    RegimeConfig("mean_reverting", volatility=0.0002, drift=0.0,     mean_reversion=0.6, order_rate_per_sec=50,  spread_ticks=2),
    RegimeConfig("volatile",       volatility=0.0012, drift=0.0,     mean_reversion=0.1, order_rate_per_sec=100, spread_ticks=5),
    RegimeConfig("low_volatility", volatility=0.0001, drift=0.0,     mean_reversion=0.4, order_rate_per_sec=30,  spread_ticks=2),
]


@dataclass
class GenerationConfig:
    # Number of simulation runs per regime
    runs_per_regime: int = 30
    # Simulation wall-clock duration per run (seconds of sim time)
    sim_duration: float = 500.0
    # L2 snapshot interval (seconds)
    l2_snapshot_interval_s: float = 0.05
    # Feature window width and stride (seconds)
    feature_window_s: float = 0.3
    feature_step_s: float = 0.05
    # Output directory (relative to project root)
    output_dir: str = "training/data"
    # RNG seed base; each run uses seed_base + run_idx * 1000
    seed_base: int = 42
    iceberg: IcebergConfig = field(default_factory=IcebergConfig)
    regimes: List[RegimeConfig] = field(default_factory=lambda: REGIMES)


@dataclass
class ModelConfig:
    # LightGBM hyperparameters
    n_estimators: int = 2000
    max_depth: int = 6
    num_leaves: int = 31
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    n_jobs: int = -1
    # Train / val / test fractions (by run_id to avoid leakage)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train - val
    output_dir: str = "training/models"
    # Class imbalance weight for positive class (target 1:5 ratio)
    scale_pos_weight: float = 1.0

