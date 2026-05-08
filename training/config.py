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
    refresh_delay_max_ms: float = 200.0
    # Fraction of "large order" slots that are icebergs
    prevalence_min: float = 0.02
    prevalence_max: float = 0.10
    # Visible tip size range (shares)
    visible_qty_min: int = 10
    visible_qty_max: int = 200
    # Fraction of icebergs that use immediate refresh (vs stochastic delay)
    immediate_refresh_fraction: float = 0.40


@dataclass
class RegimeConfig:
    name: str
    volatility: float
    drift: float
    mean_reversion: float
    order_rate_per_sec: float
    spread_ticks: int = 3


REGIMES: List[RegimeConfig] = [
    RegimeConfig("trending_up",    volatility=0.0004, drift=+0.0003, mean_reversion=0.0, order_rate_per_sec=200,  spread_ticks=3),
    RegimeConfig("trending_down",  volatility=0.0004, drift=-0.0003, mean_reversion=0.0, order_rate_per_sec=200,  spread_ticks=3),
    RegimeConfig("mean_reverting", volatility=0.0002, drift=0.0,     mean_reversion=0.6, order_rate_per_sec=150,  spread_ticks=2),
    RegimeConfig("volatile",       volatility=0.0012, drift=0.0,     mean_reversion=0.1, order_rate_per_sec=300, spread_ticks=5),
    RegimeConfig("low_volatility", volatility=0.0001, drift=0.0,     mean_reversion=0.4, order_rate_per_sec=100,  spread_ticks=2),
]


@dataclass
class GenerationConfig:
    runs_per_regime: int = 20
    sim_duration: float = 300.0
    l2_snapshot_interval_s: float = 0.05
    feature_window_s: float = 0.3
    feature_step_s: float = 0.05
    output_dir: str = "training/data"
    seed_base: int = 42
    iceberg: IcebergConfig = field(default_factory=IcebergConfig)
    regimes: List[RegimeConfig] = field(default_factory=lambda: REGIMES)


@dataclass
class ModelConfig:
    # CNN Hyperparameters
    epochs: int = 5
    batch_size: int = 512
    learning_rate: float = 0.001
    seq_len: int = 200
    
    # Train / val / test fractions (by run_id to avoid leakage)
    train_frac: float = 0.70
    val_frac: float = 0.15
    output_dir: str = "training/models"
    n_jobs: int = -1
