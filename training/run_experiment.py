"""
End-to-end iceberg detection experiment runner.

Usage:
    # From iceberg-detection/ root:
    python -m training.run_experiment [--stage all|generate|features|train]
                                      [--runs-per-regime N]
                                      [--sim-duration S]
                                      [--data-dir PATH]

Stages:
    generate  – run simulations, write L3/L2 parquet files
    features  – load parquet, extract feature matrices
    train     – train LightGBM models, print degradation report
    all       – run all three in order (default)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running as `python -m training.run_experiment` from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iceberg detection L3 vs L2 experiment")
    p.add_argument("--stage", choices=["all", "generate", "features", "train"],
                   default="all")
    p.add_argument("--runs-per-regime", type=int, default=None,
                   help="Override GenerationConfig.runs_per_regime")
    p.add_argument("--sim-duration", type=float, default=None,
                   help="Override GenerationConfig.sim_duration (seconds)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override data output directory")
    p.add_argument("--window-s", type=float, default=1.0,
                   help="Feature window width in seconds")
    p.add_argument("--step-s", type=float, default=0.1,
                   help="Feature window step in seconds")
    return p.parse_args()


def stage_generate(cfg) -> None:
    from training.data_generator import DatasetBuilder
    t0 = time.perf_counter()
    builder = DatasetBuilder(cfg)
    stats = builder.build()
    elapsed = time.perf_counter() - t0
    logger.info(f"Data generation complete in {elapsed/60:.1f} min")
    logger.info(f"  Total orders:    {stats['total_orders']:>10,}")
    logger.info(f"  Total trades:    {stats['total_trades']:>10,}")
    logger.info(f"  Total snapshots: {stats['total_snapshots']:>10,}")
    logger.info(f"  Iceberg chains:  {stats['total_iceberg_chains']:>10,}")
    if stats["total_orders"] < 5_000_000:
        logger.warning(
            f"Only {stats['total_orders']:,} orders generated. "
            "Increase runs_per_regime or sim_duration to reach 5M+ target."
        )


def stage_features(cfg) -> None:
    from training.feature_extractor import build_feature_matrices
    t0 = time.perf_counter()
    l3_feats, l2_feats = build_feature_matrices(
        data_dir=cfg.output_dir,
        window_s=getattr(cfg, "_window_s", 1.0),
        step_s=getattr(cfg, "_step_s", 0.1),
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"Feature extraction complete in {elapsed/60:.1f} min")
    logger.info(f"  L3 feature matrix: {l3_feats.shape}")
    logger.info(f"  L2 feature matrix: {l2_feats.shape}")
    logger.info(f"  L3 features ({len([c for c in l3_feats.columns if c not in {'run_id','regime','window_start','label'}])}):")
    feat_cols = [c for c in l3_feats.columns if c not in {"run_id","regime","window_start","label"}]
    logger.info(f"    {feat_cols}")


def stage_train(cfg_model, data_dir: str) -> None:
    from training.model_trainer import run_full_experiment
    t0 = time.perf_counter()
    report = run_full_experiment(data_dir=data_dir, cfg=cfg_model)
    elapsed = time.perf_counter() - t0
    logger.info(f"Training complete in {elapsed/60:.1f} min")


def main():
    args = parse_args()

    from training.config import GenerationConfig, ModelConfig

    gen_cfg = GenerationConfig()
    if args.runs_per_regime is not None:
        gen_cfg.runs_per_regime = args.runs_per_regime
    if args.sim_duration is not None:
        gen_cfg.sim_duration = args.sim_duration
    if args.data_dir is not None:
        gen_cfg.output_dir = args.data_dir

    # Stash window params on gen_cfg for convenience (not a real field)
    gen_cfg._window_s = args.window_s
    gen_cfg._step_s = args.step_s

    model_cfg = ModelConfig(output_dir=str(Path(gen_cfg.output_dir).parent / "models"))

    logger.info("=" * 60)
    logger.info("ICEBERG DETECTION EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Stage:            {args.stage}")
    logger.info(f"Runs per regime:  {gen_cfg.runs_per_regime}")
    logger.info(f"Regimes:          {[r.name for r in gen_cfg.regimes]}")
    logger.info(f"Sim duration:     {gen_cfg.sim_duration}s per run")
    logger.info(f"Feature window:   {args.window_s}s  step={args.step_s}s")
    logger.info(f"Data dir:         {gen_cfg.output_dir}")
    logger.info(f"Models dir:       {model_cfg.output_dir}")
    logger.info("=" * 60)

    if args.stage in ("all", "generate"):
        logger.info("\n[Stage 1/3] Generating simulation data …")
        stage_generate(gen_cfg)

    if args.stage in ("all", "features"):
        logger.info("\n[Stage 2/3] Extracting features …")
        stage_features(gen_cfg)

    if args.stage in ("all", "train"):
        logger.info("\n[Stage 3/3] Training models and computing degradation …")
        stage_train(model_cfg, gen_cfg.output_dir)

    logger.info("\nExperiment complete.")


if __name__ == "__main__":
    main()
