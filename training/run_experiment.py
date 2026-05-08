"""
End-to-end iceberg detection experiment runner (CNN version).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Iceberg detection L3 vs L2 experiment (CNN)")
    p.add_argument("--stage", choices=["all", "generate", "features", "train"],
                   default="all")
    p.add_argument("--runs-per-regime", type=int, default=None)
    p.add_argument("--sim-duration", type=float, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--window-s", type=float, default=0.3)
    p.add_argument("--step-s", type=float, default=0.05)
    return p.parse_args()


def stage_generate(cfg) -> None:
    from training.data_generator import DatasetBuilder
    t0 = time.perf_counter()
    builder = DatasetBuilder(cfg)
    stats = builder.build()
    elapsed = time.perf_counter() - t0
    logger.info(f"Data generation complete in {elapsed/60:.1f} min")


def stage_features(cfg) -> None:
    from training.feature_extractor import build_feature_matrices
    t0 = time.perf_counter()
    build_feature_matrices(
        data_dir=cfg.output_dir,
        window_s=getattr(cfg, "_window_s", 0.3),
        step_s=getattr(cfg, "_step_s", 0.05),
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"Feature indexing complete in {elapsed/60:.1f} min")


def stage_train(cfg_model, data_dir: str) -> None:
    from training.model_trainer import run_full_experiment
    t0 = time.perf_counter()
    run_full_experiment(data_dir=data_dir, cfg=cfg_model)
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

    gen_cfg._window_s = args.window_s
    gen_cfg._step_s = args.step_s

    model_cfg = ModelConfig(output_dir=str(Path(gen_cfg.output_dir).parent / "models"))

    logger.info("=" * 60)
    logger.info("ICEBERG DETECTION EXPERIMENT (CNN)")
    logger.info("=" * 60)
    logger.info(f"Stage:            {args.stage}")
    logger.info(f"Data dir:         {gen_cfg.output_dir}")
    logger.info("=" * 60)

    if args.stage in ("all", "generate"):
        logger.info("\n[Stage 1/3] Generating simulation data …")
        stage_generate(gen_cfg)

    if args.stage in ("all", "features"):
        logger.info("\n[Stage 2/3] Extracting features …")
        stage_features(gen_cfg)

    if args.stage in ("all", "train"):
        logger.info("\n[Stage 3/3] Training CNN models …")
        stage_train(model_cfg, gen_cfg.output_dir)

    logger.info("\nExperiment complete.")


if __name__ == "__main__":
    main()
