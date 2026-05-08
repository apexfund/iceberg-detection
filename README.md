# Iceberg Order Detection

Large institutional investors — hedge funds, banks, asset managers — often need to buy or sell enormous quantities of stock without tipping off the market. If they placed a single order for 500,000 shares, other traders would see it, move the price against them, and the execution cost would be enormous.

Their solution is the **iceberg order**: place only a small visible "tip" (say, 500 shares) in the order book. When that fills, automatically replenish it with another 500. Repeat until the full hidden quantity is exhausted. From the outside, it looks like a normal small order — but the price level never depletes. The iceberg keeps refilling.

This project builds a detector that reads only publicly observable order book data and outputs a probability that an iceberg is present at the current price level.

---

## Why It's Hard

You can't see the hidden quantity. You only see:
- The current best bid and ask prices
- The visible quantity sitting at each price level
- The stream of orders and trades as they happen

The iceberg's signature is subtle: a price level that should have been exhausted by now, but keeps refilling. This is easy to spot in hindsight but hard to detect in real time — especially when normal market noise makes levels fluctuate anyway.

---

## Approach

Since real exchange-level data is expensive and hard to label, we build a **synthetic market simulator** and inject iceberg orders ourselves. This gives us exact ground-truth labels for every moment in the simulation.

**Simulate → Inject → Label → Train → Evaluate**

1. Run a realistic order book with buy and sell agents across five market regimes (trending up, trending down, mean-reverting, volatile, low-volatility)
2. Randomly inject iceberg orders at the best bid/ask with controlled hidden sizes (5–40× the visible tip)
3. Label each 300ms window of market activity: was an iceberg active here?
4. Train a 1D-CNN to classify windows as iceberg / no iceberg
5. Compare a model that only sees price and quantity (L2) against one that also sees the raw order and trade stream (L3)

---

## The Model

Each model input is a **1-second history of the order book** — 20 timesteps at 50ms intervals, encoded as a multi-channel time series. The target label asks whether an iceberg is active in the corresponding 300ms window.

**L2 features** (4 channels): best bid, best ask (normalized to mid-price), bid quantity at L1, ask quantity at L1 (log-scaled).

**L3 features** (9 channels): everything in L2, plus per-timestep event count, trade count, total traded quantity, max single-trade quantity, and modal quantity frequency — all derived from the raw order and trade stream.

The classifier is a lightweight 1D-CNN:

```
Input (channels × 20 timesteps)
  → Conv1D(32) + BatchNorm + LeakyReLU
  → Conv1D(64) + BatchNorm + LeakyReLU
  → AdaptiveMaxPool → Dropout(0.5) → Linear → sigmoid
```

Trained on 219,434 balanced training windows (50/50 iceberg vs. no iceberg) with BCE loss and Adam. Best checkpoint selected by validation AUC on a run-level held-out validation split.

---

## Results

The metrics and plots committed under `training/models/` and `training/plots/` are example artifacts from a prior full run. Regenerate them after code or configuration changes.

| Model | AUC-ROC | Avg Precision | Test windows |
|-------|---------|---------------|--------------|
| L3 CNN (price + order flow) | **0.7429** | 0.5668 | 95,568 |
| L2 CNN (price + qty only) | **0.7200** | 0.5265 | 95,568 |

<img src="training/plots/01_model_performance.png" width="600"/>

The L3 model is still better after fixing the simulator, label export, and evaluation pipeline. The gap is smaller than the older stale artifacts suggested, but raw order flow still adds measurable signal beyond price and visible size alone.

The models are not perfect, and they shouldn't be. A 300ms window of noisy synthetic data is genuinely ambiguous much of the time. What matters is that the signal is real and learnable.

---

## Dataset

- **5 regimes** × 20 runs × 300 seconds = 100 simulated trading sessions
- **1,242 iceberg chains** injected across all sessions
- **599,300 labelled windows** (0.3s window, 0.05s step)
- Iceberg hidden size: 56–7,839 shares; visible tip: 10–199 shares; multiplier: 5–40×

<img src="training/plots/10_iceberg_signal_example.png" width="750"/>

*A real iceberg event from the volatile regime. The ask quantity (red) holds unnaturally steady while the bid pressure fluctuates — the pattern the model learns to recognize.*

---

## Project Structure

```
core/                    # Discrete-event order book simulator
  order.py               # LimitOrder, NaiveIcebergOrder (auto-refill logic)
  order_book.py          # Two-sided book with price-time priority
  matching_engine.py     # Trade generation
  event_queue.py         # Priority-queue event scheduler
  market_simulator.py    # Orchestrates simulation runs

training/
  config.py              # Regime and model hyperparameters
  data_generator.py      # Runs simulations, injects icebergs, writes parquet
  feature_extractor.py   # Sliding-window labelling and CNN sequence extraction
  model_cnn.py           # MicrostructureCNN definition and training loop
  model_trainer.py       # Full L2 + L3 experiment pipeline
  run_experiment.py      # CLI entrypoint

visualize.py             # Regenerate all plots
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline: simulate → label → train
python training/run_experiment.py

# Individual stages
python training/run_experiment.py --stage generate
python training/run_experiment.py --stage features
python training/run_experiment.py --stage train

# Regenerate plots
python visualize.py
```

Results are written to `training/models/degradation_report.json`. Model checkpoints are saved as `model_L2.pth` and `model_L3.pth`.
