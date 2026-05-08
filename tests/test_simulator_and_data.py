import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.market_simulator import MarketSimulator, SimulationConfig
from core.matching_engine import MatchingEngine
from core.order import LimitOrder, NaiveIcebergOrder, OrderSide
from core.order_book import OrderBook
from training.config import IcebergConfig
from training.data_generator import IcebergOrchestrator


class OrderBookDepthTests(unittest.TestCase):
    def test_bid_depth_returns_best_prices_first(self):
        book = OrderBook()
        for order_id, price, quantity in [
            ("b1", 99.00, 10),
            ("b2", 100.00, 20),
            ("b3", 101.00, 30),
        ]:
            book.add_order(
                LimitOrder(
                    order_id=order_id,
                    timestamp=0.0,
                    trader_id="buyer",
                    side=OrderSide.BUY,
                    price=price,
                    quantity=quantity,
                )
            )

        self.assertEqual(
            book.get_depth(OrderSide.BUY, levels=2),
            [(101.0, 30, 1), (100.0, 20, 1)],
        )
        self.assertEqual(book.get_total_depth(OrderSide.BUY, num_levels=2), 50)


class MatchingEngineTests(unittest.TestCase):
    def test_limit_order_matches_remaining_quantity_only(self):
        book = OrderBook()
        engine = MatchingEngine(book)

        engine.process_order(
            LimitOrder(
                order_id="ask",
                timestamp=0.0,
                trader_id="seller",
                side=OrderSide.SELL,
                price=100.0,
                quantity=100,
            ),
            timestamp=0.0,
        )

        first_buy = engine.process_order(
            LimitOrder(
                order_id="buy1",
                timestamp=1.0,
                trader_id="buyer",
                side=OrderSide.BUY,
                price=100.0,
                quantity=60,
            ),
            timestamp=1.0,
        )
        self.assertEqual([trade.quantity for trade in first_buy.trades], [60])

        second_buy = engine.process_order(
            LimitOrder(
                order_id="buy2",
                timestamp=2.0,
                trader_id="buyer",
                side=OrderSide.BUY,
                price=100.0,
                quantity=50,
            ),
            timestamp=2.0,
        )

        self.assertEqual([trade.quantity for trade in second_buy.trades], [40])
        self.assertEqual(second_buy.filled_quantity, 40)
        self.assertEqual(second_buy.remaining_quantity, 10)
        self.assertEqual(book.get_liquidity_at_price(100.0, OrderSide.SELL), 0)
        self.assertEqual(book.get_liquidity_at_price(100.0, OrderSide.BUY), 10)

    def test_passive_iceberg_refill_restores_only_visible_tip(self):
        book = OrderBook()
        engine = MatchingEngine(book)

        iceberg = NaiveIcebergOrder(
            order_id="ice",
            timestamp=0.0,
            trader_id="iceberg",
            side=OrderSide.SELL,
            price=100.0,
            quantity=100,
            peak_quantity=200,
            visible_quantity=100,
        )
        engine.process_order(iceberg, timestamp=0.0)

        aggressive_buy = engine.process_order(
            LimitOrder(
                order_id="buy",
                timestamp=1.0,
                trader_id="buyer",
                side=OrderSide.BUY,
                price=100.0,
                quantity=100,
            ),
            timestamp=1.0,
        )

        self.assertEqual([trade.quantity for trade in aggressive_buy.trades], [100])
        self.assertEqual(iceberg.refill_count, 1)
        self.assertEqual(book.get_liquidity_at_price(100.0, OrderSide.SELL), 100)


class IcebergOrchestratorTests(unittest.TestCase):
    def test_ground_truth_includes_completed_chain_with_correct_refill_count(self):
        sim = MarketSimulator(
            SimulationConfig(end_time=1.0, snapshot_interval=0.0, random_seed=1)
        )
        orchestrator = IcebergOrchestrator(
            sim=sim,
            rng=np.random.RandomState(1),
            cfg=IcebergConfig(),
        )

        orchestrator.inject(
            side="SELL",
            price=100.0,
            total_hidden=250,
            visible_qty=100,
            refresh_delay_s=0.0,
        )

        for idx, delay in enumerate([0.0, 0.001, 0.002], start=1):
            sim.submit_order(
                LimitOrder(
                    order_id=f"buy{idx}",
                    timestamp=delay,
                    trader_id="buyer",
                    side=OrderSide.BUY,
                    price=101.0,
                    quantity=100,
                ),
                delay=delay,
            )

        sim.run()

        ground_truth = orchestrator.ground_truth_df("run_1", "volatile")
        self.assertEqual(len(ground_truth), 1)

        row = ground_truth.iloc[0]
        self.assertEqual(row["total_hidden"], 250)
        self.assertEqual(row["visible_qty"], 100)
        self.assertEqual(row["n_refills"], 2)
        self.assertIsNotNone(row["end_time"])


if __name__ == "__main__":
    unittest.main()
