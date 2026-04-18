"""
Buy and sell agents that submit limit orders at a target price with Gaussian randomness.

Used to simulate a real market with multiple orders per second from both sides.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from core.order import LimitOrder, OrderSide


@dataclass
class BuyAgent:
    """
    Agent that submits buy limit orders with price = target_price + N(0, price_std).
    """

    agent_id: str = "buy_agent"
    target_price: float = 100.0
    price_std: float = 0.05
    quantity_min: int = 10
    quantity_max: int = 100
    order_id_prefix: str = "B"
    _order_counter: int = field(default=0, repr=False)

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"{self.order_id_prefix}{self._order_counter:06d}"

    def _sample_price(self) -> float:
        """Price with Gaussian noise around target; keep positive."""
        p = self.target_price + random.gauss(0.0, self.price_std)
        return max(0.01, p)

    def _sample_quantity(self) -> int:
        """Uniform quantity in [quantity_min, quantity_max]."""
        return random.randint(self.quantity_min, self.quantity_max)

    def submit_orders(
        self,
        simulator: Any,
        orders_per_tick: Optional[int] = None,
        current_time: float = 0.0,
    ) -> int:
        """
        Submit one or more buy limit orders to the simulator.

        Price is rounded to the book tick. Orders are scheduled with delay=0
        so they are processed at the current simulation time when the callback runs.

        Args:
            simulator: MarketSimulator instance (has submit_order, order_book).
            orders_per_tick: Number of orders to submit this tick (default random 1–3).
            current_time: Current sim time (for order timestamp; usually from simulator.current_time).

        Returns:
            Number of orders submitted.
        """
        if orders_per_tick is None:
            orders_per_tick = random.randint(1, 3)
        count = 0
        for _ in range(orders_per_tick):
            price = self._sample_price()
            price = simulator.order_book.round_price(price)
            quantity = self._sample_quantity()
            order = LimitOrder(
                order_id=self._next_order_id(),
                timestamp=current_time,
                trader_id=self.agent_id,
                side=OrderSide.BUY,
                price=price,
                quantity=quantity,
            )
            simulator.submit_order(order, delay=0.0)
            count += 1
        return count


@dataclass
class SellAgent:
    """
    Agent that submits sell limit orders with price = target_price + N(0, price_std).
    """

    agent_id: str = "sell_agent"
    target_price: float = 100.0
    price_std: float = 0.05
    quantity_min: int = 10
    quantity_max: int = 100
    order_id_prefix: str = "S"
    _order_counter: int = field(default=0, repr=False)

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"{self.order_id_prefix}{self._order_counter:06d}"

    def _sample_price(self) -> float:
        """Price with Gaussian noise around target; keep positive."""
        p = self.target_price + random.gauss(0.0, self.price_std)
        return max(0.01, p)

    def _sample_quantity(self) -> int:
        return random.randint(self.quantity_min, self.quantity_max)

    def submit_orders(
        self,
        simulator: Any,
        orders_per_tick: Optional[int] = None,
        current_time: float = 0.0,
    ) -> int:
        """
        Submit one or more sell limit orders to the simulator.
        """
        if orders_per_tick is None:
            orders_per_tick = random.randint(1, 3)
        count = 0
        for _ in range(orders_per_tick):
            price = self._sample_price()
            price = simulator.order_book.round_price(price)
            quantity = self._sample_quantity()
            order = LimitOrder(
                order_id=self._next_order_id(),
                timestamp=current_time,
                trader_id=self.agent_id,
                side=OrderSide.SELL,
                price=price,
                quantity=quantity,
            )
            simulator.submit_order(order, delay=0.0)
            count += 1
        return count
