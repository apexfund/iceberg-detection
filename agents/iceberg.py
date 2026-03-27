"""
Iceberg order injection for the simulation.

Injects one buy iceberg over the run (e.g. 60s).
Sell iceberg is commented out so it does not cancel out with the buy at the same price.
Uses NaiveIcebergOrder: large peak size with a small visible tip; refill logic is in core.
"""

from typing import Any

from core.order import NaiveIcebergOrder, OrderSide


# Order IDs used in logs to identify iceberg orders
ICEBERG_BUY_ORDER_ID = "ICEBERG_BUY"
ICEBERG_SELL_ORDER_ID = "ICEBERG_SELL"

# Default iceberg sizing: large hidden size, small visible tip
DEFAULT_PEAK_QUANTITY = 15_000
DEFAULT_VISIBLE_QUANTITY = 25


def inject_iceberg_buy(
    simulator: Any,
    order_id: str,
    current_time: float = 0.0,
    *,
    target_price: float = 100.0,
    peak_quantity: int = DEFAULT_PEAK_QUANTITY,
    visible_quantity: int = DEFAULT_VISIBLE_QUANTITY,
) -> None:
    """Submit one buy NaiveIcebergOrder with a custom order_id."""
    price = simulator.order_book.round_price(target_price)
    iceberg = NaiveIcebergOrder(
        order_id=order_id,
        timestamp=current_time,
        trader_id="iceberg_agent",
        side=OrderSide.BUY,
        price=price,
        peak_quantity=peak_quantity,
        visible_quantity=visible_quantity,
        quantity=visible_quantity,
    )
    simulator.submit_order(iceberg, delay=0.0)


def inject_iceberg_orders(
    simulator: Any,
    current_time: float = 0.0,
    *,
    target_price: float = 100.0,
    peak_quantity: int = DEFAULT_PEAK_QUANTITY,
    visible_quantity: int = DEFAULT_VISIBLE_QUANTITY,
) -> None:
    """
    Submit one buy iceberg to the simulator.
    (Sell iceberg commented out so buy/sell at same price do not cancel out.)

    Order has order_id ICEBERG_BUY so logs can confirm presence.
    Refilling of the visible tip is handled by the core matching/refill logic.

    Args:
        simulator: MarketSimulator instance.
        current_time: Simulation time at submission.
        target_price: Limit price for the iceberg.
        peak_quantity: Total (hidden) size.
        visible_quantity: Visible tip size in the book.
    """
    inject_iceberg_buy(
        simulator,
        ICEBERG_BUY_ORDER_ID,
        current_time,
        target_price=target_price,
        peak_quantity=peak_quantity,
        visible_quantity=visible_quantity,
    )

    # Sell iceberg at same price would cancel out with buy; disabled for now.
    # sell_iceberg = NaiveIcebergOrder(
    #     order_id=ICEBERG_SELL_ORDER_ID,
    #     timestamp=current_time,
    #     trader_id="iceberg_agent",
    #     side=OrderSide.SELL,
    #     price=price,
    #     peak_quantity=peak_quantity,
    #     visible_quantity=visible_quantity,
    #     quantity=visible_quantity,
    # )
    # simulator.submit_order(sell_iceberg, delay=0.0)
