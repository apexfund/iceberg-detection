"""
Matching engine for processing orders and executing trades.

The matching engine validates orders, matches them against the order book,
generates trade records, and handles special order types like icebergs.
"""

from dataclasses import dataclass
from typing import List, Optional

from core.order import (
    Order, LimitOrder, MarketOrder, CancelOrder, NaiveIcebergOrder,
    OrderSide, OrderStatus
)
from core.order_book import OrderBook


@dataclass
class Trade:
    trade_id: str
    timestamp: float
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    aggressor_side: OrderSide


@dataclass
class OrderResult:
    order: Order
    trades: List[Trade]
    accepted: bool
    rejection_reason: Optional[str] = None
    filled_quantity: int = 0
    remaining_quantity: int = 0
    
    @property
    def is_filled(self) -> bool:
        return self.remaining_quantity == 0 and self.filled_quantity > 0
    
    @property
    def is_partial(self) -> bool:
        return self.remaining_quantity > 0 and self.filled_quantity > 0


class MatchingEngine:
    """
    Core matching engine that processes orders and generates trades.
    
    Handles order validation, matching logic, trade generation,
    and special handling for iceberg orders.
    """
    
    def __init__(self, order_book: OrderBook):
        """
        Initialize the matching engine.
        
        Args:
            order_book: Order book to match against
        """
        self.order_book = order_book
        self._trade_id_counter = 0
        self._pending_icebergs: dict[str, NaiveIcebergOrder] = {}  # Track icebergs needing refill
    
    def _generate_trade_id(self) -> str:
        """Generate a unique trade ID"""
        trade_id = f"T{self._trade_id_counter:08d}"
        self._trade_id_counter += 1
        return trade_id
    
    def process_order(self, order: Order, timestamp: float) -> OrderResult:

        if isinstance(order, CancelOrder):
            return self._process_cancel(order, timestamp)
        elif isinstance(order, MarketOrder):
            return self._process_market_order(order, timestamp)
        elif isinstance(order, NaiveIcebergOrder):
            return self._process_iceberg_order(order, timestamp)
        elif isinstance(order, LimitOrder):
            return self._process_limit_order(order, timestamp)
        else:
            return OrderResult(
                order=order,
                trades=[],
                accepted=False,
                rejection_reason=f"Unknown order type: {type(order)}"
            )
    
    def _process_cancel(self, cancel_order: CancelOrder, timestamp: float) -> OrderResult:
        """
        Process a cancellation request.
        
        Args:
            cancel_order: Cancellation order
            timestamp: Current time
        Returns:
            Result indicating success/failure
        """
        cancelled = self.order_book.cancel_order(cancel_order.cancel_order_id)
        
        if cancelled is None:
            return OrderResult(
                order=cancel_order,
                trades=[],
                accepted=False,
                rejection_reason=f"Order {cancel_order.cancel_order_id} not found"
            )
        
        # Also remove from iceberg tracking if applicable
        if cancel_order.cancel_order_id in self._pending_icebergs:
            del self._pending_icebergs[cancel_order.cancel_order_id]
        
        return OrderResult(
            order=cancel_order,
            trades=[],
            accepted=True
        )
    
    def _process_market_order(self, order: MarketOrder, timestamp: float) -> OrderResult:
        """
        Process a market order (immediate execution at best price).
        
        Args:
            order: Market order
            timestamp: Current time
        Returns:
            Result with generated trades
        """
        # Market orders match against opposite side
        matches = self.order_book.match_market_order(order.side, order.quantity)
        
        if not matches:
            # No liquidity available
            order.status = OrderStatus.REJECTED
            return OrderResult(
                order=order,
                trades=[],
                accepted=False,
                rejection_reason="No liquidity available",
                remaining_quantity=order.quantity
            )
        
        # Generate trades
        trades = []
        total_filled = 0
        
        for passive_order, matched_qty, price in matches:
            trade = self._create_trade(
                timestamp=timestamp,
                price=price,
                quantity=matched_qty,
                aggressor_order=order,
                passive_order=passive_order
            )
            trades.append(trade)
            
            order.fill(matched_qty)
            total_filled += matched_qty
            
            # Check if passive order was an iceberg that needs refill
            if isinstance(passive_order, NaiveIcebergOrder) and passive_order.needs_refill:
                self._pending_icebergs[passive_order.order_id] = passive_order
        
        return OrderResult(
            order=order,
            trades=trades,
            accepted=True,
            filled_quantity=total_filled,
            remaining_quantity=order.remaining_quantity
        )
    
    # def _process_limit_order(self, order: LimitOrder, timestamp: float) -> OrderResult:
    #     """
    #     Process a limit order.
    #     First attempts to match against existing orders. Any unfilled
    #     quantity is added to the book.
        
    #     Args:
    #         order: Limit order
    #         timestamp: Current time
    #     Returns:
    #         Result with trades and book placement info
    #     """
    #     trades = []
        
    #     # Check if this order can match immediately
    #     if order.side == OrderSide.BUY:
    #         # Buy order: check if we can match against asks
    #         best_ask = self.order_book.best_ask
    #         can_match = best_ask is not None and order.price >= best_ask
    #     else:
    #         # Sell order: check if we can match against bids
    #         best_bid = self.order_book.best_bid
    #         can_match = best_bid is not None and order.price <= best_bid
        
    #     # Execute immediate matches
    #     if can_match:
    #         # Use market matching logic but respect price limit
    #         remaining = order.remaining_quantity
            
    #         while remaining > 0:
    #             matches = self.order_book.match_market_order(order.side, remaining)
                
    #             if not matches:
    #                 break
                
    #             # Check if match price is acceptable
    #             for passive_order, matched_qty, match_price in matches:
    #                 # Verify price is within limit
    #                 if order.side == OrderSide.BUY and match_price > order.price:
    #                     break
    #                 if order.side == OrderSide.SELL and match_price < order.price:
    #                     break
                    
    #                 trade = self._create_trade(
    #                     timestamp=timestamp,
    #                     price=match_price,
    #                     quantity=matched_qty,
    #                     aggressor_order=order,
    #                     passive_order=passive_order
    #                 )
    #                 trades.append(trade)
                    
    #                 order.fill(matched_qty)
    #                 remaining = order.remaining_quantity
                    
    #                 # Track iceberg refills
    #                 if isinstance(passive_order, NaiveIcebergOrder) and passive_order.needs_refill:
    #                     self._pending_icebergs[passive_order.order_id] = passive_order
                
    #             # Stop if we can't match at acceptable price
    #             if order.side == OrderSide.BUY:
    #                 best_ask = self.order_book.best_ask
    #                 if best_ask is None or best_ask > order.price:
    #                     break
    #             else:
    #                 best_bid = self.order_book.best_bid
    #                 if best_bid is None or best_bid < order.price:
    #                     break
        
    #     # Add any remaining quantity to the book
    #     if order.remaining_quantity > 0:
    #         order.status = OrderStatus.ACTIVE
    #         self.order_book.add_order(order)
        
    #     return OrderResult(
    #         order=order,
    #         trades=trades,
    #         accepted=True,
    #         filled_quantity=order.filled_quantity,
    #         remaining_quantity=order.remaining_quantity
    #     )

    def _process_limit_order(self, order: LimitOrder, timestamp: float) -> OrderResult:
        """
        Process a limit order with atomic iceberg support.
        """
        trades = []
        
        # 1. Match continuously until filled or no more valid liquidity
        while order.remaining_quantity > 0:
            # Check for liquidity
            if order.side == OrderSide.BUY:
                best_price = self.order_book.best_ask
                # Stop if no sellers or price is too high
                if best_price is None or best_price > order.price:
                    break
                passive_side = OrderSide.SELL
            else: # SELL
                best_price = self.order_book.best_bid
                # Stop if no buyers or price is too low
                if best_price is None or best_price < order.price:
                    break
                passive_side = OrderSide.BUY
            
            # Get the actual price level object
            # Note: We need to access the internal book structure or use a helper
            # Since get_level isn't in the provided snippets, we use get_liquidity logic
            # or simply rely on a loop over the best price orders.
            
            # Let's peek at the first order at the best price
            # We need to access the level directly to modify it
            side_dict = self.order_book._asks if passive_side == OrderSide.SELL else self.order_book._bids
            level = side_dict.get(best_price)
            
            if not level or level.is_empty():
                # Should not happen if best_price was valid, but safety check
                if level and level.is_empty():
                    del side_dict[best_price]
                continue
                
            passive_order = level.peek_front()
            
            # Calculate match quantity
            match_qty = min(order.remaining_quantity, passive_order.remaining_quantity)
            
            # Execute Trade
            trade = self._create_trade(
                timestamp=timestamp,
                price=best_price,
                quantity=match_qty,
                aggressor_order=order,
                passive_order=passive_order
            )
            trades.append(trade)
            
            # Update Quantities
            order.fill(match_qty)
            passive_order.fill(match_qty)
            level._total_quantity -= match_qty
            
            # Handle Passive Order Logic (Removal or Refill)
            if passive_order.is_filled:
                # Remove the finished tip/order from the front
                level.orders.popleft()
                # Don't remove from map yet if it might refill
                
                if isinstance(passive_order, NaiveIcebergOrder) and passive_order.needs_refill:
                    if passive_order.refill():
                        # REFILL SUCCESS: Add back to end of queue (priority loss)
                        level.orders.append(passive_order)
                        level._total_quantity += passive_order.remaining_quantity
                        # It stays in the order_map, so no map updates needed
                    else:
                        # REFILL FAILED: Totally done
                        if passive_order.order_id in level._order_map:
                            del level._order_map[passive_order.order_id]
                        self.order_book._discard_order_tracking(passive_order.order_id)
                else:
                    # STANDARD ORDER: Totally done
                    if passive_order.order_id in level._order_map:
                        del level._order_map[passive_order.order_id]
                    self.order_book._discard_order_tracking(passive_order.order_id)
            
            # If the level became empty after this match (and no refill happened), clean it up
            if level.is_empty():
                del side_dict[best_price]
                
        # 2. Add remaining quantity to book
        if order.remaining_quantity > 0:
            order.status = OrderStatus.ACTIVE
            self.order_book.add_order(order)
            
        return OrderResult(
            order=order,
            trades=trades,
            accepted=True,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity
        )
    
    def _process_iceberg_order(self, order: NaiveIcebergOrder, timestamp: float) -> OrderResult:
        """
        Process an iceberg order (only visible tip is shown).
        
        The visible portion is treated as a regular limit order.
        When filled, the order is marked for refill.
        
        Args:
            order: Iceberg order
            timestamp: Current time
            
        Returns:
            Result with trades
        """
        # Set the order quantity to just the visible tip
        order.quantity = order.visible_quantity
        
        # Process as a regular limit order
        result = self._process_limit_order(order, timestamp)
        
        # If filled, mark for refill
        if order.needs_refill:
            self._pending_icebergs[order.order_id] = order
        
        return result
    
    def _create_trade(
        self,
        timestamp: float,
        price: float,
        quantity: int,
        aggressor_order: Order,
        passive_order: Order
    ) -> Trade:
        """
        Create a trade record.
        
        Args:
            timestamp: When trade occurred
            price: Execution price
            quantity: Amount traded
            aggressor_order: Order that crossed the spread
            passive_order: Order sitting in the book
            
        Returns:
            Trade record
        """
        # Determine which order is buy vs sell
        if aggressor_order.side == OrderSide.BUY:
            buy_order_id = aggressor_order.order_id
            sell_order_id = passive_order.order_id
        else:
            buy_order_id = passive_order.order_id
            sell_order_id = aggressor_order.order_id
        
        return Trade(
            trade_id=self._generate_trade_id(),
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            aggressor_side=aggressor_order.side
        )
    
    def get_pending_iceberg_refills(self) -> List[NaiveIcebergOrder]:
        return list(self._pending_icebergs.values())
    
    def refill_iceberg(self, order_id: str) -> bool:
        """
        Refill an iceberg order's visible portion.
        """
        # 1. Check if the order is actually in the pending list
        if order_id not in self._pending_icebergs:
            return False
        
        iceberg = self._pending_icebergs[order_id]
        
        # 2. Attempt to refill (updates quantity and hidden_remaining)
        success = iceberg.refill()
        
        if success:
            # 3. Add refilled order back to the book (this resets its time priority)
            iceberg.status = OrderStatus.ACTIVE
            self.order_book.add_order(iceberg)
            
            # 4. Remove from pending list now that it's back in the book
            del self._pending_icebergs[order_id]
            return True
        else:
            # No more hidden quantity, order is complete
            del self._pending_icebergs[order_id]
            return False
    
    def refill_all_icebergs(self) -> int:
        """
        Refill all pending iceberg orders.
        
        Returns:
            Number of icebergs refilled
        """
        count = 0
        # Use list() to avoid modifying dict during iteration
        for order_id in list(self._pending_icebergs.keys()):
            if self.refill_iceberg(order_id):
                count += 1
        
        return count
