"""
Limit Order Book implementation with price-time priority.

The order book maintains buy (bid) and sell (ask) sides as sorted
price levels. Each price level contains a queue of orders at that price,
processed in time priority (FIFO).
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from sortedcontainers import SortedDict

from core.order import LimitOrder, OrderSide


class PriceLevel:
    """
    A single price level in the order book.
    
    Maintains a FIFO queue of orders at this price, implementing
    price-time priority matching.
    """
    
    def __init__(self, price: float):
        self.price = price
        self.orders: Deque[LimitOrder] = deque()
        self._total_quantity = 0
        self._order_map: Dict[str, LimitOrder] = {}  # Fast lookup by order_id
    
    @property
    def total_quantity(self) -> int:
        return self._total_quantity
    
    @property
    def num_orders(self) -> int:
        return len(self.orders)
    
    def is_empty(self) -> bool:
        return self.num_orders == 0
    
    def add_order(self, order: LimitOrder):
        """
        Add an order to the back of the queue (time priority).
        
        Args:
            order: Limit order to add
        """
        if order.price != self.price:
            raise ValueError(
                f"Order price {order.price} does not match level price {self.price}"
            )
        
        self.orders.append(order)
        self._order_map[order.order_id] = order
        self._total_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> Optional[LimitOrder]:
        """
        Remove an order from this price level.
        
        Args:
            order_id: ID of order to remove
            
        Returns:
            The removed order, or None if not found
        """
        if order_id not in self._order_map:
            return None
        
        order = self._order_map.pop(order_id)
        self.orders.remove(order)
        self._total_quantity -= order.remaining_quantity
        
        return order
    
    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        return self._order_map.get(order_id)
    
    def peek_front(self) -> Optional[LimitOrder]:
        return self.orders[0] if self.orders else None
    
    def match(self, quantity: int) -> List[Tuple[LimitOrder, int]]:
        """
        Match incoming quantity against orders at this level.
        Processes orders in FIFO order until quantity is exhausted
        or no more orders remain.
        
        Args:
            quantity: Amount to match
        Returns:
            List of (order, matched_quantity) tuples
        """
        matches = []
        remaining = quantity
        
        while remaining > 0 and self.orders:
            order = self.orders[0]
            available = order.remaining_quantity
            matched = min(remaining, available)
            
            # Record the match
            matches.append((order, matched))
            
            # Update order
            order.fill(matched)
            self._total_quantity -= matched
            remaining -= matched
            
            # Remove filled orders
            if order.is_filled:
                self.orders.popleft()
                self._order_map.pop(order.order_id)
        
        return matches
    
    def __repr__(self):
        return f"PriceLevel(price={self.price}, qty={self.total_quantity}, orders={self.num_orders})"


class OrderBook:
    """
    Full limit order book with buy and sell sides.
    
    Maintains price-time priority for order matching. Buy orders are
    sorted descending (highest first), sell orders ascending (lowest first).
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialize the order book.
        
        Args:
            tick_size: Minimum price increment
        """
        self.tick_size = tick_size
        
        # Buy side: sorted descending (highest price first)
        # Using negative prices as keys for descending sort
        self._bids: SortedDict[float, PriceLevel] = SortedDict()
        
        # Sell side: sorted ascending (lowest price first)
        self._asks: SortedDict[float, PriceLevel] = SortedDict()
        
        # Fast order lookup
        self._all_orders: Dict[str, LimitOrder] = {}
        
        # Track which side each order is on
        self._order_side_map: Dict[str, OrderSide] = {}
    
    def round_price(self, price: float) -> float:
        """Round price to nearest tick"""
        return round(price / self.tick_size) * self.tick_size
    
    @property
    def best_bid(self) -> Optional[float]:
        """Highest buy price"""
        if not self._bids:
            return None
        return self._bids.peekitem(-1)[0]  # Last item = highest price
    
    @property
    def best_ask(self) -> Optional[float]:
        """Lowest sell price"""
        if not self._asks:
            return None
        return self._asks.peekitem(0)[0]  # First item = lowest price
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid
    
    @property
    def mid_price(self) -> Optional[float]:
        """Midpoint between best bid and ask"""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2
    
    def add_order(self, order: LimitOrder):
        """
        Add a limit order to the book.
        
        Args:
            order: Limit order to add
        """
        price = self.round_price(order.price)
        side_dict = self._bids if order.side == OrderSide.BUY else self._asks
        
        # Create price level if it doesn't exist
        if price not in side_dict:
            side_dict[price] = PriceLevel(price)
        
        order.price = price
        level = side_dict[price]
        level.add_order(order)
        
        # Track the order
        self._all_orders[order.order_id] = order
        self._order_side_map[order.order_id] = order.side
    
    def cancel_order(self, order_id: str) -> Optional[LimitOrder]:
        """
        Cancel an order by ID.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            The cancelled order, or None if not found
        """
        if order_id not in self._all_orders:
            return None
        
        order = self._all_orders[order_id]
        side = self._order_side_map[order_id]
        side_dict = self._bids if side == OrderSide.BUY else self._asks
        
        # Remove from price level
        level = side_dict.get(order.price)
        if level:
            level.remove_order(order_id)
            
            # Clean up empty price levels
            if level.is_empty():
                del side_dict[order.price]
        
        # Clean up tracking
        del self._all_orders[order_id]
        del self._order_side_map[order_id]
        
        order.cancel()
        return order
    
    def get_order(self, order_id: str) -> Optional[LimitOrder]:
        """Look up an order by ID"""
        return self._all_orders.get(order_id)

    def get_level(self, price: float, side: OrderSide) -> Optional[PriceLevel]:
        """Return the price level at `price` on the requested side, if present."""
        side_dict = self._bids if side == OrderSide.BUY else self._asks
        return side_dict.get(self.round_price(price))

    def _discard_order_tracking(self, order_id: str):
        """Remove an order from global tracking if it is no longer in the book."""
        self._all_orders.pop(order_id, None)
        self._order_side_map.pop(order_id, None)
    
    def match_market_order(self, side: OrderSide, quantity: int) -> List[Tuple[LimitOrder, int, float]]:
        """
        Match a market order against the book.
        
        Args:
            side: BUY (matches against asks) or SELL (matches against bids)
            quantity: Amount to match
            
        Returns:
            List of (order, matched_quantity, price) tuples
        """
        # Market buy orders match against asks (sell side)
        # Market sell orders match against bids (buy side)
        side_dict = self._asks if side == OrderSide.BUY else self._bids
        
        matches = []
        remaining = quantity
        
        # Iterate through price levels in order
        prices_to_remove = []
        
        # Get prices in the right order
        if side == OrderSide.BUY:
            # Buy: take from best ask upward
            price_iter = iter(side_dict.items())
        else:
            # Sell: take from best bid downward
            price_iter = reversed(list(side_dict.items()))
        
        for price, level in price_iter:
            if remaining <= 0:
                break
            
            # Match at this level
            level_matches = level.match(remaining)
            
            for order, matched_qty in level_matches:
                matches.append((order, matched_qty, price))
                remaining -= matched_qty
                if order.is_filled:
                    self._discard_order_tracking(order.order_id)
            
            # Mark empty levels for removal
            if level.is_empty():
                prices_to_remove.append(price)
        
        # Clean up empty levels
        for price in prices_to_remove:
            del side_dict[price]
        
        return matches
    
    def get_depth(self, side: OrderSide, levels: int = 10) -> List[Tuple[float, int, int]]:
        """
        Get order book depth on one side.
        
        Args:
            side: Which side to query
            levels: Number of price levels to return
            
        Returns:
            List of (price, total_quantity, num_orders) tuples
        """
        side_dict = self._bids if side == OrderSide.BUY else self._asks

        items = list(side_dict.items())
        if side == OrderSide.BUY:
            items = list(reversed(items[-levels:]))
        else:
            items = items[:levels]

        depth = []
        for price, level in items:
            depth.append((price, level.total_quantity, level.num_orders))
        
        return depth
    
    def get_total_depth(self, side: OrderSide, num_levels: Optional[int] = None) -> int:
        """
        Get total quantity on one side of the book.
        
        Args:
            side: Which side to query
            num_levels: Limit to top N levels (None = all levels)
            
        Returns:
            Total quantity
        """
        side_dict = self._bids if side == OrderSide.BUY else self._asks
        
        if num_levels is None:
            return sum(level.total_quantity for level in side_dict.values())

        items = list(side_dict.items())
        if side == OrderSide.BUY:
            items = items[-num_levels:]
        else:
            items = items[:num_levels]

        total = 0
        for _, level in items:
            total += level.total_quantity
        
        return total
    
    def get_liquidity_at_price(self, price: float, side: OrderSide) -> int:
        """Get total quantity at a specific price level"""
        side_dict = self._bids if side == OrderSide.BUY else self._asks
        price = self.round_price(price)
        
        level = side_dict.get(price)
        return level.total_quantity if level else 0
    
    def snapshot(self) -> Dict:
        """
        Get a complete snapshot of the current book state.
        
        Returns:
            Dict with timestamp, best bid/ask, spread, and depth
        """
        return {
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'bid_depth': self.get_depth(OrderSide.BUY, levels=10),
            'ask_depth': self.get_depth(OrderSide.SELL, levels=10),
            'total_bid_quantity': self.get_total_depth(OrderSide.BUY),
            'total_ask_quantity': self.get_total_depth(OrderSide.SELL),
        }
    
    def __repr__(self):
        bid_str = f"${self.best_bid:.2f}" if self.best_bid else "N/A"
        ask_str = f"${self.best_ask:.2f}" if self.best_ask else "N/A"
        spread_str = f"${self.spread:.2f}" if self.spread else "N/A"
        
        return (
            f"OrderBook(bid={bid_str}, ask={ask_str}, spread={spread_str}, "
            f"orders={len(self._all_orders)})"
        )
