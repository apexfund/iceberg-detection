"""
Market simulator - main orchestrator for the LOB simulation.

Coordinates the order book, matching engine, event queue, and agents
to run a complete market simulation with iceberg order injection.
"""

import random
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from core.order import Order, OrderStatus, NaiveIcebergOrder
from core.order_book import OrderBook
from core.matching_engine import MatchingEngine, Trade, OrderResult
from core.event_queue import EventQueue, EventType, PeriodicEvent


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation"""
    initial_mid_price: float = 100.0
    tick_size: float = 0.01
    start_time: float = 0.0
    end_time: float = 3600.0  # 1 hour in seconds
    snapshot_interval: float = 1.0  # Take snapshot every second
    random_seed: Optional[int] = None


@dataclass
class SimulationStats:
    """Statistics collected during simulation"""
    total_orders: int = 0
    total_trades: int = 0
    total_volume: int = 0
    total_value: float = 0.0
    iceberg_orders_submitted: int = 0
    iceberg_refills: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0


class MarketSimulator:
    """
    Main simulation orchestrator.
    
    Manages the simulation lifecycle, processes events, collects data,
    and coordinates all components.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Core components
        self.order_book = OrderBook(tick_size=config.tick_size)
        self.matching_engine = MatchingEngine(self.order_book)
        self.event_queue = EventQueue(start_time=config.start_time)
        
        # Statistics
        self.stats = SimulationStats()
        
        # Event history
        self.order_history: List[Order] = []
        self.trade_history: List[Trade] = []
        self.snapshots: List[Dict] = []
        
        # Callbacks
        self._on_trade_callbacks: List[Callable[[Trade], None]] = []
        self._on_order_callbacks: List[Callable[[Order, OrderResult], None]] = []
        self._on_snapshot_callbacks: List[Callable[[Dict], None]] = []
        
        # Setup periodic snapshots
        if config.snapshot_interval > 0:
            self._setup_snapshots()
    
    @property
    def current_time(self) -> float:
        """Current simulation time"""
        return self.event_queue.current_time
    
    def _setup_snapshots(self):
        """Schedule periodic order book snapshots"""
        def take_snapshot():
            snapshot = {
                'timestamp': self.current_time,
                **self.order_book.snapshot()
            }
            self.snapshots.append(snapshot)
            
            # Trigger callbacks
            for callback in self._on_snapshot_callbacks:
                callback(snapshot)
        
        PeriodicEvent(
            event_queue=self.event_queue,
            interval=self.config.snapshot_interval,
            event_type=EventType.SNAPSHOT,
            callback=take_snapshot,
            start_time=self.config.start_time + self.config.snapshot_interval
        )
    
    # def submit_order(self, order: Order, delay: float = 0.0):
    #     """
    #     Submit an order to the market.
        
    #     Args:
    #         order: Order to submit
    #         delay: Delay before processing (0 = immediate)
    #     """
    #     def process_order_event():
    #         # Set timestamp
    #         order.timestamp = self.current_time
            
    #         # Process the order
    #         result = self.matching_engine.process_order(order, self.current_time)
            
    #         # Update statistics
    #         self.stats.total_orders += 1
    #         if isinstance(order, NaiveIcebergOrder):
    #             self.stats.iceberg_orders_submitted += 1
            
    #         if not result.accepted:
    #             self.stats.rejected_orders += 1
            
    #         # Record order
    #         self.order_history.append(order)
            
    #         # Record trades
    #         for trade in result.trades:
    #             self.trade_history.append(trade)
    #             self.stats.total_trades += 1
    #             self.stats.total_volume += trade.quantity
    #             self.stats.total_value += trade.price * trade.quantity
            
    #         # Trigger callbacks
    #         for callback in self._on_order_callbacks:
    #             callback(order, result)
            
    #         for trade in result.trades:
    #             for callback in self._on_trade_callbacks:
    #                 callback(trade)
            
    #         # Schedule iceberg refills
    #         if isinstance(order, NaiveIcebergOrder) and order.needs_refill:
    #             self._schedule_iceberg_refill(order)
            
    #         return result
        
    #     # Schedule the order processing
    #     if delay <= 0:
    #         self.event_queue.schedule(
    #             timestamp=self.current_time,
    #             event_type=EventType.ORDER_SUBMISSION,
    #             callback=process_order_event
    #         )
    #     else:
    #         self.event_queue.schedule_after(
    #             delay=delay,
    #             event_type=EventType.ORDER_SUBMISSION,
    #             callback=process_order_event
    #         )
        
    def submit_order(self, order: Order, delay: float = 0.0):
        """
        Submit an order to the market.
        """
        def process_order_event():
            # Set timestamp
            order.timestamp = self.current_time

            # Process the order
            result = self.matching_engine.process_order(order, self.current_time)

            # Update statistics
            self.stats.total_orders += 1
            if isinstance(order, NaiveIcebergOrder):
                self.stats.iceberg_orders_submitted += 1
                pass

            if not result.accepted:
                self.stats.rejected_orders += 1

            # Record order
            self.order_history.append(order)

            # Record trades
            for trade in result.trades:
                self.trade_history.append(trade)
                self.stats.total_trades += 1
                self.stats.total_volume += trade.quantity
                self.stats.total_value += trade.price * trade.quantity

            # Trigger callbacks
            for callback in self._on_order_callbacks:
                callback(order, result)

            for trade in result.trades:
                for callback in self._on_trade_callbacks:
                    callback(trade)

            # Refill any icebergs that were fully filled (e.g. by this market order)
            refilled = self.matching_engine.refill_all_icebergs()
            self.stats.iceberg_refills += refilled

            return result

        # Schedule the order processing
        if delay <= 0:
            self.event_queue.schedule(
                timestamp=self.current_time,
                event_type=EventType.ORDER_SUBMISSION,
                callback=process_order_event
            )
        else:
            self.event_queue.schedule_after(
                delay=delay,
                event_type=EventType.ORDER_SUBMISSION,
                callback=process_order_event
            )
    
    def _schedule_iceberg_refill(self, iceberg: NaiveIcebergOrder, delay: float = 0.001):
        """
        Schedule refill of an iceberg order.
        
        Args:
            iceberg: Iceberg order to refill
            delay: Small delay to simulate processing time
        """
        def refill_event():
            success = self.matching_engine.refill_iceberg(iceberg.order_id)
            if success:
                self.stats.iceberg_refills += 1
                
                # Check if it needs another refill after being added back
                if iceberg.needs_refill:
                    self._schedule_iceberg_refill(iceberg)
            
            return success
        
        self.event_queue.schedule_after(
            delay=delay,
            event_type=EventType.ICEBERG_REFILL,
            callback=refill_event
        )
    
    def cancel_order(self, order_id: str, delay: float = 0.0):
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            delay: Delay before cancellation
        """
        from core.order import CancelOrder
        
        cancel = CancelOrder(
            order_id=f"CANCEL_{order_id}",
            timestamp=self.current_time,
            trader_id="SYSTEM",
            side=None,  # Not relevant for cancels
            quantity=1,  # placeholder; CancelOrder.__post_init__ sets quantity=0
            cancel_order_id=order_id,
        )
        
        def process_cancel():
            cancel.timestamp = self.current_time
            result = self.matching_engine.process_order(cancel, self.current_time)
            if result.accepted:
                self.stats.cancelled_orders += 1
                self.order_history.append(cancel)
            return result
        
        if delay <= 0:
            self.event_queue.schedule(
                timestamp=self.current_time,
                event_type=EventType.ORDER_CANCELLATION,
                callback=process_cancel
            )
        else:
            self.event_queue.schedule_after(
                delay=delay,
                event_type=EventType.ORDER_CANCELLATION,
                callback=process_cancel
            )
    
    def run(self, until_time: Optional[float] = None) -> SimulationStats:
        """
        Run the simulation.
        
        Args:
            until_time: Run until this time (None = run until end_time)
            
        Returns:
            Simulation statistics
        """
        end = until_time if until_time is not None else self.config.end_time
        
        # Schedule end of simulation
        self.event_queue.schedule(
            timestamp=end,
            event_type=EventType.END_SIMULATION,
            callback=lambda: None
        )
        
        # Process all events
        while not self.event_queue.is_empty:
            event, result = self.event_queue.process_next()
            
            if event.event_type == EventType.END_SIMULATION:
                break
        
        return self.stats
    
    def step(self) -> bool:
        """
        Process the next event in the queue.
        
        Returns:
            True if an event was processed, False if queue is empty
        """
        if self.event_queue.is_empty:
            return False
        
        self.event_queue.process_next()
        return True
    
    def run_steps(self, num_steps: int) -> int:
        """
        Run a specific number of simulation steps.
        
        Args:
            num_steps: Number of events to process
            
        Returns:
            Actual number of steps processed
        """
        count = 0
        for _ in range(num_steps):
            if not self.step():
                break
            count += 1
        
        return count
    
    # Callback registration
    
    def on_trade(self, callback: Callable[[Trade], None]):
        """Register a callback for when trades occur"""
        self._on_trade_callbacks.append(callback)
    
    def on_order(self, callback: Callable[[Order, OrderResult], None]):
        """Register a callback for when orders are processed"""
        self._on_order_callbacks.append(callback)
    
    def on_snapshot(self, callback: Callable[[Dict], None]):
        """Register a callback for when snapshots are taken"""
        self._on_snapshot_callbacks.append(callback)
    
    # Utility methods
    
    def get_current_state(self) -> Dict:
        """Get current market state"""
        return {
            'time': self.current_time,
            'book': self.order_book.snapshot(),
            'stats': {
                'total_orders': self.stats.total_orders,
                'total_trades': self.stats.total_trades,
                'total_volume': self.stats.total_volume,
                'avg_price': self.stats.total_value / self.stats.total_volume if self.stats.total_volume > 0 else 0,
            }
        }
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.order_book = OrderBook(tick_size=self.config.tick_size)
        self.matching_engine = MatchingEngine(self.order_book)
        self.event_queue = EventQueue(start_time=self.config.start_time)
        
        self.stats = SimulationStats()
        self.order_history.clear()
        self.trade_history.clear()
        self.snapshots.clear()
        
        if self.config.snapshot_interval > 0:
            self._setup_snapshots()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
    
    def __repr__(self):
        return (
            f"MarketSimulator(time={self.current_time:.2f}, "
            f"orders={self.stats.total_orders}, "
            f"trades={self.stats.total_trades}, "
            f"volume={self.stats.total_volume})"
        )


def create_default_simulator(
    duration: float = 3600.0,
    random_seed: Optional[int] = 42
) -> MarketSimulator:
    """
    Create a simulator with default configuration.
    
    Args:
        duration: Simulation duration in seconds
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured MarketSimulator
    """
    config = SimulationConfig(
        initial_mid_price=100.0,
        tick_size=0.01,
        start_time=0.0,
        end_time=duration,
        snapshot_interval=1.0,
        random_seed=random_seed
    )
    
    return MarketSimulator(config)