"""
Paper Trading Engine Module.

Real-time paper trading simulation with live data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional
import time

from ..core.models import Allocation, Portfolio, Price
from ..strategies.base import BaseStrategy
from ..data.sources import BinanceRESTSource


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    fee: float

    def __str__(self) -> str:
        return (
            f"{self.side} {self.symbol}: {self.quantity:.6f} @ "
            f"${self.price:,.2f} (fee: ${self.fee:.2f})"
        )


@dataclass
class PaperTradingState:
    """Current state of paper trading session."""
    portfolio: Portfolio
    current_allocation: Optional[Allocation] = None
    trades: List[Trade] = field(default_factory=list)
    portfolio_history: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def pnl(self) -> float:
        """Profit/Loss from initial capital."""
        if not self.portfolio_history:
            return 0.0
        initial = self.portfolio_history[0]
        current = self.portfolio_history[-1]
        return current - initial

    @property
    def pnl_percent(self) -> float:
        """PnL as percentage."""
        if not self.portfolio_history or self.portfolio_history[0] == 0:
            return 0.0
        initial = self.portfolio_history[0]
        return self.pnl / initial


class PaperTradingEngine:
    """
    Paper trading engine for real-time simulation.

    Features:
    - Live price updates from Binance
    - Real-time portfolio tracking
    - Trade execution simulation with fees
    - Dashboard display
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        symbol_a: str = "BTCUSDT",
        symbol_b: str = "ETHUSDT",
        price_provider: Optional[Callable[[], Price]] = None,
    ):
        self.strategy = strategy
        self.fee_rate = fee_rate
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b

        self.state = PaperTradingState(
            portfolio=Portfolio(cash=initial_capital)
        )
        self._price_source = BinanceRESTSource(symbol_a, symbol_b)
        self._price_provider = price_provider
        self._price_history: List[Price] = []
        self._epoch = 0

    def _execute_rebalance(self, allocation: Allocation, price: Price) -> List[Trade]:
        """Execute trades to achieve target allocation."""
        trades = []
        portfolio = self.state.portfolio
        current_value = portfolio.value(price)

        
        target_a_qty = (
            (current_value * allocation.asset_a) / price.asset_a
            if price.asset_a > 0
            else 0
        )
        target_b_qty = (
            (current_value * allocation.asset_b) / price.asset_b
            if price.asset_b > 0
            else 0
        )

        
        delta_a = target_a_qty - portfolio.asset_a_qty
        if abs(delta_a) > 1e-8:
            trade_value = abs(delta_a) * price.asset_a
            fee = trade_value * self.fee_rate

            trades.append(Trade(
                timestamp=datetime.now(),
                symbol=self.symbol_a,
                side='BUY' if delta_a > 0 else 'SELL',
                quantity=abs(delta_a),
                price=price.asset_a,
                fee=fee,
            ))

        
        delta_b = target_b_qty - portfolio.asset_b_qty
        if abs(delta_b) > 1e-8:
            trade_value = abs(delta_b) * price.asset_b
            fee = trade_value * self.fee_rate

            trades.append(Trade(
                timestamp=datetime.now(),
                symbol=self.symbol_b,
                side='BUY' if delta_b > 0 else 'SELL',
                quantity=abs(delta_b),
                price=price.asset_b,
                fee=fee,
            ))

        
        portfolio.rebalance(allocation, price, self.fee_rate)

        return trades

    def _get_current_price(self) -> Price:
        """Obtain the current price from provider or the default REST source.

        The helper centralizes error handling and allows the engine to
        be easily tested by mocking this method.
        """
        if self._price_provider:
            return self._price_provider()
        return self._price_source.get_current_price()

    def tick(self) -> Dict:
        """
        Process one trading tick.

        Returns current state summary.
        """
        
        try:
            price = self._get_current_price()
        except Exception as e:
            return {'error': str(e)}

        self._price_history.append(price)

        
        allocation = self.strategy.decide(self._epoch, self._price_history)
        self.state.current_allocation = allocation

        
        trades = self._execute_rebalance(allocation, price)
        self.state.trades.extend(trades)

        
        current_value = self.state.portfolio.value(price)
        self.state.portfolio_history.append(current_value)

        self._epoch += 1

        return {
            'epoch': self._epoch,
            'price': price,
            'allocation': allocation,
            'portfolio_value': current_value,
            'pnl': self.state.pnl,
            'pnl_percent': self.state.pnl_percent,
            'trades': trades,
        }

    def run(
        self,
        duration_seconds: int = 3600,
        interval_seconds: float = 60.0,
        on_tick: Optional[Callable[[Dict], None]] = None,
    ) -> PaperTradingState:
        """
        Run paper trading session.

        Args:
            duration_seconds: Total duration to run
            interval_seconds: Seconds between ticks
            on_tick: Callback function for each tick

        Returns:
            Final PaperTradingState
        """
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            result = self.tick()

            if on_tick:
                result['elapsed'] = time.time() - start_time
                result['remaining'] = duration_seconds - result['elapsed']
                on_tick(result)

            time.sleep(interval_seconds)

        return self.state


# =============================================================================
# DASHBOARD DISPLAY
# =============================================================================

class DashboardDisplay:
    """
    Terminal dashboard for paper trading visualization.
    """

    def __init__(self, symbol_a: str = "BTCUSDT", symbol_b: str = "ETHUSDT"):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b

    def render(self, tick_result: Dict) -> str:
        """Render dashboard string."""
        if 'error' in tick_result:
            return f"⚠️  Error: {tick_result['error']}"

        price = tick_result['price']
        allocation = tick_result['allocation']
        elapsed = tick_result.get('elapsed', 0)
        remaining = tick_result.get('remaining', 0)

        lines = [
            "=" * 60,
            "🤖 PAPER TRADING DASHBOARD",
            "=" * 60,
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"⏱️  Elapsed: {int(elapsed)}s / Remaining: {int(remaining)}s",
            "",
            "📊 LIVE PRICES:",
            f"  {self.symbol_a}: ${price.asset_a:,.2f}",
            f"  {self.symbol_b}: ${price.asset_b:,.2f}",
            "",
            "💰 PORTFOLIO:",
            f"  Value: ${tick_result['portfolio_value']:,.2f}",
            f"  PnL: ${tick_result['pnl']:,.2f} ({tick_result['pnl_percent']:+.2%})",
            "",
            "🎯 ALLOCATION:",
            f"  Asset A: {allocation.asset_a:.1%}",
            f"  Asset B: {allocation.asset_b:.1%}",
            f"  Cash: {allocation.cash:.1%}",
            "",
        ]

        if tick_result.get('trades'):
            lines.append("📝 TRADES THIS TICK:")
            for trade in tick_result['trades'][-5:]:
                lines.append(f"  {trade}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def display(self, tick_result: Dict) -> None:
        """Print dashboard to terminal (clears screen first)."""
        
        print("\033[H\033[J", end="")
        print(self.render(tick_result))
        print("\nPress Ctrl+C to stop")
