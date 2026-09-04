"""Market Microstructure Signals: Tick data analysis, trade flow toxicity, adverse selection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import numpy as np


@dataclass
class TickData:
    """Single tick data point."""
    timestamp: float
    price: float
    size: float
    side: int  # +1 buy, -1 sell, 0 unknown
    bid: float
    ask: float


@dataclass
class MicrostructureMetrics:
    """Computed microstructure metrics."""
    vpin: float
    kyle_lambda: float
    amihud_illiquidity: float
    roll_spread: float
    order_flow_imbalance: float
    adverse_selection: float
    toxicity_score: float
    bid_ask_spread_bps: float
    effective_spread_bps: float
    realized_spread_bps: float
    price_impact: float
    order_book_resilience: float


class TickBuffer:
    """Rolling buffer for tick data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.ticks: Deque[TickData] = deque(maxlen=max_size)
        self._prices: Deque[float] = deque(maxlen=max_size)
        self._sizes: Deque[float] = deque(maxlen=max_size)
        self._sides: Deque[int] = deque(maxlen=max_size)
        self._bids: Deque[float] = deque(maxlen=max_size)
        self._asks: Deque[float] = deque(maxlen=max_size)
        self._timestamps: Deque[float] = deque(maxlen=max_size)
    
    def add_tick(self, tick: TickData):
        self.ticks.append(tick)
        self._prices.append(tick.price)
        self._sizes.append(tick.size)
        self._sides.append(tick.side)
        self._bids.append(tick.bid)
        self._asks.append(tick.ask)
        self._timestamps.append(tick.timestamp)
    
    def get_arrays(self, window: Optional[int] = None) -> tuple:
        """Get numpy arrays for computation."""
        if window is None or window >= len(self._prices):
            return (
                np.array(self._prices),
                np.array(self._sizes),
                np.array(self._sides),
                np.array(self._bids),
                np.array(self._asks),
                np.array(self._timestamps)
            )
        return (
            np.array(list(self._prices)[-window:]),
            np.array(list(self._sizes)[-window:]),
            np.array(list(self._sides)[-window:]),
            np.array(list(self._bids)[-window:]),
            np.array(list(self._asks)[-window:]),
            np.array(list(self._timestamps)[-window:])
        )
    
    def clear(self):
        self.ticks.clear()
        self._prices.clear()
        self._sizes.clear()
        self._sides.clear()
        self._bids.clear()
        self._asks.clear()
        self._timestamps.clear()


class MicrostructureAnalyzer:
    """Analyzes tick data for microstructure signals."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        micro_cfg = cfg.get("microstructure", {})
        self.bucket_size = micro_cfg.get("vpin_bucket_size", 50)
        self.lookback_ticks = micro_cfg.get("lookback_ticks", 1000)
        self.toxicity_threshold = micro_cfg.get("toxicity_threshold", 0.7)
        
        # Per-symbol tick buffers
        self._buffers: dict[str, TickBuffer] = {}
    
    def get_buffer(self, symbol: str) -> TickBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = TickBuffer(self.lookback_ticks)
        return self._buffers[symbol]
    
    def add_tick(self, symbol: str, tick: TickData):
        buffer = self.get_buffer(symbol)
        buffer.add_tick(tick)
    
    def compute_metrics(self, symbol: str) -> Optional[MicrostructureMetrics]:
        buffer = self.get_buffer(symbol)
        if len(buffer._prices) < 50:
            return None
        
        prices, sizes, sides, bids, asks, timestamps = buffer.get_arrays(self.lookback_ticks)
        
        # VPIN
        vpin = self._compute_vpin(sides, sizes)
        
        # Kyle's Lambda
        kyle_lambda = self._compute_kyle_lambda(prices, sides, sizes)
        
        # Amihud Illiquidity
        amihud = self._compute_amihud(prices, sizes)
        
        # Roll Spread
        roll_spread = self._compute_roll_spread(prices)
        
        # Order Flow Imbalance
        ofi = self._compute_ofi(bids, asks)
        
        # Adverse Selection
        adv_sel = self._compute_adverse_selection(bids, asks, prices)
        
        # Bid-ask spread
        spread_bps = np.mean((asks - bids) / ((asks + bids) / 2)) * 10000
        
        # Effective spread (assuming mid-price)
        mid = (bids + asks) / 2
        effective_spread = 2 * np.abs(prices - mid[:-1]) / mid[:-1] * 10000
        effective_spread_bps = np.mean(effective_spread) if len(effective_spread) > 0 else spread_bps
        
        # Realized spread (price reversal after trade)
        realized_spread = self._compute_realized_spread(prices, sides)
        
        # Price impact
        price_impact = self._compute_price_impact(prices, sides, sizes)
        
        # Order book resilience
        resilience = self._compute_resilience(bids, asks)
        
        # Toxicity score (composite)
        toxicity = self._compute_toxicity(vpin, kyle_lambda, adv_sel, effective_spread_bps)
        
        return MicrostructureMetrics(
            vpin=vpin,
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud,
            roll_spread=roll_spread,
            order_flow_imbalance=ofi,
            adverse_selection=adv_sel,
            toxicity_score=toxicity,
            bid_ask_spread_bps=spread_bps,
            effective_spread_bps=float(effective_spread_bps),
            realized_spread_bps=realized_spread,
            price_impact=price_impact,
            order_book_resilience=resilience
        )
    
    def _compute_vpin(self, sides: np.ndarray, sizes: np.ndarray) -> float:
        """Volume-synchronized Probability of Informed Trading."""
        if len(sides) < self.bucket_size:
            return 0.0
        
        buy_vol = 0.0
        sell_vol = 0.0
        buckets = []
        
        for side, size in zip(sides, sizes):
            if side > 0:
                buy_vol += size
            elif side < 0:
                sell_vol += size
            
            total_vol = buy_vol + sell_vol
            if total_vol >= self.bucket_size:
                buckets.append(abs(buy_vol - sell_vol) / total_vol)
                buy_vol = 0.0
                sell_vol = 0.0
        
        return float(np.mean(buckets)) if buckets else 0.0
    
    def _compute_kyle_lambda(self, prices: np.ndarray, sides: np.ndarray, sizes: np.ndarray) -> float:
        """Kyle's Lambda: price impact per unit signed flow."""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        signed_volume = sides[1:] * sizes[1:]
        
        if np.std(signed_volume) == 0:
            return 0.0
        
        # Regression: returns = lambda * signed_volume + epsilon
        cov = np.cov(returns, signed_volume)[0, 1]
        var = np.var(signed_volume)
        
        return float(cov / var) if var > 0 else 0.0
    
    def _compute_amihud(self, prices: np.ndarray, sizes: np.ndarray) -> float:
        """Amihud illiquidity ratio."""
        if len(prices) < 2:
            return 0.0
        
        returns = np.abs(np.diff(prices) / prices[:-1])
        dollar_vol = sizes[1:] * prices[1:]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = returns / (dollar_vol + 1e-9)
        
        return float(np.nanmean(ratios))
    
    def _compute_roll_spread(self, prices: np.ndarray) -> float:
        """Roll spread estimator from serial covariance."""
        if len(prices) < 3:
            return 0.0
        
        dp = np.diff(prices)
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        
        if cov < 0:
            return float(2 * np.sqrt(-cov))
        return 0.0
    
    def _compute_ofi(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Order Flow Imbalance at top of book."""
        if len(bids) < 2:
            return 0.0
        
        dbid = np.diff(bids)
        dask = np.diff(asks)
        denom = np.abs(dbid) + np.abs(dask) + 1e-9
        
        ofi = (dbid - dask) / denom
        return float(np.mean(ofi))
    
    def _compute_adverse_selection(self, bids: np.ndarray, asks: np.ndarray, prices: np.ndarray) -> float:
        """Adverse selection: fraction of spread explained by subsequent price move."""
        if len(bids) < 3:
            return 0.0
        
        spreads = asks - bids
        mid = (bids + asks) / 2
        mid_returns = np.diff(mid) / mid[:-1]
        
        if np.std(spreads[:-1]) == 0 or np.std(mid_returns) == 0:
            return 0.0
        
        corr = np.corrcoef(spreads[:-1], mid_returns)[0, 1]
        return float(abs(corr)) if not np.isnan(corr) else 0.0
    
    def _compute_realized_spread(self, prices: np.ndarray, sides: np.ndarray) -> float:
        """Realized spread: effective spread minus price reversal."""
        if len(prices) < 3:
            return 0.0
        
        # Simplified: look at price 5 ticks after trade
        horizon = 5
        realized = []
        
        for i in range(horizon, len(prices)):
            if sides[i] != 0:
                effective = 2 * sides[i] * (prices[i] - prices[i-1]) / prices[i-1]
                reversal = sides[i] * (prices[i+horizon] - prices[i]) / prices[i] if i + horizon < len(prices) else 0
                realized.append(effective - reversal)
        
        return float(np.mean(realized) * 10000) if realized else 0.0
    
    def _compute_price_impact(self, prices: np.ndarray, sides: np.ndarray, sizes: np.ndarray) -> float:
        """Average price impact per trade."""
        if len(prices) < 2:
            return 0.0
        
        impacts = []
        for i in range(1, len(prices)):
            if sides[i] != 0 and sizes[i] > 0:
                impact = sides[i] * (prices[i] - prices[i-1]) / prices[i-1] / sizes[i]
                impacts.append(impact * 10000)  # bps per share
        
        return float(np.mean(impacts)) if impacts else 0.0
    
    def _compute_resilience(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Order book resilience: how fast liquidity returns after depletion."""
        if len(bids) < 20:
            return 1.0
        
        # Find large spread events and measure recovery
        spreads = asks - bids
        mid_spread = np.median(spreads)
        wide_spreads = spreads > mid_spread * 2
        
        recoveries = []
        in_wide = False
        start_idx = 0
        
        for i, is_wide in enumerate(wide_spreads):
            if is_wide and not in_wide:
                in_wide = True
                start_idx = i
            elif not is_wide and in_wide:
                in_wide = False
                recovery_time = i - start_idx
                if recovery_time > 0:
                    recoveries.append(1.0 / recovery_time)
        
        return float(np.mean(recoveries)) if recoveries else 1.0
    
    def _compute_toxicity(self, vpin: float, kyle_lambda: float, adv_sel: float, eff_spread: float) -> float:
        """Composite toxicity score."""
        # Normalize components
        vpin_norm = min(vpin * 2, 1.0)  # VPIN > 0.5 is high
        lambda_norm = min(abs(kyle_lambda) * 10000, 1.0)
        adv_norm = adv_sel
        spread_norm = min(eff_spread / 20.0, 1.0)  # 20 bps spread is high
        
        toxicity = 0.3 * vpin_norm + 0.3 * lambda_norm + 0.2 * adv_norm + 0.2 * spread_norm
        return float(np.clip(toxicity, 0, 1))
    
    def is_toxic(self, symbol: str) -> bool:
        """Check if market is toxic for trading."""
        metrics = self.compute_metrics(symbol)
        if metrics is None:
            return False
        return metrics.toxicity_score > self.toxicity_threshold


class TradeFlowClassifier:
    """Classifies trade flow for toxicity detection."""
    
    def __init__(self):
        self._recent_trades: Deque[dict] = deque(maxlen=1000)
    
    def add_trade(self, symbol: str, price: float, size: float, side: int, timestamp: float):
        self._recent_trades.append({
            'symbol': symbol,
            'price': price,
            'size': size,
            'side': side,
            'timestamp': timestamp
        })
    
    def get_flow_toxicity(self, symbol: str, window: int = 100) -> float:
        """Get flow toxicity for symbol."""
        trades = [t for t in self._recent_trades if t['symbol'] == symbol][-window:]
        if len(trades) < 10:
            return 0.0
        
        # Large sell orders in declining market = toxic
        prices = np.array([t['price'] for t in trades])
        sides = np.array([t['side'] for t in trades])
        sizes = np.array([t['size'] for t in trades])
        
        # Price trend
        price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        # Sell pressure in declining market
        sell_vol = np.sum(sizes[sides < 0])
        buy_vol = np.sum(sizes[sides > 0])
        total_vol = sell_vol + buy_vol
        
        if total_vol == 0:
            return 0.0
        
        sell_ratio = sell_vol / total_vol
        
        # Toxic if heavy selling in downtrend
        if price_trend < -0.001 and sell_ratio > 0.7:
            return min(sell_ratio * abs(price_trend) * 100, 1.0)
        
        return 0.0