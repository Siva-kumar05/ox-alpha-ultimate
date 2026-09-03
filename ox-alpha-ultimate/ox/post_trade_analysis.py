"""Post-Trade Analysis and Alpha Decay Tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from .core import LOG, iso, DB


@dataclass
class TradeAnalysis:
    """Analysis of a single closed trade."""
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    side: str
    pnl: float
    pnl_pct: float
    hold_bars: int
    exit_reason: str
    max_favorable_pct: float
    max_adverse_pct: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion
    efficiency: float  # Actual PnL / MFE
    slippage_bps: float
    commissions: float


@dataclass
class AlphaDecayMetrics:
    """Alpha decay metrics for a strategy."""
    strategy_id: str
    avg_hold_bars: float
    decay_half_life: float  # Bars until alpha decays to 50%
    decay_rate: float  # Exponential decay rate
    initial_edge_bps: float
    final_edge_bps: float
    trades_analyzed: int
    profitable_pct: float
    avg_efficiency: float
    regime_performance: Dict[str, float]


@dataclass
class StrategyPerformance:
    """Comprehensive strategy performance."""
    strategy_id: str
    total_trades: int
    win_rate: float
    avg_pnl: float
    sharpe: float
    sortino: float
    max_dd: float
    profit_factor: float
    avg_hold_bars: float
    turnover: float
    cost_drag: float
    alpha_decay: Optional[AlphaDecayMetrics]
    regime_breakdown: Dict[str, Dict]
    time_of_day_breakdown: Dict[str, Dict]


class PostTradeAnalyzer:
    """Analyzes closed trades for alpha decay and performance attribution."""
    
    def __init__(self, cfg, db: DB):
        self.cfg = cfg
        self.db = db
        self.analysis_cfg = cfg.get("post_trade_analysis", {})
        self.lookback_trades = self.analysis_cfg.get("lookback_trades", 500)
        self.decay_window = self.analysis_cfg.get("decay_window_bars", 20)
        self.min_trades_for_decay = self.analysis_cfg.get("min_trades_for_decay", 30)
        
        # Cached results
        self._cached_analysis: Dict[str, AlphaDecayMetrics] = {}
        self._cache_timestamp: Optional[str] = None
    
    def analyze_trade(self, trade_row: tuple, price_series: pd.DataFrame) -> TradeAnalysis:
        """Analyze a single trade with price path."""
        tid, sym, side, qty, inpx, outpx, pnl, charges, strat, intime, outtime, exit_reason = trade_row
        
        # Get price path during trade
        entry_ts = pd.Timestamp(intime).timestamp() if isinstance(intime, str) else intime
        exit_ts = pd.Timestamp(outtime).timestamp() if isinstance(outtime, str) else outtime
        
        # Filter price series to trade window
        mask = (price_series['ts'] >= entry_ts) & (price_series['ts'] <= exit_ts)
        trade_prices = price_series[mask]['c'].values
        
        if len(trade_prices) < 2:
            trade_prices = np.array([inpx, outpx])
        
        entry_price = float(inpx)
        exit_price = float(outpx)
        
        # Calculate MAE/MFE
        if side == "LONG":
            price_changes = (trade_prices - entry_price) / entry_price
        else:
            price_changes = (entry_price - trade_prices) / entry_price
        
        max_favorable = float(np.max(price_changes)) if len(price_changes) > 0 else 0.0
        max_adverse = float(np.min(price_changes)) if len(price_changes) > 0 else 0.0
        
        mae = abs(max_adverse) * 100
        mfe = max_favorable * 100
        actual_pnl_pct = (pnl / (entry_price * qty)) * 100
        
        efficiency = actual_pnl_pct / mfe if mfe > 0 else 0.0
        
        # Slippage
        slippage_bps = 0.0
        # Would need reference price at signal time
        
        hold_bars = len(trade_prices)
        
        return TradeAnalysis(
            symbol=sym,
            entry_time=intime,
            exit_time=outtime,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=qty,
            side=side,
            pnl=pnl,
            pnl_pct=actual_pnl_pct,
            hold_bars=hold_bars,
            exit_reason=exit_reason,
            max_favorable_pct=max_favorable * 100,
            max_adverse_pct=max_adverse * 100,
            mae=mae,
            mfe=mfe,
            efficiency=efficiency,
            slippage_bps=slippage_bps,
            commissions=charges
        )
    
    def compute_alpha_decay(self, strategy_id: str) -> Optional[AlphaDecayMetrics]:
        """Compute alpha decay for a strategy."""
        # Get trades for strategy
        rows = self.db.q("""
            SELECT tid, sym, side, qty, inpx, outpx, pnl, charges, strat, intime, outtime, exit_reason
            FROM trades WHERE strat LIKE ? ORDER BY tid DESC LIMIT ?
        """, (f"%{strategy_id}%", self.lookback_trades))
        
        if len(rows) < self.min_trades_for_decay:
            return None
        
        # Get price data for each trade
        symbols = list(set(r[1] for r in rows))
        price_data = {}
        for sym in symbols:
            pf = self.db.q("SELECT ts, c FROM candles WHERE sym=? ORDER BY ts", (sym,))
            if pf:
                price_data[sym] = pd.DataFrame(pf, columns=['ts', 'c'])
        
        analyses = []
        for row in rows:
            sym = row[1]
            if sym in price_data:
                analyses.append(self.analyze_trade(row, price_data[sym]))
        
        if len(analyses) < self.min_trades_for_decay:
            return None
        
        # Compute decay by hold time bins
        hold_bins = defaultdict(list)
        for a in analyses:
            bin_key = min(a.hold_bars // 5 * 5, self.decay_window)
            hold_bins[bin_key].append(a.pnl_pct)
        
        if not hold_bins:
            return None
        
        # Fit exponential decay
        bins = sorted(hold_bins.keys())
        avg_returns = [np.mean(hold_bins[b]) for b in bins]
        
        if len(bins) < 3:
            return None
        
        # Exponential fit: return = a * exp(-b * t)
        try:
            log_returns = np.log(np.maximum(np.array(avg_returns), 1e-6))
            t = np.array(bins, dtype=float)
            
            # Linear regression on log returns
            A = np.vstack([t, np.ones_like(t)]).T
            slope, intercept = np.linalg.lstsq(A, log_returns, rcond=None)[0]
            
            decay_rate = -slope
            half_life = np.log(2) / decay_rate if decay_rate > 0 else float('inf')
            initial_edge = np.exp(intercept)
            final_edge = initial_edge * np.exp(-decay_rate * self.decay_window)
        except Exception:
            decay_rate = 0.0
            half_life = float('inf')
            initial_edge = np.mean(avg_returns) if avg_returns else 0.0
            final_edge = initial_edge
        
        # Regime performance
        regime_perf = self._compute_regime_performance(analyses)
        
        return AlphaDecayMetrics(
            strategy_id=strategy_id,
            avg_hold_bars=float(np.mean([a.hold_bars for a in analyses])),
            decay_half_life=half_life,
            decay_rate=decay_rate,
            initial_edge_bps=initial_edge * 100,
            final_edge_bps=final_edge * 100,
            trades_analyzed=len(analyses),
            profitable_pct=float(np.mean([a.pnl > 0 for a in analyses])),
            avg_efficiency=float(np.mean([a.efficiency for a in analyses])),
            regime_performance=regime_perf
        )
    
    def _compute_regime_performance(self, analyses: List[TradeAnalysis]) -> Dict[str, float]:
        """Compute performance by regime (would need regime data per trade)."""
        # Placeholder - would need regime at entry time
        return {}
    
    def analyze_strategy(self, strategy_id: str) -> StrategyPerformance:
        """Comprehensive strategy analysis."""
        rows = self.db.q("""
            SELECT tid, sym, side, qty, inpx, outpx, pnl, charges, strat, intime, outtime, exit_reason
            FROM trades WHERE strat LIKE ? ORDER BY tid
        """, (f"%{strategy_id}%",))
        
        if not rows:
            return StrategyPerformance(
                strategy_id=strategy_id,
                total_trades=0,
                win_rate=0.0,
                avg_pnl=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_dd=0.0,
                profit_factor=0.0,
                avg_hold_bars=0.0,
                turnover=0.0,
                cost_drag=0.0,
                alpha_decay=None,
                regime_breakdown={},
                time_of_day_breakdown={}
            )
        
        pnls = [float(r[6]) for r in rows]
        charges = [float(r[7]) for r in rows]
        win_rate = float(np.mean([p > 0 for p in pnls]))
        
        # Sharpe/Sortino
        returns = np.array(pnls) / self.cfg["capital"]
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        downside = returns[returns < 0]
        sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252)) if len(downside) > 1 and np.std(downside) > 0 else 0.0
        
        # Max drawdown
        equity = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / np.maximum(peak, 1e-9)))
        
        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Cost drag
        cost_drag = sum(charges) / sum(abs(p) for p in pnls) if sum(abs(p) for p in pnls) > 0 else 0.0
        
        # Turnover
        turnover = sum(abs(float(r[3]) * float(r[4])) for r in rows) / self.cfg["capital"]
        
        # Alpha decay
        alpha_decay = self.compute_alpha_decay(strategy_id)
        
        # Time of day breakdown
        tod_breakdown = self._compute_tod_breakdown(rows)
        
        return StrategyPerformance(
            strategy_id=strategy_id,
            total_trades=len(rows),
            win_rate=win_rate,
            avg_pnl=float(np.mean(pnls)),
            sharpe=sharpe,
            sortino=sortino,
            max_dd=max_dd,
            profit_factor=pf,
            avg_hold_bars=float(alpha_decay.avg_hold_bars) if alpha_decay else 0.0,
            turnover=turnover,
            cost_drag=cost_drag,
            alpha_decay=alpha_decay,
            regime_breakdown={},
            time_of_day_breakdown=tod_breakdown
        )
    
    def _compute_tod_breakdown(self, rows: List[tuple]) -> Dict[str, Dict]:
        """Breakdown by time of day."""
        tod_buckets = defaultdict(list)
        
        for row in rows:
            intime = row[9]
            try:
                hour = pd.Timestamp(intime).hour
                bucket = f"{hour//2*2:02d}-{(hour//2*2)+2:02d}"
                tod_buckets[bucket].append(float(row[6]))
            except Exception:
                continue
        
        result = {}
        for bucket, pnls in tod_buckets.items():
            result[bucket] = {
                "trades": len(pnls),
                "win_rate": float(np.mean([p > 0 for p in pnls])),
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls))
            }
        
        return result
    
    def get_all_strategy_decay(self) -> Dict[str, AlphaDecayMetrics]:
        """Get alpha decay for all active strategies."""
        # Get unique strategy IDs from recent trades
        rows = self.db.q("""
            SELECT DISTINCT strat FROM trades 
            WHERE tid > (SELECT MAX(tid) FROM trades) - ?
        """, (self.lookback_trades,))
        
        results = {}
        for (strat,) in rows:
            decay = self.compute_alpha_decay(strat)
            if decay:
                results[strat] = decay
        
        return results
    
    def should_retire_strategy(self, strategy_id: str) -> Tuple[bool, str]:
        """Determine if strategy should be retired based on alpha decay."""
        decay = self.compute_alpha_decay(strategy_id)
        if decay is None:
            return False, "insufficient_data"
        
        # Retire if alpha decayed significantly
        if decay.decay_rate > 0.1 and decay.final_edge_bps < decay.initial_edge_bps * 0.3:
            return True, f"alpha_decayed: rate={decay.decay_rate:.3f}, edge_retention={decay.final_edge_bps/decay.initial_edge_bps:.1%}"
        
        # Retire if efficiency too low
        if decay.avg_efficiency < 0.2 and decay.trades_analyzed > 50:
            return True, f"low_efficiency: {decay.avg_efficiency:.1%}"
        
        return False, "ok"


class AlphaDecayMonitor:
    """Monitors alpha decay in real-time and alerts."""
    
    def __init__(self, cfg, db: DB):
        self.cfg = cfg
        self.db = db
        self.analyzer = PostTradeAnalyzer(cfg, db)
        self.alert_threshold = cfg.get("alpha_decay", {}).get("alert_threshold", 0.5)
        self.check_interval = cfg.get("alpha_decay", {}).get("check_interval_hours", 4)
        self._last_check = 0.0
    
    def check_and_alert(self) -> List[dict]:
        """Check all strategies for alpha decay alerts."""
        import time
        if time.time() - self._last_check < self.check_interval * 3600:
            return []
        
        self._last_check = time.time()
        alerts = []
        
        decays = self.analyzer.get_all_strategy_decay()
        for strat_id, decay in decays.items():
            edge_retention = decay.final_edge_bps / decay.initial_edge_bps if decay.initial_edge_bps > 0 else 1.0
            
            if edge_retention < self.alert_threshold:
                alerts.append({
                    "type": "alpha_decay",
                    "strategy": strat_id,
                    "severity": "HIGH" if edge_retention < 0.2 else "MEDIUM",
                    "message": f"Alpha decayed to {edge_retention:.1%} of initial edge",
                    "decay_rate": decay.decay_rate,
                    "half_life": decay.decay_half_life,
                    "trades": decay.trades_analyzed
                })
                
                # Log to database
                self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('ALPHA_DECAY_ALERT',?,?)",
                          (f"{strat_id}: {edge_retention:.1%} retention", iso()))
        
        return alerts