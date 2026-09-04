"""Cost-Aware Strategy Selection and Adaptive Parameter Drift Detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from .core import iso, LOG
from .charges import ChargesCalculator
from .risk import RiskManager


@dataclass
class StrategyCostMetrics:
    """Cost metrics for a strategy."""
    strategy_id: str
    total_trades: int
    total_commissions: float
    total_slippage: float
    total_charges: float
    avg_commission_per_trade: float
    avg_slippage_bps: float
    turnover: float  # Annual turnover
    cost_drag: float  # Costs as % of gross profit
    net_profit_after_costs: float
    gross_profit: float
    cost_ratio: float  # Total costs / Gross profit


@dataclass
class CostAdjustedScore:
    """Strategy score adjusted for costs."""
    strategy_id: str
    raw_score: float
    cost_penalty: float
    adjusted_score: float
    rank: int
    cost_efficiency: float  # Net profit / Total costs


class CostAwareSelector:
    """Selects strategies based on cost-adjusted performance."""
    
    def __init__(self, cfg, risk_manager: RiskManager, db):
        self.cfg = cfg
        self.risk = risk_manager
        self.db = db
        self.calculator = ChargesCalculator(cfg.get("costs", {}))
        self.select_cfg = cfg.get("cost_aware_selection", {})
        self.enabled = self.select_cfg.get("enabled", True)
        self.cost_weight = self.select_cfg.get("cost_weight", 0.3)
        self.min_trades = self.select_cfg.get("min_trades_for_cost", 20)
        self.max_cost_drag = self.select_cfg.get("max_cost_drag", 0.5)
        self.turnover_penalty = self.select_cfg.get("turnover_penalty", 0.1)
    
    def compute_strategy_costs(self, strategy_id: str, lookback_trades: int = 200) -> Optional[StrategyCostMetrics]:
        """Compute cost metrics for a strategy."""
        rows = self.db.q("""
            SELECT qty, inpx, outpx, pnl, charges, intime, outtime
            FROM trades WHERE strat LIKE ? ORDER BY tid DESC LIMIT ?
        """, (f"%{strategy_id}%", lookback_trades))
        
        if len(rows) < self.min_trades:
            return None
        
        total_commissions = 0.0
        total_slippage = 0.0
        total_charges = 0.0
        total_pnl = 0.0
        total_notional = 0.0
        
        for qty, inpx, outpx, pnl, charges, intime, outtime in rows:
            total_charges += float(charges)
            total_pnl += float(pnl)
            
            # Estimate slippage from fill prices vs mid
            # Would need reference price at signal time
            # For now, use charges breakdown
            total_notional += float(qty) * float(inpx) + float(qty) * float(outpx)
        
        avg_commission = total_charges / len(rows) if rows else 0.0
        avg_slippage_bps = 0.0  # Would need tick data
        
        # Turnover
        first_trade = pd.Timestamp(rows[-1][5]) if rows else pd.Timestamp.now()
        last_trade = pd.Timestamp(rows[0][6]) if rows else pd.Timestamp.now()
        days = max((last_trade - first_trade).days, 1)
        annual_turnover = (total_notional / self.cfg["capital"]) * (252 / days)
        
        gross_profit = sum(float(r[3]) for r in rows if float(r[3]) > 0)
        cost_drag = total_charges / gross_profit if gross_profit > 0 else 1.0
        
        return StrategyCostMetrics(
            strategy_id=strategy_id,
            total_trades=len(rows),
            total_commissions=total_commissions,
            total_slippage=total_slippage,
            total_charges=total_charges,
            avg_commission_per_trade=avg_commission,
            avg_slippage_bps=avg_slippage_bps,
            turnover=annual_turnover,
            cost_drag=cost_drag,
            net_profit_after_costs=total_pnl - total_charges,
            gross_profit=gross_profit,
            cost_ratio=total_charges / gross_profit if gross_profit > 0 else 0.0
        )
    
    def compute_cost_adjusted_scores(
        self,
        strategy_scores: Dict[str, float]
    ) -> List[CostAdjustedScore]:
        """Compute cost-adjusted scores for strategies."""
        if not self.enabled:
            return [
                CostAdjustedScore(sid, score, 0.0, score, 0, 1.0)
                for sid, score in strategy_scores.items()
            ]
        
        adjusted = []
        
        for strategy_id, raw_score in strategy_scores.items():
            costs = self.compute_strategy_costs(strategy_id)
            
            if costs is None:
                adjusted.append(CostAdjustedScore(
                    strategy_id=strategy_id,
                    raw_score=raw_score,
                    cost_penalty=0.0,
                    adjusted_score=raw_score,
                    rank=0,
                    cost_efficiency=1.0
                ))
                continue
            
            # Cost penalty based on cost drag
            cost_penalty = 0.0
            if costs.cost_drag > self.max_cost_drag:
                cost_penalty = (costs.cost_drag - self.max_cost_drag) * self.cost_weight * 10
            
            # Turnover penalty
            if costs.turnover > 5.0:  # 500% annual turnover
                turnover_penalty = min((costs.turnover - 5.0) / 10.0, 1.0) * self.turnover_penalty
                cost_penalty += turnover_penalty
            
            # Efficiency ratio
            cost_efficiency = costs.net_profit_after_costs / costs.total_charges if costs.total_charges > 0 else 1.0
            
            adjusted_score = max(0.0, raw_score - cost_penalty)
            
            adjusted.append(CostAdjustedScore(
                strategy_id=strategy_id,
                raw_score=raw_score,
                cost_penalty=cost_penalty,
                adjusted_score=adjusted_score,
                rank=0,
                cost_efficiency=cost_efficiency
            ))
        
        # Rank by adjusted score
        adjusted.sort(key=lambda x: x.adjusted_score, reverse=True)
        for i, a in enumerate(adjusted):
            a.rank = i + 1
        
        return adjusted
    
    def select_strategies(
        self,
        strategy_scores: Dict[str, float],
        max_strategies: int = 5
    ) -> List[str]:
        """Select top strategies by cost-adjusted score."""
        adjusted = self.compute_cost_adjusted_scores(strategy_scores)
        
        # Filter out strategies with excessive cost drag
        viable = [a for a in adjusted if a.cost_penalty < 2.0]
        
        # Return top N
        return [a.strategy_id for a in viable[:max_strategies]]
    
    def get_cost_report(self, strategy_id: str) -> Optional[Dict]:
        """Get detailed cost report for a strategy."""
        costs = self.compute_strategy_costs(strategy_id)
        if costs is None:
            return None
        
        return {
            "strategy_id": costs.strategy_id,
            "total_trades": costs.total_trades,
            "total_charges": round(costs.total_charges, 2),
            "avg_commission": round(costs.avg_commission_per_trade, 2),
            "avg_slippage_bps": round(costs.avg_slippage_bps, 2),
            "annual_turnover": round(costs.turnover, 2),
            "cost_drag_pct": round(costs.cost_drag * 100, 2),
            "gross_profit": round(costs.gross_profit, 2),
            "net_profit": round(costs.net_profit_after_costs, 2),
            "cost_efficiency": round(costs.gross_profit / costs.total_charges, 2) if costs.total_charges > 0 else 0
        }


@dataclass
class ParameterDriftSignal:
    """Signal indicating parameter drift."""
    strategy_id: str
    parameter: str
    original_value: float
    current_value: float
    drift_magnitude: float  # 0-1
    drift_direction: str  # "increasing", "decreasing", "oscillating"
    statistical_significance: float  # p-value
    recommendation: str  # "retrain", "monitor", "revert"


class ParameterDriftDetector:
    """Detects parameter drift in live strategies."""
    
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.drift_cfg = cfg.get("parameter_drift", {})
        self.enabled = self.drift_cfg.get("enabled", True)
        self.lookback_trades = self.drift_cfg.get("lookback_trades", 100)
        self.drift_threshold = self.drift_cfg.get("drift_threshold", 0.15)  # 15% change
        self.min_significance = self.drift_cfg.get("min_significance", 0.05)
        self.check_interval = self.drift_cfg.get("check_interval_hours", 6)
        
        # Parameter history per strategy
        self._param_history: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
        self._last_check = 0.0
    
    def record_parameters(self, strategy_id: str, params: Dict):
        """Record strategy parameters at a point in time."""
        timestamp = iso()
        for param, value in params.items():
            if isinstance(value, (int, float)):
                self._param_history[strategy_id][param].append((timestamp, float(value)))
        
        # Trim history
        for param in self._param_history[strategy_id]:
            if len(self._param_history[strategy_id][param]) > self.lookback_trades * 2:
                self._param_history[strategy_id][param] = \
                    self._param_history[strategy_id][param][-self.lookback_trades:]
    
    def detect_drift(self, strategy_id: str) -> List[ParameterDriftSignal]:
        """Detect parameter drift for a strategy."""
        if not self.enabled:
            return []
        
        signals = []
        history = self._param_history.get(strategy_id, {})
        
        for param, values in history.items():
            if len(values) < 20:  # Need minimum history
                continue
            
            param_values = np.array([v[1] for v in values])
            
            # Skip if constant
            if np.std(param_values) < 1e-6:
                continue
            
            # Split into two halves for comparison
            mid = len(param_values) // 2
            first_half = param_values[:mid]
            second_half = param_values[mid:]
            
            # Statistical test (Mann-Whitney U test for non-parametric)
            try:
                from scipy.stats import mannwhitneyu
                stat, p_value = mannwhitneyu(first_half, second_half, alternative='two-sided')
            except ImportError:
                # Fallback: simple mean comparison
                mean_diff = abs(np.mean(second_half) - np.mean(first_half))
                rel_diff = mean_diff / max(abs(np.mean(first_half)), 1e-9)
                p_value = 0.05 if rel_diff > self.drift_threshold else 0.5
            
            # Drift magnitude
            drift_magnitude = abs(np.mean(second_half) - np.mean(first_half)) / max(abs(np.mean(first_half)), 1e-9)
            
            # Direction
            if np.mean(second_half) > np.mean(first_half):
                direction = "increasing"
            elif np.mean(second_half) < np.mean(first_half):
                direction = "decreasing"
            else:
                direction = "stable"
            
            # Check for oscillation (high variance in recent)
            recent_var = np.var(param_values[-10:])
            historical_var = np.var(param_values[:-10])
            if recent_var > historical_var * 2:
                direction = "oscillating"
            
            if p_value < self.min_significance and drift_magnitude > self.drift_threshold:
                # Get original value (from strategy definition)
                original = self._get_original_param(strategy_id, param)
                
                signals.append(ParameterDriftSignal(
                    strategy_id=strategy_id,
                    parameter=param,
                    original_value=original,
                    current_value=float(param_values[-1]),
                    drift_magnitude=float(drift_magnitude),
                    drift_direction=direction,
                    statistical_significance=float(p_value),
                    recommendation="retrain" if drift_magnitude > 0.3 else "monitor"
                ))
                
                LOG.warning(
                    f"Parameter drift detected: {strategy_id}.{param} "
                    f"drift={drift_magnitude:.1%} p={p_value:.4f} dir={direction}"
                )
        
        return signals
    
    def _get_original_param(self, strategy_id: str, param: str) -> float:
        """Get original parameter value from database."""
        rows = self.db.q("SELECT json FROM strategies WHERE sid=?", (strategy_id,))
        if rows:
            import json
            data = json.loads(rows[0][0])
            return float(data.get("params", {}).get(param, 0))
        return 0.0
    
    def should_retrain(self, strategy_id: str) -> Tuple[bool, List[ParameterDriftSignal]]:
        """Check if strategy should be retrained due to drift."""
        signals = self.detect_drift(strategy_id)
        
        retrain_signals = [s for s in signals if s.recommendation == "retrain"]
        return len(retrain_signals) > 0, signals
    
    def get_drift_report(self, strategy_id: str) -> Dict:
        """Get comprehensive drift report."""
        signals = self.detect_drift(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "signals": [
                {
                    "parameter": s.parameter,
                    "original": s.original_value,
                    "current": s.current_value,
                    "drift_pct": round(s.drift_magnitude * 100, 1),
                    "direction": s.drift_direction,
                    "p_value": round(s.statistical_significance, 4),
                    "recommendation": s.recommendation
                }
                for s in signals
            ],
            "max_drift": max([s.drift_magnitude for s in signals], default=0.0),
            "needs_retrain": any(s.recommendation == "retrain" for s in signals)
        }


class LivePerformanceMonitor:
    """Monitors live strategy performance for degradation."""
    
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.monitor_cfg = cfg.get("live_performance_monitor", {})
        self.enabled = self.monitor_cfg.get("enabled", True)
        self.window_trades = self.monitor_cfg.get("window_trades", 50)
        self.degradation_threshold = self.monitor_cfg.get("degradation_threshold", -0.1)
        self.min_trades = self.monitor_cfg.get("min_trades", 20)
    
    def check_strategy_health(self, strategy_id: str) -> Dict:
        """Check live health of a strategy."""
        rows = self.db.q("""
            SELECT pnl, intime FROM trades 
            WHERE strat LIKE ? ORDER BY tid DESC LIMIT ?
        """, (f"%{strategy_id}%", self.window_trades))
        
        if len(rows) < self.min_trades:
            return {"status": "insufficient_data", "trades": len(rows)}
        
        pnls = [float(r[0]) for r in rows]
        
        # Recent performance
        recent_pnl = np.mean(pnls[:20]) if len(pnls) >= 20 else np.mean(pnls)
        older_pnl = np.mean(pnls[20:40]) if len(pnls) >= 40 else np.mean(pnls)
        
        # Win rate
        recent_wr = np.mean([p > 0 for p in pnls[:20]]) if len(pnls) >= 20 else np.mean([p > 0 for p in pnls])
        older_wr = np.mean([p > 0 for p in pnls[20:40]]) if len(pnls) >= 40 else recent_wr
        
        # Degradation
        pnl_degradation = (recent_pnl - older_pnl) / max(abs(older_pnl), 1e-6) if older_pnl != 0 else 0
        wr_degradation = recent_wr - older_wr
        
        is_degraded = pnl_degradation < self.degradation_threshold or wr_degradation < -0.15
        
        return {
            "status": "degraded" if is_degraded else "healthy",
            "trades_analyzed": len(pnls),
            "recent_avg_pnl": round(recent_pnl, 2),
            "older_avg_pnl": round(older_pnl, 2),
            "pnl_degradation_pct": round(pnl_degradation * 100, 1),
            "recent_win_rate": round(recent_wr * 100, 1),
            "older_win_rate": round(older_wr * 100, 1),
            "wr_degradation_pct": round(wr_degradation * 100, 1),
            "alert": is_degraded
        }
    
    def check_all_strategies(self) -> List[Dict]:
        """Check health of all active strategies."""
        # Get unique strategy IDs from recent trades
        rows = self.db.q("""
            SELECT DISTINCT strat FROM trades 
            WHERE tid > (SELECT MAX(tid) FROM trades) - ?
        """, (self.window_trades * 3,))
        
        results = []
        for (strat,) in rows:
            health = self.check_strategy_health(strat)
            health["strategy_id"] = strat
            results.append(health)
        
        return results