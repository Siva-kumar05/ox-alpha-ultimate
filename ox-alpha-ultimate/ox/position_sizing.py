"""Advanced position sizing: Kelly, risk parity, order-flow driven, correlation-aware."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from .charges import ChargesCalculator
from .risk import CovarianceMatrix


@dataclass
class OrderFlowMetrics:
    """Order flow metrics for position sizing."""
    book_imbalance: float
    flow_imbalance: float
    pressure_ema: float
    positive_streak: int
    liquidity_score: float
    spread_bps: float
    bid_notional: float
    ask_notional: float
    microprice_edge_bps: float


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    quantity: int
    notional: float
    risk_amount: float
    kelly_fraction: float
    regime_multiplier: float
    flow_multiplier: float
    correlation_multiplier: float
    drawdown_multiplier: float
    final_fraction: float


class PositionSizer:
    def __init__(self, cfg, risk_manager):
        self.cfg = cfg
        self.risk = risk_manager
        self.calculator = ChargesCalculator(cfg.get("costs", {}))
        
        # Order flow sizing config
        flow_cfg = cfg.get("position_sizing", {}).get("order_flow", {})
        self.flow_enabled = flow_cfg.get("enabled", True)
        self.max_flow_mult = flow_cfg.get("max_multiplier", 2.0)
        self.min_flow_mult = flow_cfg.get("min_multiplier", 0.25)
        self.flow_imbalance_weight = flow_cfg.get("flow_imbalance_weight", 0.4)
        self.book_imbalance_weight = flow_cfg.get("book_imbalance_weight", 0.3)
        self.pressure_weight = flow_cfg.get("pressure_weight", 0.2)
        self.liquidity_weight = flow_cfg.get("liquidity_weight", 0.1)
        
        # Correlation sizing config
        corr_cfg = cfg.get("position_sizing", {}).get("correlation", {})
        self.corr_enabled = corr_cfg.get("enabled", True)
        self.max_corr_mult = corr_cfg.get("max_multiplier", 1.5)
        self.min_corr_mult = corr_cfg.get("min_multiplier", 0.5)
        
        # Drawdown config
        dd_cfg = cfg.get("position_sizing", {}).get("drawdown", {})
        self.dd_enabled = dd_cfg.get("enabled", True)
        self.max_dd_mult = dd_cfg.get("max_multiplier", 1.0)
        self.min_dd_mult = dd_cfg.get("min_multiplier", 0.1)
        
        # Regime config
        regime_cfg = cfg.get("position_sizing", {}).get("regime", {})
        self.regime_enabled = regime_cfg.get("enabled", True)

    def size_for_smallcap(self, price: float, stop_distance: float, capital: float, is_smallcap: bool) -> int:
        """For small-caps with limited capital: allow smaller notionals, enforce breakeven."""
        base_qty = self.risk.size_with_kelly(price, stop_distance)
        notional = base_qty * price
        min_notional_small = float(self.cfg.get("position_sizing", {}).get("min_smallcap_notional", 5000))
        if is_smallcap and base_qty > 0:
            breakeven = self.calculator.min_breakeven_sell_price(price, 1)
            if breakeven <= price * 1.02:
                return max(1, base_qty)
            if notional < min_notional_small and base_qty > 1:
                max_qty = int(capital * 0.02 / max(stop_distance, 1.0))
                return min(max_qty, max(1, int(min_notional_small / max(price, 1.0))))
        return base_qty

    def risk_parity_weights(self, symbols: list[str], volatilities: dict[str, float]) -> dict[str, float]:
        inv = {s: 1.0 / max(float(volatilities.get(s, 0.02)), 0.005) for s in symbols}
        total = sum(inv.values()) or 1.0
        return {s: round(v / total, 4) for s, v in inv.items()}

    def calculate_order_flow_multiplier(self, flow: OrderFlowMetrics) -> float:
        """Calculate position size multiplier based on order flow metrics."""
        if not self.flow_enabled:
            return 1.0
        
        # Normalize metrics to 0-1 range
        flow_imbalance_norm = np.clip((flow.flow_imbalance + 1) / 2, 0, 1)
        book_imbalance_norm = np.clip((flow.book_imbalance + 1) / 2, 0, 1)
        pressure_norm = np.clip((flow.pressure_ema + 1) / 2, 0, 1)
        liquidity_norm = np.clip(flow.liquidity_score, 0, 1)
        streak_norm = min(flow.positive_streak / 10, 1.0)
        
        # Weighted combination
        score = (
            self.flow_imbalance_weight * flow_imbalance_norm +
            self.book_imbalance_weight * book_imbalance_norm +
            self.pressure_weight * pressure_norm +
            self.liquidity_weight * liquidity_norm
        )
        
        # Bonus for persistent positive streak
        score += 0.1 * streak_norm
        
        # Penalize wide spreads
        spread_penalty = min(flow.spread_bps / 20.0, 0.5)
        score -= spread_penalty
        
        # Convert to multiplier
        multiplier = self.min_flow_mult + (self.max_flow_mult - self.min_flow_mult) * np.clip(score, 0, 1)
        return float(np.clip(multiplier, self.min_flow_mult, self.max_flow_mult))

    def calculate_correlation_multiplier(
        self,
        symbol: str,
        current_positions: Dict[str, float],
        covariance: Optional[CovarianceMatrix]
    ) -> float:
        """Calculate multiplier based on correlation with existing positions."""
        if not self.corr_enabled or covariance is None or symbol not in covariance.symbols:
            return 1.0
        
        if not current_positions:
            return 1.0
        
        idx = covariance.symbols.index(symbol)
        correlations = covariance.correlation[idx]
        
        # Weight by position size
        total_weight = 0.0
        weighted_corr = 0.0
        
        for pos_symbol, weight in current_positions.items():
            if pos_symbol in covariance.symbols:
                pos_idx = covariance.symbols.index(pos_symbol)
                weighted_corr += abs(correlations[pos_idx]) * weight
                total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        avg_corr = weighted_corr / total_weight
        
        # High correlation = reduce size
        if avg_corr > 0.7:
            multiplier = self.min_corr_mult + (1.0 - self.min_corr_mult) * (1.0 - avg_corr) / 0.3
        elif avg_corr < -0.3:
            # Negative correlation = can increase size (hedge benefit)
            multiplier = min(self.max_corr_mult, 1.0 + abs(avg_corr) * 0.5)
        else:
            multiplier = 1.0
        
        return float(np.clip(multiplier, self.min_corr_mult, self.max_corr_mult))

    def calculate_drawdown_multiplier(self, current_dd: float, max_dd: float) -> float:
        """Calculate multiplier based on current drawdown."""
        if not self.dd_enabled or max_dd <= 0:
            return 1.0
        
        dd_ratio = abs(current_dd) / max_dd
        
        if dd_ratio > 0.8:
            return self.min_dd_mult
        elif dd_ratio > 0.5:
            return self.min_dd_mult + (1.0 - self.min_dd_mult) * (0.8 - dd_ratio) / 0.3
        elif dd_ratio > 0.2:
            return 0.75 + 0.25 * (0.5 - dd_ratio) / 0.3
        else:
            return 1.0

    def calculate_regime_multiplier(self, regime: str) -> float:
        """Calculate multiplier based on market regime."""
        if not self.regime_enabled:
            return 1.0
        
        regime_multipliers = {
            "TRENDING_UP": 1.2,
            "TRENDING_DOWN": 0.7,
            "RANGING": 1.0,
            "VOLATILE": 0.5,
            "LOW_VOL_TRENDING": 1.3,
            "HIGH_VOL_MEAN_REVERTING": 0.8,
        }
        return regime_multipliers.get(regime, 1.0)

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_distance: float,
        order_flow: Optional[OrderFlowMetrics] = None,
        current_positions: Optional[Dict[str, float]] = None,
        covariance: Optional[CovarianceMatrix] = None,
        current_drawdown: float = 0.0,
        max_drawdown: float = 0.0,
        regime: str = "RANGING"
    ) -> SizingResult:
        """Calculate position size with all multipliers."""
        base_fraction = self.risk.rules["risk_per_trade_pct"] / 100.0
        
        # Kelly fraction
        kelly_stats = self.risk._kelly_stats()
        kelly_fraction = 0.0
        if kelly_stats is not None:
            kelly_fraction = self.risk.kelly_fraction(kelly_stats[0], kelly_stats[1], kelly_stats[2], cap=0.25)
        
        # Multipliers
        regime_mult = self.calculate_regime_multiplier(regime)
        flow_mult = self.calculate_order_flow_multiplier(order_flow) if order_flow else 1.0
        corr_mult = self.calculate_correlation_multiplier(symbol, current_positions or {}, covariance) if self.corr_enabled else 1.0
        dd_mult = self.calculate_drawdown_multiplier(current_drawdown, max_drawdown) if self.dd_enabled else 1.0
        
        # Combined multiplier (geometric mean) - high-leverage mode allows up to 3.5x
        leverage_cfg = self.cfg.get("leverage_engine", {})
        cap = float(leverage_cfg.get("sizer_cap_mult", 3.5))
        combined_mult = (regime_mult * flow_mult * corr_mult * dd_mult) ** 0.25
        combined_mult = np.clip(combined_mult, 0.1, cap)
        
        # Effective risk fraction - cap scales with leverage tier
        effective_fraction = base_fraction * combined_mult
        effective_fraction = min(effective_fraction, base_fraction * cap)  # Cap at sizer_cap_mult
        
        risk_amount = self.cfg["capital"] * effective_fraction
        
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in (price, stop_distance)):
            return SizingResult(0, 0.0, 0.0, kelly_fraction, regime_mult, flow_mult, corr_mult, dd_mult, combined_mult)
        
        raw_quantity = int(risk_amount / stop_distance)
        notional_quantity = int(self.risk.rules["max_notional_per_trade"] / price)
        quantity = max(0, min(raw_quantity, notional_quantity))
        notional = quantity * price
        
        return SizingResult(
            quantity=quantity,
            notional=notional,
            risk_amount=risk_amount,
            kelly_fraction=kelly_fraction,
            regime_multiplier=regime_mult,
            flow_multiplier=flow_mult,
            correlation_multiplier=corr_mult,
            drawdown_multiplier=dd_mult,
            final_fraction=combined_mult
        )

    def order_flow_to_metrics(self, flow_assessment) -> OrderFlowMetrics:
        """Convert order flow assessment to metrics."""
        return OrderFlowMetrics(
            book_imbalance=flow_assessment.book_imbalance,
            flow_imbalance=flow_assessment.flow_imbalance,
            pressure_ema=flow_assessment.pressure_ema,
            positive_streak=flow_assessment.positive_streak,
            liquidity_score=flow_assessment.liquidity_score,
            spread_bps=flow_assessment.spread_bps,
            bid_notional=flow_assessment.bid_notional,
            ask_notional=flow_assessment.ask_notional,
            microprice_edge_bps=flow_assessment.microprice_edge_bps
        )
