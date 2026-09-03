"""Market regime detection for adaptive trading."""
from __future__ import annotations
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .core import LOG, iso
from .indicators import ind

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    LOW_VOL_TRENDING = "LOW_VOL_TRENDING"
    HIGH_VOL_MEAN_REVERTING = "HIGH_VOL_MEAN_REVERTING"

@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    volatility_percentile: float
    trend_strength: float
    regime_duration: int
    ts: str
    hurst_exponent: float = 0.5
    mean_reversion_strength: float = 0.0
    volatility_regime: str = "NORMAL"

@dataclass
class StrategyWeights:
    """Strategy weights for each regime."""
    weights: Dict[str, float]
    max_positions: int
    risk_multiplier: float
    preferred_timeframe: int
    stop_type: str

class RegimeDetector:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        rcfg = cfg.get("regime", {})
        self.lookback = int(rcfg.get("lookback_candles", 500))
        self.vol_high = float(rcfg.get("vol_high_percentile", 80))
        self.vol_low = float(rcfg.get("vol_low_percentile", 20))
        self.trend_thresh = float(rcfg.get("trend_adx_threshold", 25))
        self.hurst_threshold = float(rcfg.get("hurst_threshold", 0.5))
        self._history = []
        self._regime_count = 0
        self._prev_regime = None
        self._regime_changes = 0
        
        # Strategy configuration per regime
        self.regime_strategies = {
            MarketRegime.TRENDING_UP: StrategyWeights(
                weights={"breakout": 1.5, "ma_crossover": 1.3, "donchian_breakout": 1.4, "core": 1.0, "scalp": 0.5, "pullback_to_ema": 1.2},
                max_positions=5,
                risk_multiplier=1.2,
                preferred_timeframe=15,
                stop_type="trailing"
            ),
            MarketRegime.TRENDING_DOWN: StrategyWeights(
                weights={"core": 0.8, "scalp": 0.7, "rsi2_bounce": 0.5, "mean_reversion": 0.6},
                max_positions=3,
                risk_multiplier=0.7,
                preferred_timeframe=5,
                stop_type="tight"
            ),
            MarketRegime.RANGING: StrategyWeights(
                weights={"core": 1.3, "scalp": 1.0, "bb_fade": 1.4, "rsi2_bounce": 1.3, "mean_reversion": 1.5, "keltner_mr": 1.2},
                max_positions=4,
                risk_multiplier=1.0,
                preferred_timeframe=5,
                stop_type="mean_reversion"
            ),
            MarketRegime.VOLATILE: StrategyWeights(
                weights={"scalp": 1.4, "breakout": 0.5, "core": 0.6, "volatility_squeeze_breakout": 1.2},
                max_positions=2,
                risk_multiplier=0.5,
                preferred_timeframe=1,
                stop_type="wide"
            ),
            MarketRegime.LOW_VOL_TRENDING: StrategyWeights(
                weights={"breakout": 1.6, "ma_crossover": 1.4, "core": 1.1, "pullback_to_ema": 1.3},
                max_positions=5,
                risk_multiplier=1.3,
                preferred_timeframe=15,
                stop_type="trailing"
            ),
            MarketRegime.HIGH_VOL_MEAN_REVERTING: StrategyWeights(
                weights={"bb_fade": 1.5, "rsi2_bounce": 1.4, "mean_reversion": 1.6, "keltner_mr": 1.3, "scalp": 1.1},
                max_positions=3,
                risk_multiplier=0.8,
                preferred_timeframe=5,
                stop_type="mean_reversion"
            ),
        }

    def detect(self, df):
        if df is None or len(df) < 50:
            return RegimeState(MarketRegime.RANGING, 0.5, 50.0, 0.0, 0, iso())

        h = df["h"].to_numpy(dtype=float)
        l = df["l"].to_numpy(dtype=float)
        c = df["c"].to_numpy(dtype=float)
        v = df["v"].to_numpy(dtype=float)

        # Parkinson volatility
        log_hl = np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9))
        park = np.sqrt(np.mean(log_hl[-20:] ** 2) / (4 * np.log(2)))

        # Rolling vol percentile
        all_park = []
        for i in range(min(200, len(h) - 20), len(h)):
            seg = np.log(np.maximum(h[i-20:i], 1e-9) / np.maximum(l[i-20:i], 1e-9))
            all_park.append(np.sqrt(np.mean(seg ** 2) / (4 * np.log(2))))
        vol_pct = float(np.searchsorted(sorted(all_park + [park]), park) / max(len(all_park) + 1, 1) * 100)

        # EMA slope
        ema20 = pd.Series(c).ewm(span=20).mean().values
        ema_slope = (ema20[-1] - ema20[min(5, len(ema20)-1)]) / max(ema20[min(5, len(ema20)-1)], 1e-9) * 100

        # ADX
        plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0)
        minus_dm = np.maximum(-np.diff(l, prepend=l[0]), 0)
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
        plus_di = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / np.maximum(atr14, 1e-9) * 100
        minus_di = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / np.maximum(atr14, 1e-9) * 100
        dx = np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9) * 100
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values[-1]

        # Hurst exponent for trend vs mean reversion
        try:
            hurst = ind(84)(c, max_lag=20)[-1]
            if np.isnan(hurst):
                hurst = 0.5
        except Exception:
            hurst = 0.5

        # Mean reversion strength (z-score based)
        try:
            zscore_vals = ind(85)(c, n=20)
            mean_rev_strength = float(np.abs(zscore_vals[-1])) if not np.isnan(zscore_vals[-1]) else 0.0
        except Exception:
            mean_rev_strength = 0.0

        # Volatility regime
        if vol_pct > self.vol_high:
            vol_regime = "HIGH"
        elif vol_pct < self.vol_low:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"

        is_high_vol = vol_pct > self.vol_high
        is_low_vol = vol_pct < self.vol_low
        is_trending = adx > self.trend_thresh
        is_hurst_trending = hurst > self.hurst_threshold

        # Enhanced regime classification
        if is_high_vol and is_hurst_trending:
            regime = MarketRegime.HIGH_VOL_MEAN_REVERTING
            conf = min(1.0, (vol_pct - self.vol_high) / 20 + 0.5)
        elif is_high_vol:
            regime = MarketRegime.VOLATILE
            conf = min(1.0, (vol_pct - self.vol_high) / 20 + 0.5)
        elif is_low_vol and is_hurst_trending and ema_slope > 0:
            regime = MarketRegime.LOW_VOL_TRENDING
            conf = min(1.0, (self.vol_low - vol_pct) / 20 + 0.5)
        elif is_trending and ema_slope > 0:
            regime = MarketRegime.TRENDING_UP
            conf = min(1.0, adx / 50)
        elif is_trending and ema_slope < 0:
            regime = MarketRegime.TRENDING_DOWN
            conf = min(1.0, adx / 50)
        else:
            regime = MarketRegime.RANGING
            conf = min(1.0, 1.0 - adx / 50)

        if self._history and self._history[-1].regime == regime:
            self._regime_count += 1
        else:
            self._regime_count = 1
            if self._prev_regime is not None and self._prev_regime != regime:
                self._regime_changes += 1
            self._prev_regime = regime

        state = RegimeState(
            regime=regime,
            confidence=round(conf, 4),
            volatility_percentile=round(vol_pct, 2),
            trend_strength=round(float(adx), 2),
            regime_duration=self._regime_count,
            ts=iso(),
            hurst_exponent=round(hurst, 4),
            mean_reversion_strength=round(mean_rev_strength, 4),
            volatility_regime=vol_regime
        )
        self._history.append(state)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        return state

    def regime_weights(self):
        current = self._history[-1].regime if self._history else MarketRegime.RANGING
        return self.regime_strategies.get(current, self.regime_strategies[MarketRegime.RANGING]).weights

    def get_strategy_config(self) -> StrategyWeights:
        """Get full strategy configuration for current regime."""
        current = self._history[-1].regime if self._history else MarketRegime.RANGING
        return self.regime_strategies.get(current, self.regime_strategies[MarketRegime.RANGING])

    def should_switch_regime(self) -> bool:
        """Check if regime has changed recently."""
        if len(self._history) < 2:
            return False
        return self._history[-1].regime != self._history[-2].regime

    def get_regime_persistence(self) -> float:
        """Get regime persistence (0-1, higher = more stable)."""
        if len(self._history) < 10:
            return 0.5
        recent = self._history[-10:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i].regime != recent[i-1].regime)
        return 1.0 - (changes / 9.0)

    def get_regime_transition_prob(self, from_regime: MarketRegime, to_regime: MarketRegime) -> float:
        """Estimate transition probability from historical data."""
        if len(self._history) < 20:
            return 0.25  # Uniform prior
        transitions = 0
        from_count = 0
        for i in range(1, len(self._history)):
            if self._history[i-1].regime == from_regime:
                from_count += 1
                if self._history[i].regime == to_regime:
                    transitions += 1
        return transitions / from_count if from_count > 0 else 0.25