"""Dynamic leverage engine - volatility-targeted, confidence-scaled leverage for small capital."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class LeverageDecision:
    leverage: float
    tier: str
    base_leverage: float
    vol_scalar: float
    confidence_scalar: float
    regime_scalar: float
    dd_scalar: float
    liquidation_price: float
    buffer_pct: float
    reason: str

class LeverageEngine:
    """Calculates safe dynamic leverage. Hard cap 10x equity/crypto (venues reject 20x for retail-size accounts)."""
    TIER_CAPS = {
        "equity_intraday": 5.0,
        "equity_scalp": 7.0,
        "crypto_major": 10.0,
        "crypto_alt": 10.0,
        "crypto_meme": 5.0,
        "crypto_scalp": 10.0,  # venue-realistic cap for retail-size accounts
        "options": 7.0,
    }

    def __init__(self, cfg):
        lev = cfg.get("leverage_engine", {})
        self.enabled = lev.get("enabled", True)
        self.target_vol_pct = float(lev.get("target_vol_pct", 0.8)) # 0.8% daily move target
        self.min_leverage = float(lev.get("min_leverage", 1.0))
        self.max_leverage_global = float(lev.get("max_leverage_global", 10.0))
        self.max_crypto_scalp = float(lev.get("max_crypto_scalp", 20.0))
        self.liquidation_buffer = float(lev.get("liquidation_buffer", 0.4)) # keep 40% away from liq
        self.confidence_weight = float(lev.get("confidence_weight", 0.6))

    def decide(
        self,
        symbol: str,
        tier: str,
        price: float,
        atr_pct: float,
        confidence: float, # 0-1 from vote/regime/flow
        regime: str,
        dd_ratio: float,
        hold_minutes: int,
    ) -> LeverageDecision:
        cap = self.TIER_CAPS.get(tier, 3.0)
        cap = min(cap, self.max_leverage_global if tier != "crypto_scalp" else self.max_crypto_scalp)

        # 1. Vol targeting: lev = target_vol / atr_pct
        if atr_pct and atr_pct > 0:
            vol_scalar = float(np.clip(self.target_vol_pct / max(atr_pct, 0.2), 0.4, 1.8))
        else:
            vol_scalar = 1.0

        # 2. Confidence 0.5-1.0 maps to 0.6-1.4
        confidence_scalar = 0.6 + 0.8 * float(np.clip(confidence, 0, 1))

        # 3. Regime
        regime_map = {"TRENDING_UP":1.15,"LOW_VOL_TRENDING":1.25,"RANGING":0.9,"VOLATILE":0.55,"HIGH_VOL_MEAN_REVERTING":0.7,"TRENDING_DOWN":0.75}
        regime_scalar = regime_map.get(regime, 1.0)

        # 4. DD: 0 at 0% DD, 0.3 at 80% max DD
        dd_scalar = float(np.clip(1.0 - 0.7*dd_ratio, 0.3, 1.0))

        # 5. Hold time: scalp <15m can use higher lev
        hold_scalar = 1.25 if tier=="crypto_scalp" and hold_minutes<=15 else 1.0

        base = 3.0 if "equity" in tier else 6.0
        raw = base * vol_scalar * confidence_scalar * regime_scalar * dd_scalar * hold_scalar
        leverage = float(np.clip(raw, self.min_leverage, cap))

        # Liquidation estimate (linear perp approx)
        if leverage > 0:
            liq_dist = (0.9 / leverage) # ~90% of 1/lev
            buffer_price = price * (1 - liq_dist * (1 - self.liquidation_buffer)) if leverage else price
        else:
            liq_dist, buffer_price = 0, price

        return LeverageDecision(
            leverage=round(leverage,2),
            tier=tier,
            base_leverage=base,
            vol_scalar=round(vol_scalar,2),
            confidence_scalar=round(confidence_scalar,2),
            regime_scalar=round(regime_scalar,2),
            dd_scalar=round(dd_scalar,2),
            liquidation_price=round(buffer_price,2),
            buffer_pct=self.liquidation_buffer,
            reason=f"{tier} vol{vol_scalar:.2f} conf{confidence_scalar:.2f} regime{regime_scalar:.2f} dd{dd_scalar:.2f}",
        )

    def tier_for(self, symbol: str, venue: str) -> str:
        s = symbol.upper()
        if venue=="crypto":
            if s in ("BTCUSDT","ETHUSDT","SOLUSDT"): return "crypto_major"
            if s.endswith("USDT") and any(k in s for k in ("PEPE","WIF","BONK","DOGE","SHIB")): return "crypto_meme"
            if "USDT" in s: return "crypto_alt"
            return "crypto_alt"
        if "scalper" in venue: return "equity_scalp"
        return "equity_intraday"
