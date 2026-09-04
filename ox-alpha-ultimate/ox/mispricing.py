"""Mispricing detector. Flags when market price deviates > threshold from fair value.
Threshold 85%+ confidence: only high-confidence fair values generate flags.
"""
from __future__ import annotations

class MispricingDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold_bps = float(cfg.get("mispricing", {}).get("threshold_bps", 85))  # 0.85%
        self.min_confidence = float(cfg.get("mispricing", {}).get("min_confidence", 0.6))

    def check(self, last_price: float, fair_value: float, confidence: float) -> dict:
        if fair_value <= 0 or last_price <= 0 or confidence < self.min_confidence:
            return {"flagged": False, "mispricing_bps": 0.0, "direction": "NONE", "confidence": confidence}
        mispricing_bps = (last_price - fair_value) / max(abs(fair_value), 1.0) * 10000
        flagged = abs(mispricing_bps) >= self.threshold_bps and confidence >= 0.6
        direction = "OVERPRICED" if mispricing_bps > 0 else "UNDERPRICED" if mispricing_bps < 0 else "NONE"
        return {"flagged": flagged, "mispricing_bps": round(mispricing_bps, 2), "direction": direction,
                "fair_value": round(fair_value, 2), "last_price": round(last_price, 2), "confidence": confidence}
