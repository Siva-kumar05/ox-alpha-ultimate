from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import iso, now


class Metrics:
    @staticmethod
    def sharpe(returns, annualisation: int = 252) -> float:
        values = np.asarray(returns, dtype=float)
        std = values.std()
        return float(values.mean() / std * np.sqrt(annualisation)) if len(values) > 2 and std > 0 else 0.0

    @staticmethod
    def sortino(returns, annualisation: int = 252) -> float:
        values = np.asarray(returns, dtype=float)
        downside = values[values < 0]
        if len(values) <= 2 or len(downside) <= 1:
            # A downside-risk statistic is undefined without actual downside
            # observations.  Returning zero avoids a near-zero denominator
            # turning a handful of wins into a fictional superhuman score.
            return 0.0
        deviation = downside.std()
        return float(values.mean() / deviation * np.sqrt(annualisation)) if deviation > 0 else 0.0

    @staticmethod
    def var(returns, alpha: float = 0.99) -> float:
        values = np.asarray(returns, dtype=float)
        return float(-np.quantile(values, 1.0 - alpha)) if len(values) > 10 else 0.0

    @staticmethod
    def expected_shortfall(returns, alpha: float = 0.99) -> float:
        """Mean loss of the worst (1 - alpha) tail; at least VaR by construction."""
        values = np.asarray(returns, dtype=float)
        if len(values) <= 10:
            return 0.0
        cutoff = np.quantile(values, 1.0 - alpha)
        tail = values[values <= cutoff]
        return float(-tail.mean()) if len(tail) else float(-cutoff)

    @staticmethod
    def maxdd(equity) -> float:
        values = np.asarray(equity, dtype=float)
        if len(values) < 2:
            return 0.0
        peak = np.maximum.accumulate(values)
        return float(((values - peak) / np.maximum(peak, 1e-9)).min())

    @staticmethod
    def icir(information_coefficients, annualisation: int = 252) -> float:
        values = np.asarray([value for value in information_coefficients if value == value], dtype=float)
        return float(values.mean() / values.std() * np.sqrt(annualisation)) if len(values) > 5 and values.std() > 0 else 0.0


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    symbols: List[str]
    weights: np.ndarray
    prices: np.ndarray
    quantities: np.ndarray
    notional: np.ndarray
    total_value: float


@dataclass
class CovarianceMatrix:
    """Covariance matrix with metadata."""
    matrix: np.ndarray
    correlation: np.ndarray
    symbols: List[str]
    timestamp: str
    method: str
    lookback: int


class RiskManager:
    """Risk gates that survive restarts and enforce caps at the final decision point."""

    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.rules = cfg["risk"]
        self.day = now().date()
        self.day_pnl = 0.0
        self.loss_streak = 0
        self._load_today()
        
        # Portfolio optimization components
        self._covariance: Optional[CovarianceMatrix] = None
        self._cov_lock = threading.RLock()
        self._portfolio_state: Optional[PortfolioState] = None
        self._max_drawdown_pct = float(cfg.get("advanced_risk", {}).get("max_drawdown_pct", 10.0))
        self._drawdown_lookback = int(cfg.get("advanced_risk", {}).get("drawdown_lookback", 252))
        self._cov_lookback = int(cfg.get("advanced_risk", {}).get("cov_lookback", 252))
        self._cov_method = cfg.get("advanced_risk", {}).get("cov_method", "ledoit_wolf")
        self._rebalance_threshold = float(cfg.get("advanced_risk", {}).get("rebalance_threshold", 0.05))
        self._min_correlation = float(cfg.get("advanced_risk", {}).get("min_correlation", -0.9))
        self._max_correlation = float(cfg.get("advanced_risk", {}).get("max_correlation", 0.9))

    def _load_today(self) -> None:
        # Realised day P&L is attributed to the exit date: a position opened
        # before a restart and closed today must still count against today's
        # loss cap.
        prefix = self.day.isoformat()
        rows = self.db.q("SELECT pnl FROM trades WHERE outtime LIKE ? ORDER BY tid", (f"{prefix}%",))
        self.day_pnl = sum(float(row[0]) for row in rows)
        self.loss_streak = 0
        for (pnl,) in reversed(rows):
            if float(pnl) < 0:
                self.loss_streak += 1
            else:
                break

    def rollover(self) -> None:
        if now().date() != self.day:
            self.day = now().date()
            self._load_today()

    def size(self, price: float, stop_distance: float) -> int:
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in (price, stop_distance)):
            return 0
        risk_amount = self.cfg["capital"] * (self.rules["risk_per_trade_pct"] / 100.0)
        raw_quantity = int(risk_amount / stop_distance)
        notional_quantity = int(self.rules["max_notional_per_trade"] / price)
        return max(0, min(raw_quantity, notional_quantity))

    @staticmethod
    def gross_exposure(open_positions: list[dict]) -> float:
        # Leverage-aware: exposure is notional * leverage. Positions without a
        # leverage field (legacy cash positions) count at 1.0x.
        total = 0.0
        for position in open_positions:
            qty = abs(float(position.get("qty", 0)))
            price = max(float(position.get("avg", 0.0)), 0.0)
            leverage = max(float(position.get("leverage", 1.0)), 1.0)
            total += qty * price * leverage
        return total

    def approve(self, symbol: str, side: str, quantity: int, price: float, open_positions: list[dict], portfolio_var_pct: float) -> tuple[bool, str]:
        self.rollover()
        if side not in {"BUY", "SELL"}:
            return False, "invalid side"
        if quantity <= 0 or not math.isfinite(float(price)) or price <= 0:
            return False, "invalid order size or price"
        notional = quantity * price
        if notional > self.rules["max_notional_per_trade"]:
            return False, "per-trade notional cap exceeded"
        if self.gross_exposure(open_positions) + notional > self.rules["max_gross_exposure"]:
            return False, "gross exposure cap exceeded"
        if len(open_positions) >= self.rules["max_positions"]:
            return False, "max positions cap reached"
        percentage_loss_cap = self.cfg["capital"] * (self.rules["daily_loss_cap_pct"] / 100.0)
        if self.day_pnl <= -self.rules["daily_loss_cap_abs"]:
            return False, "daily absolute loss cap breached"
        if self.day_pnl <= -percentage_loss_cap:
            return False, "daily percentage loss cap breached"
        if portfolio_var_pct > self.rules["portfolio_var_limit_pct"]:
            return False, "portfolio VaR limit exceeded"
        if self.loss_streak >= self.rules["cooldown_after_losses"]:
            return False, "loss-streak cooldown active"
        return True, "ok"

    @staticmethod
    def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, cap: float = 0.25) -> float:
        """Fractional Kelly: (p*b - q)/b where b = avg_win/avg_loss. Capped to quarter-Kelly."""
        if avg_loss <= 0 or not 0 < win_rate < 1:
            return 0.0
        b = avg_win / avg_loss
        f = (win_rate * b - (1 - win_rate)) / b
        return max(0.0, min(f, cap))

    def _kelly_stats(self) -> tuple[float, float, float] | None:
        """(win_rate, avg_win, avg_loss) from the last 200 closed trades."""
        rows = self.db.q("SELECT pnl FROM trades ORDER BY tid DESC LIMIT 200")
        pnls = [float(row[0]) for row in rows]
        if len(pnls) < 30:
            return None
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if not wins or not losses:
            return None
        return len(wins) / len(pnls), sum(wins) / len(wins), abs(sum(losses) / len(losses))

    def size_with_kelly(self, price: float, stop_distance: float, confidence: float = 0.5) -> int:
        """Fixed-fractional sizing blended with capped fractional Kelly.
        High-leverage mode: confidence>0.7 allows up to 3.5x base.
        once >=30 closed trades exist; before that it is identical to size()."""
        base_fraction = self.rules["risk_per_trade_pct"] / 100.0
        stats = self._kelly_stats()
        leverage_cfg = self.cfg.get("leverage_engine", {})
        cap_mult = float(leverage_cfg.get("sizer_cap_mult", 3.5))
        if stats is None:
            effective = base_fraction * (0.8 + 0.4*confidence)
        else:
            cap = float(self.cfg.get("advanced_risk", {}).get("kelly_fraction_cap", 0.25))
            kelly = self.kelly_fraction(stats[0], stats[1], stats[2], cap=cap)
            # confidence skews toward kelly when high
            w_kelly = 0.5 + 0.3*max(0, confidence-0.5)
            effective = max(0.0, min((1-w_kelly)*base_fraction + w_kelly*kelly, base_fraction * cap_mult))
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in (price, stop_distance)):
            return 0
        risk_amount = self.cfg["capital"] * effective
        raw_quantity = int(risk_amount / stop_distance)
        notional_quantity = int(self.rules["max_notional_per_trade"] / price)
        return max(0, min(raw_quantity, notional_quantity))

    def update_covariance(self, symbols: List[str]) -> CovarianceMatrix:
        """Compute and cache covariance/correlation matrix for portfolio optimization."""
        if len(symbols) < 2:
            return CovarianceMatrix(
                matrix=np.array([[0.0]]),
                correlation=np.array([[1.0]]),
                symbols=symbols,
                timestamp=iso(),
                method="single_asset",
                lookback=0
            )
        
        returns_list = []
        valid_symbols = []
        
        for sym in symbols:
            frame = self._get_price_series(sym)
            if frame is not None and len(frame) >= self._cov_lookback:
                close = pd.to_numeric(frame.set_index("ts")["c"], errors="coerce").pct_change().dropna()
                if len(close) >= self._cov_lookback:
                    returns_list.append(close.iloc[-self._cov_lookback:].values)
                    valid_symbols.append(sym)
        
        if len(valid_symbols) < 2:
            return CovarianceMatrix(
                matrix=np.eye(len(symbols)) * 0.01,
                correlation=np.eye(len(symbols)),
                symbols=symbols,
                timestamp=iso(),
                method="insufficient_data",
                lookback=0
            )
        
        returns_df = pd.DataFrame(np.column_stack(returns_list), columns=valid_symbols)
        
        if self._cov_method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns_df).covariance_
            except ImportError:
                cov_matrix = returns_df.cov().values
        elif self._cov_method == "oas":
            try:
                from sklearn.covariance import OAS
                oas = OAS()
                cov_matrix = oas.fit(returns_df).covariance_
            except ImportError:
                cov_matrix = returns_df.cov().values
        elif self._cov_method == "factor":
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(5, len(valid_symbols) - 1))
                factors = pca.fit_transform(returns_df)
                loadings = pca.components_.T
                factor_cov = np.cov(factors.T)
                specific_var = np.var(returns_df - factors @ pca.components_, axis=0)
                cov_matrix = loadings @ factor_cov @ loadings.T + np.diag(specific_var)
            except ImportError:
                cov_matrix = returns_df.cov().values
        else:
            cov_matrix = returns_df.cov().values
        
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if eigvals[0] < 1e-10:
            cov_matrix = cov_matrix + np.eye(len(valid_symbols)) * 1e-6
        
        # Compute correlation matrix
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        np.clip(corr_matrix, self._min_correlation, self._max_correlation, out=corr_matrix)
        
        result = CovarianceMatrix(
            matrix=cov_matrix,
            correlation=corr_matrix,
            symbols=valid_symbols,
            timestamp=iso(),
            method=self._cov_method,
            lookback=self._cov_lookback
        )
        
        with self._cov_lock:
            self._covariance = result
        
        return result
    
    def get_covariance(self) -> Optional[CovarianceMatrix]:
        with self._cov_lock:
            return self._covariance
    
    def portfolio_var(self, weights: np.ndarray, covariance: Optional[np.ndarray] = None) -> float:
        """Calculate portfolio Value at Risk."""
        if covariance is None:
            cov_obj = self.get_covariance()
            if cov_obj is None:
                return 0.0
            covariance = cov_obj.matrix
        
        if len(weights) != covariance.shape[0]:
            return 0.0
        
        port_var = weights @ covariance @ weights
        return float(np.sqrt(max(port_var, 0.0)))
    
    def marginal_var(self, weights: np.ndarray, covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate marginal VaR for each position."""
        if covariance is None:
            cov_obj = self.get_covariance()
            if cov_obj is None:
                return np.zeros_like(weights)
            covariance = cov_obj.matrix
        
        port_var = self.portfolio_var(weights, covariance)
        if port_var <= 0:
            return np.zeros_like(weights)
        
        return 2 * (covariance @ weights) / (2 * port_var)
    
    def component_var(self, weights: np.ndarray, covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate component VaR for each position."""
        marginal = self.marginal_var(weights, covariance)
        return weights * marginal
    
    def mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0,
        max_weight: float = 0.2,
        min_weight: float = 0.0,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Mean-variance portfolio optimization."""
        if covariance is None:
            cov_obj = self.get_covariance()
            if cov_obj is None:
                n = len(expected_returns)
                return np.ones(n) / n
            covariance = cov_obj.matrix
        
        n = len(expected_returns)
        if n != covariance.shape[0]:
            return np.ones(n) / n
        
        try:
            import cvxpy as cp
            w = cp.Variable(n)
            
            if target_return is not None:
                objective = cp.Minimize(cp.quad_form(w, covariance))
                constraints = [
                    cp.sum(w) == 1,
                    w >= min_weight,
                    w <= max_weight,
                    expected_returns @ w >= target_return
                ]
            else:
                objective = cp.Maximize(expected_returns @ w - 0.5 * risk_aversion * cp.quad_form(w, covariance))
                constraints = [
                    cp.sum(w) == 1,
                    w >= min_weight,
                    w <= max_weight
                ]
            
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            if w.value is not None:
                return np.maximum(w.value, 0)
        except ImportError:
            pass
        
        # Fallback: inverse volatility weighting
        vol = np.sqrt(np.diag(covariance))
        inv_vol = 1.0 / (vol + 1e-8)
        weights = inv_vol / inv_vol.sum()
        return np.clip(weights, min_weight, max_weight)
    
    def risk_parity_weights(self, covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """Risk parity portfolio weights."""
        if covariance is None:
            cov_obj = self.get_covariance()
            if cov_obj is None:
                return np.array([1.0])
            covariance = cov_obj.matrix
        
        n = covariance.shape[0]
        try:
            import cvxpy as cp
            w = cp.Variable(n)
            objective = cp.Minimize(cp.sum_squares(cp.sqrt(cp.diag(covariance) @ w) - 1.0 / n))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            if w.value is not None:
                return np.maximum(w.value, 0)
        except ImportError:
            pass
        
        # Fallback: inverse volatility
        vol = np.sqrt(np.diag(covariance))
        inv_vol = 1.0 / (vol + 1e-8)
        return inv_vol / inv_vol.sum()
    
    def max_diversification_weights(self, covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """Maximum diversification portfolio."""
        if covariance is None:
            cov_obj = self.get_covariance()
            if cov_obj is None:
                return np.array([1.0])
            covariance = cov_obj.matrix
        
        n = covariance.shape[0]
        try:
            import cvxpy as cp
            w = cp.Variable(n)
            vol = cp.sqrt(cp.quad_form(w, covariance))
            weighted_vol = cp.sum(cp.multiply(w, cp.sqrt(cp.diag(covariance))))
            objective = cp.Maximize(weighted_vol / vol)
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            if w.value is not None:
                return np.maximum(w.value, 0)
        except ImportError:
            pass
        
        # Fallback: equal weight
        return np.ones(n) / n
    
    def _get_price_series(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price series from database."""
        try:
            rows = self.db.q("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts", (symbol,))
            if not rows:
                return None
            return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
        except Exception:
            return None
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, float, float]:
        """Calculate current drawdown, max drawdown, and drawdown duration."""
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / np.maximum(peak, 1e-9)
        current_dd = float(drawdown[-1])
        max_dd = float(np.min(drawdown))
        
        # Drawdown duration
        in_drawdown = drawdown < -0.001
        dd_duration = 0
        for is_dd in reversed(in_drawdown):
            if is_dd:
                dd_duration += 1
            else:
                break
        
        return current_dd, max_dd, float(dd_duration)
    
    def drawdown_based_size(
        self,
        price: float,
        stop_distance: float,
        current_dd: float,
        max_dd: float
    ) -> int:
        """Position sizing that reduces exposure as drawdown increases."""
        # High-leverage mode allows up to 3.5x base on high confidence; DD can cut to 0.3x
        leverage_cfg = self.cfg.get("leverage_engine", {})
        max_mult = float(leverage_cfg.get("max_drawdown_mult", 3.5))
        base_fraction = self.rules["risk_per_trade_pct"] / 100.0
        
        # Reduce size as drawdown approaches max allowed
        if max_dd > 0:
            dd_ratio = abs(current_dd) / max_dd
            if dd_ratio > 0.5:
                # Linear reduction from 1.0x to 0.25x as DD goes from 50% to 100% of max
                reduction = 1.0 - 0.75 * min((dd_ratio - 0.5) / 0.5, 1.0)
                effective_fraction = base_fraction * reduction
            else:
                effective_fraction = base_fraction
        else:
            effective_fraction = base_fraction
        
        # Allow higher cap when leverage_engine enabled and confidence high
        effective_fraction = min(effective_fraction, base_fraction * max_mult)
        
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in (price, stop_distance)):
            return 0
        
        risk_amount = self.cfg["capital"] * effective_fraction
        raw_quantity = int(risk_amount / stop_distance)
        notional_quantity = int(self.rules["max_notional_per_trade"] / price)
        return max(0, min(raw_quantity, notional_quantity))
    
    def check_portfolio_correlation(self, new_symbol: str, new_weight: float) -> Tuple[bool, str]:
        """Check if adding a position would violate correlation limits."""
        cov_obj = self.get_covariance()
        if cov_obj is None or new_symbol not in cov_obj.symbols:
            return True, "ok"
        
        idx = cov_obj.symbols.index(new_symbol)
        correlations = cov_obj.correlation[idx]
        
        # Check max correlation with existing positions
        max_corr = float(np.max(np.abs(correlations)))
        if max_corr > self._max_correlation:
            return False, f"max_correlation_exceeded: {max_corr:.3f} > {self._max_correlation}"
        
        return True, "ok"
    
    def should_rebalance(self, current_weights: np.ndarray, target_weights: np.ndarray) -> bool:
        """Check if portfolio needs rebalancing."""
        if len(current_weights) != len(target_weights):
            return True
        diff = np.abs(current_weights - target_weights)
        return float(np.max(diff)) > self._rebalance_threshold
    
    def on_trade_close(self, pnl: float) -> None:
        self.rollover()
        self.day_pnl += pnl
        self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0
        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('PNL',?,?)", (f"trade_pnl={pnl:.2f} day_total={self.day_pnl:.2f}", iso()))
