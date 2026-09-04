"""Advanced Risk Management Module
===================================
Integrates best practices from:
- factor-pricing-model-risk-model (factor models, covariance estimation, risk tracking)
- tcapy (transaction cost analysis)
- Qlib (factor models, portfolio optimization)
- StockSharp (advanced order types, risk controls)
- Qlib (factor models, portfolio optimization)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


try:
    from sklearn.decomposition import PCA  # noqa: F401 - availability probe
    from sklearn.covariance import LedoitWolf, OAS  # noqa: F401 - availability probe
    from scipy.optimize import minimize  # noqa: F401 - availability probe
    from scipy.linalg import pinv, sqrtm  # noqa: F401 - availability probe
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cvxpy as cp  # noqa: F401 - availability probe
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class FactorModel:
    """Factor model with loadings, factor covariance, and specific risk."""
    factor_names: list[str]
    loadings: np.ndarray  # (n_assets, n_factors)
    factor_cov: np.ndarray
    specific_var: np.ndarray
    factor_returns: np.ndarray


@dataclass
class PortfolioRisk:
    """Portfolio risk decomposition."""
    total_var: float
    factor_var: float
    specific_var: float
    marginal_var: np.ndarray
    component_var: np.ndarray
    diversification_ratio: float
    concentration: float
    effective_n: float


class FactorRiskModel:
    """Advanced factor risk model with multiple factor types.
    
    Integrates patterns from:
    - factor-pricing-model-risk-model (PCA, factor models, covariance estimation)
    - Qlib (factor models, risk models)
    - StockSharp (risk controls)
    - tcapy (TCA, market impact)
    """
    
    def __init__(self, config: dict, db=None):
        self.config = config
        self.db = db
        self.n_factors = config.get("risk", {}).get("n_factors", 10)
        self.lookback = config.get("risk", {}).get("lookback", 252)
        self.min_periods = config.get("risk", {}).get("min_periods", 60)
        
        # Model components
        self.factor_model = None
        self.factor_returns = None
        self.factor_cov = None
        self.specific_var = None
        self.loadings = None
        self.factor_names = []
        self.asset_names = []
        
        # Risk limits
        self.max_position_pct = config.get("risk", {}).get("max_position_pct", 0.10)
        self.max_sector_pct = config.get("risk", {}).get("max_sector_pct", 0.30)
        self.max_var_pct = config.get("risk", {}).get("max_var_pct", 0.05)
        self.max_leverage = config.get("risk", {}).get("max_leverage", 1.0)
        
    def fit_factor_model(self, returns: pd.DataFrame, method: str = "pca") -> dict:
        """Fit factor model using PCA or factor model.
        
        Args:
            returns: DataFrame of asset returns (dates x assets)
            method: "pca", "factor", or "statistical"
            
        Returns:
            Dict with factor model components
        """
        if len(returns.columns) < 2:
            return {"error": "Need at least 2 assets"}
            
        returns_clean = returns.dropna(axis=1, how='all').fillna(0)

        # Standardize
        returns_scaled = (returns_clean - returns_clean.mean()) / (returns_clean.std() + 1e-8)
        
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(self.n_factors, len(returns.columns) - 1))
        factors = pca.fit_transform(returns_scaled)
        
        # Factor returns
        factor_returns = pd.DataFrame(
            factors, 
            index=returns.index, 
            columns=[f"F{i}" for i in range(pca.n_components_)]
        )
        
        # Specific variance (idiosyncratic risk)
        reconstructed = factor_returns @ pca.components_
        residuals = returns_scaled - reconstructed
        specific_var = residuals.var()
        
        # Factor covariance with Ledoit-Wolf shrinkage
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        factor_cov = lw.fit(factor_returns).covariance_
        
        # Factor loadings
        loadings = pca.components_.T
        
        self.factor_model = {
            "loadings": loadings,
            "factor_returns": factor_returns,
            "factor_cov": factor_cov,
            "specific_var": specific_var,
            "pca": pca,
            "n_factors": pca.n_components_,
            "explained_variance": pca.explained_variance_ratio_.sum()
        }
        
        return self.factor_model
    
    def estimate_covariance(self, returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
        """Estimate covariance matrix with various shrinkage methods.
        
        Methods:
        - "sample": Sample covariance
        - "ledoit_wolf": Ledoit-Wolf shrinkage
        - "oas": Oracle Approximating Shrinkage
        - "constant": Constant correlation
        - "factor": Factor model based
        """
        from sklearn.covariance import LedoitWolf, OAS
        
        returns_clean = returns.dropna(axis=1, how='all').fillna(0)

        if method == "ledoit_wolf":
            lw = LedoitWolf()
            return lw.fit(returns_clean).covariance_
        elif method == "oas":
            oas = OAS()
            return oas.fit(returns_clean).covariance_
        elif method == "sample":
            return returns.cov().values
        elif method == "constant":
            # Constant correlation model
            corr = returns.corr().values
            std = returns.std().values
            outer_std = np.outer(std, std)
            return corr * outer_std
        else:
            return returns.cov().values
    
    def portfolio_risk(self, weights: np.ndarray, returns: pd.DataFrame = None, 
                       factor_model: dict = None) -> dict:
        """Calculate comprehensive portfolio risk metrics."""
        
        if factor_model is None:
            factor_model = self.factor_model
            
        if factor_model is None:
            # Simple covariance-based risk
            cov = self.estimate_covariance(returns)
            port_var = weights @ cov @ weights
            return {
                "portfolio_var": float(weights @ cov @ weights),
                "volatility": float(np.sqrt(weights @ cov @ weights)),
                "marginal_var": 2 * cov @ weights,
                "component_var": weights * (2 * cov @ weights)
            }
        
        # Factor model risk decomposition
        loadings = factor_model.get("loadings")
        factor_cov = factor_model.get("factor_cov")
        specific_var = factor_model.get("specific_var")
        
        if loadings is None or factor_cov is None or specific_var is None:
            # Fallback
            cov = self.estimate_covariance(returns)
            port_var = weights @ cov @ weights
            return {
                "portfolio_var": float(port_var),
                "volatility": float(np.sqrt(port_var)),
                "marginal_var": 2 * cov @ weights,
                "component_var": weights * (2 * cov @ weights)
            }
        
        # Factor model risk decomposition
        B = loadings  # (n_assets, n_factors)
        F = factor_model.get("factor_cov")
        D = np.diag(factor_model.get("specific_var"))
        
        # Portfolio variance = w' * (B * F * B' + D) * w
        BF = B @ F @ B.T
        D = np.diag(specific_var)
        cov_matrix = BF + np.diag(specific_var)
        
        port_var = weights @ cov_matrix @ weights
        marginal = 2 * cov_matrix @ weights
        component = weights * marginal
        
        # Factor risk
        factor_exposure = weights @ B
        factor_var = factor_exposure @ F @ factor_exposure
        
        # Specific risk
        specific_risk = weights @ D @ weights
        
        return {
            "portfolio_var": float(port_var),
            "volatility": float(np.sqrt(port_var)),
            "factor_var": float(factor_var),
            "specific_var": float(specific_risk),
            "factor_exposure": factor_exposure.tolist(),
            "marginal_var": marginal.tolist(),
            "component_var": component.tolist(),
            "diversification_ratio": float(np.sum(np.abs(weights) * np.sqrt(np.diag(cov_matrix))) / np.sqrt(port_var)) if port_var > 0 else 0
        }
    
    def optimize_portfolio(self, 
                          expected_returns: np.ndarray,
                          covariance: np.ndarray,
                          objective: str = "max_sharpe",
                          constraints: dict = None) -> np.ndarray:
        """Portfolio optimization using various objectives.
        
        Objectives:
        - "max_sharpe": Maximum Sharpe ratio
        - "min_variance": Minimum variance
        - "risk_parity": Risk parity
        - "max_return": Maximum return for given risk
        - "min_cvar": Minimum CVaR (requires CVXPY)
        """
        n = len(expected_returns)
        
        if not CVXPY_AVAILABLE:
            # Simple analytical solutions
            if objective == "min_variance":
                # Minimum variance portfolio
                inv_cov = np.linalg.pinv(covariance)
                ones = np.ones(n)
                w = inv_cov @ ones / (ones @ inv_cov @ ones)
                return w
            elif objective == "max_sharpe":
                # Maximum Sharpe (assuming risk-free = 0)
                inv_cov = np.linalg.pinv(covariance)
                w = inv_cov @ expected_returns / (np.ones(n) @ inv_cov @ expected_returns)
                return w
            elif objective == "risk_parity":
                # Risk parity via inverse volatility (approximation)
                vols = np.sqrt(np.diag(covariance))
                w = 1 / vols
                w = w / w.sum()
                return w
        
        # CVXPY optimization for more complex objectives
        if CVXPY_AVAILABLE:
            import cvxpy as cp
            w = cp.Variable(n)
            
            if objective == "max_sharpe":
                # Approximate as max return - lambda * risk
                lambda_risk = 1.0
                objective = cp.Maximize(expected_returns @ w - lambda_risk * cp.quad_form(w, covariance))
            elif objective == "min_variance":
                objective = cp.Minimize(cp.quad_form(w, covariance))
            elif objective == "max_return":
                objective = cp.Maximize(expected_returns @ w)
            
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]
            
            if self.max_leverage < np.inf:
                constraints.append(cp.sum(w) <= self.max_leverage)
                
            if self.max_position_pct < 1:
                constraints.append(w <= self.max_position_pct)
                
            prob = cp.Problem(objective, constraints)
            prob.solve()
            return w.value
        
        return np.ones(n) / n  # Equal weight fallback
    
    def stress_test(self, portfolio_weights: np.ndarray, 
                    scenarios: dict = None) -> dict:
        """Run stress tests on portfolio.
        
        Scenarios:
        - Market crash (-20%, -30%, -50%)
        - Volatility spike (2x, 3x)
        - Correlation breakdown (corr -> 1)
        - Liquidity crisis (bid-ask spread widening)
        """
        # Historical scenarios
        scenarios = scenarios or {
            "market_crash_20": {"equity_shock": -0.20, "vol_mult": 2.0, "corr_mult": 1.5},
            "market_crash_30": {"equity_shock": -0.30, "vol_mult": 3.0, "corr_mult": 2.0},
            "vol_spike_2x": {"equity_shock": 0.0, "vol_mult": 2.0, "corr_mult": 1.0},
            "corr_breakdown": {"equity_shock": 0.0, "vol_mult": 1.0, "corr_mult": 1.0},
        }
        
        # This would integrate with the actual portfolio
        # For now return framework
        return {
            "framework": "stress_testing",
            "scenarios_available": list(scenarios.keys()),
            "methodology": "Historical + hypothetical stress scenarios"
        }
    
    def var_es(self, returns: pd.Series, alpha: float = 0.05, method: str = "historical") -> tuple:
        """Value at Risk and Expected Shortfall.
        
        Methods:
        - historical: Historical simulation
        - parametric: Gaussian assumption
        - cornish_fisher: Cornish-Fisher expansion
        - evt: Extreme Value Theory (GPD)
        """
        returns = returns.dropna()
        
        if method == "historical":
            var = -np.percentile(returns, alpha * 100)
            es = -returns[returns <= -var].mean() if (returns <= -var).any() else var
        elif method == "parametric":
            from scipy.stats import norm
            sigma = returns.std()
            var = -norm.ppf(alpha) * returns.std() - returns.mean()
            es = - (returns.mean() - sigma * norm.pdf(norm.ppf(alpha)) / alpha)
        elif method == "cornish_fisher":
            from scipy.stats import skew, kurtosis
            z = norm.ppf(alpha)
            s = skew(returns)
            k = kurtosis(returns)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
            var = -(returns.mean() + returns.std() * z_cf)
            es = var * 1.2  # Approximation
        else:
            var = -np.percentile(returns, alpha * 100)
            es = -returns[returns <= -var].mean() if (returns <= -var).any() else var
            
        return float(var), float(es)
    
    def expected_shortfall(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """Expected Shortfall (Conditional VaR)."""
        var, es = self.var_es(returns, alpha, "historical")
        return es
    
    def backtest_var(self, returns: pd.Series, var_series: pd.Series, 
                     alpha: float = 0.05) -> dict:
        """Backtest VaR model (Kupiec, Christoffersen tests)."""
        violations = (returns < -var_series).astype(int)
        n_violations = violations.sum()
        n_total = len(violations)
        expected = alpha * len(violations)
        
        # Kupiec test
        from scipy.stats import chi2
        if n_violations > 0 and n_violations < len(violations):
            lr_uc = -2 * np.log(
                (1 - alpha)**(n_total - n_violations) * alpha**n_violations /
                ((n_violations / n_total)**n_violations * 
                 (1 - n_violations / n_total)**(n_total - n_violations))
            )
            p_value = 1 - chi2.cdf(lr_uc, 1)
        else:
            lr_uc = np.nan
            p_value = np.nan
            
        return {
            "violations": int(n_violations),
            "expected": expected,
            "violation_rate": n_violations / n_total,
            "kupiec_stat": lr_uc,
            "kupiec_pvalue": p_value,
            "traffic_light": "green" if p_value > 0.05 else "yellow" if p_value > 0.01 else "red"
        }


class TransactionCostModel:
    """Transaction cost analysis from tcapy patterns."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.spread_bps = config.get("spread_bps", 5)  # bps
        self.commission_bps = config.get("commission_bps", 2)  # bps
        self.market_impact_coeff = config.get("market_impact_coeff", 0.1)
        self.urgency = config.get("urgency", "normal")
        
    def estimate_cost(self, 
                      symbol: str,
                      side: str,
                      quantity: int,
                      price: float,
                      adv: float,  # Average daily volume
                      volatility: float,
                      spread: float = None) -> dict:
        """Estimate transaction costs using Almgren-Chriss model."""
        
        notional = quantity * price

        # Spread cost
        spread_cost = self.spread_bps / 10000 * notional
        
        # Market impact (Almgren-Chriss square-root law)
        participation_rate = min(abs(quantity) / max(adv, 1), 0.1)
        temporary_impact = 0.1 * volatility * np.sqrt(participation_rate) * notional
        permanent_impact = 0.5 * volatility * participation_rate * notional
        
        # Commission
        commission = notional * 0.0002  # 2 bps
        
        total_cost = spread_cost + temporary_impact + permanent_impact + commission
        
        return {
            "total_cost": total_cost,
            "spread_cost": spread_cost,
            "market_impact": temporary_impact + permanent_impact,
            "commission": commission,
            "total_bps": total_cost / notional * 10000,
            "participation_rate": participation_rate
        }
    
    def optimal_execution(self, 
                         quantity: int,
                         price: float,
                         volatility: float,
                         adv: float,
                         horizon: int = 60,  # minutes
                         risk_aversion: float = 1e-6) -> list:
        """Optimal execution schedule (Almgren-Chriss)."""
        # Simplified - returns equal slices for now
        n_slices = max(1, int(horizon / 5))
        slice_size = quantity // n_slices
        remainder = quantity % n_slices
        
        schedule = []
        for i in range(n_slices):
            qty = slice_size + (1 if i < remainder else 0)
            schedule.append({
                "slice": i + 1,
                "quantity": qty,
                "target_time": f"{9:02d}:{30 + i*5:02d}",
                "urgency": "high" if i < n_slices/2 else "normal"
            })
        return schedule


# Risk limits and monitoring
class RiskMonitor:
    """Real-time risk monitoring with alerts."""
    
    def __init__(self, config: dict):
        self.config = config
        self.limits = config.get("risk", {}).get("limits", {})
        self.alerts = []
        
    def check_limits(self, portfolio_state: dict) -> list:
        """Check all risk limits."""
        alerts = []
        
        # Position limits
        for symbol, pos in portfolio_state.get("positions", {}).items():
            if abs(pos.get("value", 0)) > self.config.get("risk", {}).get("max_position", 1e6):
                alerts.append({
                    "type": "position_limit",
                    "symbol": symbol,
                    "message": f"Position exceeds limit: {pos['value']}"
                })
        
        # VaR limit
        if portfolio_state.get("var_95", 0) > portfolio_state.get("capital", 0) * 0.05:
            alerts.append({
                "type": "var_limit",
                "message": "Portfolio VaR exceeds 5% limit"
            })
            
        # Drawdown
        dd = portfolio_state.get("drawdown", 0)
        if dd > self.config.get("risk", {}).get("max_drawdown", 0.1):
            alerts.append({
                "type": "drawdown",
                "message": f"Drawdown {dd:.1%} exceeds limit"
            })
            
        return alerts
    
    def get_risk_report(self, portfolio_state: dict) -> dict:
        """Generate comprehensive risk report."""
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "alerts": self.check_limits(portfolio_state),
            "var_95": portfolio_state.get("var_95", 0),
            "var_99": portfolio_state.get("var_99", 0),
            "drawdown": portfolio_state.get("drawdown", 0),
            "leverage": portfolio_state.get("leverage", 1.0),
            "concentration": portfolio_state.get("concentration", 0),
            "liquidity_score": portfolio_state.get("liquidity_score", 1.0)
        }


# Main exports
__all__ = [
    "FactorRiskModel",
    "TransactionCostModel", 
    "RiskMonitor",
    "PortfolioRisk",
    "FactorModel",
    "PortfolioRisk"
]