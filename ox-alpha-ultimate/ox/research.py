"""Backtesting and research techniques: rigorous validation guardrails.

Implements the research-side techniques — walk-forward folding with embargo,
Monte Carlo resampling, deflated Sharpe, multiple-testing correction, stress
and scenario analysis, bootstrap CIs, signal decay, overfitting detection —
as reusable functions over simple return series, complementing the agent's
existing Backtester/Scorer pipeline.
"""

from __future__ import annotations

import numpy as np

REGISTRY: dict = {}


def _reg(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _clean(returns) -> np.ndarray:
    r = _f(returns)
    return r[~np.isnan(r)]


# ── Validation ────────────────────────────────────────────────────────────
@_reg("walk_forward_folds")
def walk_forward_folds(n: int, folds: int, oos_length: int, embargo: int) -> list[tuple[range, range]]:
    """Expanding-train / forward-test folds separated by an embargo gap."""
    first_train_end = n - folds * (oos_length + embargo)
    if first_train_end <= 0:
        raise ValueError("series too short for the requested folds")
    out = []
    for k in range(folds):
        train_end = first_train_end + k * (oos_length + embargo)
        test = range(train_end + embargo, train_end + embargo + oos_length)
        out.append((range(0, train_end), test))
    return out


@_reg("kfold_timeseries")
def kfold_timeseries(n: int, k: int = 5, gap: int = 1) -> list[tuple[range, range]]:
    """Purged time-series K-fold: blocks in order, `gap` bars removed before test."""
    block = n // k
    folds = []
    for i in range(k):
        start = i * block
        test = range(start, min(start + block, n))
        train = list(range(0, max(0, start - gap))) + list(range(min(start + block, n), n))
        if train and len(test) > 0:
            folds.append((train, test))
    return folds


@_reg("monte_carlo_paths")
def monte_carlo_paths(returns, n_paths: int = 1000, horizon: int = 60, seed: int = 0) -> dict:
    """IID-bootstrap equity paths; block=False keeps this honest for research."""
    r = _clean(returns)
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_paths, horizon), replace=True)
    equity = np.cumprod(1 + draws, axis=1)
    final = equity[:, -1]
    return {"p5": float(np.percentile(final, 5)), "p50": float(np.percentile(final, 50)),
            "p95": float(np.percentile(final, 95)),
            "prob_loss": float(np.mean(final < 1.0)),
            "maxdd_p95": float(np.percentile(_drawdowns(equity), 95))}


def _drawdowns(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity, axis=1)
    return np.min((equity - peak) / peak, axis=1)


@_reg("slippage_stress")
def slippage_stress(returns, costs_bps: list[float]) -> dict:
    """Net returns after per-trade cost scenarios."""
    r = _clean(returns)
    out = {}
    for bps in costs_bps:
        net = r - bps / 1e4
        out[f"{bps}bps"] = {"mean": float(np.mean(net)), "sharpe": sharpe(net)}
    return out


@_reg("deflated_sharpe")
def deflated_sharpe(sharpe_hat: float, n_trials: int, t_stats: np.ndarray | None,
                    n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & López de Prado DSR: Sharpe after multiple-testing deflation."""
    if t_stats is not None and len(t_stats) > 1:
        trial_var = float(np.var(_f(t_stats), ddof=1))
    else:
        trial_var = 1.0 / n_trials
    expected_max = np.sqrt(trial_var) * (2 * np.log(n_trials) - np.log(np.log(n_trials))
                                         - np.log(np.pi)) if n_trials > 2 else 0.0
    denom = np.sqrt(max(1 - skew * sharpe_hat + (kurt - 1) / 4 * sharpe_hat ** 2, 1e-9) / n_obs)
    dsr = float((sharpe_hat - expected_max * denom) / max(denom * np.sqrt(n_obs), 1e-9))
    return {"deflated_sharpe": dsr, "expected_max_sharpe": float(expected_max),
            "deflated_ok": dsr > 0}


@_reg("bonferroni_holm")
def bonferroni_holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm step-down correction over multiple strategy comparisons."""
    order = np.argsort(_f(p_values))
    m = len(p_values)
    results = [None] * m
    rejected = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        keep = rejected and p_values[idx] <= threshold
        if not keep:
            rejected = False
        results[idx] = {"p": p_values[idx], "adjusted": min(1.0, p_values[idx] * (m - rank)),
                        "significant": bool(keep)}
    return results


@_reg("whites_reality_check")
def whites_reality_check(strategy_returns: np.ndarray, benchmark_returns: np.ndarray,
                         n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap data-snooping test for the best strategy vs benchmark."""
    S, B = _clean(strategy_returns), _clean(benchmark_returns)
    n = min(len(S), len(B))
    diff = S[:n] - B[:n]
    observed = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(np.mean(diff[idx]))
    p_value = float(np.mean(np.array(stats) >= observed))
    return {"observed_edge": observed, "p_value": p_value, "significant": p_value < 0.05}


# ── Bias and stability ────────────────────────────────────────────────────
@_reg("lookahead_probe")
def lookahead_probe(signals_fn, data_factory, perturb_last: bool = True) -> dict:
    """A signal built without look-ahead must not change when the future does.

    Recomputes the signal on a truncated series vs the full series; the
    overlapping prefix must be identical.
    """
    full = signals_fn(data_factory(300))
    truncated = signals_fn(data_factory(250))
    prefix_equal = bool(np.allclose(full[:250], truncated[:250], equal_nan=True))
    return {"lookahead_free": prefix_equal, "checked_bars": 250}


@_reg("parameter_stability")
def parameter_stability(scores: dict[str, float]) -> dict:
    """Score surface flatness: a real edge does not collapse at neighbouring params."""
    values = np.array(list(scores.values()))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {"median": median, "mad": mad, "stable": mad <= max(0.1 * abs(median), 0.05),
            "n_params": len(values)}


@_reg("overfitting_score")
def overfitting_score(is_scores: list[float], oos_scores: list[float]) -> dict:
    """PBO-flavoured gauge: rank correlation between IS and OOS performance."""
    is_s, oos_s = _f(is_scores), _f(oos_scores)
    if len(is_s) != len(oos_s) or len(is_s) < 3:
        return {"overfit": True, "reason": "insufficient paired samples"}
    is_rank = np.argsort(np.argsort(is_s))
    oos_rank = np.argsort(np.argsort(oos_s))
    corr = float(np.corrcoef(is_rank, oos_rank)[0, 1]) if np.std(is_rank) > 0 else 0.0
    decay = float(np.mean((is_s - oos_s) / np.where(np.abs(is_s) > 1e-9, is_s, np.nan)))
    return {"rank_correlation": corr, "is_oos_decay": decay,
            "overfit": corr < 0.0 or decay > 0.7}


# ── Measurement ───────────────────────────────────────────────────────────
@_reg("sharpe")
def sharpe(returns, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))


@_reg("sortino")
def sortino(returns, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    if len(downside) < 1 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(r) / np.std(downside, ddof=1) * np.sqrt(periods_per_year))


@_reg("max_drawdown")
def max_drawdown(equity) -> float:
    e = _f(equity)
    peak = np.maximum.accumulate(e)
    return float(np.min((e - peak) / peak))


@_reg("signal_decay")
def signal_decay(signals: np.ndarray, returns: np.ndarray, horizons=(1, 3, 5, 10, 20)) -> dict:
    """Forward-return profile of signals across horizons (alpha persistence)."""
    s, r = _f(signals), _f(returns)
    cum = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(r))])
    out = {}
    for h in horizons:
        fwd = np.full(len(r), np.nan)
        fwd[:-h] = cum[h + 1:] / cum[1:len(r) - h + 1] - 1
        active = (s != 0)[:len(r) - h]
        out[f"h={h}"] = float(np.nanmean(fwd[:len(r) - h][active])) if np.any(active) else 0.0
    return out


@_reg("stress_scenarios")
def stress_scenarios(equity_curve, shocks=(-0.05, -0.10, -0.20, 0.10)) -> dict:
    """Apply instantaneous gap shocks to the current equity."""
    e = _f(equity_curve)
    current = float(e[-1]) if len(e) else 1.0
    return {f"shock{s:+.0%}": round(current * (1 + s), 4) for s in shocks}


@_reg("shadow_trading_gap")
def shadow_trading_gap(live_returns, backtest_returns) -> dict:
    """Live-vs-backtest divergence: slippage and fill-quality reality check."""
    live, bt = _clean(live_returns), _clean(backtest_returns)
    n = min(len(live), len(bt))
    if n < 2:
        return {"tracking_error": 0.0, "ok": True}
    diff = live[:n] - bt[:n]
    return {"tracking_error": float(np.std(diff)),
            "mean_gap": float(np.mean(diff)),
            "ok": float(np.mean(diff)) > -0.002}


def technique(name: str):
    return REGISTRY[name]


def self_test() -> tuple[int, list[str]]:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0004, 0.01, 500)
    equity = np.cumprod(1 + r)
    failures = []
    checks = {
        "walk_forward_folds": lambda: walk_forward_folds(1000, 3, 60, 5),
        "kfold_timeseries": lambda: kfold_timeseries(500, 5, 2),
        "monte_carlo_paths": lambda: monte_carlo_paths(r, n_paths=100),
        "slippage_stress": lambda: slippage_stress(r, [0, 5, 10]),
        "deflated_sharpe": lambda: deflated_sharpe(1.5, 50, None, 500),
        "bonferroni_holm": lambda: bonferroni_holm([0.01, 0.04, 0.03, 0.2]),
        "whites_reality_check": lambda: whites_reality_check(r, r * 0.5, n_boot=100),
        "parameter_stability": lambda: parameter_stability({"a": 1.0, "b": 1.05, "c": 0.98}),
        "overfitting_score": lambda: overfitting_score([1, 2, 3, 4], [1, 2, 3, 4]),
        "sharpe": lambda: sharpe(r),
        "sortino": lambda: sortino(r),
        "max_drawdown": lambda: max_drawdown(equity),
        "signal_decay": lambda: signal_decay(np.where(r > 0, 1, 0), r),
        "stress_scenarios": lambda: stress_scenarios(equity),
        "shadow_trading_gap": lambda: shadow_trading_gap(r, r),
    }
    for name, fn in checks.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    return len(checks), failures
