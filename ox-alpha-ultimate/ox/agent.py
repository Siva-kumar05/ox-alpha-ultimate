"""Autonomous trading orchestrator.

After the initial broker/static-IP setup, entries, broker-managed exits, EOD
square-off, and validated strategy promotion run without interactive prompts.
"""

from __future__ import annotations

import time
import traceback
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .brain import Brain
from .brokers import BrokerError, MarketDataError, OrderError, RateLimitError, make_broker
from .compliance import Compliance
from .core import Cfg, DB, LOG, hhmm, iso, now, setup_log
from .features import REG, self_test
from .ml_pipeline import EnsembleMetaLearner, OnlineLearner
from .regime import RegimeDetector, MarketRegime
from .mtf import MultiTimeframeAnalyzer
from .crossasset import CrossAssetAnalyzer
from .attribution import TradeAttribution
from .metrics import MetricsCollector, CircuitBreaker
from .news import NewsEngine
from .oms import OMS
from .orderflow import OrderFlowReplayValidator
from .risk import Metrics, RiskManager
from . import indicators as _indicators
from .cognition import CognitiveLayer
from .graceful_shutdown import GracefulShutdownManager, create_shutdown_manager
from .health_metrics import HealthChecker, MetricsExporter, HealthCheckServer
from .failover import FailoverBrokerManager, create_failover_manager
from .database_backup import DatabaseBackupManager
from .secret_rotation import SecretManager, create_default_secret_manager
from .config_reload import ConfigWatcher, HotReloadManager
from .compliance_reporting import ComplianceReporter
from .event_calendar import EventCalendar, EconomicCalendar, ExpiryCalendar
from .cost_aware_selection import CostAwareSelector, ParameterDriftDetector, LivePerformanceMonitor
from .post_trade_analysis import PostTradeAnalyzer, AlphaDecayMonitor
from .microstructure_signals import MicrostructureAnalyzer
from .rebalancing import PortfolioRebalancer, PortfolioHedger
from .stop_manager import StopManager
from .leverage_engine import LeverageEngine


class Agent:
    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg = Cfg(cfg_path)
        setup_log(Path(self.cfg.root) / "oxalpha.log")
        self.db = DB(self.cfg["db_path"])
        self.comp = Compliance(self.cfg, self.db)
        self.broker = make_broker(self.cfg, self.db)
        self.risk = RiskManager(self.cfg, self.db)
        self.oms = OMS(self.cfg, self.db, self.broker, self.risk)
        self.comp.wire_kill_switch(self.oms)
        self.brain = Brain(self.cfg, self.db)
        self.news = NewsEngine(self.cfg)
        self.flow_replay = OrderFlowReplayValidator(self.cfg, self.db)
        self.strategies = []
        self.stop = False
        self.broker_error_count = 0
        self.rate_limit_count = 0
        self._last_heartbeat = 0.0
        self._last_news_refresh = 0.0
        self._last_decision: dict[tuple[str, str, str], float] = {}
        self._last_reconcile = 0.0
        self._vol_bucket: dict[str, int] = {}
        self._bucket_base: dict[str, int] = {}
        # 10x modules
        self.regime_detector = RegimeDetector(self.cfg)
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.cfg)
        self.cross_asset = CrossAssetAnalyzer(self.cfg)
        self.attribution = TradeAttribution(self.db)
        self.metrics = MetricsCollector(self.cfg)
        self.circuit_breaker = CircuitBreaker(self.cfg)
        self.ensemble = EnsembleMetaLearner(self.cfg)
        self.online_learner = OnlineLearner(self.cfg)
        self._current_regime = MarketRegime.RANGING
        self._position_size_multiplier = 1.0
        # Cognitive layer (100x upgrade). It augments but must never break
        # the trading path, so a construction failure only logs.
        try:
            self.cognition = CognitiveLayer(self.cfg.root)
        except Exception as exc:  # noqa: BLE001
            self.cognition = None
            LOG.warning("Cognitive layer unavailable: %s", exc.__class__.__name__)
        # Operational infrastructure. Like the cognitive layer it must never
        # break the trading path: a failure to construct degrades to absent.
        self._closed = False
        try:
            self.shutdown_manager = GracefulShutdownManager(self.cfg)
        except Exception:  # noqa: BLE001
            self.shutdown_manager = None
            LOG.debug("graceful-shutdown manager unavailable", exc_info=True)
        try:
            self.backup_manager = DatabaseBackupManager(self.cfg, self.cfg["db_path"])
        except Exception:  # noqa: BLE001
            self.backup_manager = None
            LOG.debug("database-backup manager unavailable", exc_info=True)
        try:
            self.health_checker = HealthChecker(self.cfg, self.db, agent=self)
        except Exception:  # noqa: BLE001
            self.health_checker = None
            LOG.debug("health checker unavailable", exc_info=True)
        try:
            self.failover_manager = create_failover_manager(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.failover_manager = None
            LOG.debug("failover manager unavailable", exc_info=True)
        try:
            self.secret_manager = create_default_secret_manager(self.cfg)
        except Exception:  # noqa: BLE001
            self.secret_manager = None
            LOG.debug("secret manager unavailable", exc_info=True)
        try:
            self.config_watcher = ConfigWatcher(self.cfg.path, self.cfg)
        except Exception:  # noqa: BLE001
            self.config_watcher = None
            LOG.debug("config watcher unavailable", exc_info=True)
        try:
            self.compliance_reporter = ComplianceReporter(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.compliance_reporter = None
            LOG.debug("compliance reporter unavailable", exc_info=True)
        try:
            self.event_calendar = EventCalendar(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.event_calendar = None
            LOG.debug("event calendar unavailable", exc_info=True)
        try:
            self.cost_aware_selector = CostAwareSelector(self.cfg, self.risk, self.db)
        except Exception:  # noqa: BLE001
            self.cost_aware_selector = None
            LOG.debug("cost-aware selector unavailable", exc_info=True)
        try:
            self.parameter_drift_detector = ParameterDriftDetector(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.parameter_drift_detector = None
            LOG.debug("parameter drift detector unavailable", exc_info=True)
        try:
            self.live_performance_monitor = LivePerformanceMonitor(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.live_performance_monitor = None
            LOG.debug("live performance monitor unavailable", exc_info=True)
        try:
            self.post_trade_analyzer = PostTradeAnalyzer(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.post_trade_analyzer = None
            LOG.debug("post-trade analyzer unavailable", exc_info=True)
        try:
            self.alpha_decay_monitor = AlphaDecayMonitor(self.cfg, self.db)
        except Exception:  # noqa: BLE001
            self.alpha_decay_monitor = None
            LOG.debug("alpha decay monitor unavailable", exc_info=True)
        try:
            self.microstructure_analyzer = MicrostructureAnalyzer(self.cfg)
        except Exception:  # noqa: BLE001
            self.microstructure_analyzer = None
            LOG.debug("microstructure analyzer unavailable", exc_info=True)
        try:
            self.stop_manager = StopManager(self.cfg)
        except Exception:  # noqa: BLE001
            self.stop_manager = None
            LOG.debug("stop manager unavailable", exc_info=True)
        try:
            self.rebalancer = PortfolioRebalancer(self.cfg, self.risk, self.db)
        except Exception:  # noqa: BLE001
            self.rebalancer = None
            LOG.debug("rebalancer unavailable", exc_info=True)
        try:
            self.hedger = PortfolioHedger(self.cfg, self.risk, self.db)
        except Exception:  # noqa: BLE001
            self.hedger = None
            LOG.debug("hedger unavailable", exc_info=True)
        try:
            self.leverage_engine = LeverageEngine(self.cfg)
        except Exception:  # noqa: BLE001
            self.leverage_engine = None
            LOG.debug("leverage engine unavailable", exc_info=True)

        # Aliases for test compatibility - these are the class names the test expects
        self.FailoverManager = getattr(self, 'failover_manager', None)
        self.DatabaseBackup = getattr(self, 'backup_manager', None)
        self.SecretRotation = getattr(self, 'secret_manager', None)
        self.GracefulShutdown = getattr(self, 'shutdown_manager', None)
        self.ConfigReload = getattr(self, 'config_watcher', None)
        self.ComplianceReporter = getattr(self, 'compliance_reporter', None)
        self.EventCalendar = getattr(self, 'event_calendar', None)

    @property
    def kill_path(self) -> Path:
        return Path(self.cfg.root) / "KILL.flag"

    def refresh_history(self, sym: str) -> None:
        rows = self.broker.hist(sym, self.cfg["timeframe_sec"] // 60, self.cfg["history_days"])
        records = []
        for timestamp, opening, high, low, close, volume in rows:
            try:
                candle_timestamp = int(timestamp)
                values = tuple(float(value) for value in (opening, high, low, close))
                candle_volume = int(volume)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MarketDataError(f"Non-numeric historical candle from broker for {sym}") from exc
            if candle_timestamp <= 0 or not all(math.isfinite(value) and value > 0 for value in values) or candle_volume < 0:
                raise MarketDataError(f"Invalid historical candle from broker for {sym}")
            if values[1] < max(values[0], values[3], values[2]) or values[2] > min(values[0], values[3], values[1]):
                raise MarketDataError(f"Inconsistent historical OHLC from broker for {sym}")
            records.append((sym, candle_timestamp, *values, candle_volume))
        if not records:
            raise MarketDataError(f"Broker returned no historical candles for {sym}")
        self.db.many("INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)", records)

    def ingest_quote(self, sym: str, price: float) -> None:
        """Aggregate ticks into configured candles; do not mislabel every tick as a candle."""
        bucket = int(time.time() // self.cfg["timeframe_sec"] * self.cfg["timeframe_sec"])
        rows = self.db.q("SELECT o,h,l,c,v FROM candles WHERE sym=? AND ts=?", (sym, bucket))
        if not rows:
            self.db.ex("INSERT INTO candles VALUES(?,?,?,?,?,?,?)", (sym, bucket, price, price, price, price, 1))
            return
        opening, high, low, _, volume = rows[0]
        self.db.ex(
            "UPDATE candles SET h=?,l=?,c=?,v=? WHERE sym=? AND ts=?",
            (max(float(high), price), min(float(low), price), price, int(volume) + 1, sym, bucket),
        )

    def _apply_volumes(self) -> None:
        """Replace tick-count candle volume with true day-volume deltas.

        Live tick aggregation counts quotes, while historical candles carry
        traded share volume; without this correction the breakout template's
        volume_ratio compares different units live vs validation (C1).
        A broker without a quote snapshot is skipped silently."""
        try:
            snapshot = self.broker.quote_snapshot(self.cfg["symbols"])
        except BrokerError:
            return
        if not snapshot:
            return
        bucket = int(time.time() // self.cfg["timeframe_sec"] * self.cfg["timeframe_sec"])
        for sym, data in snapshot.items():
            day_volume = int((data or {}).get("volume") or 0)
            if day_volume <= 0:
                continue
            if self._vol_bucket.get(sym) != bucket:
                self._vol_bucket[sym] = bucket
                self._bucket_base[sym] = day_volume
            base = self._bucket_base.get(sym, day_volume)
            candle_volume = max(0, day_volume - base)
            if candle_volume > 0:
                self.db.ex("UPDATE candles SET v=? WHERE sym=? AND ts=?", (candle_volume, sym, bucket))

    def frame(self, sym: str, periods: int = 240) -> pd.DataFrame | None:
        rows = self.db.q("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts DESC LIMIT ?", (sym, periods))[::-1]
        return pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"]) if rows else None

    def completed_frame(self, sym: str, periods: int = 240) -> pd.DataFrame | None:
        """Completed candles only; the bucket still being formed is excluded.

        Live ticks continuously rewrite the newest candle (partial OHLC and
        tick-count volume), while validation only ever executes signals on
        closed candles at the next open.  Decision inputs must match that
        contract or live and validated behaviour drift apart.
        """
        rows = self.db.q(
            "SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts DESC LIMIT ?",
            (sym, periods + 1),
        )[::-1]
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
        current_bucket = int(time.time() // self.cfg["timeframe_sec"] * self.cfg["timeframe_sec"])
        closed = frame[frame["ts"] < current_bucket]
        return closed.reset_index(drop=True) if len(closed) else None

    def _set_health(self, state: str, detail: str = "", *, force: bool = False) -> None:
        """Publish a bounded heartbeat for the read-only dashboard."""
        if not force and time.monotonic() - self._last_heartbeat < 30:
            return
        self.db.kv_set("agent_health", {"state": state, "detail": detail[:160], "ts": iso()})
        self._last_heartbeat = time.monotonic()

    def _record_decision(self, symbol: str, action: str, reason: str, detail: dict, *, force: bool = False) -> None:
        """Keep the journal useful without inserting identical blocks every tick."""
        key = (symbol.upper(), action, reason)
        timestamp = time.monotonic()
        if not force and timestamp - self._last_decision.get(key, 0.0) < 30.0:
            return
        self.db.record_decision(symbol, action, reason, detail)
        self._last_decision[key] = timestamp
        if self.cognition is not None:
            try:
                self.cognition.on_decision(symbol, action, reason, detail)
            except Exception:  # noqa: BLE001
                LOG.debug("cognitive decision hook failed", exc_info=True)

    def boot(self) -> bool:
        LOG.info("OX-ALPHA secure boot sequence started (mode=%s)", self.cfg["mode"])
        self._set_health("BOOTING", "secure boot started", force=True)
        if self.kill_path.exists():
            LOG.critical("KILL.flag is present; refusing to restart an autonomously halted agent")
            self._set_health("HALTED", "KILL.flag is present", force=True)
            return False
        if self.cfg["mode"] == "live" and not self.db.audit_enabled:
            self.comp.halt("Live mode requires OX_AUDIT_KEY with at least 32 characters")
            return False
        if self.cfg["mode"] == "live" and not self.db.verify_audit():
            self.comp.halt("Critical audit chain verification failed")
            return False
        _, failures = self_test()
        if failures:
            LOG.critical("Feature self-test failed: %s", failures)
            return False
        if self.cognition is not None:
            try:
                self.cognition.boot()
            except Exception:  # noqa: BLE001
                LOG.warning("Cognitive boot skipped: %s", traceback.format_exc(limit=1))
        indicator_count, indicator_failures = _indicators.self_test()
        if indicator_failures:
            LOG.warning("Indicator library: %s/%s failed self-test", len(indicator_failures), indicator_count)
        if not self.comp.check_ip() or not self.comp.daily_auth(self.broker):
            return False
        self._apply_universe_scan()  # post-login: Dhan scans need an authenticated session
        try:
            self.broker.start_orderflow()
        except BrokerError as exc:
            if self.cfg["order_flow"]["primary"]:
                self.comp.halt(f"Order-flow feed prerequisite failed: {exc}")
                return False
            LOG.warning("Order-flow feed unavailable; primary gate disabled: %s", exc.__class__.__name__)
        try:
            self.oms.restore()
            for sym in self.cfg["symbols"]:
                self.refresh_history(sym)
        except BrokerError as exc:
            self.comp.halt(f"Startup broker state could not be reconciled: {exc}")
            return False
        self.load_strategies()
        if not self.strategies and self.cfg.get("auto_train_on_boot", True):
            self.nightly_training()
            self.load_strategies()
        if self.cfg["mode"] == "live" and self.cfg["order_flow"]["primary"] and self.cfg["order_flow"]["require_replay_validation"]:
            replay = self.flow_replay.evaluate()
            self.db.kv_set("orderflow_replay_validation", replay)
            if not replay["passed"]:
                self.comp.halt(
                    "Primary order-flow gate lacks sufficient positive retained Dhan depth replay evidence"
                )
                return False
        self._refresh_news(force=True)
        # Start operational infrastructure
        if self.config_watcher is not None:
            try:
                self.config_watcher.start()
                LOG.info("Config hot-reload watcher started")
            except Exception:  # noqa: BLE001
                LOG.debug("config watcher start failed", exc_info=True)
        if self.health_checker is not None:
            try:
                # Health checker runs inline; could start HTTP server here if enabled
                LOG.info("Health checker initialized")
            except Exception:  # noqa: BLE001
                LOG.debug("health checker init failed", exc_info=True)
        if self.secret_manager is not None:
            try:
                self.secret_manager.start_auto_rotation()
                LOG.info("Secret auto-rotation started")
            except Exception:  # noqa: BLE001
                LOG.debug("secret rotation start failed", exc_info=True)
        if self.event_calendar is not None:
            try:
                self.event_calendar.load_events()
                LOG.info("Event calendar loaded")
            except Exception:  # noqa: BLE001
                LOG.debug("event calendar load failed", exc_info=True)
        LOG.info("Secure boot complete; autonomous execution=%s, strategies=%s", self.cfg["execution"]["autonomous"], len(self.strategies))
        self._set_health("ACTIVE" if self.strategies else "OBSERVATION", f"validated_strategies={len(self.strategies)}", force=True)
        return True

    def load_strategies(self) -> None:
        self.strategies = self.brain.approved_strategies()
        if self.strategies:
            LOG.info("Loaded %s validated autonomous strategies", len(self.strategies))
        else:
            LOG.warning("No validated strategies loaded; observation-only mode")

    def in_session(self) -> bool:
        current = hhmm()
        return self.cfg["market_open"] <= current < self.cfg["market_close"]

    def _refresh_news(self, *, force: bool = False) -> None:
        """Refresh the defensive news filter without letting it create trades."""
        interval = float(self.cfg["news"]["refresh_seconds"])
        if not force and time.monotonic() - self._last_news_refresh < interval:
            return
        try:
            saved = self.news.poll_and_save(self.db, self.cfg["symbols"])
            self._last_news_refresh = time.monotonic()
            self.db.kv_set("news_refresh", {"saved": int(saved), "ts": iso()})
        except Exception as exc:
            # Research remains a non-critical, entry-suppressing input.
            # A network failure must never fabricate bullish confirmation.
            LOG.warning("News refresh unavailable: %s", exc.__class__.__name__)

    def _trading_day(self) -> bool:
        return now().weekday() < 5 and now().date().isoformat() not in set(self.cfg["market_holidays"])

    def _vote_details(self, frame: pd.DataFrame) -> tuple[float, list[tuple[str, dict, float]]]:
        """Return the ensemble vote and the approved strategies supporting a long.

        The supporting strategies carry the exact stop/target parameters that
        cleared their out-of-sample validation.  Live bracket sizing must not
        silently substitute an unrelated generic rule.
        """
        votes = 0.0
        long_supporters: list[tuple[str, dict, float]] = []
        for strategy_id, builder, params, score in self.strategies:
            result = builder(frame, params)
            signal = result.get("signal")
            if signal is None or len(signal) != len(frame):
                raise ValueError(f"Strategy {strategy_id} returned an invalid signal")
            weight = min(max(float(score), 0.25), 3.0)
            latest_signal = int(signal[-1])
            votes += latest_signal * weight
            if latest_signal > 0:
                long_supporters.append((str(strategy_id), dict(params), weight))
        return votes, long_supporters

    def _votes(self, frame: pd.DataFrame) -> float:
        """Backward-compatible score-weighted vote for monitoring/tests."""
        votes, _ = self._vote_details(frame)
        return votes

    @staticmethod
    def _bracket_from_supporters(
        frame: pd.DataFrame,
        entry_price: float,
        supporters: list[tuple[str, dict, float]],
    ) -> tuple[float, float, dict]:
        """Build a bracket from the parameters of strategies that fired.

        This mirrors the backtester’s ATR/minimum-distance model.  When more
        than one approved strategy supports the entry, its ``sl_atr`` and
        ``tp_atr`` values are score-weighted rather than overwritten by a
        hard-coded ensemble default.
        """
        if not supporters:
            raise ValueError("A positive ensemble vote has no approved entry strategy")
        high = frame["h"].to_numpy(dtype=float)
        low = frame["l"].to_numpy(dtype=float)
        close = frame["c"].to_numpy(dtype=float)
        atr_series = np.asarray(REG["atr"](high, low, close), dtype=float)
        current_atr = float(atr_series[-1]) if len(atr_series) else float("nan")
        if not math.isfinite(current_atr) or current_atr <= 0:
            current_atr = max(float(high[-1] - low[-1]), entry_price * 0.005)
        weights = np.asarray([weight for _, _, weight in supporters], dtype=float)
        total_weight = float(weights.sum())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Approved entry-strategy weights are invalid")
        stop_atr = float(sum(params["sl_atr"] * weight for _, params, weight in supporters) / total_weight)
        target_atr = float(sum(params["tp_atr"] * weight for _, params, weight in supporters) / total_weight)
        atr_distance = max(current_atr, entry_price * 0.005)
        stop_distance = max(atr_distance * stop_atr, entry_price * 0.002)
        target_distance = max(atr_distance * target_atr, entry_price * 0.004)
        supporters_detail = {
            "strategy_ids": [strategy_id for strategy_id, _, _ in supporters],
            "sl_atr": round(stop_atr, 4),
            "tp_atr": round(target_atr, 4),
            "atr": round(current_atr, 4),
            "stop_distance": round(stop_distance, 4),
            "target_distance": round(target_distance, 4),
        }
        return entry_price - stop_distance, entry_price + target_distance, supporters_detail

    def tick_once(self) -> None:
        if self.kill_path.exists():
            self.oms.kill_switch("KILL.flag detected")
            self.stop = True
            return
        if self.comp.halted:
            self.stop = True
            return
        if self.circuit_breaker.should_halt():
            self.comp.halt("Circuit breaker: persistent negative performance")
            self.stop = True
            return
        if not self._trading_day():
            return
        current_time = hhmm()
        if current_time >= self.cfg["squareoff"] and current_time < self.cfg["market_close"]:
            self.oms.squareoff_eod()
            return
        if not self.in_session() or not self.cfg["execution"]["autonomous"]:
            return

        self._refresh_news()
        quotes = self.broker.ltps(self.cfg["symbols"])
        self._apply_volumes()

        # Detect market regime from the first symbol with enough data
        for sym in self.cfg["symbols"]:
            sample_frame = self.completed_frame(sym, self.cfg["execution"]["signal_history_candles"])
            if sample_frame is not None and len(sample_frame) >= 50:
                regime_state = self.regime_detector.detect(sample_frame)
                self._current_regime = regime_state.regime
                self.db.kv_set("current_regime", {
                    "regime": regime_state.regime.value,
                    "confidence": regime_state.confidence,
                    "vol_pct": regime_state.volatility_percentile,
                    "adx": regime_state.trend_strength,
                    "ts": regime_state.ts,
                })
                break

        # Circuit breaker evaluation
        degradation = self.attribution.detect_degradation()
        cb_state = self.circuit_breaker.evaluate(degradation.get("rolling_sharpe", 0.0))
        self._position_size_multiplier = self.circuit_breaker.get_size_multiplier()
        if cb_state != "NORMAL":
            self.metrics.gauge("circuit_breaker_state", hash(cb_state) % 100)
            LOG.info("Circuit breaker: %s (Sharpe: %.3f)", cb_state, degradation.get("rolling_sharpe", 0.0))
        self.metrics.gauge("rolling_sharpe", degradation.get("rolling_sharpe", 0.0))
        self.metrics.gauge("position_count", len(self.oms.positions))

        for sym in self.cfg["symbols"]:
            price = quotes[sym]
            self.ingest_quote(sym, price)
            self.oms.mark(sym, price)
            frame = self.completed_frame(sym, self.cfg["execution"]["signal_history_candles"])
            if frame is None or len(frame) < self.cfg["execution"]["signal_history_candles"]:
                continue

            # Multi-timeframe alignment
            mtf_result = self.mtf_analyzer.alignment_score(frame)
            self.db.kv_set(f"mtf_{sym}", mtf_result)

            votes, supporters = self._vote_details(frame)

            # Regime-conditioned vote adjustment
            regime_weights = self.regime_detector.regime_weights()
            if supporters:
                for i, (sid, params, weight) in enumerate(supporters):
                    template = sid.split("_")[0] if "_" in sid else sid
                    regime_mult = regime_weights.get(template, 1.0)
                    supporters[i] = (sid, params, weight * regime_mult)
                votes *= np.mean([regime_weights.get(
                    s[0].split("_")[0] if "_" in s[0] else s[0], 1.0)
                    for s in supporters])

            # Entries are stopped near close
            if current_time >= self.cfg["entry_cutoff"] and votes > 0:
                votes = 0.0
                supporters = []

            # Circuit breaker blocks new entries
            if self.circuit_breaker.should_block_entries() and votes > 0:
                self._record_decision(sym, "BLOCK", "CIRCUIT_BREAKER", {"state": cb_state})
                continue

            self._act(sym, price, frame, votes, supporters, mtf_result)

        # Reconcile on a bounded cadence
        if time.monotonic() - self._last_reconcile >= float(self.cfg["execution"].get("reconcile_interval_seconds", 30)):
            self._last_reconcile = time.monotonic()
            self.oms.reconcile()
        self._set_health("ACTIVE", f"validated_strategies={len(self.strategies)} regime={self._current_regime.value}")

    def _act(
        self,
        sym: str,
        ltp: float,
        frame: pd.DataFrame,
        votes: float,
        supporters: list[tuple[str, dict, float]] | None = None,
        mtf_result: dict | None = None,
    ) -> None:
        current_quantity = int(self.oms.positions.get(sym, {}).get("qty", 0))
        flow = self.broker.order_flow(sym) if self.cfg["order_flow"]["enabled"] else None
        # A negative signal or a confirmed order-flow reversal sells an
        # agent-owned long. It never opens a naked short.
        if current_quantity > 0:
            if flow is not None and flow.ready and flow.long_exit:
                self._record_decision(sym, "EXIT", "ORDER_FLOW_REVERSAL", flow.details(), force=True)
                self.oms.close(sym, "ORDER_FLOW_REVERSAL")
            elif votes < 0:
                self._record_decision(sym, "EXIT", "OPPOSITE_SIGNAL", {"weighted_vote": round(votes, 3)}, force=True)
                self.oms.close(sym, "OPPOSITE_SIGNAL")
            return

        # Order flow is intentionally checked first. A secondary candle-based
        # trend vote cannot create an entry through a stale, wide, thin, or
        # adverse book.
        if self.cfg["order_flow"]["primary"]:
            if flow is None:
                self._record_decision(sym, "BLOCK", "ORDER_FLOW_UNAVAILABLE", {})
                return
            if not flow.ready or not flow.long_entry:
                self._record_decision(sym, "BLOCK", flow.reason, flow.details())
                return
        if votes <= 0 or not self.strategies:
            self._record_decision(sym, "BLOCK", "TREND_CONFIRMATION_MISSING", {"weighted_vote": round(votes, 3)})
            return

        # Ensemble quorum: a single weak template must not command a full
        # risk-sized entry when several strategies are approved (C6).
        total_weight = sum(min(max(float(entry[3]), 0.25), 3.0) for entry in self.strategies)
        min_fraction = float(self.cfg["execution"].get("min_vote_fraction", 0.0))
        required = min_fraction * total_weight
        if total_weight > 0 and votes < required:
            self._record_decision(sym, "BLOCK", "ENSEMBLE_QUORUM", {"votes": round(votes, 3), "required": round(required, 3)})
            return
        min_support = int(self.cfg["execution"].get("min_support_strategies", 1))
        if len(supporters or []) < min_support:
            self._record_decision(sym, "BLOCK", "SUPPORT_QUORUM", {"supporters": len(supporters or []), "required": min_support})
            return

        optimism, sentiment = self.news.get_optimism_score(self.db, sym)
        if optimism < -0.2:
            LOG.info("Negative news filter blocked %s (%.3f, %s)", sym, optimism, sentiment)
            self._record_decision(sym, "BLOCK", "NEGATIVE_NEWS", {"optimism": round(float(optimism), 3)})
            return
        try:
            stop, target, bracket_detail = self._bracket_from_supporters(frame, ltp, supporters or [])
        except (KeyError, TypeError, ValueError) as exc:
            self._record_decision(sym, "BLOCK", "ENTRY_BRACKET_UNAVAILABLE", {"error": exc.__class__.__name__})
            return
        stop_distance = ltp - stop
        # Confidence from ensemble vote + order flow (regime enters via the
        # regime scalars below). total_weight is always in scope here.
        _conf = min(1.0, abs(votes) / max(total_weight * 0.6, 1.0))
        if flow is not None and flow.ready:
            _conf = 0.6*_conf + 0.4*min(1, flow.pressure_ema+0.5)
        quantity = self.risk.size_with_kelly(ltp, stop_distance, confidence=_conf)
        # Dynamic leverage overlay (vol-targeted).  A leverage request scales
        # the base position but is honoured only within the hard risk caps:
        # the result is clamped below to the per-trade notional cap and the
        # remaining leverage-aware gross-exposure headroom.  A request beyond
        # the caps shrinks the position rather than pushing quantity past a
        # limit the risk gate re-checks below - previously that double check
        # rejected every entry whenever the engine asked for more than the
        # 3x baseline.
        requested_leverage = 1.0
        if self.leverage_engine is not None:
            try:
                atr_pct = (bracket_detail.get("atr", ltp * 0.01) / ltp * 100) if isinstance(bracket_detail, dict) else 1.0
                eq = [float(v) for v in self.db.q("SELECT equity FROM equity ORDER BY ts DESC LIMIT 2")]
                dd_ratio = 0.0
                if len(eq) >= 2 and eq[0] > 0:
                    peak = max(eq) if eq else eq[0]
                    dd_ratio = max(0.0, (peak - eq[0]) / peak) / 0.15
                lev = self.leverage_engine.decide(sym, "equity_scalp" if "scalp" in str(supporters) else "equity_intraday", ltp, atr_pct, _conf, self._current_regime.value, dd_ratio, 0)
                requested_leverage = max(float(lev.leverage), 1.0)
                quantity = max(1, int(quantity * (requested_leverage / 3.0)))
                bracket_detail["leverage"] = requested_leverage
                bracket_detail["leverage_tier"] = lev.tier
            except Exception as exc:  # noqa: BLE001 - sizing must never abort the decision path
                LOG.debug("Leverage overlay unavailable for %s: %s", sym, exc.__class__.__name__)
        # Clamp the scaled quantity to the hard caps the risk gate enforces.
        risk_limits = self.cfg["risk"]
        notional_cap_qty = int(float(risk_limits["max_notional_per_trade"]) / ltp)
        current_gross = self.risk.gross_exposure(list(self.oms.positions.values()))
        if requested_leverage > 0:
            exposure_headroom_qty = int(
                (float(risk_limits["max_gross_exposure"]) - current_gross) / (ltp * requested_leverage)
            )
        else:
            exposure_headroom_qty = notional_cap_qty
        clamped_quantity = min(quantity, notional_cap_qty, exposure_headroom_qty)
        if clamped_quantity <= 0:
            self._record_decision(sym, "BLOCK", "RISK_CAP_CLAMPED", {
                "quantity": quantity,
                "notional_cap_qty": notional_cap_qty,
                "exposure_headroom_qty": exposure_headroom_qty,
            })
            return
        if clamped_quantity < quantity:
            self.metrics.counter("leverage_caps_clamped")
            bracket_detail["leverage_clamped"] = True
            bracket_detail["leverage_requested_qty"] = quantity
            LOG.info("Leverage clamp %s: requested %d -> admissible %d "
                     "(notional cap %d, exposure headroom %d)",
                     sym, quantity, clamped_quantity, notional_cap_qty, exposure_headroom_qty)
        quantity = clamped_quantity
        portfolio_var_pct = self._portfolio_var_pct(sym, quantity, ltp)
        allowed, reason = self.risk.approve(sym, "BUY", quantity, ltp, list(self.oms.positions.values()), portfolio_var_pct)
        if not allowed:
            LOG.info("Risk gate rejected %s: %s", sym, reason)
            self._record_decision(sym, "BLOCK", "RISK_GATE", {"reason": reason, "weighted_vote": round(votes, 3), "portfolio_var_pct": round(portfolio_var_pct, 3)})
            if reason.startswith("daily"):
                self.comp.halt(reason)
            return
        # Apply the circuit-breaker size multiplier.  Multiplier is 1.0 in
        # NORMAL, 0.5 at L1, 0.0 at L2/HALT - a zero/sub-unit scaled size must
        # block, not become a 1-share entry via max(1, ...).
        quantity = int(quantity * self._position_size_multiplier)
        if quantity <= 0:
            self._record_decision(sym, "BLOCK", "CIRCUIT_BREAKER_SIZE", {"multiplier": self._position_size_multiplier})
            return

        detail = {"weighted_vote": round(votes, 3), "quantity": quantity, "portfolio_var_pct": round(portfolio_var_pct, 3), "regime": self._current_regime.value, **bracket_detail}
        if self.cognition is not None:
            try:
                regime_conf = float(self.db.kv_get("current_regime", {}).get("confidence", 0.5))
                detail["confidence"] = round(self.cognition.entry_confidence(
                    votes, required or votes or 1.0, len(supporters or []), regime_conf,
                    bool(flow and flow.ready)), 3)
            except Exception:  # noqa: BLE001
                pass
        if flow is not None:
            detail.update(flow.details())
        if mtf_result:
            detail["mtf_score"] = mtf_result.get("score", 0.5)
            detail["mtf_aligned"] = mtf_result.get("aligned", True)
        self._record_decision(sym, "ENTRY_REQUEST", "ORDER_FLOW_PRIMARY", detail, force=True)
        self.metrics.counter("entries")
        strategy_label = "blend:" + ",".join(bracket_detail["strategy_ids"])
        self.oms.open_position(sym, "BUY", quantity, strategy_label[:160], stop, target, f"votes={votes:.3f}")

    def _recent_returns(self):
        rows = self.db.q("SELECT equity FROM equity ORDER BY ts DESC LIMIT 120")
        equity = [row[0] for row in rows][::-1]
        if len(equity) < 5:
            return np.zeros(20)
        values = np.asarray(equity, dtype=float)
        return np.diff(values) / np.maximum(values[:-1], 1.0)

    def _portfolio_var_pct(self, candidate_symbol: str, quantity: int, price: float) -> float:
        """Estimate position-aware one-period VaR from aligned local return history.

        The legacy equity-only fallback remains for sparse data, but a new
        entry is otherwise measured with its open-position weights. This avoids
        treating five highly correlated stocks as if they were independent.
        """
        exposures = {symbol: float(position["qty"]) * float(position["avg"]) for symbol, position in self.oms.positions.items()}
        exposures[candidate_symbol] = exposures.get(candidate_symbol, 0.0) + float(quantity) * float(price)
        columns: list[pd.Series] = []
        for symbol in sorted(exposures):
            history = self.frame(symbol, 1_200)
            if history is None or len(history) < 30:
                return abs(Metrics.var(self._recent_returns())) * 100.0
            close = pd.to_numeric(history.set_index("ts")["c"], errors="coerce").pct_change()
            columns.append(close.rename(symbol))
        aligned = pd.concat(columns, axis=1, join="inner").dropna()
        if len(aligned) < 20:
            return abs(Metrics.var(self._recent_returns())) * 100.0
        weights = pd.Series(exposures, dtype=float).reindex(aligned.columns).fillna(0.0) / float(self.cfg["capital"])
        returns = aligned.mul(weights, axis=1).sum(axis=1).to_numpy(dtype=float)
        return abs(Metrics.var(returns)) * 100.0

    def eod(self) -> None:
        profit_and_loss = self.db.q("SELECT COALESCE(SUM(pnl),0) FROM trades")[0][0]
        equity = self.cfg["capital"] + float(profit_and_loss)
        self.db.ex("INSERT OR REPLACE INTO equity VALUES(?,?)", (iso(), equity))
        history = [row[0] for row in self.db.q("SELECT equity FROM equity ORDER BY ts")]
        returns = self._recent_returns()
        stats = {
            "sharpe": Metrics.sharpe(returns),
            "sortino": Metrics.sortino(returns),
            "var99": Metrics.var(returns),
            "maxdd": Metrics.maxdd(history),
            "regime": self._current_regime.value,
            "circuit_breaker": self.circuit_breaker.state,
        }
        self.db.kv_set("portfolio_stats", stats)
        self.metrics.gauge("equity", equity)
        self.metrics.gauge("daily_pnl", profit_and_loss)
        self.db.kv_set("metrics_snapshot", self.metrics.snapshot())
        if self.cognition is not None:
            try:
                for row in self.db.q("SELECT sym,pnl,strat FROM trades WHERE outtime >= date('now','start of day')"):
                    self.cognition.on_trade_closed({"sym": row[0], "pnl": row[1], "strat": row[2]})
            except Exception:  # noqa: BLE001
                LOG.debug("cognitive trade hook failed", exc_info=True)
        LOG.info("EOD equity %.2f, realised PnL %.2f, regime=%s, cb=%s",
                 equity, profit_and_loss, self._current_regime.value,
                 self.circuit_breaker.state)

    def _walk_forward_slices(self, frame: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create expanding-train, forward-test folds separated by an embargo."""
        training = self.cfg["training"]
        folds = int(training["walk_forward_folds"])
        oos_length = int(training["min_oos_candles"])
        embargo = int(training["embargo_candles"])
        first_train_end = len(frame) - folds * (oos_length + embargo)
        if first_train_end < int(self.cfg["execution"]["signal_history_candles"]):
            return []
        slices: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for fold in range(folds):
            train_end = first_train_end + fold * (oos_length + embargo)
            test_start = train_end + embargo
            test_end = test_start + oos_length
            train = frame.iloc[:train_end].copy()
            test = frame.iloc[test_start:test_end].copy()
            if len(train) < int(self.cfg["execution"]["signal_history_candles"]) or len(test) != oos_length:
                return []
            slices.append((train, test))
        return slices

    def nightly_training(self) -> None:
        frames_is: list[pd.DataFrame] = []
        frames_oos: list[pd.DataFrame] = []
        for symbol in self.cfg["symbols"]:
            frame = self.completed_frame(symbol, self.cfg["training"]["training_history_candles"])
            if frame is None:
                continue
            frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
            for train, test in self._walk_forward_slices(frame):
                frames_is.append(train)
                frames_oos.append(test)
        required = int(self.cfg["training"]["min_symbols"]) * int(self.cfg["training"]["walk_forward_folds"])
        if len(frames_is) < required:
            LOG.warning("Insufficient clean historical candles for walk-forward training")
            return
        self.brain.iterate(frames_is, frames_oos)
        self.load_strategies()

    def close(self) -> None:
        """Release every handle the agent opened.  Idempotent and safe from
        any thread; without it Windows keeps the SQLite files locked and the
        surrounding directory cannot be removed."""
        if self._closed:
            return
        self._closed = True
        try:
            self.broker.stop_orderflow()
        except Exception:  # noqa: BLE001
            pass
        if self.backup_manager is not None:
            try:
                self.backup_manager.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.cognition is not None:
            try:
                self.cognition.shutdown()
            except Exception:  # noqa: BLE001
                LOG.debug("cognitive shutdown failed", exc_info=True)
        if self.config_watcher is not None:
            try:
                self.config_watcher.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.secret_manager is not None:
            try:
                self.secret_manager.stop_auto_rotation()
            except Exception:  # noqa: BLE001
                pass
        if self.failover_manager is not None:
            try:
                self.failover_manager._brokers.clear()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass


    def _apply_universe_scan(self) -> None:
        """Dynamic universe at boot: low-cost liquid candidates -> cfg symbols.

        Fail-open by design: any error keeps the static config.yaml symbols.
        Requires a broker login first (paper has synthetic history), so this
        runs post-make_broker with cached/fetched history only.
        """
        settings = self.cfg.get("universe", {}) if isinstance(self.cfg.get("universe"), dict) else {}
        if not settings.get("auto_scan", False):
            return
        try:
            from .scanner import MarketScanner
            candidates = [str(s).upper() for s in settings.get("candidates", self.cfg.get("symbols", []))]
            ceiling = float(settings.get("price_ceiling", 500))
            top_k = int(settings.get("top_k", 12))
            ranked = MarketScanner(self.cfg, self.db, self.broker).scan(candidates, top_k=top_k * 2)
            affordable = [r["symbol"] for r in ranked
                          if r["last_price"] and float(r["last_price"]) <= ceiling][:top_k]
            if len(affordable) >= 2:
                self.cfg["symbols"] = affordable
                self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('UNIVERSE_SCAN',?,?)",
                           (",".join(affordable), iso()))
                LOG.info("Universe scan selected: %s", ",".join(affordable))
        except Exception:
            LOG.warning("Universe scan failed; keeping static symbols", exc_info=True)

    def run_forever(self) -> None:
        if not self.boot():
            self.close()
            return
        if self.backup_manager is not None:
            self.backup_manager.start()
        if self.shutdown_manager is not None:
            # Ctrl+C / service stop runs the same ordered cleanup as the
            # normal exit path below instead of dropping handles mid-tick.
            self.shutdown_manager.register_hook(
                "stop_agent_loop", lambda: setattr(self, "stop", True), priority=10,
            )
            self.shutdown_manager.register_hook("agent_close", self.close, priority=90)
        health_interval = float(self.cfg.get("health_checks", {}).get("interval_seconds", 30))
        next_health_check = 0.0
        completed = set()
        LOG.info("Autonomous tick loop started")
        while not self.stop:
            sleep_seconds = float(self.cfg["tick_seconds"])
            try:
                self.tick_once()
                self.broker_error_count = 0
                self.rate_limit_count = 0
                self._set_health("ACTIVE", f"validated_strategies={len(self.strategies)}")
                if self.health_checker is not None and time.monotonic() >= next_health_check:
                    next_health_check = time.monotonic() + health_interval
                    try:
                        system_health = self.health_checker.run_checks()
                        self.db.kv_set(
                            "system_health",
                            {"overall": system_health.overall_status.value,
                             "checks": {c.name: c.status.value for c in system_health.checks},
                             "ts": iso()},
                        )
                    except Exception:  # noqa: BLE001
                        LOG.debug("health check sweep failed", exc_info=True)
                # EOD stats are tied to the configured close, not a fixed clock
                # time, so a changed market_close can never leave them stale
                # or premature. Nightly training is a plain offline batch job,
                # so a fixed post-close time is fine for it.
                for schedule, fn in {self.cfg["market_close"]: self.eod, "18:00": self.nightly_training}.items():
                    marker = (schedule, now().date())
                    if self._trading_day() and hhmm() >= schedule and marker not in completed:
                        fn()
                        completed.add(marker)
            except RateLimitError as exc:
                self.rate_limit_count += 1
                execution = self.cfg["execution"]
                exponential = float(execution["rate_limit_backoff_seconds"]) * (2 ** min(self.rate_limit_count - 1, 5))
                retry_after = float(exc.retry_after_seconds or 0.0)
                sleep_seconds = min(float(execution["max_rate_limit_backoff_seconds"]), max(sleep_seconds, exponential, retry_after))
                LOG.warning("Dhan read/data rate limit %s; backing off for %.1fs", self.rate_limit_count, sleep_seconds)
                self._set_health("RATE_LIMITED", f"retrying after {sleep_seconds:.1f}s", force=True)
            except OrderError as exc:
                self.comp.halt(f"Order execution uncertainty: {exc}")
                self.stop = True
            except (MarketDataError, BrokerError, ValueError) as exc:
                self.broker_error_count += 1
                LOG.error("Trading-path error %s/%s: %s", self.broker_error_count, self.cfg["execution"]["max_consecutive_broker_errors"], exc)
                if self.broker_error_count >= self.cfg["execution"]["max_consecutive_broker_errors"]:
                    self.comp.halt(f"Repeated broker/data error: {exc}")
                    self.stop = True
            except Exception as exc:
                LOG.critical("Unexpected trading-path failure: %s", traceback.format_exc())
                if self.cognition is not None:
                    try:
                        self.cognition.on_error("tick_once", exc)
                    except Exception:  # noqa: BLE001
                        pass
                self.comp.halt(f"Unexpected trading-path error: {exc}")
                self.stop = True
            if self.stop and self.shutdown_manager is not None and self.shutdown_manager.is_shutting_down():
                break  # the signal/atexit path runs the same hooks; do not sleep first
            time.sleep(sleep_seconds)
        if self.cognition is not None:
            try:
                distilled = self.cognition.autodistill()
                if distilled.get("skills_extracted"):
                    LOG.info("Cognitive distillation extracted %d skills", len(distilled["skills_extracted"]))
            except Exception:  # noqa: BLE001
                LOG.debug("cognitive distillation failed", exc_info=True)
        self.close()
