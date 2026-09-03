"""10x ML Pipeline: Advanced feature engineering and ensemble learning."""
from __future__ import annotations
import threading
from typing import Dict, List
import numpy as np
import pandas as pd
from .core import LOG, iso
from .features import REG


class FeatureEngineer:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.feature_names = []

    def generate_features(self, df, include_labels=False):
        if df is None or len(df) < 50:
            return pd.DataFrame()
        features = pd.DataFrame(index=df.index)
        o = df["o"].to_numpy(dtype=float)
        h = df["h"].to_numpy(dtype=float)
        l = df["l"].to_numpy(dtype=float)
        c = df["c"].to_numpy(dtype=float)
        v = df["v"].to_numpy(dtype=float)
        features = self._momentum(features, c, o, h, l)
        features = self._volatility(features, h, l, c, o)
        features = self._volume_feat(features, v, c, h, l)
        features = self._statistical(features, c)
        features = self._patterns(features, o, h, l, c)
        if include_labels:
            features = self._labels(features, c)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = list(features.columns)
        return features

    def _momentum(self, f, c, o, h, l):
        for p in [1, 3, 5, 10, 20, 50]:
            if len(c) > p:
                roc = np.zeros(len(c))
                roc[p:] = c[p:] / np.maximum(c[:-p], 1e-9) - 1.0
                f["roc_" + str(p)] = roc
        for p in [10, 20, 50]:
            sma = pd.Series(c).rolling(p, min_periods=1).mean().values
            f["psma_" + str(p)] = (c - sma) / np.maximum(sma, 1e-9)
        for fs, sl in [(5, 20), (9, 21), (12, 26)]:
            ef = pd.Series(c).ewm(span=fs).mean().values
            es = pd.Series(c).ewm(span=sl).mean().values
            f["emax_" + str(fs) + "_" + str(sl)] = (ef - es) / np.maximum(es, 1e-9)
        ml, sl = REG["macd"](c)
        f["macd_l"] = ml / np.maximum(np.abs(c), 1e-9)
        f["macd_s"] = sl / np.maximum(np.abs(c), 1e-9)
        f["macd_h"] = (ml - sl) / np.maximum(np.abs(c), 1e-9)
        for p in [7, 14, 21]:
            rsi = REG["rsi"](c, p)
            f["rsi_" + str(p)] = (rsi - 50) / 50.0
        tp = (h + l + c) / 3.0
        sma_tp = pd.Series(tp).rolling(20).mean().values
        f["cci"] = (tp - sma_tp) / np.maximum(
            0.015 * pd.Series(tp).rolling(20).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            ).values, 1e-9)
        pc = np.diff(c, prepend=c[0])
        dsp = pd.Series(pd.Series(pc).ewm(span=25).mean()).ewm(span=13).mean().values
        dsa = pd.Series(pd.Series(np.abs(pc)).ewm(span=25).mean()).ewm(span=13).mean().values
        f["tsi"] = dsp / np.maximum(dsa, 1e-9) * 100
        return f

    def _volatility(self, f, h, l, c, o):
        for p in [7, 14, 21]:
            atr = REG["atr"](h, l, c, p)
            f["atrp_" + str(p)] = atr / np.maximum(c, 1e-9) * 100
        ret = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(max(c[0], 1e-9)))
        for p in [10, 20, 50]:
            f["hvol_" + str(p)] = pd.Series(ret).rolling(p).std().values * np.sqrt(252 * 375)
        log_hl = np.log(np.maximum(h, 1e-9) / np.maximum(l, 1e-9))
        f["park"] = np.sqrt(pd.Series(log_hl ** 2).rolling(20).mean().values / (4 * np.log(2)))
        f["gkv"] = REG["garman_klass"](h, l, c, o, window=20)
        f["vr"] = pd.Series(ret).rolling(5).std().values / np.maximum(
            pd.Series(ret).rolling(20).std().values, 1e-9)
        sma20 = pd.Series(c).rolling(20).mean().values
        std20 = pd.Series(c).rolling(20).std().values
        f["bbw"] = 2 * std20 / np.maximum(sma20, 1e-9)
        f["bbp"] = (c - (sma20 - 2 * std20)) / np.maximum(4 * std20, 1e-9)
        f["irange"] = (h - l) / np.maximum(c, 1e-9)
        f["rexp"] = f["irange"] / np.maximum(
            pd.Series(f["irange"]).rolling(20).mean().values, 1e-9)
        f["gap"] = np.zeros(len(c))
        if len(c) > 1:
            f["gap"][1:] = (o[1:] - c[:-1]) / np.maximum(c[:-1], 1e-9)
        return f

    def _volume_feat(self, f, v, c, h, l):
        for p in [10, 20, 50]:
            vma = pd.Series(v).rolling(p, min_periods=1).mean().values
            f["vrp_" + str(p)] = v / np.maximum(vma, 1.0)
        obv = np.where(np.diff(c, prepend=c[0]) > 0, v, -v)
        f["obv"] = np.cumsum(obv) / np.maximum(np.cumsum(np.abs(obv)), 1.0)
        f["vpct"] = pd.Series(v).rolling(100).rank(pct=True).values
        f["bp"] = (c - l) / np.maximum(h - l, 1e-9)
        f["sp"] = (h - c) / np.maximum(h - l, 1e-9)
        return f

    def _statistical(self, f, c):
        ret = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(max(c[0], 1e-9)))
        for p in [20, 50]:
            f["skew_" + str(p)] = pd.Series(ret).rolling(p).skew().values
        f["zs"] = (ret - pd.Series(ret).rolling(20).mean().values) / np.maximum(
            pd.Series(ret).rolling(20).std().values, 1e-9)
        f["pr"] = pd.Series(c).rolling(50).rank(pct=True).values
        for p in [20, 50]:
            m = pd.Series(c).rolling(p).mean().values
            s = pd.Series(c).rolling(p).std().values
            f["dm_" + str(p)] = (c - m) / np.maximum(s, 1e-9)
        return f

    def _patterns(self, f, o, h, l, c):
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-9)
        f["br"] = body / rng
        f["us"] = (h - np.maximum(o, c)) / rng
        f["ls"] = (np.minimum(o, c) - l) / rng
        f["doji"] = (body / rng < 0.1).astype(float)
        d = np.sign(c - o)
        cc = np.zeros(len(c))
        for i in range(1, len(c)):
            if d[i] == d[i - 1] and d[i] != 0:
                cc[i] = cc[i - 1] + 1
        f["cc"] = cc
        return f

    def _labels(self, f, c):
        for h in [1, 3, 5, 10, 20]:
            fwd = np.roll(c, -h) / np.maximum(c, 1e-9) - 1.0
            if h < len(fwd):
                fwd[-h:] = np.nan
            f["fwd_" + str(h)] = fwd
        f["fu5"] = (f["fwd_5"] > 0).astype(float)
        return f


class EnsembleMetaLearner:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.perf = {}
        self.weights = {}
        self.decay = float(self.cfg.get("ensemble_decay", 0.95))
        self.min_w = float(self.cfg.get("ensemble_min_weight", 0.1))
        self.max_w = float(self.cfg.get("ensemble_max_weight", 3.0))
        self._lock = threading.RLock()

    def update_performance(self, sid, pnl):
        with self._lock:
            self.perf.setdefault(sid, []).append(pnl)
            if len(self.perf[sid]) > 100:
                self.perf[sid] = self.perf[sid][-100:]
            self._update_weight(sid)

    def _update_weight(self, sid):
        p = self.perf.get(sid, [])
        if len(p) < 5:
            self.weights[sid] = 1.0
            return
        w = np.array([self.decay ** i for i in range(len(p) - 1, -1, -1)])
        r = np.array(p)
        wm = np.average(r, weights=w)
        wv = np.average((r - wm) ** 2, weights=w)
        score = wm / np.sqrt(wv) * np.sqrt(252) if wv > 0 else 0.0
        self.weights[sid] = float(np.clip(1.0 + score * 0.5, self.min_w, self.max_w))

    def get_weights(self):
        with self._lock:
            return dict(self.weights)

    def combine_signals(self, signals):
        with self._lock:
            if not signals:
                return 0.0
            tw = 0.0
            ws = 0.0
            for sid, sig in signals.items():
                w = self.weights.get(sid, 1.0)
                ws += sig * w
                tw += abs(w)
            return ws / max(tw, 1e-9)


class OnlineLearner:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.lookback = int(self.cfg.get("online_lookback", 100))
        self.regime_perf = {}
        self._lock = threading.RLock()

    def record_trade(self, sid, regime, pnl):
        with self._lock:
            self.regime_perf.setdefault(regime, {}).setdefault(sid, []).append(pnl)
            p = self.regime_perf[regime][sid]
            if len(p) > self.lookback:
                self.regime_perf[regime][sid] = p[-self.lookback:]

    def get_regime_weights(self, regime):
        with self._lock:
            if regime not in self.regime_perf:
                return {}
            w = {}
            for sid, pnls in self.regime_perf[regime].items():
                if len(pnls) < 5:
                    w[sid] = 1.0
                    continue
                r = np.array(pnls[-50:])
                d = np.array([0.95 ** i for i in range(len(r) - 1, -1, -1)])
                wm = np.average(r, weights=d)
                wv = np.average((r - wm) ** 2, weights=d)
                sh = wm / np.sqrt(wv) * np.sqrt(252) if wv > 0 else 0.0
                w[sid] = float(np.clip(1.0 + sh * 0.3, 0.5, 3.0))
            return w

    def get_confidence(self, sid, regime):
        with self._lock:
            pnls = self.regime_perf.get(regime, {}).get(sid, [])
            if len(pnls) < 10:
                return 0.5
            r = np.array(pnls[-20:])
            wr = np.mean(r > 0)
            cons = 1.0 - min(np.std(r) / max(np.abs(np.mean(r)), 1e-9), 1.0)
            sf = min(len(pnls) / 50, 1.0)
            return float(np.clip(0.4 * wr + 0.3 * cons + 0.3 * sf, 0.0, 1.0))


__all__ = ["FeatureEngineer", "EnsembleMetaLearner", "OnlineLearner"]
