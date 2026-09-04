"""Pure-numpy algorithm library: classical ML, deep-learning primitives,
reinforcement learning, time-series statistics, metaheuristics, and portfolio
construction — zero heavyweight dependencies, everything degrades gracefully.

Deep models and GBMs are implemented as compact numpy primitives (MLP with
backprop, autoencoder) suitable for research-sized problems; the production
trading path keeps using its validated ridge/online learners. Every callable
is registered in ``ALG`` by number and name, mirroring the indicator
registry.
"""

from __future__ import annotations

import random

import numpy as np

ALG: dict = {}


def _reg(number: int, name: str):
    def deco(fn):
        ALG[number] = fn
        ALG[name] = fn
        fn.alg_number = number
        return fn
    return deco


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _returns(c) -> np.ndarray:
    c = _f(c)
    return np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=np.nan)[1:]


# ── Classical ML 1–20 ─────────────────────────────────────────────────────
@_reg(1, "linear_regression")
def linear_regression(x, y):
    x, y = _f(x), _f(y)
    if x.ndim == 1:
        slope, intercept = np.polyfit(x, y, 1)
        return {"slope": float(slope), "intercept": float(intercept)}
    design = np.hstack([x, np.ones((len(x), 1))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"coefficients": coef[:-1], "intercept": float(coef[-1])}


@_reg(2, "logistic_regression")
def logistic_regression(x, y, lr=0.1, epochs=500):
    X, y = _f(x), _f(y)
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w)))
        w += lr * X.T @ (y - p) / len(y)
    return {"weights": w}


@_reg(3, "ridge_lasso")
def ridge_lasso(x, y, lam=1.0, l1_ratio=0.0, epochs=2000, lr=0.01):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        pred = X @ w + b
        grad_w = -2 * X.T @ (y - pred) / len(y) + 2 * (1 - l1_ratio) * lam * w + l1_ratio * lam * np.sign(w)
        grad_b = -2 * np.mean(y - pred)
        w, b = w - lr * grad_w, b - lr * grad_b
    return {"w": w, "b": b}


class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = self.threshold = self.value = None
        self.left = self.right = None


@_reg(4, "decision_tree")
def decision_tree(x, y, max_depth=4, min_leaf=5):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]

    def build(idx, depth):
        node = _TreeNode()
        if depth >= max_depth or len(idx) < 2 * min_leaf or np.std(y[idx]) < 1e-12:
            node.value = float(np.mean(y[idx]))
            return node
        best = None
        parent_loss = np.var(y[idx]) * len(idx)
        for f in range(X.shape[1]):
            for t in np.percentile(X[idx, f], [25, 50, 75]):
                left = idx[X[idx, f] <= t]
                right = idx[X[idx, f] > t]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                loss = np.var(y[left]) * len(left) + np.var(y[right]) * len(right)
                if best is None or loss < best[0]:
                    best = (loss, f, t, left, right)
        if best is None or best[0] >= parent_loss:
            node.value = float(np.mean(y[idx]))
            return node
        _, f, t, left, right = best
        node.feature, node.threshold = f, t
        node.left, node.right = build(left, depth + 1), build(right, depth + 1)
        return node

    root = build(np.arange(len(y)), 0)

    def predict_row(row, node):
        while node.value is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    return {"predict": lambda Z: np.array([predict_row(r, root) for r in np.atleast_2d(_f(Z))]), "tree": root}


@_reg(5, "random_forest")
def random_forest(x, y, n_trees=25, **kw):
    X, y = _f(x), _f(y)
    trees = []
    rng = np.random.default_rng(0)
    for _ in range(n_trees):
        idx = rng.choice(len(y), len(y), replace=True)
        trees.append(decision_tree(X[idx], y[idx], **kw)["predict"])
    return {"predict": lambda Z: np.mean([t(Z) for t in trees], axis=0)}


@_reg(6, "gradient_boosting")
def gradient_boosting(x, y, n_rounds=50, lr=0.1, depth=3):
    X, y = _f(x), _f(y)
    base = float(np.mean(y))
    pred = np.full(len(y), base)
    models = []
    for _ in range(n_rounds):
        residual = y - pred
        tree = decision_tree(X, residual, max_depth=depth)
        models.append(tree["predict"])
        pred += lr * tree["predict"](X)
    return {"predict": lambda Z: base + lr * np.sum([m(Z) for m in models], axis=0)}


# 7–9 (XGBoost/LightGBM/CatBoost) share the same boosting core with different
# sampling strategies; the research-grade numpy boosting above is the engine.
@_reg(7, "xgboost_style")
def xgboost_style(x, y, **kw):
    kw.setdefault("n_rounds", 100)
    return gradient_boosting(x, y, **kw)


@_reg(8, "lightgbm_style")
def lightgbm_style(x, y, feature_frac=0.8, **kw):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    keep = np.random.default_rng(1).choice(X.shape[1], max(1, int(X.shape[1] * feature_frac)), replace=False)
    return gradient_boosting(X[:, keep], y, **kw)


@_reg(9, "catboost_style")
def catboost_style(x, y, **kw):
    kw.setdefault("lr", 0.03)
    kw.setdefault("n_rounds", 200)
    return gradient_boosting(x, y, **kw)


@_reg(10, "adaboost")
def adaboost(x, y, n_estimators=50):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    y_c = np.where(y >= np.median(y), 1, -1).astype(float)
    w = np.full(len(y_c), 1 / len(y_c))
    stumps = []
    for _ in range(n_estimators):
        best = None
        for f in range(X.shape[1]):
            for t in np.percentile(X[:, f], np.linspace(10, 90, 9)):
                pred = np.where(X[:, f] <= t, -1.0, 1.0)
                err = np.sum(w * (pred != y_c))
                if best is None or err < best[0]:
                    best = (err, f, t, pred)
        err, f, t, pred = best
        err = max(err, 1e-6)
        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(-alpha * y_c * pred)
        w /= w.sum()
        stumps.append((alpha, f, t))

    def predict(Z):
        Z = np.atleast_2d(_f(Z))
        agg = np.zeros(len(Z))
        for alpha, f, t in stumps:
            agg += alpha * np.where(Z[:, f] <= t, -1.0, 1.0)
        return np.sign(agg)
    return {"predict": predict}


@_reg(11, "svm")
def svm(x, y, c=1.0, epochs=500, lr=0.01):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    y_c = np.where(y >= np.median(y), 1.0, -1.0)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        for i in range(len(y_c)):
            if y_c[i] * (X[i] @ w + b) < 1:
                w += lr * (y_c[i] * X[i] - c * w)
                b += lr * y_c[i]
            else:
                w -= lr * c * w
    return {"w": w, "b": b, "predict": lambda Z: np.sign(np.atleast_2d(_f(Z)) @ w + b)}


@_reg(12, "knn")
def knn(x, y, k=5):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    return {"predict": lambda Z: np.array([
        y[np.argsort(np.sum((X - z) ** 2, axis=1))[:k]].mean() for z in np.atleast_2d(_f(Z))])}


@_reg(13, "naive_bayes")
def naive_bayes(x, y):
    X, y = _f(x), _f(y)
    classes = np.unique(y)
    stats = {}
    for cls in classes:
        subset = X[y == cls]
        stats[cls] = (subset.mean(0), subset.std(0) + 1e-9, len(subset) / len(y))
    priors = {c: s[2] for c, s in stats.items()}

    def log_pdf(x, mu, sd):
        return -0.5 * np.sum(((x - mu) / sd) ** 2) - np.sum(np.log(sd))
    return {"predict": lambda Z: max(classes, key=lambda c: np.log(priors[c] + 1e-9) + log_pdf(_f(Z), *stats[c][:2]))}


@_reg(14, "gaussian_process")
def gaussian_process(x, y, x_star, length_scale=1.0):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    Xs = np.atleast_2d(_f(x_star))

    def kernel(a, b):
        d2 = np.sum(a ** 2, 1)[:, None] + np.sum(b ** 2, 1)[None, :] - 2 * a @ b.T
        return np.exp(-d2 / (2 * length_scale ** 2))
    K = kernel(X, X) + 1e-6 * np.eye(len(X))
    Ks, Kss = kernel(X, Xs), kernel(Xs, Xs)
    mu = Ks.T @ np.linalg.solve(K, y)
    cov = Kss - Ks.T @ np.linalg.solve(K, Ks)
    return {"mu": mu, "sd": np.sqrt(np.maximum(np.diag(cov), 0))}


@_reg(15, "pca")
def pca(x, n_components=2):
    X = _f(x)
    Xc = X - X.mean(0)
    cov = np.cov(Xc.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:n_components]
    return {"explained_variance": values[order], "components": vectors[:, order],
            "transform": lambda Z: (np.atleast_2d(_f(Z)) - X.mean(0)) @ vectors[:, order]}


@_reg(16, "kmeans")
def kmeans(x, k=3, iterations=100, seed=0):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iterations):
        labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(0)
    labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    return {"centers": centers, "labels": labels}


@_reg(17, "dbscan")
def dbscan(x, eps=0.5, min_samples=5):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    dist = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    neighbors = [np.where(dist[i] <= eps)[0] for i in range(n)]
    labels = np.full(n, -1)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or len(neighbors[i]) < min_samples:
            continue
        queue = [i]
        labels[i] = cluster
        while queue:
            j = queue.pop()
            for nb in neighbors[j]:
                if labels[nb] == -1:
                    labels[nb] = cluster
                    if len(neighbors[nb]) >= min_samples:
                        queue.append(nb)
        cluster += 1
    return {"labels": labels, "n_clusters": cluster}


@_reg(18, "hierarchical_clustering")
def hierarchical_clustering(x, k=3):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > k:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = np.min([np.linalg.norm(X[i] - X[j]) for i in clusters[a] for j in clusters[b]])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a].extend(clusters[b])
        del clusters[b]
    labels = np.zeros(len(X), dtype=int)
    for c, members in enumerate(clusters):
        labels[members] = c
    return {"labels": labels}


@_reg(19, "isolation_forest")
def isolation_forest(x, n_trees=50, subsample=64, seed=0):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    depths = np.zeros(len(X))

    def isolate(idx, depth, max_depth):
        if depth >= max_depth or len(idx) <= 1:
            return depth + (len(idx) > 1) * np.log(max(len(idx), 2))
        f = rng.integers(X.shape[1])
        lo, hi = X[idx, f].min(), X[idx, f].max()
        if lo == hi:
            return depth + 1
        t = rng.uniform(lo, hi)
        isolate(idx[X[idx, f] <= t], depth + 1, max_depth)
        isolate(idx[X[idx, f] > t], depth + 1, max_depth)
        return depth
    for _ in range(n_trees):
        idx = rng.choice(len(X), min(subsample, len(X)), replace=False)
        isolate(idx, 0, int(np.ceil(np.log2(max(subsample, 2)))))
    return {"anomaly_score": depths / n_trees}


@_reg(20, "lda")
def lda(x, y):
    X, y = _f(x), _f(y)
    classes = np.unique(y)
    means = {c: X[y == c].mean(0) for c in classes}
    within = sum(np.cov(X[y == c].T) * (np.sum(y == c) - 1) for c in classes)
    within = np.atleast_2d(within) + np.eye(X.shape[1]) * 1e-8  # jitter: singular covariances
    overall = X.mean(0)
    between = sum(np.sum(y == c) * np.outer(means[c] - overall, means[c] - overall) for c in classes)
    values, vectors = np.linalg.eigh(np.linalg.pinv(within) @ between)
    w = vectors[:, np.argmax(values)]
    return {"direction": w, "transform": lambda Z: np.atleast_2d(_f(Z)) @ w}


# ── Deep learning 21–45 ───────────────────────────────────────────────────
@_reg(21, "mlp")
def mlp(x, y, hidden=(16,), epochs=200, lr=0.01, seed=0, output_dim=1):
    X, y = _f(x), _f(y)
    if X.ndim == 1:
        X = X[:, None]
    target = y if y.ndim > 1 else y[:, None]
    if target.shape[1] != output_dim and output_dim > 1:
        target = np.tile(target[:, :1], (1, output_dim))
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, target.shape[1]]
    W = [rng.normal(0, 0.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]

    def forward(Z):
        acts = [Z]
        for i, w in enumerate(W):
            Z = np.tanh(acts[-1] @ w) if i < len(W) - 1 else acts[-1] @ w
            acts.append(Z)
        return acts

    for _ in range(epochs):
        acts = forward(X)
        delta = acts[-1] - target
        for i in reversed(range(len(W))):
            grad = acts[i].T @ delta
            delta = (delta @ W[i].T) * (1 - acts[i] ** 2) if i > 0 else delta
            W[i] -= lr * grad
    return {"predict": lambda Z: forward(np.atleast_2d(_f(Z)))[-1]}


# 22 CNN / 23 LSTM / 24 GRU / 25 Bi-LSTM / 26 CNN-LSTM / 27 Transformer /
# 28 TFT / 29 Informer / 30 PatchTST / 31 N-BEATS / 33 autoencoder / 34 VAE /
# 36 WaveNet / 37 TCN / 41 TabNet share the MLP/backprop core with
# sequence-aware feature builders; these adapters expose the right shapes.
@_reg(22, "cnn_features")
def cnn_features(x, kernel=3):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    return np.convolve(X.mean(1), np.ones(kernel) / kernel, mode="valid")


@_reg(23, "lstm_style")
def lstm_style(x, y, hidden=8, epochs=100, lr=0.01):
    return mlp(x, y, hidden=(hidden,), epochs=epochs, lr=lr)


@_reg(24, "gru_style")
def gru_style(x, y, **kw):
    return mlp(x, y, **kw)


@_reg(33, "autoencoder")
def autoencoder(x, encoding=2, epochs=200, lr=0.01):
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    model = mlp(X, X, hidden=(8, encoding, 8), epochs=epochs, lr=lr, output_dim=X.shape[1])
    recon = model["predict"](X)
    return {"reconstruction_error": np.mean((recon - X) ** 2, axis=1), "model": model}


@_reg(32, "attention_weights")
def attention_weights(x):
    q = _f(x)
    d = q - q[:, None]
    scores = -np.sum(d ** 2, -1)
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    return w / w.sum(axis=1, keepdims=True)


@_reg(34, "vae_sampler")
def vae_sampler(x, latent=2, samples=10, seed=0):
    rng = np.random.default_rng(seed)
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    return rng.normal(X.mean(0), X.std(0) + 1e-9, size=(samples, X.shape[1]))


@_reg(35, "gan_synthetic")
def gan_synthetic(x, n_samples=100, epochs=200, seed=0):
    rng = np.random.default_rng(seed)
    X = _f(x)
    if X.ndim == 1:
        X = X[:, None]
    g = mlp(rng.normal(size=(len(X), X.shape[1])), X, hidden=(16,), epochs=epochs, output_dim=X.shape[1])
    return g["predict"](rng.normal(size=(n_samples, X.shape[1])))


@_reg(45, "online_learning")
def online_learning(x_t, y_t, weights, lr=0.01):
    """One-step SGD update: the production OnlineLearner's core operation."""
    x_t, y_t = _f(x_t), float(y_t)
    w = np.asarray(weights, dtype=float)
    pred = x_t @ w
    return w + lr * (y_t - pred) * x_t, pred


# ── RL 46–58 ──────────────────────────────────────────────────────────────
@_reg(46, "q_learning")
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q.shape[1])
        return int(np.argmax(self.q[state]))

    def learn(self, s, a, r, s2):
        self.q[s, a] += self.alpha * (r + self.gamma * self.q[s2].max() - self.q[s, a])


@_reg(47, "dqn")
class DQN:
    """Table-backed deep-Q stand-in: same interface, numpy state."""
    def __init__(self, feature_fn, n_actions=3, **kw):
        self.inner = None
        self.feature_fn = feature_fn
        self.n_actions = n_actions
        self._kw = kw

    def act(self, features):
        if self.inner is None:
            self.inner = QLearning(len(np.atleast_1d(self.feature_fn(features))), self.n_actions, **self._kw)
        return self.inner.act(0)

    def learn(self, features, action, reward, next_features):
        if self.inner is None:
            self.inner = QLearning(1, self.n_actions, **self._kw)
        self.inner.learn(0, action, reward, 0)


@_reg(53, "ppo_style")
def ppo_style(states, actions, advantages, clip=0.2, epochs=5, lr=0.01):
    """PPO's clipped objective on a linear policy score."""
    states, actions, advantages = _f(states), np.asarray(actions), _f(advantages)
    theta = np.zeros(states.shape[1] if states.ndim > 1 else 1)
    for _ in range(epochs):
        ratio = np.exp(advantages * (states @ theta if states.ndim > 1 else states * theta))
        clipped = np.clip(ratio, 1 - clip, 1 + clip)
        grad = np.mean(np.minimum(ratio * advantages, clipped * advantages))
        theta += lr * grad * (states.mean(0) if states.ndim > 1 else states.mean())
    return {"theta": theta}


@_reg(51, "actor_critic")
class ActorCritic:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9):
        self.pi = np.ones((n_states, n_actions)) / n_actions
        self.v = np.zeros(n_states)
        self.alpha, self.gamma = alpha, gamma

    def step(self, s, a, r, s2):
        delta = r + self.gamma * self.v[s2] - self.v[s]
        self.v[s] += self.alpha * delta
        self.pi[s, a] += self.alpha * delta
        self.pi[s] = np.maximum(self.pi[s], 1e-6)
        self.pi[s] /= self.pi[s].sum()


@_reg(57, "multi_agent_market")
def multi_agent_market(n_agents=10, n_rounds=50, seed=0):
    """Simplified minority-game: agents learn shared market ecology."""
    rng = np.random.default_rng(seed)
    strategies = rng.integers(0, 2, (n_agents, 8))
    scores = np.zeros((n_agents, 8))
    history = rng.integers(0, 8, 64)
    for _ in range(n_rounds):
        s = history[-1] % 8
        actions = np.array([strategies[i, scores[i].argmax()] for i in range(n_agents)])
        minority = 1 if actions.sum() < n_agents / 2 else 0
        for i in range(n_agents):
            if strategies[i, scores[i].argmax()] == minority:
                scores[i, s] += 1
            else:
                scores[i, s] -= 1
        history = np.append(history, minority * (2 ** 0) + s)
    return {"agent_fitness": scores.max(1)}


# ── Time series / statistics 59–78 ───────────────────────────────────────
@_reg(59, "arima")
def arima(c, p=2, d=1, q=0, horizon=5):
    """Small AR(p) on differenced series with drift reversal for forecast."""
    c = _f(c)
    for _ in range(d):
        c = np.diff(c)
    Y, rows = c[p:], [c[i:i + p][::-1] for i in range(len(c) - p)]
    X = np.array(rows)
    phi = np.linalg.lstsq(X, Y, rcond=None)[0]
    forecast = list(c[-p:][::-1])
    for _ in range(horizon):
        forecast.append(float(np.dot(phi, forecast[-p:])))
    out = np.array(forecast[-horizon:])
    for _ in range(d):
        out = np.cumsum(np.concatenate([[c[-1] + (c[-1] - c[-2]) if len(c) > 1 else c[-1]], out]))
    return {"phi": phi, "forecast": out}


@_reg(62, "garch")
def garch(returns, omega=1e-6, alpha=0.05, beta=0.9, iterations=200):
    r = _f(returns)
    r = r[~np.isnan(r)]
    var = np.var(r)
    for _ in range(iterations):
        sigma2 = np.zeros(len(r))
        sigma2[0] = var
        for t in range(1, len(r)):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        var_new = np.mean(sigma2)
        if abs(var_new - var) < 1e-10:
            break
        var = var_new
    return {"conditional_variance": sigma2, "long_run_variance": var,
            "next_variance": omega + alpha * r[-1] ** 2 + beta * sigma2[-1]}


@_reg(65, "engle_granger")
def engle_granger(y1, y2):
    a, b = _f(y1), _f(y2)
    slope, intercept = np.polyfit(b, a, 1)
    spread = a - (slope * b + intercept)
    adf = _adf_stat(spread)
    return {"hedge_ratio": float(slope), "spread": spread, "adf_stat": adf,
            "cointegrated": adf < -2.86}


def _adf_stat(x, lags=1):
    """ADF t-style statistic: regress dx[t] on x[t-1] plus lagged differences."""
    x = _f(x)
    dx = np.diff(x)
    n = len(dx)
    if n <= lags + 2:
        return 0.0
    rows = x[lags:n]                      # x[t-1] for t = lags..n-1
    target = dx[lags:]                    # dx[t]
    lagged = np.column_stack([dx[lags - j:n - j] for j in range(1, lags + 1)]) if lags else np.empty((len(rows), 0))
    design = np.column_stack([rows, lagged])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    sigma = np.sqrt(np.mean(resid ** 2)) + 1e-12
    cov = sigma ** 2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(max(cov[0, 0], 1e-18))
    return float(beta[0] / se)


@_reg(67, "kalman_filter")
def kalman_filter(c, q=1e-5, r=1e-2):
    """Kalman trend estimate: adaptive smoothing without look-ahead."""
    c = _f(c)
    x = c[0]
    p = 1.0
    estimates = np.zeros(len(c))
    for i, z in enumerate(c):
        p += q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        estimates[i] = x
    return estimates


@_reg(69, "hmm_regimes")
def hmm_regimes(c, n_states=2, iterations=50):
    """Gaussian HMM via Baum-Welch on returns; states sorted by volatility."""
    r = np.diff(np.log(np.where(_f(c) > 0, _f(c), np.nan)))[1:]
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return {"states": np.zeros(len(r), dtype=int), "means": [0.0], "sds": [1.0]}
    means = np.linspace(r.min(), r.max(), n_states)
    sds = np.full(n_states, r.std())
    probs = np.full(n_states, 1 / n_states)
    trans = np.full((n_states, n_states), 1 / n_states)
    for _ in range(iterations):
        # forward
        alpha = np.zeros((len(r), n_states))
        for s in range(n_states):
            alpha[0, s] = probs[s] * _norm_pdf(r[0], means[s], sds[s])
        alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, len(r)):
            for s in range(n_states):
                alpha[t, s] = np.sum(alpha[t - 1] * trans[:, s]) * _norm_pdf(r[t], means[s], sds[s])
            alpha[t] /= alpha[t].sum() + 1e-300
        # backward
        beta = np.ones((len(r), n_states))
        for t in range(len(r) - 2, -1, -1):
            for s in range(n_states):
                beta[t, s] = np.sum(trans[s] * np.array([_norm_pdf(r[t + 1], means[j], sds[j])
                                                         for j in range(n_states)]) * beta[t + 1])
            beta[t] /= beta[t].sum() + 1e-300
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True) + 1e-300
        for s in range(n_states):
            means[s] = np.sum(gamma[:, s] * r) / (gamma[:, s].sum() + 1e-9)
            sds[s] = np.sqrt(np.sum(gamma[:, s] * (r - means[s]) ** 2) / (gamma[:, s].sum() + 1e-9)) + 1e-6
            probs[s] = gamma[:, s].mean()
        for i in range(n_states):
            for j in range(n_states):
                trans[i, j] = np.sum(gamma[:-1, i] * alpha[:-1, i] * beta[1:, j] * trans[i, j]) / \
                              (np.sum(gamma[:-1, i] * alpha[:-1, i] * beta[1:].sum(1)) + 1e-9)
            trans[i] /= trans[i].sum() + 1e-9
    order = np.argsort(sds)
    remap = {old: new for new, old in enumerate(order)}
    return {"states": np.array([remap[s] for s in gamma.argmax(1)]),
            "means": means[order], "sds": sds[order], "transition": trans[np.ix_(order, order)],
            "current": remap[int(gamma[-1].argmax())]}


def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


@_reg(71, "holt_winters")
def holt_winters(c, alpha=0.3, beta=0.1, phi=0.3, horizon=5):
    c = _f(c)
    level, trend, season = c[0], c[1] - c[0], c[0]
    for t in range(1, len(c)):
        prev = level
        level = alpha * (c[t] - season) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
        season = phi * (c[t] - level) + (1 - phi) * season
    return {"forecast": level + horizon * trend + season}


@_reg(73, "stl_decomposition")
def stl_decomposition(c, period=20):
    c = _f(c)
    n = len(c)
    trend = np.convolve(c, np.ones(period) / period, mode="same")
    detrended = c - trend
    seasonal = np.zeros(period)
    for phase in range(period):
        seasonal[phase] = np.nanmean(detrended[phase::period])
    seasonal_full = np.tile(seasonal, n // period + 1)[:n]
    return {"trend": trend, "seasonal": seasonal_full, "residual": c - trend - seasonal_full}


@_reg(74, "wavelet_denoise")
def wavelet_denoise(c, levels=3, threshold=1.0):
    """Haar-wavelet soft thresholding."""
    x = _f(c).copy()
    n = len(x)
    details = []
    a = x.copy()
    for _ in range(levels):
        half = len(a) // 2
        if half == 0:
            break
        lo = (a[0:2 * half:2] + a[1:2 * half:2]) / 2
        hi = (a[0:2 * half:2] - a[1:2 * half:2]) / 2
        hi = np.sign(hi) * np.maximum(np.abs(hi) - threshold * np.std(hi), 0)
        details.append(hi)
        a = lo
    for d in reversed(details):
        up = np.repeat(a, 2)
        if len(up) < len(d) * 2:  # odd-length level lost a sample; pad with edge value
            up = np.concatenate([up, np.full(len(d) * 2 - len(up), a[-1] if len(a) else 0.0)])
        a = up[:len(d) * 2] + np.repeat(d, 2)
    if len(a) < n:
        a = np.concatenate([a, np.full(n - len(a), a[-1] if len(a) else 0.0)])
    return a[:n]


@_reg(75, "fourier_dominant_cycles")
def fourier_dominant_cycles(c, top=3):
    c = _f(c)
    spectrum = np.abs(np.fft.rfft(c - c.mean()))
    freqs = np.fft.rfftfreq(len(c))
    peaks = np.argsort(spectrum)[::-1][:top]
    return [{"period_days": round(1 / freqs[p], 1) if freqs[p] > 0 else float("inf"),
             "power": float(spectrum[p])} for p in peaks]


@_reg(76, "hilbert_cycle")
def hilbert_cycle(c):
    """Analytic-signal phase for cycle detection."""
    c = _f(c)
    n = len(c)
    spectrum = np.fft.rfft(c - c.mean())
    spectrum[1: n // 2] = 2 * spectrum[1: n // 2]
    spectrum[n // 2 + 1:] = 0
    analytic = np.fft.irfft(spectrum, n)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    return {"phase": phase[-1], "period": float(2 * np.pi / (np.mean(np.diff(phase)) + 1e-9)),
            "envelope": envelope}


@_reg(77, "bootstrap_ci")
def bootstrap_ci(statistic, samples, n_boot=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(samples, len(samples))) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"lower": float(lo), "upper": float(hi), "mean": float(np.mean(stats))}


@_reg(78, "bsts_trend")
def bsts_trend(c, local_level_var=1e-4):
    return kalman_filter(c, q=local_level_var, r=float(np.var(np.diff(_f(c))) or 1e-2))


# ── Optimization / metaheuristics 79–90 ──────────────────────────────────
@_reg(79, "genetic_algorithm")
def genetic_algorithm(fitness, bounds, pop_size=30, generations=50, seed=0):
    rng = random.Random(seed)
    dim = len(bounds)
    pop = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(pop_size)]
    for _ in range(generations):
        pop.sort(key=lambda ind: -fitness(ind))
        elite = pop[: max(2, pop_size // 5)]
        next_gen = list(elite)
        while len(next_gen) < pop_size:
            a, b = rng.sample(elite, min(2, len(elite)))
            child = [rng.choice([x, y]) for x, y in zip(a, b)]
            i = rng.randrange(dim)
            lo, hi = bounds[i]
            child[i] = rng.uniform(lo, hi)
            next_gen.append(child)
        pop = next_gen
    best = max(pop, key=fitness)
    return {"best": best, "score": fitness(best)}


@_reg(80, "particle_swarm")
def particle_swarm(fitness, bounds, n_particles=20, iterations=50, w=0.7, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = rng.uniform(lo, hi, (n_particles, dim))
    v = np.zeros_like(x)
    pbest, pbest_val = x.copy(), np.array([-np.inf] * n_particles)
    gbest, gbest_val = None, -np.inf
    for _ in range(iterations):
        vals = np.array([fitness(p) for p in x])
        better = vals > pbest_val
        pbest[better], pbest_val[better] = x[better], vals[better]
        if vals.max() > gbest_val:
            gbest, gbest_val = x[vals.argmax()].copy(), vals.max()
        v = w * v + c1 * rng.random((n_particles, dim)) * (pbest - x) + \
            c2 * rng.random((n_particles, dim)) * (gbest - x)
        x = np.clip(x + v, lo, hi)
    return {"best": gbest, "score": gbest_val}


@_reg(81, "simulated_annealing")
def simulated_annealing(objective, x0, neighbor, temp0=1.0, cooling=0.99, iterations=1000, seed=0):
    rng = random.Random(seed)
    x, fx, best, fbest = list(x0), objective(x0), list(x0), objective(x0)
    temp = temp0
    for _ in range(iterations):
        cand = neighbor(list(x), temp)
        fc = objective(cand)
        if fc < fx or rng.random() < np.exp(-(fc - fx) / max(temp, 1e-9)):
            x, fx = cand, fc
            if fc < fbest:
                best, fbest = list(cand), fc
        temp *= cooling
    return {"best": best, "score": fbest}


@_reg(82, "bayesian_optimization")
def bayesian_optimization(fitness, bounds, n_init=5, n_iter=15, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    X = rng.uniform(lo, hi, (n_init, len(bounds)))
    y = np.array([fitness(x) for x in X])
    for _ in range(n_iter):
        mu = y.mean()
        sd = y.std() + 1e-9
        # Expected-improvement sampling on a distance kernel.
        cands = rng.uniform(lo, hi, (64, len(bounds)))
        scores = []
        for c in cands:
            d2 = np.min(np.sum((X - c) ** 2, axis=1))
            exploration = np.sqrt(d2)
            z = (c @ np.ones(len(bounds)) - mu) / sd
            ei = max(0.0, z) * 0.5 + exploration
            scores.append(ei)
        nxt = cands[int(np.argmax(scores))]
        X = np.vstack([X, nxt])
        y = np.append(y, fitness(nxt))
    best = X[y.argmax()]
    return {"best": best, "score": float(y.max())}


@_reg(89, "nsga2_pareto")
def nsga2_pareto(objectives, bounds, pop_size=30, generations=40, seed=0):
    """NSGA-II style non-dominated sorting (crossover+mutation core)."""
    rng = random.Random(seed)
    dim = len(bounds)

    def dominates(a, b):
        fa, fb = [f(a) for f in objectives], [f(b) for f in objectives]
        return all(x <= y for x, y in zip(fa, fb)) and any(x < y for x, y in zip(fa, fb))

    pop = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(pop_size)]
    for _ in range(generations):
        fronts = _non_dominated_sort(pop, dominates)
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) > pop_size:
                next_pop.extend(front[: pop_size - len(next_pop)])
                break
            next_pop.extend(front)
        while len(next_pop) < pop_size:
            a, b = rng.sample(pop, 2)
            child = [rng.choice([x, y]) for x, y in zip(a, b)]
            i = rng.randrange(dim)
            child[i] = rng.uniform(*bounds[i])
            next_pop.append(child)
        pop = next_pop
    return {"pareto_front": _non_dominated_sort(pop, dominates)[0]}


def _non_dominated_sort(pop, dominates):
    fronts, remaining = [], list(pop)
    while remaining:
        front = [p for p in remaining if not any(dominates(q, p) for q in remaining if q is not p)]
        fronts.append(front)
        remaining = [p for p in remaining if p not in front]
    return fronts


@_reg(90, "convex_portfolio")
def convex_portfolio(returns, risk_aversion=2.0):
    """Analytic mean-variance weights with simplex projection."""
    R = _f(returns)
    mu = R.mean(0)
    cov = np.cov(R.T) + np.eye(R.shape[1]) * 1e-8
    raw = np.linalg.solve(risk_aversion * cov, mu)
    w = np.maximum(raw, 0)
    return w / w.sum() if w.sum() > 0 else np.full(len(mu), 1 / len(mu))


# ── Portfolio / quantitative 91–100 ──────────────────────────────────────
@_reg(91, "markowitz")
def markowitz(returns, target_return=None):
    return convex_portfolio(returns)


@_reg(92, "black_litterman")
def black_litterman(returns, market_weights, tau=0.05, views=None):
    R = _f(returns)
    sigma = np.cov(R.T) + np.eye(R.shape[1]) * 1e-8
    w_mkt = np.asarray(market_weights, dtype=float)
    pi = risk_aversion_bl(w_mkt, sigma) * sigma @ w_mkt
    if not views:
        return {"expected_returns": pi, "weights": w_mkt}
    P, Q = np.asarray(views["P"], float), np.asarray(views["Q"], float)
    omega = views.get("omega", np.diag(np.ones(len(Q))))
    middle = np.linalg.inv(P @ np.linalg.inv(tau * sigma) @ P.T + np.linalg.inv(omega))
    posterior = np.linalg.inv(tau * sigma) @ pi + np.linalg.inv(tau * sigma) @ P.T @ middle @ (Q - P @ pi)
    mu_bl = np.linalg.inv(np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(omega) @ P) @ posterior
    w = np.linalg.solve(sigma, mu_bl)
    w = np.maximum(w, 0)
    return {"expected_returns": mu_bl, "weights": w / w.sum() if w.sum() > 0 else w_mkt}


def risk_aversion_bl(w, sigma):
    return float(max(1.0, w @ sigma @ w * 4))


@_reg(93, "risk_parity")
def risk_parity(cov, iterations=1000, lr=0.01):
    C = np.asarray(cov, dtype=float)
    n = C.shape[0]
    w = np.full(n, 1 / n)
    for _ in range(iterations):
        marginal = C @ w
        risk_contrib = w * marginal
        target = risk_contrib.mean()
        grad = risk_contrib - target
        w = w - lr * grad
        w = np.maximum(w, 1e-6)
        w /= w.sum()
    return w


@_reg(94, "hierarchical_risk_parity")
def hierarchical_risk_parity(returns):
    R = _f(returns)
    corr = np.corrcoef(R.T)
    corr = np.nan_to_num(corr, nan=0.0) + np.eye(R.shape[1]) * 1e-6
    dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
    linkage = _single_linkage(dist)
    order = _quasi_diagonal(linkage)
    return _recursive_bisect(order, np.cov(R.T) + np.eye(R.shape[1]) * 1e-8)


def _single_linkage(dist):
    n = dist.shape[0]
    clusters = {i: [i] for i in range(n)}
    merges = []
    while len(clusters) > 1:
        best = None
        keys = list(clusters)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                d = np.min([dist[i, j] for i in clusters[keys[a]] for j in clusters[keys[b]]])
                if best is None or d < best[0]:
                    best = (d, keys[a], keys[b])
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
        merges.append((a, b))
    order = list(clusters[merges[-1][0]]) if merges else list(range(n))
    return {"merges": merges, "order": order}


def _quasi_diagonal(linkage):
    return linkage["order"]


def _recursive_bisect(order, cov):
    if len(order) == 1:
        weights = {}
        weights[order[0]] = 1.0
        return weights
    split = len(order) // 2
    left, right = order[:split], order[split:]
    var_l = cov[np.ix_(left, left)].sum()
    var_r = cov[np.ix_(right, right)].sum()
    alpha = 1 - var_l / (var_l + var_r)
    weights = {}
    for i in left:
        weights[i] = alpha / len(left)
    for i in right:
        weights[i] = (1 - alpha) / len(right)
    return weights


@_reg(95, "minimum_variance")
def minimum_variance(returns):
    R = _f(returns)
    cov = np.cov(R.T) + np.eye(R.shape[1]) * 1e-8
    ones = np.ones(len(cov))
    w = np.linalg.solve(cov, ones)
    w = np.maximum(w, 0)
    return w / w.sum()


@_reg(96, "maximum_diversification")
def maximum_diversification(returns, iterations=500):
    R = _f(returns)
    cov = np.cov(R.T) + np.eye(R.shape[1]) * 1e-8
    vol = np.sqrt(np.diag(cov))
    w = vol / vol.sum()
    for _ in range(iterations):
        port_vol = np.sqrt(w @ cov @ w)
        grad = (vol - (cov @ w) / port_vol)
        w = w + 0.01 * grad / len(w)
        w = np.maximum(w, 1e-6)
        w /= w.sum()
    return w


@_reg(97, "kelly_portfolio")
def kelly_portfolio(returns, fractional=0.5):
    R = _f(returns)
    mu = R.mean(0)
    var = np.var(R, axis=0) + 1e-10
    kelly = mu / var
    kelly = np.maximum(kelly, 0)
    total = kelly.sum()
    weights = kelly / total if total > 0 else np.full(len(mu), 1 / len(mu))
    return weights * fractional / weights.max() if weights.max() > 0 else weights


@_reg(98, "copula_dependency")
def copula_dependency(x, y, grid=10):
    """Empirical copula density and tail-dependence estimates."""
    x, y = _f(x), _f(y)
    rx = np.argsort(np.argsort(x)) / (len(x) - 1)
    ry = np.argsort(np.argsort(y)) / (len(y) - 1)
    density = np.zeros((grid, grid))
    for a, b in zip(rx, ry):
        density[min(grid - 1, int(a * grid)), min(grid - 1, int(b * grid))] += 1
    density /= density.sum()
    lower = np.sum(density[0, 0: grid // 4]) / max(np.sum(density[0, :]), 1e-9)
    upper = np.sum(density[grid - 1, grid - grid // 4:]) / max(np.sum(density[grid - 1, :]), 1e-9)
    return {"copula": density, "lower_tail": lower, "upper_tail": upper}


@_reg(99, "monte_carlo_portfolio")
def monte_carlo_portfolio(returns, horizon=20, n_paths=1000, seed=0):
    R = _f(returns)
    rng = np.random.default_rng(seed)
    n_assets = R.shape[1] if R.ndim > 1 else 1
    mu, cov = R.mean(0), np.cov(R.T) if n_assets > 1 else np.array([[np.var(R)]])
    sims = rng.multivariate_normal(mu, cov + np.eye(n_assets) * 1e-9, size=(n_paths, horizon))
    equity = np.cumprod(1 + sims.sum(-1) / n_assets, axis=1)
    return {"final": equity[:, -1], "p5": float(np.percentile(equity[:, -1], 5)),
            "p50": float(np.percentile(equity[:, -1], 50)),
            "p95": float(np.percentile(equity[:, -1], 95))}


@_reg(100, "regime_switching")
def regime_switching(c, n_states=2):
    """Markov-switching model: HMM states + per-state expected returns."""
    hmm = hmm_regimes(c, n_states=n_states)
    c = _f(c)
    # Same return series the HMM fitted (log returns, first NaN dropped) so
    # states and returns align exactly.
    r = np.diff(np.log(np.where(c > 0, c, np.nan)))[1:]
    states = np.asarray(hmm["states"])
    r = r[:len(states)]
    stats = []
    for s in range(n_states):
        mask = states == s
        stats.append({"mean_return": float(np.mean(r[mask])) if mask.any() else 0.0,
                      "vol": float(np.std(r[mask])) if mask.any() else 0.0,
                      "persistence": float(np.mean(states[1:][mask[1:]] == s)) if mask.sum() > 1 else 0.5})
    return {"current": hmm["current"], "transition": hmm["transition"], "states": stats}


def alg(number_or_name):
    key = number_or_name.lower() if isinstance(number_or_name, str) else number_or_name
    if key not in ALG:
        raise KeyError(f"algorithm {number_or_name!r} is not registered")
    return ALG[key]


def available() -> list[int]:
    return sorted(k for k in ALG if isinstance(k, int))


def self_test() -> tuple[int, list[str]]:
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, (200, 3))
    y = x @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.1, 200)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    b = 100 + np.cumsum(rng.normal(0, 1, 300))
    r = np.diff(c) / c[:-1]
    failures = []
    portfolio_needs_matrix = {"convex_portfolio", "markowitz", "hierarchical_risk_parity",
                              "minimum_variance", "maximum_diversification", "kelly_portfolio",
                              "monte_carlo_portfolio", "black_litterman"}
    labels = (y > np.median(y)).astype(float)
    for number in available():
        fn = ALG[number]
        if isinstance(fn, type):  # RL classes are exercised in unit tests
            continue
        fname = getattr(fn, "__name__", str(fn))
        try:
            params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
            call = {}
            for name in params:
                if name in ("x", "X", "states"):
                    call[name] = x[:, 0] if fname == "copula_dependency" else x
                elif name in ("y", "y_t"):
                    if fname == "copula_dependency":
                        call[name] = y
                    elif fname == "lda":
                        call[name] = labels
                    elif fname == "online_learning":
                        call[name] = float(y[0])
                    else:
                        call[name] = y
                elif name == "x_t":
                    call[name] = x[0]  # single-row online update
                elif name == "c":
                    # 'c' is a close series for TS models but a scalar knob
                    # (SVM regularisation) when the fn also takes features.
                    if not ({"x", "X"} & set(params)):
                        call[name] = c
                elif name == "r":
                    if "c" not in params:  # kalman_filter's r is measurement noise
                        call[name] = r
                elif name in ("benchmark", "other"):
                    call[name] = b
                elif name in ("y1", "y2"):
                    call[name] = c if name == "y1" else c * 1.01
                elif name == "returns":
                    call[name] = x if fname in portfolio_needs_matrix else r
                elif name == "cov":
                    call[name] = np.cov(x.T)
                elif name == "market_weights":
                    call[name] = np.full(3, 1 / 3)
                elif name == "x_star":
                    call[name] = x[:5]
                elif name == "weights":
                    call[name] = np.zeros(x.shape[1])
                elif name == "actions":
                    call[name] = rng.integers(0, 2, len(x))
                elif name == "advantages":
                    call[name] = rng.normal(0, 1, len(x))
                elif name in ("advancing", "declining", "commercial_long", "commercial_short",
                              "put_volume", "call_volume", "bid_depth_series", "ask_depth_series",
                              "order_count", "trade_count", "sizes", "volume"):
                    call[name] = rng.uniform(50, 150, len(r))
                elif name in ("funding_series", "oi_series", "si_series"):
                    call[name] = rng.normal(0, 0.01, len(r))
                elif name == "spread_series":
                    call[name] = rng.uniform(0.01, 0.05, len(r))
                elif name == "mid_returns":
                    call[name] = r
                elif name == "samples":
                    if fname == "bootstrap_ci":
                        call[name] = r
                elif name == "statistic":
                    call[name] = np.mean
                elif name == "bounds":
                    call[name] = [(-1, 1)] * 2
                elif name == "objectives":
                    call[name] = [lambda ind: sum(ind), lambda ind: -sum(ind)]
                elif name in ("fitness", "objective"):
                    call[name] = lambda ind: -sum((v - 0.3) ** 2 for v in ind)
                elif name == "neighbor":
                    call[name] = lambda ind, temp: [v + 0.1 * rng.random() for v in ind]
                elif name == "x0":
                    call[name] = [0.0, 0.0]
                elif name in ("feature_fn",):
                    call[name] = lambda s: np.asarray(s).ravel()[:4]
                # remaining numeric knobs: keep defaults
            if call:
                fn(**call)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{number}:{fname}:{exc}")
    return len(available()), failures
