"""Purged walk-forward cross-validation (Lopez de Prado discipline).

For a label that looks ``h`` bars into the future, a training sample whose
label window overlaps a test fold leaks future information into training.
Purging drops those training samples; the embargo additionally removes a
buffer *after* each test fold to kill serial-correlation leakage.

``purged_walk_forward(n, train_bars, test_bars, embargo_bars, horizon_bars)``
yields (train_idx, test_idx) pairs over ``0..n-1`` — overlapping-free,
test folds disjoint and in temporal order.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def purged_walk_forward(
    n: int,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
    horizon_bars: int = 1,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train, test) index arrays with purge + embargo.

    Args:
        n: total number of bars.
        train_bars: length of each training window.
        test_bars: length of each test fold.
        embargo_bars: gap enforced AFTER each test fold (serial leakage).
        horizon_bars: label lookahead — training samples whose label window
            [i, i+horizon) intersects the test fold are purged.
    """
    if n <= 0 or train_bars <= 0 or test_bars <= 0:
        return
    horizon_bars = max(1, int(horizon_bars))
    embargo_bars = max(0, int(embargo_bars))

    # Standard walk-forward: the first test fold only starts once a full
    # training window exists, so every fold is fitted on real history.
    test_start = train_bars
    while test_start + test_bars <= n:
        test_end = test_start + test_bars
        train_start = max(0, test_start - train_bars)
        train_idx = np.arange(train_start, test_start)
        # Purge: drop train samples whose label window touches the test fold.
        keep = train_idx + horizon_bars <= test_start
        train_idx = train_idx[keep] if len(train_idx) else train_idx

        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx

        # Embargo: the next fold starts after the gap.
        test_start = test_end + embargo_bars


def leakage_report(n: int, train_bars: int, test_bars: int,
                   embargo_bars: int, horizon_bars: int) -> dict:
    """Summarise fold structure (used by tests and the validate command)."""
    folds = list(purged_walk_forward(n, train_bars, test_bars, embargo_bars, horizon_bars))
    covered = np.zeros(n, dtype=int)
    overlaps = 0
    for train_idx, test_idx in folds:
        covered[test_idx] += 1
        overlaps += int(np.intersect1d(train_idx + horizon_bars - 1, test_idx).size)
    gaps = np.where(covered == 0)[0]
    return {
        "folds": len(folds),
        "test_coverage": float((covered > 0).mean()),
        "max_fold_overlap": int(covered.max()),
        "label_overlap_violations": overlaps,
        "embargoed_bars": int(len(gaps)),
    }
