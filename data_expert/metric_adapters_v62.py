from __future__ import annotations

import numpy as np


def weighted_pinball(y_true, y_pred, quantiles, weights=None) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape[1] != len(q):
        raise ValueError("prediction columns must match quantiles")
    if y.shape[0] != pred.shape[0]:
        raise ValueError("row count mismatch")
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(y):
        raise ValueError("weights length mismatch")
    losses = []
    for j, quantile in enumerate(q):
        err = y - pred[:, j]
        losses.append(np.maximum(quantile * err, (quantile - 1.0) * err))
    loss = np.mean(np.stack(losses, axis=1), axis=1)
    return float(np.average(loss, weights=w))


def wrmsse(y_true, y_pred, scales, weights) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    scales = np.asarray(scales, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have same [series,horizon] shape")
    if len(scales) != y.shape[0] or len(weights) != y.shape[0]:
        raise ValueError("scale/weight length must equal number of series")
    if np.any(scales <= 0):
        raise ValueError("all scales must be > 0")
    rmsse = np.sqrt(np.mean((y - pred) ** 2, axis=1) / scales)
    return float(np.average(rmsse, weights=weights))
