"""Smoothing filters for noisy time-series robot signals (positions, poses)."""

from __future__ import annotations

import numpy as np


class EmaFilter:
    """Exponential moving average over a vector signal.

    The new sample contributes `alpha`; the previous smoothed value contributes
    `1 - alpha`.

        smoothed_t = alpha * x_t + (1 - alpha) * smoothed_{t-1}

    Effective averaging window ~= 1 / alpha samples. Pick alpha by how
    aggressively you want to smooth at your sample rate:

        alpha = 0.10  -> heavy smoothing, lots of latency (~10-sample window)
        alpha = 0.30  -> moderate smoothing, low lag        ~3-sample window)
        alpha = 0.60  -> light smoothing                    (~1.7-sample window)
        alpha = 1.00  -> no smoothing (pass-through)

    Dropouts: pass `None` to indicate a missing sample. The filter holds the
    last value internally but reports `None` for that step too, so downstream
    consumers can decide what to do.
    """

    def __init__(self, alpha: float, max_dropouts: int = 5):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_dropouts = max_dropouts
        self._state: np.ndarray | None = None
        self._consecutive_dropouts = 0

    def update(self, x: np.ndarray | None) -> np.ndarray | None:
        """Push one sample (or None for a dropout). Return the smoothed value."""
        if x is None:
            self._consecutive_dropouts += 1
            if self._consecutive_dropouts > self.max_dropouts:
                # Stale; forget the previous state so the next valid sample
                # initializes cleanly instead of slowly drifting in.
                self._state = None
            return None

        self._consecutive_dropouts = 0
        x = np.asarray(x, dtype=np.float64)

        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state

        return self._state.copy()

    def reset(self):
        self._state = None
        self._consecutive_dropouts = 0
