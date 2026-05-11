"""Signal processing filters for gim_arm_control."""

import math

import numpy as np

class ButterworthFilter:
    """Second-order Butterworth low-pass filter for smoothing."""

    def __init__(self, cutoff_hz: float, dt: float, size: int):
        self.cutoff = cutoff_hz
        self.dt = dt
        self.size = size
        self._compute_coeffs()
        self.x1 = np.zeros(size, dtype=np.float64)
        self.x2 = np.zeros(size, dtype=np.float64)
        self.y1 = np.zeros(size, dtype=np.float64)
        self.y2 = np.zeros(size, dtype=np.float64)

    def _compute_coeffs(self):
        if self.cutoff <= 0.0:
            self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
            self.a1, self.a2 = 0.0, 0.0
            return
        k = math.tan(math.pi * self.cutoff * self.dt)
        norm = 1.0 + math.sqrt(2.0) * k + k * k
        self.b0 = (k * k) / norm
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (k * k - 1.0) / norm
        self.a2 = (1.0 - math.sqrt(2.0) * k + k * k) / norm

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.cutoff <= 0.0:
            return x.copy()
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1.copy(), x.copy()
        self.y2, self.y1 = self.y1.copy(), y.copy()
        return y