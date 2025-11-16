"""Common data containers used by the translated algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

@dataclass
class LaplaceApproximation:
    curr: np.ndarray
    p_gam: int
    llh: float = 0.0
    lmp: float = 0.0
    log_post: float = 0.0
    log_det_sigma: np.ndarray | None = None
    A: np.ndarray | None = None

    def copy(self) -> "LaplaceApproximation":
        return LaplaceApproximation(
            curr=self.curr.copy(),
            p_gam=int(self.p_gam),
            llh=float(self.llh),
            lmp=float(self.lmp),
            log_post=float(self.log_post),
            log_det_sigma=None if self.log_det_sigma is None else self.log_det_sigma.copy(),
            A=None if self.A is None else self.A.copy(),
        )

__all__ = ["LaplaceApproximation"]
