"""CS = α·RC + β·(1 − SE) + γ·(1 − HP), with α=0.4, β=0.35, γ=0.25 (spec §4).

HP costs a cross-encoder pass per probe, so it only runs for borderline cases.
The spec doesn't say what CS is when HP is skipped; here it's the two cheap
signals renormalized over α+β, which keeps it on the same [0, 1] scale:

    CS_prelim = (α·RC + β·(1 − SE)) / (α + β)

If CS_prelim is inside the HP band (default [0.35, 0.65]) the full formula is
used, otherwise CS = CS_prelim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

DEFAULT_HP_BAND: Tuple[float, float] = (0.35, 0.65)
_BAND_EPS = 1e-9  # so a preliminary score of exactly 0.35 isn't lost to float rounding


@dataclass(frozen=True)
class ScoringWeights:
    alpha: float = 0.40  # retrieval confidence
    beta: float = 0.35  # 1 - semantic entropy
    gamma: float = 0.25  # 1 - hallucination probability

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError(f"weights must be non-negative: {self}")
        if not math.isclose(self.alpha + self.beta + self.gamma, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {self.alpha + self.beta + self.gamma:.4f}")
        if self.alpha + self.beta == 0:
            raise ValueError("alpha + beta must be > 0 to compute the preliminary score")

    @classmethod
    def parse(cls, spec: str) -> "ScoringWeights":
        parts = [float(p) for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected 3 comma-separated weights, got {spec!r}")
        return cls(*parts)


@dataclass(frozen=True)
class CoverageScore:
    score: float
    rc: float
    se: float
    hp: Optional[float]  # None when HP was not needed
    preliminary: float

    @property
    def hp_computed(self) -> bool:
        return self.hp is not None


DEFAULT_WEIGHTS = ScoringWeights()

HPInput = Union[float, Callable[[], float], None]


def composite_coverage_score(
    rc: float,
    se: float,
    hp: HPInput,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    hp_band: Tuple[float, float] = DEFAULT_HP_BAND,
) -> CoverageScore:
    # hp can be a zero-arg callable so the cross-encoder only runs when the
    # preliminary score lands in the band. A plain float is also only used
    # in-band, so precomputed and lazy HP give identical scores.
    for name, value in (("rc", rc), ("se", se)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    lo, hi = hp_band
    a, b, g = weights.alpha, weights.beta, weights.gamma

    preliminary = (a * rc + b * (1.0 - se)) / (a + b)

    hp_value: Optional[float] = None
    in_band = lo - _BAND_EPS <= preliminary <= hi + _BAND_EPS
    if hp is not None and in_band and g > 0:
        hp_value = float(hp() if callable(hp) else hp)
        if not 0.0 <= hp_value <= 1.0:
            raise ValueError(f"hp must be in [0, 1], got {hp_value}")

    if hp_value is None:
        score = preliminary
    else:
        score = a * rc + b * (1.0 - se) + g * (1.0 - hp_value)

    return CoverageScore(score=float(score), rc=float(rc), se=float(se), hp=hp_value, preliminary=float(preliminary))
