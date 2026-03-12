"""Probabilistic observation model for the rescue drone simulator."""

from __future__ import annotations

import random
from enum import Enum
from typing import Iterable

from .actions import Action
from .map_loader import MapConfig
from .state import DroneState

# Survivor-signal table requested by assignment spec.
_SURVIVOR_SIGNAL_TABLE: tuple[tuple[int, float], ...] = (
    (0, 0.95),
    (1, 0.90),
    (2, 0.80),
    (3, 0.70),
    (4, 0.60),
    (5, 0.50),
    (6, 0.40),
    (7, 0.30),
    (8, 0.22),
    (9, 0.15),
)

# Hazard-warning table requested by assignment spec.
_HAZARD_WARNING_TABLE: tuple[tuple[int, float], ...] = (
    (1, 0.85),
    (3, 0.60),
    (6, 0.30),
)


class Observation(str, Enum):
    """Observation symbols emitted by `env.step`."""

    NONE = "NONE"
    SURVIVOR_SIGNAL = "SURVIVOR_SIGNAL"
    HAZARD_WARNING = "HAZARD_WARNING"
    NO_SIGNAL = "NO_SIGNAL"
    # Backward-compatible aliases used by earlier tests/materials.
    POSITIVE = "SURVIVOR_SIGNAL"
    NEGATIVE = "NO_SIGNAL"


def survivor_signal_probability(
    state: DroneState,
    true_survivor_cell: tuple[int, int] | None,
) -> float:
    """Return `P(SURVIVOR_SIGNAL)` from distance to hidden survivor."""

    distance = _distance_to_cell(state, true_survivor_cell)
    if distance is None:
        return 0.10
    for max_distance, probability in _SURVIVOR_SIGNAL_TABLE:
        if distance <= max_distance:
            return probability
    return 0.10


def hazard_warning_probability(
    state: DroneState,
    active_hazard_cells: Iterable[tuple[int, int]],
) -> float:
    """Return `P(HAZARD_WARNING)` from nearest active hazard distance."""

    distance = _distance_to_nearest(state, active_hazard_cells)
    if distance is None:
        return 0.10
    for max_distance, probability in _HAZARD_WARNING_TABLE:
        if distance <= max_distance:
            return probability
    return 0.10


def scan_positive_probability(
    map_config: MapConfig,
    state: DroneState,
    *,
    true_survivor_cell: tuple[int, int] | None = None,
    active_hazard_cells: Iterable[tuple[int, int]] | None = None,
    true_positive: float | None = None,
    false_positive: float | None = None,
) -> float:
    """Compatibility helper returning survivor-signal probability.

    This keeps older map-analysis tests working while the environment now emits
    three scan outcomes.
    """

    _ = map_config
    _ = active_hazard_cells
    _ = true_positive
    _ = false_positive
    return survivor_signal_probability(state, true_survivor_cell)


def scan_observation_distribution(
    state: DroneState,
    *,
    true_survivor_cell: tuple[int, int] | None,
    active_hazard_cells: Iterable[tuple[int, int]],
) -> dict[Observation, float]:
    """Return normalized distribution over scan observations."""

    p_survivor = survivor_signal_probability(state, true_survivor_cell)
    p_hazard = hazard_warning_probability(state, active_hazard_cells)
    p_survivor = _clamp_probability(p_survivor)
    p_hazard = _clamp_probability(p_hazard)

    # Prioritize survivor signal; hazard warning appears if survivor signal is absent.
    p_survivor_signal = p_survivor
    p_hazard_warning = (1.0 - p_survivor_signal) * p_hazard
    p_no_signal = max(0.0, 1.0 - p_survivor_signal - p_hazard_warning)

    total = p_survivor_signal + p_hazard_warning + p_no_signal
    if total <= 0.0:
        return {
            Observation.SURVIVOR_SIGNAL: 0.0,
            Observation.HAZARD_WARNING: 0.0,
            Observation.NO_SIGNAL: 1.0,
        }
    return {
        Observation.SURVIVOR_SIGNAL: p_survivor_signal / total,
        Observation.HAZARD_WARNING: p_hazard_warning / total,
        Observation.NO_SIGNAL: p_no_signal / total,
    }


def sample_observation(
    map_config: MapConfig,
    state: DroneState,
    action: Action,
    rng: random.Random,
    *,
    true_survivor_cell: tuple[int, int] | None,
    active_hazard_cells: Iterable[tuple[int, int]],
) -> Observation:
    """Sample and return an observation emitted after taking `action`."""

    _ = map_config
    if action != Action.SCAN:
        return Observation.NONE

    distribution = scan_observation_distribution(
        state,
        true_survivor_cell=true_survivor_cell,
        active_hazard_cells=active_hazard_cells,
    )
    draw = rng.random()
    cumulative = 0.0
    for observation in (
        Observation.SURVIVOR_SIGNAL,
        Observation.HAZARD_WARNING,
        Observation.NO_SIGNAL,
    ):
        cumulative += distribution[observation]
        if draw <= cumulative:
            return observation
    return Observation.NO_SIGNAL


def _distance_to_cell(
    state: DroneState,
    target_cell: tuple[int, int] | None,
) -> int | None:
    """Return Manhattan distance to one cell, or `None` if no cell exists."""

    if target_cell is None:
        return None
    return abs(target_cell[0] - state.row) + abs(target_cell[1] - state.col)


def _distance_to_nearest(
    state: DroneState,
    cells: Iterable[tuple[int, int]],
) -> int | None:
    """Return Manhattan distance to nearest cell in iterable, or `None` if empty."""

    best: int | None = None
    for row, col in cells:
        distance = abs(row - state.row) + abs(col - state.col)
        if best is None or distance < best:
            best = distance
    return best


def _clamp_probability(value: float) -> float:
    """Clamp numeric probability-like values into `[0.0, 1.0]`."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
