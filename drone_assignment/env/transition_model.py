"""Deterministic state transitions for the rescue drone simulator."""

from __future__ import annotations

from .actions import Action, MOVEMENT_ACTIONS
from .map_loader import MapConfig
from .state import DroneState

MOVE_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_NORTH: (-1, 0),
    Action.MOVE_SOUTH: (1, 0),
    Action.MOVE_EAST: (0, 1),
    Action.MOVE_WEST: (0, -1),
}


def apply_action(
    map_config: MapConfig,
    state: DroneState,
    action: Action,
    *,
    max_battery: int,
    scan_cost: int = 1,
    recharge_amount: int | str = "full",
) -> DroneState:
    """Apply one action and return the next visible state.

    Transition highlights:
    - movement: battery -1, blocked by boundary/obstacle
    - scan: battery -`scan_cost`
    - recharge: if on `B` cell, restore battery to `max_battery`
    """

    next_row = state.row
    next_col = state.col
    next_battery = state.battery
    next_time_step = state.time_step + 1
    used_battery_stations = set(state.used_battery_stations)
    collected_reward_cells = set(state.collected_reward_cells)

    if action in MOVEMENT_ACTIONS and state.battery > 0:
        next_battery = max(0, state.battery - 1)
        delta_row, delta_col = MOVE_DELTAS[action]
        candidate_row = state.row + delta_row
        candidate_col = state.col + delta_col
        if map_config.in_bounds(candidate_row, candidate_col) and (
            candidate_row,
            candidate_col,
        ) not in map_config.obstacles:
            next_row = candidate_row
            next_col = candidate_col
    elif action == Action.SCAN and state.battery > 0:
        next_battery = max(0, state.battery - scan_cost)
    elif action == Action.RECHARGE and is_battery_cell(map_config, state):
        current_cell = state.position
        if current_cell not in used_battery_stations:
            if recharge_amount == "full":
                next_battery = max_battery
            else:
                next_battery = min(max_battery, state.battery + max(0, int(recharge_amount)))
            used_battery_stations.add(current_cell)

    current_cell = (next_row, next_col)
    if current_cell in map_config.reward_cells:
        collected_reward_cells.add(current_cell)

    return DroneState(
        row=next_row,
        col=next_col,
        battery=next_battery,
        time_step=next_time_step,
        used_battery_stations=_sorted_coordinate_tuple(used_battery_stations),
        collected_reward_cells=_sorted_coordinate_tuple(collected_reward_cells),
    )


def is_battery_cell(map_config: MapConfig, state: DroneState) -> bool:
    """Return whether the drone is currently on a battery station cell."""

    return state.position in map_config.battery_stations


def is_hazard_state(map_config: MapConfig, state: DroneState) -> bool:
    """Return whether the position is a possible hazard-region (`H`) cell."""

    return state.position in map_config.hazards


def is_survivor_zone_state(map_config: MapConfig, state: DroneState) -> bool:
    """Return whether the drone position is a possible survivor-zone (`G`) cell."""

    return state.position in map_config.survivors


def is_survivor_state(map_config: MapConfig, state: DroneState) -> bool:
    """Backward-compatible alias for `is_survivor_zone_state`."""

    return is_survivor_zone_state(map_config, state)


def is_reward_cell(map_config: MapConfig, state: DroneState) -> bool:
    """Return whether the drone position is a reward cell (`R`)."""

    return state.position in map_config.reward_cells


def _sorted_coordinate_tuple(
    coordinates: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return deterministic tuple encoding for coordinate-set fields."""

    return tuple(sorted(coordinates))
