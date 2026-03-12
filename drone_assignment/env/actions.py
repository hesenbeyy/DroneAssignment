"""Action definitions for the rescue drone simulator."""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """Discrete actions supported by the rescue drone environment."""

    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    SCAN = "SCAN"
    RECHARGE = "RECHARGE"


MOVEMENT_ACTIONS: tuple[Action, ...] = (
    Action.MOVE_NORTH,
    Action.MOVE_SOUTH,
    Action.MOVE_EAST,
    Action.MOVE_WEST,
)


def is_movement_action(action: Action) -> bool:
    """Return whether the given action is a grid movement action."""

    return action in MOVEMENT_ACTIONS

