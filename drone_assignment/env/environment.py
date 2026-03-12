"""Main student-facing environment API for the rescue drone assignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from pathlib import Path

from .actions import Action, MOVEMENT_ACTIONS
from .map_loader import MapConfig, load_map
from .observation_model import Observation, sample_observation
from .state import DroneState
from .transition_model import (
    apply_action,
    is_battery_cell,
)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Mission constraints and reward settings for one experiment setup."""
    max_time_steps: int = 25
    initial_battery: int = 10
    max_battery: int = 10
    step_cost: float = -1.0
    scan_cost: float = -1.0
    scan_battery_usage: int = 1
    recharge_amount: int | str = "full"
    hazard_prior: float = 0.35
    hazard_penalty: float = -15.0
    battery_depletion_penalty: float = -100.0
    goal_reward: float = 100.0
    reward_cell_value: float = 10.0
    reward_requires_success: bool = True
    invalid_move_penalty: float = -3.0

    @classmethod
    def preset(cls, name: str = "DEFAULT") -> "EnvironmentConfig":
        """Return a named config preset.

        Supported names:
        - `DEFAULT`
        - `SET_A` (alias: `A`)
        - `SET_B` (alias: `B`)
        - `SET_C` (alias: `C`)
        """

        return get_environment_config(name)

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """Return available preset names."""

        return tuple(_ENVIRONMENT_CONFIG_PRESETS.keys())


_ENVIRONMENT_CONFIG_PRESETS: dict[str, EnvironmentConfig] = {
    "DEFAULT": EnvironmentConfig(),
    "SET_A": EnvironmentConfig(
        max_time_steps=25,
        initial_battery=10,
        max_battery=10,
        step_cost=-1.0,
        scan_cost=-1.0,
        scan_battery_usage=1,
        recharge_amount="full",
        hazard_prior=0.40,
        hazard_penalty=-35.0,
        battery_depletion_penalty=-120.0,
        goal_reward=100.0,
        reward_cell_value=10.0,
        reward_requires_success=True,
        invalid_move_penalty=-5.0,
    ),
    "SET_B": EnvironmentConfig(
        max_time_steps=20,
        initial_battery=7,
        max_battery=7,
        step_cost=-1.0,
        scan_cost=-1.5,
        scan_battery_usage=1,
        recharge_amount="full",
        hazard_prior=0.35,
        hazard_penalty=-15.0,
        battery_depletion_penalty=-150.0,
        goal_reward=100.0,
        reward_cell_value=10.0,
        reward_requires_success=True,
        invalid_move_penalty=-3.0,
    ),
    "SET_C": EnvironmentConfig(
        max_time_steps=25,
        initial_battery=10,
        max_battery=10,
        step_cost=-1.0,
        scan_cost=-1.0,
        scan_battery_usage=1,
        recharge_amount="full",
        hazard_prior=0.35,
        hazard_penalty=-15.0,
        battery_depletion_penalty=-100.0,
        goal_reward=100.0,
        reward_cell_value=25.0,
        reward_requires_success=True,
        invalid_move_penalty=-3.0,
    ),
}

_ENVIRONMENT_CONFIG_ALIASES: dict[str, str] = {
    "A": "SET_A",
    "B": "SET_B",
    "C": "SET_C",
}


def get_environment_config(name: str = "DEFAULT") -> EnvironmentConfig:
    """Return a copy of a named mission-parameter preset."""

    normalized_name = name.strip().upper()
    resolved_name = _ENVIRONMENT_CONFIG_ALIASES.get(normalized_name, normalized_name)
    if resolved_name not in _ENVIRONMENT_CONFIG_PRESETS:
        valid_names = ", ".join(sorted(_ENVIRONMENT_CONFIG_PRESETS.keys()))
        raise ValueError(
            f"Unknown EnvironmentConfig preset '{name}'. Valid names: {valid_names}."
        )
    return replace(_ENVIRONMENT_CONFIG_PRESETS[resolved_name])


class RescueDroneEnv:
    """Rescue drone simulator with a stable student API.

    Public API:
    - `reset() -> DroneState`
    - `available_actions(state) -> list[Action]`
    - `step(state, action) -> tuple[DroneState, Observation]`
    - `state_id(state) -> str`
    """

    def __init__(
        self,
        map_path: str | Path,
        *,
        config: EnvironmentConfig | None = None,
        rng_seed: int | None = 0,
        # Legacy compatibility overrides.
        max_battery: int | None = None,
        max_steps: int | None = None,
        scan_cost: int | float | None = None,
        scan_true_positive: float | None = None,
        scan_false_positive: float | None = None,
        scan_radius: int | None = None,
        success_reward: float | None = None,
        move_cost: float | None = None,
    ) -> None:
        """Create an environment from a map file and mission configuration."""

        self._map: MapConfig = load_map(map_path)
        base_config = config or EnvironmentConfig()
        self.config = self._apply_legacy_overrides(
            base_config,
            max_battery=max_battery,
            max_steps=max_steps,
            scan_cost=scan_cost,
            success_reward=success_reward,
            move_cost=move_cost,
        )

        # Backward-compatible public attributes used by earlier helper code.
        self.scan_true_positive = 0.90 if scan_true_positive is None else scan_true_positive
        self.scan_false_positive = 0.10 if scan_false_positive is None else scan_false_positive
        self.scan_radius = 1 if scan_radius is None else max(1, int(scan_radius))

        self._rng = random.Random(rng_seed)
        self._true_survivor_cell: tuple[int, int] | None = None
        self._active_hazard_cells: frozenset[tuple[int, int]] = frozenset()
        self._last_transition_reward: float = 0.0
        self._resample_hidden_truth()

    def reset(self) -> DroneState:
        """Reset hidden truth and return the initial visible state."""

        self._resample_hidden_truth()
        self._last_transition_reward = 0.0
        row, col = self._map.start_position
        return DroneState(
            row=row,
            col=col,
            battery=self.config.initial_battery,
            time_step=0,
            used_battery_stations=tuple(),
            collected_reward_cells=tuple(),
        )

    def available_actions(self, state: DroneState) -> list[Action]:
        """Return legal actions for the given visible state."""

        if self.is_terminal(state):
            return []

        actions: list[Action] = []
        if state.battery > 0:
            actions.extend(MOVEMENT_ACTIONS)
            actions.append(Action.SCAN)

        if (
            state.battery > 0
            and is_battery_cell(self._map, state)
            and state.position not in state.used_battery_set
            and state.battery < self.config.max_battery
        ):
            actions.append(Action.RECHARGE)
        return actions

    def step(self, state: DroneState, action: Action | str) -> tuple[DroneState, Observation]:
        """Apply one action and return `(next_state, observation)`."""

        parsed_action = action if isinstance(action, Action) else Action(action)
        legal_actions = self.available_actions(state)
        if parsed_action not in legal_actions:
            raise ValueError(
                f"Illegal action {parsed_action.value} for state {state}. "
                f"Legal actions are: {[a.value for a in legal_actions]}"
            )

        next_state = apply_action(
            self._map,
            state,
            parsed_action,
            max_battery=self.config.max_battery,
            scan_cost=self.config.scan_battery_usage,
            recharge_amount=self.config.recharge_amount,
        )
        self._last_transition_reward = self.transition_reward(state, parsed_action, next_state)
        observation = sample_observation(
            self._map,
            next_state,
            parsed_action,
            self._rng,
            true_survivor_cell=self._true_survivor_cell,
            active_hazard_cells=self._active_hazard_cells,
        )
        return next_state, observation

    def state_id(self, state: DroneState) -> str:
        """Return a canonical state ID over visible planning-relevant fields."""

        used_key = "-".join(f"{row}:{col}" for row, col in state.used_battery_stations) or "none"
        reward_key = (
            "-".join(f"{row}:{col}" for row, col in state.collected_reward_cells) or "none"
        )
        return (
            f"r{state.row}_c{state.col}_b{state.battery}"
            f"_u{used_key}_r{reward_key}"
        )

    def is_terminal(self, state: DroneState) -> bool:
        """Return whether the mission is terminal at `state`."""

        if state.time_step >= self.config.max_time_steps:
            return True
        if self._is_true_survivor_state(state):
            return True
        if state.battery <= 0:
            return True
        return False

    def transition_reward(
        self,
        previous_state: DroneState,
        action: Action | str,
        next_state: DroneState,
    ) -> float:
        """Return one-step reward using environment-defined mission constraints."""

        parsed_action = action if isinstance(action, Action) else Action(action)
        reward = self.config.step_cost

        if parsed_action == Action.SCAN:
            reward += self.config.scan_cost

        if parsed_action in MOVEMENT_ACTIONS and next_state.position == previous_state.position:
            reward += self.config.invalid_move_penalty

        if next_state.position in self._active_hazard_cells:
            reward += self.config.hazard_penalty

        if not self.config.reward_requires_success:
            newly_collected = (
                len(next_state.collected_reward_cells) - len(previous_state.collected_reward_cells)
            )
            if newly_collected > 0:
                reward += newly_collected * self.config.reward_cell_value

        success = self._is_true_survivor_state(next_state)
        if success:
            reward += self.config.goal_reward
            if self.config.reward_requires_success:
                reward += (
                    len(next_state.collected_reward_cells) * self.config.reward_cell_value
                )
            return reward

        if next_state.battery <= 0:
            reward += self.config.battery_depletion_penalty
        return reward

    def last_transition_reward(self) -> float:
        """Return the reward computed for the most recent `step` transition."""

        return self._last_transition_reward

    def render(self, state: DroneState) -> str:
        """Return plain-text map rendering with current drone location overlay."""

        rows = [list(row) for row in self._map.grid]
        rows[state.row][state.col] = "D"
        body = "\n".join("".join(row) for row in rows)
        header = (
            f"time_step={state.time_step} battery={state.battery} "
            f"used_B={len(state.used_battery_stations)} "
            f"collected_R={len(state.collected_reward_cells)} "
            f"terminal={self.is_terminal(state)}"
        )
        return f"{header}\n{body}"

    def _resample_hidden_truth(self) -> None:
        """Sample hidden survivor and hidden active hazard cells."""

        survivor_cells = sorted(self._map.survivors)
        if survivor_cells:
            self._true_survivor_cell = self._rng.choice(survivor_cells)
        else:
            self._true_survivor_cell = None

        active_hazards: set[tuple[int, int]] = set()
        for cell in sorted(self._map.hazards):
            if self._rng.random() < self.config.hazard_prior:
                active_hazards.add(cell)
        self._active_hazard_cells = frozenset(active_hazards)

    def _is_true_survivor_state(self, state: DroneState) -> bool:
        """Return whether `state` equals hidden true survivor location."""

        return self._true_survivor_cell is not None and state.position == self._true_survivor_cell

    @property
    def max_battery(self) -> int:
        """Read-only alias for configured maximum battery."""

        return self.config.max_battery

    @property
    def max_steps(self) -> int:
        """Read-only alias for configured maximum time steps."""

        return self.config.max_time_steps

    @property
    def scan_cost(self) -> int:
        """Read-only alias for scan battery consumption per action."""

        return self.config.scan_battery_usage

    @property
    def success_reward(self) -> float:
        """Read-only alias for configured mission success reward."""

        return self.config.goal_reward

    @property
    def move_cost(self) -> float:
        """Read-only alias for configured per-step movement cost."""

        return self.config.step_cost

    @staticmethod
    def _apply_legacy_overrides(
        config: EnvironmentConfig,
        *,
        max_battery: int | None,
        max_steps: int | None,
        scan_cost: int | float | None,
        success_reward: float | None,
        move_cost: float | None,
    ) -> EnvironmentConfig:
        """Apply backward-compatible constructor overrides onto config."""

        updated = config
        if max_battery is not None:
            clamped_initial = min(updated.initial_battery, int(max_battery))
            updated = replace(updated, max_battery=int(max_battery), initial_battery=clamped_initial)
        if max_steps is not None:
            updated = replace(updated, max_time_steps=int(max_steps))
        if scan_cost is not None:
            if isinstance(scan_cost, int) and scan_cost > 0:
                updated = replace(updated, scan_battery_usage=int(scan_cost))
            else:
                updated = replace(updated, scan_cost=float(scan_cost))
        if success_reward is not None:
            updated = replace(updated, goal_reward=float(success_reward))
        if move_cost is not None:
            updated = replace(updated, step_cost=float(move_cost))
        return updated
