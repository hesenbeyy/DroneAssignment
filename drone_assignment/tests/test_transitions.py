"""Transition-rule tests for battery, reward cells, and mission outcomes."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_assignment.env import Action, DroneState, RescueDroneEnv


class TestTransitions(unittest.TestCase):
    """Validate movement constraints and new mission semantics."""

    def setUp(self) -> None:
        """Create fresh deterministic environment for each test."""

        maps_dir = Path(__file__).resolve().parents[1] / "maps"
        self.env = RescueDroneEnv(maps_dir / "map_2.txt", rng_seed=23)
        self.env.reset()

    def test_boundary_blocked_move_applies_invalid_move_penalty(self) -> None:
        """Blocked movement should keep position and include invalid-move penalty."""

        state = DroneState(row=0, col=0, battery=8, time_step=1)
        next_state, _ = self.env.step(state, Action.MOVE_NORTH)
        self.assertEqual(next_state.position, state.position)
        self.assertEqual(next_state.battery, state.battery - 1)
        expected_reward = self.env.config.step_cost + self.env.config.invalid_move_penalty
        self.assertAlmostEqual(self.env.last_transition_reward(), expected_reward, places=6)

    def test_battery_station_is_single_use(self) -> None:
        """Recharge should work once per station and then become unavailable."""

        battery_row, battery_col = sorted(self.env._map.battery_stations)[0]
        state = DroneState(row=battery_row, col=battery_col, battery=4, time_step=2)
        self.assertIn(Action.RECHARGE, self.env.available_actions(state))

        recharged_state, _ = self.env.step(state, Action.RECHARGE)
        self.assertEqual(recharged_state.battery, self.env.config.max_battery)
        self.assertIn((battery_row, battery_col), recharged_state.used_battery_stations)

        actions_after = self.env.available_actions(recharged_state)
        self.assertNotIn(Action.RECHARGE, actions_after)

    def test_reward_cell_collected_once(self) -> None:
        """Reward cells should be tracked and not double-counted."""

        reward_cell = sorted(self.env._map.reward_cells)[0]
        start_state, move_action = self._find_move_into_cell(reward_cell)
        next_state, _ = self.env.step(start_state, move_action)
        self.assertIn(reward_cell, next_state.collected_reward_cells)

        # Leaving and re-entering should keep one collected token for that cell.
        back_action = _reverse_action(move_action)
        back_state, _ = self.env.step(next_state, back_action)
        reenter_state, _ = self.env.step(back_state, move_action)
        occurrences = sum(1 for cell in reenter_state.collected_reward_cells if cell == reward_cell)
        self.assertEqual(occurrences, 1)

    def test_reward_tokens_apply_only_on_success(self) -> None:
        """Collected `R` value should be awarded only when mission succeeds."""

        self.env._active_hazard_cells = frozenset()
        self.env.reset()
        true_survivor = self.env._true_survivor_cell
        self.assertIsNotNone(true_survivor)

        collected = tuple(sorted(self.env._map.reward_cells)[:2])
        success_prev = DroneState(row=0, col=0, battery=5, time_step=3, collected_reward_cells=collected)
        success_next = DroneState(
            row=true_survivor[0],
            col=true_survivor[1],
            battery=4,
            time_step=4,
            collected_reward_cells=collected,
        )
        success_reward = self.env.transition_reward(success_prev, Action.MOVE_SOUTH, success_next)
        expected_success = (
            self.env.config.step_cost
            + self.env.config.goal_reward
            + len(collected) * self.env.config.reward_cell_value
        )
        self.assertAlmostEqual(success_reward, expected_success, places=6)

        failure_prev = DroneState(row=0, col=0, battery=1, time_step=5, collected_reward_cells=collected)
        failure_next = DroneState(row=0, col=1, battery=0, time_step=6, collected_reward_cells=collected)
        failure_reward = self.env.transition_reward(failure_prev, Action.MOVE_EAST, failure_next)
        expected_failure = self.env.config.step_cost + self.env.config.battery_depletion_penalty
        self.assertAlmostEqual(failure_reward, expected_failure, places=6)

    def test_battery_depletion_terminal_failure(self) -> None:
        """Battery reaching zero should terminate mission."""

        state = DroneState(row=0, col=0, battery=1, time_step=0)
        next_state, _ = self.env.step(state, Action.MOVE_EAST)
        self.assertEqual(next_state.battery, 0)
        self.assertTrue(self.env.is_terminal(next_state))
        self.assertEqual(self.env.available_actions(next_state), [])

    def _find_move_into_cell(self, target_cell: tuple[int, int]) -> tuple[DroneState, Action]:
        """Return legal state/action pair that enters `target_cell` in one step."""

        map_config = self.env._map
        target_row, target_col = target_cell
        for action, delta_row, delta_col in [
            (Action.MOVE_NORTH, -1, 0),
            (Action.MOVE_SOUTH, 1, 0),
            (Action.MOVE_EAST, 0, 1),
            (Action.MOVE_WEST, 0, -1),
        ]:
            start_row = target_row - delta_row
            start_col = target_col - delta_col
            if not map_config.in_bounds(start_row, start_col):
                continue
            if (start_row, start_col) in map_config.obstacles:
                continue
            state = DroneState(row=start_row, col=start_col, battery=8, time_step=0)
            if action in self.env.available_actions(state):
                return state, action
        raise AssertionError(f"No legal one-step entry found for cell {target_cell}.")


def _reverse_action(action: Action) -> Action:
    """Return opposite movement action."""

    if action == Action.MOVE_NORTH:
        return Action.MOVE_SOUTH
    if action == Action.MOVE_SOUTH:
        return Action.MOVE_NORTH
    if action == Action.MOVE_EAST:
        return Action.MOVE_WEST
    if action == Action.MOVE_WEST:
        return Action.MOVE_EAST
    raise ValueError(f"Action {action.value} is not reversible movement.")


if __name__ == "__main__":
    unittest.main()
