"""API-level tests for the instructor-provided rescue drone environment."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_assignment.env import Action, DroneState, RescueDroneEnv


class TestEnvironmentAPI(unittest.TestCase):
    """Validate stable student-facing API and environment-defined constraints."""

    def setUp(self) -> None:
        """Create fresh deterministic environment for each API test."""

        maps_dir = Path(__file__).resolve().parents[1] / "maps"
        self.env = RescueDroneEnv(maps_dir / "map_2.txt", rng_seed=17)

    def test_reset_returns_expected_initial_state_fields(self) -> None:
        """`reset` should initialize visible mission-tracking fields."""

        state = self.env.reset()
        self.assertEqual((state.row, state.col), self.env._map.start_position)
        self.assertEqual(state.battery, self.env.config.initial_battery)
        self.assertEqual(state.time_step, 0)
        self.assertEqual(state.used_battery_stations, tuple())
        self.assertEqual(state.collected_reward_cells, tuple())

    def test_constraints_are_defined_in_environment_config(self) -> None:
        """Mission constraints should live in env config and be readable."""

        config = self.env.config
        self.assertEqual(config.max_time_steps, 25)
        self.assertEqual(config.initial_battery, 10)
        self.assertEqual(config.max_battery, 10)
        self.assertEqual(config.goal_reward, 100.0)
        self.assertEqual(config.reward_cell_value, 10.0)
        self.assertTrue(config.reward_requires_success)

    def test_reset_samples_true_survivor_from_g_cells(self) -> None:
        """Hidden true survivor must always be chosen from map `G` cells."""

        survivors = set(self.env._map.survivors)
        self.assertGreaterEqual(len(survivors), 2)
        sampled: set[tuple[int, int]] = set()
        for _ in range(40):
            self.env.reset()
            sampled.add(self.env._true_survivor_cell)
        self.assertTrue(sampled.issubset(survivors))
        self.assertGreaterEqual(len(sampled), 2)

    def test_reset_samples_hidden_hazards_from_h_cells(self) -> None:
        """Active hidden hazards should be sampled only from possible `H` cells."""

        hazard_cells = set(self.env._map.hazards)
        seen_non_empty = False
        for _ in range(30):
            self.env.reset()
            active = set(self.env._active_hazard_cells)
            self.assertTrue(active.issubset(hazard_cells))
            if active:
                seen_non_empty = True
        self.assertTrue(seen_non_empty)

    def test_true_survivor_not_exposed_in_student_visible_api(self) -> None:
        """Student-visible state must not expose hidden true survivor."""

        state = self.env.reset()
        self.assertFalse(hasattr(state, "true_survivor_cell"))
        self.assertFalse(hasattr(self.env, "true_survivor_cell"))

    def test_state_id_distinguishes_used_battery_and_collected_rewards(self) -> None:
        """State identity should include single-use resource tracking fields."""

        base = DroneState(row=1, col=1, battery=5, time_step=3)
        used = DroneState(
            row=1,
            col=1,
            battery=5,
            time_step=9,
            used_battery_stations=((2, 3),),
        )
        rewarded = DroneState(
            row=1,
            col=1,
            battery=5,
            time_step=9,
            collected_reward_cells=((4, 4),),
        )
        self.assertNotEqual(self.env.state_id(base), self.env.state_id(used))
        self.assertNotEqual(self.env.state_id(base), self.env.state_id(rewarded))

    def test_terminal_conditions_follow_environment_rules(self) -> None:
        """Mission should end on true survivor, battery depletion, or max-time."""

        self.env.reset()
        true_survivor = self.env._true_survivor_cell
        self.assertIsNotNone(true_survivor)

        survivor_state = DroneState(
            row=true_survivor[0],
            col=true_survivor[1],
            battery=3,
            time_step=4,
        )
        self.assertTrue(self.env.is_terminal(survivor_state))

        depleted_state = DroneState(row=0, col=0, battery=0, time_step=2)
        self.assertTrue(self.env.is_terminal(depleted_state))
        self.assertEqual(self.env.available_actions(depleted_state), [])

        timeout_state = DroneState(
            row=0,
            col=0,
            battery=5,
            time_step=self.env.config.max_time_steps,
        )
        self.assertTrue(self.env.is_terminal(timeout_state))

    def test_available_actions_include_recharge_only_if_station_unused(self) -> None:
        """Recharge should be legal only on an unused battery station."""

        battery_row, battery_col = sorted(self.env._map.battery_stations)[0]
        unused_state = DroneState(
            row=battery_row,
            col=battery_col,
            battery=6,
            time_step=3,
        )
        self.assertIn(Action.RECHARGE, self.env.available_actions(unused_state))

        used_state = DroneState(
            row=battery_row,
            col=battery_col,
            battery=6,
            time_step=4,
            used_battery_stations=((battery_row, battery_col),),
        )
        self.assertNotIn(Action.RECHARGE, self.env.available_actions(used_state))


if __name__ == "__main__":
    unittest.main()
