"""Regression tests for benchmark map semantics and structure."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from drone_assignment.env import DroneState, RescueDroneEnv
from drone_assignment.env.map_loader import load_map
from drone_assignment.env.observation_model import survivor_signal_probability


class TestMaps(unittest.TestCase):
    """Validate benchmark maps for assignment-intended behavior."""

    def test_map_4_loads_with_new_symbol_sets(self) -> None:
        """`map_4` should include core symbols including reward cells (`R`)."""

        map_path = Path(__file__).resolve().parents[1] / "maps" / "map_4.txt"
        config = load_map(map_path)
        self.assertGreaterEqual(config.rows, 8)
        self.assertLessEqual(config.rows, 12)
        self.assertGreaterEqual(config.cols, 8)
        self.assertLessEqual(config.cols, 12)
        self.assertEqual(config.start_position, (0, 0))
        self.assertGreaterEqual(len(config.obstacles), 8)
        self.assertGreaterEqual(len(config.hazards), 2)
        self.assertGreaterEqual(len(config.survivors), 2)
        self.assertGreaterEqual(len(config.battery_stations), 1)
        self.assertGreaterEqual(len(config.reward_cells), 1)

    def test_map_2_has_repeated_reachable_state_ids(self) -> None:
        """`map_2` should permit repeated reachable state IDs."""

        map_path = Path(__file__).resolve().parents[1] / "maps" / "map_2.txt"
        env = RescueDroneEnv(map_path, rng_seed=0)
        initial_state = env.reset()
        counts = self._explore_state_id_occurrences(env, initial_state, depth_limit=3)
        repeated_ids = [sid for sid, count in counts.items() if count > 1]
        self.assertTrue(repeated_ids)

    def test_map_1_makes_scan_meaningful(self) -> None:
        """`map_1` should produce different signal rates by distance."""

        map_path = Path(__file__).resolve().parents[1] / "maps" / "map_1.txt"
        config = load_map(map_path)
        true_survivor = sorted(config.survivors)[0]
        near_state = DroneState(row=true_survivor[0], col=true_survivor[1], battery=6, time_step=0)
        far_state = DroneState(row=0, col=0, battery=6, time_step=0)
        near_signal = survivor_signal_probability(near_state, true_survivor)
        far_signal = survivor_signal_probability(far_state, true_survivor)
        self.assertGreater(near_signal, far_signal)
        self.assertAlmostEqual(near_signal, 0.95, places=6)

    def _explore_state_id_occurrences(
        self,
        env: RescueDroneEnv,
        start_state: DroneState,
        depth_limit: int,
    ) -> Counter[str]:
        """Return state-id visit counts by expanding legal actions."""

        frontier: list[tuple[DroneState, int]] = [(start_state, 0)]
        counts: Counter[str] = Counter()
        while frontier:
            state, depth = frontier.pop()
            counts[env.state_id(state)] += 1
            if depth >= depth_limit or env.is_terminal(state):
                continue
            for action in env.available_actions(state):
                next_state, _ = env.step(state, action)
                frontier.append((next_state, depth + 1))
        return counts


if __name__ == "__main__":
    unittest.main()
