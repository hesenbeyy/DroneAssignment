"""Observation-model tests for survivor and hazard uncertainty."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_assignment.env import Action, DroneState, Observation, RescueDroneEnv
from drone_assignment.env.observation_model import hazard_warning_probability, survivor_signal_probability


class TestObservations(unittest.TestCase):
    """Validate scan outputs and distance-based probability behavior."""

    def setUp(self) -> None:
        """Create deterministic environment for observation tests."""

        maps_dir = Path(__file__).resolve().parents[1] / "maps"
        self.env = RescueDroneEnv(maps_dir / "map_2.txt", rng_seed=101)
        self.env.reset()
        # Force deterministic hidden truth for empirical checks.
        self.env._true_survivor_cell = sorted(self.env._map.survivors)[0]
        self.env._active_hazard_cells = frozenset(sorted(self.env._map.hazards)[:1])

    def test_non_scan_actions_return_none_observation(self) -> None:
        """Non-scan actions should emit `Observation.NONE`."""

        state = DroneState(row=0, col=0, battery=6, time_step=0)
        _, observation = self.env.step(state, Action.MOVE_EAST)
        self.assertEqual(observation, Observation.NONE)

    def test_scan_outputs_only_valid_symbols(self) -> None:
        """Scan should emit one of SURVIVOR_SIGNAL/HAZARD_WARNING/NO_SIGNAL."""

        state = DroneState(row=0, col=0, battery=8, time_step=0)
        valid = {Observation.SURVIVOR_SIGNAL, Observation.HAZARD_WARNING, Observation.NO_SIGNAL}
        seen: set[Observation] = set()
        for _ in range(300):
            _, observation = self.env.step(state, Action.SCAN)
            self.assertIn(observation, valid)
            seen.add(observation)
        self.assertTrue(seen.issuperset({Observation.NO_SIGNAL}))

    def test_survivor_signal_probability_decreases_with_distance(self) -> None:
        """Survivor signal model should be higher near true survivor than far away."""

        true_survivor = self.env._true_survivor_cell
        self.assertIsNotNone(true_survivor)
        near_state = DroneState(row=true_survivor[0], col=true_survivor[1], battery=5, time_step=0)
        far_state = DroneState(row=0, col=0, battery=5, time_step=0)
        p_near = survivor_signal_probability(near_state, true_survivor)
        p_far = survivor_signal_probability(far_state, true_survivor)
        self.assertGreater(p_near, p_far)
        self.assertAlmostEqual(p_near, 0.95, places=6)

    def test_hazard_warning_probability_decreases_with_distance(self) -> None:
        """Hazard warning model should be higher near active hazards."""

        hazard_cell = next(iter(self.env._active_hazard_cells))
        near_state = DroneState(row=hazard_cell[0], col=hazard_cell[1], battery=5, time_step=0)
        far_state = DroneState(row=0, col=0, battery=5, time_step=0)
        p_near = hazard_warning_probability(near_state, self.env._active_hazard_cells)
        p_far = hazard_warning_probability(far_state, self.env._active_hazard_cells)
        self.assertGreaterEqual(p_near, p_far)
        self.assertAlmostEqual(p_near, 0.85, places=6)

    def test_empirical_hazard_warning_rate_near_hazard_is_high(self) -> None:
        """Repeated scans near active hazard should produce non-trivial warnings."""

        hazard_cell = next(iter(self.env._active_hazard_cells))
        state = DroneState(row=hazard_cell[0], col=hazard_cell[1], battery=10, time_step=0)
        warnings = 0
        trials = 600
        for _ in range(trials):
            _, observation = self.env.step(state, Action.SCAN)
            if observation == Observation.HAZARD_WARNING:
                warnings += 1
        rate = warnings / trials
        # Survivor signals are prioritized in the observation model, so this
        # lower bound only checks that hazard warnings still occur regularly.
        self.assertGreater(rate, 0.05)


if __name__ == "__main__":
    unittest.main()
