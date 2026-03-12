"""Tests for named `EnvironmentConfig` presets."""

from __future__ import annotations

import unittest

from drone_assignment.env import EnvironmentConfig, get_environment_config


class TestConfigPresets(unittest.TestCase):
    """Validate instructor-provided mission-parameter preset sets."""

    def test_preset_names_include_all_expected_sets(self) -> None:
        """Preset name listing should include default and all alternatives."""

        names = set(EnvironmentConfig.preset_names())
        self.assertEqual(names, {"DEFAULT", "SET_A", "SET_B", "SET_C"})

    def test_set_a_values_match_spec(self) -> None:
        """Alternative Set A should expose the expected parameter values."""

        config = get_environment_config("SET_A")
        self.assertEqual(config.max_time_steps, 25)
        self.assertEqual(config.initial_battery, 10)
        self.assertEqual(config.max_battery, 10)
        self.assertEqual(config.hazard_prior, 0.40)
        self.assertEqual(config.hazard_penalty, -35.0)
        self.assertEqual(config.battery_depletion_penalty, -120.0)
        self.assertEqual(config.invalid_move_penalty, -5.0)

    def test_alias_lookup(self) -> None:
        """Short aliases A/B/C should map to SET_A/SET_B/SET_C."""

        self.assertEqual(get_environment_config("A"), get_environment_config("SET_A"))
        self.assertEqual(get_environment_config("B"), get_environment_config("SET_B"))
        self.assertEqual(get_environment_config("C"), get_environment_config("SET_C"))

    def test_unknown_preset_raises(self) -> None:
        """Unknown preset names should raise a clear value error."""

        with self.assertRaises(ValueError):
            _ = get_environment_config("SET_Z")


if __name__ == "__main__":
    unittest.main()
