"""Smoke test for student-style API usage without planner implementation."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_assignment.env import Action, RescueDroneEnv
from drone_assignment.env.viz import write_state_graph_dot


class TestStudentSmokeWorkflow(unittest.TestCase):
    """Validate a tiny end-to-end workflow that students are expected to perform."""

    def test_student_like_interaction_and_dot_export(self) -> None:
        """Import API, explore actions, build a tiny graph, and export DOT."""

        maps_dir = Path(__file__).resolve().parents[1] / "maps"
        env = RescueDroneEnv(maps_dir / "map_2.txt", rng_seed=44)

        state = env.reset()
        nodes = {env.state_id(state)}
        edges: list[tuple[str, str, str]] = []

        action_plan = [Action.SCAN, Action.MOVE_EAST, Action.MOVE_SOUTH]
        for action in action_plan:
            legal_actions = env.available_actions(state)
            self.assertIn(action, legal_actions)
            source_id = env.state_id(state)
            next_state, _ = env.step(state, action)
            target_id = env.state_id(next_state)
            nodes.add(target_id)
            edges.append((source_id, target_id, action.value))
            state = next_state

        output_dir = Path(__file__).resolve().parents[1] / "tests" / "_tmp_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "student_tiny_graph.dot"
        output.unlink(missing_ok=True)
        write_state_graph_dot(nodes, edges, output)
        self.assertTrue(output.exists())
        content = output.read_text(encoding="utf-8")
        self.assertIn("digraph state_graph", content)
        self.assertIn("SCAN", content)
        output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
