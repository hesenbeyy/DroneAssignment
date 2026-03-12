"""Tests for DOT export visualization helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_assignment.env.viz import write_search_tree_dot, write_state_graph_dot


class TestVizExport(unittest.TestCase):
    """Validate DOT file generation for graph and tree helpers."""

    def test_write_state_graph_dot_generates_file_and_content(self) -> None:
        """State graph helper should write a valid DOT file with nodes and edges."""

        output = self._output_dir() / "state_graph.dot"
        output.unlink(missing_ok=True)
        write_state_graph_dot(
            node_ids={"r0_c0_b8", "r0_c1_b7"},
            edges=[("r0_c0_b8", "r0_c1_b7", "MOVE_EAST")],
            output_path=output,
        )

        self.assertTrue(output.exists())
        content = output.read_text(encoding="utf-8")
        self.assertIn("digraph state_graph", content)
        self.assertIn("MOVE_EAST", content)
        output.unlink(missing_ok=True)

    def test_write_search_tree_dot_generates_file_and_content(self) -> None:
        """Search tree helper should write a valid DOT file with labeled nodes."""

        output = self._output_dir() / "search_tree.dot"
        output.unlink(missing_ok=True)
        write_search_tree_dot(
            nodes=[("n0", "root"), ("n1", "child")],
            edges=[("n0", "n1", "SCAN")],
            output_path=output,
        )

        self.assertTrue(output.exists())
        content = output.read_text(encoding="utf-8")
        self.assertIn("digraph search_tree", content)
        self.assertIn("shape=box", content)
        self.assertIn("SCAN", content)
        output.unlink(missing_ok=True)

    def _output_dir(self) -> Path:
        """Return a writable directory for test-generated artifacts."""

        output_dir = Path(__file__).resolve().parents[1] / "tests" / "_tmp_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


if __name__ == "__main__":
    unittest.main()
