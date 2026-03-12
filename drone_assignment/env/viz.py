"""DOT export helpers for state-transition graphs and search trees."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

Edge = tuple[str, str, str]
NodeWithLabel = tuple[str, str]


def write_state_graph_dot(
    node_ids: Iterable[str],
    edges: Iterable[Edge],
    output_path: str | Path,
    *,
    graph_name: str = "state_graph",
) -> Path:
    """Write a directed state-transition graph in Graphviz DOT format."""

    normalized_nodes = sorted(set(node_ids))
    normalized_edges = sorted(set(edges))
    lines: list[str] = [f"digraph {graph_name} {{", "  rankdir=LR;"]

    for node_id in normalized_nodes:
        escaped = _dot_escape(node_id)
        lines.append(f'  "{escaped}" [label="{escaped}"];')

    for source_id, target_id, label in normalized_edges:
        lines.append(
            f'  "{_dot_escape(source_id)}" -> "{_dot_escape(target_id)}" '
            f'[label="{_dot_escape(label)}"];'
        )

    lines.append("}")
    return _write_lines(output_path, lines)


def write_search_tree_dot(
    nodes: Iterable[NodeWithLabel],
    edges: Iterable[Edge],
    output_path: str | Path,
    *,
    graph_name: str = "search_tree",
) -> Path:
    """Write a search tree in Graphviz DOT format."""

    normalized_nodes = sorted(set(nodes))
    normalized_edges = sorted(set(edges))
    lines: list[str] = [f"digraph {graph_name} {{", "  rankdir=TB;"]

    for node_id, label in normalized_nodes:
        lines.append(
            f'  "{_dot_escape(node_id)}" [label="{_dot_escape(label)}", shape=box];'
        )

    for source_id, target_id, label in normalized_edges:
        lines.append(
            f'  "{_dot_escape(source_id)}" -> "{_dot_escape(target_id)}" '
            f'[label="{_dot_escape(label)}"];'
        )

    lines.append("}")
    return _write_lines(output_path, lines)


def _write_lines(output_path: str | Path, lines: list[str]) -> Path:
    """Write DOT content to disk and return the output path."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _dot_escape(value: str) -> str:
    """Escape a string for safe inclusion in Graphviz DOT labels."""

    return value.replace("\\", "\\\\").replace('"', '\\"')

