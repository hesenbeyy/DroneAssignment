"""Small instructor demo showing how to use the rescue-drone API."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from drone_assignment.env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from drone_assignment.env.viz import write_search_tree_dot, write_state_graph_dot
    from drone_assignment.student.planner_template import build_state_graph, build_search_tree, bayes_update
except ModuleNotFoundError:
    from env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from env.viz import write_search_tree_dot, write_state_graph_dot
    from student.planner_template import build_state_graph, build_search_tree, bayes_update

def _resolve_map_path(root: Path, requested_map: str | None) -> Path:
    """Resolve a map filename from `maps/`, with fallback for local map variants."""

    maps_dir = root / "maps"
    if requested_map:
        candidate = maps_dir / requested_map
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Map not found: {candidate}")

    preferred_order = (
        "map_1.txt",
        "map_2.txt",
        "map_3.txt",
        "map_4.txt",
    )
    for name in preferred_order:
        candidate = maps_dir / name
        if candidate.exists():
            return candidate

    available = sorted(path.name for path in maps_dir.glob("*.txt"))
    if available:
        return maps_dir / available[0]
    raise FileNotFoundError(f"No .txt maps found in {maps_dir}")


def _build_parser() -> argparse.ArgumentParser:
    """Return CLI parser for demo options."""

    parser = argparse.ArgumentParser(description="Run rescue-drone demo with optional config presets.")
    parser.add_argument(
        "--map",
        default=None,
        help="Map filename inside drone_assignment/maps (default: auto-select).",
    )
    parser.add_argument(
        "--config-set",
        default="DEFAULT",
        help=(
            "Environment config preset name (DEFAULT, SET_A, SET_B, SET_C; "
            "aliases A/B/C also supported)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for hidden survivor/hazard sampling.",
    )
    return parser


def main() -> None:
    """Run a short scripted interaction and export example DOT files."""

    root = Path(__file__).resolve().parent
    args = _build_parser().parse_args()
    map_path = _resolve_map_path(root, args.map)
    config: EnvironmentConfig = get_environment_config(args.config_set)
    env = RescueDroneEnv(map_path, config=config, rng_seed=args.seed)

    state = env.reset()
    graph_nodes, graph_edges = build_state_graph(env, state, max_depth=3, algorithm="bfs")   # or "dfs"
    tree_nodes, tree_edges = build_search_tree(env, state, depth_limit=3)
    print("Map:", map_path.name)
    print("Config set:", args.config_set)
    print("Config:", config)
    print("Initial state:", state)
    print("Initial state_id:", env.state_id(state))
    print("Initial legal actions:", [action.value for action in env.available_actions(state)])
    print(env.render(state))

    state_dot_path = root / "state_graph_example.dot"
    tree_dot_path = root / "search_tree_example.dot"
    write_state_graph_dot(graph_nodes, graph_edges, state_dot_path)
    write_search_tree_dot(tree_nodes, tree_edges, tree_dot_path)

    print(f"Wrote DOT file: {state_dot_path}")
    print(f"Wrote DOT file: {tree_dot_path}")
'''

    posterior = bayes_update(
    prior_survivor=0.5,
    observation="SURVIVOR_SIGNAL",
    p_signal_given_survivor_nearby=0.8,
    p_signal_given_no_survivor_nearby=0.2)

    print("Posterior:", posterior)
'''

if __name__ == "__main__":
    main()
