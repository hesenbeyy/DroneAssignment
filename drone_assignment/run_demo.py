"""Small instructor demo showing how to use the rescue-drone API."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from drone_assignment.env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from drone_assignment.env.viz import write_search_tree_dot, write_state_graph_dot
except ModuleNotFoundError:
    from env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from env.viz import write_search_tree_dot, write_state_graph_dot


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
    print("Map:", map_path.name)
    print("Config set:", args.config_set)
    print("Config:", config)
    print("Initial state:", state)
    print("Initial state_id:", env.state_id(state))
    print("Initial legal actions:", [action.value for action in env.available_actions(state)])
    print(env.render(state))

    scripted_actions = [
        Action.SCAN,
        Action.MOVE_EAST,
        Action.MOVE_EAST,
        Action.MOVE_EAST,
        Action.MOVE_SOUTH,
        Action.MOVE_SOUTH,
        Action.MOVE_WEST,
        Action.RECHARGE,
        Action.MOVE_EAST,
    ]

    state_graph_nodes: set[str] = {env.state_id(state)}
    state_graph_edges: list[tuple[str, str, str]] = []
    tree_nodes: list[tuple[str, str]] = [("n0", f"{env.state_id(state)}\\nt={state.time_step}")]
    tree_edges: list[tuple[str, str, str]] = []
    tree_node_index = 1
    parent_tree_node_id = "n0"

    for action in scripted_actions:
        legal_actions = env.available_actions(state)
        if action not in legal_actions:
            print(
                f"action={action.value:>10} skipped (illegal for current state); "
                f"legal={ [a.value for a in legal_actions] }"
            )
            break

        source_state_id = env.state_id(state)
        next_state, observation = env.step(state, action)
        target_state_id = env.state_id(next_state)

        state_graph_nodes.add(target_state_id)
        state_graph_edges.append((source_state_id, target_state_id, action.value))

        child_tree_node_id = f"n{tree_node_index}"
        tree_node_index += 1
        tree_nodes.append(
            (
                child_tree_node_id,
                f"{target_state_id}\\nt={next_state.time_step}\\nobs={observation.value}",
            )
        )
        tree_edges.append((parent_tree_node_id, child_tree_node_id, action.value))

        print(
            f"action={action.value:>10} -> state_id={target_state_id:>12} "
            f"obs={observation.value}"
        )

        parent_tree_node_id = child_tree_node_id
        state = next_state
        if env.is_terminal(state):
            print("Reached terminal state.")
            break

    state_dot_path = root / "state_graph_example.dot"
    tree_dot_path = root / "search_tree_example.dot"
    write_state_graph_dot(state_graph_nodes, state_graph_edges, state_dot_path)
    write_search_tree_dot(tree_nodes, tree_edges, tree_dot_path)

    print(f"Wrote DOT file: {state_dot_path}")
    print(f"Wrote DOT file: {tree_dot_path}")


if __name__ == "__main__":
    main()
