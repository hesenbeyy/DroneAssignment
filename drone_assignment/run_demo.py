"""Small instructor demo showing how to use the rescue-drone API."""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from pathlib import Path

try:
    from drone_assignment.env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from drone_assignment.env.viz import write_search_tree_dot, write_state_graph_dot
    from drone_assignment.student.planner_template import (build_state_graph, build_search_tree, bayes_update, choose_best_action)
except ModuleNotFoundError:
    from env import Action, EnvironmentConfig, RescueDroneEnv, get_environment_config
    from env.viz import write_search_tree_dot, write_state_graph_dot
    from student.planner_template import (build_state_graph, build_search_tree, bayes_update, choose_best_action)


def plot_environment_map(env: RescueDroneEnv):
    """Visualize the static environment map with all cell types."""
    rows, cols = env._map.rows, env._map.cols
    fig, ax = plt.subplots(figsize=(cols + 1, rows + 1))
    for r in range(rows + 1):
        ax.axhline(r, color='gray', linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c, color='gray', linewidth=0.5)
    symbol_colors = {'X': ('black', 'Obstacle'), 'H': ('orange', 'Hazard'),
                     'G': ('green', 'Survivor Zone'), 'B': ('blue', 'Battery Station'),
                     'R': ('gold', 'Reward Cell'), 'S': ('cyan', 'Start')}
    for r in range(rows):
        for c in range(cols):
            sym = env._map.symbol_at(r, c)
            if sym in symbol_colors:
                color, _ = symbol_colors[sym]
                ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color=color, alpha=0.6))
            ax.text(c + 0.5, rows - 1 - r + 0.5, sym, ha='center', va='center', fontsize=9, fontweight='bold')
    legend_patches = [mpatches.Patch(color=v[0], alpha=0.6, label=v[1]) for v in symbol_colors.values()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Environment Map: {env._map.name}")
    plt.tight_layout()
    plt.show()

def plot_drone_path(env: RescueDroneEnv, path: list[tuple[int, int]]):
    """
    Plot the drone's path on the environment map.
    Uses the same coordinate system as plot_environment_map:
    x=col, y=rows-1-r, ylim=(0,rows), NO invert_yaxis.
    """
    rows, cols = env._map.rows, env._map.cols
    fig, ax = plt.subplots(figsize=(max(6, cols + 1), max(5, rows + 1)))

    # Draw grid
    for r in range(rows + 1):
        ax.axhline(r, color='gray', linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c, color='gray', linewidth=0.5)

    # Draw all cell types — same transform (rows-1-r) as plot_environment_map
    cell_layers = [
        (env._map.obstacles,        'dimgray', 0.7, 'Obstacle'),
        (env._map.hazards,          'orange',  0.5, 'Hazard'),
        (env._map.survivors,        'green',   0.5, 'Survivor Zone'),
        (env._map.battery_stations, 'blue',    0.5, 'Battery Station'),
        (env._map.reward_cells,     'gold',    0.5, 'Reward Cell'),
    ]
    for cells, color, alpha, label in cell_layers:
        for r, c in cells:
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color=color, alpha=alpha, label=label))

    # Mark start cell
    sr, sc = env._map.start_position
    ax.add_patch(plt.Rectangle((sc, rows - 1 - sr), 1, 1, color='cyan', alpha=0.6, label='Start'))

    # Shade visited cells
    for r, c in set(path):
        ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, color='green', alpha=0.25, label='Visited'))

    # Plot path through cell centers
    path_xs = [c + 0.5 for r, c in path]
    path_ys = [rows - 1 - r + 0.5 for r, c in path]
    ax.plot(path_xs, path_ys, marker='o', color='green', linewidth=2, markersize=8, label='Drone Path')

    # Start and end labels
    ax.text(path_xs[0],  path_ys[0],  'Start', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(path_xs[-1], path_ys[-1], 'End',   ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)  # NO invert_yaxis — rows-1-r already places row 0 at top
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Drone Path Visualization")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

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
    path = [(state.row, state.col)]
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

# ... (previous code in main)
    print("\nChoosing best action")
    belief = {"visited": {}}
    for row in range(env._map.rows):
        for col in range(env._map.cols):
            belief[(row, col)] = 0.3

    plot_environment_map(env)

    step_count = 0
    while not env.is_terminal(state):
        # Unpack the tuple correctly
        action, utility = choose_best_action(env, state, belief)

        # Apply action
        state, obs = env.step(state, action)
        path.append((state.row, state.col))

        # Update visited count
        pos = (state.row, state.col)
        belief["visited"][pos] = belief["visited"].get(pos, 0) + 1

        # Bayes Update
        if action == Action.SCAN and obs in ("SURVIVOR_SIGNAL", "NO_SIGNAL"):
            belief[pos] = bayes_update(
                prior_survivor=belief[pos],
                observation=obs,
                p_signal_given_survivor_nearby=0.8,
                p_signal_given_no_survivor_nearby=0.2,
            )
        
        print(f"Step {step_count+1}: {action.value} | Utility: {utility:.2f} | Batt: {state.battery}")
        # print(env.render(state)) # Optional: uncomment for text grid each step
        step_count += 1
        
    print(f"Mission finished in {step_count} steps.")
    plot_drone_path(env, path)
    
if __name__ == "__main__":
    main()