"""Student starter template for planner-related assignment functions."""

from __future__ import annotations

from typing import Any
from collections import deque

try:
    from drone_assignment.env import Action, DroneState, RescueDroneEnv
except ModuleNotFoundError:
    from env import Action, DroneState, RescueDroneEnv


def build_state_graph(
    env: RescueDroneEnv,
    start_state: DroneState,
    max_depth: int,
    algorithm: str = "bfs"
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Build a state transition graph from environment interactions.

    TODO:
    - Explore actions from each discovered state.
    - Use `env.state_id(state)` for node identity.
    - Add directed edges labeled by action names.
    """
    start_id = env.state_id(start_state)
    nodes = {start_id}
    edges = []
    visited = {start_id}

    frontier = deque()
    frontier.append((start_state, 0))

    while frontier:
        if algorithm == 'bfs':
            current_state, depth = frontier.popleft()
        elif algorithm == 'dfs':
            current_state, depth = frontier.pop()
        else:
            raise ValueError("Unknown algorithm: must be 'bfs' or 'dfs'")

        if depth >= max_depth or env.is_terminal(current_state):
            continue

        current_state_id = env.state_id(current_state)
        for action in env.available_actions(current_state):
            next_state, _ = env.step(current_state, action)
            next_id = env.state_id(next_state)

       

            edges.append((current_state_id, next_id, action.value))
            nodes.add(next_id)

            if next_id not in visited:
                visited.add(next_id)
                frontier.append((next_state, depth + 1))

    return nodes, edges


def build_search_tree(
    env: RescueDroneEnv,
    start_state: DroneState,
    depth_limit: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Build a search tree with unique node IDs for each expansion path.

    TODO:
    - Keep parent->child edges for each chosen action.
    - Allow duplicate environment states in different tree branches.
    - Return `(nodes, edges)` in the format expected by `env.viz`.
    """

    root_id = "root"
    root_env_id = env.state_id(start_state)

    nodes: list[tuple[str, str]] = [(root_id, root_env_id)]
    edges: list[tuple[str, str, str]] = []

    frontier = deque()
    frontier.append((root_id, start_state, 0))

    while frontier:
        node_id, current_state, depth = frontier.popleft()
        if depth >= depth_limit or env.is_terminal(current_state):
            continue

        for action in env.available_actions(current_state):
            next_state, _ = env.step(current_state, action)

            child_id = f"{node_id}->{action.value}"
            child_env_id = env.state_id(next_state)

            nodes.append((child_id, child_env_id))
            edges.append((node_id, child_id, action.value))

            frontier.append((child_id, next_state, depth + 1))

    return nodes, edges

def bayes_update(
    prior_survivor: float,
    observation: str,
    p_signal_given_survivor_nearby: float,
    p_signal_given_no_survivor_nearby: float,
) -> float:
    """Update `P(survivor)` after a scan observation using Bayes' rule.
    
    TODO:
    - Implement posterior computation for SURVIVOR_SIGNAL and NO_SIGNAL observations.
    - Return the posterior probability in `[0.0, 1.0]`.
    """
    p_s = prior_survivor #P(S)
    if observation == "SURVIVOR_SIGNAL":
        p_o_given_s = p_signal_given_survivor_nearby # P(O/S)
        p_o_given_not_s = p_signal_given_no_survivor_nearby # P(O/not S)
    elif observation == "NO_SIGNAL":
        p_o_given_s = 1 - p_signal_given_survivor_nearby # P(O/S)
        p_o_given_not_s = 1- p_signal_given_no_survivor_nearby # P(O/not S)
    else:
        raise ValueError("Unknown observation")

    p_not_s = 1 - p_s
    p_o = p_s * p_o_given_s + p_not_s * p_o_given_not_s

    if p_o != 0:
        posterior = (p_s * p_o_given_s) / p_o
    else:
        return prior_survivor

    return posterior


def choose_best_action(
    env: RescueDroneEnv,
    state: DroneState,
    belief: dict[str, float],
) -> Action:
    """Choose an action by expected utility under the student's belief model.

    Core formula (Appendix H):
        U(s) = P(success) * R_goal - C_path - C_hazard + R_reward

    Battery feasibility is handled as a hard constraint / large penalty.
    """
    R_GOAL = env.config.goal_reward
    C_STEP = abs(env.config.step_cost)
    C_HAZARD = abs(env.config.hazard_penalty)
    R_REWARD_CELL = env.config.reward_cell_value
    SCAN_COST = abs(env.config.scan_cost)
    C_BATTERY_DEPLETION = abs(env.config.battery_depletion_penalty)
    PRIOR = env.config.hazard_prior

    best_action = None
    best_utility = -float('inf')

    curr_pos = (state.row, state.col)
    visited = belief.get("visited", {})

    # Minimum Manhattan distance to any unused battery station from current pos
    unused_stations = [
        b for b in env._map.battery_stations
        if b not in state.used_battery_set
    ]
    min_dist_to_battery = min(
        abs(curr_pos[0] - b[0]) + abs(curr_pos[1] - b[1])
        for b in unused_stations
    ) if unused_stations else env.config.max_battery  # no station available → treat as far

    # Battery is urgent when remaining charge can barely reach the nearest station
    battery_urgent = state.battery <= min_dist_to_battery + 1

    for action in env.available_actions(state):
        next_state, _ = env.step(state, action)
        next_pos = (next_state.row, next_state.col)

        # --- P(success) * R_goal  (survivor belief at next cell) ---
        p_survivor = belief.get(next_pos, 0.0)
        utility = p_survivor * R_GOAL

        # --- C_path: movement / scan step cost ---
        if action != Action.RECHARGE:
            utility -= C_STEP

        # Extra penalty for bumping into a wall (position unchanged after move)
        if action.value.startswith("MOVE") and next_pos == curr_pos:
            utility += env.config.invalid_move_penalty  # invalid_move_penalty is already negative

        # --- C_hazard: expected hazard cost at next position ---
        if next_pos in env._map.hazards:
            p_hazard = belief.get(next_pos, PRIOR)  # use belief if updated, else prior
            utility -= p_hazard * C_HAZARD

        # --- R_reward: reward cell bonus ---
        if next_pos in env._map.reward_cells and next_pos not in state.collected_reward_set:
            utility += R_REWARD_CELL

        # --- Attraction toward survivor zones not yet confirmed empty ---
        for g in env._map.survivors:
            g_belief = belief.get(g, PRIOR)
            if g_belief > 0.05:
                dist_now  = abs(curr_pos[0] - g[0]) + abs(curr_pos[1] - g[1])
                dist_next = abs(next_pos[0] - g[0]) + abs(next_pos[1] - g[1])
                if dist_next < dist_now:
                    # Scale attraction by belief strength and goal reward
                    utility += g_belief * (R_GOAL * 0.15)

        # --- Battery feasibility penalty ---
        if next_state.battery <= 0:
            # Running out of battery is catastrophic
            utility -= C_BATTERY_DEPLETION
        elif battery_urgent and unused_stations:
            # When urgent, only reward moves that strictly reduce distance to station.
            # Clamp to >= 0 so moving *away* from station never gets a bonus (fixes oscillation).
            min_dist_next = min(
                abs(next_pos[0] - b[0]) + abs(next_pos[1] - b[1])
                for b in unused_stations
            )
            dist_improvement = max(0, min_dist_to_battery - min_dist_next)
            utility += dist_improvement * (C_BATTERY_DEPLETION / env.config.max_battery)
            # Extra penalty for moving away when urgent — breaks symmetry that causes oscillation
            if min_dist_next > min_dist_to_battery:
                utility -= C_STEP * 3

        # --- Safety check: don't grab reward cell if it risks battery depletion ---
        if next_pos in env._map.reward_cells and env.config.reward_requires_success:
            # Only count reward if we can still feasibly reach the survivor after
            if unused_stations:
                dist_reward_to_station = min(
                    abs(next_pos[0] - b[0]) + abs(next_pos[1] - b[1])
                    for b in unused_stations
                )
                # If battery after moving won't cover distance to station, discount the reward
                if next_state.battery < dist_reward_to_station:
                    utility -= R_REWARD_CELL  # cancel the bonus added earlier

        # --- SCAN action: override with information-gain based utility ---
        if action == Action.SCAN:
            if battery_urgent:
                utility = -C_BATTERY_DEPLETION  # Never scan when battery is critically low
            else:
                current_p = belief.get(curr_pos, PRIOR)
                scan_count = visited.get(curr_pos, 0)
                # Information gain peaks when belief is near 0.5; diminishes at extremes
                info_gain = (current_p * (1.0 - current_p)) * R_GOAL
                if scan_count >= 2:
                    info_gain *= 0.1  # Heavily discount repeated scans at same cell
                utility = info_gain - SCAN_COST

        # --- RECHARGE action: utility based on battery recovered ---
        if action == Action.RECHARGE:
            if curr_pos in env._map.battery_stations and state.battery < env.config.max_battery:
                battery_gained = env.config.max_battery - state.battery
                # Recharge value scales with how much battery we recover
                utility = battery_gained * (C_BATTERY_DEPLETION / env.config.max_battery)
                if not battery_urgent:
                    # Small discount if not urgent — exploring is still preferred
                    utility *= 0.5
            else:
                utility = -C_BATTERY_DEPLETION

        # --- Exploration penalty: discourage revisiting cells ---
        # Use 3x step cost so it's strong enough to break symmetric oscillation
        utility -= visited.get(next_pos, 0) * C_STEP * 3

        if utility > best_utility:
            best_utility = utility
            best_action = action

    return (best_action if best_action else env.available_actions(state)[0], best_utility)


def student_notes() -> dict[str, Any]:
    """

    all good hocam

    """

    return {}
