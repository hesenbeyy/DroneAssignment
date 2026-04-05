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
    """Build a state transition graph from environment interactions."""
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
    """Build a search tree with unique node IDs for each expansion path."""

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

    """Update `P(survivor)` after a scan observation using Bayes' rule."""
    p_s = prior_survivor # P(S)

    if observation == "SURVIVOR_SIGNAL":
        p_o_given_s = p_signal_given_survivor_nearby # P(O|S)
        p_o_given_not_s = p_signal_given_no_survivor_nearby # P(O|not S)
    elif observation == "NO_SIGNAL":
        p_o_given_s = 1 - p_signal_given_survivor_nearby # P(O|S)
        p_o_given_not_s = 1 - p_signal_given_no_survivor_nearby # P(O|not S)

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
    """Choose an action by expected utility under the student's belief model."""
    
    # 1. Pull parameters dynamically from the environment configuration
    # Note: If your env stores this under a different name (e.g., env._config), update the prefix below.
    config = getattr(env, "config", getattr(env, "_config", None))
    
    # Provide fallbacks just in case the config object isn't attached during a specific test
    max_batt = config.max_battery if config else 10.0
    hazard_pen = config.hazard_penalty if config else -35.0
    depletion_pen = config.battery_depletion_penalty if config else -120.0
    invalid_move_pen = config.invalid_move_penalty if config else -5.0
    
    # 2. Base Heuristics 
    R_GOAL = 100.0
    C_STEP = 2.0
    R_REWARD_CELL = 5.0

    best_action = None
    best_utility = -float('inf')

    curr_pos = (state.row, state.col)
<<<<<<< HEAD
    
    # 3. BUG FIX: Extract 'visited' here so it is available for Action.SCAN and revisiting penalties
    visited = belief.get("visited", {})
=======
    visited = belief.get("visited", {})

    # Compute min distance from current position to nearest battery station (dynamic)
    min_dist_to_battery = min(
        abs(curr_pos[0] - b[0]) + abs(curr_pos[1] - b[1])
        for b in env._map.battery_stations
    ) if env._map.battery_stations else 0

    # Urgency threshold: need enough battery to actually reach the nearest station
    battery_urgent = state.battery <= min_dist_to_battery + 2
>>>>>>> f12612040dd8c5b640ae6766cbfc6cb7805985c8

    for action in env.available_actions(state):
        next_state, _ = env.step(state, action)
        next_pos = (next_state.row, next_state.col)

<<<<<<< HEAD
        # Force recharge if battery is at <= 80% of dynamic max capacity
        if action.value == "RECHARGE" and state.battery <= (max_batt * 0.8):
            return action
=======
        # Only early-return RECHARGE if battery is genuinely low (can't afford to explore)
        if action.value == "RECHARGE" and battery_urgent:
            return action, (env.config.max_battery - state.battery) * 50.0
>>>>>>> f12612040dd8c5b640ae6766cbfc6cb7805985c8

        p_survivor = belief.get(next_pos, 0.0)
        utility = p_survivor * R_GOAL

        # Hitting walls (staying in the same position)
        if action.value.startswith("MOVE") and next_pos == curr_pos:
            utility += invalid_move_pen # We ADD because the config penalty is already negative

        if action != Action.SCAN and action != Action.RECHARGE:
            utility -= C_STEP

        if next_pos in env._map.hazards:
            utility += hazard_pen # We ADD because the config penalty is already negative

        if next_pos in env._map.reward_cells:
            utility += R_REWARD_CELL

        # Attraction toward survivor zones not yet confirmed empty
        for g in env._map.survivors:
            g_belief = belief.get(g, 0.3)
            if g_belief > 0.1:
                dist_now  = abs(curr_pos[0] - g[0]) + abs(curr_pos[1] - g[1])
                dist_next = abs(next_pos[0] - g[0]) + abs(next_pos[1] - g[1])
                if dist_next < dist_now:
                    utility += g_belief * 20.0  # reward getting closer

        if next_state.battery <= 0:
<<<<<<< HEAD
            utility += depletion_pen # We ADD because the config penalty is already negative
        else:
            if env._map.battery_stations:
                min_distance = min(
                    abs(next_pos[0] - b[0]) + abs(next_pos[1] - b[1])
                    for b in env._map.battery_stations
                )
            else:
                min_distance = 0
                
            # Move closer to the station when battery is at <= 50% capacity
            if state.battery <= (max_batt * 0.5):
                utility += (max_batt - min_distance) * 150.0

        if action == Action.SCAN:
            current_p = belief.get(curr_pos, 0.3)
            scan_count = visited.get(curr_pos, 0)

            if scan_count >= 2:
                utility -= 500.0
                
            if current_p > 0.8 or current_p < 0.2:
                info_gain = -100.0
            else:
                info_gain = (current_p * (1 - current_p)) * 200
            
            # Use env.scan_cost if available, fallback to 0
            scan_cost = getattr(env, 'scan_cost', 0)
            utility += info_gain - scan_cost

        if action == Action.RECHARGE:
            if state.battery < max_batt and curr_pos in env._map.battery_stations:
                utility += (max_batt - state.battery) * 50
=======
            if next_pos not in env._map.battery_stations:
                utility -= 2000.0  # Avoid total depletion at all costs
            else:
                utility -= 3000.0  # Arriving at station with 0 battery still game-over
        else:
            min_distance = min(
                abs(next_pos[0] - b[0]) + abs(next_pos[1] - b[1])
                for b in env._map.battery_stations
            )
            if battery_urgent:
                #Move closer to the station before battery = 0
                utility += (env.config.max_battery - min_distance) * 150.0

        if action == Action.SCAN:
            if battery_urgent:
                utility = -2000.0  # Never scan when battery is critically low
            else:
                current_p = belief.get(curr_pos, 0.3)
                scan_count = visited.get(curr_pos, 0)

                if scan_count >= 2:
                    utility = -500.0
                if current_p > 0.8 or current_p < 0.2:
                    info_gain = -100.0
                else:
                    info_gain = (current_p * (1 - current_p)) * 200
               
                utility = info_gain - env.scan_cost

        if action == Action.RECHARGE:
            if state.battery < env.config.max_battery and curr_pos in env._map.battery_stations:
                utility = (env.config.max_battery - state.battery) * 50
>>>>>>> f12612040dd8c5b640ae6766cbfc6cb7805985c8
            else:
                utility -= 100.0

<<<<<<< HEAD
        # Penalize revisiting locations
        utility -= visited.get(next_pos, 0) * 15.0
=======
        utility -= visited.get(next_pos, 0) * 30.0
>>>>>>> f12612040dd8c5b640ae6766cbfc6cb7805985c8

        if utility > best_utility:
            best_utility = utility
            best_action = action

<<<<<<< HEAD
    return best_action if best_action else env.available_actions(state)[0]
=======
    return (best_action if best_action else env.available_actions(state)[0], best_utility)
>>>>>>> f12612040dd8c5b640ae6766cbfc6cb7805985c8


def student_notes() -> dict[str, Any]:
    """Notes from the student."""
    return {
        "status": "All good hocam",
        
    }