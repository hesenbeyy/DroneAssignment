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
    start_id = env.state_id(start_state)
    nodes = {start_id}
    edges = []
    visited = {start_id}
    frontier = deque([(start_state, 0)])

    while frontier:
        current_state, depth = frontier.popleft() if algorithm == 'bfs' else frontier.pop()
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
    root_id = "root"
    nodes = [(root_id, env.state_id(start_state))]
    edges = []
    frontier = deque([(root_id, start_state, 0)])

    while frontier:
        node_id, current_state, depth = frontier.popleft()
        if depth >= depth_limit or env.is_terminal(current_state):
            continue
        for action in env.available_actions(current_state):
            next_state, _ = env.step(current_state, action)
            child_id = f"{node_id}->{action.value}_{depth}_{len(nodes)}"
            nodes.append((child_id, env.state_id(next_state)))
            edges.append((node_id, child_id, action.value))
            frontier.append((child_id, next_state, depth + 1))
    return nodes, edges

def bayes_update(
    prior_survivor: float,
    observation: str,
    p_signal_given_survivor_nearby: float,
    p_signal_given_no_survivor_nearby: float,
) -> float:
    p_s = prior_survivor
    if observation == "SURVIVOR_SIGNAL":
        p_o_given_s = p_signal_given_survivor_nearby
        p_o_given_not_s = p_signal_given_no_survivor_nearby
    elif observation == "NO_SIGNAL":
        p_o_given_s = 1 - p_signal_given_survivor_nearby
        p_o_given_not_s = 1 - p_signal_given_no_survivor_nearby
    else:
        return prior_survivor

    p_o = (p_s * p_o_given_s) + ((1 - p_s) * p_o_given_not_s)
    return (p_s * p_o_given_s) / p_o if p_o > 0 else prior_survivor

def choose_best_action(
    env: RescueDroneEnv,
    state: DroneState,
    belief: dict[str, float],
) -> tuple[Action, float]:
    """Choose action by expected utility. Returns (Action, utility_value)."""
    config = getattr(env, "config", None)
    max_batt = config.max_battery if config else 10.0
    depletion_pen = config.battery_depletion_penalty if config else -120.0
    
    R_GOAL = 100.0
    visited = belief.get("visited", {})
    curr_pos = (state.row, state.col)

    # Calculate distance to nearest charging station
    min_dist_to_batt = min([abs(curr_pos[0]-b[0]) + abs(curr_pos[1]-b[1]) for b in env._map.battery_stations]) if env._map.battery_stations else 0
    battery_urgent = state.battery <= (min_dist_to_batt + 2)

    best_action = None
    best_utility = -float('inf')

    for action in env.available_actions(state):
        next_state, _ = env.step(state, action)
        next_pos = (next_state.row, next_state.col)
        
        # Base utility from belief
        utility = belief.get(next_pos, 0.3) * R_GOAL

        # Dynamic logic for actions
        if action == Action.RECHARGE:
            if curr_pos in env._map.battery_stations:
                utility = (max_batt - state.battery) * 50.0
                if battery_urgent: utility += 500.0 
            else:
                utility = -100.0
        
        elif action == Action.SCAN:
            if battery_urgent:
                utility = -1000.0
            else:
                p = belief.get(curr_pos, 0.3)
                utility = (p * (1-p) * 200.0) - getattr(env, 'scan_cost', 2.0)
                if visited.get(curr_pos, 0) >= 2: utility -= 500.0

        # Move penalties/rewards
        if next_pos == curr_pos and action.value.startswith("MOVE"):
            utility += getattr(config, 'invalid_move_penalty', -5.0)
        
        if next_pos in env._map.hazards:
            utility += getattr(config, 'hazard_penalty', -35.0)

        if next_state.battery <= 0:
            utility += depletion_pen * 10 # Massive penalty for dying

        # Exploration/Distance logic
        utility -= visited.get(next_pos, 0) * 20.0

        if utility > best_utility:
            best_utility = utility
            best_action = action

    return (best_action if best_action else Action.WAIT, float(best_utility))

def student_notes() -> dict[str, Any]:
    return {"status": "Fixed merge conflicts and return types."}