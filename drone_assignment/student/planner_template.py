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

    # raise NotImplementedError("Student task: implement search tree construction.")


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

    TODO:
    - Evaluate legal actions from `env.available_actions(state)`.
    - Use expected utility with your transition/belief assumptions.
    - Return one action that maximizes your objective.
    """

    raise NotImplementedError("Student task: implement expected-utility action choice.")


def student_notes() -> dict[str, Any]:
    """

    all good hocam

    
    """

    return {}
