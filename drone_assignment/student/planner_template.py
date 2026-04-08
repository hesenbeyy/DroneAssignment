"""Student starter template for planner-related assignment functions."""

from __future__ import annotations
from typing import Any
from collections import deque

try:
    from drone_assignment.env import Action, DroneState, RescueDroneEnv
    from drone_assignment.env import observation_model
except ModuleNotFoundError:
    from env import Action, DroneState, RescueDroneEnv
    import env.observation_model as observation_model


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
    probability_survivor = prior_survivor
    if observation == "SURVIVOR_SIGNAL":
        probability_observation_given_survivor = p_signal_given_survivor_nearby
        probability_observation_given_no_survivor = p_signal_given_no_survivor_nearby
    elif observation == "NO_SIGNAL":
        probability_observation_given_survivor = 1 - p_signal_given_survivor_nearby
        probability_observation_given_no_survivor = 1 - p_signal_given_no_survivor_nearby
    else:
        return prior_survivor

    probability_observation = (
        (probability_survivor * probability_observation_given_survivor)
        + ((1 - probability_survivor) * probability_observation_given_no_survivor)
    )
    return (
        (probability_survivor * probability_observation_given_survivor) / probability_observation
        if probability_observation > 0
        else prior_survivor
    )


def choose_best_action(
    env: RescueDroneEnv,
    state: DroneState,
    belief: dict[str, float],
    lookahead_depth: int = 4,
) -> tuple[Action, float]:
    """Choose action by expected utility with lookahead. Returns (Action, utility_value)."""

    def _manhattan_to_nearest(position: tuple[int, int], cells: frozenset) -> int | None:
        """Thin wrapper: converts a (row, col) position into a DroneState-like
        object so we can reuse observation_model._distance_to_nearest."""
        if not cells:
            return None
        # observation_model._distance_to_nearest expects an object with .row/.col
        class _PosProxy:
            def __init__(self, row, col):
                self.row = row
                self.col = col
        return observation_model._distance_to_nearest(_PosProxy(*position), cells)

    def _hazard_penalty(position: tuple[int, int]) -> float:
        """Use observation_model.hazard_warning_probability to penalise proximity
        to active hazards. Returns a negative penalty value."""
        class _PosProxy:
            def __init__(self, row, col):
                self.row = row
                self.col = col
        active_hazards = list(env._map.hazards) if hasattr(env._map, "hazards") else []
        p_hazard = observation_model.hazard_warning_probability(
            _PosProxy(*position), active_hazards
        )
        # Scale into a penalty: max ~-8.5 when p_hazard == 1.0
        return -10.0 * p_hazard

    def _step_utility(
        state: DroneState,
        action: Action,
        next_state: DroneState,
        visited_counts: dict,
    ) -> float:
        reward = env.transition_reward(state, action, next_state)
        position = next_state.position
        survivor_prob = belief.get(position, 0.0)
        belief_bonus = survivor_prob * env.config.goal_reward if isinstance(survivor_prob, float) else 0.0
        revisit_penalty = -3.0 * visited_counts.get(position, 0)
        distance = _manhattan_to_nearest(position, env._map.survivors)
        proximity_bonus = (8.0 / (distance + 1)) if distance is not None else 0.0
        hazard_penalty = _hazard_penalty(position)
        battery_urgency = 0.0
        if next_state.battery <= 2:
            distance_to_battery = _manhattan_to_nearest(position, env._map.battery_stations)
            if distance_to_battery is not None and distance_to_battery > next_state.battery:
                battery_urgency = env.config.battery_depletion_penalty * 0.2
        return reward + belief_bonus + revisit_penalty + proximity_bonus + hazard_penalty + battery_urgency

    def _rollout_utility(
        state: DroneState, depth: int, visited_counts: dict, discount: float = 0.9
    ) -> float:
        if depth == 0 or env.is_terminal(state):
            distance = _manhattan_to_nearest(state.position, env._map.survivors)
            proximity_bonus = (10.0 / (distance + 1)) if distance is not None else 0.0
            battery_ratio = state.battery / env.config.max_battery
            hazard_penalty = _hazard_penalty(state.position)
            return proximity_bonus + 5.0 * battery_ratio + hazard_penalty

        best = float("-inf")
        for action in env.available_actions(state):
            next_state, _ = env.step(state, action)
            total = _step_utility(state, action, next_state, visited_counts) + discount * _rollout_utility(
                next_state, depth - 1, visited_counts, discount
            )
            if total > best:
                best = total
        return best

    visited: dict[tuple[int, int], int] = belief.get("visited", {})
    best_action = None
    best_utility = float("-inf")

    for action in env.available_actions(state):
        next_state, _ = env.step(state, action)
        utility = _step_utility(state, action, next_state, visited) + 0.9 * _rollout_utility(
            next_state, lookahead_depth - 1, visited
        )
        if utility > best_utility:
            best_utility = utility
            best_action = action

    if best_action is None:
        best_action = env.available_actions(state)[0]
        best_utility = 0.0

    return best_action, best_utility


def student_notes() -> dict[str, Any]:
    return {"all good hocam"}