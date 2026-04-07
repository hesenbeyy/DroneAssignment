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
    lookahead_depth: int = 3,
) -> tuple[Action, float]:
    """Choose action by expected utility with lookahead. Returns (Action, utility_value)."""

    def _manhattan_to_nearest(pos: tuple[int, int], cells: frozenset) -> int | None:
        if not cells:
            return None
        return min(abs(pos[0] - r) + abs(pos[1] - c) for r, c in cells)

    def _rollout_utility(
        s: DroneState,
        depth: int,
        visited_counts: dict,
        discount: float = 0.9,
    ) -> float:
        """Greedy rollout: accumulate discounted reward up to `depth` steps."""
        if depth == 0 or env.is_terminal(s):
            # Terminal heuristic: reward proximity to survivor zones
            dist = _manhattan_to_nearest(s.position, env._map.survivors)
            proximity_bonus = (10.0 / (dist + 1)) if dist is not None else 0.0
            battery_ratio = s.battery / env.config.max_battery
            return proximity_bonus + 5.0 * battery_ratio
        
        best = float("-inf")
        for a in env.available_actions(s):
            ns, _ = env.step(s, a)
            r = env.transition_reward(s, a, ns)

            pos = ns.position
            survivor_prob = belief.get(pos, 0.0)
            belief_bonus = survivor_prob * env.config.goal_reward if isinstance(survivor_prob, float) else 0.0

            visit_count = visited_counts.get(pos, 0)
            revisit_penalty = -3.0 * visit_count

            # Distance shaping: reward getting closer to survivor zones
            dist = _manhattan_to_nearest(pos, env._map.survivors)
            proximity_bonus = (8.0 / (dist + 1)) if dist is not None else 0.0

            # Battery conservation: penalize low battery far from recharge
            battery_urgency = 0.0
            if ns.battery <= 2:
                dist_to_b = _manhattan_to_nearest(pos, env._map.battery_stations)
                if dist_to_b is not None and dist_to_b > ns.battery:
                    battery_urgency = -20.0  # Can't reach recharge in time

            step_utility = r + belief_bonus + revisit_penalty + proximity_bonus + battery_urgency
            future = discount * _rollout_utility(ns, depth - 1, visited_counts, discount)
            total = step_utility + future
            if total > best:
                best = total

        return best

    visited: dict[tuple[int, int], int] = belief.get("visited", {})
    best_action = None
    best_utility = float("-inf")

    for action in env.available_actions(state):
        next_state, _ = env.step(state, action)
        r = env.transition_reward(state, action, next_state)

        pos = next_state.position
        survivor_prob = belief.get(pos, 0.0)
        belief_bonus = survivor_prob * env.config.goal_reward if isinstance(survivor_prob, float) else 0.0

        visit_count = visited.get(pos, 0)
        revisit_penalty = -3.0 * visit_count

        dist = _manhattan_to_nearest(pos, env._map.survivors)
        proximity_bonus = (8.0 / (dist + 1)) if dist is not None else 0.0

        battery_urgency = 0.0
        if next_state.battery <= 2:
            dist_to_b = _manhattan_to_nearest(pos, env._map.battery_stations)
            if dist_to_b is not None and dist_to_b > next_state.battery:
                battery_urgency = -20.0

        immediate = r + belief_bonus + revisit_penalty + proximity_bonus + battery_urgency
        future = 0.9 * _rollout_utility(next_state, lookahead_depth - 1, visited)
        utility = immediate + future

        if utility > best_utility:
            best_utility = utility
            best_action = action

    if best_action is None:
        best_action = env.available_actions(state)[0]
        best_utility = 0.0

    return best_action, best_utility


def student_notes() -> dict[str, Any]:
    return {"status": "Fixed merge conflicts and return types."}