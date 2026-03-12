"""Student starter template for planner-related assignment functions."""

from __future__ import annotations

from typing import Any

try:
    from drone_assignment.env import Action, DroneState, RescueDroneEnv
except ModuleNotFoundError:
    from env import Action, DroneState, RescueDroneEnv


def build_state_graph(
    env: RescueDroneEnv,
    start_state: DroneState,
    max_depth: int,
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Build a state transition graph from environment interactions.

    TODO:
    - Explore actions from each discovered state.
    - Use `env.state_id(state)` for node identity.
    - Add directed edges labeled by action names.
    """

    raise NotImplementedError("Student task: implement state graph construction.")


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

    raise NotImplementedError("Student task: implement search tree construction.")


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

    raise NotImplementedError("Student task: implement Bayesian update.")


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
    """Provide a small scratch space for student debugging metadata.

    TODO:
    - Optionally store counters, diagnostics, or experiment settings here.
    """

    return {}
