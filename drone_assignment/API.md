# Rescue Drone Planning Assignment

## Environment Provided

You are given an instructor-built simulator. Treat it as the source of truth for all rules.

### Map symbols
- `S`: start
- `.`: empty
- `X`: obstacle
- `B`: battery station (single use)
- `H`: possible hazard region
- `G`: possible survivor zone
- `R`: reward cell

### Visible state fields
- `row`, `col`
- `battery`
- `time_step`
- `used_battery_stations`
- `collected_reward_cells`

### Available actions
- `MOVE_NORTH`
- `MOVE_SOUTH`
- `MOVE_EAST`
- `MOVE_WEST`
- `SCAN`
- `RECHARGE`

### Observation outputs
- `SURVIVOR_SIGNAL`
- `HAZARD_WARNING`
- `NO_SIGNAL`
- `NONE` (for non-scan actions)

### API you should use

```python
state = env.reset()
actions = env.available_actions(state)
next_state, observation = env.step(state, action)
state_id = env.state_id(state)
```

Optional helpers:
- `env.is_terminal(state)`
- `env.render(state)`
- `env.last_transition_reward()`

## Config Presets for Comparison Runs

You can run the same planner under different environment parameter presets:

- `DEFAULT`
- `SET_A` (alias `A`)
- `SET_B` (alias `B`)
- `SET_C` (alias `C`)

Example:

```python
from drone_assignment.env import EnvironmentConfig, RescueDroneEnv

config = EnvironmentConfig.preset("SET_A")
env = RescueDroneEnv("drone_assignment/maps/map_1.txt", config=config, rng_seed=0)
```

## Implementation Required

Implement these functions in `student/planner_template.py`:

1. `build_state_graph(env, start_state, max_depth)`
2. `build_search_tree(env, start_state, depth_limit)`
3. `bayes_update(prior_survivor, observation, p_signal_given_survivor_nearby, p_signal_given_no_survivor_nearby)`
4. `choose_best_action(env, state, belief)`

### Required behavior
- Build a depth-limited **state graph** using `env.state_id(...)`.
- Build a depth-limited **search tree** from the current state.
- Perform a **Bayesian update** for survivor belief after scan observations.
- Choose only **legal actions** from `env.available_actions(state)`.
- Use an **expected-utility style** rule for action choice.

## Run and Check

Run demo:

```bash
python -m drone_assignment.run_demo
```

Run tests:

```bash
python -m unittest discover -s drone_assignment/tests -v
```
