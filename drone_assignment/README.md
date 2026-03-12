# Rescue Drone Assignment (Student Starter)

This package provides the instructor-built simulator and starter code for the rescue-drone planning assignment.

Students implement planner logic in:

- `drone_assignment/student/planner_template.py`

Students should not modify simulator internals in:

- `drone_assignment/env/`

## Quick Start

Run the demo:

```bash
python -m drone_assignment.run_demo
```

Run tests:

```bash
python -m unittest discover -s drone_assignment/tests -v
```

## Student-Facing API

```python
from drone_assignment.env import RescueDroneEnv

env = RescueDroneEnv("drone_assignment/maps/map_1.txt")
state = env.reset()
actions = env.available_actions(state)
next_state, observation = env.step(state, actions[0])
state_id = env.state_id(next_state)
```

Optional helpers:

- `env.is_terminal(state)`
- `env.render(state)`
- `env.last_transition_reward()`

## Map Symbols

- `S`: start
- `.`: empty
- `X`: obstacle
- `B`: battery station (single-use recharge)
- `H`: possible hazard region
- `G`: possible survivor zone
- `R`: reward cell

## Default Mission Constraints

Defined in `env/environment.py` (`EnvironmentConfig`):

- `max_time_steps=25`
- `initial_battery=10`
- `max_battery=10`
- `step_cost=-1`
- `scan_cost=-1`
- `scan_battery_usage=1`
- `recharge_amount="full"`
- `hazard_prior=0.35`
- `hazard_penalty=-15`
- `battery_depletion_penalty=-100`
- `goal_reward=100`
- `reward_cell_value=10`
- `reward_requires_success=True`
- `invalid_move_penalty=-3`

## Preset Parameter Sets

Students can compare planner behavior under multiple mission settings:

- `DEFAULT`
- `SET_A` (or `A`)
- `SET_B` (or `B`)
- `SET_C` (or `C`)

Python usage:

```python
from drone_assignment.env import EnvironmentConfig, RescueDroneEnv

config = EnvironmentConfig.preset("SET_B")
env = RescueDroneEnv("drone_assignment/maps/map_1.txt", config=config, rng_seed=0)
```

CLI demo usage:

```bash
python -m drone_assignment.run_demo --config-set SET_A
python -m drone_assignment.run_demo --config-set SET_B
python -m drone_assignment.run_demo --config-set SET_C
```

## Included Maps

Map files are plain-text grids under `drone_assignment/maps/` (for example `map_1.txt` to `map_4.txt`).

## DOT Visualization

`run_demo.py` writes Graphviz DOT files:

- `drone_assignment/state_graph_example.dot`
- `drone_assignment/search_tree_example.dot`

Render with Graphviz if installed:

```bash
dot -Tpng drone_assignment/state_graph_example.dot -o drone_assignment/state_graph_example.png
dot -Tpng drone_assignment/search_tree_example.dot -o drone_assignment/search_tree_example.png
```
