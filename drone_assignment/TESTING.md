# Testing Guide

Use these commands from repository root.

## Run all provided tests

```bash
python -m unittest discover -s drone_assignment/tests -v
```

## Run only student workflow smoke test

```bash
python -m unittest drone_assignment.tests.test_student_smoke -v
```

## Run demo script

```bash
python -m drone_assignment.run_demo
```

The demo should print state transitions and write:

- `drone_assignment/state_graph_example.dot`
- `drone_assignment/search_tree_example.dot`
