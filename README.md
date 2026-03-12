# CNGB242 Assignment 1: Rescue Drone Planning

This repository contains a rescue-drone planning assignment with:

- A provided simulation environment (`drone_assignment/env/`)
- Student implementation template (`drone_assignment/student/planner_template.py`)
- Maps, tests, and API documentation
- Optional instructor reference code (`instructor_solution/`)

## Repository Layout

- `drone_assignment/`: assignment package used by students
- `drone_assignment/API.md`: student-facing API and task requirements
- `drone_assignment/tests/`: automated tests
- `instructor_solution/`: reference implementation and experiment scripts

## Quick Start

Run the demo:

```bash
python -m drone_assignment.run_demo
```

Run tests:

```bash
python -m unittest discover -s drone_assignment/tests -v
```

## What Students Implement

Edit:

- `drone_assignment/student/planner_template.py`

Do not modify simulator internals in:

- `drone_assignment/env/`

## Notes for GitHub Upload

- Generated outputs and caches are ignored via `.gitignore`.
- If publishing publicly, remove `instructor_solution/` before pushing.
