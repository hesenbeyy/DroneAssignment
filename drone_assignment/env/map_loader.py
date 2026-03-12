"""Map loading utilities for the rescue drone simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

Coordinate = tuple[int, int]
ALLOWED_SYMBOLS = {"S", ".", "X", "H", "G", "B", "R"}


@dataclass(frozen=True, slots=True)
class MapConfig:
    """Parsed map data used internally by the simulator.

    Note: `survivors` stores coordinates of possible survivor zones (`G` cells).
    The environment samples one hidden true survivor from these zones per episode.
    """

    name: str
    grid: tuple[str, ...]
    rows: int
    cols: int
    start_position: Coordinate
    obstacles: frozenset[Coordinate]
    hazards: frozenset[Coordinate]
    survivors: frozenset[Coordinate]
    battery_stations: frozenset[Coordinate]
    reward_cells: frozenset[Coordinate]

    def in_bounds(self, row: int, col: int) -> bool:
        """Return whether `(row, col)` is inside map boundaries."""

        return 0 <= row < self.rows and 0 <= col < self.cols

    def symbol_at(self, row: int, col: int) -> str:
        """Return the map symbol at the given coordinate."""

        return self.grid[row][col]


def load_map(map_path: str | Path) -> MapConfig:
    """Load a plain-text map and return a validated `MapConfig`."""

    path = Path(map_path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError(f"Map file is empty: {path}")

    width = len(lines[0])
    if width == 0:
        raise ValueError(f"Map rows must not be empty: {path}")

    start_positions: list[Coordinate] = []
    obstacles: set[Coordinate] = set()
    hazards: set[Coordinate] = set()
    survivors: set[Coordinate] = set()
    battery_stations: set[Coordinate] = set()
    reward_cells: set[Coordinate] = set()

    for row_idx, row_text in enumerate(lines):
        if len(row_text) != width:
            raise ValueError(
                f"Map must be rectangular. Row 0 has width {width}, "
                f"row {row_idx} has width {len(row_text)}."
            )
        for col_idx, symbol in enumerate(row_text):
            if symbol not in ALLOWED_SYMBOLS:
                raise ValueError(
                    f"Unsupported map symbol '{symbol}' at ({row_idx}, {col_idx})."
                )
            coord = (row_idx, col_idx)
            if symbol == "S":
                start_positions.append(coord)
            elif symbol == "X":
                obstacles.add(coord)
            elif symbol == "H":
                hazards.add(coord)
            elif symbol == "G":
                survivors.add(coord)
            elif symbol == "B":
                battery_stations.add(coord)
            elif symbol == "R":
                reward_cells.add(coord)

    if len(start_positions) != 1:
        raise ValueError(
            f"Map must contain exactly one start cell 'S'. Found {len(start_positions)}."
        )

    return MapConfig(
        name=path.stem,
        grid=tuple(lines),
        rows=len(lines),
        cols=width,
        start_position=start_positions[0],
        obstacles=frozenset(obstacles),
        hazards=frozenset(hazards),
        survivors=frozenset(survivors),
        battery_stations=frozenset(battery_stations),
        reward_cells=frozenset(reward_cells),
    )
