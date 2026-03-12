"""Student-visible state object for the rescue drone simulator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DroneState:
    """Visible simulator state used by students for planning and search."""

    row: int
    col: int
    battery: int
    time_step: int
    used_battery_stations: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    collected_reward_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def position(self) -> tuple[int, int]:
        """Return the `(row, col)` position of the drone."""

        return (self.row, self.col)

    @property
    def used_battery_set(self) -> frozenset[tuple[int, int]]:
        """Return used battery stations as a set-like view."""

        return frozenset(self.used_battery_stations)

    @property
    def collected_reward_set(self) -> frozenset[tuple[int, int]]:
        """Return collected reward cells as a set-like view."""

        return frozenset(self.collected_reward_cells)
