from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NgaMode = Literal["none", "two_point", "elite_heavy_mutation", "staged_hypermutation"]
SolverMode = Literal["classic", "restart_rescue"]


@dataclass(slots=True)
class ProblemInstance:
    items: list[str]
    prices: list[int]
    hidden_vector: list[int]
    target_sum: int
    generation_mode: str


@dataclass(slots=True)
class GeneticAlgorithmConfig:
    solver_mode: SolverMode = "classic"
    population_size: int = 100
    generations: int = 200
    stagnation: int = 30
    repeat_limit: int | None = None
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    tournament_size: int = 3
    nga_mode: NgaMode = "none"
    nga_mutation_fraction: float = 0.4
    nga_trigger_points: tuple[int, ...] = ()
    nga_mutate_points: tuple[int, ...] = (40,)
    restart_max_count: int | None = None
    rescue_min_mutated_bits_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.solver_mode not in ("classic", "restart_rescue"):
            raise ValueError("solver_mode must be one of: classic, restart_rescue")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.stagnation < 1:
            raise ValueError("stagnation must be at least 1")
        if self.repeat_limit is not None and self.repeat_limit < 1:
            raise ValueError("repeat_limit must be at least 1")
        if self.restart_max_count is not None and self.restart_max_count < 0:
            raise ValueError("restart_max_count must be at least 0")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.nga_mode not in ("none", "two_point", "elite_heavy_mutation", "staged_hypermutation"):
            raise ValueError(
                "nga_mode must be one of: none, two_point, elite_heavy_mutation, staged_hypermutation"
            )
        for name, value in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("nga_mutation_fraction", self.nga_mutation_fraction),
            ("rescue_min_mutated_bits_ratio", self.rescue_min_mutated_bits_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.solver_mode == "classic":
            if self.nga_mode in ("two_point", "elite_heavy_mutation") and self.repeat_limit is None:
                raise ValueError("repeat_limit must be provided when nga_mode is two_point or elite_heavy_mutation")
            if self.nga_mode == "staged_hypermutation":
                if self.repeat_limit is not None:
                    raise ValueError("repeat_limit is not used in staged_hypermutation mode")
                if not self.nga_trigger_points:
                    raise ValueError("nga_trigger_points must be provided for staged_hypermutation mode")
                previous_point = 0
                for point in self.nga_trigger_points:
                    if point < 1:
                        raise ValueError("nga_trigger_points must contain only positive integers")
                    if point >= self.stagnation:
                        raise ValueError("each nga_trigger_point must be less than stagnation")
                    if point <= previous_point:
                        raise ValueError("nga_trigger_points must be strictly increasing")
                    previous_point = point
                if not self.nga_mutate_points:
                    raise ValueError("nga_mutate_points must be provided for staged_hypermutation mode")
                if len(self.nga_mutate_points) not in (1, len(self.nga_trigger_points)):
                    raise ValueError(
                        "nga_mutate_points must contain either one value or match nga_trigger_points length"
                    )
                for mutate_percent in self.nga_mutate_points:
                    if not 0 <= mutate_percent <= 100:
                        raise ValueError("nga_mutate_points must be between 0 and 100")
        else:
            if self.nga_mode != "none":
                raise ValueError("nga_mode is only available in classic solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is only available in classic solver_mode")


@dataclass(slots=True)
class GeneticAlgorithmResult:
    best_vector: list[int]
    best_sum: int
    fitness: int
    difference: int
    generations_used: int
    exact_match: bool
    stop_reason: str
    nga_used: bool = False
    nga_trigger_generation: int | None = None
    nga_trigger_generations: list[int] = field(default_factory=list)
    restart_count: int = 0
    rescue_used: bool = False
