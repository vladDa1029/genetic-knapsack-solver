from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProblemInstance:
    items: list[str]
    prices: list[int]
    hidden_vector: list[int]
    target_sum: int
    generation_mode: str


@dataclass(slots=True)
class GeneticAlgorithmConfig:
    population_size: int = 100
    generations: int = 200
    stagnation: int = 30
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    tournament_size: int = 3

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.stagnation < 1:
            raise ValueError("stagnation must be at least 1")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        for name, value in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass(slots=True)
class GeneticAlgorithmResult:
    best_vector: list[int]
    best_sum: int
    fitness: int
    difference: int
    generations_used: int
    exact_match: bool
    stop_reason: str
