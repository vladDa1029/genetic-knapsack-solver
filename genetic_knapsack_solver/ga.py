from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import Random
from typing import Callable

from genetic_knapsack_solver.generator import dot_product
from genetic_knapsack_solver.models import GeneticAlgorithmConfig, GeneticAlgorithmResult


@dataclass(slots=True)
class _RunResult:
    best_vector: list[int]
    best_sum: int
    best_difference: int
    generations_used: int
    stop_reason: str
    exact_match: bool
    nga_used: bool
    nga_trigger_generation: int | None


def evaluate_vector(
    prices: list[int],
    target_sum: int,
    vector: list[int],
) -> tuple[int, int]:
    current_sum = dot_product(prices, vector)
    difference = abs(target_sum - current_sum)
    return difference, current_sum


def fitness(prices: list[int], target_sum: int, vector: list[int]) -> int:
    difference, _ = evaluate_vector(prices, target_sum, vector)
    return difference


def build_initial_population(
    gene_count: int,
    population_size: int,
    rng: Random,
) -> list[list[int]]:
    population: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    repeated_attempts = 0

    while len(population) < population_size:
        bias = rng.uniform(0.35, 0.65)
        candidate = [1 if rng.random() < bias else 0 for _ in range(gene_count)]
        signature = tuple(candidate)

        if signature in seen and repeated_attempts < population_size * 3:
            repeated_attempts += 1
            continue

        seen.add(signature)
        population.append(candidate)

    return population


def crossover(
    first: list[int],
    second: list[int],
    rng: Random,
) -> tuple[list[int], list[int]]:
    if len(first) < 2:
        return first[:], second[:]

    point = rng.randint(1, len(first) - 1)
    child_a = first[:point] + second[point:]
    child_b = second[:point] + first[point:]
    return child_a, child_b


def crossover_two_points(
    first: list[int],
    second: list[int],
    rng: Random,
) -> tuple[list[int], list[int]]:
    if len(first) < 3:
        return crossover(first, second, rng)

    left, right = sorted(rng.sample(range(1, len(first)), 2))
    child_a = first[:left] + second[left:right] + first[right:]
    child_b = second[:left] + first[left:right] + second[right:]
    return child_a, child_b


def mutate(vector: list[int], mutation_rate: float, rng: Random) -> list[int]:
    mutated = vector[:]
    if mutated and rng.random() < mutation_rate:
        index = rng.randrange(len(mutated))
        mutated[index] = 1 - mutated[index]
    return mutated


def mutate_two_points(vector: list[int], mutation_rate: float, rng: Random) -> list[int]:
    mutated = vector[:]
    if not mutated or rng.random() >= mutation_rate:
        return mutated

    if len(mutated) == 1:
        mutated[0] = 1 - mutated[0]
        return mutated

    for index in rng.sample(range(len(mutated)), 2):
        mutated[index] = 1 - mutated[index]
    return mutated


def mutate_many_bits(
    vector: list[int],
    mutation_fraction: float,
    rng: Random,
) -> list[int]:
    mutated = vector[:]
    if not mutated or mutation_fraction <= 0.0:
        return mutated

    flip_count = min(len(mutated), max(1, ceil(len(mutated) * mutation_fraction)))
    for index in rng.sample(range(len(mutated)), flip_count):
        mutated[index] = 1 - mutated[index]
    return mutated


def _evaluate_population(
    population: list[list[int]],
    prices: list[int],
    target_sum: int,
) -> list[tuple[int, int]]:
    return [evaluate_vector(prices, target_sum, vector) for vector in population]


def _pick_parent(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    tournament_size: int,
    rng: Random,
) -> list[int]:
    contestant_count = min(tournament_size, len(population))
    contestant_indices = [rng.randrange(len(population)) for _ in range(contestant_count)]
    best_index = min(contestant_indices, key=lambda index: evaluations[index][0])
    return population[best_index]


def _rank_population_indices(evaluations: list[tuple[int, int]]) -> list[int]:
    return sorted(
        range(len(evaluations)),
        key=lambda index: evaluations[index][0],
    )


def _select_survivors(
    candidates: list[list[int]],
    prices: list[int],
    target_sum: int,
) -> list[list[int]]:
    ranked_candidates = sorted(
        enumerate(candidates),
        key=lambda item: (evaluate_vector(prices, target_sum, item[1])[0], item[0]),
    )
    return [candidates[index][:] for index, _ in ranked_candidates[:2]]


def _build_next_population(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
    crossover_fn: Callable[[list[int], list[int], Random], tuple[list[int], list[int]]],
    mutate_fn: Callable[[list[int], float, Random], list[int]],
) -> list[list[int]]:
    ranked_indices = _rank_population_indices(evaluations)
    next_generation = [population[ranked_indices[0]][:]]

    while len(next_generation) < len(population):
        parent_a = _pick_parent(
            population,
            evaluations,
            config.tournament_size,
            rng,
        )
        parent_b = _pick_parent(
            population,
            evaluations,
            config.tournament_size,
            rng,
        )

        if rng.random() < config.crossover_rate:
            child_a, child_b = crossover_fn(parent_a, parent_b, rng)
        else:
            child_a, child_b = parent_a[:], parent_b[:]

        candidate_pool = [
            parent_a[:],
            parent_b[:],
            mutate_fn(child_a, config.mutation_rate, rng),
            mutate_fn(child_b, config.mutation_rate, rng),
        ]
        survivors = _select_survivors(candidate_pool, prices, target_sum)
        for survivor in survivors:
            if len(next_generation) >= len(population):
                break
            next_generation.append(survivor)

    return next_generation


def _apply_nga_two_point(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> list[list[int]]:
    return _build_next_population(
        population,
        evaluations,
        prices,
        target_sum,
        config,
        rng,
        crossover_two_points,
        mutate_two_points,
    )


def _apply_nga_elite_heavy_mutation(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> list[list[int]]:
    ranked_indices = _rank_population_indices(evaluations)
    best_index = ranked_indices[0]
    next_generation = [population[best_index][:]]

    for index, vector in enumerate(population):
        if index == best_index:
            continue
        next_generation.append(mutate_many_bits(vector, config.nga_mutation_fraction, rng))

    return next_generation


def _apply_nga_intervention(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> list[list[int]]:
    if config.nga_mode == "two_point":
        return _apply_nga_two_point(population, evaluations, prices, target_sum, config, rng)
    if config.nga_mode == "elite_heavy_mutation":
        return _apply_nga_elite_heavy_mutation(population, evaluations, config, rng)
    return population


def _best_from_population(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
) -> tuple[list[int], int, int]:
    best_index = min(range(len(population)), key=lambda index: evaluations[index][0])
    best_difference, best_sum = evaluations[best_index]
    return population[best_index][:], best_sum, best_difference


def _run_search(
    population: list[list[int]],
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> _RunResult:
    evaluations = _evaluate_population(population, prices, target_sum)
    best_vector, best_sum, best_difference = _best_from_population(population, evaluations)

    if best_difference == 0:
        return _RunResult(
            best_vector=best_vector,
            best_sum=best_sum,
            best_difference=best_difference,
            generations_used=0,
            stop_reason="exact_match",
            exact_match=True,
            nga_used=False,
            nga_trigger_generation=None,
        )

    no_improvement_streak = 0
    current_population = population
    nga_used = False
    nga_trigger_generation: int | None = None

    for generation in range(1, config.generations + 1):
        current_population = _build_next_population(
            current_population,
            evaluations,
            prices,
            target_sum,
            config,
            rng,
            crossover,
            mutate,
        )
        evaluations = _evaluate_population(current_population, prices, target_sum)
        current_best_vector, current_best_sum, current_best_difference = _best_from_population(
            current_population,
            evaluations,
        )

        if current_best_difference < best_difference:
            best_vector = current_best_vector
            best_sum = current_best_sum
            best_difference = current_best_difference
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        if best_difference == 0:
            return _RunResult(
                best_vector=best_vector,
                best_sum=best_sum,
                best_difference=best_difference,
                generations_used=generation,
                stop_reason="exact_match",
                exact_match=True,
                nga_used=nga_used,
                nga_trigger_generation=nga_trigger_generation,
            )

        if (
            config.nga_mode != "none"
            and not nga_used
            and config.repeat_limit is not None
            and no_improvement_streak >= config.repeat_limit
        ):
            current_population = _apply_nga_intervention(
                current_population,
                evaluations,
                prices,
                target_sum,
                config,
                rng,
            )
            evaluations = _evaluate_population(current_population, prices, target_sum)
            current_best_vector, current_best_sum, current_best_difference = _best_from_population(
                current_population,
                evaluations,
            )
            nga_used = True
            nga_trigger_generation = generation

            if current_best_difference < best_difference:
                best_vector = current_best_vector
                best_sum = current_best_sum
                best_difference = current_best_difference

            no_improvement_streak = 0

            if best_difference == 0:
                return _RunResult(
                    best_vector=best_vector,
                    best_sum=best_sum,
                    best_difference=best_difference,
                    generations_used=generation,
                    stop_reason="exact_match",
                    exact_match=True,
                    nga_used=nga_used,
                    nga_trigger_generation=nga_trigger_generation,
                )

            continue

        if no_improvement_streak >= config.stagnation:
            return _RunResult(
                best_vector=best_vector,
                best_sum=best_sum,
                best_difference=best_difference,
                generations_used=generation,
                stop_reason="stagnation",
                exact_match=False,
                nga_used=nga_used,
                nga_trigger_generation=nga_trigger_generation,
            )

    return _RunResult(
        best_vector=best_vector,
        best_sum=best_sum,
        best_difference=best_difference,
        generations_used=config.generations,
        stop_reason="generation_limit",
        exact_match=False,
        nga_used=nga_used,
        nga_trigger_generation=nga_trigger_generation,
    )


def solve_with_genetic_algorithm(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> GeneticAlgorithmResult:
    population = build_initial_population(
        gene_count=len(prices),
        population_size=config.population_size,
        rng=rng,
    )
    run = _run_search(
        population=population,
        prices=prices,
        target_sum=target_sum,
        config=config,
        rng=rng,
    )
    return GeneticAlgorithmResult(
        best_vector=run.best_vector,
        best_sum=run.best_sum,
        fitness=run.best_difference,
        difference=run.best_difference,
        generations_used=run.generations_used,
        exact_match=run.exact_match,
        stop_reason=run.stop_reason,
        nga_used=run.nga_used,
        nga_trigger_generation=run.nga_trigger_generation,
    )
