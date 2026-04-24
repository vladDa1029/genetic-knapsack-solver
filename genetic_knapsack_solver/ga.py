from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import Random
from typing import Callable

from genetic_knapsack_solver.generator import dot_product
from genetic_knapsack_solver.models import GeneticAlgorithmConfig, GeneticAlgorithmResult, OperatorType


CrossoverFunction = Callable[[list[int], list[int], Random], tuple[list[int], list[int]]]
MutationFunction = Callable[[list[int], float, Random], list[int]]


@dataclass(slots=True)
class _SingleRunResult:
    population: list[list[int]]
    best_vector: list[int]
    best_sum: int
    difference: int
    best_fitness: int
    generations_used: int
    exact_match: bool
    stop_reason: str
    nga_used: bool
    nga_trigger_generation: int | None
    nga_trigger_generations: list[int]


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


def rescue_heavy_mutation(
    vector: list[int],
    mutation_ratio: float,
    rng: Random,
) -> list[int]:
    mutated = vector[:]
    if not mutated or mutation_ratio <= 0.0:
        return mutated

    flip_count = min(len(mutated), max(1, ceil(mutation_ratio * len(mutated))))
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


def _build_next_population(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    config: GeneticAlgorithmConfig,
    rng: Random,
    crossover_fn: CrossoverFunction,
    mutate_fn: MutationFunction,
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

        next_generation.append(mutate_fn(child_a, config.mutation_rate, rng))
        if len(next_generation) < len(population):
            next_generation.append(mutate_fn(child_b, config.mutation_rate, rng))

    return next_generation


def _apply_nga_two_point(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> list[list[int]]:
    return _build_next_population(
        population,
        evaluations,
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
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> list[list[int]]:
    if config.nga_mode == "two_point":
        return _apply_nga_two_point(population, evaluations, config, rng)
    if config.nga_mode == "elite_heavy_mutation":
        return _apply_nga_elite_heavy_mutation(population, evaluations, config, rng)
    return population


def _resolve_staged_mutation_fraction(
    config: GeneticAlgorithmConfig,
    stagnation_counter: int,
) -> float:
    trigger_index = config.nga_trigger_points.index(stagnation_counter)
    raw_points = config.nga_mutate_points
    mutate_percent = raw_points[0] if len(raw_points) == 1 else raw_points[trigger_index]
    return mutate_percent / 100.0


def _apply_staged_hypermutation(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
    mutation_fraction: float,
    rng: Random,
) -> list[list[int]]:
    ranked_indices = _rank_population_indices(evaluations)
    best_index = ranked_indices[0]
    next_generation = [population[best_index][:]]

    for index, vector in enumerate(population):
        if index == best_index:
            continue
        next_generation.append(mutate_many_bits(vector, mutation_fraction, rng))

    return next_generation


def _best_from_population(
    population: list[list[int]],
    evaluations: list[tuple[int, int]],
) -> tuple[list[int], int, int]:
    best_index = min(range(len(population)), key=lambda index: evaluations[index][0])
    best_difference, best_sum = evaluations[best_index]
    return population[best_index][:], best_sum, best_difference


def _copy_population(population: list[list[int]]) -> list[list[int]]:
    return [vector[:] for vector in population]


def _resolve_crossover_fn(operator_type: OperatorType) -> CrossoverFunction:
    if operator_type == "two_point":
        return crossover_two_points
    return crossover


def _resolve_mutation_fn(operator_type: OperatorType) -> MutationFunction:
    if operator_type == "two_point":
        return mutate_two_points
    return mutate


def _single_run(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
    initial_population: list[list[int]] | None = None,
    stop_on_stagnation: bool = True,
    allow_nga: bool = False,
    crossover_fn: CrossoverFunction = crossover,
    mutate_fn: MutationFunction = mutate,
) -> _SingleRunResult:
    current_population = (
        build_initial_population(len(prices), config.population_size, rng)
        if initial_population is None
        else _copy_population(initial_population)
    )
    evaluations = _evaluate_population(current_population, prices, target_sum)
    best_vector, best_sum, best_difference = _best_from_population(current_population, evaluations)
    best_fitness = best_difference
    nga_used = False
    nga_trigger_generation: int | None = None
    nga_trigger_generations: list[int] = []

    if best_difference == 0:
        return _SingleRunResult(
            population=current_population,
            best_vector=best_vector,
            best_sum=best_sum,
            difference=best_difference,
            best_fitness=best_fitness,
            generations_used=0,
            exact_match=True,
            stop_reason="exact_match",
            nga_used=False,
            nga_trigger_generation=None,
            nga_trigger_generations=[],
        )

    no_improvement_streak = 0

    for generation in range(1, config.generations + 1):
        current_population = _build_next_population(
            current_population,
            evaluations,
            config,
            rng,
            crossover_fn,
            mutate_fn,
        )
        evaluations = _evaluate_population(current_population, prices, target_sum)
        current_best_vector, current_best_sum, current_best_difference = _best_from_population(
            current_population,
            evaluations,
        )
        current_best_fitness = current_best_difference

        if current_best_difference < best_difference:
            best_vector = current_best_vector
            best_sum = current_best_sum
            best_difference = current_best_difference
            best_fitness = current_best_fitness
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        if best_difference == 0:
            return _SingleRunResult(
                population=current_population,
                best_vector=best_vector,
                best_sum=best_sum,
                difference=best_difference,
                best_fitness=best_fitness,
                generations_used=generation,
                exact_match=True,
                stop_reason="exact_match",
                nga_used=nga_used,
                nga_trigger_generation=nga_trigger_generation,
                nga_trigger_generations=nga_trigger_generations[:],
            )

        if allow_nga and config.nga_mode != "none":
            nga_population: list[list[int]] | None = None

            if (
                config.nga_mode == "staged_hypermutation"
                and no_improvement_streak in config.nga_trigger_points
            ):
                mutation_fraction = _resolve_staged_mutation_fraction(config, no_improvement_streak)
                nga_population = _apply_staged_hypermutation(
                    current_population,
                    evaluations,
                    mutation_fraction,
                    rng,
                )
            elif (
                config.nga_mode in ("two_point", "elite_heavy_mutation")
                and not nga_used
                and config.repeat_limit is not None
                and no_improvement_streak >= config.repeat_limit
            ):
                nga_population = _apply_nga_intervention(
                    current_population,
                    evaluations,
                    config,
                    rng,
                )

            if nga_population is not None:
                current_population = nga_population
                evaluations = _evaluate_population(current_population, prices, target_sum)
                current_best_vector, current_best_sum, current_best_difference = _best_from_population(
                    current_population,
                    evaluations,
                )
                current_best_fitness = current_best_difference
                nga_used = True
                if nga_trigger_generation is None:
                    nga_trigger_generation = generation
                nga_trigger_generations.append(generation)

                if current_best_difference < best_difference:
                    best_vector = current_best_vector
                    best_sum = current_best_sum
                    best_difference = current_best_difference
                    best_fitness = current_best_fitness
                    no_improvement_streak = 0

                if best_difference == 0:
                    return _SingleRunResult(
                        population=current_population,
                        best_vector=best_vector,
                        best_sum=best_sum,
                        difference=best_difference,
                        best_fitness=best_fitness,
                        generations_used=generation,
                        exact_match=True,
                        stop_reason="exact_match",
                        nga_used=nga_used,
                        nga_trigger_generation=nga_trigger_generation,
                        nga_trigger_generations=nga_trigger_generations[:],
                    )

        if stop_on_stagnation and no_improvement_streak >= config.stagnation:
            return _SingleRunResult(
                population=current_population,
                best_vector=best_vector,
                best_sum=best_sum,
                difference=best_difference,
                best_fitness=best_fitness,
                generations_used=generation,
                exact_match=False,
                stop_reason="stagnation_limit",
                nga_used=nga_used,
                nga_trigger_generation=nga_trigger_generation,
                nga_trigger_generations=nga_trigger_generations[:],
            )

    return _SingleRunResult(
        population=current_population,
        best_vector=best_vector,
        best_sum=best_sum,
        difference=best_difference,
        best_fitness=best_fitness,
        generations_used=config.generations,
        exact_match=best_difference == 0,
        stop_reason="exact_match" if best_difference == 0 else "generation_limit",
        nga_used=nga_used,
        nga_trigger_generation=nga_trigger_generation,
        nga_trigger_generations=nga_trigger_generations[:],
    )


def _build_rescue_population(
    base_population: list[list[int]],
    global_best_vector: list[int],
    mutation_ratio: float,
    rng: Random,
) -> list[list[int]]:
    rescue_population = [global_best_vector[:]]
    for vector in base_population[1:]:
        rescue_population.append(rescue_heavy_mutation(vector, mutation_ratio, rng))
    return rescue_population


def _build_restart_population(
    base_population: list[list[int]],
    elite_vector: list[int],
    population_mode: str,
    mutation_fraction: float,
    rng: Random,
) -> list[list[int]]:
    if population_mode != "elite_from_last_population":
        raise ValueError(f"unsupported restart_population_mode: {population_mode}")
    if not base_population:
        return [elite_vector[:]]

    elite_index = next((index for index, vector in enumerate(base_population) if vector == elite_vector), 0)
    restart_population = [elite_vector[:]]

    for index, vector in enumerate(base_population):
        if index == elite_index:
            continue
        restart_population.append(mutate_many_bits(vector, mutation_fraction, rng))

    return restart_population


def _choose_better_run(
    first: _SingleRunResult,
    second: _SingleRunResult,
) -> _SingleRunResult:
    if second.difference < first.difference:
        return second
    if second.difference > first.difference:
        return first
    if second.best_fitness < first.best_fitness:
        return second
    return first


def _build_result(
    run: _SingleRunResult,
    generations_used: int,
    stop_reason: str,
    restart_count: int,
    rescue_used: bool,
    stage1_run_count: int = 0,
    stage2_run_count: int = 0,
    stage2_used: bool = False,
    stage1_best_differences: list[int] | None = None,
    stage2_best_differences: list[int] | None = None,
) -> GeneticAlgorithmResult:
    return GeneticAlgorithmResult(
        best_vector=run.best_vector,
        best_sum=run.best_sum,
        fitness=run.best_fitness,
        difference=run.difference,
        generations_used=generations_used,
        exact_match=run.exact_match,
        stop_reason=stop_reason,
        nga_used=run.nga_used,
        nga_trigger_generation=run.nga_trigger_generation,
        nga_trigger_generations=run.nga_trigger_generations[:],
        restart_count=restart_count,
        rescue_used=rescue_used,
        stage1_run_count=stage1_run_count,
        stage2_run_count=stage2_run_count,
        stage2_used=stage2_used,
        stage1_best_differences=[] if stage1_best_differences is None else stage1_best_differences[:],
        stage2_best_differences=[] if stage2_best_differences is None else stage2_best_differences[:],
    )


def _solve_classic(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> GeneticAlgorithmResult:
    run = _single_run(
        prices=prices,
        target_sum=target_sum,
        config=config,
        rng=rng,
        initial_population=None,
        stop_on_stagnation=True,
        allow_nga=True,
    )
    stop_reason = "stagnation" if run.stop_reason == "stagnation_limit" else run.stop_reason
    return _build_result(
        run=run,
        generations_used=run.generations_used,
        stop_reason=stop_reason,
        restart_count=0,
        rescue_used=False,
    )


def _solve_restart_rescue(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> GeneticAlgorithmResult:
    run_0 = _single_run(
        prices=prices,
        target_sum=target_sum,
        config=config,
        rng=rng,
        initial_population=None,
        stop_on_stagnation=True,
        allow_nga=False,
    )

    if run_0.stop_reason in {"exact_match", "generation_limit"}:
        return _build_result(
            run=run_0,
            generations_used=run_0.generations_used,
            stop_reason=run_0.stop_reason,
            restart_count=0,
            rescue_used=False,
        )

    global_best = run_0
    total_generations = run_0.generations_used
    restart_count = 0

    while True:
        if config.restart_max_count is not None and restart_count >= config.restart_max_count:
            return _build_result(
                run=global_best,
                generations_used=total_generations,
                stop_reason="restart_limit",
                restart_count=restart_count,
                rescue_used=False,
            )

        restart_count += 1
        run_i = _single_run(
            prices=prices,
            target_sum=target_sum,
            config=config,
            rng=rng,
            initial_population=None,
            stop_on_stagnation=True,
            allow_nga=False,
        )
        total_generations += run_i.generations_used

        if run_i.stop_reason in {"exact_match", "generation_limit"}:
            return _build_result(
                run=run_i,
                generations_used=total_generations,
                stop_reason=run_i.stop_reason,
                restart_count=restart_count,
                rescue_used=False,
            )

        if run_i.difference < global_best.difference:
            global_best = run_i
            continue

        rescue_population = _build_rescue_population(
            base_population=run_i.population,
            global_best_vector=global_best.best_vector,
            mutation_ratio=config.rescue_min_mutated_bits_ratio,
            rng=rng,
        )
        rescue_run = _single_run(
            prices=prices,
            target_sum=target_sum,
            config=config,
            rng=rng,
            initial_population=rescue_population,
            stop_on_stagnation=True,
            allow_nga=False,
        )
        total_generations += rescue_run.generations_used

        if rescue_run.stop_reason == "exact_match":
            return _build_result(
                run=rescue_run,
                generations_used=total_generations,
                stop_reason="exact_match",
                restart_count=restart_count,
                rescue_used=True,
            )

        if rescue_run.stop_reason == "stagnation_limit":
            return _build_result(
                run=global_best,
                generations_used=total_generations,
                stop_reason="rescue_stagnation",
                restart_count=restart_count,
                rescue_used=True,
            )

        if rescue_run.stop_reason == "generation_limit":
            selected_run = _choose_better_run(global_best, rescue_run)
            return _build_result(
                run=selected_run,
                generations_used=total_generations,
                stop_reason="rescue_generation_limit",
                restart_count=restart_count,
                rescue_used=True,
            )

        return _build_result(
            run=rescue_run,
            generations_used=total_generations,
            stop_reason=rescue_run.stop_reason,
            restart_count=restart_count,
            rescue_used=True,
        )


def _solve_two_stage_restart(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> GeneticAlgorithmResult:
    total_generations = 0
    stage = 1
    current_population: list[list[int]] | None = None
    stage1_best_differences: list[int] = []
    stage2_best_differences: list[int] = []
    stage1_run_count = 0
    stage2_run_count = 0
    stage2_used = False
    current_crossover_fn = crossover
    current_mutate_fn = mutate

    while True:
        run = _single_run(
            prices=prices,
            target_sum=target_sum,
            config=config,
            rng=rng,
            initial_population=current_population,
            stop_on_stagnation=True,
            allow_nga=False,
            crossover_fn=current_crossover_fn,
            mutate_fn=current_mutate_fn,
        )
        total_generations += run.generations_used

        if stage == 1:
            stage1_run_count += 1
            stage1_best_differences.append(run.difference)
        else:
            stage2_run_count += 1
            stage2_best_differences.append(run.difference)
            stage2_used = True

        if run.stop_reason == "exact_match":
            return _build_result(
                run=run,
                generations_used=total_generations,
                stop_reason="exact_match",
                restart_count=0,
                rescue_used=False,
                stage1_run_count=stage1_run_count,
                stage2_run_count=stage2_run_count,
                stage2_used=stage2_used,
                stage1_best_differences=stage1_best_differences,
                stage2_best_differences=stage2_best_differences,
            )

        if run.stop_reason == "generation_limit":
            return _build_result(
                run=run,
                generations_used=total_generations,
                stop_reason="generation_limit",
                restart_count=0,
                rescue_used=False,
                stage1_run_count=stage1_run_count,
                stage2_run_count=stage2_run_count,
                stage2_used=stage2_used,
                stage1_best_differences=stage1_best_differences,
                stage2_best_differences=stage2_best_differences,
            )

        current_population = _build_restart_population(
            base_population=run.population,
            elite_vector=run.best_vector,
            population_mode=config.restart_population_mode,
            mutation_fraction=config.restart_mutation_fraction,
            rng=rng,
        )

        if stage == 1:
            if (
                len(stage1_best_differences) >= 2
                and stage1_best_differences[-1] == stage1_best_differences[-2]
            ):
                stage = 2
                current_crossover_fn = _resolve_crossover_fn(config.stage2_crossover_type)
                current_mutate_fn = _resolve_mutation_fn(config.stage2_mutation_type)
                continue

            current_crossover_fn = crossover
            current_mutate_fn = mutate
            continue

        if (
            len(stage2_best_differences) >= 2
            and stage2_best_differences[-1] == stage2_best_differences[-2]
        ):
            return _build_result(
                run=run,
                generations_used=total_generations,
                stop_reason="stagnation",
                restart_count=0,
                rescue_used=False,
                stage1_run_count=stage1_run_count,
                stage2_run_count=stage2_run_count,
                stage2_used=stage2_used,
                stage1_best_differences=stage1_best_differences,
                stage2_best_differences=stage2_best_differences,
            )


def solve_with_genetic_algorithm(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig,
    rng: Random,
) -> GeneticAlgorithmResult:
    if config.solver_mode == "restart_rescue":
        return _solve_restart_rescue(
            prices=prices,
            target_sum=target_sum,
            config=config,
            rng=rng,
        )
    if config.solver_mode == "two_stage_restart":
        return _solve_two_stage_restart(
            prices=prices,
            target_sum=target_sum,
            config=config,
            rng=rng,
        )
    return _solve_classic(
        prices=prices,
        target_sum=target_sum,
        config=config,
        rng=rng,
    )
