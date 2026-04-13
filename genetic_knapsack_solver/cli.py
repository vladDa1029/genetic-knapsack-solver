from __future__ import annotations

import argparse
from random import Random
from typing import Sequence

from genetic_knapsack_solver.ga import solve_with_genetic_algorithm
from genetic_knapsack_solver.generator import (
    GENERATION_MODE_LABELS,
    GENERATION_MODE_RANDOM,
    generate_problem,
)
from genetic_knapsack_solver.models import GeneticAlgorithmConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Решение subset-sum задачи генетическим алгоритмом.",
    )
    parser.add_argument("--items", type=int, default=12, help="Количество предметов.")
    parser.add_argument(
        "--generation-mode",
        choices=tuple(GENERATION_MODE_LABELS),
        default=GENERATION_MODE_RANDOM,
        help="Режим генерации задачи.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Размер популяции.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Максимум поколений.",
    )
    parser.add_argument(
        "--stagnation",
        type=int,
        default=30,
        help="Порог поколений без улучшения.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Вероятность одноточечного кроссовера.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.05,
        help="Вероятность мутации.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Размер турнира при селекции.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Опциональный seed для воспроизводимости.",
    )
    return parser


def _format_vector(vector: list[int]) -> str:
    return "[" + ", ".join(str(bit) for bit in vector) + "]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = Random(args.seed)
    problem = generate_problem(
        item_count=args.items,
        rng=rng,
        generation_mode=args.generation_mode,
    )
    config = GeneticAlgorithmConfig(
        population_size=args.population_size,
        generations=args.generations,
        stagnation=args.stagnation,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
    )
    result = solve_with_genetic_algorithm(
        prices=problem.prices,
        target_sum=problem.target_sum,
        config=config,
        rng=rng,
    )

    print(f"Режим генерации: {GENERATION_MODE_LABELS[problem.generation_mode]}")
    print(f"Предметы: {', '.join(problem.items)}")
    print(f"Цены: {problem.prices}")
    print(f"Целевая сумма: {problem.target_sum}")
    print(f"Скрытый вектор: {_format_vector(problem.hidden_vector)}")
    print(f"Лучшее решение: {_format_vector(result.best_vector)}")
    print(f"Найденная сумма: {result.best_sum}")
    print(f"Fitness: {result.fitness}")
    print(f"Абсолютная разница: {result.difference}")
    print(f"Поколений: {result.generations_used}")
    print(f"Точное совпадение: {'Да' if result.exact_match else 'Нет'}")
    print(f"Причина остановки: {result.stop_reason}")
    return 0
