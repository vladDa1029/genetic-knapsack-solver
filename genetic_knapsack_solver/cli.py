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

NGA_MODE_LABELS = {
    "none": "Без NGA",
    "two_point": "NGA: двухточечный кроссовер и двухточечная мутация",
    "elite_heavy_mutation": "NGA: сохранить лучшую особь и сильно мутировать остальные",
}


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
        "--repeat-limit",
        type=int,
        default=None,
        help="Лимит поколений без улучшения для одноразового вызова NGA.",
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
        "--nga-mode",
        choices=tuple(NGA_MODE_LABELS),
        default="none",
        help="Режим одноразового вмешательства NGA.",
    )
    parser.add_argument(
        "--nga-mutation-fraction",
        type=float,
        default=0.4,
        help="Доля битов для сильной мутации в режиме elite_heavy_mutation.",
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
        repeat_limit=args.repeat_limit,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        nga_mode=args.nga_mode,
        nga_mutation_fraction=args.nga_mutation_fraction,
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
    print(f"NGA режим: {NGA_MODE_LABELS[config.nga_mode]}")
    print(f"Лимит NGA без улучшения: {config.repeat_limit if config.repeat_limit is not None else 'Не используется'}")
    print(f"Лучшее решение: {_format_vector(result.best_vector)}")
    print(f"Найденная сумма: {result.best_sum}")
    print(f"Fitness: {result.fitness}")
    print(f"Абсолютная разница: {result.difference}")
    print(f"Поколений: {result.generations_used}")
    print(f"Точное совпадение: {'Да' if result.exact_match else 'Нет'}")
    print(f"NGA использован: {'Да' if result.nga_used else 'Нет'}")
    print(f"Поколение NGA: {result.nga_trigger_generation if result.nga_trigger_generation is not None else 'Не применялся'}")
    print(f"Причина остановки: {result.stop_reason}")
    return 0
