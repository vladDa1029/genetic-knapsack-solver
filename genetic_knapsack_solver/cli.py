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
from genetic_knapsack_solver.rust_core import solve_with_rust_core

SOLVER_MODE_LABELS = {
    "classic": "Классический режим",
    "restart_rescue": "Режим рестартов и rescue",
    "two_stage_restart": "Двухэтапный каскад запусков с переносом элиты",
    "five_stage_restart": "Five-stage restart (Rust core)",
}

NGA_MODE_LABELS = {
    "none": "Без NGA",
    "two_point": "NGA: двухточечный кроссовер и двухточечная мутация",
    "elite_heavy_mutation": "NGA: сохранить лучшую особь и сильно мутировать остальные",
    "staged_hypermutation": "NGA: многошаговая hypermutation в точках стагнации",
}

RESTART_POPULATION_MODE_LABELS = {
    "elite_from_last_population": "Элита прошлого запуска + сильная мутация остальных",
}

RESTART_MUTATION_TYPE_LABELS = {
    "many_bits": "Сильная bit-flip мутация",
    "reverse": "Разворот вектора",
}

CROSSOVER_TYPE_LABELS = {
    "one_point": "Одноточечный",
    "two_point": "Двухточечный",
}

MUTATION_TYPE_LABELS = {
    "one_point": "Одноточечная",
    "two_point": "Двухточечная",
    "reverse": "Разворот вектора",
}

MULTISTAGE_MUTATION_TYPE_LABELS = {
    "one_point": "Одноточечная",
    "two_point": "Двухточечная",
}


STAGE2_OFFSPRING_MODE_LABELS = {
    "two_children": "2 родителя -> 2 ребёнка",
    "four_children_select_two": "2 родителя -> 4 ребёнка -> выбрать 2 лучших",
}


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


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
        "--solver-mode",
        choices=tuple(SOLVER_MODE_LABELS),
        default="classic",
        help="Режим работы solver.",
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
        help="Режим вмешательства NGA.",
    )
    parser.add_argument(
        "--nga-mutation-fraction",
        type=float,
        default=0.4,
        help="Доля битов для сильной мутации в режиме elite_heavy_mutation.",
    )
    parser.add_argument(
        "--nga-trigger-points",
        type=_parse_int_list,
        default=(),
        help="Точные точки стагнации для staged_hypermutation, например 100,160,215.",
    )
    parser.add_argument(
        "--nga-mutate-points",
        type=_parse_int_list,
        default=(40,),
        help="Проценты сильной мутации для staged_hypermutation: одно значение или список по trigger-point.",
    )
    parser.add_argument(
        "--restart-max-count",
        type=int,
        default=None,
        help="Максимум полных рестартов в режиме restart_rescue.",
    )
    parser.add_argument(
        "--rescue-min-mutated-bits-ratio",
        type=float,
        default=0.4,
        help="Минимальная доля инвертируемых битов в rescue-этапе.",
    )
    parser.add_argument(
        "--restart-population-mode",
        choices=tuple(RESTART_POPULATION_MODE_LABELS),
        default="elite_from_last_population",
        help="Способ построения новой популяции между запусками two_stage_restart.",
    )
    parser.add_argument(
        "--restart-mutation-fraction",
        type=float,
        default=0.4,
        help="Доля битов для сильной мутации между запусками two_stage_restart.",
    )
    parser.add_argument(
        "--restart-mutation-type",
        choices=tuple(RESTART_MUTATION_TYPE_LABELS),
        default="many_bits",
        help="Оператор мутации для restart-популяции two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-crossover-type",
        choices=tuple(CROSSOVER_TYPE_LABELS),
        default="two_point",
        help="Тип кроссовера для 2 этапа two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-mutation-type",
        choices=tuple(MUTATION_TYPE_LABELS),
        default="two_point",
        help="Тип мутации для 2 этапа two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-offspring-mode",
        choices=tuple(STAGE2_OFFSPRING_MODE_LABELS),
        default="two_children",
        help="РЎС…РµРјР° РїРѕСЃС‚СЂРѕРµРЅРёСЏ РїРѕС‚РѕРјРєРѕРІ РЅР° 2 СЌС‚Р°РїРµ two_stage_restart.",
    )
    parser.add_argument(
        "--multistage-crossover-type",
        choices=tuple(CROSSOVER_TYPE_LABELS),
        default="one_point",
        help="Crossover type for five_stage_restart.",
    )
    parser.add_argument(
        "--multistage-mutation-type",
        choices=tuple(MULTISTAGE_MUTATION_TYPE_LABELS),
        default="one_point",
        help="Mutation type for five_stage_restart.",
    )
    parser.add_argument(
        "--multistage-elite-count",
        type=int,
        default=1,
        help="Elite count preserved between five_stage_restart stages.",
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
        solver_mode=args.solver_mode,
        population_size=args.population_size,
        generations=args.generations,
        stagnation=args.stagnation,
        repeat_limit=args.repeat_limit,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        nga_mode=args.nga_mode,
        nga_mutation_fraction=args.nga_mutation_fraction,
        nga_trigger_points=args.nga_trigger_points,
        nga_mutate_points=args.nga_mutate_points,
        restart_max_count=args.restart_max_count,
        rescue_min_mutated_bits_ratio=args.rescue_min_mutated_bits_ratio,
        restart_population_mode=args.restart_population_mode,
        restart_mutation_type=args.restart_mutation_type,
        restart_mutation_fraction=args.restart_mutation_fraction,
        stage2_crossover_type=args.stage2_crossover_type,
        stage2_mutation_type=args.stage2_mutation_type,
        stage2_offspring_mode=args.stage2_offspring_mode,
        multistage_crossover_type=args.multistage_crossover_type,
        multistage_mutation_type=args.multistage_mutation_type,
        multistage_elite_count=args.multistage_elite_count,
    )
    if config.solver_mode == "five_stage_restart":
        result = solve_with_rust_core(
            prices=problem.prices,
            target_sum=problem.target_sum,
            config=config,
            rng=rng,
        )
    else:
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
    print(f"Режим solver: {SOLVER_MODE_LABELS[config.solver_mode]}")
    print(f"NGA режим: {NGA_MODE_LABELS[config.nga_mode]}")
    print(
        "Лимит NGA без улучшения: "
        f"{config.repeat_limit if config.repeat_limit is not None else 'Не используется'}"
    )
    print(
        "Точки staged NGA: "
        f"{list(config.nga_trigger_points) if config.nga_trigger_points else 'Не используются'}"
    )
    print(f"Сила staged NGA, %: {list(config.nga_mutate_points)}")
    print(
        "Максимум рестартов: "
        f"{config.restart_max_count if config.restart_max_count is not None else 'Без ограничения'}"
    )
    print(f"Минимальная доля битов rescue: {config.rescue_min_mutated_bits_ratio}")
    print(f"Режим restart-популяции: {RESTART_POPULATION_MODE_LABELS[config.restart_population_mode]}")
    print(f"Оператор restart-мутации: {RESTART_MUTATION_TYPE_LABELS[config.restart_mutation_type]}")
    print(f"Сила restart-мутации: {config.restart_mutation_fraction}")
    print(f"Кроссовер 2 этапа: {CROSSOVER_TYPE_LABELS[config.stage2_crossover_type]}")
    print(f"Мутация 2 этапа: {MUTATION_TYPE_LABELS[config.stage2_mutation_type]}")
    print(f"Схема потомков 2 этапа: {STAGE2_OFFSPRING_MODE_LABELS[config.stage2_offspring_mode]}")
    print(f"Кроссовер five-stage: {CROSSOVER_TYPE_LABELS[config.multistage_crossover_type]}")
    print(f"Мутация five-stage: {MULTISTAGE_MUTATION_TYPE_LABELS[config.multistage_mutation_type]}")
    print(f"Элит five-stage: {config.multistage_elite_count}")
    print(f"Лучшее решение: {_format_vector(result.best_vector)}")
    print(f"Найденная сумма: {result.best_sum}")
    print(f"Fitness: {result.fitness}")
    print(f"Абсолютная разница: {result.difference}")
    print(f"Поколений: {result.generations_used}")
    print(f"Точное совпадение: {'Да' if result.exact_match else 'Нет'}")
    print(f"NGA использован: {'Да' if result.nga_used else 'Нет'}")
    print(
        "Поколения NGA: "
        f"{result.nga_trigger_generations if result.nga_trigger_generations else 'Не применялся'}"
    )
    print(f"Количество рестартов: {result.restart_count}")
    print(f"Rescue использован: {'Да' if result.rescue_used else 'Нет'}")
    print(f"Запусков 1 этапа: {result.stage1_run_count}")
    print(f"Запусков 2 этапа: {result.stage2_run_count}")
    print(f"2 этап использован: {'Да' if result.stage2_used else 'Нет'}")
    print(f"История differance 1 этапа: {result.stage1_best_differences}")
    print(f"История differance 2 этапа: {result.stage2_best_differences}")
    print(f"Запуски five-stage по этапам: {result.stage_run_counts}")
    print(f"История difference five-stage: {result.stage_best_differences}")
    print(f"Финальный этап five-stage: {result.final_stage}")
    print(f"Причина остановки: {result.stop_reason}")
    return 0
