from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter
from typing import Sequence

from genetic_knapsack_solver.ga import solve_with_genetic_algorithm
from genetic_knapsack_solver.generator import (
    GENERATION_MODE_LABELS,
    GENERATION_MODE_SUPERINCREASING_DISGUISED,
    generate_problem,
)
from genetic_knapsack_solver.models import GeneticAlgorithmConfig, ProblemInstance

STOP_REASON_LABELS = {
    "exact_match": "точное совпадение",
    "stagnation": "стагнация",
    "generation_limit": "лимит поколений",
}

CSV_FIELD_LABELS = {
    "task_index": "Индекс задачи",
    "problem_seed": "Seed задачи",
    "solver_seed": "Seed решателя",
    "generation_mode": "Режим генерации",
    "items": "Количество предметов",
    "population_size": "Размер популяции",
    "stagnation": "Порог остановки без улучшения",
    "generations": "Максимум поколений",
    "mutation_rate": "Вероятность мутации",
    "crossover_rate": "Вероятность кроссовера",
    "tournament_size": "Размер турнира",
    "target_sum": "Целевая сумма",
    "hidden_vector": "Скрытый вектор",
    "best_vector": "Лучший вектор",
    "best_sum": "Найденная сумма",
    "fitness": "Fitness",
    "difference": "Абсолютная разница",
    "generations_used": "Использовано поколений",
    "exact_match": "Точное совпадение",
    "stop_reason": "Причина остановки",
    "elapsed_seconds": "Время выполнения (сек)",
}


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    item_counts: tuple[int, ...] = (25,)
    task_count: int = 25
    generations: int = 500000
    mutation_rate: float = 0.9
    crossover_rate: float = 0.95
    tournament_size: int = 3
    population_stop_pairs: tuple[tuple[int, int], ...] = (
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (3000, 3000),
    )
    generation_mode: str = GENERATION_MODE_SUPERINCREASING_DISGUISED
    seed: int = 42
    output_root: Path = Path("benchmark_results")


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_population_stop_pairs(raw: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        population_size_raw, separator, stop_limit_raw = candidate.partition(":")
        if separator != ":":
            raise argparse.ArgumentTypeError(
                "expected pairs in format population:stop_limit"
            )
        population_size = int(population_size_raw.strip())
        stop_limit = int(stop_limit_raw.strip())
        pairs.append((population_size, stop_limit))

    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one pair")
    return tuple(pairs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Бенчмарк базового генетического алгоритма без NGA.",
    )
    parser.add_argument(
        "--items",
        type=_parse_int_list,
        default=(25,),
        help="Список размеров задачи n через запятую.",
    )
    parser.add_argument("--tasks", type=int, default=25, help="Количество генерируемых задач.")
    parser.add_argument(
        "--generation-mode",
        choices=tuple(GENERATION_MODE_LABELS),
        default=GENERATION_MODE_SUPERINCREASING_DISGUISED,
        help="Режим генерации задач для бенчмарка.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500000,
        help="Максимум поколений для одного прогона.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.9,
        help="Вероятность мутации.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.95,
        help="Вероятность кроссовера.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Размер турнира при селекции.",
    )
    parser.add_argument(
        "--population-stop-pairs",
        type=_parse_population_stop_pairs,
        default=((500, 500), (1000, 1000), (2000, 2000), (3000, 3000)),
        help="Список пар population:stop_limit через запятую.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Базовый seed для генерации задач и решателя.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_results"),
        help="Корневая папка для результатов.",
    )
    return parser


def _format_vector(vector: list[int]) -> str:
    return json.dumps(vector, ensure_ascii=False)


def _format_stop_reason(stop_reason: str) -> str:
    return STOP_REASON_LABELS.get(stop_reason, stop_reason)


def _format_exact_match(exact_match: bool) -> str:
    return "Да" if exact_match else "Нет"


def _build_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _build_problems(
    settings: BenchmarkSettings,
) -> list[tuple[int, int, int, ProblemInstance]]:
    problems: list[tuple[int, int, int, ProblemInstance]] = []
    for item_count in settings.item_counts:
        for task_index in range(settings.task_count):
            task_seed = settings.seed + item_count * 1_000 + task_index
            problem = generate_problem(
                item_count=item_count,
                rng=Random(task_seed),
                generation_mode=settings.generation_mode,
            )
            problems.append((item_count, task_index, task_seed, problem))
    return problems


def _summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["items"]),
            int(row["population_size"]),
            int(row["stagnation"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (item_count, population_size, stagnation), group_rows in sorted(grouped.items()):
        exact_matches = sum(1 for row in group_rows if row["exact_match"] == "Да")
        avg_elapsed = mean(float(row["elapsed_seconds"]) for row in group_rows)
        avg_generations = mean(int(row["generations_used"]) for row in group_rows)
        avg_fitness = mean(int(row["fitness"]) for row in group_rows)
        summary_rows.append(
            {
                "items": item_count,
                "population_size": population_size,
                "stagnation": stagnation,
                "runs": len(group_rows),
                "exact_matches": exact_matches,
                "exact_match_rate": 100.0 * exact_matches / len(group_rows),
                "avg_fitness": avg_fitness,
                "avg_generations_used": avg_generations,
                "avg_elapsed_seconds": avg_elapsed,
            }
        )
    return summary_rows


def _write_csv(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    csv_path = output_dir / "results.csv"
    field_order = [
        "task_index",
        "problem_seed",
        "solver_seed",
        "generation_mode",
        "items",
        "population_size",
        "stagnation",
        "generations",
        "mutation_rate",
        "crossover_rate",
        "tournament_size",
        "target_sum",
        "hidden_vector",
        "best_vector",
        "best_sum",
        "fitness",
        "difference",
        "generations_used",
        "exact_match",
        "stop_reason",
        "elapsed_seconds",
    ]
    fieldnames = [CSV_FIELD_LABELS[field] for field in field_order]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({CSV_FIELD_LABELS[field]: row[field] for field in field_order})
    return csv_path


def _write_summary(
    output_dir: Path,
    settings: BenchmarkSettings,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
) -> Path:
    summary_path = output_dir / "summary.md"
    best_config = min(
        summary_rows,
        key=lambda row: (
            -float(row["exact_match_rate"]),
            float(row["avg_fitness"]),
            float(row["avg_elapsed_seconds"]),
        ),
    )

    lines = [
        "# Сводка бенчмарка",
        "",
        "## Параметры запуска",
        "",
        f"- Время начала: `{started_at.isoformat(timespec='microseconds')}`",
        f"- Время окончания: `{finished_at.isoformat(timespec='microseconds')}`",
        f"- Общее время, сек: `{elapsed_seconds:.6f}`",
        f"- Режим генерации: `{GENERATION_MODE_LABELS[settings.generation_mode]}`",
        f"- Размеры задачи n: `{list(settings.item_counts)}`",
        f"- Количество задач: `{settings.task_count}`",
        f"- Максимум поколений: `{settings.generations}`",
        f"- Вероятность мутации: `{settings.mutation_rate}`",
        f"- Вероятность кроссовера: `{settings.crossover_rate}`",
        f"- Размер турнира: `{settings.tournament_size}`",
        f"- Пары population:stop_limit: `{[f'{population}:{stop_limit}' for population, stop_limit in settings.population_stop_pairs]}`",
        f"- Базовый seed: `{settings.seed}`",
        "",
        "## Сводные результаты",
        "",
        f"- Всего прогонов: `{len(rows)}`",
        f"- Папка результатов: `{output_dir}`",
        (
            "- Лучшая конфигурация: "
            f"`n={best_config['items']}, "
            f"размер популяции={best_config['population_size']}, "
            f"порог остановки без улучшения={best_config['stagnation']}`"
        ),
        "",
        "| n | Размер популяции | Порог остановки без улучшения | Прогонов | Точных совпадений | Доля точных совпадений % | Средний fitness | Среднее число поколений | Среднее время, сек |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['items']} | "
            f"{row['population_size']} | "
            f"{row['stagnation']} | "
            f"{row['runs']} | "
            f"{row['exact_matches']} | "
            f"{float(row['exact_match_rate']):.2f} | "
            f"{float(row['avg_fitness']):.2f} | "
            f"{float(row['avg_generations_used']):.2f} | "
            f"{float(row['avg_elapsed_seconds']):.6f} |"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_benchmark(settings: BenchmarkSettings) -> Path:
    output_dir = _build_output_dir(settings.output_root)
    started_at = datetime.now().astimezone()
    started_at_perf = perf_counter()
    rows: list[dict[str, object]] = []
    problems = _build_problems(settings)

    for population_size, stagnation in settings.population_stop_pairs:
        config = GeneticAlgorithmConfig(
            population_size=population_size,
            generations=settings.generations,
            stagnation=stagnation,
            crossover_rate=settings.crossover_rate,
            mutation_rate=settings.mutation_rate,
            tournament_size=settings.tournament_size,
        )
        for item_count, task_index, problem_seed, problem in problems:
            solver_seed = (
                settings.seed * 1_000_000
                + item_count * 10_000
                + population_size * 10
                + stagnation
                + task_index
            )
            task_started_at = perf_counter()
            result = solve_with_genetic_algorithm(
                prices=problem.prices,
                target_sum=problem.target_sum,
                config=config,
                rng=Random(solver_seed),
            )
            task_elapsed_seconds = perf_counter() - task_started_at
            rows.append(
                {
                    "task_index": task_index,
                    "problem_seed": problem_seed,
                    "solver_seed": solver_seed,
                    "generation_mode": GENERATION_MODE_LABELS[problem.generation_mode],
                    "items": item_count,
                    "population_size": population_size,
                    "stagnation": stagnation,
                    "generations": settings.generations,
                    "mutation_rate": settings.mutation_rate,
                    "crossover_rate": settings.crossover_rate,
                    "tournament_size": settings.tournament_size,
                    "target_sum": problem.target_sum,
                    "hidden_vector": _format_vector(problem.hidden_vector),
                    "best_vector": _format_vector(result.best_vector),
                    "best_sum": result.best_sum,
                    "fitness": result.fitness,
                    "difference": result.difference,
                    "generations_used": result.generations_used,
                    "exact_match": _format_exact_match(result.exact_match),
                    "stop_reason": _format_stop_reason(result.stop_reason),
                    "elapsed_seconds": f"{task_elapsed_seconds:.6f}",
                }
            )

    finished_at = datetime.now().astimezone()
    total_elapsed_seconds = perf_counter() - started_at_perf
    summary_rows = _summarize_rows(rows)
    _write_csv(output_dir, rows)
    _write_summary(
        output_dir=output_dir,
        settings=settings,
        rows=rows,
        summary_rows=summary_rows,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=total_elapsed_seconds,
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = BenchmarkSettings(
        item_counts=args.items,
        task_count=args.tasks,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        population_stop_pairs=args.population_stop_pairs,
        generation_mode=args.generation_mode,
        seed=args.seed,
        output_root=args.output_root,
    )
    output_dir = run_benchmark(settings)
    print(f"Результаты бенчмарка сохранены в: {output_dir}")
    print(f"Режим генерации: {GENERATION_MODE_LABELS[settings.generation_mode]}")
    print(f"CSV-файл: {output_dir / 'results.csv'}")
    print(f"Markdown-файл: {output_dir / 'summary.md'}")
    return 0
