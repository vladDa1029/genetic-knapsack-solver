from __future__ import annotations

import argparse
import csv
import json
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter
from typing import Any, Literal, Sequence

from genetic_knapsack_solver.ga import solve_with_genetic_algorithm
from genetic_knapsack_solver.generator import (
    GENERATION_MODE_LABELS,
    GENERATION_MODE_SUPERINCREASING_DISGUISED,
    generate_problem,
)
from genetic_knapsack_solver.models import (
    CrossoverType,
    GeneticAlgorithmConfig,
    MutationType,
    NgaMode,
    ProblemInstance,
    RestartPopulationMode,
    Stage2OffspringMode,
)

BenchmarkMode = Literal[
    "none",
    "two_point",
    "elite_heavy_mutation",
    "staged_hypermutation",
    "restart_rescue",
    "two_stage_restart",
]

ALGORITHM_MODE_LABELS: dict[BenchmarkMode, str] = {
    "none": "Без NGA",
    "two_point": "NGA-1: двухточечный кроссовер и двухточечная мутация",
    "elite_heavy_mutation": "NGA-2: сохранить лучшую особь и сильно мутировать остальные",
    "staged_hypermutation": "NGA-3: staged hypermutation в точках стагнации",
    "restart_rescue": "Restart rescue: полные рестарты и rescue-этап",
    "two_stage_restart": "Two-stage restart: перенос элиты и отдельный 2 этап",
}

STOP_REASON_LABELS = {
    "exact_match": "точное совпадение",
    "stagnation": "стагнация",
    "generation_limit": "лимит поколений",
    "restart_limit": "лимит рестартов",
    "rescue_stagnation": "rescue завершился стагнацией",
    "rescue_generation_limit": "rescue достиг лимита поколений",
}

CSV_FIELD_LABELS = {
    "item_count": "Количество предметов",
    "repeat_index": "Индекс повтора",
    "problem_seed": "Seed задачи",
    "solver_seed": "Seed решателя",
    "generation_mode": "Режим генерации",
    "algorithm_mode": "Режим алгоритма",
    "population_size": "Размер популяции",
    "repeat_limit": "Лимит NGA без улучшения",
    "stagnation": "Порог остановки без улучшения",
    "generations": "Максимум поколений",
    "mutation_rate": "Вероятность мутации",
    "crossover_rate": "Вероятность кроссовера",
    "tournament_size": "Размер турнира",
    "nga_mutation_fraction": "Доля сильной мутации NGA",
    "nga_trigger_points": "Точки staged NGA",
    "nga_mutate_points": "Сила staged NGA (%)",
    "restart_max_count": "Максимум рестартов",
    "rescue_min_mutated_bits_ratio": "Минимальная доля rescue-мутации",
    "target_sum": "Целевая сумма",
    "hidden_vector": "Скрытый вектор",
    "best_vector": "Лучший вектор",
    "best_sum": "Найденная сумма",
    "fitness": "Fitness",
    "difference": "Абсолютная разница",
    "generations_used": "Использовано поколений",
    "exact_match": "Точное совпадение",
    "nga_used": "NGA использован",
    "nga_trigger_generation": "Поколение NGA",
    "nga_trigger_generations": "Поколения NGA",
    "restart_count": "Количество рестартов",
    "rescue_used": "Rescue использован",
    "restart_population_mode": "Режим restart-популяции",
    "restart_mutation_fraction": "Сила restart-мутации",
    "stage2_crossover_type": "Кроссовер 2 этапа",
    "stage2_mutation_type": "Мутация 2 этапа",
    "stage2_offspring_mode": "Схема потомков 2 этапа",
    "stage1_run_count": "Запусков 1 этапа",
    "stage2_run_count": "Запусков 2 этапа",
    "stage2_used": "2 этап использован",
    "stage1_best_differences": "История difference 1 этапа",
    "stage2_best_differences": "История difference 2 этапа",
    "stop_reason": "Причина остановки",
    "elapsed_seconds": "Время выполнения (сек)",
    "status": "Статус",
    "reason": "Причина ошибки",
}


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    item_counts: tuple[int, ...] = (25, 26)
    repeats: int = 33
    generations: int = 500000
    mutation_rate: float = 0.9
    crossover_rate: float = 0.95
    tournament_size: int = 3
    population_stop_pairs: tuple[tuple[int, int], ...] = (
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (3000, 3000),
        (5000, 5000),
    )
    algorithm_modes: tuple[BenchmarkMode, ...] = (
        "none",
        "two_point",
        "elite_heavy_mutation",
    )
    nga_mutation_fraction: float = 0.4
    nga_trigger_points: tuple[int, ...] = ()
    nga_mutate_points: tuple[int, ...] = (40,)
    restart_max_count: int | None = None
    rescue_min_mutated_bits_ratio: float = 0.4
    restart_population_mode: RestartPopulationMode = "elite_from_last_population"
    restart_mutation_fraction: float = 0.4
    stage2_crossover_type: CrossoverType = "two_point"
    stage2_mutation_type: MutationType = "two_point"
    stage2_offspring_mode: Stage2OffspringMode = "two_children"
    generation_mode: str = GENERATION_MODE_SUPERINCREASING_DISGUISED
    seed: int = 42
    output_root: Path = Path("benchmark_results")
    output_dir: Path | None = None
    resume: bool = False


@dataclass(slots=True)
class BenchmarkRecord:
    item_count: int
    repeat_index: int
    generation_mode: str
    algorithm_mode: str
    population_size: int
    repeat_limit: int | None
    stagnation: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    nga_mutation_fraction: float
    nga_trigger_points: str = "[]"
    nga_mutate_points: str = "[40]"
    restart_max_count: int | None = None
    rescue_min_mutated_bits_ratio: float = 0.4
    restart_population_mode: str = "elite_from_last_population"
    restart_mutation_fraction: float = 0.4
    stage2_crossover_type: str = "two_point"
    stage2_mutation_type: str = "two_point"
    stage2_offspring_mode: str = "two_children"
    problem_seed: int = 0
    solver_seed: int = 0
    target_sum: int | None = None
    hidden_vector: str | None = None
    best_vector: str | None = None
    best_sum: int | None = None
    fitness: int | None = None
    difference: int | None = None
    generations_used: int | None = None
    exact_match: bool | None = None
    nga_used: bool | None = None
    nga_trigger_generation: int | None = None
    nga_trigger_generations: str | None = None
    restart_count: int | None = None
    rescue_used: bool | None = None
    stage1_run_count: int | None = None
    stage2_run_count: int | None = None
    stage2_used: bool | None = None
    stage1_best_differences: str | None = None
    stage2_best_differences: str | None = None
    stop_reason: str = "failed"
    elapsed_seconds: float | None = None
    status: str = "failed"
    reason: str = ""


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
            raise argparse.ArgumentTypeError("expected pairs in format population:stop_limit")
        population_size = int(population_size_raw.strip())
        stop_limit = int(stop_limit_raw.strip())
        pairs.append((population_size, stop_limit))

    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one pair")
    return tuple(pairs)


def _parse_algorithm_modes(raw: str) -> tuple[BenchmarkMode, ...]:
    modes = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not modes:
        raise argparse.ArgumentTypeError("expected at least one algorithm mode")

    invalid_modes = [mode for mode in modes if mode not in ALGORITHM_MODE_LABELS]
    if invalid_modes:
        raise argparse.ArgumentTypeError(f"unknown algorithm modes: {', '.join(invalid_modes)}")
    return modes  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Бенчмарк базового ГА и доступных NGA-режимов.",
    )
    parser.add_argument(
        "--items",
        type=_parse_int_list,
        default=(25, 26),
        help="Список размеров задачи n через запятую.",
    )
    parser.add_argument(
        "--repeats",
        "--tasks",
        dest="repeats",
        type=int,
        default=33,
        help="Количество независимых задач на каждую конфигурацию.",
    )
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
        help="Вероятность мутации базового ГА.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.95,
        help="Вероятность кроссовера базового ГА.",
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
        default=((500, 500), (1000, 1000), (2000, 2000), (3000, 3000), (5000, 5000)),
        help="Список пар population:stop_limit через запятую.",
    )
    parser.add_argument(
        "--algorithm-modes",
        type=_parse_algorithm_modes,
        default=("none", "two_point", "elite_heavy_mutation"),
        help="Список режимов алгоритма через запятую.",
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
        help="Точные точки стагнации для staged_hypermutation.",
    )
    parser.add_argument(
        "--nga-mutate-points",
        type=_parse_int_list,
        default=(40,),
        help="Проценты сильной мутации для staged_hypermutation.",
    )
    parser.add_argument(
        "--restart-max-count",
        type=int,
        default=None,
        help="Максимальное число полных рестартов для restart_rescue; если не задано, лимит не применяется.",
    )
    parser.add_argument(
        "--rescue-min-mutated-bits-ratio",
        type=float,
        default=0.4,
        help="Минимальная доля мутируемых битов для rescue-этапа restart_rescue.",
    )
    parser.add_argument(
        "--restart-population-mode",
        choices=("elite_from_last_population",),
        default="elite_from_last_population",
        help="Способ построения новой популяции для two_stage_restart.",
    )
    parser.add_argument(
        "--restart-mutation-fraction",
        type=float,
        default=0.4,
        help="Доля битов для сильной мутации между запусками two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-crossover-type",
        choices=("one_point", "two_point"),
        default="two_point",
        help="Тип кроссовера на 2 этапе two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-mutation-type",
        choices=("one_point", "two_point", "reverse"),
        default="two_point",
        help="Тип мутации на 2 этапе two_stage_restart.",
    )
    parser.add_argument(
        "--stage2-offspring-mode",
        choices=("two_children", "four_children_select_two"),
        default="two_children",
        help="РЎС…РµРјР° РїРѕСЃС‚СЂРѕРµРЅРёСЏ РїРѕС‚РѕРјРєРѕРІ РЅР° 2 СЌС‚Р°РїРµ two_stage_restart.",
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
        help="Корневая папка для новых каталогов результатов.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Явный каталог результатов для нового запуска или resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить benchmark из существующего каталога результатов.",
    )
    return parser


def _format_vector(vector: list[int]) -> str:
    return json.dumps(vector, ensure_ascii=False)


def _format_int_tuple(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _format_stop_reason(stop_reason: str) -> str:
    return STOP_REASON_LABELS.get(stop_reason, stop_reason)


def _format_exact_match(exact_match: bool | None) -> str:
    if exact_match is None:
        return "n/a"
    return "Да" if exact_match else "Нет"


def _format_nga_used(nga_used: bool | None) -> str:
    if nga_used is None:
        return "n/a"
    return "Да" if nga_used else "Нет"


def _format_rescue_used(rescue_used: bool | None) -> str:
    if rescue_used is None:
        return "n/a"
    return "Да" if rescue_used else "Нет"


def _format_stage2_used(stage2_used: bool | None) -> str:
    if stage2_used is None:
        return "n/a"
    return "Да" if stage2_used else "Нет"


def _build_output_dir(settings: BenchmarkSettings) -> Path:
    if settings.resume:
        if settings.output_dir is None:
            raise ValueError("resume requires --output-dir")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        return settings.output_dir

    if settings.output_dir is not None:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        return settings.output_dir

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = settings.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _build_problem_seed(settings: BenchmarkSettings, item_count: int, repeat_index: int) -> int:
    return settings.seed + item_count * 10_000 + repeat_index


def _build_solver_seed(
    settings: BenchmarkSettings,
    item_count: int,
    repeat_index: int,
    algorithm_mode: str,
    population_size: int,
    stop_limit: int,
) -> int:
    algorithm_index = settings.algorithm_modes.index(algorithm_mode) + 1
    return (
        settings.seed * 1_000_000_000
        + algorithm_index * 100_000_000
        + item_count * 1_000_000
        + repeat_index * 10_000
        + population_size * 10
        + stop_limit
    )


def _build_problem_cache(settings: BenchmarkSettings) -> dict[tuple[int, int], tuple[int, ProblemInstance]]:
    problems: dict[tuple[int, int], tuple[int, ProblemInstance]] = {}
    for item_count in settings.item_counts:
        for repeat_index in range(1, settings.repeats + 1):
            problem_seed = _build_problem_seed(settings, item_count, repeat_index)
            problem = generate_problem(
                item_count=item_count,
                rng=Random(problem_seed),
                generation_mode=settings.generation_mode,
            )
            problems[(item_count, repeat_index)] = (problem_seed, problem)
    return problems


def _build_record_key(payload: dict[str, Any] | BenchmarkRecord) -> tuple[int, int, str, int, int]:
    data = asdict(payload) if isinstance(payload, BenchmarkRecord) else payload
    return (
        int(data["item_count"]),
        int(data["repeat_index"]),
        str(data["algorithm_mode"]),
        int(data["population_size"]),
        int(data["stagnation"]),
    )


def _iter_matrix(settings: BenchmarkSettings) -> list[tuple[int, int, str, int, int]]:
    matrix: list[tuple[int, int, str, int, int]] = []
    for item_count in settings.item_counts:
        for repeat_index in range(1, settings.repeats + 1):
            for algorithm_mode in settings.algorithm_modes:
                for population_size, stop_limit in settings.population_stop_pairs:
                    matrix.append((item_count, repeat_index, algorithm_mode, population_size, stop_limit))
    return matrix


def _state_path(output_dir: Path) -> Path:
    return output_dir / "benchmark_state.json"


def _csv_path(output_dir: Path) -> Path:
    return output_dir / "results.csv"


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.md"


def _record_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field_info in fields(BenchmarkRecord):
        if field_info.default is not MISSING:
            defaults[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            defaults[field_info.name] = field_info.default_factory()  # type: ignore[misc]
    return defaults


def _benchmark_record_from_payload(payload: dict[str, Any]) -> BenchmarkRecord:
    merged = _record_defaults()
    merged.update(payload)
    return BenchmarkRecord(**merged)


def _load_existing_records(output_dir: Path) -> dict[tuple[int, int, str, int, int], BenchmarkRecord]:
    state_path = _state_path(output_dir)
    if not state_path.exists():
        return {}

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return {_build_record_key(record): _benchmark_record_from_payload(record) for record in records}


def _records_payload(
    settings: BenchmarkSettings,
    records: list[BenchmarkRecord],
    started_at: datetime,
    finished_at: datetime | None,
) -> dict[str, Any]:
    serialized_records = [asdict(record) for record in records]
    issues = [
        {"type": record.status, "record": asdict(record)}
        for record in records
        if record.status != "completed"
    ]
    return {
        "metadata": {
            "started_at": started_at.isoformat(timespec="microseconds"),
            "finished_at": None if finished_at is None else finished_at.isoformat(timespec="microseconds"),
            "settings": {
                "item_counts": list(settings.item_counts),
                "repeats": settings.repeats,
                "generations": settings.generations,
                "mutation_rate": settings.mutation_rate,
                "crossover_rate": settings.crossover_rate,
                "tournament_size": settings.tournament_size,
                "population_stop_pairs": [list(pair) for pair in settings.population_stop_pairs],
                "algorithm_modes": list(settings.algorithm_modes),
                "nga_mutation_fraction": settings.nga_mutation_fraction,
                "nga_trigger_points": list(settings.nga_trigger_points),
                "nga_mutate_points": list(settings.nga_mutate_points),
                "restart_max_count": settings.restart_max_count,
                "rescue_min_mutated_bits_ratio": settings.rescue_min_mutated_bits_ratio,
                "restart_population_mode": settings.restart_population_mode,
                "restart_mutation_fraction": settings.restart_mutation_fraction,
                "stage2_crossover_type": settings.stage2_crossover_type,
                "stage2_mutation_type": settings.stage2_mutation_type,
                "stage2_offspring_mode": settings.stage2_offspring_mode,
                "generation_mode": settings.generation_mode,
                "seed": settings.seed,
                "resume": settings.resume,
            },
        },
        "records": serialized_records,
        "issues": issues,
    }


def _write_csv(output_dir: Path, records: list[BenchmarkRecord]) -> None:
    field_order = [
        "item_count",
        "repeat_index",
        "problem_seed",
        "solver_seed",
        "generation_mode",
        "algorithm_mode",
        "population_size",
        "repeat_limit",
        "stagnation",
        "generations",
        "mutation_rate",
        "crossover_rate",
        "tournament_size",
        "nga_mutation_fraction",
        "nga_trigger_points",
        "nga_mutate_points",
        "restart_max_count",
        "rescue_min_mutated_bits_ratio",
        "restart_population_mode",
        "restart_mutation_fraction",
        "stage2_crossover_type",
        "stage2_mutation_type",
        "stage2_offspring_mode",
        "target_sum",
        "hidden_vector",
        "best_vector",
        "best_sum",
        "fitness",
        "difference",
        "generations_used",
        "exact_match",
        "nga_used",
        "nga_trigger_generation",
        "nga_trigger_generations",
        "restart_count",
        "rescue_used",
        "stage1_run_count",
        "stage2_run_count",
        "stage2_used",
        "stage1_best_differences",
        "stage2_best_differences",
        "stop_reason",
        "elapsed_seconds",
        "status",
        "reason",
    ]

    with _csv_path(output_dir).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[CSV_FIELD_LABELS[field] for field in field_order])
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["generation_mode"] = GENERATION_MODE_LABELS[row["generation_mode"]]
            row["algorithm_mode"] = ALGORITHM_MODE_LABELS[row["algorithm_mode"]]
            row["exact_match"] = _format_exact_match(row["exact_match"])
            row["nga_used"] = _format_nga_used(row["nga_used"])
            row["rescue_used"] = _format_rescue_used(row["rescue_used"])
            row["stage2_used"] = _format_stage2_used(row["stage2_used"])
            row["stop_reason"] = _format_stop_reason(row["stop_reason"])
            row["elapsed_seconds"] = "n/a" if row["elapsed_seconds"] is None else f"{float(row['elapsed_seconds']):.6f}"
            writer.writerow({CSV_FIELD_LABELS[field]: row[field] for field in field_order})


def _group_summary(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, int, int], list[BenchmarkRecord]] = {}
    for record in records:
        key = (record.item_count, record.algorithm_mode, record.population_size, record.stagnation)
        groups.setdefault(key, []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for (item_count, algorithm_mode, population_size, stop_limit), group_records in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
    ):
        completed_records = [record for record in group_records if record.status == "completed"]
        exact_matches = sum(1 for record in completed_records if record.exact_match)
        nga_used_runs = sum(1 for record in completed_records if record.nga_used)
        rescue_used_runs = sum(1 for record in completed_records if record.rescue_used)
        stage2_used_runs = sum(1 for record in completed_records if record.stage2_used)

        average_time = None
        average_generations = None
        average_fitness = None
        average_restart_count = None
        exact_rate = 0.0
        if completed_records:
            average_time = mean(float(record.elapsed_seconds or 0.0) for record in completed_records)
            average_generations = mean(int(record.generations_used or 0) for record in completed_records)
            average_fitness = mean(int(record.fitness or 0) for record in completed_records)
            average_restart_count = mean(int(record.restart_count or 0) for record in completed_records)
            exact_rate = 100.0 * exact_matches / len(completed_records)

        summary_rows.append(
            {
                "item_count": item_count,
                "algorithm_mode": algorithm_mode,
                "population_size": population_size,
                "stop_limit": stop_limit,
                "runs": len(completed_records),
                "exact_matches": exact_matches,
                "nga_used_runs": nga_used_runs,
                "rescue_used_runs": rescue_used_runs,
                "stage2_used_runs": stage2_used_runs,
                "exact_match_rate": exact_rate,
                "avg_fitness": average_fitness,
                "avg_generations_used": average_generations,
                "avg_restart_count": average_restart_count,
                "avg_elapsed_seconds": average_time,
            }
        )
    return summary_rows


def _write_summary(
    output_dir: Path,
    settings: BenchmarkSettings,
    records: list[BenchmarkRecord],
    started_at: datetime,
    finished_at: datetime | None,
) -> None:
    summary_rows = _group_summary(records)
    completed_records = [record for record in records if record.status == "completed"]
    elapsed_seconds = None if finished_at is None else (finished_at - started_at).total_seconds()

    lines = [
        "# Сводка бенчмарка",
        "",
        "## Параметры запуска",
        "",
        f"- Время начала: `{started_at.isoformat(timespec='microseconds')}`",
        f"- Время окончания: `{finished_at.isoformat(timespec='microseconds') if finished_at is not None else 'выполняется'}`",
        f"- Общее время, сек: `{f'{elapsed_seconds:.6f}' if elapsed_seconds is not None else 'выполняется'}`",
        f"- Режим генерации: `{GENERATION_MODE_LABELS[settings.generation_mode]}`",
        f"- Размеры задачи n: `{list(settings.item_counts)}`",
        f"- Повторов на конфигурацию: `{settings.repeats}`",
        f"- Режимы алгоритма: `{[ALGORITHM_MODE_LABELS[mode] for mode in settings.algorithm_modes]}`",
        f"- Максимум поколений: `{settings.generations}`",
        f"- Вероятность мутации: `{settings.mutation_rate}`",
        f"- Вероятность кроссовера: `{settings.crossover_rate}`",
        f"- Размер турнира: `{settings.tournament_size}`",
        f"- Пары population:stop_limit: `{[f'{population}:{stop_limit}' for population, stop_limit in settings.population_stop_pairs]}`",
        f"- Доля сильной мутации NGA: `{settings.nga_mutation_fraction}`",
        f"- Точки staged NGA: `{list(settings.nga_trigger_points)}`",
        f"- Сила staged NGA (%): `{list(settings.nga_mutate_points)}`",
        f"- Максимум рестартов restart_rescue: `{settings.restart_max_count if settings.restart_max_count is not None else 'без лимита'}`",
        f"- Минимальная доля rescue-мутации: `{settings.rescue_min_mutated_bits_ratio}`",
        f"- Режим restart-популяции: `{settings.restart_population_mode}`",
        f"- Сила restart-мутации: `{settings.restart_mutation_fraction}`",
        f"- Кроссовер 2 этапа: `{settings.stage2_crossover_type}`",
        f"- Мутация 2 этапа: `{settings.stage2_mutation_type}`",
        f"- Схема потомков 2 этапа: `{settings.stage2_offspring_mode}`",
        f"- Базовый seed: `{settings.seed}`",
        "",
        "## Сводные результаты",
        "",
        f"- Всего завершённых прогонов: `{len(completed_records)}`",
        f"- Всего запланированных прогонов: `{len(_iter_matrix(settings))}`",
        f"- Папка результатов: `{output_dir}`",
        "",
        "| n | Режим алгоритма | Размер популяции | Лимит без улучшения | Завершённых прогонов | Точных совпадений | Использований NGA | Использований rescue | Использований 2 этапа | Доля точных совпадений % | Средний fitness | Среднее число поколений | Среднее число рестартов | Среднее время, сек |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary_rows:
        avg_fitness = "n/a" if row["avg_fitness"] is None else f"{row['avg_fitness']:.2f}"
        avg_generations = "n/a" if row["avg_generations_used"] is None else f"{row['avg_generations_used']:.2f}"
        avg_restarts = "n/a" if row["avg_restart_count"] is None else f"{row['avg_restart_count']:.2f}"
        avg_elapsed = "n/a" if row["avg_elapsed_seconds"] is None else f"{row['avg_elapsed_seconds']:.6f}"
        lines.append(
            "| "
            f"{row['item_count']} | "
            f"{ALGORITHM_MODE_LABELS[row['algorithm_mode']]} | "
            f"{row['population_size']} | "
            f"{row['stop_limit']} | "
            f"{row['runs']} | "
            f"{row['exact_matches']} | "
            f"{row['nga_used_runs']} | "
            f"{row['rescue_used_runs']} | "
            f"{row['stage2_used_runs']} | "
            f"{row['exact_match_rate']:.2f} | "
            f"{avg_fitness} | "
            f"{avg_generations} | "
            f"{avg_restarts} | "
            f"{avg_elapsed} |"
        )

    _summary_path(output_dir).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist_results(
    output_dir: Path,
    settings: BenchmarkSettings,
    records_by_key: dict[tuple[int, int, str, int, int], BenchmarkRecord],
    started_at: datetime,
    finished_at: datetime | None,
) -> None:
    ordered_records = [records_by_key[key] for key in sorted(records_by_key)]
    payload = _records_payload(settings, ordered_records, started_at, finished_at)
    _state_path(output_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir, ordered_records)
    _write_summary(output_dir, settings, ordered_records, started_at, finished_at)


def run_benchmark(settings: BenchmarkSettings) -> Path:
    output_dir = _build_output_dir(settings)
    records_by_key = _load_existing_records(output_dir) if settings.resume else {}
    started_at = datetime.now().astimezone()
    if settings.resume and _state_path(output_dir).exists():
        payload = json.loads(_state_path(output_dir).read_text(encoding="utf-8"))
        started_at = datetime.fromisoformat(payload["metadata"]["started_at"])

    problems = _build_problem_cache(settings)
    matrix = _iter_matrix(settings)

    for item_count, repeat_index, algorithm_mode, population_size, stop_limit in matrix:
        record_key = (item_count, repeat_index, algorithm_mode, population_size, stop_limit)
        if record_key in records_by_key and records_by_key[record_key].status == "completed":
            continue

        if algorithm_mode == "restart_rescue":
            solver_mode = "restart_rescue"
        elif algorithm_mode == "two_stage_restart":
            solver_mode = "two_stage_restart"
        else:
            solver_mode = "classic"
        nga_mode: NgaMode = algorithm_mode if algorithm_mode in {
            "none",
            "two_point",
            "elite_heavy_mutation",
            "staged_hypermutation",
        } else "none"
        repeat_limit = stop_limit if algorithm_mode in {"two_point", "elite_heavy_mutation"} else None
        nga_trigger_points = settings.nga_trigger_points if algorithm_mode == "staged_hypermutation" else ()
        nga_mutate_points = settings.nga_mutate_points if algorithm_mode == "staged_hypermutation" else (40,)
        problem_seed, problem = problems[(item_count, repeat_index)]
        solver_seed = _build_solver_seed(
            settings,
            item_count,
            repeat_index,
            algorithm_mode,
            population_size,
            stop_limit,
        )

        try:
            config = GeneticAlgorithmConfig(
                solver_mode=solver_mode,
                population_size=population_size,
                generations=settings.generations,
                stagnation=stop_limit,
                repeat_limit=repeat_limit,
                crossover_rate=settings.crossover_rate,
                mutation_rate=settings.mutation_rate,
                tournament_size=settings.tournament_size,
                nga_mode=nga_mode,
                nga_mutation_fraction=settings.nga_mutation_fraction,
                nga_trigger_points=nga_trigger_points,
                nga_mutate_points=nga_mutate_points,
                restart_max_count=settings.restart_max_count,
                rescue_min_mutated_bits_ratio=settings.rescue_min_mutated_bits_ratio,
                restart_population_mode=settings.restart_population_mode,
                restart_mutation_fraction=settings.restart_mutation_fraction,
                stage2_crossover_type=settings.stage2_crossover_type,
                stage2_mutation_type=settings.stage2_mutation_type,
                stage2_offspring_mode=settings.stage2_offspring_mode,
            )
            task_started_at = perf_counter()
            result = solve_with_genetic_algorithm(
                prices=problem.prices,
                target_sum=problem.target_sum,
                config=config,
                rng=Random(solver_seed),
            )
            task_elapsed_seconds = perf_counter() - task_started_at
            record = BenchmarkRecord(
                item_count=item_count,
                repeat_index=repeat_index,
                generation_mode=problem.generation_mode,
                algorithm_mode=algorithm_mode,
                population_size=population_size,
                repeat_limit=repeat_limit,
                stagnation=stop_limit,
                generations=settings.generations,
                mutation_rate=settings.mutation_rate,
                crossover_rate=settings.crossover_rate,
                tournament_size=settings.tournament_size,
                nga_mutation_fraction=settings.nga_mutation_fraction,
                nga_trigger_points=_format_int_tuple(nga_trigger_points),
                nga_mutate_points=_format_int_tuple(nga_mutate_points),
                restart_max_count=settings.restart_max_count,
                rescue_min_mutated_bits_ratio=settings.rescue_min_mutated_bits_ratio,
                restart_population_mode=settings.restart_population_mode,
                restart_mutation_fraction=settings.restart_mutation_fraction,
                stage2_crossover_type=settings.stage2_crossover_type,
                stage2_mutation_type=settings.stage2_mutation_type,
                stage2_offspring_mode=settings.stage2_offspring_mode,
                problem_seed=problem_seed,
                solver_seed=solver_seed,
                target_sum=problem.target_sum,
                hidden_vector=_format_vector(problem.hidden_vector),
                best_vector=_format_vector(result.best_vector),
                best_sum=result.best_sum,
                fitness=result.fitness,
                difference=result.difference,
                generations_used=result.generations_used,
                exact_match=result.exact_match,
                nga_used=result.nga_used,
                nga_trigger_generation=result.nga_trigger_generation,
                nga_trigger_generations=_format_vector(result.nga_trigger_generations),
                restart_count=result.restart_count,
                rescue_used=result.rescue_used,
                stage1_run_count=result.stage1_run_count,
                stage2_run_count=result.stage2_run_count,
                stage2_used=result.stage2_used,
                stage1_best_differences=_format_vector(result.stage1_best_differences),
                stage2_best_differences=_format_vector(result.stage2_best_differences),
                stop_reason=result.stop_reason,
                elapsed_seconds=task_elapsed_seconds,
                status="completed",
            )
        except Exception as error:
            record = BenchmarkRecord(
                item_count=item_count,
                repeat_index=repeat_index,
                generation_mode=settings.generation_mode,
                algorithm_mode=algorithm_mode,
                population_size=population_size,
                repeat_limit=repeat_limit,
                stagnation=stop_limit,
                generations=settings.generations,
                mutation_rate=settings.mutation_rate,
                crossover_rate=settings.crossover_rate,
                tournament_size=settings.tournament_size,
                nga_mutation_fraction=settings.nga_mutation_fraction,
                nga_trigger_points=_format_int_tuple(nga_trigger_points),
                nga_mutate_points=_format_int_tuple(nga_mutate_points),
                restart_max_count=settings.restart_max_count,
                rescue_min_mutated_bits_ratio=settings.rescue_min_mutated_bits_ratio,
                restart_population_mode=settings.restart_population_mode,
                restart_mutation_fraction=settings.restart_mutation_fraction,
                stage2_crossover_type=settings.stage2_crossover_type,
                stage2_mutation_type=settings.stage2_mutation_type,
                stage2_offspring_mode=settings.stage2_offspring_mode,
                problem_seed=problem_seed,
                solver_seed=solver_seed,
                stop_reason="failed",
                status="failed",
                reason=str(error),
            )

        records_by_key[record_key] = record
        _persist_results(output_dir, settings, records_by_key, started_at, None)

    finished_at = datetime.now().astimezone()
    _persist_results(output_dir, settings, records_by_key, started_at, finished_at)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = BenchmarkSettings(
        item_counts=args.items,
        repeats=args.repeats,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        population_stop_pairs=args.population_stop_pairs,
        algorithm_modes=args.algorithm_modes,
        nga_mutation_fraction=args.nga_mutation_fraction,
        nga_trigger_points=args.nga_trigger_points,
        nga_mutate_points=args.nga_mutate_points,
        restart_max_count=args.restart_max_count,
        rescue_min_mutated_bits_ratio=args.rescue_min_mutated_bits_ratio,
        restart_population_mode=args.restart_population_mode,
        restart_mutation_fraction=args.restart_mutation_fraction,
        stage2_crossover_type=args.stage2_crossover_type,
        stage2_mutation_type=args.stage2_mutation_type,
        stage2_offspring_mode=args.stage2_offspring_mode,
        generation_mode=args.generation_mode,
        seed=args.seed,
        output_root=args.output_root,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    output_dir = run_benchmark(settings)
    print(f"Результаты бенчмарка сохранены в: {output_dir}")
    print(f"Режим генерации: {GENERATION_MODE_LABELS[settings.generation_mode]}")
    print(f"Режимы алгоритма: {[ALGORITHM_MODE_LABELS[mode] for mode in settings.algorithm_modes]}")
    print(f"Точки staged NGA: {list(settings.nga_trigger_points)}")
    print(f"Сила staged NGA (%): {list(settings.nga_mutate_points)}")
    print(f"Максимум рестартов restart_rescue: {settings.restart_max_count if settings.restart_max_count is not None else 'без лимита'}")
    print(f"Минимальная доля rescue-мутации: {settings.rescue_min_mutated_bits_ratio}")
    print(f"Режим restart-популяции: {settings.restart_population_mode}")
    print(f"Сила restart-мутации: {settings.restart_mutation_fraction}")
    print(f"Кроссовер 2 этапа: {settings.stage2_crossover_type}")
    print(f"Мутация 2 этапа: {settings.stage2_mutation_type}")
    print(f"Схема потомков 2 этапа: {settings.stage2_offspring_mode}")
    print(f"State-файл: {_state_path(output_dir)}")
    print(f"CSV-файл: {_csv_path(output_dir)}")
    print(f"Markdown-файл: {_summary_path(output_dir)}")
    return 0
