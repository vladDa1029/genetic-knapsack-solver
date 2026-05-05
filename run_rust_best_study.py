from __future__ import annotations

import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

from genetic_knapsack_solver.generator import generate_problem
from genetic_knapsack_solver.models import GeneticAlgorithmConfig
from genetic_knapsack_solver.rust_core import rust_core_available, solve_with_rust_core

# =========================
# Настройки исследования
# =========================
# Меняй значения в этом блоке и запускай:
#   uv run python run_rust_best_study.py
#
# Скрипт использует Rust-ядро и текущий лучший метод по умолчанию:
# two_stage_restart + two_point/two_point/four_children_select_two.

# Размерности задач. Например: [28], [28, 29] или [27, 28, 29].
ITEM_COUNTS = [28, 29]

# Число независимых задач/повторов для каждого n.
REPEATS = 50

# Сколько n считать одновременно. Для [28, 29] и PROCESSES=2 будет два процесса.
PROCESSES = 2

# Пара population:stagnation. Для текущих исследований обычно 5000:5000.
POPULATION_SIZE = 5000
STAGNATION = 5000

# Жёсткий верхний лимит поколений на один запуск. Обычно остановка происходит раньше по STAGNATION.
GENERATIONS = 500_000

# Текущий подтверждённый best-конфиг для n=28.
CROSSOVER_RATE = 0.95
MUTATION_RATE = 0.9
TOURNAMENT_SIZE = 3
RESTART_MUTATION_FRACTION = 0.45

# Базовый seed. Для repeat_index и n из него детерминированно строятся problem_seed/solver_seed.
SEED = 42

# Метод второго этапа в two_stage_restart.
STAGE2_CROSSOVER_TYPE = "two_point"
STAGE2_MUTATION_TYPE = "two_point"
STAGE2_OFFSPRING_MODE = "four_children_select_two"
GENERATION_MODE = "superincreasing_disguised"

# Пустая строка: автоматически создать benchmark_results/<timestamp>_rust_best_...
# Можно указать путь явно, чтобы продолжить/перезапустить конкретное исследование.
OUTPUT_DIR = ""

# Если True, уже завершённые повторы из OUTPUT_DIR не пересчитываются.
RESUME = True

# Если True, удалить OUTPUT_DIR перед запуском. Осторожно: удаляет старые результаты.
FORCE_RECREATE = False

# Как часто главный процесс обновляет progress.md, пока работают дочерние процессы.
PROGRESS_INTERVAL_SECONDS = 30

CSV_FIELDS = (
    "item_count",
    "repeat_index",
    "problem_seed",
    "solver_seed",
    "generation_mode",
    "algorithm_mode",
    "population_size",
    "stagnation",
    "generations",
    "mutation_rate",
    "crossover_rate",
    "tournament_size",
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
    "stage1_run_count",
    "stage2_run_count",
    "stage2_used",
    "stage1_best_differences",
    "stage2_best_differences",
    "stop_reason",
    "elapsed_seconds",
    "status",
    "reason",
)


@dataclass(frozen=True)
class StudyConfig:
    item_counts: list[int]
    repeats: int
    processes: int
    population_size: int
    stagnation: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int
    restart_mutation_fraction: float
    seed: int
    stage2_crossover_type: str
    stage2_mutation_type: str
    stage2_offspring_mode: str
    generation_mode: str
    output_dir: Path
    resume: bool


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def build_output_dir() -> Path:
    root = repo_root()
    if OUTPUT_DIR:
        path = Path(OUTPUT_DIR)
        return path if path.is_absolute() else Path.cwd() / path

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    items = "_".join(str(item) for item in ITEM_COUNTS)
    return root / "benchmark_results" / f"{timestamp}_rust_best_items_{items}_r{REPEATS}"


def build_config() -> StudyConfig:
    if not ITEM_COUNTS:
        raise ValueError("ITEM_COUNTS must not be empty")
    if REPEATS < 1:
        raise ValueError("REPEATS must be >= 1")
    if PROCESSES < 1:
        raise ValueError("PROCESSES must be >= 1")
    return StudyConfig(
        item_counts=[int(item) for item in ITEM_COUNTS],
        repeats=int(REPEATS),
        processes=min(int(PROCESSES), len(ITEM_COUNTS)),
        population_size=int(POPULATION_SIZE),
        stagnation=int(STAGNATION),
        generations=int(GENERATIONS),
        crossover_rate=float(CROSSOVER_RATE),
        mutation_rate=float(MUTATION_RATE),
        tournament_size=int(TOURNAMENT_SIZE),
        restart_mutation_fraction=float(RESTART_MUTATION_FRACTION),
        seed=int(SEED),
        stage2_crossover_type=STAGE2_CROSSOVER_TYPE,
        stage2_mutation_type=STAGE2_MUTATION_TYPE,
        stage2_offspring_mode=STAGE2_OFFSPRING_MODE,
        generation_mode=GENERATION_MODE,
        output_dir=build_output_dir(),
        resume=bool(RESUME),
    )


def ensure_rust_core() -> None:
    """Проверяет доступность Rust DLL и собирает rust_core, если DLL ещё нет."""

    if rust_core_available():
        return

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("Rust core is unavailable and `cargo` was not found in PATH")

    manifest = repo_root() / "rust_core" / "Cargo.toml"
    print("Rust core is not built. Running cargo build --release...")
    subprocess.run([cargo, "build", "--release", "--manifest-path", str(manifest)], check=True)
    if not rust_core_available():
        raise RuntimeError("Rust core is still unavailable after cargo build")


def item_dir(config: StudyConfig, item_count: int) -> Path:
    return config.output_dir / f"n{item_count}"


def state_path(config: StudyConfig, item_count: int) -> Path:
    return item_dir(config, item_count) / "benchmark_state.json"


def csv_path(config: StudyConfig, item_count: int) -> Path:
    return item_dir(config, item_count) / "results.csv"


def summary_path(config: StudyConfig, item_count: int) -> Path:
    return item_dir(config, item_count) / "summary.md"


def stdout_path(config: StudyConfig, item_count: int) -> Path:
    return item_dir(config, item_count) / "stdout.log"


def stderr_path(config: StudyConfig, item_count: int) -> Path:
    return item_dir(config, item_count) / "stderr.log"


def log(config: StudyConfig, item_count: int, message: str) -> None:
    with stdout_path(config, item_count).open("a", encoding="utf-8") as file:
        file.write(f"[{now_iso()}] {message}\n")


def log_error(config: StudyConfig, item_count: int, message: str) -> None:
    with stderr_path(config, item_count).open("a", encoding="utf-8") as file:
        file.write(f"[{now_iso()}] {message}\n")


def problem_seed(config: StudyConfig, item_count: int, repeat_index: int) -> int:
    return config.seed + item_count * 10_000 + repeat_index


def solver_seed(config: StudyConfig, item_count: int, repeat_index: int) -> int:
    # Seed зависит от n/repeat/population/stagnation, чтобы разные сценарии были воспроизводимыми.
    return (
        config.seed * 1_000_000_000
        + 1 * 100_000_000
        + item_count * 1_000_000
        + repeat_index * 10_000
        + config.population_size * 10
        + config.stagnation
    )


def load_records(config: StudyConfig, item_count: int) -> dict[int, dict[str, Any]]:
    """Загружает уже завершённые повторы для resume."""

    if not config.resume:
        return {}
    path = state_path(config, item_count)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["repeat_index"]): record for record in payload.get("records", [])}


def write_csv(config: StudyConfig, item_count: int, records: list[dict[str, Any]]) -> None:
    with csv_path(config, item_count).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Считает основные метрики по завершённым повторам."""

    completed = [record for record in records if record.get("status") == "completed"]
    exact = sum(1 for record in completed if record.get("exact_match") is True)
    stage2 = sum(1 for record in completed if record.get("stage2_used") is True)
    return {
        "completed": len(completed),
        "exact": exact,
        "exact_rate": 100.0 * exact / len(completed) if completed else 0.0,
        "avg_diff": statistics.mean(float(record["difference"]) for record in completed) if completed else None,
        "median_diff": statistics.median(float(record["difference"]) for record in completed) if completed else None,
        "avg_generations": statistics.mean(float(record["generations_used"]) for record in completed) if completed else None,
        "avg_seconds": statistics.mean(float(record["elapsed_seconds"]) for record in completed) if completed else None,
        "stage2_rate": 100.0 * stage2 / len(completed) if completed else 0.0,
        "avg_stage1_runs": statistics.mean(float(record["stage1_run_count"]) for record in completed) if completed else None,
        "avg_stage2_runs": statistics.mean(float(record["stage2_run_count"]) for record in completed) if completed else None,
    }


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def config_name(config: StudyConfig) -> str:
    return f"rf{config.restart_mutation_fraction}_m{config.mutation_rate}_t{config.tournament_size}"


def write_summary(config: StudyConfig, item_count: int, records: list[dict[str, Any]], finished: bool) -> None:
    summary = summarize_records(records)
    lines = [
        "# Сводка Rust-исследования",
        "",
        f"- n: `{item_count}`",
        f"- Конфигурация: `{config_name(config)}`",
        f"- restart_mutation_fraction: `{config.restart_mutation_fraction}`",
        f"- mutation_rate: `{config.mutation_rate}`",
        f"- tournament_size: `{config.tournament_size}`",
        f"- population:stagnation: `{config.population_size}:{config.stagnation}`",
        f"- Повторов: `{config.repeats}`",
        f"- Завершено полностью: `{finished}`",
        "",
        "| завершено | exact | exact % | ср. diff | медиана diff | ср. поколений | ср. секунд | stage2 % | ср. запусков stage1 | ср. запусков stage2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['completed']} | {summary['exact']} | {fmt(summary['exact_rate'])} | "
            f"{fmt(summary['avg_diff'])} | {fmt(summary['median_diff'])} | "
            f"{fmt(summary['avg_generations'])} | {fmt(summary['avg_seconds'])} | "
            f"{fmt(summary['stage2_rate'])} | {fmt(summary['avg_stage1_runs'])} | "
            f"{fmt(summary['avg_stage2_runs'])} |"
        ),
    ]
    summary_path(config, item_count).write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist(config: StudyConfig, item_count: int, records_by_repeat: dict[int, dict[str, Any]], finished: bool) -> None:
    """Сохраняет состояние после каждого повтора, чтобы долгий запуск можно было продолжить."""

    records = [records_by_repeat[index] for index in sorted(records_by_repeat)]
    payload = {
        "metadata": {
            "updated_at": now_iso(),
            "finished_at": now_iso() if finished else None,
            "solver": "rust_core",
            "config_name": config_name(config),
            "settings": {
                "item_count": item_count,
                "repeats": config.repeats,
                "population_size": config.population_size,
                "stagnation": config.stagnation,
                "generations": config.generations,
                "crossover_rate": config.crossover_rate,
                "mutation_rate": config.mutation_rate,
                "tournament_size": config.tournament_size,
                "restart_mutation_fraction": config.restart_mutation_fraction,
                "stage2_crossover_type": config.stage2_crossover_type,
                "stage2_mutation_type": config.stage2_mutation_type,
                "stage2_offspring_mode": config.stage2_offspring_mode,
                "generation_mode": config.generation_mode,
                "seed": config.seed,
            },
        },
        "records": records,
        "issues": [record for record in records if record.get("status") != "completed"],
    }
    state_path(config, item_count).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(config, item_count, records)
    write_summary(config, item_count, records, finished)


def initialize_output(config: StudyConfig) -> None:
    if FORCE_RECREATE and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for item_count in config.item_counts:
        item_dir(config, item_count).mkdir(parents=True, exist_ok=True)
        for path in (stdout_path(config, item_count), stderr_path(config, item_count)):
            if not path.exists() or not config.resume:
                path.write_text("", encoding="utf-8")
    manifest = {
        "generated_at": now_iso(),
        "repo_root": str(repo_root()),
        "output_dir": str(config.output_dir),
        "solver": "rust_core",
        "config_name": config_name(config),
        "item_counts": config.item_counts,
        "repeats": config.repeats,
        "processes": config.processes,
        "population_size": config.population_size,
        "stagnation": config.stagnation,
        "generations": config.generations,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "tournament_size": config.tournament_size,
        "restart_mutation_fraction": config.restart_mutation_fraction,
        "stage2_crossover_type": config.stage2_crossover_type,
        "stage2_mutation_type": config.stage2_mutation_type,
        "stage2_offspring_mode": config.stage2_offspring_mode,
        "generation_mode": config.generation_mode,
        "seed": config.seed,
    }
    (config.output_dir / "study_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(config)


def run_item(config: StudyConfig, item_count: int) -> None:
    """Считает все repeats для одного n в отдельном процессе."""

    records_by_repeat = load_records(config, item_count)
    log(config, item_count, f"START n={item_count}")
    for repeat_index in range(1, config.repeats + 1):
        existing = records_by_repeat.get(repeat_index)
        if existing and existing.get("status") == "completed":
            continue

        p_seed = problem_seed(config, item_count, repeat_index)
        s_seed = solver_seed(config, item_count, repeat_index)
        problem = generate_problem(
            item_count=item_count,
            rng=Random(p_seed),
            generation_mode=config.generation_mode,
        )
        ga_config = GeneticAlgorithmConfig(
            solver_mode="two_stage_restart",
            population_size=config.population_size,
            generations=config.generations,
            stagnation=config.stagnation,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            tournament_size=config.tournament_size,
            nga_mode="none",
            restart_mutation_fraction=config.restart_mutation_fraction,
            stage2_crossover_type=config.stage2_crossover_type,
            stage2_mutation_type=config.stage2_mutation_type,
            stage2_offspring_mode=config.stage2_offspring_mode,
        )
        started = perf_counter()
        try:
            result = solve_with_rust_core(
                prices=problem.prices,
                target_sum=problem.target_sum,
                config=ga_config,
                seed=s_seed,
            )
            elapsed = perf_counter() - started
            record = {
                "item_count": item_count,
                "repeat_index": repeat_index,
                "problem_seed": p_seed,
                "solver_seed": s_seed,
                "generation_mode": config.generation_mode,
                "algorithm_mode": "two_stage_restart",
                "population_size": config.population_size,
                "stagnation": config.stagnation,
                "generations": config.generations,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
                "tournament_size": config.tournament_size,
                "restart_mutation_fraction": config.restart_mutation_fraction,
                "stage2_crossover_type": config.stage2_crossover_type,
                "stage2_mutation_type": config.stage2_mutation_type,
                "stage2_offspring_mode": config.stage2_offspring_mode,
                "target_sum": problem.target_sum,
                "hidden_vector": json.dumps(problem.hidden_vector),
                "best_vector": json.dumps(result.best_vector),
                "best_sum": result.best_sum,
                "fitness": result.fitness,
                "difference": result.difference,
                "generations_used": result.generations_used,
                "exact_match": result.exact_match,
                "stage1_run_count": result.stage1_run_count,
                "stage2_run_count": result.stage2_run_count,
                "stage2_used": result.stage2_used,
                "stage1_best_differences": json.dumps(result.stage1_best_differences),
                "stage2_best_differences": json.dumps(result.stage2_best_differences),
                "stop_reason": result.stop_reason,
                "elapsed_seconds": elapsed,
                "status": "completed",
                "reason": "",
            }
            records_by_repeat[repeat_index] = record
            log(config, item_count, f"repeat={repeat_index} exact={result.exact_match} diff={result.difference} seconds={elapsed:.6f}")
        except Exception as error:
            elapsed = perf_counter() - started
            records_by_repeat[repeat_index] = {
                "item_count": item_count,
                "repeat_index": repeat_index,
                "problem_seed": p_seed,
                "solver_seed": s_seed,
                "generation_mode": config.generation_mode,
                "algorithm_mode": "two_stage_restart",
                "population_size": config.population_size,
                "stagnation": config.stagnation,
                "generations": config.generations,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
                "tournament_size": config.tournament_size,
                "restart_mutation_fraction": config.restart_mutation_fraction,
                "stage2_crossover_type": config.stage2_crossover_type,
                "stage2_mutation_type": config.stage2_mutation_type,
                "stage2_offspring_mode": config.stage2_offspring_mode,
                "elapsed_seconds": elapsed,
                "status": "failed",
                "reason": str(error),
            }
            log_error(config, item_count, f"repeat={repeat_index} failed: {error}\n{traceback.format_exc()}")
            persist(config, item_count, records_by_repeat, finished=False)
            raise
        persist(config, item_count, records_by_repeat, finished=False)
    persist(config, item_count, records_by_repeat, finished=True)
    log(config, item_count, f"DONE n={item_count}")


def collect(config: StudyConfig) -> tuple[list[dict[str, Any]], list[str], int]:
    """Собирает промежуточные или финальные результаты по всем n."""

    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    total_completed = 0
    for item_count in config.item_counts:
        path = state_path(config, item_count)
        if not path.exists():
            issues.append(f"`n{item_count}`: missing benchmark_state.json")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        completed = [record for record in records if record.get("status") == "completed"]
        failed = [record for record in records if record.get("status") != "completed"]
        total_completed += len(completed)
        if len(completed) != config.repeats:
            issues.append(f"`n{item_count}`: completed records {len(completed)} / {config.repeats}")
        if failed:
            issues.append(f"`n{item_count}`: failed records {len(failed)}")
        if stderr_path(config, item_count).exists() and stderr_path(config, item_count).stat().st_size > 0:
            issues.append(f"`n{item_count}`: non-empty stderr.log")
        if completed:
            summary = summarize_records(completed)
            summary["item_count"] = item_count
            rows.append(summary)
    return rows, issues, total_completed


def write_progress(config: StudyConfig) -> None:
    rows, issues, completed = collect(config)
    finished = sum(1 for row in rows if row["completed"] == config.repeats)
    total = len(config.item_counts) * config.repeats
    lines = [
        "# Прогресс Rust-исследования",
        "",
        f"- Обновлено: `{now_iso()}`",
        f"- Завершено n: `{finished} / {len(config.item_counts)}`",
        f"- Завершено повторов: `{completed} / {total}`",
        f"- Найдено проблем: `{len(issues)}`",
    ]
    (config.output_dir / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(config: StudyConfig) -> None:
    rows, issues, completed = collect(config)
    rows = sorted(rows, key=lambda row: int(row["item_count"]))
    total = len(config.item_counts) * config.repeats
    lines = [
        "# Сравнение Rust-исследования",
        "",
        "## Параметры запуска",
        "",
        f"- Сформировано: `{now_iso()}`",
        f"- Папка результатов: `{config.output_dir}`",
        "- Решатель: `rust_core`",
        f"- Конфигурация: `{config_name(config)}`",
        f"- Метод 2 этапа: `{config.stage2_crossover_type}/{config.stage2_mutation_type}/{config.stage2_offspring_mode}`",
        f"- restart_mutation_fraction: `{config.restart_mutation_fraction}`",
        f"- mutation_rate: `{config.mutation_rate}`",
        f"- tournament_size: `{config.tournament_size}`",
        f"- population:stagnation: `{config.population_size}:{config.stagnation}`",
        f"- Повторов на каждое n: `{config.repeats}`",
        f"- Запланировано Rust-вызовов: `{total}`",
        f"- Завершено Rust-вызовов: `{completed}`",
        "",
        "## Результаты",
        "",
        "| n | завершено | exact | exact % | ср. diff | медиана diff | ср. поколений | ср. секунд | stage2 % | ср. запусков stage1 | ср. запусков stage2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['item_count']} | {row['completed']} | {row['exact']} | {fmt(row['exact_rate'])} | "
            f"{fmt(row['avg_diff'])} | {fmt(row['median_diff'])} | {fmt(row['avg_generations'])} | "
            f"{fmt(row['avg_seconds'])} | {fmt(row['stage2_rate'])} | {fmt(row['avg_stage1_runs'])} | "
            f"{fmt(row['avg_stage2_runs'])} |"
        )

    lines.extend(["", "## Итоговая сводка", ""])
    if rows:
        for row in rows:
            lines.append(
                f"- `n={row['item_count']}`: `{row['exact']}/{row['completed']}` exact, "
                f"`{fmt(row['exact_rate'])}%`, средний diff `{fmt(row['avg_diff'])}`."
            )
    else:
        lines.append("- Завершённых результатов пока нет.")

    lines.extend(["", "## Проверки и проблемы", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- Ошибок, пропущенных повторов и непустых stderr-логов не найдено.")
    (config.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_progress(config)


def worker_main(config: StudyConfig, item_count: int) -> int:
    try:
        ensure_rust_core()
        run_item(config, item_count)
        return 0
    except Exception:
        item_dir(config, item_count).mkdir(parents=True, exist_ok=True)
        log_error(config, item_count, traceback.format_exc())
        return 1


def parent_main(config: StudyConfig) -> int:
    ensure_rust_core()
    initialize_output(config)
    print(f"Папка результатов: {config.output_dir}")
    print(
        "Конфигурация: rust_core, two_stage_restart, "
        f"{config.stage2_crossover_type}/{config.stage2_mutation_type}/{config.stage2_offspring_mode}, "
        f"rf={config.restart_mutation_fraction}, mutation={config.mutation_rate}, tournament={config.tournament_size}"
    )

    queue = list(config.item_counts)
    running: list[tuple[int, subprocess.Popen[bytes]]] = []
    all_processes: list[dict[str, Any]] = []

    while queue or running:
        while queue and len(running) < config.processes:
            item_count = queue.pop(0)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(item_count)])
            running.append((item_count, process))
            all_processes.append({"item_count": item_count, "pid": process.pid, "started_at": now_iso()})
            (config.output_dir / "process_manifest.json").write_text(
                json.dumps(all_processes, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        time.sleep(PROGRESS_INTERVAL_SECONDS)
        still_running: list[tuple[int, subprocess.Popen[bytes]]] = []
        for item_count, process in running:
            exit_code = process.poll()
            if exit_code is None:
                still_running.append((item_count, process))
            elif exit_code != 0:
                print(f"Worker n={item_count} завершился с ошибкой, exit code {exit_code}")
        running = still_running
        write_progress(config)
        print((config.output_dir / "progress.md").read_text(encoding="utf-8").strip())

    write_report(config)
    print(f"Готово. Отчёт: {config.output_dir / 'comparison.md'}")
    return 0


def main() -> int:
    config = build_config()
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return worker_main(config, int(sys.argv[2]))
    if len(sys.argv) != 1:
        print("Usage: uv run python run_rust_best_study.py", file=sys.stderr)
        return 2
    return parent_main(config)


if __name__ == "__main__":
    raise SystemExit(main())
