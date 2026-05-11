from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from genetic_knapsack_solver.benchmark import (
    BenchmarkSettings,
    CSV_FIELD_LABELS,
    _parse_algorithm_modes,
    _parse_population_stop_pairs,
    run_benchmark,
)
from genetic_knapsack_solver.generator import GENERATION_MODE_SUPERINCREASING_DISGUISED
from genetic_knapsack_solver.rust_core import rust_core_available


class BenchmarkTests(unittest.TestCase):
    def test_parse_population_stop_pairs(self) -> None:
        self.assertEqual(
            _parse_population_stop_pairs("3000:3000, 5000:5000"),
            ((3000, 3000), (5000, 5000)),
        )

    def test_parse_algorithm_modes(self) -> None:
        self.assertEqual(
            _parse_algorithm_modes(
                "none,two_point,elite_heavy_mutation,staged_hypermutation,restart_rescue,two_stage_restart,five_stage_restart"
            ),
            (
                "none",
                "two_point",
                "elite_heavy_mutation",
                "staged_hypermutation",
                "restart_rescue",
                "two_stage_restart",
                "five_stage_restart",
            ),
        )

    def test_benchmark_writes_csv_markdown_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result_dir = run_benchmark(
                BenchmarkSettings(
                    item_counts=(6,),
                    repeats=2,
                    generations=12,
                    mutation_rate=0.4,
                    crossover_rate=0.6,
                    tournament_size=2,
                    population_stop_pairs=((10, 3),),
                    algorithm_modes=("none", "staged_hypermutation"),
                    nga_mutation_fraction=0.5,
                    nga_trigger_points=(1,),
                    nga_mutate_points=(40,),
                    generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                    seed=123,
                    output_dir=output_dir,
                )
            )

            csv_path = result_dir / "results.csv"
            markdown_path = result_dir / "summary.md"
            state_path = result_dir / "benchmark_state.json"

            self.assertTrue(csv_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(state_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 4)
            self.assertIn("Режим алгоритма", rows[0])
            self.assertIn("Точки staged NGA", rows[0])
            self.assertIn("Поколения NGA", rows[0])
            self.assertIn("Максимум рестартов", rows[0])
            self.assertIn("Минимальная доля rescue-мутации", rows[0])
            self.assertIn("Количество рестартов", rows[0])
            self.assertIn("Rescue использован", rows[0])

            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Режимы алгоритма", markdown_text)
            self.assertIn("Без NGA", markdown_text)
            self.assertIn("NGA-3: staged hypermutation в точках стагнации", markdown_text)
            self.assertIn("Использований rescue", markdown_text)
            self.assertIn("Среднее число рестартов", markdown_text)

    def test_benchmark_supports_restart_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result_dir = run_benchmark(
                BenchmarkSettings(
                    item_counts=(6,),
                    repeats=1,
                    generations=12,
                    mutation_rate=0.4,
                    crossover_rate=0.6,
                    tournament_size=2,
                    population_stop_pairs=((10, 3),),
                    algorithm_modes=("restart_rescue",),
                    restart_max_count=1,
                    rescue_min_mutated_bits_ratio=0.5,
                    generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                    seed=555,
                    output_dir=output_dir,
                )
            )

            with (result_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Режим алгоритма"], "Restart rescue: полные рестарты и rescue-этап")
            self.assertEqual(rows[0]["Максимум рестартов"], "1")
            self.assertEqual(rows[0]["Минимальная доля rescue-мутации"], "0.5")
            self.assertIn("Количество рестартов", rows[0])
            self.assertIn("Rescue использован", rows[0])

            markdown_text = (result_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Restart rescue: полные рестарты и rescue-этап", markdown_text)
            self.assertIn("Использований rescue", markdown_text)

    def test_resume_does_not_duplicate_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            settings = BenchmarkSettings(
                item_counts=(6,),
                repeats=1,
                generations=8,
                mutation_rate=0.3,
                crossover_rate=0.5,
                tournament_size=2,
                population_stop_pairs=((10, 3),),
                algorithm_modes=("none", "staged_hypermutation"),
                nga_trigger_points=(1,),
                nga_mutate_points=(40,),
                generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                seed=321,
                output_dir=output_dir,
            )
            run_benchmark(settings)
            run_benchmark(replace(settings, resume=True))

            with (output_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)

    def test_benchmark_supports_two_stage_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result_dir = run_benchmark(
                BenchmarkSettings(
                    item_counts=(6,),
                    repeats=1,
                    generations=12,
                    mutation_rate=0.4,
                    crossover_rate=0.6,
                    tournament_size=2,
                    population_stop_pairs=((10, 3),),
                    algorithm_modes=("two_stage_restart",),
                    restart_mutation_fraction=0.5,
                    stage2_crossover_type="two_point",
                    stage2_mutation_type="reverse",
                    stage2_offspring_mode="four_children_select_two",
                    generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                    seed=777,
                    output_dir=output_dir,
                )
            )

            with (result_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Режим алгоритма"], "Two-stage restart: перенос элиты и отдельный 2 этап")
            self.assertEqual(rows[0]["Мутация 2 этапа"], "reverse")
            self.assertEqual(rows[0]["Схема потомков 2 этапа"], "four_children_select_two")
            self.assertIn("Запусков 1 этапа", rows[0])
            self.assertIn("Запусков 2 этапа", rows[0])
            self.assertIn("2 этап использован", rows[0])

            markdown_text = (result_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Использований 2 этапа", markdown_text)
            self.assertIn("Two-stage restart: перенос элиты и отдельный 2 этап", markdown_text)
            self.assertIn("Схема потомков 2 этапа", markdown_text)

    @unittest.skipUnless(rust_core_available(), "Rust core library is not built")
    def test_benchmark_supports_five_stage_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result_dir = run_benchmark(
                BenchmarkSettings(
                    item_counts=(6,),
                    repeats=1,
                    generations=3,
                    mutation_rate=0.0,
                    crossover_rate=0.0,
                    tournament_size=2,
                    population_stop_pairs=((6, 3),),
                    algorithm_modes=("five_stage_restart",),
                    multistage_crossover_type="two_point",
                    multistage_mutation_type="two_point",
                    multistage_elite_count=2,
                    generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                    seed=888,
                    output_dir=output_dir,
                )
            )

            with (result_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0][CSV_FIELD_LABELS["algorithm_mode"]],
                "Five-stage restart: 5 этапов в Rust core",
            )
            self.assertEqual(rows[0][CSV_FIELD_LABELS["multistage_crossover_type"]], "two_point")
            self.assertEqual(rows[0][CSV_FIELD_LABELS["multistage_mutation_type"]], "two_point")
            self.assertEqual(rows[0][CSV_FIELD_LABELS["multistage_elite_count"]], "2")
            self.assertIn(CSV_FIELD_LABELS["final_stage"], rows[0])
            self.assertIn(CSV_FIELD_LABELS["stage_best_differences"], rows[0])

            markdown_text = (result_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Five-stage restart: 5 этапов в Rust core", markdown_text)
            self.assertIn("Средний финальный этап", markdown_text)


if __name__ == "__main__":
    unittest.main()
