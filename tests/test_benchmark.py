from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from genetic_knapsack_solver.benchmark import (
    BenchmarkSettings,
    _parse_algorithm_modes,
    _parse_population_stop_pairs,
    run_benchmark,
)
from genetic_knapsack_solver.generator import GENERATION_MODE_SUPERINCREASING_DISGUISED


class BenchmarkTests(unittest.TestCase):
    def test_parse_population_stop_pairs(self) -> None:
        self.assertEqual(
            _parse_population_stop_pairs("3000:3000, 5000:5000"),
            ((3000, 3000), (5000, 5000)),
        )

    def test_parse_algorithm_modes(self) -> None:
        self.assertEqual(
            _parse_algorithm_modes("none,two_point,elite_heavy_mutation"),
            ("none", "two_point", "elite_heavy_mutation"),
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
                    algorithm_modes=("none", "two_point"),
                    nga_mutation_fraction=0.5,
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
            self.assertIn("Лимит NGA без улучшения", rows[0])
            self.assertIn("NGA использован", rows[0])

            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Режимы алгоритма", markdown_text)
            self.assertIn("Без NGA", markdown_text)
            self.assertIn("NGA-1: двухточечный кроссовер и двухточечная мутация", markdown_text)

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
                algorithm_modes=("none", "two_point"),
                generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                seed=321,
                output_dir=output_dir,
            )
            run_benchmark(settings)
            run_benchmark(replace(settings, resume=True))

            with (output_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
