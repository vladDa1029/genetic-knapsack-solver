from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from genetic_knapsack_solver.benchmark import (
    BenchmarkSettings,
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

    def test_benchmark_writes_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = run_benchmark(
                BenchmarkSettings(
                    item_counts=(6, 7),
                    task_count=2,
                    generations=12,
                    mutation_rate=0.4,
                    crossover_rate=0.6,
                    tournament_size=2,
                    population_stop_pairs=((10, 3),),
                    generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
                    seed=123,
                    output_root=Path(temp_dir),
                )
            )

            csv_path = output_dir / "results.csv"
            markdown_path = output_dir / "summary.md"

            self.assertTrue(csv_path.exists())
            self.assertTrue(markdown_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 4)
            self.assertIn("Режим генерации", rows[0])
            self.assertIn("Количество предметов", rows[0])
            self.assertIn("Размер популяции", rows[0])
            self.assertIn("Точное совпадение", rows[0])
            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Размеры задачи n", markdown_text)
            self.assertIn("Режим генерации", markdown_text)
            self.assertIn("Замаскированная суперпоследовательность", markdown_text)


if __name__ == "__main__":
    unittest.main()
