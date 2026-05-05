from __future__ import annotations

import unittest

from genetic_knapsack_solver.models import GeneticAlgorithmConfig
from genetic_knapsack_solver.rust_core import rust_core_available, solve_with_rust_core


@unittest.skipUnless(rust_core_available(), "Rust core library is not built")
class RustCoreTests(unittest.TestCase):
    def test_rust_core_solves_with_python_config(self) -> None:
        config = GeneticAlgorithmConfig(
            population_size=6,
            generations=5,
            stagnation=2,
            crossover_rate=0.0,
            mutation_rate=0.0,
            tournament_size=2,
        )

        result = solve_with_rust_core(
            prices=[4, 8],
            target_sum=3,
            config=config,
            seed=7,
        )

        self.assertFalse(result.exact_match)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertEqual(result.generations_used, 2)
        self.assertEqual(result.best_sum, 4)
        self.assertEqual(result.difference, 1)

    def test_rust_core_supports_two_stage_restart(self) -> None:
        config = GeneticAlgorithmConfig(
            solver_mode="two_stage_restart",
            population_size=6,
            generations=3,
            stagnation=1,
            crossover_rate=0.0,
            mutation_rate=0.0,
            tournament_size=2,
            restart_mutation_fraction=0.4,
            stage2_crossover_type="two_point",
            stage2_mutation_type="two_point",
            stage2_offspring_mode="four_children_select_two",
        )

        result = solve_with_rust_core(
            prices=[4, 8],
            target_sum=3,
            config=config,
            seed=7,
        )

        self.assertFalse(result.exact_match)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertTrue(result.stage2_used)
        self.assertGreaterEqual(result.stage1_run_count, 2)
        self.assertGreaterEqual(result.stage2_run_count, 2)


if __name__ == "__main__":
    unittest.main()
