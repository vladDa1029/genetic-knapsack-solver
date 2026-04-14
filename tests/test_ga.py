from __future__ import annotations

import unittest
from random import Random

from genetic_knapsack_solver.ga import (
    crossover,
    crossover_two_points,
    fitness,
    mutate,
    mutate_many_bits,
    mutate_two_points,
    solve_with_genetic_algorithm,
)
from genetic_knapsack_solver.models import GeneticAlgorithmConfig


class GeneticAlgorithmTests(unittest.TestCase):
    def test_basic_operators(self) -> None:
        self.assertEqual(fitness([10, 20, 30], 30, [1, 0, 1]), 10)

        child_a, child_b = crossover([1, 1, 1, 1], [0, 0, 0, 0], Random(2))
        child_c, child_d = crossover_two_points([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], Random(5))
        self.assertEqual(len(child_a), 4)
        self.assertEqual(len(child_b), 4)
        self.assertEqual(len(child_c), 5)
        self.assertEqual(len(child_d), 5)
        self.assertTrue(all(bit in (0, 1) for bit in child_a + child_b))
        self.assertTrue(all(bit in (0, 1) for bit in child_c + child_d))

        mutated = mutate([0, 0, 0, 0], 1.0, Random(3))
        mutated_two_points = mutate_two_points([0, 0, 0, 0], 1.0, Random(3))
        mutated_many_bits = mutate_many_bits([0] * 10, 0.4, Random(11))
        self.assertEqual(sum(mutated), 1)
        self.assertEqual(sum(mutated_two_points), 2)
        self.assertEqual(sum(mutated_many_bits), 4)

    def test_solver_stops_on_stagnation_without_nga(self) -> None:
        config = GeneticAlgorithmConfig(
            population_size=6,
            generations=5,
            stagnation=2,
            crossover_rate=0.0,
            mutation_rate=0.0,
            tournament_size=2,
        )
        result = solve_with_genetic_algorithm(
            prices=[4, 8],
            target_sum=3,
            config=config,
            rng=Random(7),
        )

        self.assertFalse(result.exact_match)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertEqual(result.generations_used, 2)
        self.assertFalse(result.nga_used)
        self.assertIsNone(result.nga_trigger_generation)

    def test_solver_triggers_two_point_nga_once(self) -> None:
        config = GeneticAlgorithmConfig(
            population_size=6,
            generations=5,
            stagnation=2,
            repeat_limit=1,
            crossover_rate=0.0,
            mutation_rate=0.0,
            tournament_size=2,
            nga_mode="two_point",
        )
        result = solve_with_genetic_algorithm(
            prices=[2, 4],
            target_sum=1,
            config=config,
            rng=Random(7),
        )

        self.assertFalse(result.exact_match)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertEqual(result.generations_used, 3)
        self.assertTrue(result.nga_used)
        self.assertEqual(result.nga_trigger_generation, 1)

    def test_solver_triggers_elite_heavy_mutation_nga_once(self) -> None:
        config = GeneticAlgorithmConfig(
            population_size=6,
            generations=5,
            stagnation=2,
            repeat_limit=1,
            crossover_rate=0.0,
            mutation_rate=0.0,
            tournament_size=2,
            nga_mode="elite_heavy_mutation",
            nga_mutation_fraction=0.5,
        )
        result = solve_with_genetic_algorithm(
            prices=[2, 4],
            target_sum=1,
            config=config,
            rng=Random(7),
        )

        self.assertFalse(result.exact_match)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertEqual(result.generations_used, 3)
        self.assertTrue(result.nga_used)
        self.assertEqual(result.nga_trigger_generation, 1)


if __name__ == "__main__":
    unittest.main()
