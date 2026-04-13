from __future__ import annotations

import unittest
from random import Random

from genetic_knapsack_solver.ga import crossover, fitness, mutate, solve_with_genetic_algorithm
from genetic_knapsack_solver.models import GeneticAlgorithmConfig


class GeneticAlgorithmTests(unittest.TestCase):
    def test_basic_operators(self) -> None:
        self.assertEqual(fitness([10, 20, 30], 30, [1, 0, 1]), 10)

        child_a, child_b = crossover([1, 1, 1, 1], [0, 0, 0, 0], Random(2))
        self.assertEqual(len(child_a), 4)
        self.assertEqual(len(child_b), 4)
        self.assertTrue(all(bit in (0, 1) for bit in child_a + child_b))

        mutated = mutate([0, 0, 0, 0], 1.0, Random(3))
        self.assertEqual(sum(mutated), 1)

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


if __name__ == "__main__":
    unittest.main()
