from __future__ import annotations

import unittest
from random import Random

from genetic_knapsack_solver.generator import (
    GENERATION_MODE_RANDOM,
    GENERATION_MODE_SUPERINCREASING_DISGUISED,
    count_target_solutions,
    dot_product,
    generate_problem,
)


class ProblemGenerationTests(unittest.TestCase):
    def test_generate_random_problem_builds_unique_target(self) -> None:
        problem = generate_problem(8, Random(1234), generation_mode=GENERATION_MODE_RANDOM)

        self.assertEqual(problem.generation_mode, GENERATION_MODE_RANDOM)
        self.assertEqual(len(problem.items), 8)
        self.assertEqual(len(problem.prices), 8)
        self.assertEqual(len(problem.hidden_vector), 8)
        self.assertEqual(len(set(problem.prices)), 8)
        self.assertEqual(problem.target_sum, dot_product(problem.prices, problem.hidden_vector))
        self.assertEqual(count_target_solutions(problem.prices, problem.target_sum), 1)

    def test_generate_disguised_superincreasing_problem_builds_unique_target(self) -> None:
        problem = generate_problem(
            8,
            Random(4321),
            generation_mode=GENERATION_MODE_SUPERINCREASING_DISGUISED,
        )

        self.assertEqual(problem.generation_mode, GENERATION_MODE_SUPERINCREASING_DISGUISED)
        self.assertEqual(len(problem.items), 8)
        self.assertEqual(len(problem.prices), 8)
        self.assertEqual(len(problem.hidden_vector), 8)
        self.assertEqual(problem.target_sum, dot_product(problem.prices, problem.hidden_vector))
        self.assertEqual(count_target_solutions(problem.prices, problem.target_sum), 1)


if __name__ == "__main__":
    unittest.main()
