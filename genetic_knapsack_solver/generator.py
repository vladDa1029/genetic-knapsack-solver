from __future__ import annotations

from math import gcd
from random import Random

from genetic_knapsack_solver.models import ProblemInstance

GENERATION_MODE_RANDOM = "random"
GENERATION_MODE_SUPERINCREASING_DISGUISED = "superincreasing_disguised"

GENERATION_MODE_LABELS = {
    GENERATION_MODE_RANDOM: "Случайные цены",
    GENERATION_MODE_SUPERINCREASING_DISGUISED: "Замаскированная суперпоследовательность",
}


def dot_product(values: list[int], vector: list[int]) -> int:
    return sum(value * bit for value, bit in zip(values, vector))


def build_item_names(item_count: int) -> list[str]:
    return [f"Предмет-{index + 1}" for index in range(item_count)]


def build_subset_sum_counts(values: list[int]) -> dict[int, int]:
    sums = [0]
    for value in values:
        sums += [subtotal + value for subtotal in sums]

    counts: dict[int, int] = {}
    for subtotal in sums:
        counts[subtotal] = min(2, counts.get(subtotal, 0) + 1)
    return counts


def count_target_solutions(prices: list[int], target_sum: int) -> int:
    midpoint = len(prices) // 2
    left_counts = build_subset_sum_counts(prices[:midpoint])
    right_counts = build_subset_sum_counts(prices[midpoint:])

    solutions = 0
    for left_sum, left_ways in left_counts.items():
        right_ways = right_counts.get(target_sum - left_sum, 0)
        if right_ways == 0:
            continue
        solutions = min(2, solutions + left_ways * right_ways)
        if solutions >= 2:
            return 2
    return solutions


def _build_hidden_vector(item_count: int, rng: Random) -> list[int]:
    hidden_vector = [rng.randint(0, 1) for _ in range(item_count)]
    if sum(hidden_vector) == 0:
        hidden_vector[rng.randrange(item_count)] = 1
    return hidden_vector


def build_superincreasing_prices(item_count: int, rng: Random) -> list[int]:
    prices: list[int] = []
    current = rng.randint(2, 9)
    prices.append(current)

    for _ in range(1, item_count):
        current = 2 * current + 1 + rng.randint(1, 3)
        prices.append(current)

    return prices


def disguise_superincreasing_prices(prices: list[int], rng: Random) -> list[int]:
    price_sum = sum(prices)
    modulus = price_sum + rng.randint(price_sum // 2 + 1, price_sum * 2)

    multiplier = rng.randint(2, modulus - 1)
    while gcd(multiplier, modulus) != 1:
        multiplier = rng.randint(2, modulus - 1)

    order = list(range(len(prices)))
    rng.shuffle(order)
    return [(multiplier * prices[index]) % modulus for index in order]


def _build_prices(item_count: int, rng: Random, generation_mode: str) -> list[int]:
    if generation_mode == GENERATION_MODE_RANDOM:
        upper_bound = max(20, item_count * 1000 + 10)
        return rng.sample(range(10, upper_bound), item_count)

    if generation_mode == GENERATION_MODE_SUPERINCREASING_DISGUISED:
        return disguise_superincreasing_prices(
            build_superincreasing_prices(item_count, rng),
            rng,
        )

    raise ValueError(f"unknown generation_mode: {generation_mode}")


def generate_problem(
    item_count: int,
    rng: Random,
    generation_mode: str = GENERATION_MODE_RANDOM,
) -> ProblemInstance:
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if generation_mode not in GENERATION_MODE_LABELS:
        raise ValueError(f"unknown generation_mode: {generation_mode}")

    items = build_item_names(item_count)

    while True:
        prices = _build_prices(item_count, rng, generation_mode)
        hidden_vector = _build_hidden_vector(item_count, rng)
        target_sum = dot_product(prices, hidden_vector)

        if target_sum == 0:
            continue

        if count_target_solutions(prices, target_sum) == 1:
            return ProblemInstance(
                items=items,
                prices=prices,
                hidden_vector=hidden_vector,
                target_sum=target_sum,
                generation_mode=generation_mode,
            )
