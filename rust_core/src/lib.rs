use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Deserialize)]
struct SolveRequest {
    prices: Vec<i64>,
    target_sum: i64,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    config: GeneticAlgorithmConfig,
}

// Конфигурация должна зеркалить Python dataclass GeneticAlgorithmConfig.
// JSON приходит через FFI, поэтому значения по умолчанию и validate() держат
// границу Rust-ядра устойчивой к неполным или некорректным payload.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct GeneticAlgorithmConfig {
    solver_mode: String,
    population_size: usize,
    generations: usize,
    stagnation: usize,
    repeat_limit: Option<usize>,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    nga_mode: String,
    nga_mutation_fraction: f64,
    nga_trigger_points: Vec<usize>,
    nga_mutate_points: Vec<usize>,
    restart_max_count: Option<usize>,
    rescue_min_mutated_bits_ratio: f64,
    restart_population_mode: String,
    restart_mutation_fraction: f64,
    stage2_crossover_type: String,
    stage2_mutation_type: String,
    stage2_offspring_mode: String,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            solver_mode: "classic".to_string(),
            population_size: 100,
            generations: 200,
            stagnation: 30,
            repeat_limit: None,
            crossover_rate: 0.8,
            mutation_rate: 0.05,
            tournament_size: 3,
            nga_mode: "none".to_string(),
            nga_mutation_fraction: 0.4,
            nga_trigger_points: Vec::new(),
            nga_mutate_points: vec![40],
            restart_max_count: None,
            rescue_min_mutated_bits_ratio: 0.4,
            restart_population_mode: "elite_from_last_population".to_string(),
            restart_mutation_fraction: 0.4,
            stage2_crossover_type: "two_point".to_string(),
            stage2_mutation_type: "two_point".to_string(),
            stage2_offspring_mode: "two_children".to_string(),
        }
    }
}

impl GeneticAlgorithmConfig {
    fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.solver_mode.as_str(),
            "classic" | "restart_rescue" | "two_stage_restart"
        ) {
            return Err("solver_mode must be one of: classic, restart_rescue, two_stage_restart".to_string());
        }
        if self.population_size < 2 {
            return Err("population_size must be at least 2".to_string());
        }
        if self.generations < 1 {
            return Err("generations must be at least 1".to_string());
        }
        if self.stagnation < 1 {
            return Err("stagnation must be at least 1".to_string());
        }
        if let Some(repeat_limit) = self.repeat_limit {
            if repeat_limit < 1 {
                return Err("repeat_limit must be at least 1".to_string());
            }
        }
        if let Some(restart_max_count) = self.restart_max_count {
            if restart_max_count == usize::MAX {
                return Err("restart_max_count is too large".to_string());
            }
        }
        if self.tournament_size < 2 {
            return Err("tournament_size must be at least 2".to_string());
        }
        if !matches!(
            self.nga_mode.as_str(),
            "none" | "two_point" | "elite_heavy_mutation" | "staged_hypermutation"
        ) {
            return Err(
                "nga_mode must be one of: none, two_point, elite_heavy_mutation, staged_hypermutation"
                    .to_string(),
            );
        }
        if self.restart_population_mode != "elite_from_last_population" {
            return Err("restart_population_mode must be one of: elite_from_last_population".to_string());
        }
        if !matches!(self.stage2_crossover_type.as_str(), "one_point" | "two_point") {
            return Err("stage2_crossover_type must be one of: one_point, two_point".to_string());
        }
        if !matches!(
            self.stage2_mutation_type.as_str(),
            "one_point" | "two_point" | "reverse"
        ) {
            return Err("stage2_mutation_type must be one of: one_point, two_point, reverse".to_string());
        }
        if !matches!(
            self.stage2_offspring_mode.as_str(),
            "two_children" | "four_children_select_two"
        ) {
            return Err(
                "stage2_offspring_mode must be one of: two_children, four_children_select_two"
                    .to_string(),
            );
        }
        for (name, value) in [
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("nga_mutation_fraction", self.nga_mutation_fraction),
            (
                "rescue_min_mutated_bits_ratio",
                self.rescue_min_mutated_bits_ratio,
            ),
            ("restart_mutation_fraction", self.restart_mutation_fraction),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be between 0.0 and 1.0"));
            }
        }

        match self.solver_mode.as_str() {
            "classic" => {
                if matches!(self.nga_mode.as_str(), "two_point" | "elite_heavy_mutation")
                    && self.repeat_limit.is_none()
                {
                    return Err(
                        "repeat_limit must be provided when nga_mode is two_point or elite_heavy_mutation"
                            .to_string(),
                    );
                }
                if self.nga_mode == "staged_hypermutation" {
                    if self.repeat_limit.is_some() {
                        return Err("repeat_limit is not used in staged_hypermutation mode".to_string());
                    }
                    if self.nga_trigger_points.is_empty() {
                        return Err("nga_trigger_points must be provided for staged_hypermutation mode".to_string());
                    }
                    let mut previous_point = 0;
                    for point in &self.nga_trigger_points {
                        if *point < 1 {
                            return Err("nga_trigger_points must contain only positive integers".to_string());
                        }
                        if *point >= self.stagnation {
                            return Err("each nga_trigger_point must be less than stagnation".to_string());
                        }
                        if *point <= previous_point {
                            return Err("nga_trigger_points must be strictly increasing".to_string());
                        }
                        previous_point = *point;
                    }
                    if self.nga_mutate_points.is_empty() {
                        return Err("nga_mutate_points must be provided for staged_hypermutation mode".to_string());
                    }
                    if self.nga_mutate_points.len() != 1
                        && self.nga_mutate_points.len() != self.nga_trigger_points.len()
                    {
                        return Err(
                            "nga_mutate_points must contain either one value or match nga_trigger_points length"
                                .to_string(),
                        );
                    }
                    if self.nga_mutate_points.iter().any(|value| *value > 100) {
                        return Err("nga_mutate_points must be between 0 and 100".to_string());
                    }
                }
            }
            "restart_rescue" => {
                if self.nga_mode != "none" {
                    return Err("nga_mode is only available in classic solver_mode".to_string());
                }
                if self.repeat_limit.is_some() {
                    return Err("repeat_limit is only available in classic solver_mode".to_string());
                }
            }
            "two_stage_restart" => {
                if self.nga_mode != "none" {
                    return Err("nga_mode is not used in two_stage_restart solver_mode".to_string());
                }
                if self.repeat_limit.is_some() {
                    return Err("repeat_limit is not used in two_stage_restart solver_mode".to_string());
                }
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in two_stage_restart solver_mode".to_string());
                }
            }
            _ => {}
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct SingleRunResult {
    population: Vec<Vec<u8>>,
    best_vector: Vec<u8>,
    best_sum: i64,
    difference: i64,
    best_fitness: i64,
    generations_used: usize,
    exact_match: bool,
    stop_reason: String,
    nga_used: bool,
    nga_trigger_generation: Option<usize>,
    nga_trigger_generations: Vec<usize>,
}

// Результат сериализуется обратно в Python без дополнительного преобразования.
#[derive(Clone, Debug, Serialize)]
struct GeneticAlgorithmResult {
    best_vector: Vec<u8>,
    best_sum: i64,
    fitness: i64,
    difference: i64,
    generations_used: usize,
    exact_match: bool,
    stop_reason: String,
    nga_used: bool,
    nga_trigger_generation: Option<usize>,
    nga_trigger_generations: Vec<usize>,
    restart_count: usize,
    rescue_used: bool,
    stage1_run_count: usize,
    stage2_run_count: usize,
    stage2_used: bool,
    stage1_best_differences: Vec<i64>,
    stage2_best_differences: Vec<i64>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Clone, Copy)]
enum CrossoverKind {
    OnePoint,
    TwoPoint,
}

#[derive(Clone, Copy)]
enum MutationKind {
    OnePoint,
    TwoPoint,
    Reverse,
}

#[derive(Clone, Copy)]
enum OffspringMode {
    TwoChildren,
    FourChildrenSelectTwo,
}

#[derive(Clone, Debug)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    // SplitMix64: быстрый детерминированный PRNG для воспроизводимых Rust-прогонов.
    fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random_f64()
    }

    fn randrange(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        let upper_u64 = upper as u64;
        let zone = u64::MAX - (u64::MAX % upper_u64);
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % upper_u64) as usize;
            }
        }
    }

    fn randint_inclusive(&mut self, low: usize, high: usize) -> usize {
        low + self.randrange(high - low + 1)
    }

    fn sample_range(&mut self, upper: usize, count: usize) -> Vec<usize> {
        let mut values: Vec<usize> = (0..upper).collect();
        let take = count.min(upper);
        for index in 0..take {
            let swap_index = index + self.randrange(upper - index);
            values.swap(index, swap_index);
        }
        values.truncate(take);
        values
    }
}

fn evaluate_vector(prices: &[i64], target_sum: i64, vector: &[u8]) -> (i64, i64) {
    let current_sum = prices
        .iter()
        .zip(vector.iter())
        .map(|(price, bit)| price * i64::from(*bit))
        .sum::<i64>();
    ((target_sum - current_sum).abs(), current_sum)
}

fn evaluate_population(
    population: &[Vec<u8>],
    prices: &[i64],
    target_sum: i64,
) -> Vec<(i64, i64)> {
    population
        .iter()
        .map(|vector| evaluate_vector(prices, target_sum, vector))
        .collect()
}

fn rank_population_indices(evaluations: &[(i64, i64)]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..evaluations.len()).collect();
    indices.sort_by_key(|index| (evaluations[*index].0, *index));
    indices
}

fn best_from_population(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
) -> (Vec<u8>, i64, i64) {
    let best_index = (0..population.len())
        .min_by_key(|index| (evaluations[*index].0, *index))
        .unwrap_or(0);
    let (best_difference, best_sum) = evaluations[best_index];
    (population[best_index].clone(), best_sum, best_difference)
}

fn build_initial_population(
    gene_count: usize,
    population_size: usize,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let mut population = Vec::with_capacity(population_size);
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut repeated_attempts = 0usize;

    while population.len() < population_size {
        let bias = rng.uniform(0.35, 0.65);
        let candidate: Vec<u8> = (0..gene_count)
            .map(|_| u8::from(rng.random_f64() < bias))
            .collect();

        if seen.contains(&candidate) && repeated_attempts < population_size * 3 {
            repeated_attempts += 1;
            continue;
        }

        seen.insert(candidate.clone());
        population.push(candidate);
    }

    population
}

fn crossover(first: &[u8], second: &[u8], rng: &mut Rng64) -> (Vec<u8>, Vec<u8>) {
    if first.len() < 2 {
        return (first.to_vec(), second.to_vec());
    }
    let point = rng.randint_inclusive(1, first.len() - 1);
    let mut child_a = Vec::with_capacity(first.len());
    child_a.extend_from_slice(&first[..point]);
    child_a.extend_from_slice(&second[point..]);

    let mut child_b = Vec::with_capacity(first.len());
    child_b.extend_from_slice(&second[..point]);
    child_b.extend_from_slice(&first[point..]);
    (child_a, child_b)
}

fn crossover_two_points(first: &[u8], second: &[u8], rng: &mut Rng64) -> (Vec<u8>, Vec<u8>) {
    if first.len() < 3 {
        return crossover(first, second, rng);
    }
    let mut points = rng.sample_range(first.len() - 1, 2);
    points.iter_mut().for_each(|point| *point += 1);
    points.sort_unstable();
    let left = points[0];
    let right = points[1];

    let mut child_a = Vec::with_capacity(first.len());
    child_a.extend_from_slice(&first[..left]);
    child_a.extend_from_slice(&second[left..right]);
    child_a.extend_from_slice(&first[right..]);

    let mut child_b = Vec::with_capacity(first.len());
    child_b.extend_from_slice(&second[..left]);
    child_b.extend_from_slice(&first[left..right]);
    child_b.extend_from_slice(&second[right..]);
    (child_a, child_b)
}

fn mutate(vector: &[u8], mutation_rate: f64, rng: &mut Rng64) -> Vec<u8> {
    let mut mutated = vector.to_vec();
    if !mutated.is_empty() && rng.random_f64() < mutation_rate {
        let index = rng.randrange(mutated.len());
        mutated[index] = 1 - mutated[index];
    }
    mutated
}

fn mutate_two_points(vector: &[u8], mutation_rate: f64, rng: &mut Rng64) -> Vec<u8> {
    let mut mutated = vector.to_vec();
    if mutated.is_empty() || rng.random_f64() >= mutation_rate {
        return mutated;
    }
    if mutated.len() == 1 {
        mutated[0] = 1 - mutated[0];
        return mutated;
    }
    for index in rng.sample_range(mutated.len(), 2) {
        mutated[index] = 1 - mutated[index];
    }
    mutated
}

fn mutate_reverse(vector: &[u8], mutation_rate: f64, rng: &mut Rng64) -> Vec<u8> {
    if vector.is_empty() || rng.random_f64() >= mutation_rate {
        return vector.to_vec();
    }
    vector.iter().rev().copied().collect()
}

fn mutate_many_bits(vector: &[u8], mutation_fraction: f64, rng: &mut Rng64) -> Vec<u8> {
    let mut mutated = vector.to_vec();
    if mutated.is_empty() || mutation_fraction <= 0.0 {
        return mutated;
    }
    let flip_count = mutated
        .len()
        .min(usize::max(1, ((mutated.len() as f64) * mutation_fraction).ceil() as usize));
    for index in rng.sample_range(mutated.len(), flip_count) {
        mutated[index] = 1 - mutated[index];
    }
    mutated
}

fn rescue_heavy_mutation(vector: &[u8], mutation_ratio: f64, rng: &mut Rng64) -> Vec<u8> {
    mutate_many_bits(vector, mutation_ratio, rng)
}

fn pick_parent(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    tournament_size: usize,
    rng: &mut Rng64,
) -> Vec<u8> {
    let contestant_count = tournament_size.min(population.len());
    let mut best_index = rng.randrange(population.len());
    for _ in 1..contestant_count {
        let index = rng.randrange(population.len());
        if evaluations[index].0.cmp(&evaluations[best_index].0) == Ordering::Less {
            best_index = index;
        }
    }
    population[best_index].clone()
}

fn apply_crossover(
    kind: CrossoverKind,
    first: &[u8],
    second: &[u8],
    rng: &mut Rng64,
) -> (Vec<u8>, Vec<u8>) {
    match kind {
        CrossoverKind::OnePoint => crossover(first, second, rng),
        CrossoverKind::TwoPoint => crossover_two_points(first, second, rng),
    }
}

fn apply_mutation(kind: MutationKind, vector: &[u8], mutation_rate: f64, rng: &mut Rng64) -> Vec<u8> {
    match kind {
        MutationKind::OnePoint => mutate(vector, mutation_rate, rng),
        MutationKind::TwoPoint => mutate_two_points(vector, mutation_rate, rng),
        MutationKind::Reverse => mutate_reverse(vector, mutation_rate, rng),
    }
}

fn produce_offspring(
    parent_a: &[u8],
    parent_b: &[u8],
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
    crossover_kind: CrossoverKind,
    mutation_kind: MutationKind,
    offspring_mode: OffspringMode,
) -> [Vec<u8>; 2] {
    match offspring_mode {
        OffspringMode::FourChildrenSelectTwo => {
            // Для режима four_children_select_two строим четыре варианта детей,
            // мутируем их и оставляем две лучшие особи по близости к target_sum.
            let raw_children = if rng.random_f64() < config.crossover_rate {
                let (child_a, child_b) = apply_crossover(crossover_kind, parent_a, parent_b, rng);
                let (child_c, child_d) = apply_crossover(crossover_kind, parent_a, parent_b, rng);
                [child_a, child_b, child_c, child_d]
            } else {
                [
                    parent_a.to_vec(),
                    parent_b.to_vec(),
                    parent_a.to_vec(),
                    parent_b.to_vec(),
                ]
            };

            let mut candidates: Vec<(Vec<u8>, i64, i64, usize)> = raw_children
                .into_iter()
                .enumerate()
                .map(|(index, child)| {
                    let mutated = apply_mutation(mutation_kind, &child, config.mutation_rate, rng);
                    let (difference, current_sum) = evaluate_vector(prices, target_sum, &mutated);
                    (mutated, difference, current_sum, index)
                })
                .collect();
            candidates.sort_by_key(|(_, difference, current_sum, index)| (*difference, *current_sum, *index));

            [
                candidates.remove(0).0,
                candidates.remove(0).0,
            ]
        }
        OffspringMode::TwoChildren => {
            let (child_a, child_b) = if rng.random_f64() < config.crossover_rate {
                apply_crossover(crossover_kind, parent_a, parent_b, rng)
            } else {
                (parent_a.to_vec(), parent_b.to_vec())
            };
            [
                apply_mutation(mutation_kind, &child_a, config.mutation_rate, rng),
                apply_mutation(mutation_kind, &child_b, config.mutation_rate, rng),
            ]
        }
    }
}

fn build_next_population(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
    crossover_kind: CrossoverKind,
    mutation_kind: MutationKind,
    offspring_mode: OffspringMode,
) -> Vec<Vec<u8>> {
    let ranked_indices = rank_population_indices(evaluations);
    let mut next_generation = Vec::with_capacity(population.len());
    // Элитизм: лучшая особь всегда переносится в следующее поколение без изменений.
    next_generation.push(population[ranked_indices[0]].clone());

    while next_generation.len() < population.len() {
        let parent_a = pick_parent(population, evaluations, config.tournament_size, rng);
        let parent_b = pick_parent(population, evaluations, config.tournament_size, rng);
        let [child_a, child_b] = produce_offspring(
            &parent_a,
            &parent_b,
            prices,
            target_sum,
            config,
            rng,
            crossover_kind,
            mutation_kind,
            offspring_mode,
        );
        next_generation.push(child_a);
        if next_generation.len() < population.len() {
            next_generation.push(child_b);
        }
    }

    next_generation
}

fn apply_nga_two_point(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    build_next_population(
        population,
        evaluations,
        prices,
        target_sum,
        config,
        rng,
        CrossoverKind::TwoPoint,
        MutationKind::TwoPoint,
        OffspringMode::TwoChildren,
    )
}

fn apply_nga_elite_heavy_mutation(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let ranked_indices = rank_population_indices(evaluations);
    let best_index = ranked_indices[0];
    let mut next_generation = Vec::with_capacity(population.len());
    next_generation.push(population[best_index].clone());

    for (index, vector) in population.iter().enumerate() {
        if index == best_index {
            continue;
        }
        next_generation.push(mutate_many_bits(vector, config.nga_mutation_fraction, rng));
    }
    next_generation
}

fn apply_nga_intervention(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    match config.nga_mode.as_str() {
        "two_point" => apply_nga_two_point(population, evaluations, prices, target_sum, config, rng),
        "elite_heavy_mutation" => apply_nga_elite_heavy_mutation(population, evaluations, config, rng),
        _ => population.to_vec(),
    }
}

fn resolve_staged_mutation_fraction(
    config: &GeneticAlgorithmConfig,
    stagnation_counter: usize,
) -> f64 {
    let trigger_index = config
        .nga_trigger_points
        .iter()
        .position(|point| *point == stagnation_counter)
        .unwrap_or(0);
    let mutate_percent = if config.nga_mutate_points.len() == 1 {
        config.nga_mutate_points[0]
    } else {
        config.nga_mutate_points[trigger_index]
    };
    mutate_percent as f64 / 100.0
}

fn apply_staged_hypermutation(
    population: &[Vec<u8>],
    evaluations: &[(i64, i64)],
    mutation_fraction: f64,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let ranked_indices = rank_population_indices(evaluations);
    let best_index = ranked_indices[0];
    let mut next_generation = Vec::with_capacity(population.len());
    next_generation.push(population[best_index].clone());

    for (index, vector) in population.iter().enumerate() {
        if index == best_index {
            continue;
        }
        next_generation.push(mutate_many_bits(vector, mutation_fraction, rng));
    }
    next_generation
}

#[allow(clippy::too_many_arguments)]
fn single_run(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
    initial_population: Option<Vec<Vec<u8>>>,
    stop_on_stagnation: bool,
    allow_nga: bool,
    crossover_kind: CrossoverKind,
    mutation_kind: MutationKind,
    offspring_mode: OffspringMode,
) -> SingleRunResult {
    let mut current_population = initial_population
        .unwrap_or_else(|| build_initial_population(prices.len(), config.population_size, rng));
    let mut evaluations = evaluate_population(&current_population, prices, target_sum);
    let (mut best_vector, mut best_sum, mut best_difference) =
        best_from_population(&current_population, &evaluations);
    let mut best_fitness = best_difference;
    let mut nga_used = false;
    let mut nga_trigger_generation = None;
    let mut nga_trigger_generations = Vec::new();

    if best_difference == 0 {
        return SingleRunResult {
            population: current_population,
            best_vector,
            best_sum,
            difference: best_difference,
            best_fitness,
            generations_used: 0,
            exact_match: true,
            stop_reason: "exact_match".to_string(),
            nga_used: false,
            nga_trigger_generation: None,
            nga_trigger_generations,
        };
    }

    let mut no_improvement_streak = 0usize;

    for generation in 1..=config.generations {
        current_population = build_next_population(
            &current_population,
            &evaluations,
            prices,
            target_sum,
            config,
            rng,
            crossover_kind,
            mutation_kind,
            offspring_mode,
        );
        evaluations = evaluate_population(&current_population, prices, target_sum);
        let (current_best_vector, current_best_sum, current_best_difference) =
            best_from_population(&current_population, &evaluations);
        let current_best_fitness = current_best_difference;

        if current_best_difference < best_difference {
            best_vector = current_best_vector;
            best_sum = current_best_sum;
            best_difference = current_best_difference;
            best_fitness = current_best_fitness;
            no_improvement_streak = 0;
        } else {
            no_improvement_streak += 1;
        }

        if best_difference == 0 {
            return SingleRunResult {
                population: current_population,
                best_vector,
                best_sum,
                difference: best_difference,
                best_fitness,
                generations_used: generation,
                exact_match: true,
                stop_reason: "exact_match".to_string(),
                nga_used,
                nga_trigger_generation,
                nga_trigger_generations,
            };
        }

        if allow_nga && config.nga_mode != "none" {
            let mut nga_population = None;

            if config.nga_mode == "staged_hypermutation"
                && config.nga_trigger_points.contains(&no_improvement_streak)
            {
                let mutation_fraction =
                    resolve_staged_mutation_fraction(config, no_improvement_streak);
                nga_population = Some(apply_staged_hypermutation(
                    &current_population,
                    &evaluations,
                    mutation_fraction,
                    rng,
                ));
            } else if matches!(config.nga_mode.as_str(), "two_point" | "elite_heavy_mutation")
                && !nga_used
                && config
                    .repeat_limit
                    .is_some_and(|repeat_limit| no_improvement_streak >= repeat_limit)
            {
                nga_population = Some(apply_nga_intervention(
                    &current_population,
                    &evaluations,
                    prices,
                    target_sum,
                    config,
                    rng,
                ));
            }

            if let Some(population) = nga_population {
                current_population = population;
                evaluations = evaluate_population(&current_population, prices, target_sum);
                let (current_best_vector, current_best_sum, current_best_difference) =
                    best_from_population(&current_population, &evaluations);
                let current_best_fitness = current_best_difference;
                nga_used = true;
                if nga_trigger_generation.is_none() {
                    nga_trigger_generation = Some(generation);
                }
                nga_trigger_generations.push(generation);

                if current_best_difference < best_difference {
                    best_vector = current_best_vector;
                    best_sum = current_best_sum;
                    best_difference = current_best_difference;
                    best_fitness = current_best_fitness;
                    no_improvement_streak = 0;
                }

                if best_difference == 0 {
                    return SingleRunResult {
                        population: current_population,
                        best_vector,
                        best_sum,
                        difference: best_difference,
                        best_fitness,
                        generations_used: generation,
                        exact_match: true,
                        stop_reason: "exact_match".to_string(),
                        nga_used,
                        nga_trigger_generation,
                        nga_trigger_generations,
                    };
                }
            }
        }

        if stop_on_stagnation && no_improvement_streak >= config.stagnation {
            return SingleRunResult {
                population: current_population,
                best_vector,
                best_sum,
                difference: best_difference,
                best_fitness,
                generations_used: generation,
                exact_match: false,
                stop_reason: "stagnation_limit".to_string(),
                nga_used,
                nga_trigger_generation,
                nga_trigger_generations,
            };
        }
    }

    SingleRunResult {
        population: current_population,
        best_vector,
        best_sum,
        difference: best_difference,
        best_fitness,
        generations_used: config.generations,
        exact_match: best_difference == 0,
        stop_reason: if best_difference == 0 {
            "exact_match".to_string()
        } else {
            "generation_limit".to_string()
        },
        nga_used,
        nga_trigger_generation,
        nga_trigger_generations,
    }
}

fn build_rescue_population(
    base_population: &[Vec<u8>],
    global_best_vector: &[u8],
    mutation_ratio: f64,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let mut rescue_population = Vec::with_capacity(base_population.len().max(1));
    rescue_population.push(global_best_vector.to_vec());
    for vector in base_population.iter().skip(1) {
        rescue_population.push(rescue_heavy_mutation(vector, mutation_ratio, rng));
    }
    rescue_population
}

fn build_restart_population(
    base_population: &[Vec<u8>],
    elite_vector: &[u8],
    population_mode: &str,
    mutation_fraction: f64,
    rng: &mut Rng64,
) -> Result<Vec<Vec<u8>>, String> {
    if population_mode != "elite_from_last_population" {
        return Err(format!("unsupported restart_population_mode: {population_mode}"));
    }
    if base_population.is_empty() {
        return Ok(vec![elite_vector.to_vec()]);
    }

    let elite_index = base_population
        .iter()
        .position(|vector| vector == elite_vector)
        .unwrap_or(0);
    let mut restart_population = Vec::with_capacity(base_population.len());
    restart_population.push(elite_vector.to_vec());

    for (index, vector) in base_population.iter().enumerate() {
        if index == elite_index {
            continue;
        }
        restart_population.push(mutate_many_bits(vector, mutation_fraction, rng));
    }

    Ok(restart_population)
}

fn choose_better_run(first: &SingleRunResult, second: &SingleRunResult) -> SingleRunResult {
    if second.difference < first.difference {
        return second.clone();
    }
    if second.difference > first.difference {
        return first.clone();
    }
    if second.best_fitness < first.best_fitness {
        return second.clone();
    }
    first.clone()
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    run: SingleRunResult,
    generations_used: usize,
    stop_reason: &str,
    restart_count: usize,
    rescue_used: bool,
    stage1_run_count: usize,
    stage2_run_count: usize,
    stage2_used: bool,
    stage1_best_differences: Vec<i64>,
    stage2_best_differences: Vec<i64>,
) -> GeneticAlgorithmResult {
    GeneticAlgorithmResult {
        best_vector: run.best_vector,
        best_sum: run.best_sum,
        fitness: run.best_fitness,
        difference: run.difference,
        generations_used,
        exact_match: run.exact_match,
        stop_reason: stop_reason.to_string(),
        nga_used: run.nga_used,
        nga_trigger_generation: run.nga_trigger_generation,
        nga_trigger_generations: run.nga_trigger_generations,
        restart_count,
        rescue_used,
        stage1_run_count,
        stage2_run_count,
        stage2_used,
        stage1_best_differences,
        stage2_best_differences,
    }
}

fn solve_restart_rescue(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> GeneticAlgorithmResult {
    let run_0 = single_run(
        prices,
        target_sum,
        config,
        rng,
        None,
        true,
        false,
        CrossoverKind::OnePoint,
        MutationKind::OnePoint,
        OffspringMode::TwoChildren,
    );

    if matches!(run_0.stop_reason.as_str(), "exact_match" | "generation_limit") {
        return build_result(
            run_0.clone(),
            run_0.generations_used,
            &run_0.stop_reason,
            0,
            false,
            0,
            0,
            false,
            Vec::new(),
            Vec::new(),
        );
    }

    let mut global_best = run_0.clone();
    let mut total_generations = run_0.generations_used;
    let mut restart_count = 0usize;

    loop {
        if config
            .restart_max_count
            .is_some_and(|restart_max_count| restart_count >= restart_max_count)
        {
            return build_result(
                global_best,
                total_generations,
                "restart_limit",
                restart_count,
                false,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            );
        }

        restart_count += 1;
        let run_i = single_run(
            prices,
            target_sum,
            config,
            rng,
            None,
            true,
            false,
            CrossoverKind::OnePoint,
            MutationKind::OnePoint,
            OffspringMode::TwoChildren,
        );
        total_generations += run_i.generations_used;

        if matches!(run_i.stop_reason.as_str(), "exact_match" | "generation_limit") {
            return build_result(
                run_i.clone(),
                total_generations,
                &run_i.stop_reason,
                restart_count,
                false,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            );
        }

        if run_i.difference < global_best.difference {
            global_best = run_i;
            continue;
        }

        let rescue_population = build_rescue_population(
            &run_i.population,
            &global_best.best_vector,
            config.rescue_min_mutated_bits_ratio,
            rng,
        );
        let rescue_run = single_run(
            prices,
            target_sum,
            config,
            rng,
            Some(rescue_population),
            true,
            false,
            CrossoverKind::OnePoint,
            MutationKind::OnePoint,
            OffspringMode::TwoChildren,
        );
        total_generations += rescue_run.generations_used;

        if rescue_run.stop_reason == "exact_match" {
            return build_result(
                rescue_run,
                total_generations,
                "exact_match",
                restart_count,
                true,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            );
        }
        if rescue_run.stop_reason == "stagnation_limit" {
            return build_result(
                global_best,
                total_generations,
                "rescue_stagnation",
                restart_count,
                true,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            );
        }
        if rescue_run.stop_reason == "generation_limit" {
            let selected_run = choose_better_run(&global_best, &rescue_run);
            return build_result(
                selected_run,
                total_generations,
                "rescue_generation_limit",
                restart_count,
                true,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            );
        }

        return build_result(
            rescue_run.clone(),
            total_generations,
            &rescue_run.stop_reason,
            restart_count,
            true,
            0,
            0,
            false,
            Vec::new(),
            Vec::new(),
        );
    }
}

fn resolve_crossover_kind(operator_type: &str) -> CrossoverKind {
    if operator_type == "two_point" {
        CrossoverKind::TwoPoint
    } else {
        CrossoverKind::OnePoint
    }
}

fn resolve_mutation_kind(operator_type: &str) -> MutationKind {
    match operator_type {
        "two_point" => MutationKind::TwoPoint,
        "reverse" => MutationKind::Reverse,
        _ => MutationKind::OnePoint,
    }
}

fn resolve_offspring_mode(operator_type: &str) -> OffspringMode {
    match operator_type {
        "four_children_select_two" => OffspringMode::FourChildrenSelectTwo,
        _ => OffspringMode::TwoChildren,
    }
}

fn solve_two_stage_restart(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    // Двухэтапный restart:
    // 1. Stage 1 использует обычные one_point/one_point/two_children операторы.
    // 2. После двух подряд restart-run с одинаковым best difference переключаемся на stage 2.
    // 3. Stage 2 использует настраиваемые операторы stage2_* и завершает весь solver,
    //    когда два подряд stage2-run дают одинаковый best difference.
    let mut total_generations = 0usize;
    let mut stage = 1usize;
    let mut current_population = None;
    let mut stage1_best_differences = Vec::new();
    let mut stage2_best_differences = Vec::new();
    let mut stage1_run_count = 0usize;
    let mut stage2_run_count = 0usize;
    let mut stage2_used = false;
    let mut current_crossover_kind = CrossoverKind::OnePoint;
    let mut current_mutation_kind = MutationKind::OnePoint;
    let mut current_offspring_mode = OffspringMode::TwoChildren;

    loop {
        let run = single_run(
            prices,
            target_sum,
            config,
            rng,
            current_population.take(),
            true,
            false,
            current_crossover_kind,
            current_mutation_kind,
            current_offspring_mode,
        );
        total_generations += run.generations_used;

        if stage == 1 {
            stage1_run_count += 1;
            stage1_best_differences.push(run.difference);
        } else {
            stage2_run_count += 1;
            stage2_best_differences.push(run.difference);
            stage2_used = true;
        }

        if run.stop_reason == "exact_match" {
            return Ok(build_result(
                run,
                total_generations,
                "exact_match",
                0,
                false,
                stage1_run_count,
                stage2_run_count,
                stage2_used,
                stage1_best_differences,
                stage2_best_differences,
            ));
        }

        if run.stop_reason == "generation_limit" {
            return Ok(build_result(
                run,
                total_generations,
                "generation_limit",
                0,
                false,
                stage1_run_count,
                stage2_run_count,
                stage2_used,
                stage1_best_differences,
                stage2_best_differences,
            ));
        }

        // Следующий run стартует не с полностью случайной популяции, а с элиты
        // последнего run и её сильных мутаций. Сила задаётся restart_mutation_fraction.
        current_population = Some(build_restart_population(
            &run.population,
            &run.best_vector,
            &config.restart_population_mode,
            config.restart_mutation_fraction,
            rng,
        )?);

        if stage == 1 {
            // Стабильный best difference на двух stage1 restart-run считается сигналом,
            // что обычные операторы исчерпались и пора перейти на stage2-операторы.
            if stage1_best_differences.len() >= 2
                && stage1_best_differences[stage1_best_differences.len() - 1]
                    == stage1_best_differences[stage1_best_differences.len() - 2]
            {
                stage = 2;
                current_crossover_kind = resolve_crossover_kind(&config.stage2_crossover_type);
                current_mutation_kind = resolve_mutation_kind(&config.stage2_mutation_type);
                current_offspring_mode = resolve_offspring_mode(&config.stage2_offspring_mode);
                continue;
            }

            current_crossover_kind = CrossoverKind::OnePoint;
            current_mutation_kind = MutationKind::OnePoint;
            current_offspring_mode = OffspringMode::TwoChildren;
            continue;
        }

        // На втором этапе такой же повтор best difference уже является финальной
        // стагнацией всего two_stage_restart solver.
        if stage2_best_differences.len() >= 2
            && stage2_best_differences[stage2_best_differences.len() - 1]
                == stage2_best_differences[stage2_best_differences.len() - 2]
        {
            return Ok(build_result(
                run,
                total_generations,
                "stagnation",
                0,
                false,
                stage1_run_count,
                stage2_run_count,
                stage2_used,
                stage1_best_differences,
                stage2_best_differences,
            ));
        }
    }
}

fn solve_with_genetic_algorithm(request: SolveRequest) -> Result<GeneticAlgorithmResult, String> {
    request.config.validate()?;
    let mut rng = Rng64::new(request.seed);
    match request.config.solver_mode.as_str() {
        "restart_rescue" => Ok(solve_restart_rescue(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        )),
        "two_stage_restart" => solve_two_stage_restart(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        _ => {
            let run = single_run(
                &request.prices,
                request.target_sum,
                &request.config,
                &mut rng,
                None,
                true,
                true,
                CrossoverKind::OnePoint,
                MutationKind::OnePoint,
                OffspringMode::TwoChildren,
            );
            let generations_used = run.generations_used;
            let stop_reason = if run.stop_reason == "stagnation_limit" {
                "stagnation"
            } else {
                run.stop_reason.as_str()
            }
            .to_string();
            Ok(build_result(
                run,
                generations_used,
                &stop_reason,
                0,
                false,
                0,
                0,
                false,
                Vec::new(),
                Vec::new(),
            ))
        }
    }
}

fn solve_json_inner(input: &str) -> String {
    let response = serde_json::from_str::<SolveRequest>(input)
        .map_err(|error| error.to_string())
        .and_then(solve_with_genetic_algorithm);

    match response {
        Ok(result) => serde_json::to_string(&result)
            .unwrap_or_else(|error| serde_json::to_string(&ErrorResponse { error: error.to_string() }).unwrap()),
        Err(error) => serde_json::to_string(&ErrorResponse { error }).unwrap(),
    }
}

#[no_mangle]
pub extern "C" fn solve_json(input: *const c_char) -> *mut c_char {
    // Единственная FFI-точка входа: Python передаёт JSON как C-строку,
    // Rust возвращает heap-allocated C-строку, которую Python обязан освободить
    // через free_rust_string(). catch_unwind не даёт panic пересечь FFI-границу.
    let output = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            return serde_json::to_string(&ErrorResponse {
                error: "input pointer is null".to_string(),
            })
            .unwrap();
        }

        let input = unsafe { CStr::from_ptr(input) };
        match input.to_str() {
            Ok(value) => solve_json_inner(value),
            Err(error) => serde_json::to_string(&ErrorResponse {
                error: error.to_string(),
            })
            .unwrap(),
        }
    }))
    .unwrap_or_else(|_| {
        serde_json::to_string(&ErrorResponse {
            error: "Rust core panic".to_string(),
        })
        .unwrap()
    });

    CString::new(output).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn free_rust_string(value: *mut c_char) {
    if !value.is_null() {
        unsafe {
            let _ = CString::from_raw(value);
        }
    }
}
