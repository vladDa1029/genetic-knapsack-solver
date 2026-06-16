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
    restart_mutation_type: String,
    restart_mutation_fraction: f64,
    stage1_crossover_type: String,
    stage1_mutation_type: String,
    stage1_offspring_mode: String,
    stage2_crossover_type: String,
    stage2_mutation_type: String,
    stage2_offspring_mode: String,
    stage2_restart_fraction: f64,
    multistage_crossover_type: String,
    multistage_mutation_type: String,
    multistage_offspring_mode: String,
    multistage_elite_count: usize,
    multistage_stage4_fraction: f64,
    multistage_stage5_fraction: f64,
    multistage_fresh_fraction: f64,
    multistage_double_mutation: bool,
    multistage_min_diversity: f64,
    multistage_late_tournament_size: usize,
    hybrid_max_outer_restarts: usize,
    // hybrid_targeted_restart: между outer-рестартами генерируем популяцию в k-bit
    // окрестности лучшей особи. k выбирается uniform в [k_min, k_max] для каждой особи.
    hybrid_targeted_k_min: usize,
    hybrid_targeted_k_max: usize,
    // hybrid_gene_fix: между outer-рестартами детектируем "замороженные" гены
    // (биты где >= threshold доля особей совпадают) и форсируем их инверсию
    // в части новой популяции.
    hybrid_gene_fix_threshold: f64,
    hybrid_gene_fix_invert_count: usize,
    // CHC (Eshelman 1991): HUX crossover + incest prevention + cataclysmic restart.
    // divergence_rate — доля бит инвертируемая в каждой особи при cataclysmic restart (типично 0.35).
    // initial_threshold — стартовый Hamming-порог для incest prevention (0 = auto = n_genes/4).
    // max_restarts — лимит cataclysmic перезапусков.
    chc_divergence_rate: f64,
    chc_initial_threshold: usize,
    chc_max_restarts: usize,
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
            restart_mutation_type: "many_bits".to_string(),
            restart_mutation_fraction: 0.4,
            stage1_crossover_type: "one_point".to_string(),
            stage1_mutation_type: "one_point".to_string(),
            stage1_offspring_mode: "two_children".to_string(),
            stage2_crossover_type: "two_point".to_string(),
            stage2_mutation_type: "two_point".to_string(),
            stage2_offspring_mode: "two_children".to_string(),
            stage2_restart_fraction: 0.40,
            multistage_crossover_type: "one_point".to_string(),
            multistage_mutation_type: "one_point".to_string(),
            multistage_offspring_mode: "two_children".to_string(),
            multistage_elite_count: 1,
            multistage_stage4_fraction: 0.60,
            multistage_stage5_fraction: 0.80,
            multistage_fresh_fraction: 0.0,
            multistage_double_mutation: false,
            multistage_min_diversity: 0.0,
            multistage_late_tournament_size: 0,
            hybrid_max_outer_restarts: 2,
            hybrid_targeted_k_min: 3,
            hybrid_targeted_k_max: 6,
            hybrid_gene_fix_threshold: 0.95,
            hybrid_gene_fix_invert_count: 3,
            chc_divergence_rate: 0.35,
            chc_initial_threshold: 0,
            chc_max_restarts: 5,
        }
    }
}

impl GeneticAlgorithmConfig {
    fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.solver_mode.as_str(),
            "classic" | "restart_rescue" | "two_stage_restart" | "five_stage_restart" | "hybrid_restart" | "progressive_restart" | "hybrid_targeted_restart" | "hybrid_gene_fix" | "cascading_pipeline" | "chc"
        ) {
            return Err(
                "solver_mode must be one of: classic, restart_rescue, two_stage_restart, five_stage_restart, hybrid_restart, progressive_restart, hybrid_targeted_restart, hybrid_gene_fix, cascading_pipeline, chc"
                    .to_string(),
            );
        }
        if self.population_size < 2 {
            return Err("population_size must be at least 2".to_string());
        }
        if self.multistage_elite_count < 1 {
            return Err("multistage_elite_count must be at least 1".to_string());
        }
        if self.multistage_elite_count > self.population_size {
            return Err("multistage_elite_count must not exceed population_size".to_string());
        }
        if self.multistage_fresh_fraction < 0.0 || self.multistage_fresh_fraction > 1.0 {
            return Err("multistage_fresh_fraction must be between 0.0 and 1.0".to_string());
        }
        if self.multistage_min_diversity < 0.0 || self.multistage_min_diversity > 1.0 {
            return Err("multistage_min_diversity must be between 0.0 and 1.0".to_string());
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
        if !matches!(self.restart_mutation_type.as_str(), "many_bits" | "reverse") {
            return Err("restart_mutation_type must be one of: many_bits, reverse".to_string());
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
            "two_children" | "four_children_select_two" | "six_children_from_three_select_two"
        ) {
            return Err(
                "stage2_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
                    .to_string(),
            );
        }
        if !matches!(self.stage1_crossover_type.as_str(), "one_point" | "two_point") {
            return Err("stage1_crossover_type must be one of: one_point, two_point".to_string());
        }
        if !matches!(
            self.stage1_mutation_type.as_str(),
            "one_point" | "two_point" | "reverse"
        ) {
            return Err("stage1_mutation_type must be one of: one_point, two_point, reverse".to_string());
        }
        if !matches!(
            self.stage1_offspring_mode.as_str(),
            "two_children" | "four_children_select_two" | "six_children_from_three_select_two"
        ) {
            return Err(
                "stage1_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
                    .to_string(),
            );
        }
        if !matches!(self.multistage_crossover_type.as_str(), "one_point" | "two_point") {
            return Err("multistage_crossover_type must be one of: one_point, two_point".to_string());
        }
        if !matches!(self.multistage_mutation_type.as_str(), "one_point" | "two_point") {
            return Err("multistage_mutation_type must be one of: one_point, two_point".to_string());
        }
        if !matches!(
            self.multistage_offspring_mode.as_str(),
            "two_children" | "four_children_select_two" | "six_children_from_three_select_two"
        ) {
            return Err(
                "multistage_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
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
            ("stage2_restart_fraction", self.stage2_restart_fraction),
            ("multistage_stage4_fraction", self.multistage_stage4_fraction),
            ("multistage_stage5_fraction", self.multistage_stage5_fraction),
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
            "five_stage_restart" => {
                if self.nga_mode != "none" {
                    return Err("nga_mode is not used in five_stage_restart solver_mode".to_string());
                }
                if self.repeat_limit.is_some() {
                    return Err("repeat_limit is not used in five_stage_restart solver_mode".to_string());
                }
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in five_stage_restart solver_mode".to_string());
                }
            }
            "hybrid_restart" => {
                if self.nga_mode != "none" {
                    return Err("nga_mode is not used in hybrid_restart solver_mode".to_string());
                }
                if self.repeat_limit.is_some() {
                    return Err("repeat_limit is not used in hybrid_restart solver_mode".to_string());
                }
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in hybrid_restart solver_mode".to_string());
                }
                if self.hybrid_max_outer_restarts == 0 {
                    return Err("hybrid_max_outer_restarts must be at least 1".to_string());
                }
            }
            "hybrid_targeted_restart" => {
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in hybrid_targeted_restart solver_mode".to_string());
                }
                if self.hybrid_max_outer_restarts == 0 {
                    return Err("hybrid_max_outer_restarts must be at least 1".to_string());
                }
                if self.hybrid_targeted_k_min == 0 || self.hybrid_targeted_k_max < self.hybrid_targeted_k_min {
                    return Err("hybrid_targeted_k_min must be >= 1 and k_max >= k_min".to_string());
                }
            }
            "hybrid_gene_fix" => {
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in hybrid_gene_fix solver_mode".to_string());
                }
                if self.hybrid_max_outer_restarts == 0 {
                    return Err("hybrid_max_outer_restarts must be at least 1".to_string());
                }
                if self.hybrid_gene_fix_threshold < 0.5 || self.hybrid_gene_fix_threshold > 1.0 {
                    return Err("hybrid_gene_fix_threshold must be in [0.5, 1.0]".to_string());
                }
                if self.hybrid_gene_fix_invert_count == 0 {
                    return Err("hybrid_gene_fix_invert_count must be >= 1".to_string());
                }
            }
            "cascading_pipeline" => {
                if self.generations < self.stagnation {
                    return Err("generations must be at least stagnation in cascading_pipeline solver_mode".to_string());
                }
                if self.hybrid_max_outer_restarts == 0 {
                    return Err("hybrid_max_outer_restarts must be at least 1".to_string());
                }
            }
            "chc" => {
                if !(0.05..=0.95).contains(&self.chc_divergence_rate) {
                    return Err("chc_divergence_rate must be in [0.05, 0.95]".to_string());
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
    stage_run_counts: Vec<usize>,
    stage_best_differences: Vec<i64>,
    final_stage: usize,
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
    SixChildrenFromThreeSelectTwo,
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

fn build_random_vector(gene_count: usize, rng: &mut Rng64) -> Vec<u8> {
    (0..gene_count).map(|_| u8::from(rng.random_f64() < 0.5)).collect()
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
                    let mutated = if config.multistage_double_mutation {
                        apply_mutation(mutation_kind, &mutated, config.mutation_rate, rng)
                    } else {
                        mutated
                    };
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
            let mutated_a = apply_mutation(mutation_kind, &child_a, config.mutation_rate, rng);
            let mutated_a = if config.multistage_double_mutation {
                apply_mutation(mutation_kind, &mutated_a, config.mutation_rate, rng)
            } else {
                mutated_a
            };
            let mutated_b = apply_mutation(mutation_kind, &child_b, config.mutation_rate, rng);
            let mutated_b = if config.multistage_double_mutation {
                apply_mutation(mutation_kind, &mutated_b, config.mutation_rate, rng)
            } else {
                mutated_b
            };
            [mutated_a, mutated_b]
        }
        OffspringMode::SixChildrenFromThreeSelectTwo => {
            // Этот режим обрабатывается напрямую в build_next_population (нужен 3-й родитель).
            // produce_offspring для него никогда не вызывается.
            unreachable!("SixChildrenFromThreeSelectTwo is handled in build_next_population")
        }
    }
}

fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

// Если потомок слишком похож на элиту (hamming < min_diff), применяем мутации
// до достижения порога (не более max_attempts попыток).
fn enforce_min_diversity(
    mut individual: Vec<u8>,
    elite: &[u8],
    min_diversity: f64,
    mutation_kind: MutationKind,
    rng: &mut Rng64,
) -> Vec<u8> {
    let n = individual.len();
    if n == 0 { return individual; }
    let min_diff = ((n as f64) * min_diversity).ceil() as usize;
    let mut attempts = 0;
    while hamming_distance(&individual, elite) < min_diff && attempts < 5 {
        individual = apply_mutation(mutation_kind, &individual, 1.0, rng);
        attempts += 1;
    }
    individual
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
    // best_difference > 0 означает, что элита ещё не решила задачу.
    // Diversity enforcement включается только в этом случае — когда нужно исследовать,
    // а не когда популяция уже сходится к правильному ответу.
    let best_difference = evaluations[ranked_indices[0]].0;
    let diversity_active = config.multistage_min_diversity > 0.0 && best_difference > 0;
    let mut next_generation = Vec::with_capacity(population.len());
    // Элитизм: лучшая особь всегда переносится в следующее поколение без изменений.
    next_generation.push(population[ranked_indices[0]].clone());

    while next_generation.len() < population.len() {
        match offspring_mode {
            OffspringMode::SixChildrenFromThreeSelectTwo => {
                // Режим three_parents: выбираем 3 родителей, делаем 3 кроссовера (P1×P2, P1×P3, P2×P3),
                // получаем 6 потомков, мутируем каждого и оставляем 2 лучших.
                let parent_a = pick_parent(population, evaluations, config.tournament_size, rng);
                let parent_b = pick_parent(population, evaluations, config.tournament_size, rng);
                let parent_c = pick_parent(population, evaluations, config.tournament_size, rng);

                let raw_children: Vec<Vec<u8>> = if rng.random_f64() < config.crossover_rate {
                    let (c1, c2) = apply_crossover(crossover_kind, &parent_a, &parent_b, rng);
                    let (c3, c4) = apply_crossover(crossover_kind, &parent_a, &parent_c, rng);
                    let (c5, c6) = apply_crossover(crossover_kind, &parent_b, &parent_c, rng);
                    vec![c1, c2, c3, c4, c5, c6]
                } else {
                    vec![
                        parent_a.clone(), parent_b.clone(),
                        parent_a.clone(), parent_c.clone(),
                        parent_b.clone(), parent_c.clone(),
                    ]
                };

                let mut candidates: Vec<(Vec<u8>, i64, i64, usize)> = raw_children
                    .into_iter()
                    .enumerate()
                    .map(|(index, child)| {
                        let mutated = apply_mutation(mutation_kind, &child, config.mutation_rate, rng);
                        let mutated = if config.multistage_double_mutation {
                            apply_mutation(mutation_kind, &mutated, config.mutation_rate, rng)
                        } else {
                            mutated
                        };
                        let (difference, current_sum) = evaluate_vector(prices, target_sum, &mutated);
                        (mutated, difference, current_sum, index)
                    })
                    .collect();
                candidates.sort_by_key(|(_, difference, current_sum, index)| (*difference, *current_sum, *index));

                let child1 = if diversity_active {
                    enforce_min_diversity(candidates.remove(0).0, &next_generation[0], config.multistage_min_diversity, mutation_kind, rng)
                } else { candidates.remove(0).0 };
                next_generation.push(child1);
                if next_generation.len() < population.len() {
                    let child2 = if diversity_active {
                        enforce_min_diversity(candidates.remove(0).0, &next_generation[0], config.multistage_min_diversity, mutation_kind, rng)
                    } else { candidates.remove(0).0 };
                    next_generation.push(child2);
                }
            }
            _ => {
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
                let child_a = if diversity_active {
                    enforce_min_diversity(child_a, &next_generation[0], config.multistage_min_diversity, mutation_kind, rng)
                } else { child_a };
                next_generation.push(child_a);
                if next_generation.len() < population.len() {
                    let child_b = if diversity_active {
                        enforce_min_diversity(child_b, &next_generation[0], config.multistage_min_diversity, mutation_kind, rng)
                    } else { child_b };
                    next_generation.push(child_b);
                }
            }
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
    generation_limit: Option<usize>,
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
    let mut generation = 0usize;

    loop {
        if generation_limit.is_some_and(|limit| generation >= limit) {
            return SingleRunResult {
                population: current_population,
                best_vector,
                best_sum,
                difference: best_difference,
                best_fitness,
                generations_used: generation,
                exact_match: best_difference == 0,
                stop_reason: if best_difference == 0 {
                    "exact_match".to_string()
                } else {
                    "generation_limit".to_string()
                },
                nga_used,
                nga_trigger_generation,
                nga_trigger_generations,
            };
        }

        generation += 1;
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
    mutation_type: &str,
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
        let restarted_vector = match mutation_type {
            "many_bits" => mutate_many_bits(vector, mutation_fraction, rng),
            // Restart reverse is intentionally deterministic: it replaces the strong
            // bit-flip restart operator with a full vector reversal for every non-elite.
            "reverse" => mutate_reverse(vector, 1.0, rng),
            _ => return Err(format!("unsupported restart_mutation_type: {mutation_type}")),
        };
        restart_population.push(restarted_vector);
    }

    Ok(restart_population)
}

fn split_reverse_halves(vector: &[u8]) -> Vec<u8> {
    let mid = vector.len() / 2;
    let mut mutated = Vec::with_capacity(vector.len());
    mutated.extend(vector[..mid].iter().rev().copied());
    mutated.extend(vector[mid..].iter().rev().copied());
    mutated
}

fn swap_halves(vector: &[u8]) -> Vec<u8> {
    let mid = vector.len() / 2;
    let mut mutated = Vec::with_capacity(vector.len());
    mutated.extend_from_slice(&vector[mid..]);
    mutated.extend_from_slice(&vector[..mid]);
    mutated
}

fn build_multistage_restart_population(
    base_population: &[Vec<u8>],
    prices: &[i64],
    target_sum: i64,
    elite_count: usize,
    next_stage: usize,
    stage4_fraction: f64,
    stage5_fraction: f64,
    fresh_fraction: f64,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    if base_population.is_empty() {
        return Vec::new();
    }

    let n = base_population[0].len();
    let pop_size = base_population.len();
    let evaluations = evaluate_population(base_population, prices, target_sum);
    let ranked_indices = rank_population_indices(&evaluations);
    let elite_count = elite_count.min(pop_size);
    let elite_indices: HashSet<usize> = ranked_indices.iter().take(elite_count).copied().collect();
    let mut next_population = Vec::with_capacity(pop_size);

    // Элита переносится без изменений.
    for index in ranked_indices.iter().take(elite_count) {
        next_population.push(base_population[*index].clone());
    }

    // Оставшиеся слоты: часть (fresh_fraction) заполняется свежими случайными особями,
    // остальные — стандартным преобразованием для данного этапа.
    let non_elite_count = pop_size - elite_count;
    let fresh_count = ((non_elite_count as f64) * fresh_fraction).round() as usize;
    let mut fresh_added = 0usize;

    for (index, vector) in base_population.iter().enumerate() {
        if elite_indices.contains(&index) {
            continue;
        }
        let restarted_vector = if fresh_added < fresh_count {
            fresh_added += 1;
            build_random_vector(n, rng)
        } else {
            match next_stage {
                2 => split_reverse_halves(vector),
                3 => swap_halves(vector),
                4 => mutate_many_bits(vector, stage4_fraction, rng),
                5 => mutate_many_bits(vector, stage5_fraction, rng),
                _ => vector.to_vec(),
            }
        };
        next_population.push(restarted_vector);
    }

    next_population
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
        stage_run_counts: Vec::new(),
        stage_best_differences: Vec::new(),
        final_stage: 0,
    }
}

fn build_result_with_stage_metadata(
    run: SingleRunResult,
    generations_used: usize,
    stop_reason: &str,
    stage_run_counts: Vec<usize>,
    stage_best_differences: Vec<i64>,
    final_stage: usize,
) -> GeneticAlgorithmResult {
    let mut result = build_result(
        run,
        generations_used,
        stop_reason,
        0,
        false,
        0,
        0,
        false,
        Vec::new(),
        Vec::new(),
    );
    result.stage_run_counts = stage_run_counts;
    result.stage_best_differences = stage_best_differences;
    result.final_stage = final_stage;
    result
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
        Some(config.generations),
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
            Some(config.generations),
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
            Some(config.generations),
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
        "six_children_from_three_select_two" => OffspringMode::SixChildrenFromThreeSelectTwo,
        _ => OffspringMode::TwoChildren,
    }
}

// Внутренняя функция: один цикл из max_stages этапов, стартует с заданной популяции.
// max_stages=5 — полный цикл (стандартное поведение).
// max_stages=1..4 — ранняя остановка для progressive_restart.
// Возвращает (последний SingleRunResult, счётчики этапов, разницы этапов, суммарные поколения).
fn run_five_stage_cycle(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
    initial_population: Option<Vec<Vec<u8>>>,
    max_stages: usize,
) -> (SingleRunResult, Vec<usize>, Vec<i64>, usize) {
    let max_stages = max_stages.clamp(1, 5);
    let crossover_kind = resolve_crossover_kind(&config.multistage_crossover_type);
    let mutation_kind = resolve_mutation_kind(&config.multistage_mutation_type);
    let offspring_mode = resolve_offspring_mode(&config.multistage_offspring_mode);
    let mut current_population = initial_population;
    let mut total_generations = 0usize;
    let mut stage_run_counts = Vec::with_capacity(5);
    let mut stage_best_differences = Vec::with_capacity(5);

    for stage in 1..=max_stages {
        // Если задан late_tournament_size — этапы 4 и 5 используют другой размер турнира.
        // Этапы 1–3: стандартный tournament_size (exploration).
        // Этапы 4–5: late_tournament_size (можно поставить меньше для доп. exploration
        //             на агрессивно-мутированной популяции, или больше для exploitation).
        let stage_config;
        let effective_config: &GeneticAlgorithmConfig =
            if config.multistage_late_tournament_size > 0 && stage >= 4 {
                stage_config = GeneticAlgorithmConfig {
                    tournament_size: config.multistage_late_tournament_size,
                    ..config.clone()
                };
                &stage_config
            } else {
                config
            };
        let run = single_run(
            prices,
            target_sum,
            effective_config,
            rng,
            current_population.take(),
            true,
            false,
            crossover_kind,
            mutation_kind,
            offspring_mode,
            None,
        );
        total_generations += run.generations_used;
        stage_run_counts.push(1);
        stage_best_differences.push(run.difference);

        if run.stop_reason == "exact_match" || stage == max_stages {
            return (run, stage_run_counts, stage_best_differences, total_generations);
        }

        current_population = Some(build_multistage_restart_population(
            &run.population,
            prices,
            target_sum,
            config.multistage_elite_count,
            stage + 1,
            config.multistage_stage4_fraction,
            config.multistage_stage5_fraction,
            config.multistage_fresh_fraction,
            rng,
        ));
    }
    unreachable!("run_five_stage_cycle loop exits via return")
}

// Примечание: функция оставлена для возможного будущего использования.
// В текущей реализации solve_hybrid_restart каждый внешний рестарт стартует
// с полностью случайной популяции (initial_population = None), что даёт лучшие
// результаты: элита из предыдущего цикла доминирует в турнирной селекции и
// возвращает алгоритм к тому же локальному оптимуму, из которого он пытался вырваться.
#[allow(dead_code)]
fn build_hybrid_outer_restart_population(
    final_population: &[Vec<u8>],
    prices: &[i64],
    target_sum: i64,
    elite_count: usize,
    mutation_fraction: f64,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let pop_size = final_population.len();
    if pop_size == 0 {
        return Vec::new();
    }
    let n = final_population[0].len();
    let evaluations = evaluate_population(final_population, prices, target_sum);
    let ranked = rank_population_indices(&evaluations);
    let elite_count = elite_count.min(pop_size);

    let mut new_pop: Vec<Vec<u8>> = Vec::with_capacity(pop_size);

    for &idx in ranked.iter().take(elite_count) {
        new_pop.push(final_population[idx].clone());
    }

    let elite_vecs: Vec<&Vec<u8>> = ranked.iter().take(elite_count).map(|&i| &final_population[i]).collect();
    let mutated_slots = (pop_size - elite_count) / 2;
    for i in 0..mutated_slots {
        let src = elite_vecs[i % elite_count];
        new_pop.push(mutate_many_bits(src, mutation_fraction, rng));
    }

    while new_pop.len() < pop_size {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(if rng.random_f64() < 0.5 { 1u8 } else { 0u8 });
        }
        new_pop.push(v);
    }

    new_pop
}

fn solve_five_stage_restart(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let (run, stage_run_counts, stage_best_differences, total_generations) =
        run_five_stage_cycle(prices, target_sum, config, rng, None, 5);
    let stop_reason = if run.stop_reason == "exact_match" { "exact_match" } else { "stagnation" };
    let final_stage = stage_run_counts.len();
    Ok(build_result_with_stage_metadata(
        run,
        total_generations,
        stop_reason,
        stage_run_counts,
        stage_best_differences,
        final_stage,
    ))
}

// Прогрессивный рестарт: эскалирует число этапов с каждой попыткой.
//
// Схема (при N = 5 уровнях):
//   Попытка 1: [этап 1]           — чистая случайная популяция
//   Попытка 2: [этап 1 → этап 2]  — 1 лучшая особь + 4999 случайных
//   Попытка 3: [этапы 1–3]        — 1 лучшая особь + 4999 случайных
//   Попытка 4: [этапы 1–4]        — 1 лучшая особь + 4999 случайных
//   Попытка 5: [этапы 1–5]        — 1 лучшая особь + 4999 случайных
//
// Ключевые принципы:
//   - Лучшая особь от предыдущей попытки "засевает" следующую популяцию.
//   - 4999 случайных особей обеспечивают diversity и выход из локальных оптимумов.
//   - 1/5000 = 0.02% — не доминирует в турнирной селекции, но "зерно" знания сохраняется.
//   - Рандом инжектируется ТОЛЬКО при старте каждой попытки (в stage 1).
//     Внутри попытки five_stage работает как обычно: трансформации между этапами не трогаем.
//   - Суммарная стоимость в worst case = 1+2+3+4+5 = 15 этапов ≈ 1.5× hybrid N=2.
fn solve_progressive_restart(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let n_genes = prices.len();
    let pop_size = config.population_size;
    let mut total_generations = 0usize;
    let mut best_run: Option<SingleRunResult> = None;
    let mut all_stage_run_counts: Vec<usize> = Vec::new();
    let mut all_stage_best_differences: Vec<i64> = Vec::new();

    for max_stages in 1..=5usize {
        // Строим начальную популяцию:
        //   - Попытка 1: полностью случайная (нет предыдущего знания).
        //   - Попытки 2–5: 1 лучшая особь + (pop_size-1) случайных.
        //     Это сохраняет "зерно" лучшего найденного решения при высоком diversity.
        let initial_pop = match &best_run {
            None => None, // attempt 1: полностью случайная
            Some(prev_best) => {
                let mut pop = build_initial_population(n_genes, pop_size, rng);
                pop[0] = prev_best.best_vector.clone(); // инжектируем лучшую особь
                Some(pop)
            }
        };

        let (run, stage_run_counts, stage_best_diffs, gens) =
            run_five_stage_cycle(prices, target_sum, config, rng, initial_pop, max_stages);
        total_generations += gens;
        all_stage_run_counts.extend_from_slice(&stage_run_counts);
        all_stage_best_differences.extend_from_slice(&stage_best_diffs);

        let exact = run.stop_reason == "exact_match";
        let is_better = best_run.as_ref().map_or(true, |b: &SingleRunResult| run.difference < b.difference);
        if is_better {
            best_run = Some(run);
        }

        if exact {
            break;
        }
    }

    let final_run = best_run.expect("best_run is always set after first cycle");
    let exact = final_run.stop_reason == "exact_match";
    let stop = if exact { "exact_match" } else { "stagnation" };
    let final_stage = all_stage_run_counts.len();
    Ok(build_result_with_stage_metadata(
        final_run,
        total_generations,
        stop,
        all_stage_run_counts,
        all_stage_best_differences,
        final_stage,
    ))
}

// Гибридный алгоритм: объединяет five_stage_restart и two_stage_restart.
//
// Схема:
//   1. Запустить полный цикл five_stage (5 этапов с нарастающей мутацией).
//   2. Если найдено точное решение — вернуть результат.
//   3. Если не найдено и outer_restart < hybrid_max_outer_restarts:
//      a. Построить «ударную» популяцию из элиты + сильных мутаций + свежего random.
//      b. Запустить five_stage снова с этой популяцией.
//      c. Повторять до исчерпания outer_restart.
//   4. Вернуть глобально лучший результат.
//
// Почему это лучше:
//   - five_stage прекрасно ищет, но иногда застревает в глубоком локальном оптимуме.
//   - two_stage умеет выбираться из ям через aggressive restart.
//   - Здесь мы используем five_stage как основной движок, а outer_restart как механизм
//     two_stage для принудительного escape из застрявших состояний.
fn solve_hybrid_restart(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let max_outer_restarts = config.hybrid_max_outer_restarts;
    let mut current_population: Option<Vec<Vec<u8>>> = None;
    let mut total_generations = 0usize;
    let mut best_run: Option<SingleRunResult> = None;
    let mut all_stage_run_counts: Vec<usize> = Vec::new();
    let mut all_stage_best_differences: Vec<i64> = Vec::new();

    for outer_restart in 0..=max_outer_restarts {
        let (run, stage_run_counts, stage_best_diffs, gens) =
            run_five_stage_cycle(prices, target_sum, config, rng, current_population.take(), 5);
        total_generations += gens;
        all_stage_run_counts.extend_from_slice(&stage_run_counts);
        all_stage_best_differences.extend_from_slice(&stage_best_diffs);

        let exact = run.stop_reason == "exact_match";

        // Обновляем глобально лучший результат
        let is_better = best_run.as_ref().map_or(true, |b: &SingleRunResult| run.difference < b.difference);
        if is_better {
            best_run = Some(run);
        }

        if exact || outer_restart == max_outer_restarts {
            let final_run = best_run.expect("best_run is always set after first cycle");
            let stop = if exact { "exact_match" } else { "stagnation" };
            let final_stage = all_stage_run_counts.len();
            return Ok(build_result_with_stage_metadata(
                final_run,
                total_generations,
                stop,
                all_stage_run_counts,
                all_stage_best_differences,
                final_stage,
            ));
        }

        // Следующий цикл стартует с чистой случайной популяции.
        //
        // Почему НЕ используем шоковую популяцию с элитой:
        // Если нести elite_count лучших векторов из предыдущего цикла, они получают
        // разницу ~100, тогда как свежие случайные векторы имеют разницу ~200.
        // Турнирная селекция (tournament_size=3) перетягивает всю популяцию обратно
        // к тому же локальному оптимуму, из которого мы пытались вырваться.
        // Cycle 2 повторяет путь cycle 1 к той же «яме» → результаты хуже, чем
        // один five_stage прогон.
        //
        // Правильная модель: hybrid_restart = best-of-N независимых five_stage прогонов.
        // Глобально лучший результат (best_run) сохраняется через все циклы.
        current_population = None;
    }

    Err("hybrid_restart finished without a terminal stage".to_string())
}

// Строит "целевую" начальную популяцию: каждая особь — результат инверсии k случайных
// битов лучшей особи, где k выбирается uniform из [k_min, k_max] для каждой особи.
// Сама лучшая особь не включается (только её соседи), это обеспечивает diversity при
// сохранении локальности поиска.
fn build_targeted_neighborhood_population(
    best_individual: &[u8],
    population_size: usize,
    k_min: usize,
    k_max: usize,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let n = best_individual.len();
    let k_min = k_min.max(1).min(n);
    let k_max = k_max.max(k_min).min(n);
    let mut population: Vec<Vec<u8>> = Vec::with_capacity(population_size);
    // Первая особь — сам "best" (элита, без изменений) для сохранения зерна.
    population.push(best_individual.to_vec());
    while population.len() < population_size {
        let k = rng.randint_inclusive(k_min, k_max);
        let mut candidate = best_individual.to_vec();
        // Выбираем k уникальных индексов и инвертируем их.
        let mut flipped: HashSet<usize> = HashSet::with_capacity(k);
        while flipped.len() < k {
            let idx = rng.randint_inclusive(0, n - 1);
            flipped.insert(idx);
        }
        for idx in flipped {
            candidate[idx] ^= 1;
        }
        population.push(candidate);
    }
    population
}

// Детектирует "замороженные" биты: позиции, где >= threshold доля особей популяции
// имеет одинаковое значение. Возвращает список (index, frozen_value).
fn detect_frozen_bits(population: &[Vec<u8>], threshold: f64) -> Vec<(usize, u8)> {
    if population.is_empty() {
        return Vec::new();
    }
    let n = population[0].len();
    let pop_size = population.len() as f64;
    let min_count = (pop_size * threshold).ceil() as usize;
    let mut frozen = Vec::new();
    for bit_idx in 0..n {
        let ones = population.iter().filter(|v| v[bit_idx] == 1).count();
        let zeros = population.len() - ones;
        if ones >= min_count {
            frozen.push((bit_idx, 1u8));
        } else if zeros >= min_count {
            frozen.push((bit_idx, 0u8));
        }
    }
    frozen
}

// Строит "gene-fix" популяцию: берём лучшую особь, в каждой новой особи форсируем
// инверсию случайно выбранных invert_count битов из замороженного множества.
// Если замороженных битов меньше invert_count — инвертируем все доступные.
// Если замороженных битов нет — fallback на случайную популяцию.
fn build_gene_fix_population(
    best_individual: &[u8],
    final_population: &[Vec<u8>],
    population_size: usize,
    threshold: f64,
    invert_count: usize,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let n = best_individual.len();
    let frozen = detect_frozen_bits(final_population, threshold);
    if frozen.is_empty() {
        // Нет замороженных битов — возвращаем случайную популяцию с элитой.
        let mut pop = build_initial_population(n, population_size, rng);
        pop[0] = best_individual.to_vec();
        return pop;
    }
    let mut population: Vec<Vec<u8>> = Vec::with_capacity(population_size);
    population.push(best_individual.to_vec());
    let actual_invert = invert_count.min(frozen.len());
    while population.len() < population_size {
        let mut candidate = best_individual.to_vec();
        // Выбираем actual_invert случайных индексов из frozen и инвертируем.
        let mut chosen: HashSet<usize> = HashSet::with_capacity(actual_invert);
        while chosen.len() < actual_invert {
            let idx = rng.randint_inclusive(0, frozen.len() - 1);
            chosen.insert(idx);
        }
        for idx in chosen {
            let (bit_pos, _frozen_val) = frozen[idx];
            candidate[bit_pos] ^= 1;
        }
        // Добавляем небольшую случайную мутацию (1-2 бита) на не-замороженных позициях
        // чтобы избежать полной идентичности у особей с одинаковым набором инвертированных битов.
        let extra_flips = rng.randint_inclusive(1, 2);
        for _ in 0..extra_flips {
            let idx = rng.randint_inclusive(0, n - 1);
            candidate[idx] ^= 1;
        }
        population.push(candidate);
    }
    population
}

// hybrid_targeted_restart: между outer-рестартами генерируем популяцию в k-bit
// окрестности глобально лучшей особи (а не случайную). Идея: если cycle 1 застрял
// в локальном оптимуме diff=D, точное решение скорее всего в нескольких битах от
// этого оптимума — нет смысла перезапускаться полностью случайно.
fn solve_hybrid_targeted_restart(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let max_outer_restarts = config.hybrid_max_outer_restarts;
    let pop_size = config.population_size;
    let mut current_population: Option<Vec<Vec<u8>>> = None;
    let mut total_generations = 0usize;
    let mut best_run: Option<SingleRunResult> = None;
    let mut all_stage_run_counts: Vec<usize> = Vec::new();
    let mut all_stage_best_differences: Vec<i64> = Vec::new();

    for outer_restart in 0..=max_outer_restarts {
        let (run, stage_run_counts, stage_best_diffs, gens) =
            run_five_stage_cycle(prices, target_sum, config, rng, current_population.take(), 5);
        total_generations += gens;
        all_stage_run_counts.extend_from_slice(&stage_run_counts);
        all_stage_best_differences.extend_from_slice(&stage_best_diffs);

        let exact = run.stop_reason == "exact_match";
        let is_better = best_run.as_ref().map_or(true, |b: &SingleRunResult| run.difference < b.difference);
        if is_better {
            best_run = Some(run);
        }

        if exact || outer_restart == max_outer_restarts {
            let final_run = best_run.expect("best_run is always set after first cycle");
            let stop = if exact { "exact_match" } else { "stagnation" };
            let final_stage = all_stage_run_counts.len();
            return Ok(build_result_with_stage_metadata(
                final_run, total_generations, stop,
                all_stage_run_counts, all_stage_best_differences, final_stage,
            ));
        }

        // Целевая популяция: k-bit окрестность глобально лучшего вектора.
        let best_vec = best_run.as_ref().expect("best_run set").best_vector.clone();
        current_population = Some(build_targeted_neighborhood_population(
            &best_vec, pop_size,
            config.hybrid_targeted_k_min, config.hybrid_targeted_k_max,
            rng,
        ));
    }
    Err("hybrid_targeted_restart finished without a terminal stage".to_string())
}

// hybrid_gene_fix: между outer-рестартами анализируем финальную популяцию и
// детектируем "замороженные" биты — позиции, где >= threshold (по умолчанию 95%)
// особей сошлись к одному значению. Новая популяция строится из лучшей особи с
// форсированной инверсией случайных подмножеств замороженных битов. Идея: GA
// застряла именно потому, что не может вырваться из консенсуса по этим битам,
// и точное решение требует инверсии нескольких из них.
fn solve_hybrid_gene_fix(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let max_outer_restarts = config.hybrid_max_outer_restarts;
    let pop_size = config.population_size;
    let mut current_population: Option<Vec<Vec<u8>>> = None;
    let mut total_generations = 0usize;
    let mut best_run: Option<SingleRunResult> = None;
    let mut all_stage_run_counts: Vec<usize> = Vec::new();
    let mut all_stage_best_differences: Vec<i64> = Vec::new();

    for outer_restart in 0..=max_outer_restarts {
        let (run, stage_run_counts, stage_best_diffs, gens) =
            run_five_stage_cycle(prices, target_sum, config, rng, current_population.take(), 5);
        total_generations += gens;
        all_stage_run_counts.extend_from_slice(&stage_run_counts);
        all_stage_best_differences.extend_from_slice(&stage_best_diffs);

        let exact = run.stop_reason == "exact_match";
        let final_population = run.population.clone();
        let is_better = best_run.as_ref().map_or(true, |b: &SingleRunResult| run.difference < b.difference);
        if is_better {
            best_run = Some(run);
        }

        if exact || outer_restart == max_outer_restarts {
            let final_run = best_run.expect("best_run is always set after first cycle");
            let stop = if exact { "exact_match" } else { "stagnation" };
            let final_stage = all_stage_run_counts.len();
            return Ok(build_result_with_stage_metadata(
                final_run, total_generations, stop,
                all_stage_run_counts, all_stage_best_differences, final_stage,
            ));
        }

        let best_vec = best_run.as_ref().expect("best_run set").best_vector.clone();
        current_population = Some(build_gene_fix_population(
            &best_vec, &final_population, pop_size,
            config.hybrid_gene_fix_threshold, config.hybrid_gene_fix_invert_count,
            rng,
        ));
    }
    Err("hybrid_gene_fix finished without a terminal stage".to_string())
}

// cascading_pipeline: волновой пайплайн. На каждой итерации все активные слои
// продвигаются на один шаг вперёд, и стартует новый свежий Stage1.
//
// Схема:
//   Iter 1: [S1_a (fresh)]                                          — 1 stage-run
//   Iter 2: [S1_b (fresh)] + [S2_a (из pop S1_a)]                   — 2 stage-runs
//   Iter 3: [S1_c]         + [S2_b (из S1_b)] + [S3_a (из S2_a)]    — 3 stage-runs
//   Iter 4: [S1_d]+[S2_c]+[S3_b]+[S4_a]                              — 4
//   Iter 5: [S1_e]+[S2_d]+[S3_c]+[S4_b]+[S5_a]                       — 5
//   Всего: 15 stage-runs (≈ 1.5× hybrid N=2 = 10 stage-runs).
//
// Если на любой итерации в любом слое найдено exact — возвращаем сразу.
// Если после 5 итераций exact не найден — переходим к "hybrid part 2":
// дополнительный полный five_stage_cycle с свежей популяцией.
//
// Принципиальное отличие от progressive_restart: ВСЕ слои работают одновременно,
// каждый продолжая ту цепочку трансформаций которая для него корректна. Stage 5
// получает популяцию, которая прошла через S1→S2→S3→S4 (как и в обычном
// five_stage), но параллельно с этим запускаются и более "молодые" пайплайны
// (которые могут найти решение раньше своей длины).
fn solve_cascading_pipeline(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let _n_genes = prices.len();
    let _pop_size = config.population_size;
    let mut total_generations = 0usize;
    let mut best_run: Option<SingleRunResult> = None;
    let mut all_stage_run_counts: Vec<usize> = Vec::new();
    let mut all_stage_best_differences: Vec<i64> = Vec::new();

    // pipelines[i] = популяция готовая для запуска на этапе (i+1) на следующей итерации.
    // Длина растёт от 1 до 5 по мере "взросления" волнового фронта.
    // pipelines[0] = популяция для следующего Stage1, pipelines[1] = для Stage2, ...
    let mut pipelines: Vec<Vec<Vec<u8>>> = Vec::with_capacity(5);

    let mut found_exact = false;
    let mut early_return_run: Option<SingleRunResult> = None;

    for iter in 1..=5usize {
        // На этой итерации запускаем слои 1..=iter.
        // Слой j использует pipelines[j-1] если есть, иначе fresh (для j=1 всегда fresh
        // на старте, а на последующих итерациях pipelines[0] всегда обновляется).
        let mut new_pipelines: Vec<Vec<Vec<u8>>> = Vec::with_capacity(iter);

        for stage in 1..=iter {
            // Готовим input population для этого слоя.
            let input_pop: Option<Vec<Vec<u8>>> = if stage == 1 {
                // Stage1 всегда fresh.
                None
            } else {
                // Stage j (j>=2) использует трансформированную популяцию из pipelines[j-2].
                let prev = &pipelines[stage - 2];
                Some(build_multistage_restart_population(
                    prev, prices, target_sum,
                    config.multistage_elite_count, stage,
                    config.multistage_stage4_fraction,
                    config.multistage_stage5_fraction,
                    config.multistage_fresh_fraction,
                    rng,
                ))
            };

            // Применяем late tournament size если нужно (как в run_five_stage_cycle).
            let stage_config;
            let effective_config: &GeneticAlgorithmConfig =
                if config.multistage_late_tournament_size > 0 && stage >= 4 {
                    stage_config = GeneticAlgorithmConfig {
                        tournament_size: config.multistage_late_tournament_size,
                        ..config.clone()
                    };
                    &stage_config
                } else {
                    config
                };

            let crossover_kind = resolve_crossover_kind(&config.multistage_crossover_type);
            let mutation_kind = resolve_mutation_kind(&config.multistage_mutation_type);
            let offspring_mode = resolve_offspring_mode(&config.multistage_offspring_mode);

            let run = single_run(
                prices, target_sum, effective_config, rng,
                input_pop, true, false,
                crossover_kind, mutation_kind, offspring_mode, None,
            );
            total_generations += run.generations_used;
            all_stage_run_counts.push(1);
            all_stage_best_differences.push(run.difference);

            let exact = run.stop_reason == "exact_match";
            let is_better = best_run.as_ref().map_or(true, |b: &SingleRunResult| run.difference < b.difference);

            // Сохраняем популяцию для следующей итерации (слой j → станет слоем j+1).
            new_pipelines.push(run.population.clone());

            if is_better {
                best_run = Some(run.clone());
            }
            if exact {
                found_exact = true;
                early_return_run = Some(run);
                break;
            }
        }

        if found_exact {
            break;
        }

        // Сдвиг волнового фронта: pipelines теперь = new_pipelines.
        // Если iter < 5, на следующей итерации мы запустим (iter+1) слоёв:
        // Stage1 (fresh) + Stage2 из new_pipelines[0] + ... + Stage(iter+1) из new_pipelines[iter-1].
        pipelines = new_pipelines;
        // Если pipelines уже содержит 5 элементов (после iter=5), мы не дойдём до этой точки
        // потому что цикл закончится. На iter=5 запускается до 5 слоёв включительно, и
        // дальше переходим к hybrid-part-2.
    }

    if found_exact {
        let final_run = early_return_run.unwrap_or_else(|| best_run.clone().expect("best_run is set"));
        let final_stage = all_stage_run_counts.len();
        return Ok(build_result_with_stage_metadata(
            final_run, total_generations, "exact_match",
            all_stage_run_counts, all_stage_best_differences, final_stage,
        ));
    }

    // Hybrid part 2: дополнительный полный five_stage_cycle с чистой случайной популяцией.
    // (так же как outer_restart в обычном hybrid_restart с N=1 дополнительным циклом.)
    let (run2, stage_counts2, stage_diffs2, gens2) =
        run_five_stage_cycle(prices, target_sum, config, rng, None, 5);
    total_generations += gens2;
    all_stage_run_counts.extend_from_slice(&stage_counts2);
    all_stage_best_differences.extend_from_slice(&stage_diffs2);

    let exact2 = run2.stop_reason == "exact_match";
    let is_better2 = best_run.as_ref().map_or(true, |b: &SingleRunResult| run2.difference < b.difference);
    if is_better2 {
        best_run = Some(run2);
    }

    let final_run = best_run.expect("best_run is set after iterations");
    let stop = if exact2 || final_run.stop_reason == "exact_match" { "exact_match" } else { "stagnation" };
    let final_stage = all_stage_run_counts.len();
    Ok(build_result_with_stage_metadata(
        final_run, total_generations, stop,
        all_stage_run_counts, all_stage_best_differences, final_stage,
    ))
}

// ============================================================================
// CHC (Eshelman 1991): Cross-generational elitist selection, Heterogeneous
// recombination (HUX), Cataclysmic mutation.
//
// Особенности:
//   - HUX crossover: считаем биты различия, ровно половину обмениваем.
//   - Incest prevention: кроссовер только если hamming(p1,p2)/2 > threshold.
//   - НЕТ обычной мутации в цикле: разнообразие только через HUX.
//   - Elitist replacement: новое поколение = top-N из P ∪ C.
//   - Cataclysmic restart: когда threshold падает до 0, новая популяция =
//     [best] + (N-1) × mutate(best, ~35% бит). Threshold ресетается на ~r*L.
//
// Зачем: спроектирован специально для трудных бинарных оптимизационных задач
// с обилием локальных оптимумов. Принципиально иная парадигма от five_stage:
// никакой эскалации мутации, всё разнообразие из crossover. Если current
// hypothesis "high selection pressure кладёт нас в локальный оптимум" верна,
// CHC её обходит полностью.
// ============================================================================

// HUX (Half-Uniform Crossover): для всех бит где родители различаются,
// случайно выбираем ровно половину и обмениваем. Возвращает 2 ребёнка.
fn hux_crossover(parent_a: &[u8], parent_b: &[u8], rng: &mut Rng64) -> (Vec<u8>, Vec<u8>) {
    let n = parent_a.len();
    let differing: Vec<usize> = (0..n).filter(|&i| parent_a[i] != parent_b[i]).collect();
    let swap_count = differing.len() / 2;
    // Выбираем swap_count случайных позиций из differing для обмена.
    let swap_indices_in_differing = rng.sample_range(differing.len(), swap_count);
    let swap_set: HashSet<usize> = swap_indices_in_differing.iter().map(|&i| differing[i]).collect();

    let mut child_a = parent_a.to_vec();
    let mut child_b = parent_b.to_vec();
    for &pos in &swap_set {
        child_a[pos] = parent_b[pos];
        child_b[pos] = parent_a[pos];
    }
    (child_a, child_b)
}

// Cataclysmic restart: новая популяция = [best_individual] + (pop_size-1)
// мутаций best_individual где каждый бит инвертирован с вероятностью divergence_rate.
fn build_cataclysmic_population(
    best_individual: &[u8],
    pop_size: usize,
    divergence_rate: f64,
    rng: &mut Rng64,
) -> Vec<Vec<u8>> {
    let mut population = Vec::with_capacity(pop_size);
    population.push(best_individual.to_vec());
    while population.len() < pop_size {
        let mut individual = best_individual.to_vec();
        for bit in individual.iter_mut() {
            if rng.random_f64() < divergence_rate {
                *bit ^= 1;
            }
        }
        population.push(individual);
    }
    population
}

fn solve_chc(
    prices: &[i64],
    target_sum: i64,
    config: &GeneticAlgorithmConfig,
    rng: &mut Rng64,
) -> Result<GeneticAlgorithmResult, String> {
    let n_genes = prices.len();
    let pop_size = config.population_size;
    let divergence_rate = config.chc_divergence_rate;
    let max_restarts = config.chc_max_restarts;
    let max_generations = config.generations;
    let stagnation_limit = config.stagnation;

    // Стартовый Hamming-порог. Eshelman рекомендует L/4 где L = длина хромосомы.
    let initial_threshold = if config.chc_initial_threshold > 0 {
        config.chc_initial_threshold
    } else {
        n_genes / 4
    };
    // Порог после cataclysmic restart: r * (1-r) * L по Eshelman.
    let restart_threshold = ((n_genes as f64) * divergence_rate * (1.0 - divergence_rate)).ceil() as usize;

    let mut population = build_initial_population(n_genes, pop_size, rng);
    let mut evaluations = evaluate_population(&population, prices, target_sum);
    let mut threshold = initial_threshold as i64; // i64 чтобы можно было уйти в отрицательные при saturating
    let mut generations_used = 0usize;
    let mut restart_count = 0usize;

    // Глобально лучшее найденное решение.
    // best_from_population возвращает (vector, sum, difference) — порядок не совпадает с именованием!
    let (mut best_vec, mut best_sum, mut best_diff) = best_from_population(&population, &evaluations);
    let mut no_improvement_streak = 0usize;
    // Стадия диагностики (для метаданных): счётчик поколений в каждом "эпохе" между рестартами.
    let mut stage_run_counts: Vec<usize> = vec![0];
    let mut stage_best_differences: Vec<i64> = Vec::new();

    let mut exact_found = best_diff == 0;

    while !exact_found && generations_used < max_generations {
        // Формируем N/2 родительских пар через перемешивание индексов.
        let shuffled = rng.sample_range(pop_size, pop_size);

        let mut offspring: Vec<Vec<u8>> = Vec::new();
        for pair in shuffled.chunks(2) {
            if pair.len() < 2 { break; }
            let p1 = &population[pair[0]];
            let p2 = &population[pair[1]];
            let hamming = hamming_distance(p1, p2);
            // Incest prevention: половина differ-бит должна превышать threshold.
            if (hamming as i64) / 2 > threshold {
                let (c1, c2) = hux_crossover(p1, p2, rng);
                offspring.push(c1);
                offspring.push(c2);
            }
        }

        if offspring.is_empty() {
            // Кроссоверов не было — популяция слишком похожа на саму себя.
            threshold -= 1;
        } else {
            let off_evaluations = evaluate_population(&offspring, prices, target_sum);

            // Elitist survival: top pop_size из P ∪ C по difference (затем по sum как tiebreaker).
            // Собираем (diff, sum, source_index, is_offspring).
            let mut combined: Vec<(i64, i64, usize, bool)> = Vec::with_capacity(pop_size + offspring.len());
            for (i, (diff, sum)) in evaluations.iter().enumerate() {
                combined.push((*diff, *sum, i, false));
            }
            for (i, (diff, sum)) in off_evaluations.iter().enumerate() {
                combined.push((*diff, *sum, i, true));
            }
            combined.sort_by_key(|&(diff, sum, _, _)| (diff, sum));

            let mut new_population = Vec::with_capacity(pop_size);
            let mut new_evaluations = Vec::with_capacity(pop_size);
            let mut entered_count = 0usize;
            for &(diff, sum, idx, is_off) in combined.iter().take(pop_size) {
                if is_off {
                    new_population.push(offspring[idx].clone());
                    entered_count += 1;
                } else {
                    new_population.push(population[idx].clone());
                }
                new_evaluations.push((diff, sum));
            }
            // Если ни одно потомство не попало в новое поколение — популяция стагнирует.
            if entered_count == 0 {
                threshold -= 1;
            }
            population = new_population;
            evaluations = new_evaluations;
        }

        generations_used += 1;
        *stage_run_counts.last_mut().unwrap() += 1;

        // Обновляем глобальный best.
        let (cur_vec, cur_sum, cur_diff) = best_from_population(&population, &evaluations);
        if cur_diff < best_diff {
            best_vec = cur_vec;
            best_diff = cur_diff;
            best_sum = cur_sum;
            no_improvement_streak = 0;
        } else {
            no_improvement_streak += 1;
        }
        if best_diff == 0 {
            exact_found = true;
            break;
        }

        // Cataclysmic restart triggers: threshold <= 0 ИЛИ стагнация по поколениям.
        let trigger_threshold = threshold <= 0;
        let trigger_stagnation = no_improvement_streak >= stagnation_limit;
        if trigger_threshold || trigger_stagnation {
            if restart_count >= max_restarts {
                break;
            }
            stage_best_differences.push(best_diff);
            population = build_cataclysmic_population(&best_vec, pop_size, divergence_rate, rng);
            evaluations = evaluate_population(&population, prices, target_sum);
            threshold = restart_threshold as i64;
            no_improvement_streak = 0;
            restart_count += 1;
            stage_run_counts.push(0);
        }
    }
    // Финальная запись diff для последней эпохи.
    stage_best_differences.push(best_diff);

    let stop_reason = if exact_found { "exact_match" } else if generations_used >= max_generations { "generation_limit" } else { "stagnation" };
    let run = SingleRunResult {
        population,
        best_vector: best_vec,
        best_sum,
        difference: best_diff,
        best_fitness: -best_diff,
        generations_used,
        exact_match: exact_found,
        stop_reason: stop_reason.to_string(),
        nga_used: false,
        nga_trigger_generation: None,
        nga_trigger_generations: Vec::new(),
    };
    let final_stage = stage_run_counts.len();
    Ok(build_result_with_stage_metadata(
        run,
        generations_used,
        stop_reason,
        stage_run_counts,
        stage_best_differences,
        final_stage,
    ))
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
    let mut current_crossover_kind = resolve_crossover_kind(&config.stage1_crossover_type);
    let mut current_mutation_kind = resolve_mutation_kind(&config.stage1_mutation_type);
    let mut current_offspring_mode = resolve_offspring_mode(&config.stage1_offspring_mode);

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
            Some(config.generations),
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
        let restart_fraction = if stage == 1 {
            config.restart_mutation_fraction
        } else {
            config.stage2_restart_fraction
        };
        current_population = Some(build_restart_population(
            &run.population,
            &run.best_vector,
            &config.restart_population_mode,
            &config.restart_mutation_type,
            restart_fraction,
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

            current_crossover_kind = resolve_crossover_kind(&config.stage1_crossover_type);
            current_mutation_kind = resolve_mutation_kind(&config.stage1_mutation_type);
            current_offspring_mode = resolve_offspring_mode(&config.stage1_offspring_mode);
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
        "five_stage_restart" => solve_five_stage_restart(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "hybrid_restart" => solve_hybrid_restart(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "progressive_restart" => solve_progressive_restart(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "hybrid_targeted_restart" => solve_hybrid_targeted_restart(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "hybrid_gene_fix" => solve_hybrid_gene_fix(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "cascading_pipeline" => solve_cascading_pipeline(
            &request.prices,
            request.target_sum,
            &request.config,
            &mut rng,
        ),
        "chc" => solve_chc(
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
                Some(request.config.generations),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reverse_halves_reverses_each_half() {
        assert_eq!(
            split_reverse_halves(&[1, 0, 1, 1, 0]),
            vec![0, 1, 0, 1, 1]
        );
    }

    #[test]
    fn swap_halves_moves_right_half_first() {
        assert_eq!(swap_halves(&[1, 0, 1, 1, 0]), vec![1, 1, 0, 1, 0]);
    }

    #[test]
    fn multistage_population_preserves_multiple_elites() {
        let population = vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 1, 1],
        ];
        let mut rng = Rng64::new(1);
        let restarted =
            build_multistage_restart_population(&population, &[10, 7, 3], 10, 2, 2, 0.6, 0.8, 0.0, &mut rng);

        assert_eq!(restarted[0], vec![1, 0, 0]);
        assert_eq!(restarted[1], vec![0, 1, 0]);
        assert_eq!(restarted.len(), population.len());
    }

    #[test]
    fn multistage_population_applies_sixty_and_eighty_percent_mutations() {
        let population = vec![vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0], vec![0; 10], vec![0; 10]];
        let prices = vec![1; 10];
        let mut rng = Rng64::new(1);
        let stage4 = build_multistage_restart_population(&population, &prices, 1, 1, 4, 0.6, 0.8, 0.0, &mut rng);
        let mut rng = Rng64::new(1);
        let stage5 = build_multistage_restart_population(&population, &prices, 1, 1, 5, 0.6, 0.8, 0.0, &mut rng);

        assert_eq!(stage4[1].iter().filter(|bit| **bit == 1).count(), 6);
        assert_eq!(stage4[2].iter().filter(|bit| **bit == 1).count(), 6);
        assert_eq!(stage5[1].iter().filter(|bit| **bit == 1).count(), 8);
        assert_eq!(stage5[2].iter().filter(|bit| **bit == 1).count(), 8);
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
