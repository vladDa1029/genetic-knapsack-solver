from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NgaMode = Literal["none", "two_point", "elite_heavy_mutation", "staged_hypermutation"]
SolverMode = Literal[
    "classic", "restart_rescue", "two_stage_restart", "five_stage_restart",
    "hybrid_restart", "progressive_restart",
    "hybrid_targeted_restart", "hybrid_gene_fix", "cascading_pipeline",
    "chc", "deterministic_crowding", "hybrid_portfolio",
]
RestartPopulationMode = Literal["elite_from_last_population"]
RestartMutationType = Literal["many_bits", "reverse"]
CrossoverType = Literal["one_point", "two_point"]
MutationType = Literal["one_point", "two_point", "reverse"]
MultistageMutationType = Literal["one_point", "two_point"]
MultistageOffspringMode = Literal["two_children", "four_children_select_two", "six_children_from_three_select_two"]
Stage2OffspringMode = Literal["two_children", "four_children_select_two", "six_children_from_three_select_two"]
OperatorType = Literal["one_point", "two_point", "reverse"]


@dataclass(slots=True)
class ProblemInstance:
    items: list[str]
    prices: list[int]
    hidden_vector: list[int]
    target_sum: int
    generation_mode: str


@dataclass(slots=True)
class GeneticAlgorithmConfig:
    solver_mode: SolverMode = "classic"
    population_size: int = 100
    generations: int = 200
    stagnation: int = 30
    repeat_limit: int | None = None
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    tournament_size: int = 3
    nga_mode: NgaMode = "none"
    nga_mutation_fraction: float = 0.4
    nga_trigger_points: tuple[int, ...] = ()
    nga_mutate_points: tuple[int, ...] = (40,)
    restart_max_count: int | None = None
    rescue_min_mutated_bits_ratio: float = 0.4
    restart_population_mode: RestartPopulationMode = "elite_from_last_population"
    restart_mutation_type: RestartMutationType = "many_bits"
    restart_mutation_fraction: float = 0.4
    stage2_crossover_type: CrossoverType = "two_point"
    stage2_mutation_type: MutationType = "two_point"
    stage2_offspring_mode: Stage2OffspringMode = "two_children"
    multistage_crossover_type: CrossoverType = "one_point"
    multistage_mutation_type: MultistageMutationType = "one_point"
    multistage_offspring_mode: MultistageOffspringMode = "two_children"
    multistage_elite_count: int = 1
    stage1_crossover_type: CrossoverType = "one_point"
    stage1_mutation_type: MutationType = "one_point"
    stage1_offspring_mode: Stage2OffspringMode = "two_children"
    stage2_restart_fraction: float = 0.40
    multistage_stage4_fraction: float = 0.60
    multistage_stage5_fraction: float = 0.80
    multistage_fresh_fraction: float = 0.0
    multistage_double_mutation: bool = False
    multistage_min_diversity: float = 0.0
    multistage_late_tournament_size: int = 0
    hybrid_max_outer_restarts: int = 2
    hybrid_targeted_k_min: int = 3
    hybrid_targeted_k_max: int = 6
    hybrid_gene_fix_threshold: float = 0.95
    hybrid_gene_fix_invert_count: int = 3
    chc_divergence_rate: float = 0.35
    chc_initial_threshold: int = 0
    chc_max_restarts: int = 5
    dc_mutation_fraction: float = 0.03

    def __post_init__(self) -> None:
        if self.solver_mode not in (
            "classic", "restart_rescue", "two_stage_restart", "five_stage_restart",
            "hybrid_restart", "progressive_restart",
            "hybrid_targeted_restart", "hybrid_gene_fix", "cascading_pipeline",
            "chc", "deterministic_crowding", "hybrid_portfolio",
        ):
            raise ValueError(
                "solver_mode must be one of: classic, restart_rescue, two_stage_restart, "
                "five_stage_restart, hybrid_restart, progressive_restart, "
                "hybrid_targeted_restart, hybrid_gene_fix, cascading_pipeline, chc, "
                "deterministic_crowding, hybrid_portfolio"
            )
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.multistage_elite_count < 1:
            raise ValueError("multistage_elite_count must be at least 1")
        if self.multistage_elite_count > self.population_size:
            raise ValueError("multistage_elite_count must not exceed population_size")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.stagnation < 1:
            raise ValueError("stagnation must be at least 1")
        if self.repeat_limit is not None and self.repeat_limit < 1:
            raise ValueError("repeat_limit must be at least 1")
        if self.restart_max_count is not None and self.restart_max_count < 0:
            raise ValueError("restart_max_count must be at least 0")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.nga_mode not in ("none", "two_point", "elite_heavy_mutation", "staged_hypermutation"):
            raise ValueError(
                "nga_mode must be one of: none, two_point, elite_heavy_mutation, staged_hypermutation"
            )
        if self.restart_population_mode not in ("elite_from_last_population",):
            raise ValueError("restart_population_mode must be one of: elite_from_last_population")
        if self.restart_mutation_type not in ("many_bits", "reverse"):
            raise ValueError("restart_mutation_type must be one of: many_bits, reverse")
        if self.stage2_crossover_type not in ("one_point", "two_point"):
            raise ValueError("stage2_crossover_type must be one of: one_point, two_point")
        if self.stage2_mutation_type not in ("one_point", "two_point", "reverse"):
            raise ValueError("stage2_mutation_type must be one of: one_point, two_point, reverse")
        if self.stage2_offspring_mode not in ("two_children", "four_children_select_two", "six_children_from_three_select_two"):
            raise ValueError(
                "stage2_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
            )
        if self.multistage_crossover_type not in ("one_point", "two_point"):
            raise ValueError("multistage_crossover_type must be one of: one_point, two_point")
        if self.multistage_mutation_type not in ("one_point", "two_point"):
            raise ValueError("multistage_mutation_type must be one of: one_point, two_point")
        if self.multistage_offspring_mode not in ("two_children", "four_children_select_two", "six_children_from_three_select_two"):
            raise ValueError(
                "multistage_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
            )
        if self.stage1_crossover_type not in ("one_point", "two_point"):
            raise ValueError("stage1_crossover_type must be one of: one_point, two_point")
        if self.stage1_mutation_type not in ("one_point", "two_point", "reverse"):
            raise ValueError("stage1_mutation_type must be one of: one_point, two_point, reverse")
        if self.stage1_offspring_mode not in ("two_children", "four_children_select_two", "six_children_from_three_select_two"):
            raise ValueError(
                "stage1_offspring_mode must be one of: two_children, four_children_select_two, six_children_from_three_select_two"
            )
        for name, value in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("nga_mutation_fraction", self.nga_mutation_fraction),
            ("rescue_min_mutated_bits_ratio", self.rescue_min_mutated_bits_ratio),
            ("restart_mutation_fraction", self.restart_mutation_fraction),
            ("stage2_restart_fraction", self.stage2_restart_fraction),
            ("multistage_stage4_fraction", self.multistage_stage4_fraction),
            ("multistage_stage5_fraction", self.multistage_stage5_fraction),
            ("multistage_fresh_fraction", self.multistage_fresh_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.solver_mode == "classic":
            if self.nga_mode in ("two_point", "elite_heavy_mutation") and self.repeat_limit is None:
                raise ValueError("repeat_limit must be provided when nga_mode is two_point or elite_heavy_mutation")
            if self.nga_mode == "staged_hypermutation":
                if self.repeat_limit is not None:
                    raise ValueError("repeat_limit is not used in staged_hypermutation mode")
                if not self.nga_trigger_points:
                    raise ValueError("nga_trigger_points must be provided for staged_hypermutation mode")
                previous_point = 0
                for point in self.nga_trigger_points:
                    if point < 1:
                        raise ValueError("nga_trigger_points must contain only positive integers")
                    if point >= self.stagnation:
                        raise ValueError("each nga_trigger_point must be less than stagnation")
                    if point <= previous_point:
                        raise ValueError("nga_trigger_points must be strictly increasing")
                    previous_point = point
                if not self.nga_mutate_points:
                    raise ValueError("nga_mutate_points must be provided for staged_hypermutation mode")
                if len(self.nga_mutate_points) not in (1, len(self.nga_trigger_points)):
                    raise ValueError(
                        "nga_mutate_points must contain either one value or match nga_trigger_points length"
                    )
                for mutate_percent in self.nga_mutate_points:
                    if not 0 <= mutate_percent <= 100:
                        raise ValueError("nga_mutate_points must be between 0 and 100")
        elif self.solver_mode == "restart_rescue":
            if self.nga_mode != "none":
                raise ValueError("nga_mode is only available in classic solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is only available in classic solver_mode")
        elif self.solver_mode == "two_stage_restart":
            if self.nga_mode != "none":
                raise ValueError("nga_mode is not used in two_stage_restart solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is not used in two_stage_restart solver_mode")
            if self.generations < self.stagnation:
                raise ValueError("generations must be at least stagnation in two_stage_restart solver_mode")
        elif self.solver_mode == "five_stage_restart":
            if self.nga_mode != "none":
                raise ValueError("nga_mode is not used in five_stage_restart solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is not used in five_stage_restart solver_mode")
            if self.generations < self.stagnation:
                raise ValueError("generations must be at least stagnation in five_stage_restart solver_mode")
        elif self.solver_mode == "chc":
            if self.nga_mode != "none":
                raise ValueError("nga_mode is not used in chc solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is not used in chc solver_mode")
            if not 0.05 <= self.chc_divergence_rate <= 0.95:
                raise ValueError("chc_divergence_rate must be in [0.05, 0.95]")
            if self.chc_initial_threshold < 0:
                raise ValueError("chc_initial_threshold must be >= 0")
            if self.chc_max_restarts < 0:
                raise ValueError("chc_max_restarts must be >= 0")
        elif self.solver_mode == "deterministic_crowding":
            if self.nga_mode != "none":
                raise ValueError("nga_mode is not used in deterministic_crowding solver_mode")
            if self.repeat_limit is not None:
                raise ValueError("repeat_limit is not used in deterministic_crowding solver_mode")
            if not 0.0 <= self.dc_mutation_fraction <= 1.0:
                raise ValueError("dc_mutation_fraction must be in [0.0, 1.0]")
            if self.hybrid_max_outer_restarts < 1:
                raise ValueError("hybrid_max_outer_restarts must be at least 1")
        elif self.solver_mode in ("hybrid_restart", "hybrid_targeted_restart", "hybrid_gene_fix", "cascading_pipeline", "hybrid_portfolio"):
            if self.nga_mode != "none":
                raise ValueError(f"nga_mode is not used in {self.solver_mode} solver_mode")
            if self.repeat_limit is not None:
                raise ValueError(f"repeat_limit is not used in {self.solver_mode} solver_mode")
            if self.generations < self.stagnation:
                raise ValueError(f"generations must be at least stagnation in {self.solver_mode} solver_mode")
            if self.hybrid_max_outer_restarts < 1:
                raise ValueError("hybrid_max_outer_restarts must be at least 1")
            if self.solver_mode == "hybrid_targeted_restart":
                if self.hybrid_targeted_k_min < 1 or self.hybrid_targeted_k_max < self.hybrid_targeted_k_min:
                    raise ValueError("hybrid_targeted_k_min must be >= 1 and k_max >= k_min")
            if self.solver_mode == "hybrid_gene_fix":
                if not 0.5 <= self.hybrid_gene_fix_threshold <= 1.0:
                    raise ValueError("hybrid_gene_fix_threshold must be in [0.5, 1.0]")
                if self.hybrid_gene_fix_invert_count < 1:
                    raise ValueError("hybrid_gene_fix_invert_count must be >= 1")


@dataclass(slots=True)
class GeneticAlgorithmResult:
    best_vector: list[int]
    best_sum: int
    fitness: int
    difference: int
    generations_used: int
    exact_match: bool
    stop_reason: str
    nga_used: bool = False
    nga_trigger_generation: int | None = None
    nga_trigger_generations: list[int] = field(default_factory=list)
    restart_count: int = 0
    rescue_used: bool = False
    stage1_run_count: int = 0
    stage2_run_count: int = 0
    stage2_used: bool = False
    stage1_best_differences: list[int] = field(default_factory=list)
    stage2_best_differences: list[int] = field(default_factory=list)
    stage_run_counts: list[int] = field(default_factory=list)
    stage_best_differences: list[int] = field(default_factory=list)
    final_stage: int = 0
