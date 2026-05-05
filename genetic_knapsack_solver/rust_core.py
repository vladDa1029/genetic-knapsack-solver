from __future__ import annotations

import ctypes
import json
import os
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Any

from genetic_knapsack_solver.models import GeneticAlgorithmConfig, GeneticAlgorithmResult


class RustCoreUnavailable(RuntimeError):
    """Raised when the compiled Rust core library cannot be loaded."""


def _library_filename() -> str:
    if sys.platform.startswith("win"):
        return "genetic_knapsack_solver_core.dll"
    if sys.platform == "darwin":
        return "libgenetic_knapsack_solver_core.dylib"
    return "libgenetic_knapsack_solver_core.so"


def _candidate_library_paths() -> list[Path]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    library_filename = _library_filename()
    paths: list[Path] = []
    if env_path := os.environ.get("GENETIC_KNAPSACK_RUST_CORE_DLL"):
        paths.append(Path(env_path))
    paths.extend(
        [
            project_root / "rust_core" / "target" / "release" / library_filename,
            project_root / "rust_core" / "target" / "debug" / library_filename,
        ]
    )
    return paths


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    attempted_paths = _candidate_library_paths()
    for path in attempted_paths:
        if not path.exists():
            continue
        library = ctypes.CDLL(str(path))
        library.solve_json.argtypes = [ctypes.c_char_p]
        library.solve_json.restype = ctypes.c_void_p
        library.free_rust_string.argtypes = [ctypes.c_void_p]
        library.free_rust_string.restype = None
        return library

    attempted = ", ".join(str(path) for path in attempted_paths)
    project_root = Path(__file__).resolve().parent.parent
    raise RustCoreUnavailable(
        "Rust core library was not found. Build it with "
        f"`cargo build --release --manifest-path {project_root / 'rust_core' / 'Cargo.toml'}`. "
        f"Checked: {attempted}"
    )


def rust_core_available() -> bool:
    try:
        _load_library()
    except RustCoreUnavailable:
        return False
    return True


def _call_rust_core(payload: dict[str, Any]) -> dict[str, Any]:
    library = _load_library()
    encoded_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    pointer = library.solve_json(encoded_payload)
    if not pointer:
        raise RuntimeError("Rust core returned a null pointer")

    try:
        raw_response = ctypes.string_at(pointer).decode("utf-8")
    finally:
        library.free_rust_string(pointer)

    response = json.loads(raw_response)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def solve_with_rust_core(
    prices: list[int],
    target_sum: int,
    config: GeneticAlgorithmConfig | None = None,
    *,
    seed: int = 0,
    rng: Random | None = None,
) -> GeneticAlgorithmResult:
    """Run the genetic solver in the compiled Rust core.

    The Rust implementation uses its own deterministic PRNG. Pass `seed` for
    reproducible Rust runs, or pass a Python `Random` instance to derive a seed.
    """

    if config is None:
        config = GeneticAlgorithmConfig()
    if rng is not None:
        seed = rng.getrandbits(64)

    response = _call_rust_core(
        {
            "prices": prices,
            "target_sum": target_sum,
            "seed": seed,
            "config": asdict(config),
        }
    )
    return GeneticAlgorithmResult(**response)
