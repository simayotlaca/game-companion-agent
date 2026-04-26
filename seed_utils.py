"""Centralized seed management for reproducibility.

Goal: same seed + same config → same game trajectory and agent decisions.

Covers:
    - Python built-in random
    - numpy (if installed)
    - torch CPU (if installed)
    - torch CUDA (if installed + deterministic_mode=True)

Usage:
    from seed_utils import apply_seed, SeedContext

    # Global seed at startup:
    apply_seed(42, deterministic=True)

    # Per-experiment isolated seed (does not leak into global state):
    with SeedContext(seed=42):
        result_a = run_experiment()
    with SeedContext(seed=42):
        result_b = run_experiment()
    assert result_a == result_b
"""

import random
from typing import Optional


def apply_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds globally for all active random sources.

    Args:
        seed: The integer seed to apply.
        deterministic: If True, also sets torch CUDA deterministic flags
            (disables some optimizations, may reduce GPU throughput).
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_seed_state() -> dict:
    """Capture current RNG state from all active sources.

    Returns a snapshot that can be restored with restore_seed_state().
    Useful for debugging: save state before a run, restore if you need
    to reproduce exactly the same sequence.
    """
    state: dict = {
        "python_random": random.getstate(),
    }

    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass

    try:
        import torch
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    return state


def restore_seed_state(state: dict) -> None:
    """Restore RNG state captured by get_seed_state().

    Args:
        state: Dict returned by get_seed_state().
    """
    random.setstate(state["python_random"])

    if "numpy" in state:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass

    if "torch_cpu" in state:
        try:
            import torch
            torch.set_rng_state(state["torch_cpu"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except ImportError:
            pass


class SeedContext:
    """Context manager for isolated, reproducible random sessions.

    Saves global RNG state on entry, applies a fresh seed, and
    restores the original state on exit. This means the seeded block
    does not affect randomness outside of it.

    Example:
        with SeedContext(seed=42):
            result = run_agent()
        # Global RNG state is unchanged after the block.
    """

    def __init__(self, seed: int, deterministic: bool = False) -> None:
        self.seed = seed
        self.deterministic = deterministic
        self._saved_state: Optional[dict] = None

    def __enter__(self) -> "SeedContext":
        self._saved_state = get_seed_state()
        apply_seed(self.seed, deterministic=self.deterministic)
        return self

    def __exit__(self, *_) -> None:
        if self._saved_state is not None:
            restore_seed_state(self._saved_state)
