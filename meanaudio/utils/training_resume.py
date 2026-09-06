"""Pure helpers for deterministic training resume coordinates."""

from __future__ import annotations


def resume_coordinates(completed_iterations: int, batches_per_epoch: int) -> tuple[int, int]:
    if completed_iterations < 0:
        raise ValueError("completed_iterations must be non-negative")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    return divmod(completed_iterations, batches_per_epoch)


def remaining_batch_indices(completed_iterations: int, batches_per_epoch: int) -> list[int]:
    """Indices consumed next in the current epoch after an exact resume."""
    _, offset = resume_coordinates(completed_iterations, batches_per_epoch)
    return list(range(offset, batches_per_epoch))
