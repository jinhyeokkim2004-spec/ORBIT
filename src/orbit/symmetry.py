"""Space-group and target-site-stabilizer operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .periodic import minimum_image_distance, wrap_fractional


@dataclass(frozen=True)
class SymmetryOperations:
    rotations: Any
    translations: Any

    def __len__(self) -> int:
        return len(self.rotations)


def structure_symmetry_operations(cell: Any, frac: Any, numbers: Any, symprec: float) -> SymmetryOperations:
    import numpy as np
    import spglib

    symmetry = spglib.get_symmetry((cell, frac, numbers), symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib could not determine structure symmetry operations")
    return SymmetryOperations(
        rotations=np.asarray(symmetry["rotations"], dtype=int),
        translations=wrap_fractional(symmetry["translations"]),
    )


def target_site_stabilizer(
    operations: SymmetryOperations,
    target_frac: Any,
    cell: Any,
    tolerance_angstrom: float,
) -> SymmetryOperations:
    """Return distinct space-group operations that fix the target modulo PBC."""
    import numpy as np

    target = wrap_fractional(target_frac)
    rotations = []
    translations = []
    for rotation, translation in zip(operations.rotations, operations.translations):
        mapped = wrap_fractional(rotation @ target + translation)
        if minimum_image_distance(mapped, target, cell) > tolerance_angstrom:
            continue
        duplicate = any(
            np.array_equal(rotation, old_rotation)
            and minimum_image_distance(translation, old_translation, cell)
            <= 1.0e-10
            for old_rotation, old_translation in zip(rotations, translations)
        )
        if not duplicate:
            rotations.append(rotation)
            translations.append(translation)
    if not rotations:
        raise RuntimeError(
            "No symmetry operation fixes the target; the identity should be present"
        )
    return SymmetryOperations(
        rotations=np.asarray(rotations, dtype=int),
        translations=np.asarray(translations, dtype=float),
    )

