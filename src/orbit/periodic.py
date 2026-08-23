"""Periodic-coordinate operations shared by sampling, paths, and plotting."""

from __future__ import annotations

from typing import Any


def wrap_fractional(frac: Any):
    """Wrap fractional coordinate(s) into the half-open unit cell [0, 1)."""
    import numpy as np

    wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=1.0e-12)] = 0.0
    return wrapped


def fractional_to_cartesian(frac: Any, cell: Any):
    """Convert row-vector fractional coordinates using `cart = frac @ cell`."""
    import numpy as np

    return np.asarray(frac, dtype=float) @ np.asarray(cell, dtype=float)


def minimum_image_displacement(frac_a: Any, frac_b: Any, cell: Any):
    """Shortest Cartesian displacement from periodic point b to point a."""
    import numpy as np
    from ase.geometry import find_mic

    displacement = (
        np.asarray(frac_a, dtype=float) - np.asarray(frac_b, dtype=float)
    ) @ np.asarray(cell, dtype=float)
    mic, _distance = find_mic(displacement, cell=cell, pbc=True)
    return np.asarray(mic, dtype=float)


def minimum_image_distance(frac_a: Any, frac_b: Any, cell: Any) -> float:
    """Shortest periodic Cartesian distance in angstrom for a general cell."""
    import numpy as np

    return float(np.linalg.norm(minimum_image_displacement(frac_a, frac_b, cell)))

