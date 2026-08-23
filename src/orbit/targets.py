"""Explicit project target selection."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .structure import SiteRecord


TARGET_FIELDS = (
    "target_order",
    "target_id",
    "site_id",
    "element",
    "ase_index_zero_based",
    "cif_label",
    "symmetry_orbit_id",
    "orbit_size",
    "selection_mode",
    "selected_at",
)


def select_targets(
    sites: list[SiteRecord],
    *,
    all_inequivalent: bool = False,
    elements: list[str] | None = None,
    site_ids: list[str] | None = None,
) -> tuple[list[SiteRecord], str]:
    modes = sum((all_inequivalent, elements is not None, site_ids is not None))
    if modes != 1:
        raise ValueError("choose exactly one target-selection mode")
    by_id = {site.site_id: site for site in sites}

    if site_ids is not None:
        missing = [site_id for site_id in site_ids if site_id not in by_id]
        if missing:
            raise ValueError(f"Unknown site IDs: {', '.join(missing)}")
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("Explicit site IDs contain duplicates")
        return [by_id[site_id] for site_id in site_ids], "explicit_sites"

    representatives = [site for site in sites if site.is_orbit_representative]
    if all_inequivalent:
        return sorted(representatives, key=lambda site: site.site_id), "all_inequivalent"

    assert elements is not None
    requested = list(dict.fromkeys(elements))
    available = {site.element for site in sites}
    missing_elements = [element for element in requested if element not in available]
    if missing_elements:
        raise ValueError(f"Elements not found in structure: {', '.join(missing_elements)}")
    chosen = [site for site in representatives if site.element in requested]
    chosen.sort(key=lambda site: (requested.index(site.element), site.site_id))
    return chosen, "inequivalent_elements"


def write_targets_manifest(path: Path, targets: list[SiteRecord], selection_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    selected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        for order, site in enumerate(targets):
            writer.writerow(
                {
                    "target_order": order,
                    "target_id": site.site_id,
                    "site_id": site.site_id,
                    "element": site.element,
                    "ase_index_zero_based": site.ase_index_zero_based,
                    "cif_label": site.cif_label,
                    "symmetry_orbit_id": site.symmetry_orbit_id,
                    "orbit_size": site.orbit_size,
                    "selection_mode": selection_mode,
                    "selected_at": selected_at,
                }
            )
    temporary.replace(path)
