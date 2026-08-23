"""Standalone interactive 3D gap heatmaps."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config import ProjectConfig
from .gaps import selected_targets
from .periodic import fractional_to_cartesian, minimum_image_distance, wrap_fractional
from .project import ProjectLayout
from .structure import load_site_records, validate_manifest_source
from .symmetry import structure_symmetry_operations, target_site_stabilizer


PLOT_SCHEMA_VERSION = 1

CPK_COLORS = {
    "H": "#FFFFFF",
    "C": "#505050",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Ti": "#BFC2C7",
    "Sr": "#00FF00",
    "Hf": "#4DC2FF",
    "Pb": "#575961",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "target"


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_complete_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Extracted gap table not found: {path}; run orbit extract first"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    complete = []
    for row in rows:
        status = row.get("output_status", "")
        counts[status] = counts.get(status, 0) + 1
        if status == "COMPLETE" and row.get("gap_ev", "") != "":
            complete.append(row)
    if not complete:
        raise ValueError(f"No COMPLETE gap rows are available in {path}")
    return complete, dict(sorted(counts.items()))


def _cell_edges(cell: Any) -> tuple[list[float | None], list[float | None], list[float | None]]:
    import numpy as np

    corners = np.asarray(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=float,
    )
    cart = corners @ np.asarray(cell, dtype=float)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for first in range(8):
        for second in range(first + 1, 8):
            if int(np.sum(np.abs(corners[first] - corners[second]))) != 1:
                continue
            for axis, values in enumerate((x, y, z)):
                values.extend([float(cart[first, axis]), float(cart[second, axis]), None])
    return x, y, z


def _distinct_stabilizer_images(
    frac: Any,
    stabilizer: Any,
    cell: Any,
    tolerance_angstrom: float = 1.0e-6,
) -> list[Any]:
    images = []
    for rotation, translation in zip(stabilizer.rotations, stabilizer.translations):
        image = wrap_fractional(rotation @ frac + translation)
        if not any(
            minimum_image_distance(image, old, cell) <= tolerance_angstrom
            for old in images
        ):
            images.append(image)
    return images


def _build_figure(
    config: ProjectConfig,
    target_id: str,
    rows: list[dict[str, str]],
    global_min: float,
    global_max: float,
    representatives_only: bool,
):
    try:
        import numpy as np
        import plotly.graph_objects as go
        from ase.data import atomic_numbers, covalent_radii
        from ase.io import read as ase_read
    except ImportError as exc:
        raise RuntimeError(
            "Interactive plotting requires Plotly, NumPy, and ASE. Reinstall "
            "ORBIT in the active environment with: python -m pip install -e PATH"
        ) from exc

    atoms = ase_read(config.structure)
    cell = np.asarray(atoms.cell.array, dtype=float)
    sites = load_site_records(config.root / "manifests" / "sites.csv")
    target_matches = [site for site in sites if site.site_id == target_id]
    if len(target_matches) != 1:
        raise ValueError(f"Expected one site record for {target_id}")
    target = target_matches[0]
    operations = structure_symmetry_operations(
        cell,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
        config.symprec_angstrom,
    )
    stabilizer = target_site_stabilizer(
        operations,
        np.asarray(target.frac, dtype=float),
        cell,
        config.symprec_angstrom,
    )

    representative_frac = np.asarray(
        [[float(row[axis]) for axis in ("frac_x", "frac_y", "frac_z")] for row in rows]
    )
    representative_cart = representative_frac @ cell
    gaps = np.asarray([float(row["gap_ev"]) for row in rows])
    custom = np.asarray(
        [
            [
                row["sample_name"],
                float(row["gap_ev"]),
                float(row["displacement_norm_A"]),
                int(float(row["orbit_size"])),
            ]
            for row in rows
        ],
        dtype=object,
    )
    traces = []
    edge_x, edge_y, edge_z = _cell_edges(cell)
    traces.append(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "#667085", "width": 4},
            name="unit cell",
            hoverinfo="skip",
        )
    )

    symbols = atoms.get_chemical_symbols()
    atom_cart = np.asarray(atoms.get_positions(), dtype=float)
    for element in sorted(set(symbols)):
        indices = [index for index, symbol in enumerate(symbols) if symbol == element]
        sizes = [
            max(7.0, min(18.0, 9.0 * float(covalent_radii[atomic_numbers[element]])))
            for _index in indices
        ]
        traces.append(
            go.Scatter3d(
                x=atom_cart[indices, 0],
                y=atom_cart[indices, 1],
                z=atom_cart[indices, 2],
                mode="markers",
                marker={
                    "size": sizes,
                    "color": CPK_COLORS.get(element, "#9AA0A6"),
                    "line": {"color": "#20242A", "width": 1},
                    "opacity": 1.0,
                },
                name=f"{element} atoms",
                text=[sites[index].site_id for index in indices],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if config.fold_by_symmetry and not representatives_only:
        equivalent_frac = []
        equivalent_gaps = []
        equivalent_custom = []
        for row, frac, gap in zip(rows, representative_frac, gaps):
            images = _distinct_stabilizer_images(frac, stabilizer, cell)
            for image in images:
                if minimum_image_distance(image, frac, cell) <= 1.0e-6:
                    continue
                equivalent_frac.append(image)
                equivalent_gaps.append(float(gap))
                equivalent_custom.append(
                    [row["sample_name"], float(gap), float(row["displacement_norm_A"])]
                )
        if equivalent_frac:
            equivalent_cart = fractional_to_cartesian(
                np.asarray(equivalent_frac), cell
            )
            traces.append(
                go.Scatter3d(
                    x=equivalent_cart[:, 0],
                    y=equivalent_cart[:, 1],
                    z=equivalent_cart[:, 2],
                    mode="markers",
                    marker={
                        "size": 4.5,
                        "symbol": "diamond",
                        "color": equivalent_gaps,
                        "colorscale": "RdBu",
                        "reversescale": True,
                        "cmin": global_min,
                        "cmax": global_max,
                        "cmid": 0.0,
                        "opacity": 0.38,
                        "showscale": False,
                    },
                    customdata=np.asarray(equivalent_custom, dtype=object),
                    name="symmetry images",
                    hovertemplate=(
                        "source=%{customdata[0]}<br>gap=%{customdata[1]:.6f} eV"
                        "<br>|displacement|=%{customdata[2]:.4f} Å"
                        "<extra>symmetry image</extra>"
                    ),
                )
            )

    traces.append(
        go.Scatter3d(
            x=representative_cart[:, 0],
            y=representative_cart[:, 1],
            z=representative_cart[:, 2],
            mode="markers",
            marker={
                "size": 5.5,
                "symbol": "circle",
                "color": gaps,
                "colorscale": "RdBu",
                "reversescale": True,
                "cmin": global_min,
                "cmax": global_max,
                "cmid": 0.0,
                "opacity": 0.92,
                "colorbar": {"title": "Gap (eV)", "thickness": 18},
                "line": {"color": "#20242A", "width": 0.5},
            },
            customdata=custom,
            name="calculated representatives",
            hovertemplate=(
                "sample=%{customdata[0]}<br>gap=%{customdata[1]:.6f} eV"
                "<br>|displacement|=%{customdata[2]:.4f} Å"
                "<br>orbit size=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    target_cart = np.asarray(target.frac) @ cell
    traces.append(
        go.Scatter3d(
            x=[target_cart[0]],
            y=[target_cart[1]],
            z=[target_cart[2]],
            mode="markers+text",
            marker={"size": 8, "symbol": "x", "color": "#FFD400", "line": {"width": 2}},
            text=[f"{target_id} equilibrium"],
            textposition="top center",
            name="target equilibrium",
            hovertemplate=f"{escape(target_id)} equilibrium<extra></extra>",
        )
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=f"{target_id}: sampled HOMO–LUMO gap",
        template="plotly_dark",
        paper_bgcolor="#0F1116",
        plot_bgcolor="#0F1116",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"orientation": "h", "y": -0.03},
        scene={
            "xaxis_title": "x (Å)",
            "yaxis_title": "y (Å)",
            "zaxis_title": "z (Å)",
            "aspectmode": "data",
        },
    )
    return figure, len(stabilizer)


def write_gap_heatmap(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
    output: Path | None = None,
    *,
    representatives_only: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write one standalone tabbed HTML heatmap and a provenance manifest."""
    try:
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError(
            "Interactive plotting requires Plotly. Reinstall ORBIT in the active "
            "environment with: python -m pip install -e PATH"
        ) from exc

    validate_manifest_source(layout.structure_manifest, config.structure)
    targets = selected_targets(config, requested)
    datasets: list[tuple[str, list[dict[str, str]], dict[str, int], Path]] = []
    all_gaps: list[float] = []
    for target_id in targets:
        gaps_path = config.root / "results" / "gaps" / target_id / "gaps.csv"
        rows, counts = _load_complete_rows(gaps_path)
        datasets.append((target_id, rows, counts, gaps_path))
        all_gaps.extend(float(row["gap_ev"]) for row in rows)
    global_min = min(0.0, min(all_gaps))
    global_max = max(0.0, max(all_gaps))
    if abs(global_max - global_min) < 1.0e-12:
        global_max = global_min + 1.0e-6

    figures = []
    target_metadata = []
    for target_id, rows, counts, gaps_path in datasets:
        figure, stabilizer_order = _build_figure(
            config,
            target_id,
            rows,
            global_min,
            global_max,
            representatives_only,
        )
        figures.append(
            pio.to_html(
                figure,
                include_plotlyjs=True if not figures else False,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
        )
        target_metadata.append(
            {
                "target_id": target_id,
                "complete_count": len(rows),
                "status_counts": counts,
                "site_stabilizer_order": stabilizer_order,
                "gaps_file": str(gaps_path.relative_to(config.root)),
                "gaps_file_sha256": _sha256_file(gaps_path),
            }
        )

    if output is None:
        filename = (
            f"gap_heatmap_{_safe_name(targets[0])}.html"
            if len(targets) == 1
            else "gap_heatmaps.html"
        )
        output_path = config.root / "plots" / filename
    else:
        output_path = output.expanduser()
        if not output_path.is_absolute():
            output_path = config.root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tabs = "".join(
        f'<button class="tab {"active" if index == 0 else ""}" '
        f'onclick="showTab({index}, this)">{escape(target)}</button>'
        for index, target in enumerate(targets)
    )
    panels = "".join(
        f'<section class="panel" style="display:{"block" if index == 0 else "none"}">'
        f'<p class="summary">{len(datasets[index][1])} complete calculations; '
        f'status counts: {escape(json.dumps(datasets[index][2], sort_keys=True))}</p>'
        f'{figure}</section>'
        for index, figure in enumerate(figures)
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORBIT gap heatmaps</title>
<style>
body{{margin:0;background:#0F1116;color:#F2F4F7;font-family:Inter,Segoe UI,sans-serif}}
.header{{padding:18px 22px 8px}} h1{{font-size:22px;margin:0 0 10px}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap}} .tab{{border:1px solid #475467;background:#1D2939;color:#EAECF0;padding:8px 14px;border-radius:7px;cursor:pointer}}
.tab.active{{background:#1570EF;border-color:#53B1FD}} .summary{{margin:4px 22px;color:#98A2B3;font-size:13px}}
.panel{{min-height:760px}}
</style></head><body>
<div class="header"><h1>ORBIT sampled-gap heatmaps</h1><div class="tabs">{tabs}</div></div>
{panels}
<script>
function showTab(index, button){{
  document.querySelectorAll('.panel').forEach((p,i)=>p.style.display=i===index?'block':'none');
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  button.classList.add('active');
  setTimeout(()=>window.dispatchEvent(new Event('resize')),30);
}}
</script></body></html>
"""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output_path)

    manifest_path = output_path.with_suffix(".json")
    payload = {
        "schema_version": PLOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "targets": target_metadata,
        "representatives_only": representatives_only,
        "shared_color_range_eV": [global_min, global_max],
        "html_file": _display_path(output_path, config.root),
        "html_file_sha256": _sha256_file(output_path),
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return output_path, manifest_path, payload
