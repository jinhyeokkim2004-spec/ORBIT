"""Reference-style sampled heatmap and calculated-path viewer."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any

from .config import ProjectConfig
from .gaps import selected_targets
from .periodic import minimum_image_distance, wrap_fractional
from .plotting import CPK_COLORS, _load_complete_rows
from .project import ProjectLayout
from .qe.path_calculations import resolve_path_run
from .structure import load_site_records, validate_manifest_source
from .symmetry import structure_symmetry_operations, target_site_stabilizer


PATH_PLOT_SCHEMA_VERSION = 2
ATOM_STYLE = {
    "O": {"color": "#D62728", "radius": 0.34},
    "H": {"color": "#D9D9D9", "radius": 0.24},
}
FALLBACK_ATOM_STYLE = {"color": "#9AA0A6", "radius": 0.40}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "target"


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _formula(formula: str) -> str:
    digits = "₀₁₂₃₄₅₆₇₈₉"
    return re.sub(r"\d+", lambda match: "".join(digits[int(d)] for d in match.group()), formula)


def _color_spec(gaps: list[float]) -> dict[str, Any]:
    magnitude = max([abs(value) for value in gaps] + [1.0e-6])
    return {
        "colorscale": "RdBu",
        "reversescale": True,
        "cmin": -magnitude,
        "cmid": 0.0,
        "cmax": magnitude,
    }


def _cell_edges(cell: Any):
    import numpy as np

    corners = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)
    cart = corners @ np.asarray(cell, dtype=float)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for first in range(8):
        for second in range(first + 1, 8):
            if int(np.abs(corners[first] - corners[second]).sum()) != 1:
                continue
            x.extend([float(cart[first, 0]), float(cart[second, 0]), None])
            y.extend([float(cart[first, 1]), float(cart[second, 1]), None])
            z.extend([float(cart[first, 2]), float(cart[second, 2]), None])
    return x, y, z


def _boundary_images(frac: Any, tolerance: float = 1.0e-6):
    import numpy as np

    wrapped = wrap_fractional(frac)
    choices: list[list[float]] = []
    for value in wrapped:
        shifts = [0.0]
        if np.isclose(value, 0.0, atol=tolerance):
            shifts.append(1.0)
        elif np.isclose(value, 1.0, atol=tolerance):
            shifts.append(-1.0)
        choices.append(shifts)
    return [wrapped + np.asarray(shift) for shift in itertools.product(*choices)]


def _sphere_arrays(center: Any, radius: float, *, n: int = 16):
    import numpy as np

    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)
    return x.ravel(), y.ravel(), z.ravel()


def _sphere_mesh(center: Any, radius: float, color: str, name: str, *, n: int = 16):
    import plotly.graph_objects as go

    x, y, z = _sphere_arrays(center, radius, n=n)
    return go.Mesh3d(
        x=x, y=y, z=z, alphahull=0, color=color, opacity=1.0,
        flatshading=False, name=name, hovertext=name, hoverinfo="text",
        lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.25, "roughness": 0.5},
        showscale=False,
    )


def _atom_style(symbol: str) -> dict[str, Any]:
    if symbol in ATOM_STYLE:
        return ATOM_STYLE[symbol]
    return {
        "color": CPK_COLORS.get(symbol, FALLBACK_ATOM_STYLE["color"]),
        "radius": FALLBACK_ATOM_STYLE["radius"],
    }


def _framework_traces(atoms: Any, sites: Any, target_index: int):
    """Draw atom spheres and periodic copies shared by cell boundaries."""
    import numpy as np
    import plotly.graph_objects as go

    cell = np.asarray(atoms.cell.array, dtype=float)
    traces: list[Any] = []
    target_local_indices: list[int] = []
    for index, (symbol, frac) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=True))
    ):
        style = _atom_style(symbol)
        label = sites[index].site_id
        if index == target_index:
            label += " — target atom at equilibrium"
        for copy_number, image in enumerate(_boundary_images(frac)):
            copy_label = label if copy_number == 0 else label + " — periodic boundary copy"
            traces.append(
                _sphere_mesh(image @ cell, float(style["radius"]), str(style["color"]), copy_label)
            )
            if index == target_index:
                target_local_indices.append(len(traces))

    edge_x, edge_y, edge_z = _cell_edges(cell)
    traces.insert(0, go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line={"color": "#34495E", "width": 3}, name="unit cell", hoverinfo="skip",
    ))
    return traces, target_local_indices


def _distinct_images(frac: Any, stabilizer: Any, cell: Any) -> list[Any]:
    images = []
    for rotation, translation in zip(stabilizer.rotations, stabilizer.translations):
        image = wrap_fractional(rotation @ wrap_fractional(frac) + translation)
        if not any(minimum_image_distance(image, old, cell) <= 1.0e-6 for old in images):
            images.append(image)
    return images


def _split_wrapped_path(wrapped: Any, cell: Any):
    import numpy as np

    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for index, frac in enumerate(wrapped):
        cart = np.asarray(frac) @ cell
        if index and np.any(np.abs(frac - wrapped[index - 1]) > 0.5):
            x.append(None)
            y.append(None)
            z.append(None)
        x.append(float(cart[0]))
        y.append(float(cart[1]))
        z.append(float(cart[2]))
    return x, y, z


def _destination_label(destination: str) -> str:
    return "".join(f"+{axis}" for axis in "abc" if axis in destination) or destination


def _build_figure(
    config: ProjectConfig,
    target_id: str,
    sample_rows: list[dict[str, str]],
    path_rows: list[dict[str, str]],
    destination: str,
    colors: dict[str, Any],
    representatives_only: bool,
):
    import numpy as np
    import plotly.graph_objects as go
    from ase.io import read as ase_read

    atoms = ase_read(config.structure)
    cell = np.asarray(atoms.cell.array, dtype=float)
    sites = load_site_records(config.root / "manifests" / "sites.csv")
    matches = [site for site in sites if site.site_id == target_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one site record for {target_id}")
    target = matches[0]
    target_index = target.ase_index_zero_based

    operations = structure_symmetry_operations(
        cell, atoms.get_scaled_positions(wrap=True), atoms.get_atomic_numbers(),
        config.symprec_angstrom,
    )
    stabilizer = target_site_stabilizer(
        operations, np.asarray(target.frac, dtype=float), cell, config.symprec_angstrom,
    )

    representative_frac = np.asarray([
        [float(row[f"frac_{axis}"]) for axis in "xyz"] for row in sample_rows
    ])
    representative_cart = representative_frac @ cell
    representative_gaps = np.asarray([float(row["gap_ev"]) for row in sample_rows])
    sample_hover = [
        f"{escape(row['sample_name'])}<br>gap = {float(row['gap_ev']):+.3f} eV"
        f"<br>wrapped fractional = ({float(row['frac_x']):.5f}, "
        f"{float(row['frac_y']):.5f}, {float(row['frac_z']):.5f})"
        f"<br>displacement = {float(row['displacement_norm_A']):.4f} Å"
        for row in sample_rows
    ]
    traces: list[Any] = [go.Scatter3d(
        x=representative_cart[:, 0], y=representative_cart[:, 1], z=representative_cart[:, 2],
        mode="markers",
        marker={
            "size": 5, "color": representative_gaps, "opacity": 0.92,
            "symbol": "circle", "line": {"width": 0.4, "color": "#222222"},
            "showscale": True, "colorbar": {"title": "Egap (eV)", "thickness": 16, "len": 0.70},
            **colors,
        },
        text=sample_hover, hoverinfo="text", name="calculated representatives",
    )]
    sample_trace_index = 0

    equivalent_frac: list[Any] = []
    equivalent_gaps: list[float] = []
    for row, frac, gap in zip(sample_rows, representative_frac, representative_gaps):
        orbit_size = int(float(row.get("orbit_size", "0") or 0))
        images = [wrap_fractional(frac)] if orbit_size == 1 else _distinct_images(frac, stabilizer, cell)
        equivalent_frac.extend(images)
        equivalent_gaps.extend([float(gap)] * len(images))
    equivalent = np.asarray(equivalent_frac)
    equivalent_cart = equivalent @ cell
    traces.append(go.Scatter3d(
        x=equivalent_cart[:, 0], y=equivalent_cart[:, 1], z=equivalent_cart[:, 2],
        mode="markers",
        marker={
            "size": 4, "color": equivalent_gaps, "opacity": 0.30,
            "symbol": "diamond", "line": {"width": 0}, "showscale": False, **colors,
        },
        text=[f"symmetry-equivalent image<br>gap = {gap:+.3f} eV" for gap in equivalent_gaps],
        hoverinfo="text", name="wrapped symmetry equivalents",
    ))
    symmetry_trace_index = 1

    framework_start = len(traces)
    framework, target_local_indices = _framework_traces(atoms, sites, target_index)
    traces.extend(framework)
    target_framework_indices = [framework_start + index for index in target_local_indices]

    wrapped = np.asarray([
        [float(row[f"wrapped_frac_{axis}"]) for axis in "xyz"] for row in path_rows
    ])
    unwrapped = np.asarray([
        [float(row[f"unwrapped_frac_{axis}"]) for axis in "xyz"] for row in path_rows
    ])
    fractions = np.asarray([float(row.get("path_fraction", 0.0)) for row in path_rows])
    names = np.asarray([row["image_name"] for row in path_rows], dtype=str)
    actual = np.asarray([
        float(row["gap_eV"])
        if row.get("output_status") == "COMPLETE" and row.get("gap_eV", "") else np.nan
        for row in path_rows
    ])
    proxy = np.asarray([
        float(row["gap_proxy_eV"]) if row.get("gap_proxy_eV", "") else np.nan
        for row in path_rows
    ])

    path_x, path_y, path_z = _split_wrapped_path(wrapped, cell)
    path_hover: list[str | None] = []
    for index, (name, fraction, frac) in enumerate(zip(names, fractions, wrapped)):
        if index and np.any(np.abs(frac - wrapped[index - 1]) > 0.5):
            path_hover.append(None)
        path_hover.append(
            f"{escape(str(name))}<br>path fraction = {fraction:.4f}"
            f"<br>wrapped fractional = ({frac[0]:.6f}, {frac[1]:.6f}, {frac[2]:.6f})"
        )
    traces.append(go.Scatter3d(
        x=path_x, y=path_y, z=path_z, mode="lines+markers",
        line={"color": "#7B2CBF", "width": 5, "dash": "dot"},
        marker={
            "size": 3.5, "color": "#7B2CBF", "symbol": "circle",
            "line": {"width": 0.5, "color": "#4A176B"},
        },
        name="CIF path trajectory", text=path_hover, hoverinfo="text", visible=False,
    ))
    path_trace_index = len(traces) - 1

    target_style = _atom_style(atoms.get_chemical_symbols()[target_index])
    moving_x, moving_y, moving_z = _sphere_arrays(
        wrapped[0] @ cell, float(target_style["radius"]) * 1.15, n=18,
    )
    traces.append(go.Mesh3d(
        x=moving_x, y=moving_y, z=moving_z, alphahull=0, color="#FFD166",
        opacity=1.0, flatshading=False, name=f"moving {target_id}",
        hovertext=f"{escape(str(names[0]))}<br>path fraction = {fractions[0]:.4f}",
        hoverinfo="text",
        lighting={"ambient": 0.50, "diffuse": 0.85, "specular": 0.45, "roughness": 0.35},
        showscale=False, visible=False,
    ))
    moving_trace_index = len(traces) - 1

    finite = np.flatnonzero(np.isfinite(actual))
    if len(finite):
        gap_min = float(actual[finite].min())
        gap_max = float(actual[finite].max())
        span = max(gap_max - gap_min, 0.05)
        gap_y_range = (gap_min - 0.10 * span, gap_max + 0.10 * span)
    else:
        gap_y_range = (0.0, 1.0)

    steps = np.arange(len(names))
    traces.append(go.Scatter(
        x=steps, y=actual, mode="lines+markers",
        line={"color": "#7B2CBF", "width": 2.5}, marker={"size": 6, "color": "#7B2CBF"},
        name="calculated path E<sub>gap</sub>", customdata=names,
        hovertemplate="%{customdata}<br>step = %{x}<br>E<sub>gap</sub> = %{y:.4f} eV<extra></extra>",
        connectgaps=False, visible=False,
    ))
    gap_trace_index = len(traces) - 1

    gap_minimum_index = None
    if len(finite):
        minimum_step = int(finite[np.argmin(actual[finite])])
        minimum_gap = float(actual[minimum_step])
        traces.append(go.Scatter(
            x=[minimum_step], y=[minimum_gap], mode="markers+text",
            marker={
                "size": 14, "symbol": "star", "color": "#D62728",
                "line": {"width": 1.5, "color": "#7F1D1D"},
            },
            text=[f"minimum {minimum_gap:.4f} eV · step {minimum_step}"],
            textposition="top right" if minimum_step < len(names) / 2 else "top left",
            textfont={"color": "#7F1D1D", "size": 13}, name="minimum path gap", visible=False,
        ))
        gap_minimum_index = len(traces) - 1

    traces.append(go.Scatter(
        x=[0], y=[actual[0]], mode="markers+text",
        marker={"size": 13, "color": "#FFD166", "line": {"width": 2, "color": "#5B3A00"}},
        text=[f"{actual[0]:.4f} eV" if np.isfinite(actual[0]) else "gap unavailable"],
        textposition="bottom center", textfont={"color": "#5B3A00", "size": 13},
        name="selected path step", visible=False,
    ))
    gap_marker_index = len(traces) - 1
    traces.append(go.Scatter(
        x=[0, 0], y=list(gap_y_range), mode="lines",
        line={"color": "#FFD166", "width": 2}, name="current step",
        hoverinfo="skip", showlegend=False, visible=False,
    ))
    gap_cursor_index = len(traces) - 1

    frames: list[Any] = []
    slider_steps: list[dict[str, Any]] = []
    for index, (wrapped_frac, unwrapped_frac, fraction, name, proxy_gap) in enumerate(
        zip(wrapped, unwrapped, fractions, names, proxy)
    ):
        x, y, z = _sphere_arrays(
            wrapped_frac @ cell, float(target_style["radius"]) * 1.15, n=18,
        )
        hover = (
            f"{escape(str(name))}<br>path fraction = {fraction:.4f}"
            f"<br>wrapped fractional = ({wrapped_frac[0]:.6f}, {wrapped_frac[1]:.6f}, {wrapped_frac[2]:.6f})"
            f"<br>unwrapped fractional = ({unwrapped_frac[0]:.6f}, {unwrapped_frac[1]:.6f}, {unwrapped_frac[2]:.6f})"
        )
        if np.isfinite(proxy_gap):
            hover += f"<br>proxy gap = {proxy_gap:.4f} eV<br><i>proxy, not an SCF result for this image</i>"
        hover += (
            f"<br>calculated gap = {actual[index]:.4f} eV"
            if np.isfinite(actual[index]) else "<br>calculated gap = unavailable"
        )
        frame_name = f"{target_id}_path_{index:03d}"
        frames.append(go.Frame(
            name=frame_name,
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z, alphahull=0, color="#FFD166", opacity=1.0,
                    flatshading=False, hovertext=hover, hoverinfo="text",
                    lighting={"ambient": 0.50, "diffuse": 0.85, "specular": 0.45, "roughness": 0.35},
                    showscale=False,
                ),
                go.Scatter(
                    x=[index], y=[actual[index]], mode="markers+text",
                    marker={"size": 13, "color": "#FFD166", "line": {"width": 2, "color": "#5B3A00"}},
                    text=[f"{actual[index]:.4f} eV" if np.isfinite(actual[index]) else "gap unavailable"],
                    textposition="bottom center", textfont={"color": "#5B3A00", "size": 13},
                ),
                go.Scatter(
                    x=[index, index], y=list(gap_y_range), mode="lines",
                    line={"color": "#FFD166", "width": 2}, hoverinfo="skip", showlegend=False,
                ),
            ],
            traces=[moving_trace_index, gap_marker_index, gap_cursor_index],
        ))
        slider_steps.append({
            "method": "animate",
            "label": str(index) if index % 5 == 0 or index == len(names) - 1 else "",
            "args": [[frame_name], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        })

    figure = go.Figure(data=traces, frames=frames)
    formula = _formula(atoms.get_chemical_formula())
    suffix = "" if representatives_only else " (symmetry-filled)"
    title_heatmap = f"{formula} — HOMO–LUMO gap vs. {target_id} position{suffix}"
    title_path = (
        f"{formula} — {target_id} displacement path to {_destination_label(destination)} "
        f"({len(names)} images)"
    )

    trace_count = len(traces)
    heatmap_view = [True] * trace_count
    heatmap_view[symmetry_trace_index] = not representatives_only
    for index in (path_trace_index, moving_trace_index, gap_trace_index, gap_marker_index, gap_cursor_index):
        heatmap_view[index] = False
    if gap_minimum_index is not None:
        heatmap_view[gap_minimum_index] = False
    for trace, visible in zip(figure.data, heatmap_view):
        trace.visible = visible

    path_heatmap_on = [True] * trace_count
    path_heatmap_on[symmetry_trace_index] = not representatives_only
    for index in target_framework_indices:
        path_heatmap_on[index] = False
    path_heatmap_off = path_heatmap_on.copy()
    path_heatmap_off[sample_trace_index] = False
    path_heatmap_off[symmetry_trace_index] = False

    updatemenus = [
        {
            "type": "buttons", "direction": "right", "x": 0.01, "y": 1.18,
            "xanchor": "left", "yanchor": "top", "showactive": True, "active": 0,
            "buttons": [
                {"label": "HEATMAP", "method": "update", "args": [
                    {"visible": heatmap_view},
                    {"title.text": title_heatmap, "sliders[0].visible": False,
                     "updatemenus[1].visible": False, "updatemenus[2].visible": False,
                     "scene.domain.y": [0.0, 1.0], "xaxis.visible": False, "yaxis.visible": False},
                ]},
                {"label": "PATH", "method": "update", "args": [
                    {"visible": path_heatmap_on},
                    {"title.text": title_path, "sliders[0].visible": True,
                     "updatemenus[1].visible": True, "updatemenus[2].visible": True,
                     "scene.domain.y": [0.36, 1.0], "xaxis.visible": True, "yaxis.visible": True},
                ]},
            ],
        },
        {
            "type": "buttons", "direction": "right", "x": 0.01, "y": 1.10,
            "xanchor": "left", "yanchor": "top", "showactive": True,
            "active": 0, "visible": False,
            "buttons": [
                {"label": "Heatmap ON", "method": "update", "args": [
                    {"visible": path_heatmap_on}, {"title.text": title_path},
                ]},
                {"label": "Heatmap OFF", "method": "update", "args": [
                    {"visible": path_heatmap_off}, {"title.text": title_path},
                ]},
            ],
        },
        {
            "type": "buttons", "direction": "right", "x": 0.27, "y": 1.10,
            "xanchor": "left", "yanchor": "top", "showactive": False, "visible": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {
                    "fromcurrent": True, "mode": "immediate",
                    "frame": {"duration": 450, "redraw": True}, "transition": {"duration": 0},
                }]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {
                    "mode": "immediate", "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                }]},
            ],
        },
    ]
    sliders = [{
        "active": 0, "visible": False, "x": 0.08, "len": 0.84, "y": -0.02,
        "xanchor": "left", "yanchor": "top", "pad": {"t": 45, "b": 0},
        "currentvalue": {
            "prefix": "Path step: ", "visible": True, "xanchor": "center", "font": {"size": 15},
        },
        "transition": {"duration": 0}, "steps": slider_steps,
    }]
    figure.update_layout(
        template="plotly_white",
        title={"text": title_heatmap, "x": 0.5, "xanchor": "center", "y": 0.99, "yanchor": "top"},
        scene={
            "xaxis_title": "x (Å)", "yaxis_title": "y (Å)", "zaxis_title": "z (Å)",
            "aspectmode": "data",
            "xaxis": {"backgroundcolor": "#FAFBFC"},
            "yaxis": {"backgroundcolor": "#FAFBFC"},
            "zaxis": {"backgroundcolor": "#FAFBFC"},
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        },
        xaxis={
            "title": "Path step", "domain": [0.08, 0.92], "range": [-1.5, len(names) + 0.5],
            "dtick": 5, "visible": False, "showgrid": True, "gridcolor": "#D9E0E7",
        },
        yaxis={
            "title": "E<sub>gap</sub> (eV)", "domain": [0.02, 0.25],
            "range": list(gap_y_range), "visible": False, "showgrid": True, "gridcolor": "#D9E0E7",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.27, "x": 0.0},
        margin={"l": 0, "r": 0, "t": 155, "b": 105},
        updatemenus=updatemenus, sliders=sliders, uirevision=f"{target_id}-camera",
    )
    summary = {
        "complete_count": int(len(finite)),
        "image_count": len(path_rows),
        "minimum_gap_eV": None if not len(finite) else float(actual[finite].min()),
        "minimum_gap_image": None if not len(finite) else int(finite[np.argmin(actual[finite])]),
        "site_stabilizer_order": len(stabilizer),
        "atom_sphere_count": len(framework) - 1,
        "target_periodic_sphere_count": len(target_framework_indices),
    }
    return figure, summary


def write_path_viewer(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
    output: Path | None,
    *,
    run_id: str | None,
    representatives_only: bool = False,
):
    import plotly.io as pio

    validate_manifest_source(layout.structure_manifest, config.structure)
    targets = selected_targets(config, requested)
    if run_id is not None and len(targets) != 1:
        raise ValueError("--run-id requires exactly one --target")

    datasets = []
    all_sample_gaps: list[float] = []
    for target_id in targets:
        sample_rows, sample_counts = _load_complete_rows(
            config.root / "results" / "gaps" / target_id / "gaps.csv"
        )
        run_dir = resolve_path_run(layout, target_id, run_id if len(targets) == 1 else None)
        resolved_run = run_dir.name
        path_json = _read_json(run_dir / "path.json", "Selected-path manifest")
        path_gap_root = config.root / "results" / "path_gaps" / target_id / resolved_run
        extraction = _read_json(path_gap_root / "extraction.json", "Path-gap extraction manifest")
        gap_path = path_gap_root / "gaps.csv"
        if extraction.get("gaps_file_sha256") != _sha256(gap_path):
            raise ValueError("Path-gap table changed after extraction")
        path_rows = _read_csv(gap_path, "Path-gap table")
        datasets.append(
            (target_id, sample_rows, sample_counts, resolved_run, path_json, path_rows, gap_path)
        )
        all_sample_gaps.extend(float(row["gap_ev"]) for row in sample_rows)
    colors = _color_spec(all_sample_gaps)

    panels: list[str] = []
    metadata: list[dict[str, Any]] = []
    included_js = False
    for index, dataset in enumerate(datasets):
        target_id, sample_rows, counts, resolved_run, path_json, path_rows, gap_path = dataset
        figure, summary = _build_figure(
            config, target_id, sample_rows, path_rows,
            str(path_json["destination_id"]), colors, representatives_only,
        )
        div_id = f"gap-viewer-{_safe(target_id).lower()}"
        plot_html = pio.to_html(
            figure, include_plotlyjs=not included_js, full_html=False, div_id=div_id,
            auto_play=False,
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        )
        included_js = True
        panels.append(
            f'<section id="{div_id}-panel" class="tab-panel{" active" if index == 0 else ""}">'
            f"{plot_html}</section>"
        )
        metadata.append({
            "target_id": target_id,
            "path_run_id": resolved_run,
            "destination_id": path_json["destination_id"],
            "sample_status_counts": counts,
            "path_gaps_file": str(gap_path.relative_to(config.root)),
            **summary,
        })

    if output is None:
        name = (
            f"gap_path_viewer_{_safe(targets[0])}.html"
            if len(targets) == 1 else "gap_path_viewers.html"
        )
        output_path = config.root / "plots" / name
    else:
        output_path = output.expanduser()
        if not output_path.is_absolute():
            output_path = config.root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buttons = "".join(
        f'<button class="tab-button{" active" if index == 0 else ""}" '
        f'onclick="showViewer(event, \'gap-viewer-{_safe(target).lower()}-panel\')">'
        f"{escape(target)}</button>"
        for index, target in enumerate(targets)
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ORBIT heatmap and path viewers</title>
<style>
:root{{color-scheme:light;--page-bg:#f4f6f8;--text:#1f2933;--panel:#fff;--border:#d9e0e7;--accent:#244b74;--hover:#eef3f7;--shadow:rgba(31,41,51,.08)}}
:root[data-theme="dark"]{{color-scheme:dark;--page-bg:#0f1116;--text:#f2f4f7;--panel:#171b24;--border:#475467;--accent:#53b1fd;--hover:#293241;--shadow:rgba(0,0,0,.35)}}
html,body{{margin:0;min-height:100%;background:var(--page-bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif;transition:background-color .18s ease,color .18s ease}}
.page{{max-width:1500px;margin:0 auto;padding:18px}}
h1{{font-size:22px;margin:0}}
.toolbar{{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 14px}}
.theme-controls{{display:flex;gap:6px;flex:0 0 auto}}
.theme-button{{border:1px solid var(--border);background:var(--panel);color:var(--text);padding:7px 12px;border-radius:8px;cursor:pointer;font-weight:600}}
.theme-button:hover{{background:var(--hover)}}
.theme-button.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap}}
.tab-button{{border:1px solid var(--border);background:var(--panel);color:var(--accent);padding:9px 18px;border-radius:12px 12px 0 0;cursor:pointer;font-weight:600}}
.tab-button:hover{{background:var(--hover)}}
.tab-button.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.tab-panel{{display:none;background:var(--panel);border:1px solid var(--border);border-radius:0 12px 12px 12px;box-shadow:0 2px 8px var(--shadow);min-height:760px}}
.tab-panel.active{{display:block}}
.plotly-graph-div{{height:min(82vh,900px)!important;min-height:700px}}
:root[data-theme="dark"] .updatemenu-item-rect[style*="rgb(244, 250, 255)"]{{fill:#1570EF!important;}}
</style></head><body><main class="page"><div class="toolbar">
<h1>ORBIT sampled gaps and calculated paths</h1>
<div class="theme-controls" role="group" aria-label="Color theme">
<button id="theme-light" class="theme-button active" type="button" aria-pressed="true" onclick="setTheme('light')">LIGHT</button>
<button id="theme-dark" class="theme-button" type="button" aria-pressed="false" onclick="setTheme('dark')">DARK</button>
</div></div>
<div class="tabs">{buttons}</div>{''.join(panels)}</main>
<script>
function setTheme(theme){{
 const dark=theme==='dark';
 document.documentElement.dataset.theme=theme;
 document.querySelectorAll('.theme-button').forEach(function(button){{
  const selected=button.id==='theme-'+theme;
  button.classList.toggle('active',selected);button.setAttribute('aria-pressed',selected?'true':'false');
 }});
 const plotTheme=dark?{{
  paper_bgcolor:'#171b24',plot_bgcolor:'#171b24','font.color':'#f2f4f7',
  'modebar.bgcolor':'rgba(23,27,36,.78)','modebar.color':'#98a2b3','modebar.activecolor':'#53b1fd',
  'scene.xaxis.backgroundcolor':'#202631','scene.yaxis.backgroundcolor':'#202631','scene.zaxis.backgroundcolor':'#202631',
  'scene.xaxis.gridcolor':'#475467','scene.yaxis.gridcolor':'#475467','scene.zaxis.gridcolor':'#475467',
  'scene.xaxis.zerolinecolor':'#667085','scene.yaxis.zerolinecolor':'#667085','scene.zaxis.zerolinecolor':'#667085',
  'xaxis.gridcolor':'#475467','yaxis.gridcolor':'#475467',
  'updatemenus[0].bgcolor':'#293241','updatemenus[0].bordercolor':'#667085',
  'updatemenus[1].bgcolor':'#293241','updatemenus[1].bordercolor':'#667085',
  'updatemenus[2].bgcolor':'#293241','updatemenus[2].bordercolor':'#667085',
  'sliders[0].bgcolor':'#293241','sliders[0].bordercolor':'#667085','sliders[0].currentvalue.font.color':'#f2f4f7'
 }}:{{
  paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff','font.color':'#1f2933',
  'modebar.bgcolor':'rgba(255,255,255,.78)','modebar.color':'#697386','modebar.activecolor':'#244b74',
  'scene.xaxis.backgroundcolor':'#FAFBFC','scene.yaxis.backgroundcolor':'#FAFBFC','scene.zaxis.backgroundcolor':'#FAFBFC',
  'scene.xaxis.gridcolor':'#D9E0E7','scene.yaxis.gridcolor':'#D9E0E7','scene.zaxis.gridcolor':'#D9E0E7',
  'scene.xaxis.zerolinecolor':'#D9E0E7','scene.yaxis.zerolinecolor':'#D9E0E7','scene.zaxis.zerolinecolor':'#D9E0E7',
  'xaxis.gridcolor':'#D9E0E7','yaxis.gridcolor':'#D9E0E7',
  'updatemenus[0].bgcolor':'#ffffff','updatemenus[0].bordercolor':'#d9e0e7',
  'updatemenus[1].bgcolor':'#ffffff','updatemenus[1].bordercolor':'#d9e0e7',
  'updatemenus[2].bgcolor':'#ffffff','updatemenus[2].bordercolor':'#d9e0e7',
  'sliders[0].bgcolor':'#ffffff','sliders[0].bordercolor':'#d9e0e7','sliders[0].currentvalue.font.color':'#1f2933'
 }};
 document.querySelectorAll('.plotly-graph-div').forEach(function(plot){{Plotly.relayout(plot,plotTheme)}});
}}
function showViewer(event,panelId){{
 document.querySelectorAll('.tab-panel').forEach(function(panel){{panel.classList.remove('active')}});
 document.querySelectorAll('.tab-button').forEach(function(button){{button.classList.remove('active')}});
 document.getElementById(panelId).classList.add('active');event.currentTarget.classList.add('active');
 setTimeout(function(){{window.dispatchEvent(new Event('resize'))}},40);
}}
</script></body></html>'''
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)

    manifest_path = output_path.with_suffix(".json")
    payload = {
        "schema_version": PATH_PLOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "targets": metadata,
        "shared_sample_color_range_eV": [colors["cmin"], colors["cmax"]],
        "html_file": str(output_path.relative_to(config.root)),
        "html_file_sha256": _sha256(output_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path, manifest_path, payload
