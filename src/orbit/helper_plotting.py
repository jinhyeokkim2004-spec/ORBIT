
"""Interactive helper-iteration path viewers matching ORBIT's path viewer."""

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

import numpy as np
from ase.io import read as ase_read
import plotly.io as pio
from plotly import graph_objects as go

from .config import ProjectConfig
from .plotting import CPK_COLORS
from .structure import load_site_records

HELPER_PLOT_SCHEMA_VERSION = 3
ATOM_STYLE = {"O": {"color": "#D62728", "radius": 0.34},
              "H": {"color": "#D9D9D9", "radius": 0.24}}
FALLBACK = {"color": "#9AA0A6", "radius": 0.40}
TARGET_COLOR = "#FFD166"
HELPER_COLOR = "#E8703A"
PATH_COLOR = "#7B2CBF"
MIN_COLOR = "#D62728"
ANCHOR_COLOR = "#98A4A8"
THRESHOLD_COLOR = "#667085"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "target"


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    return rows


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _complete_rows(path: Path, description: str) -> list[dict[str, str]]:
    rows = _read_csv(path, description)
    bad = [r for r in rows if r.get("output_status") != "COMPLETE"]
    if bad:
        raise ValueError(f"{description} is incomplete: {len(bad)} non-COMPLETE rows")
    return rows


def _formula(formula: str) -> str:
    digits = "₀₁₂₃₄₅₆₇₈₉"
    return re.sub(r"\d+", lambda m: "".join(digits[int(d)] for d in m.group()), formula)


def _atom_style(symbol: str) -> dict[str, Any]:
    if symbol in ATOM_STYLE:
        return ATOM_STYLE[symbol]
    return {"color": CPK_COLORS.get(symbol, FALLBACK["color"]),
            "radius": FALLBACK["radius"]}


def _sphere(center: Any, radius: float, color: str, name: str, hover: str | None = None):
    center = np.asarray(center, dtype=float)
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 18)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)
    return go.Mesh3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(),
        alphahull=0, color=color, opacity=1.0, flatshading=False,
        name=name, hovertext=hover or name, hoverinfo="text",
        lighting={"ambient": 0.50, "diffuse": 0.85,
                  "specular": 0.45, "roughness": 0.35},
        showscale=False,
    )


def _cell_edges(cell: np.ndarray):
    corners = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)
    cart = corners @ cell
    x, y, z = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.abs(corners[i] - corners[j]).sum()) == 1:
                x += [cart[i, 0], cart[j, 0], None]
                y += [cart[i, 1], cart[j, 1], None]
                z += [cart[i, 2], cart[j, 2], None]
    return x, y, z


def _sources(config: ProjectConfig, target: str, run_id: str, latest: int):
    out = [{
        "iteration": 0,
        "label": "Iteration 0 · target-only path",
        "calc_root": config.root / "calculations" / "paths" / target / run_id,
        "gap_csv": config.root / "results" / "path_gaps" / target / run_id / "gaps.csv",
        "anchors": [],
        "helpers": set(),
    }]
    for it in range(1, latest + 1):
        root = config.root / "helpers" / target / run_id / f"iteration_{it:03d}"
        anchors = list(_read_json(root / "anchors.json", f"Iteration {it} anchors").get("anchors", []))
        helpers = {int(a["helper_index"]) for a in anchors if a.get("helper_index") not in (None, "")}
        newest = next((a for a in reversed(anchors) if int(a.get("iteration", -1)) == it), None)
        label = f"Iteration {it}"
        if newest is not None:
            label += f" · image {int(newest['image']):03d} · {newest.get('helper_site_id', 'helper')}"
        out.append({
            "iteration": it,
            "label": label,
            "calc_root": config.root / "calculations" / "helpers" / target / run_id / f"iteration_{it:03d}" / "path",
            "gap_csv": config.root / "results" / "helper_gaps" / target / run_id / f"iteration_{it:03d}" / "gaps.csv",
            "anchors": anchors,
            "helpers": helpers,
        })
    return out


def _load(source: dict[str, Any]):
    rows = _complete_rows(Path(source["gap_csv"]), f"Iteration {source['iteration']} gaps")
    frames = []
    for row in rows:
        image = int(row["image"])
        name = row.get("image_name") or f"image_{image:03d}"
        structure = Path(source["calc_root"]) / name / "structure.cif"
        frames.append({"image": image, "name": name, "gap": float(row["gap_eV"]),
                       "atoms": ase_read(structure)})
    frames.sort(key=lambda r: r["image"])
    return frames


def _figure(config: ProjectConfig, target: str, target_index: int, source: dict[str, Any]):
    frames = _load(source)
    it = int(source["iteration"])
    helpers = set(source["helpers"])
    anchors = list(source["anchors"])
    atoms0 = frames[0]["atoms"]
    symbols = atoms0.get_chemical_symbols()
    cell = np.asarray(atoms0.cell.array, dtype=float)
    formula = _formula(atoms0.get_chemical_formula())

    images = np.asarray([f["image"] for f in frames], dtype=int)
    gaps = np.asarray([f["gap"] for f in frames], dtype=float)
    threshold = float(config.helper.target_gap_eV)
    lo = min(float(gaps.min()), threshold, 0.0)
    hi = max(float(gaps.max()), threshold, 0.0)
    span = max(hi - lo, 0.05)
    yrange = [lo - 0.10 * span, hi + 0.10 * span]

    traces = []
    ex, ey, ez = _cell_edges(cell)
    traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line={"color": "#34495E", "width": 3},
                               name="unit cell", hoverinfo="skip"))

    # Static framework at step 0.
    #
    # Target and active helper atoms are omitted here because separate animated
    # overlays render them below. Otherwise the static step-0 copies remain
    # visible and appear as stationary ghost atoms during helper motion.
    cart0 = np.asarray(atoms0.get_positions(), dtype=float)
    for idx, (symbol, center) in enumerate(zip(symbols, cart0)):
        if idx == target_index or idx in helpers:
            continue
        style = _atom_style(symbol)
        traces.append(
            _sphere(
                center,
                float(style["radius"]),
                str(style["color"]),
                f"{symbol}{idx} (framework)",
            )
        )

    # Trajectories.
    tcoords = np.asarray([f["atoms"].get_positions()[target_index] for f in frames])
    traces.append(go.Scatter3d(
        x=tcoords[:, 0], y=tcoords[:, 1], z=tcoords[:, 2],
        mode="lines+markers", line={"color": PATH_COLOR, "width": 5, "dash": "dot"},
        marker={"size": 3.5, "color": PATH_COLOR}, name="target path trajectory"))
    for hidx in sorted(helpers):
        coords = np.asarray([f["atoms"].get_positions()[hidx] for f in frames])
        traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="lines+markers", line={"color": HELPER_COLOR, "width": 3, "dash": "dot"},
            marker={"size": 2.5, "color": HELPER_COLOR},
            name=f"helper trajectory {symbols[hidx]}{hidx}"))

    # Moving target/helper overlays.
    tstyle = _atom_style(symbols[target_index])
    traces.append(_sphere(tcoords[0], float(tstyle["radius"]) * 1.15,
                          TARGET_COLOR, f"moving {target}"))
    moving_target = len(traces) - 1
    moving_helpers = {}
    for hidx in sorted(helpers):
        hstyle = _atom_style(symbols[hidx])
        center = frames[0]["atoms"].get_positions()[hidx]
        traces.append(_sphere(center, float(hstyle["radius"]) * 1.15,
                              HELPER_COLOR, f"moving helper {symbols[hidx]}{hidx}"))
        moving_helpers[hidx] = len(traces) - 1

    # Gap panel.
    traces.append(go.Scatter(
        x=images, y=gaps, mode="lines+markers",
        line={"color": PATH_COLOR, "width": 2.5},
        marker={"size": 6, "color": PATH_COLOR},
        name="calculated path E<sub>gap</sub>",
        hovertemplate="step=%{x}<br>E<sub>gap</sub>=%{y:.4f} eV<extra></extra>"))

    min_i = int(np.argmin(gaps))
    min_step = int(images[min_i])
    min_gap = float(gaps[min_i])
    traces.append(go.Scatter(
        x=[min_step], y=[min_gap], mode="markers+text",
        marker={"size": 14, "symbol": "star", "color": MIN_COLOR,
                "line": {"width": 1.5, "color": "#7F1D1D"}},
        text=[f"minimum {min_gap:.4f} eV · step {min_step}"],
        textposition="top right" if min_step < len(frames)/2 else "top left",
        name="minimum path gap"))

    traces.append(go.Scatter(
        x=[int(images[0])], y=[float(gaps[0])], mode="markers+text",
        marker={"size": 13, "color": TARGET_COLOR,
                "line": {"width": 2, "color": "#5B3A00"}},
        text=[f"{float(gaps[0]):.4f} eV"], textposition="bottom center",
        name="selected path step"))
    marker_idx = len(traces) - 1

    traces.append(go.Scatter(
        x=[int(images[0]), int(images[0])], y=yrange, mode="lines",
        line={"color": TARGET_COLOR, "width": 2},
        hoverinfo="skip", showlegend=False))
    cursor_idx = len(traces) - 1

    traces.append(go.Scatter(
        x=[int(images.min()), int(images.max())], y=[threshold, threshold],
        mode="lines", line={"color": THRESHOLD_COLOR, "width": 1.6, "dash": "dash"},
        name=f"target gap {threshold:.3f} eV", hoverinfo="skip"))

    for anchor in anchors:
        aimg = int(anchor["image"])
        traces.append(go.Scatter(
            x=[aimg, aimg], y=yrange, mode="lines",
            line={"color": ANCHOR_COLOR, "width": 1.4, "dash": "dot"},
            name=f"iteration {int(anchor.get('iteration', 0))} anchor · "
                 f"{anchor.get('helper_site_id', 'helper')} @ {aimg}",
            hoverinfo="skip"))

    anim_frames = []
    slider_steps = []
    for n, frame in enumerate(frames):
        image, gap = int(frame["image"]), float(frame["gap"])
        update_data = []
        update_indices = []

        center = frame["atoms"].get_positions()[target_index]
        update_data.append(_sphere(center, float(tstyle["radius"]) * 1.15,
                                   TARGET_COLOR, f"moving {target}",
                                   f"{frame['name']}<br>Egap={gap:.4f} eV"))
        update_indices.append(moving_target)

        for hidx in sorted(helpers):
            hstyle = _atom_style(symbols[hidx])
            hcenter = frame["atoms"].get_positions()[hidx]
            update_data.append(_sphere(
                hcenter, float(hstyle["radius"]) * 1.15, HELPER_COLOR,
                f"moving helper {symbols[hidx]}{hidx}",
                f"{frame['name']}<br>{symbols[hidx]}{hidx} helper<br>Egap={gap:.4f} eV"))
            update_indices.append(moving_helpers[hidx])

        update_data.append(go.Scatter(
            x=[image], y=[gap], mode="markers+text",
            marker={"size": 13, "color": TARGET_COLOR,
                    "line": {"width": 2, "color": "#5B3A00"}},
            text=[f"{gap:.4f} eV"], textposition="bottom center"))
        update_indices.append(marker_idx)

        update_data.append(go.Scatter(
            x=[image, image], y=yrange, mode="lines",
            line={"color": TARGET_COLOR, "width": 2},
            hoverinfo="skip", showlegend=False))
        update_indices.append(cursor_idx)

        fname = f"{_safe(target)}_helper_i{it}_{n:03d}"
        anim_frames.append(go.Frame(name=fname, data=update_data, traces=update_indices))
        slider_steps.append({
            "method": "animate",
            "label": str(image) if n % 5 == 0 or n == len(frames)-1 else "",
            "args": [[fname], {"mode": "immediate",
                               "frame": {"duration": 0, "redraw": True},
                               "transition": {"duration": 0}}],
        })

    fig = go.Figure(data=traces, frames=anim_frames)
    kind = "target-only path" if it == 0 else "helper-assisted path"
    fig.update_layout(
        template="plotly_white",
        title={"text": f"{formula} — {target} {kind}: iteration {it} ({len(frames)} images)",
               "x": 0.5, "xanchor": "center", "y": 0.99, "yanchor": "top"},
        scene={
            "xaxis_title": "x (Å)", "yaxis_title": "y (Å)", "zaxis_title": "z (Å)",
            "aspectmode": "data",
            "xaxis": {"backgroundcolor": "#FAFBFC"},
            "yaxis": {"backgroundcolor": "#FAFBFC"},
            "zaxis": {"backgroundcolor": "#FAFBFC"},
            "domain": {"x": [0.0, 1.0], "y": [0.36, 1.0]},
        },
        xaxis={"title": "Path step", "domain": [0.08, 0.92],
               "range": [-1.5, len(frames) + 0.5], "dtick": 5,
               "visible": True, "showgrid": True, "gridcolor": "#D9E0E7"},
        yaxis={"title": "E<sub>gap</sub> (eV)", "domain": [0.02, 0.25],
               "range": yrange, "visible": True, "showgrid": True,
               "gridcolor": "#D9E0E7"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.27, "x": 0.0},
        margin={"l": 0, "r": 0, "t": 155, "b": 105},
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 0.01, "y": 1.10,
            "xanchor": "left", "yanchor": "top", "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"fromcurrent": True, "mode": "immediate",
                                  "frame": {"duration": 450, "redraw": True},
                                  "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0, "visible": True, "x": 0.08, "len": 0.84, "y": -0.02,
            "xanchor": "left", "yanchor": "top", "pad": {"t": 45, "b": 0},
            "currentvalue": {"prefix": "Path step: ", "visible": True,
                             "xanchor": "center", "font": {"size": 15}},
            "transition": {"duration": 0}, "steps": slider_steps,
        }],
        uirevision=f"{target}-helper-i{it}-camera",
    )
    return fig, {
        "iteration": it, "label": source["label"],
        "complete_count": len(frames), "image_count": len(frames),
        "minimum_gap_eV": min_gap, "minimum_gap_image": min_step,
        "target_gap_eV": threshold,
        "helper_indices": sorted(helpers),
        "anchor_images": [int(a["image"]) for a in anchors],
    }


CSS = """
:root{color-scheme:light;--page-bg:#f4f6f8;--text:#1f2933;--panel:#fff;--border:#d9e0e7;--accent:#244b74;--hover:#eef3f7;--shadow:rgba(31,41,51,.08)}
:root[data-theme="dark"]{color-scheme:dark;--page-bg:#0f1116;--text:#f2f4f7;--panel:#171b24;--border:#475467;--accent:#53b1fd;--hover:#293241;--shadow:rgba(0,0,0,.35)}
html,body{margin:0;min-height:100%;background:var(--page-bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}
.page{max-width:1500px;margin:0 auto;padding:18px}
h1{font-size:22px;margin:0}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 14px}
.theme-controls{display:flex;gap:6px}.theme-button{border:1px solid var(--border);background:var(--panel);color:var(--text);padding:7px 12px;border-radius:8px;cursor:pointer;font-weight:600}
.theme-button:hover{background:var(--hover)}.theme-button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tabs{display:flex;gap:8px;flex-wrap:wrap}.tab-button{border:1px solid var(--border);background:var(--panel);color:var(--accent);padding:9px 18px;border-radius:12px 12px 0 0;cursor:pointer;font-weight:600}
.tab-button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tab-panel{display:none;background:var(--panel);border:1px solid var(--border);border-radius:0 12px 12px 12px;min-height:760px}.tab-panel.active{display:block}
.single-panel{display:block;background:var(--panel);border:1px solid var(--border);border-radius:12px;min-height:760px}
.plotly-graph-div{height:min(82vh,900px)!important;min-height:700px}
"""

THEME_JS = """
function setTheme(theme){
 const dark=theme==='dark';
 document.documentElement.dataset.theme=theme;
 document.querySelectorAll('.theme-button').forEach(function(button){
  const selected=button.id==='theme-'+theme;
  button.classList.toggle('active',selected);
  button.setAttribute('aria-pressed',selected?'true':'false');
 });
 const p=dark?{
  paper_bgcolor:'#171b24',plot_bgcolor:'#171b24','font.color':'#f2f4f7',
  'scene.xaxis.backgroundcolor':'#202631','scene.yaxis.backgroundcolor':'#202631','scene.zaxis.backgroundcolor':'#202631',
  'scene.xaxis.gridcolor':'#475467','scene.yaxis.gridcolor':'#475467','scene.zaxis.gridcolor':'#475467',
  'xaxis.gridcolor':'#475467','yaxis.gridcolor':'#475467',
  'updatemenus[0].bgcolor':'#293241','updatemenus[0].bordercolor':'#667085',
  'sliders[0].bgcolor':'#293241','sliders[0].bordercolor':'#667085','sliders[0].currentvalue.font.color':'#f2f4f7'
 }:{
  paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff','font.color':'#1f2933',
  'scene.xaxis.backgroundcolor':'#FAFBFC','scene.yaxis.backgroundcolor':'#FAFBFC','scene.zaxis.backgroundcolor':'#FAFBFC',
  'scene.xaxis.gridcolor':'#D9E0E7','scene.yaxis.gridcolor':'#D9E0E7','scene.zaxis.gridcolor':'#D9E0E7',
  'xaxis.gridcolor':'#D9E0E7','yaxis.gridcolor':'#D9E0E7',
  'updatemenus[0].bgcolor':'#ffffff','updatemenus[0].bordercolor':'#d9e0e7',
  'sliders[0].bgcolor':'#ffffff','sliders[0].bordercolor':'#d9e0e7','sliders[0].currentvalue.font.color':'#1f2933'
 };
 document.querySelectorAll('.plotly-graph-div').forEach(function(plot){if(window.Plotly)Plotly.relayout(plot,p);});
}
"""


def _controls():
    return """<div class="theme-controls">
<button id="theme-light" class="theme-button active" onclick="setTheme('light')">LIGHT</button>
<button id="theme-dark" class="theme-button" onclick="setTheme('dark')">DARK</button>
</div>"""


def _single(fig: go.Figure, path: Path, title: str, div_id: str):
    ph = pio.to_html(fig, include_plotlyjs=True, full_html=False, div_id=div_id,
                     auto_play=False,
                     config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>{CSS}</style></head><body><main class="page"><div class="toolbar">
<h1>{escape(title)}</h1>{_controls()}</div><section class="single-panel">{ph}</section>
</main><script>{THEME_JS}</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def _tabs(rendered, path: Path, title: str):
    buttons, panels = [], []
    included = False
    for i, (source, fig, _summary) in enumerate(rendered):
        div = f"helper-viewer-{int(source['iteration']):03d}"
        ph = pio.to_html(fig, include_plotlyjs=not included, full_html=False,
                         div_id=div, auto_play=False,
                         config={"responsive": True, "displaylogo": False, "scrollZoom": True})
        included = True
        pid = div + "-panel"
        active = " active" if i == 0 else ""
        buttons.append(f'<button class="tab-button{active}" onclick="showIteration(event, \'{pid}\')">{escape(str(source["label"]))}</button>')
        panels.append(f'<section id="{pid}" class="tab-panel{active}">{ph}</section>')
    js = """
function pauseAll(){document.querySelectorAll('.plotly-graph-div').forEach(function(p){if(window.Plotly)Plotly.animate(p,[null],{mode:'immediate',frame:{duration:0,redraw:false},transition:{duration:0}});});}
function showIteration(event,id){pauseAll();document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(b=>b.classList.remove('active'));document.getElementById(id).classList.add('active');event.currentTarget.classList.add('active');setTimeout(()=>window.dispatchEvent(new Event('resize')),40);}
"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>{CSS}</style></head><body><main class="page"><div class="toolbar">
<h1>{escape(title)}</h1>{_controls()}</div><div class="tabs">{''.join(buttons)}</div>
{''.join(panels)}</main><script>{THEME_JS}{js}</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def write_helper_iteration_viewers(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    latest_iteration: int,
    *,
    output: Path | None = None,
):
    sites = load_site_records(config.root / "manifests" / "sites.csv")
    matches = [s for s in sites if s.site_id == target_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one site record for {target_id}")
    target_index = int(matches[0].ase_index_zero_based)

    sources = _sources(config, target_id, run_id, latest_iteration)
    root = config.root / "plots" / "helpers" / target_id / run_id
    root.mkdir(parents=True, exist_ok=True)

    rendered, files, skipped = [], [], []
    for source in sources:
        try:
            fig, summary = _figure(config, target_id, target_index, source)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({
                "iteration": int(source["iteration"]),
                "label": str(source["label"]),
                "reason": str(exc),
            })
            continue

        it = int(source["iteration"])
        ipath = root / f"iteration_{it:03d}.html"
        _single(
            fig,
            ipath,
            f"ORBIT helper path — {target_id} — iteration {it}",
            f"helper-single-{_safe(target_id)}-{it:03d}",
        )
        rendered.append((source, fig, summary))
        files.append({**summary, "plot_file": str(ipath), "plot_sha256": _sha256(ipath)})

    if not rendered:
        raise ValueError(
            f"No helper iterations are plot-ready for target {target_id} and run {run_id}. "
            "This usually means the helper path gap tables are missing, empty, or incomplete."
        )

    combined = root / f"{target_id}_helper_paths_all_iterations.html" if output is None else (
        output if output.is_absolute() else config.root / output
    )
    formula = _formula(
        ase_read(Path(rendered[0][0]["calc_root"]) / "image_000" / "structure.cif").get_chemical_formula()
    )
    plotted_min = min(int(item["iteration"]) for item in files)
    plotted_max = max(int(item["iteration"]) for item in files)
    _tabs(
        rendered,
        combined,
        f"ORBIT {formula} {target_id} helper-assisted paths — available iterations {plotted_min}–{plotted_max}",
    )

    latest = files[-1]
    return combined, {
        "schema_version": HELPER_PLOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": int(latest["iteration"]),
        "combined_plot_file": str(combined),
        "combined_plot_sha256": _sha256(combined),
        "iteration_files": files,
        "skipped_iterations": skipped,
        "minimum_gap_eV": float(latest["minimum_gap_eV"]),
        "minimum_gap_image": int(latest["minimum_gap_image"]),
        "target_gap_eV": float(config.helper.target_gap_eV),
    }
