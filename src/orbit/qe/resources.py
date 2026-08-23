"""Pseudopotential, cutoff, valence-electron, and band-count resolution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from ..config import ProjectConfig


@dataclass(frozen=True)
class ElementResource:
    element: str
    pseudopotential: str
    pseudopotential_sha256: str
    cutoff_wfc_Ry: float
    cutoff_rho_Ry: float
    z_valence: float


@dataclass(frozen=True)
class QEResources:
    pseudo_root: Path
    pseudo_dir: Path
    cutoff_file: Path
    cutoff_file_sha256: str
    elements: tuple[ElementResource, ...]
    ecutwfc_Ry: float
    ecutrho_Ry: float
    total_valence_electrons: int
    occupied_bands: int
    nbnd: int
    nbnd_source: str

    @property
    def pseudopotentials(self) -> dict[str, str]:
        return {item.element: item.pseudopotential for item in self.elements}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return result


def _discover_pseudopotential(
    library: Path,
    element: str,
    override: str | None,
) -> Path:
    if override is not None:
        if Path(override).name != override:
            raise ValueError(
                f"Pseudopotential override for {element} must be a filename, "
                f"not a path: {override!r}"
            )
        selected = library / override
        if not selected.is_file():
            raise FileNotFoundError(
                f"Configured {element} pseudopotential not found: {selected}"
            )
        return selected

    prefix = re.compile(rf"^{re.escape(element)}(?=[._-])", re.IGNORECASE)
    matches = sorted(
        path
        for path in library.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".upf"
        and prefix.match(path.name)
    )
    if not matches:
        raise FileNotFoundError(
            f"No pseudopotential matching {element}.*.upf in {library}"
        )
    if len(matches) > 1:
        choices = "\n".join(f"  {path.name}" for path in matches)
        raise RuntimeError(
            f"Multiple {element} pseudopotentials found; configure "
            f"[pseudopotentials.files].{element}:\n{choices}"
        )
    return matches[0]


def parse_upf_z_valence(path: Path) -> float:
    """Extract z_valence from XML-style or legacy UPF headers."""
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"\bz_valence\s*=\s*[\"']?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+Z\s+valence",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _positive_number(float(match.group(1)), f"z_valence in {path}")
    raise ValueError(f"Could not parse z_valence from pseudopotential: {path}")


def _automatic_nbnd(occupied_bands: int) -> int:
    empty_bands = max(8, math.ceil(0.25 * occupied_bands))
    requested = occupied_bands + empty_bands
    return 8 * math.ceil(requested / 8)


def resolve_qe_resources(
    config: ProjectConfig,
    symbols: list[str],
) -> QEResources:
    pseudo_root = config.pseudopotentials.root
    library = pseudo_root / "library"
    cutoff_file = pseudo_root / "cutoffs.json"
    if not library.is_dir():
        raise FileNotFoundError(f"Pseudopotential library not found: {library}")
    if not cutoff_file.is_file():
        raise FileNotFoundError(f"Cutoff database not found: {cutoff_file}")
    try:
        cutoff_data = json.loads(cutoff_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {cutoff_file}: {exc}") from exc
    if not isinstance(cutoff_data, dict):
        raise ValueError(f"Cutoff database must contain a JSON object: {cutoff_file}")

    element_order = tuple(dict.fromkeys(symbols))
    counts = {element: symbols.count(element) for element in element_order}
    resources: list[ElementResource] = []
    total_valence = 0.0
    for element in element_order:
        pseudo = _discover_pseudopotential(
            library,
            element,
            config.pseudopotentials.files.get(element),
        )
        entry = cutoff_data.get(element)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing cutoff object for element {element!r}")
        if "cutoff_wfc" not in entry or "cutoff_rho" not in entry:
            raise ValueError(
                f"Cutoff entry {element!r} must define cutoff_wfc and cutoff_rho"
            )
        z_valence = parse_upf_z_valence(pseudo)
        total_valence += counts[element] * z_valence
        resources.append(
            ElementResource(
                element=element,
                pseudopotential=pseudo.name,
                pseudopotential_sha256=_sha256(pseudo),
                cutoff_wfc_Ry=_positive_number(
                    entry["cutoff_wfc"], f"{element}.cutoff_wfc"
                ),
                cutoff_rho_Ry=_positive_number(
                    entry["cutoff_rho"], f"{element}.cutoff_rho"
                ),
                z_valence=z_valence,
            )
        )

    rounded_electrons = round(total_valence)
    if not math.isclose(total_valence, rounded_electrons, abs_tol=1.0e-7):
        raise ValueError(
            f"Total UPF valence electron count is nonintegral: {total_valence}"
        )
    total_electrons = int(rounded_electrons)
    if total_electrons % 2:
        raise ValueError(
            f"Neutral cell has an odd UPF valence count ({total_electrons}); "
            "spin-polarized automatic band setup is not yet supported"
        )
    occupied_bands = total_electrons // 2
    if config.qe.nbnd is None:
        nbnd = _automatic_nbnd(occupied_bands)
        nbnd_source = "auto_from_upf_valence"
    else:
        nbnd = config.qe.nbnd
        nbnd_source = "explicit_config"
    if nbnd <= occupied_bands:
        raise ValueError(
            f"nbnd={nbnd} must exceed the {occupied_bands} occupied bands "
            "so that a LUMO is calculated"
        )

    return QEResources(
        pseudo_root=pseudo_root,
        pseudo_dir=library.resolve(),
        cutoff_file=cutoff_file.resolve(),
        cutoff_file_sha256=_sha256(cutoff_file),
        elements=tuple(resources),
        ecutwfc_Ry=max(item.cutoff_wfc_Ry for item in resources),
        ecutrho_Ry=max(item.cutoff_rho_Ry for item in resources),
        total_valence_electrons=total_electrons,
        occupied_bands=occupied_bands,
        nbnd=nbnd,
        nbnd_source=nbnd_source,
    )

