"""Durable records for one path-construction attempt."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, TextIO
from uuid import uuid4


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class CandidateRecord:
    """One destination evaluated by the widest-path search."""

    candidate_order: int
    destination_id: str
    displacement_frac_a: float
    displacement_frac_b: float
    displacement_frac_c: float
    exit_grid_i: int
    exit_grid_j: int
    exit_grid_k: int
    winding_a: int
    winding_b: int
    winding_c: int
    route_found: bool
    theta_star_eV: float | None
    theta_min_eV: float
    feasibility: str
    exact_node_count: int | None
    selected: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        allowed = {"INSULATING", "CLOSES_GAP", "NO_ROUTE", "PRESCRIBED"}
        if self.feasibility not in allowed:
            raise ValueError(
                f"feasibility must be one of {sorted(allowed)}, got {self.feasibility!r}"
            )
        if self.candidate_order < 0:
            raise ValueError("candidate_order must be nonnegative")
        if self.theta_min_eV < 0:
            raise ValueError("theta_min_eV must be nonnegative")

        if self.feasibility == "PRESCRIBED":
            # A prescribed geometric route exists, but its electronic
            # bottleneck has not yet been evaluated.
            if not self.route_found:
                raise ValueError("PRESCRIBED candidate must have route_found=True")
            if self.theta_star_eV is not None:
                raise ValueError(
                    "PRESCRIBED candidate must not assign theta_star_eV before path SCFs"
                )
        else:
            if self.route_found != (self.theta_star_eV is not None):
                raise ValueError("route_found and theta_star_eV disagree")
            if self.route_found and not math.isfinite(float(self.theta_star_eV)):
                raise ValueError("theta_star_eV must be finite for a found route")

        if not self.route_found and self.exact_node_count is not None:
            raise ValueError("exact_node_count must be empty when no route was found")

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["route_found"] = int(self.route_found)
        row["selected"] = int(self.selected)
        row["theta_star_eV"] = (
            "" if self.theta_star_eV is None else f"{self.theta_star_eV:.12g}"
        )
        row["exact_node_count"] = (
            "" if self.exact_node_count is None else self.exact_node_count
        )
        return row


CANDIDATE_FIELDS = tuple(CandidateRecord.__dataclass_fields__)


class PathRunRecorder:
    """Write a run manifest, candidate table, and mirrored terminal log.

    The recorder creates `paths/<target>/runs/<run_id>/` and never replaces a
    previous run.  Call `finish` with a terminal status even when the numerical
    search finds no feasible path.
    """

    def __init__(
        self,
        paths_root: Path,
        target_id: str,
        *,
        inputs: dict[str, Any],
        parameters: dict[str, Any],
        algorithm: str = "widest_path_dijkstra",
        run_id: str | None = None,
        stream: TextIO | None = None,
    ) -> None:
        if not target_id or any(part in target_id for part in ("/", "\\", "..")):
            raise ValueError("target_id must be a nonempty path-safe identifier")

        self.target_id = target_id
        self.run_id = run_id or new_run_id()
        self.started_at = utc_now()
        self.stream = stream if stream is not None else sys.stdout
        self.target_root = Path(paths_root) / target_id
        self.run_dir = self.target_root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "cifs").mkdir()
        self.log_path = self.run_dir / "pathfinding.log"
        self.candidates_path = self.run_dir / "candidates.csv"
        self.manifest_path = self.run_dir / "run.json"
        self.index_path = self.target_root / "index.csv"
        self._log_handle = self.log_path.open("x", encoding="utf-8")
        self._candidates: list[CandidateRecord] = []
        self._finished = False
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "target_id": target_id,
            "algorithm": algorithm,
            "status": "RUNNING",
            "started_at": self.started_at,
            "finished_at": None,
            "inputs": inputs,
            "parameters": parameters,
            "grid_summary": {},
            "candidate_count": 0,
            "selected_destination": None,
            "message": "",
        }
        self._write_manifest()
        self.emit(f"[run] {self.run_id}; target={self.target_id}")

    def __enter__(self) -> "PathRunRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None and not self._finished:
            self.finish("FAILED", message=f"{exc_type.__name__}: {exc}")
        elif not self._finished:
            self.finish("INCOMPLETE", message="recorder closed without finish()")
        return False

    def emit(self, message: str = "") -> None:
        """Print and persist the same ordered diagnostic message."""
        timestamped = f"{utc_now()} {message}"
        print(message, file=self.stream, flush=True)
        print(timestamped, file=self._log_handle, flush=True)

    def set_grid_summary(self, **values: Any) -> None:
        self._ensure_open()
        self._manifest["grid_summary"] = values
        self._write_manifest()

    def add_candidate(self, candidate: CandidateRecord) -> None:
        self._ensure_open()
        if candidate.candidate_order != len(self._candidates):
            raise ValueError("candidate_order must match evaluation order")
        if any(old.destination_id == candidate.destination_id for old in self._candidates):
            raise ValueError(f"duplicate destination_id: {candidate.destination_id}")
        if candidate.selected:
            raise ValueError(
                "selection is determined after comparison; pass "
                "selected_destination to finish()"
            )
        self._candidates.append(candidate)
        self._write_candidates()
        self._manifest["candidate_count"] = len(self._candidates)
        self._write_manifest()

        threshold = (
            "-inf" if candidate.theta_star_eV is None
            else f"{candidate.theta_star_eV:.6f}"
        )
        nodes = (
            "" if candidate.exact_node_count is None
            else f"; {candidate.exact_node_count} exact nodes"
        )
        verdict = {
            "INSULATING": "INSULATING",
            "CLOSES_GAP": "closes gap",
            "NO_ROUTE": "no route",
            "PRESCRIBED": "prescribed route; electronic feasibility unevaluated",
        }[candidate.feasibility]
        self.emit(
            f"[R={candidate.destination_id}] theta* = {threshold} eV -> "
            f"{verdict}{nodes}"
        )

    def finish(
        self,
        status: str,
        *,
        selected_destination: str | None = None,
        message: str = "",
    ) -> None:
        self._ensure_open()
        allowed = {"SUCCEEDED", "INFEASIBLE", "NO_ROUTE", "FAILED", "INCOMPLETE", "CONSTRUCTED"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        if selected_destination is not None and selected_destination not in {
            item.destination_id for item in self._candidates
        }:
            raise ValueError("selected_destination was not recorded as a candidate")

        if selected_destination is not None:
            self._candidates = [
                replace(
                    item,
                    selected=(item.destination_id == selected_destination),
                )
                for item in self._candidates
            ]
            self._write_candidates()

        self._manifest.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "candidate_count": len(self._candidates),
                "selected_destination": selected_destination,
                "message": message,
            }
        )
        self.emit(
            f"[finish] status={status}; selected={selected_destination or 'none'}"
            + (f"; {message}" if message else "")
        )
        self._write_manifest()
        self._append_index()
        self._finished = True
        self._log_handle.close()

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("path run is already finished")

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _write_candidates(self) -> None:
        temporary = self.candidates_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
            writer.writeheader()
            writer.writerows(item.csv_row() for item in self._candidates)
        temporary.replace(self.candidates_path)

    def _append_index(self) -> None:
        self.target_root.mkdir(parents=True, exist_ok=True)
        exists = self.index_path.exists()
        fields = (
            "run_id",
            "target_id",
            "status",
            "started_at",
            "finished_at",
            "candidate_count",
            "selected_destination",
            "run_directory",
        )
        with self.index_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "run_id": self.run_id,
                    "target_id": self.target_id,
                    "status": self._manifest["status"],
                    "started_at": self.started_at,
                    "finished_at": self._manifest["finished_at"],
                    "candidate_count": len(self._candidates),
                    "selected_destination": self._manifest["selected_destination"] or "",
                    "run_directory": str(self.run_dir.relative_to(self.target_root)),
                }
            )
