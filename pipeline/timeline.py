from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MIN_SEGMENT_SECONDS = 4.0
MAX_SEGMENT_SECONDS = 15.0

VALID_CONTINUITY_MODES = {
    "independent",
    "chained",
    "anchored",
    "hard_cut",
    "scene_reset",
}


@dataclass(frozen=True)
class TimelineSegment:
    shot_id: str
    scene_id: str
    order: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    continuity_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "order": self.order,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "continuity_mode": self.continuity_mode,
        }


class ProductionTimeline:
    """Canonical timeline derived from the production plan.

    The timeline is a view/edit surface only. The shot plan remains the
    source of truth; applying edits updates shot durations and continuity
    policy, then deterministically rebuilds this timeline.
    """

    def __init__(self, plan: dict[str, Any]):
        self.plan = plan

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _mode_for_shot(shot: dict[str, Any], previous: dict[str, Any] | None) -> str:
        explicit = str(shot.get("continuity_mode", "") or "").strip().lower()
        if explicit in VALID_CONTINUITY_MODES:
            return explicit
        if bool(shot.get("is_scene_boundary", False)) or previous is None:
            return "scene_reset"
        return "chained"

    def build(self) -> list[TimelineSegment]:
        shots = [
            shot for shot in (self.plan.get("shots", []) or [])
            if isinstance(shot, dict)
        ]
        shots.sort(key=lambda item: (int(item.get("order", 0) or 0), str(item.get("shot_id", ""))))
        cursor = 0.0
        previous: dict[str, Any] | None = None
        segments: list[TimelineSegment] = []
        for shot in shots:
            duration = self._safe_float(shot.get("duration_seconds"), 5.2)
            if duration < MIN_SEGMENT_SECONDS or duration > MAX_SEGMENT_SECONDS:
                raise ValueError(
                    f"Duration for {shot.get('shot_id', 'unknown')} must be between "
                    f"{MIN_SEGMENT_SECONDS:g} and {MAX_SEGMENT_SECONDS:g} seconds."
                )
            start = cursor
            end = start + duration
            mode = self._mode_for_shot(shot, previous)
            shot["duration_seconds"] = round(duration, 4)
            shot["timeline_start_seconds"] = round(start, 4)
            shot["timeline_end_seconds"] = round(end, 4)
            shot["continuity_mode"] = mode
            if mode == "scene_reset":
                shot["is_scene_boundary"] = True
            segments.append(
                TimelineSegment(
                    shot_id=str(shot.get("shot_id", "")),
                    scene_id=str(shot.get("scene_id", "")),
                    order=int(shot.get("order", 0) or 0),
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=duration,
                    continuity_mode=mode,
                )
            )
            cursor = end
            previous = shot
        self.plan["timeline"] = {
            "version": 1,
            "total_duration_seconds": round(cursor, 4),
            "segments": [segment.to_dict() for segment in segments],
        }
        return segments

    def table(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for segment in self.build():
            rows.append([
                segment.shot_id,
                segment.scene_id,
                round(segment.start_seconds, 2),
                round(segment.end_seconds, 2),
                round(segment.duration_seconds, 2),
                segment.continuity_mode,
            ])
        return rows

    def apply_table(self, rows: Any) -> list[TimelineSegment]:
        if rows is None:
            return self.build()
        if hasattr(rows, "to_numpy"):
            rows = rows.to_numpy().tolist()
        rows = list(rows)
        by_id = {
            str(shot.get("shot_id", "")): shot
            for shot in (self.plan.get("shots", []) or [])
            if isinstance(shot, dict)
        }
        for row in rows:
            if not isinstance(row, (list, tuple)) or not row:
                continue
            shot = by_id.get(str(row[0]).strip())
            if shot is None:
                continue
            if len(row) >= 5:
                duration = self._safe_float(row[4], self._safe_float(shot.get("duration_seconds"), 5.2))
                if duration < MIN_SEGMENT_SECONDS or duration > MAX_SEGMENT_SECONDS:
                    raise ValueError(
                        f"Duration for {shot['shot_id']} must be between "
                        f"{MIN_SEGMENT_SECONDS:g} and {MAX_SEGMENT_SECONDS:g} seconds."
                    )
                shot["duration_seconds"] = duration
            if len(row) >= 6:
                mode = str(row[5] or "").strip().lower()
                if mode not in VALID_CONTINUITY_MODES:
                    raise ValueError(
                        f"Invalid continuity_mode {mode!r}; choose one of {sorted(VALID_CONTINUITY_MODES)}."
                    )
                shot["continuity_mode"] = mode
        return self.build()

    @staticmethod
    def validate(plan: dict[str, Any]) -> None:
        timeline = plan.get("timeline")
        if not isinstance(timeline, dict):
            raise RuntimeError("Production timeline is missing.")
        segments = timeline.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError("Production timeline segments are missing.")
        previous_end = 0.0
        ids = set()
        for item in segments:
            if not isinstance(item, dict):
                raise RuntimeError("Timeline segment must be an object.")
            shot_id = str(item.get("shot_id", "")).strip()
            if not shot_id or shot_id in ids:
                raise RuntimeError(f"Duplicate or empty timeline shot_id: {shot_id!r}")
            ids.add(shot_id)
            start = float(item.get("start_seconds", 0.0))
            end = float(item.get("end_seconds", 0.0))
            duration = float(item.get("duration_seconds", 0.0))
            if (
                start < previous_end - 1e-5
                or end <= start
                or duration < MIN_SEGMENT_SECONDS
                or duration > MAX_SEGMENT_SECONDS
                or abs((end - start) - duration) > 0.02
            ):
                raise RuntimeError(f"Invalid timeline segment for {shot_id}.")
            mode = str(item.get("continuity_mode", ""))
            if mode not in VALID_CONTINUITY_MODES:
                raise RuntimeError(f"Invalid timeline continuity mode for {shot_id}: {mode!r}")
            previous_end = end
