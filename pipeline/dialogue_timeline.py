from __future__ import annotations

import math
import re
from typing import Any

from pipeline.dialogue_duration import (
    DialogueDurationProvider,
    ExplicitOrWPMDurationProvider,
)
from schemas.dialogue import DialogueEvent
from planner.config import H3_FPS, H3_FRAMES_PER_SHOT


class DialogueTimeline:
    """Deterministically canonicalize and schedule shot dialogue.

    Planning timing uses the exact H3-legal video duration (24 FPS with the
    17*n+5 frame rule and H3's 4-15 second / 124-362 frame bounds). Speech
    duration is an estimate unless an explicit duration is supplied. Final
    rendered-media duration is validated separately with ffprobe.
    """

    FPS = float(H3_FPS)
    MIN_REQUESTED_SECONDS = 4.0
    MAX_REQUESTED_SECONDS = 15.0
    MIN_FRAMES = int(H3_FRAMES_PER_SHOT)
    MAX_FRAMES = 362

    DEFAULT_PRE_ROLL = 0.35
    DEFAULT_POST_ROLL = 0.35
    MIN_EVENT_DURATION = 0.60

    def __init__(
        self,
        characters: list[dict] | None = None,
        duration_provider: DialogueDurationProvider | None = None,
    ) -> None:
        self.characters = list(characters or [])
        self.duration_provider = duration_provider or ExplicitOrWPMDurationProvider()
        self._by_name = {
            self._norm(c.get("name", "")): c
            for c in self.characters
            if isinstance(c, dict) and self._norm(c.get("name", ""))
        }
        self._by_id = {
            str(c.get("character_id", "")).strip(): c
            for c in self.characters
            if isinstance(c, dict) and str(c.get("character_id", "")).strip()
        }

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @classmethod
    def h3_legal_frames(cls, requested_seconds: float) -> int:
        """Mirror the production H3 workflow's exact legal frame calculation."""
        seconds = max(
            cls.MIN_REQUESTED_SECONDS,
            min(cls.MAX_REQUESTED_SECONDS, float(requested_seconds or 0.0)),
        )
        requested_frames = max(5, round(seconds * cls.FPS))
        n = max(0, (requested_frames - 5 + 16) // 17)
        frames = 17 * n + 5
        return max(cls.MIN_FRAMES, min(cls.MAX_FRAMES, frames))

    @classmethod
    def h3_effective_duration_seconds(cls, requested_seconds: float) -> float:
        return cls.h3_legal_frames(requested_seconds) / cls.FPS

    @classmethod
    def _minimum_h3_duration_for_required_seconds(cls, required_seconds: float) -> tuple[int, float]:
        """Return the smallest H3-legal duration that can contain ``required_seconds``.

        H3 durations live on the ``17*n+5`` frame grid. This helper intentionally
        works from required runtime rather than the caller's requested duration so
        deterministic dialogue scheduling can expand a short Qwen-selected shot
        without changing the spoken text.
        """
        required = max(0.0, float(required_seconds))
        required_frames = max(
            cls.MIN_FRAMES,
            int(math.ceil(required * cls.FPS - 1e-9)),
        )
        minimum_n = (cls.MIN_FRAMES - 5) // 17
        n = max(
            minimum_n,
            int(math.ceil((required_frames - 5) / 17.0)),
        )
        frames = 17 * n + 5
        if frames > cls.MAX_FRAMES:
            raise ValueError(
                f"Required H3 runtime {required:.3f}s exceeds the maximum legal "
                f"duration ({cls.MAX_FRAMES / cls.FPS:.3f}s)."
            )
        return frames, frames / cls.FPS

    @classmethod
    def _legacy_events(cls, shot: dict) -> list[dict]:
        text = str(shot.get("speech_text", "") or "")
        speakers = [
            str(v).strip()
            for v in (shot.get("speaking_characters", []) or [])
            if str(v).strip()
        ]
        if not text.strip():
            return []
        if len(speakers) == 1:
            # Preserve legacy "Speaker: line" formatting when present, while
            # keeping the spoken text itself exact.
            prefix = re.match(r"^\s*([^:]{1,80}):\s*(.+?)\s*$", text)
            spoken_text = prefix.group(2) if prefix and cls._norm(prefix.group(1)) == cls._norm(speakers[0]) else text
            return [
                {
                    "speaker": speakers[0],
                    "text": spoken_text,
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                }
            ]
        events: list[dict] = []
        pattern = re.compile(r"^\s*([^:]{1,80}):\s*(.+?)\s*$")
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                events = []
                break
            events.append(
                {
                    "speaker": match.group(1).strip(),
                    "text": match.group(2),
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                }
            )
        if events:
            return events
        if speakers:
            return [
                {
                    "speaker": speakers[0],
                    "text": text,
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                }
            ]
        raise ValueError("speech_text exists but no speaking character is declared.")

    @staticmethod
    def _raw_events(shot: dict) -> list[dict]:
        supplied = shot.get("dialogue_events")
        if isinstance(supplied, list) and supplied:
            if any(not isinstance(value, dict) for value in supplied):
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue_events contains a non-object entry.")
            normalized: list[dict] = []
            for value in supplied:
                event = dict(value)
                # Canonical DialogueEvent payloads use speaker_id/speaker_name,
                # while Qwen/legacy inputs use speaker. Preserve both contracts
                # so deterministic rescheduling is idempotent.
                speaker = str(event.get("speaker", "") or "").strip()
                if not speaker:
                    speaker = str(event.get("speaker_name", "") or "").strip()
                if not speaker:
                    speaker = str(event.get("speaker_id", "") or "").strip()
                event["speaker"] = speaker
                normalized.append(event)
            return normalized
        return DialogueTimeline._legacy_events(shot)

    def _resolve_speaker(self, name_or_id: str, shot_characters: list[str]) -> dict:
        key = self._norm(name_or_id)
        if key in self._by_name:
            candidate = self._by_name[key]
        elif str(name_or_id).strip() in self._by_id:
            candidate = self._by_id[str(name_or_id).strip()]
        else:
            candidate = None
            for name in shot_characters:
                if self._norm(name) == key:
                    candidate = self._by_name.get(key)
                    break
        if candidate is None:
            raise ValueError(f"Unknown dialogue speaker: {name_or_id!r}")
        if shot_characters:
            allowed = {self._norm(v) for v in shot_characters}
            if self._norm(candidate.get("name", "")) not in allowed:
                raise ValueError(
                    f"Speaker {candidate.get('name', '')!r} is not bound to this shot."
                )
        return candidate

    def _estimate_duration(self, text: str, event: dict[str, Any] | None = None):
        estimate = self.duration_provider.estimate(text, event)
        if estimate.seconds < self.MIN_EVENT_DURATION:
            return type(estimate)(
                seconds=self.MIN_EVENT_DURATION,
                source=estimate.source,
                exact_for_source=estimate.exact_for_source,
            )
        return estimate

    def schedule_shot(
        self,
        shot: dict,
        *,
        previous_dialogue: DialogueEvent | None = None,
    ) -> list[dict]:
        requested_duration = float(shot.get("duration_seconds", 5.2) or 5.2)
        effective_frames = self.h3_legal_frames(requested_duration)
        duration = effective_frames / self.FPS
        shot["requested_duration_seconds"] = requested_duration
        shot["duration_seconds"] = round(duration, 4)
        shot["frames_per_shot"] = effective_frames
        shot["h3_effective_frames"] = effective_frames
        shot["h3_effective_duration_seconds"] = duration

        shot_characters = list(shot.get("characters", []) or [])
        raw_events = self._raw_events(shot)
        if not raw_events:
            return []

        events: list[DialogueEvent] = []
        cursor = 0.0
        count = len(raw_events)

        for index, raw in enumerate(raw_events, start=1):
            speaker = self._resolve_speaker(
                str(raw.get("speaker", "")).strip(),
                shot_characters,
            )
            text = str(raw.get("text", ""))
            if not text.strip():
                raise ValueError(f"{shot.get('shot_id', '')}: empty dialogue text.")

            continues_prev = bool(raw.get("continues_from_previous_shot", False))
            continues_next = bool(raw.get("continues_to_next_shot", False))

            if continues_prev and previous_dialogue is None:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue cannot continue from a missing previous event."
                )

            if continues_prev and previous_dialogue is not None:
                if speaker.get("character_id", "") != previous_dialogue.speaker_id:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: continued dialogue must use the previous shot's speaker."
                    )

            estimate = self._estimate_duration(text, raw)
            speech_duration = float(estimate.seconds)

            remaining_events = count - index
            reserve_for_future = remaining_events * self.MIN_EVENT_DURATION

            pre = (
                self.DEFAULT_PRE_ROLL
                if index == 1 and not continues_prev
                else 0.0
            )
            post = self.DEFAULT_POST_ROLL if index == count else 0.0

            # Preserve exact dialogue text. Dialogue is never truncated. If the
            # current H3-legal shot is too short, deterministically promote the
            # shot to the smallest legal H3 duration that can contain the event
            # plus any already-reserved future events. Optional breathing margins
            # are only removed when the maximum legal H3 duration is otherwise
            # insufficient.
            required_duration = cursor + pre + speech_duration + post + reserve_for_future

            if required_duration > duration + 1e-6:
                try:
                    _, expanded_duration = self._minimum_h3_duration_for_required_seconds(
                        required_duration
                    )
                except ValueError:
                    expanded_duration = duration

                if expanded_duration > duration + 1e-6:
                    duration = expanded_duration
                    effective_frames = int(round(duration * self.FPS))
                    shot["duration_seconds"] = round(duration, 4)
                    shot["frames_per_shot"] = effective_frames
                    shot["h3_effective_frames"] = effective_frames
                    shot["h3_effective_duration_seconds"] = duration
                    print(
                        f"[DIALOGUE] recovery type=shot_duration_extended "
                        f"shot={shot.get('shot_id', '')} "
                        f"new_duration={duration:.3f}s",
                        flush=True,
                    )

            # Pre/post roll are optional. At the H3 maximum, reclaim those
            # margins before declaring the dialogue unschedulable.
            available = duration - cursor - pre - post - reserve_for_future
            if speech_duration > available + 1e-6 and post > 0.0:
                post = 0.0
                available = duration - cursor - pre - reserve_for_future
            if speech_duration > available + 1e-6 and pre > 0.0:
                pre = 0.0
                available = duration - cursor - reserve_for_future

            if speech_duration > available + 1e-6:
                try:
                    _, max_required_duration = self._minimum_h3_duration_for_required_seconds(
                        cursor + pre + speech_duration + post + reserve_for_future
                    )
                except ValueError as exc:
                    detail = (
                        f" Explicit continuation was requested, but the dialogue segment "
                        "cannot fit in the maximum legal H3 shot and must be split."
                        if continues_next
                        else " Increase shot duration or split the dialogue into shorter events."
                    )
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: dialogue event {index} exceeds "
                        f"maximum H3 runtime; estimated_dialogue={speech_duration:.3f}s, "
                        f"available={available:.3f}s.{detail}"
                    ) from exc

                # Defensive guard: the helper should never report a duration that
                # still cannot contain the required event. Keep this deterministic
                # rather than silently clipping or shifting the spoken text.
                if max_required_duration <= duration + 1e-6:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: dialogue event {index} cannot be "
                        f"scheduled without exceeding H3 runtime; estimated_dialogue="
                        f"{speech_duration:.3f}s, available={available:.3f}s."
                    )
                duration = max_required_duration
                effective_frames = int(round(duration * self.FPS))
                shot["duration_seconds"] = round(duration, 4)
                shot["frames_per_shot"] = effective_frames
                shot["h3_effective_frames"] = effective_frames
                shot["h3_effective_duration_seconds"] = duration
                print(
                    f"[DIALOGUE] recovery type=shot_duration_extended "
                    f"shot={shot.get('shot_id', '')} "
                    f"new_duration={duration:.3f}s",
                    flush=True,
                )
                available = duration - cursor - pre - post - reserve_for_future

                if speech_duration > available + 1e-6:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: dialogue timing exceeds the "
                        f"maximum schedulable H3 runtime ({duration:.3f}s)."
                    )

            start = cursor + pre
            end = start + speech_duration

            if end > duration - post + 1e-6:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue timing exceeds the H3-effective "
                    "shot boundary after optional margins are applied."
                )

            event = DialogueEvent(
                dialogue_id=f"{shot.get('shot_id', 'shot')}_dialogue_{index:03d}",
                speaker_id=str(speaker.get("character_id", "")).strip(),
                speaker_name=str(speaker.get("name", "")).strip(),
                text=text,
                start_seconds=start,
                end_seconds=end,
                pre_roll_seconds=pre,
                post_roll_seconds=post,
                continues_from_previous_shot=continues_prev,
                continues_to_next_shot=continues_next,
                expected_duration_ms=(
                    int(round(speech_duration * 1000.0))
                    if estimate.exact_for_source
                    else None
                ),
                duration_source=estimate.source,
            )
            events.append(event)
            cursor = end + post

        previous_end = 0.0
        for event in events:
            if event.start_seconds < previous_end - 1e-6:
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue events overlap.")
            if event.end_seconds > duration + 1e-6:
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue exceeds H3-effective shot duration.")
            previous_end = event.end_seconds

        return [event.to_dict() for event in events]

    @classmethod
    def snapshot_source_dialogue(cls, plan: dict) -> dict[str, list[dict[str, str]]]:
        """Capture explicit dialogue before deterministic scheduling.

        The snapshot is intentionally limited to source identity/text, not timing,
        because timing is allowed to change during H3-legal normalization.  This
        provides a hard invariant against silently dropping or rewriting spoken
        content while the speaker identity is canonicalized upstream.
        """
        snapshot: dict[str, list[dict[str, str]]] = {}
        for shot in plan.get("shots", []) or []:
            if not isinstance(shot, dict):
                continue
            shot_id = str(shot.get("shot_id", "") or "").strip()
            if not shot_id:
                continue
            events = cls._raw_events(shot)
            if not events:
                continue
            snapshot[shot_id] = [
                {
                    "speaker": str(event.get("speaker", "") or "").strip(),
                    "text": str(event.get("text", "")),
                }
                for event in events
            ]
        return snapshot

    @classmethod
    def assert_source_dialogue_preserved(
        cls,
        plan: dict,
        source_snapshot: dict[str, list[dict[str, str]]],
    ) -> None:
        """Fail closed if deterministic normalization drops explicit dialogue.

        Speaker names are allowed to become canonical names during upstream
        entity resolution, so this invariant checks source identity presence and
        exact spoken text, while the final speaker_id/name fields are validated
        separately for canonical binding.
        """
        final_by_shot = {
            str(shot.get("shot_id", "") or "").strip(): shot
            for shot in (plan.get("shots", []) or [])
            if isinstance(shot, dict) and str(shot.get("shot_id", "") or "").strip()
        }

        for shot_id, expected_events in source_snapshot.items():
            shot = final_by_shot.get(shot_id)
            if shot is None:
                raise ValueError(
                    f"Dialogue contract violation: source dialogue shot {shot_id!r} disappeared."
                )

            final_events = shot.get("dialogue_events", []) or []
            if not isinstance(final_events, list):
                raise ValueError(
                    f"Dialogue contract violation: {shot_id!r} dialogue_events is not a list."
                )
            if len(final_events) != len(expected_events):
                raise ValueError(
                    f"Dialogue contract violation: {shot_id!r} changed dialogue event count "
                    f"from {len(expected_events)} to {len(final_events)}."
                )

            for index, (expected, actual) in enumerate(zip(expected_events, final_events), start=1):
                actual_text = str(actual.get("text", ""))
                if actual_text != expected["text"]:
                    raise ValueError(
                        f"Dialogue contract violation: {shot_id!r} event {index} text was altered."
                    )
                speaker_name = str(actual.get("speaker_name", "") or "").strip()
                speaker_id = str(actual.get("speaker_id", "") or "").strip()
                if not speaker_name or not speaker_id:
                    raise ValueError(
                        f"Dialogue contract violation: {shot_id!r} event {index} has no canonical speaker binding."
                    )

    def apply_to_plan(self, plan: dict) -> None:
        previous_by_scene: dict[str, DialogueEvent | None] = {}
        continuation_by_scene: dict[str, bool] = {}
        previous_shot_by_scene: dict[str, dict | None] = {}

        shots = [
            shot
            for shot in (plan.get("shots", []) or [])
            if isinstance(shot, dict)
        ]
        last_shot_by_scene: dict[str, dict] = {}
        for shot in shots:
            scene_id = str(shot.get("scene_id", ""))
            if scene_id:
                last_shot_by_scene[scene_id] = shot

        for shot in shots:
            scene_id = str(shot.get("scene_id", ""))
            boundary = bool(shot.get("is_scene_boundary", False))
            previous = None if boundary else previous_by_scene.get(scene_id)

            requested = self._raw_events(shot)

            # Normalize event-level continuation ownership before scheduling.
            # Only the first event may continue from a prior shot, and only the
            # final event may continue to the next shot. Scene boundaries never
            # inherit dialogue continuation from an earlier scene.
            for event in requested:
                event["continues_from_previous_shot"] = bool(
                    event.get("continues_from_previous_shot", False)
                )
                event["continues_to_next_shot"] = bool(
                    event.get("continues_to_next_shot", False)
                )
            for event in requested[1:]:
                event["continues_from_previous_shot"] = False
            for event in requested[:-1]:
                event["continues_to_next_shot"] = False

            current_continues_from = bool(
                requested and requested[0].get("continues_from_previous_shot", False)
            )

            if boundary:
                # A scene-boundary shot starts a new dialogue segment.
                if requested:
                    requested[0]["continues_from_previous_shot"] = False
                current_continues_from = False
                continuation_by_scene[scene_id] = False
                previous = None

            if previous is not None:
                previous_flag = bool(
                    continuation_by_scene.get(scene_id, False)
                )
                previous_shot = previous_shot_by_scene.get(scene_id)

                if not requested:
                    # A shot with no dialogue cannot continue an earlier speech
                    # segment. Clear the previous boundary explicitly so stale
                    # continuation metadata cannot leak into the next shot.
                    previous.continues_to_next_shot = False
                    previous_shot = previous_shot_by_scene.get(scene_id)
                    if previous_shot is not None:
                        previous_events = previous_shot.get("dialogue_events", [])
                        if previous_events:
                            previous_events[-1]["continues_to_next_shot"] = False
                else:
                    # Boundary metadata may disagree after upstream dialogue
                    # filtering. Reconcile it deterministically rather than
                    # failing on stale flags. The OR preserves any explicit
                    # continuation request; schedule_shot() then enforces the
                    # semantic same-speaker requirement.
                    continuation = previous_flag or current_continues_from
                    requested[0]["continues_from_previous_shot"] = continuation
                    # _raw_events() returns defensive copies. Persist the
                    # reconciled boundary flag on the shot so schedule_shot()
                    # sees the corrected value when it reads the events again.
                    shot["dialogue_events"] = requested
                    if previous_shot is not None:
                        previous_events = previous_shot.get("dialogue_events", [])
                        if previous_events:
                            previous_events[-1]["continues_to_next_shot"] = continuation

            events = self.schedule_shot(
                shot,
                previous_dialogue=previous,
            )
            shot["dialogue_events"] = events
            shot["speaking_characters"] = [event["speaker_name"] for event in events]
            default_language = str(
                shot.get("dialogue_language")
                or shot.get("language")
                or plan.get("dialogue_language")
                or plan.get("language")
                or "English"
            ).strip() or "English"
            shot["language"] = default_language
            shot["speech_text"] = "\n".join(
                f"({event['speaker_id']}) says: <d>[{event.get('language', default_language)}] {event['text']}</d>"
                for event in events
            )

            if shot is last_shot_by_scene.get(scene_id) and events:
                # A scene has no legal dialogue continuation edge to another
                # scene. Persist the invariant in the canonical event payload.
                for event in events:
                    event["continues_to_next_shot"] = False

            previous_by_scene[scene_id] = (
                DialogueEvent(**events[-1]) if events else None
            )
            previous_shot_by_scene[scene_id] = shot if events else None
            continuation_by_scene[scene_id] = bool(
                events and events[-1]["continues_to_next_shot"]
            )

    @staticmethod
    def validate_rendered_media(
        media_path: str,
        *,
        tolerance_seconds: float = 0.30,
    ) -> dict[str, Any]:
        from pipeline.dialogue_duration import FFProbeMediaDurationProvider

        return FFProbeMediaDurationProvider().validate_video_audio_sync(
            media_path,
            tolerance_seconds=tolerance_seconds,
        )
