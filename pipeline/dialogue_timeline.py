from __future__ import annotations

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
            return [dict(value) for value in supplied]
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

            # Preserve exact dialogue text. We never silently truncate speech
            # to squeeze it into a too-short shot. Instead, reduce optional
            # breathing margins and then fail with an actionable diagnostic.
            available = duration - cursor - pre - post - reserve_for_future

            if available < self.MIN_EVENT_DURATION:
                if post > 0.0:
                    post = 0.0
                    available = duration - cursor - pre - reserve_for_future
                if available < self.MIN_EVENT_DURATION and pre > 0.0:
                    pre = 0.0
                    available = duration - cursor - reserve_for_future

            if available < self.MIN_EVENT_DURATION:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: insufficient H3 runtime for "
                    f"dialogue event {index}; effective_duration={duration:.3f}s, "
                    f"remaining_capacity={max(0.0, available):.3f}s. "
                    "Increase shot duration or split/explicitly continue the dialogue."
                )

            if speech_duration > available + 1e-6:
                if continues_next:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: explicit continuation is marked, "
                        "but the current dialogue segment still exceeds the available "
                        "runtime. Split the text into shorter events before scheduling."
                    )
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue does not fit the H3-effective "
                    f"shot duration ({duration:.3f}s). Estimated dialogue duration="
                    f"{speech_duration:.3f}s; available={available:.3f}s."
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

    def apply_to_plan(self, plan: dict) -> None:
        previous_by_scene: dict[str, DialogueEvent | None] = {}
        continuation_by_scene: dict[str, bool] = {}

        for shot in plan.get("shots", []) or []:
            scene_id = str(shot.get("scene_id", ""))
            boundary = bool(shot.get("is_scene_boundary", False))
            previous = None if boundary else previous_by_scene.get(scene_id)

            requested = self._raw_events(shot)
            current_continues_from = bool(
                requested and requested[0].get("continues_from_previous_shot", False)
            )

            if boundary and current_continues_from:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: a scene-boundary shot cannot continue "
                    "dialogue from the previous scene."
                )

            if previous is not None:
                if continuation_by_scene.get(scene_id, False) != current_continues_from:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: dialogue continuation flags do not "
                        "match across the shot boundary."
                    )

            events = self.schedule_shot(
                shot,
                previous_dialogue=previous,
            )
            shot["dialogue_events"] = events
            shot["speaking_characters"] = [event["speaker_name"] for event in events]
            shot["speech_text"] = "\n".join(
                f"({event['speaker_id']}) says: <d>[English] {event['text']}</d>"
                for event in events
            )

            previous_by_scene[scene_id] = (
                DialogueEvent(**events[-1]) if events else None
            )
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
