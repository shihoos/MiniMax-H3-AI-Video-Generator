from __future__ import annotations

import re
from typing import Any

from schemas.dialogue import DialogueEvent
from pipeline.dialogue_duration import DialogueDurationProvider, ExplicitOrWPMDurationProvider


class DialogueTimeline:
    """Deterministically canonicalize and schedule shot dialogue.

    Timing is dynamic: no fixed two-second airlock is imposed. A modest
    settle/reaction budget is reserved when the shot has enough duration.
    """

    DEFAULT_PRE_ROLL = 0.35
    DEFAULT_POST_ROLL = 0.35
    MIN_EVENT_DURATION = 0.60
    WORDS_PER_MINUTE = 145.0

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

    @classmethod
    def _legacy_events(cls, shot: dict) -> list[dict]:
        text = str(shot.get("speech_text", "") or "")
        speakers = [str(v).strip() for v in (shot.get("speaking_characters", []) or []) if str(v).strip()]
        if not text.strip():
            return []
        if len(speakers) == 1:
            return [{"speaker": speakers[0], "text": text, "continues_to_next_shot": False}]
        events: list[dict] = []
        pattern = re.compile(r"^\s*([^:]{1,80}):\s*(.+?)\s*$")
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                events = []
                break
            events.append({
                "speaker": match.group(1).strip(),
                "text": match.group(2),
                "continues_to_next_shot": False,
            })
        if events:
            return events
        # Legacy ambiguity: preserve exact text and bind to the first declared
        # speaker rather than inventing or rewriting dialogue.
        if speakers:
            return [{"speaker": speakers[0], "text": text, "continues_to_next_shot": False}]
        raise ValueError("speech_text exists but no speaking character is declared.")

    def _raw_events(self, shot: dict) -> list[dict]:
        supplied = shot.get("dialogue_events")
        if isinstance(supplied, list) and supplied:
            return [v for v in supplied if isinstance(v, dict)]
        return self._legacy_events(shot)

    def _estimate_duration(
        self,
        text: str,
        event: dict[str, Any] | None = None,
    ):
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
        duration = float(shot.get("duration_seconds", 5.2) or 5.2)
        shot_characters = list(shot.get("characters", []) or [])
        raw_events = self._raw_events(shot)
        if not raw_events:
            return []

        events: list[DialogueEvent] = []
        cursor = 0.0
        count = len(raw_events)
        for index, raw in enumerate(raw_events, start=1):
            speaker = self._resolve_speaker(str(raw.get("speaker", "")).strip(), shot_characters)
            text = str(raw.get("text", ""))
            if not text.strip():
                raise ValueError(f"{shot.get('shot_id', '')}: empty dialogue text.")
            continues_prev = bool(raw.get("continues_from_previous_shot", False))
            continues_next = bool(raw.get("continues_to_next_shot", False))
            if continues_prev and previous_dialogue is None:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue cannot continue from a missing previous event."
                )
            duration_estimate = self._estimate_duration(text, raw)
            speech_duration = duration_estimate.seconds
            remaining_events = count - index
            reserve_for_future = remaining_events * (self.MIN_EVENT_DURATION + 0.15)
            pre = self.DEFAULT_PRE_ROLL if cursor == 0.0 and not continues_prev else 0.0
            post = self.DEFAULT_POST_ROLL if index == count else 0.0
            available = duration - cursor - pre - post - reserve_for_future
            actual_duration = min(speech_duration, max(self.MIN_EVENT_DURATION, available))
            if actual_duration < speech_duration - 1e-6 and not continues_next:
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue does not fit shot duration; "
                    "increase shot duration or mark the line as an explicit continuation."
                )
            start = cursor + pre
            end = min(duration - post, start + actual_duration)
            if end <= start:
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue timing collapsed.")
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
                    if duration_estimate.exact_for_source
                    else None
                ),
                duration_source=duration_estimate.source,
            )
            events.append(event)
            cursor = end + post

        # Hard deterministic overlap/fit checks.
        previous_end = 0.0
        for event in events:
            if event.start_seconds < previous_end - 1e-6:
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue events overlap.")
            if event.end_seconds > duration + 1e-6:
                raise ValueError(f"{shot.get('shot_id', '')}: dialogue exceeds shot duration.")
            previous_end = event.end_seconds
        return [event.to_dict() for event in events]

    def apply_to_plan(self, plan: dict) -> None:
        previous_by_scene: dict[str, DialogueEvent | None] = {}
        previous_continuation_by_scene: dict[str, bool] = {}
        for shot in plan.get("shots", []) or []:
            scene_id = str(shot.get("scene_id", ""))
            previous = None if bool(shot.get("is_scene_boundary", False)) else previous_by_scene.get(scene_id)
            requested = self._raw_events(shot)
            current_continues_from = bool(
                requested
                and requested[0].get("continues_from_previous_shot", False)
            )
            if previous_continuation_by_scene.get(scene_id, False) != current_continues_from and (
                previous is not None or current_continues_from
            ):
                raise ValueError(
                    f"{shot.get('shot_id', '')}: dialogue continuation flags do not match across the shot boundary."
                )
            if current_continues_from and previous is not None:
                speaker = str(requested[0].get("speaker", "")).strip()
                if self._norm(speaker) not in {
                    self._norm(previous.speaker_name),
                    previous.speaker_id,
                }:
                    raise ValueError(
                        f"{shot.get('shot_id', '')}: continued dialogue must use the previous shot's speaker."
                    )

            events = self.schedule_shot(shot, previous_dialogue=previous)
            shot["dialogue_events"] = events
            shot["speaking_characters"] = [event["speaker_name"] for event in events]
            shot["speech_text"] = "\n".join(
                f"{event['speaker_name']}: {event['text']}" for event in events
            )
            previous_by_scene[scene_id] = (
                DialogueEvent(**events[-1]) if events else None
            )
            previous_continuation_by_scene[scene_id] = bool(
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
