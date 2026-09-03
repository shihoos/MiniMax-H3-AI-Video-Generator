from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

from planner.cinematic_compiler import CinematicCompiler
from planner.entity_resolver import EntityResolver
from pipeline.production_checkpoint import ProductionCheckpoint

from planner.config import (
    AI_STORY_MODE,
    DIRECTOR_KAGGLE_INPUT_ROOT,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_MODEL_ENV,
    DIRECTOR_MODEL_FILENAME,
    DIRECTOR_N_BATCH,
    DIRECTOR_N_CTX,
    DIRECTOR_N_GPU_LAYERS,
    DIRECTOR_TEMPERATURE,
    DIRECTOR_THREADS,
    DIRECTOR_TOP_P,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_enabled,
)


class QwenDirector:

    # Production planning limits. Keep the narrative rich, but bound the
    # production graph so an LLM cannot accidentally explode a short film
    # into dozens of scenes and therefore dozens of expensive Qwen calls.
    MAX_SCENES = 6
    SHOTS_PER_SCENE = 2
    # Creative shot batching may cover up to two fresh adjacent scenes.
    # The runtime selects the largest batch that still fits the 8K context
    # budget with a bounded completion reserve. Missing shots are deterministic
    # fallbacks; no per-scene Qwen recovery is used.
    MAX_SHOT_BATCH_SCENES = 2

    FORBIDDEN_CHARACTER_NAMES = {
        "treat",
        "develop",
        "clarify",
        "every",
        "above",
        "far",
        "tone",
        "visual",
        "story",
        "scene",
        "scenes",
        "shot",
        "shots",
        "camera",
        "lighting",
        "sound",
        "soundscape",
        "environment",
        "action",
        "continuity",
        "mood",
        "location",
        "weather",
        "dialogue",
        "music",
        "character",
        "characters",
        "the",
        "a",
        "an",
        "he",
        "she",
        "his",
        "her",
        "it",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "when",
        "while",
        "after",
        "before",
        "finally",
        "suddenly",
        "meanwhile",
        "developing",
        "preserve",
        "expand",
        "generate",
        "generation",
        "priority",
        "description",
        "details",
        "detail",
        "camera_shot",
        "camera_movement",
        "negative_prompt",
        "visual_prompt",
    }

    VALID_GENERIC_ROLES = {
        "man",
        "woman",
        "girl",
        "boy",
        "child",
        "person",
        "hero",
        "heroine",
        "explorer",
        "detective",
        "scientist",
        "soldier",
        "warrior",
        "king",
        "queen",
        "robot",
        "android",
        "pilot",
    }

    _MODE_LABELS = {
        AI_STORY_MODE: "AI STORY MODE",
        EXPAND_USER_STORY_MODE: "EXPAND STORY MODE",
        PRESERVE_USER_STORY_MODE: "PRESERVE STORY MODE",
    }

    def __init__(
        self,
        project_root: Path,
    ):

        self.project_root = (
            Path(
                project_root
            )
            .resolve()
        )

        self._llama = None

        self._model_path = (
            self._find_model()
            if director_enabled()
            else None
        )

        self._fallback_planner = None
        self._entity_resolver = EntityResolver(self)
        self._current_visual_language: dict = {}
        self._reference_visual_context: dict[str, dict] = {}

        # Optional development diagnostics. Both are disabled unless the
        # corresponding environment variable is explicitly configured.
        self._trace_dir = self._optional_directory_env(
            "H3_DIRECTOR_TRACE_DIR"
        )
        self._cache_dir = self._optional_directory_env(
            "H3_DIRECTOR_CACHE_DIR"
        )
        self._cache_namespace = "minimax-h3-qwen-schema-v2"

        # Runtime Qwen telemetry is intentionally lightweight: keep only
        # aggregate/per-call metrics needed to diagnose latency, token usage,
        # retries, cache behavior, and deterministic recovery decisions.
        self._qwen_telemetry = {
            "calls": [],
            "total_elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries": 0,
            "cache_hits": 0,
            "deterministic_recoveries": 0,
        }

    def set_reference_visual_context(self, context: dict[str, dict] | None) -> None:
        self._reference_visual_context = {
            str(key): dict(value)
            for key, value in (context or {}).items()
            if isinstance(value, dict)
        }

    @staticmethod
    def _optional_directory_env(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        if not value:
            return None
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _strip_thinking(text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(
            r"<think>.*?</think>",
            "",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"<think>", value, flags=re.IGNORECASE):
            value = re.split(
                r"<think>",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        return value

    def _record_qwen_call(
        self,
        *,
        call_name: str,
        elapsed: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        max_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format=None,
        cache_hit: bool = False,
        error: str = "",
    ) -> None:
        """Record and print bounded runtime telemetry for one Qwen call."""
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        elapsed = max(0.0, float(elapsed or 0.0))

        record = {
            "call_name": str(call_name or "unknown"),
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "decode_tps": (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            ),
            "max_tokens": int(max_tokens or 0),
            "temperature": temperature,
            "top_p": top_p,
            "response_format": (
                "json_schema"
                if isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                else (
                    response_format.get("type")
                    if isinstance(response_format, dict)
                    else None
                )
            ),
            "cache_hit": bool(cache_hit),
            "error": str(error or ""),
        }

        calls = self._qwen_telemetry.setdefault("calls", [])
        calls.append(record)

        self._qwen_telemetry["total_elapsed_seconds"] += elapsed
        self._qwen_telemetry["prompt_tokens"] += prompt_tokens
        self._qwen_telemetry["completion_tokens"] += completion_tokens
        if cache_hit:
            self._qwen_telemetry["cache_hits"] += 1

        if "retry" in str(call_name).lower():
            self._qwen_telemetry["retries"] += 1

        print(
            "[QWEN]",
            call_name,
            f"elapsed={elapsed:.2f}s",
            f"prompt_tokens={prompt_tokens}",
            f"completion_tokens={completion_tokens}",
            f"total_tokens={prompt_tokens + completion_tokens}",
            f"decode_tps={record['decode_tps']:.2f}",
            f"max_tokens={int(max_tokens or 0)}",
            (f"cache_hit={cache_hit}" if cache_hit else ""),
            (f"error={error}" if error else ""),
            flush=True,
        )

    def _record_recovery(
        self,
        recovery_type: str,
        detail: str = "",
    ) -> None:
        self._qwen_telemetry["deterministic_recoveries"] += 1
        print(
            "[QWEN]",
            "recovery",
            f"type={recovery_type}",
            (f"detail={detail}" if detail else ""),
            flush=True,
        )

    def _print_qwen_summary(self) -> None:
        """Print a compact production-level Qwen accounting summary."""
        calls = list(self._qwen_telemetry.get("calls", []) or [])
        total_elapsed = float(
            self._qwen_telemetry.get("total_elapsed_seconds", 0.0) or 0.0
        )
        prompt_tokens = int(
            self._qwen_telemetry.get("prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            self._qwen_telemetry.get("completion_tokens", 0) or 0
        )
        retries = int(self._qwen_telemetry.get("retries", 0) or 0)
        cache_hits = int(self._qwen_telemetry.get("cache_hits", 0) or 0)
        recoveries = int(
            self._qwen_telemetry.get("deterministic_recoveries", 0) or 0
        )

        print("[QWEN] ==================== SUMMARY ====================", flush=True)
        print("[QWEN] calls=" + str(len(calls)), flush=True)
        print("[QWEN] prompt_tokens=" + str(prompt_tokens), flush=True)
        print("[QWEN] completion_tokens=" + str(completion_tokens), flush=True)
        print("[QWEN] total_tokens=" + str(prompt_tokens + completion_tokens), flush=True)
        print("[QWEN] total_elapsed=" + f"{total_elapsed:.2f}s", flush=True)
        print("[QWEN] retries=" + str(retries), flush=True)
        print("[QWEN] cache_hits=" + str(cache_hits), flush=True)
        print("[QWEN] deterministic_recoveries=" + str(recoveries), flush=True)
        if calls:
            names = ", ".join(str(item.get("call_name", "unknown")) for item in calls)
            print("[QWEN] call_sequence=" + names, flush=True)
        print("[QWEN] =====================================================", flush=True)

    def _trace_call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        response,
        elapsed: float,
        error: str = "",
        response_format=None,
    ) -> None:
        if self._trace_dir is None:
            return

        raw_content = ""
        if isinstance(response, dict):
            try:
                raw_content = str(
                    response["choices"][0]["message"]["content"] or ""
                )
            except Exception:
                raw_content = ""

        payload = {
            "call_name": call_name,
            "elapsed_seconds": elapsed,
            "error": error,
            "response_format": response_format,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_content": raw_content,
            "raw_response": response,
        }
        digest = hashlib.sha256(
            (
                self._cache_namespace
                + "\n"
                + call_name
                + "\n"
                + system_prompt
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest()
        path = self._trace_dir / f"{call_name.replace(':', '_')}_{digest[:16]}.json"
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                "[QWEN]",
                "trace_write_failed",
                call_name,
                str(exc),
                flush=True,
            )

    def _cache_key(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None,
    ) -> str:
        material = json.dumps(
            {
                "namespace": self._cache_namespace,
                "call_name": call_name,
                "system": system_prompt,
                "user": user_prompt,
                "schema": response_schema,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()

    def _cache_read(
        self,
        key: str,
    ) -> dict | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            result = payload.get("parsed")
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _cache_write(
        self,
        key: str,
        parsed: dict,
        *,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        elapsed: float,
        raw_content: str,
    ) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        payload = {
            "namespace": self._cache_namespace,
            "call_name": call_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "elapsed_seconds": elapsed,
            "raw_content": raw_content,
            "parsed": parsed,
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                "[QWEN]",
                "cache_write_failed",
                call_name,
                str(exc),
                flush=True,
            )

    # ========================================================
    # MODEL DISCOVERY
    # ========================================================

    def _find_model(
        self,
    ) -> Path:

        explicit = os.getenv(
            DIRECTOR_MODEL_ENV,
            "",
        ).strip()

        candidates: list[Path] = []

        if explicit:

            candidates.append(
                Path(
                    explicit
                )
            )

        candidates.extend(
            [
                (
                    self.project_root
                    / "data"
                    / "models"
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    self.project_root
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    DIRECTOR_KAGGLE_INPUT_ROOT
                    / DIRECTOR_MODEL_FILENAME
                ),
            ]
        )

        for root in (
            self.project_root,
            DIRECTOR_KAGGLE_INPUT_ROOT,
        ):

            if not root.exists():
                continue

            try:

                candidates.extend(
                    root.rglob(
                        DIRECTOR_MODEL_FILENAME
                    )
                )

            except OSError:

                continue

        unique: list[Path] = []
        seen: set[str] = set()

        for candidate in candidates:

            candidate = Path(
                candidate
            )

            try:

                key = str(
                    candidate.resolve()
                )

            except OSError:

                key = str(
                    candidate
                )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                candidate
            )

        existing = [
            path
            for path in unique
            if path.is_file()
        ]

        if not existing:

            raise FileNotFoundError(
                "Qwen director model was not found.\n"
                f"Expected filename: "
                f"{DIRECTOR_MODEL_FILENAME}\n"
                "Attach the Kaggle director-model dataset "
                "or set H3_DIRECTOR_MODEL_PATH."
            )

        if len(existing) > 1:

            raise RuntimeError(
                "Multiple Qwen director models were found:\n"
                + "\n".join(
                    str(path)
                    for path in existing
                )
            )

        return existing[0]

    @property
    def model_path(
        self,
    ) -> Path | None:

        return self._model_path

    @property
    def available(
        self,
    ) -> bool:

        return bool(
            director_enabled()
            and self._model_path is not None
            and self._model_path.is_file()
        )

    # ========================================================
    # DETERMINISTIC FALLBACK
    # ========================================================


    # ========================================================
    # FINAL ARCHITECTURE HELPERS
    # ========================================================

    def _resolve_scene_character_aliases(
        self,
        scenes,
        character_names,
    ) -> list[dict]:
        return self._entity_resolver.resolve_scene_aliases(
            scenes,
            character_names,
            qwen_chat=self._chat_json,
        )

    @staticmethod
    def _classify_scene_function(
        scene: dict,
        index: int,
        total: int,
    ) -> str:

        text = " ".join(
            [
                str(
                    scene.get(
                        "title",
                        "",
                    )
                    or ""
                ),
                str(
                    scene.get(
                        "description",
                        "",
                    )
                    or ""
                ),
                str(
                    scene.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ),
            ]
        ).lower()

        # ----------------------------------------------------
        # Structural boundaries
        # ----------------------------------------------------

        if index == 0:
            return "setup"

        if index == total - 1:
            return "finale"

        # ----------------------------------------------------
        # Narrative vocabulary
        #
        # IMPORTANT:
        # Midpoint/revelation must be checked BEFORE catalyst,
        # because words such as "learns", "discovers", and
        # "signal" can appear inside a revelation.
        # ----------------------------------------------------

        climax_terms = (
            "climax",
            "final confrontation",
            "decisive choice",
            "must choose",
            "must shut",
            "collapses",
            "collapse",
            "explodes",
            "final battle",
            "last chance",
            "saves",
            "destroy",
            "destroying",
            "escape",
        )

        midpoint_terms = (
            "midpoint",
            "revelation",
            "reveals the truth",
            "reveals that",
            "reveals",
            "realizes the truth",
            "realizes that",
            "realizes",
            "understands the truth",
            "understands that",
            "understands",
            "discovers the truth",
            "learns the truth",
            "hidden truth",
            "secret is revealed",
            "truth is revealed",
            "turning point",
            "identity",
            "identity is revealed",
        )

        catalyst_terms = (
            "inciting",
            "inciting incident",
            "receives a warning",
            "receives",
            "warning",
            "arrives",
            "unexpected attack",
            "attack",
            "first discovery",
            "initial discovery",
            "discovers a clue",
            "finds a clue",
            "signal appears",
            "signal",
            "learns that",
            "disruption",
        )

        # ----------------------------------------------------
        # Highest-priority dramatic state first.
        # ----------------------------------------------------

        if any(
            term in text
            for term in climax_terms
        ):
            return "climax"

        # A revelation/turning-point interpretation has higher
        # precedence than generic discovery/catalyst language.
        if any(
            term in text
            for term in midpoint_terms
        ):
            return "midpoint"

        if any(
            term in text
            for term in catalyst_terms
        ):
            return "catalyst"

        return "development"

    @classmethod
    def _annotate_scene_functions(
        cls,
        scenes: list[dict],
    ) -> list[dict]:

        result = []

        total = len(
            scenes
        )

        for index, raw_scene in enumerate(
            scenes
        ):

            scene = dict(
                raw_scene
            )

            function = (
                cls._classify_scene_function(
                    scene,
                    index,
                    total,
                )
            )

            description = str(
                scene.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            objective = str(
                scene.get(
                    "scene_objective",
                    "",
                )
                or ""
            ).strip()

            moment = (
                objective
                or
                description
            )

            moment = re.sub(
                r"\s+",
                " ",
                moment,
            ).strip()

            if len(moment) > 220:
                moment = (
                    moment[:220]
                    .rsplit(
                        " ",
                        1,
                    )[0]
                    .strip()
                )

            scene[
                "scene_function"
            ] = function

            scene[
                "obligatory_moment"
            ] = moment

            result.append(
                scene
            )

        return result

    @classmethod
    def _scene_function_coverage(
        cls,
        scenes: list[dict],
    ) -> set[str]:

        return {
            str(
                scene.get(
                    "scene_function",
                    "",
                )
                or ""
            ).strip().lower()
            for scene in scenes or []
            if isinstance(
                scene,
                dict,
            )
        }

    @staticmethod
    def _merge_scene_group(
        group: list[dict],
        index: int,
    ) -> dict:

        first = dict(
            group[0]
        )

        descriptions = [
            str(item.get("description", "") or "").strip()
            for item in group
            if str(item.get("description", "") or "").strip()
        ]

        story_summaries = [
            str(item.get("story_summary", "") or "").strip()
            for item in group
            if str(item.get("story_summary", "") or "").strip()
        ]
        obligatory = [
            str(item.get("obligatory_moment", "") or "").strip()
            for item in group
            if str(item.get("obligatory_moment", "") or "").strip()
        ]
        narrative_beats = [
            str(item.get(key, "") or "").strip()
            for item in group
            for key in ("narrative_beat", "key_event", "event")
            if str(item.get(key, "") or "").strip()
        ]

        objectives = [
            str(
                item.get(
                    "scene_objective",
                    "",
                )
                or ""
            ).strip()
            for item in group
        ]

        objectives = [
            value
            for value in objectives
            if value
        ]

        continuity = [
            str(
                item.get(
                    "continuity_notes",
                    "",
                )
                or ""
            ).strip()
            for item in group
        ]

        continuity = [
            value
            for value in continuity
            if value
        ]

        env = []
        props = []
        chars = []

        for item in group:

            for value in (
                item.get(
                    "environment_details",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in env:
                    env.append(
                        text
                    )

            for value in (
                item.get(
                    "key_props",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in props:
                    props.append(
                        text
                    )

            for value in (
                item.get(
                    "characters",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in chars:
                    chars.append(
                        text
                    )

        first[
            "scene_id"
        ] = str(
            first.get(
                "scene_id",
                f"scene_{index:03d}",
            )
            or
            f"scene_{index:03d}"
        ).strip()

        first[
            "order"
        ] = index

        first[
            "description"
        ] = " ".join(descriptions).strip()

        if story_summaries:
            first["story_summary"] = " ".join(story_summaries).strip()
        if obligatory:
            first["obligatory_moment"] = " ".join(obligatory).strip()
        if narrative_beats:
            first["narrative_beat"] = " ".join(narrative_beats).strip()

        first[
            "scene_objective"
        ] = " ".join(objectives).strip()

        first[
            "continuity_notes"
        ] = " ".join(
            continuity
        ).strip()

        first[
            "environment_details"
        ] = env[:12]

        first[
            "key_props"
        ] = props[:8]

        first[
            "characters"
        ] = chars[:6]

        return first

    @classmethod
    def _deterministic_compress_scenes(
        cls,
        scenes: list[dict],
        target_count: int = 6,
    ) -> list[dict]:

        if len(scenes) <= target_count:
            return cls._annotate_scene_functions(
                scenes
            )

        target_count = max(
            4,
            min(
                target_count,
                len(scenes),
            )
        )

        total = len(
            scenes
        )

        groups = []

        start = 0

        for group_index in range(
            target_count
        ):

            remaining_items = (
                total - start
            )

            remaining_groups = (
                target_count
                - group_index
            )

            size = (
                (remaining_items + remaining_groups - 1)
                // remaining_groups
            )

            end = min(
                total,
                start + size,
            )

            groups.append(
                scenes[
                    start:end
                ]
            )

            start = end

        merged = []

        for index, group in enumerate(
            groups,
            start=1,
        ):

            merged.append(
                cls._merge_scene_group(
                    group,
                    index,
                )
            )

        # Re-tag after merging because the structural positions changed.
        return cls._annotate_scene_functions(
            merged
        )

    @classmethod
    def _deterministic_expand_scenes(
        cls,
        story: str,
        existing_scenes: list[dict],
        characters: list[dict],
    ) -> list[dict]:

        story = str(
            story or ""
        ).strip()

        if not story:
            return []

        sentences = [
            sentence.strip()
            for sentence
            in re.split(
                r"(?<=[.!?])\s+",
                story,
            )
            if sentence.strip()
        ]

        if len(sentences) < 4:
            # If source prose itself is short, use the existing scenes
            # and do not fabricate events.
            return cls._annotate_scene_functions(
                existing_scenes
            )

        target = min(
            6,
            max(
                4,
                len(existing_scenes),
            ),
        )

        target = min(
            target,
            len(sentences),
        )

        groups = [
            []
            for _ in range(
                target
            )
        ]

        for index, sentence in enumerate(
            sentences
        ):

            bucket = min(
                target - 1,
                int(
                    index
                    * target
                    /
                    max(
                        1,
                        len(sentences),
                    )
                ),
            )

            groups[
                bucket
            ].append(
                sentence
            )

        outputs = []

        fallback_characters = [
            str(
                item.get(
                    "name",
                    "",
                )
                or ""
            ).strip()
            for item in characters
            if isinstance(
                item,
                dict,
            )
            and
            str(
                item.get(
                    "name",
                    "",
                )
                or ""
            ).strip()
        ]

        for index, group in enumerate(
            groups,
            start=1,
        ):

            description = " ".join(
                group
            ).strip()

            if not description:
                continue

            template = {}

            if existing_scenes:

                template = dict(
                    existing_scenes[
                        min(
                            index - 1,
                            len(existing_scenes) - 1,
                        )
                    ]
                )

            template.update(
                {
                    "scene_id":
                        f"scene_{index:03d}",

                    "title":
                        str(
                            template.get(
                                "title",
                                "",
                            )
                            or ""
                        ).strip()
                        or
                        f"Story Beat {index}",

                    "order":
                        index,

                    "description":
                        description,

                    "characters":
                        fallback_characters[
                            :6
                        ],

                    "scene_objective":
                        "Advance the source narrative while preserving its event.",

                    "continuity_notes":
                        "Preserve chronology and character continuity.",
                }
            )

            outputs.append(
                template
            )

        return cls._annotate_scene_functions(
            outputs
        )


    def _planner(
        self,
    ):

        if self._fallback_planner is None:

            from planner.production_planner import (
                ProductionPlanner,
            )

            self._fallback_planner = (
                ProductionPlanner(
                    self.project_root
                )
            )

        return self._fallback_planner

    def _build_deterministic_fallback(
        self,
        story: str,
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        story = str(
            story or ""
        ).strip()

        if not story:

            return (
                [],
                [],
            )

        planner = self._planner()

        characters = []
        scenes = []

        try:

            characters = (
                planner.create_characters(
                    story
                )
                or []
            )

        except Exception:

            characters = []

        try:

            scenes = (
                planner.create_scenes(
                    story,
                    characters,
                )
                or []
            )

        except Exception:

            scenes = []

        return (
            [
                character.to_dict()
                for character
                in characters
                if character is not None
            ],
            [
                scene.to_dict()
                for scene
                in scenes
                if scene is not None
            ],
        )

    # ========================================================
    # CUDA / MODEL LIFECYCLE
    # ========================================================

    @staticmethod
    def _load_nvidia_cuda_libraries() -> None:

        import site

        site_roots: list[Path] = []

        try:

            site_roots.extend(
                Path(path)
                for path
                in site.getsitepackages()
                if path
            )

        except Exception:
            pass

        try:

            user_site = (
                site.getusersitepackages()
            )

            if user_site:

                site_roots.append(
                    Path(
                        user_site
                    )
                )

        except Exception:
            pass

        # The pinned llama-cpp-python wheel is cu130.
        # Never allow CUDA 12 userspace to be selected accidentally.
        cudart: list[Path] = []
        cublas: list[Path] = []
        cublaslt: list[Path] = []

        for site_root in site_roots:

            nvidia_root = (
                site_root
                / "nvidia"
            )

            if not nvidia_root.is_dir():
                continue

            try:

                cudart.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcudart.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcudart\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

                cublas.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcublas.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcublas\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

                cublaslt.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcublasLt.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcublasLt\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

            except OSError:

                continue

        if not cudart:

            raise RuntimeError(
                "No compatible libcudart.so.13 runtime was found."
            )

        if not cublas:

            raise RuntimeError(
                "No compatible libcublas.so.13 runtime was found."
            )

        if not cublaslt:

            raise RuntimeError(
                "No compatible libcublasLt.so.13 runtime was found."
            )

        cudart_lib = cudart[0]

        matching_cublas = [
            path
            for path
            in cublas
            if path.parent
            == cudart_lib.parent
        ]

        cublas_lib = (
            matching_cublas[0]
            if matching_cublas
            else cublas[0]
        )

        matching_cublaslt = [
            path
            for path
            in cublaslt
            if path.parent
            == cudart_lib.parent
        ]

        cublaslt_lib = (
            matching_cublaslt[0]
            if matching_cublaslt
            else cublaslt[0]
        )

        directories = [
            str(
                cudart_lib.parent
            ),
            str(
                cublas_lib.parent
            ),
            str(
                cublaslt_lib.parent
            ),
        ]

        old_ld = os.environ.get(
            "LD_LIBRARY_PATH",
            "",
        )

        if old_ld:

            directories.append(
                old_ld
            )

        # Put the CUDA 13 userspace directories first so a
        # pre-existing CUDA 12 path cannot win resolution.
        os.environ[
            "LD_LIBRARY_PATH"
        ] = ":".join(
            directories
        )

        try:

            ctypes.CDLL(
                str(
                    cudart_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(
                    cublas_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(
                    cublaslt_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

        except OSError as exc:

            raise RuntimeError(
                "Unable to load NVIDIA CUDA 13 libraries:\n"
                f"CUDA runtime: {cudart_lib}\n"
                f"cuBLAS: {cublas_lib}\n"
                f"cuBLASLt: {cublaslt_lib}\n"
                f"{exc}"
            ) from exc

    def load(
        self,
    ) -> None:

        if not self.available:
            return

        if self._llama is not None:
            return

        self._load_nvidia_cuda_libraries()

        try:

            from llama_cpp import Llama

        except ImportError as exc:

            raise RuntimeError(
                "llama-cpp-python is not installed."
            ) from exc

        try:

            self._llama = Llama(
                model_path=str(
                    self._model_path
                ),
                n_ctx=DIRECTOR_N_CTX,
                n_gpu_layers=DIRECTOR_N_GPU_LAYERS,
                n_batch=DIRECTOR_N_BATCH,
                n_threads=DIRECTOR_THREADS,
                flash_attn=True,
                verbose=False,
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to initialize Qwen3-14B director.\n"
                f"Model: {self._model_path}\n"
                f"Error: {exc}"
            ) from exc

    def unload(
        self,
    ) -> None:

        model = self._llama

        self._llama = None

        if model is not None:

            del model

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                try:

                    torch.cuda.ipc_collect()

                except Exception:
                    pass

        except Exception:
            pass

    # ========================================================
    # TOKEN BUDGET
    # ========================================================

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if self._llama is None:

            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        return len(
            self._llama.tokenize(
                text.encode(
                    "utf-8"
                ),
                add_bos=True,
                special=True,
            )
        )

    def _available_output_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
    ) -> tuple[int, int]:

        context = int(
            DIRECTOR_N_CTX
        )

        safety = 128

        prompt_tokens = (
            self._count_tokens(
                system_prompt
                + "\n\n"
                + user_prompt
            )
        )

        available = (
            context
            - prompt_tokens
            - safety
        )

        if available < minimum_completion:

            raise RuntimeError(
                "Qwen director prompt is too large "
                f"for the {context}-token context window.\n"
                f"Prompt tokens: {prompt_tokens}.\n"
                f"Available completion tokens: {available}."
            )

        return (
            prompt_tokens,
            min(
                int(
                    DIRECTOR_MAX_TOKENS
                ),
                available,
            ),
        )

    # ========================================================
    # MODE PROMPTS
    # ========================================================

    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return """
AI STORY MODE.

Treat the user input as a PREMISE, not a finished story.

Create a genuinely developed cinematic narrative.

Build:
- a memorable protagonist;
- meaningful supporting characters;
- a clear desire or objective;
- conflict;
- escalating complications;
- emotional or thematic progression;
- a strong climax;
- a satisfying resolution.

The resulting story should feel like a real short-film
story, not a paraphrase of the premise.

You are allowed to invent details that improve the story.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:

            return """
EXPAND STORY MODE.

The user supplied an existing story.

The supplied story is the source of truth for its core events,
characters, chronology, setting and outcome.

Your task is to make that story substantially more compelling
without replacing it.

Preserve:
- important characters;
- important events;
- chronology;
- setting;
- outcome;
- explicit constraints.

Enrich it with:
- motivation;
- emotional depth;
- cause and effect;
- transitions;
- intermediate events;
- stakes;
- tension;
- sensory detail;
- character reactions;
- meaningful dialogue where appropriate;
- stronger escalation;
- a clearer dramatic progression;
- richer ending consequences.

Do NOT merely add adjectives.

Do NOT simply convert prose into camera directions.

The result must feel like a professionally expanded short-film
story while remaining recognizably the same story.
""".strip()

        if mode == PRESERVE_USER_STORY_MODE:

            return """
PRESERVE STORY MODE.

The supplied story text is immutable.

Do not rewrite it.

Do not add narrative events.

Use the supplied story as-is and create only the production
structure needed to visualize it.
""".strip()

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )



    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return """
You are the narrative writer for MiniMax H3.

The user provides a premise.

Write a complete cinematic short-film story.
Add a protagonist objective, meaningful conflict,
escalation, character reactions, a climax, and a resolution.
Add substantive events; do not merely paraphrase the premise.

Output ONLY the story prose.
Do not output JSON.
Do not output labels, analysis, notes, camera directions,
or explanations.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:
            return """
You are the narrative expansion writer for MiniMax H3.

Expand the supplied story substantially while preserving
its important characters, events, chronology, setting,
outcome, and explicit constraints.

Add motivation, emotional depth, cause and effect,
transitions, intermediate events, stakes, tension,
sensory detail, character reactions, stronger escalation,
and richer consequences.

Do not merely add adjectives.
Do not replace the original plot.
Do not convert the story into camera directions.

Output ONLY the expanded story prose.
Do not output JSON, labels, analysis, notes, or explanations.
""".strip()

        raise ValueError(
            "Preserve Story does not use a story-text pass."
        )

    def _story_text_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return (
            "MODE: "
            + str(mode)
            + "\n\nSOURCE STORY / PREMISE:\n"
            + self._limit_text(
                story,
                7000,
            )
        )

   


    def _sampling_for_mode(
        self,
        mode: str,
    ) -> tuple[float, float]:
    
        if mode == AI_STORY_MODE:
            return (
                0.78,
                0.90,
            )
    
        if mode == EXPAND_USER_STORY_MODE:
            return (
                0.62,
                0.88,
            )
    
        if mode == PRESERVE_USER_STORY_MODE:
            return (
                0.10,
                0.80,
            )
    
        return (
            DIRECTOR_TEMPERATURE,
            DIRECTOR_TOP_P,
        )
    
    def _shot_sampling(
        self,
    ) -> tuple[float, float]:

        return (
            0.68,
            0.92,
        )

    # ========================================================
    # STRUCTURED OUTPUT SCHEMAS
    # ========================================================

    @staticmethod
    def _metadata_json_schema() -> dict:
        scene_properties = {
            "scene_id": {"type": "string"},
            "title": {"type": "string"},
            "order": {"type": "integer"},
            "location": {"type": "string"},
            "time_of_day": {"type": "string"},
            "weather": {"type": "string"},
            "atmosphere": {"type": "string"},
            "description": {"type": "string"},
            "mood": {"type": "string"},
            "lighting": {"type": "string"},
            "color_temperature": {"type": "string"},
            "environment_details": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "key_props": {
                "type": "array",
                "items": {"type": "string"},
            },
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scene_objective": {"type": "string"},
            "continuity_notes": {"type": "string"},
        }

        character_properties = {
            "character_id": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "description": {"type": "string"},
            "personality": {"type": "string"},
            "appearance": {"type": "object"},
            "clothing": {"type": "object"},
            "distinctive_features": {
                "type": "array",
                "items": {"type": "string"},
            },
            "character_state": {"type": "object"},
            "continuity_rules": {
                "type": "array",
                "items": {"type": "string"},
            },
        }

        return {
            "type": "object",
            "properties": {
                "director_notes": {"type": "string"},
                "visual_language": {
                    "type": "object",
                    "properties": {
                        "genre_tone": {"type": "string"},
                        "color_palette": {"type": "string"},
                        "lighting_philosophy": {"type": "string"},
                        "camera_philosophy": {"type": "string"},
                        "pacing": {"type": "string"},
                    },
                    "required": [
                        "genre_tone",
                        "color_palette",
                        "lighting_philosophy",
                        "camera_philosophy",
                        "pacing",
                    ],
                    "additionalProperties": False,
                },
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": character_properties,
                        "required": [
                            "name",
                            "role",
                        ],
                        "additionalProperties": False,
                    },
                },
                "scenes": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": scene_properties,
                        "required": [
                            "scene_id",
                            "title",
                            "order",
                            "location",
                            "time_of_day",
                            "weather",
                            "atmosphere",
                            "description",
                            "mood",
                            "lighting",
                            "color_temperature",
                            "environment_details",
                            "key_props",
                            "characters",
                            "scene_objective",
                            "continuity_notes",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "director_notes",
                "visual_language",
                "characters",
                "scenes",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _scene_json_schema() -> dict:
        metadata = QwenDirector._metadata_json_schema()
        return {
            "type": "object",
            "properties": {
                "scenes": metadata["properties"]["scenes"],
            },
            "required": ["scenes"],
            "additionalProperties": False,
        }

    @staticmethod
    def _scene_compression_json_schema() -> dict:
        return QwenDirector._scene_json_schema()

    @staticmethod
    def _shot_json_schema(
        min_items: int = 2,
        max_items: int = 2,
    ) -> dict:
        shot_properties = {
            "shot_id": {"type": "string"},
            "scene_id": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "location": {"type": "string"},
            "action": {"type": "string"},
            "camera_shot": {"type": "string"},
            "camera_movement": {"type": "string"},
            "lens_and_depth_of_field": {"type": "string"},
            "composition_notes": {"type": "string"},
            "lighting": {"type": "string"},
            "color_temperature": {"type": "string"},
            "mood": {"type": "string"},
            "visual_prompt": {"type": "string"},
            "speaking_characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "speech_text": {"type": "string"},
            "dialogue_events": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "text": {"type": "string"},
                        "continues_from_previous_shot": {"type": "boolean"},
                        "continues_to_next_shot": {"type": "boolean"},
                    },
                    "required": [
                        "speaker",
                        "text",
                        "continues_from_previous_shot",
                        "continues_to_next_shot",
                    ],
                    "additionalProperties": False,
                },
            },
            "continuity_start_state": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "lighting": {"type": "string"},
                    "environment": {"type": "string"},
                    "props": {"type": "array", "items": {"type": "string"}},
                    "camera_side": {"type": "string"},
                    "state_description": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "continuity_end_state": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "lighting": {"type": "string"},
                    "environment": {"type": "string"},
                    "props": {"type": "array", "items": {"type": "string"}},
                    "camera_side": {"type": "string"},
                    "state_description": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "is_scene_boundary": {"type": "boolean"},
            "character_spatial_bboxes": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_regions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "character_spatial_bboxes_start": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_bboxes_end": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_regions_start": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "character_spatial_regions_end": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        }
        required = list(shot_properties.keys())
        return {
            "type": "object",
            "properties": {
                "shots": {
                    "type": "array",
                    "minItems": min_items,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": shot_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["shots"],
            "additionalProperties": False,
        }

    @staticmethod
    def _shot_batch_json_schema(
        scene_count: int = 2,
    ) -> dict:
        shot_schema = QwenDirector._shot_json_schema()[
            "properties"
        ]["shots"]["items"]
        return {
            "type": "object",
            "properties": {
                "scene_shots": {
                    "type": "array",
                    "minItems": scene_count,
                    "maxItems": scene_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "shots": {
                                "type": "array",
                                "minItems": QwenDirector.SHOTS_PER_SCENE,
                                "maxItems": QwenDirector.SHOTS_PER_SCENE,
                                "items": shot_schema,
                            },
                        },
                        "required": [
                            "scene_id",
                            "shots",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["scene_shots"],
            "additionalProperties": False,
        }

   
    # ========================================================
    # JSON / TEXT HELPERS
    # ========================================================

    @staticmethod
    def _limit_text(
        text: str,
        max_chars: int,
    ) -> str:

        value = str(
            text or ""
        ).strip()

        if len(value) <= max_chars:
            return value

        return (
            value[
                :max_chars
            ]
            .rstrip()
            + "…"
        )

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict:

        value = QwenDirector._strip_thinking(
            text
        )

        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\s*```$",
            "",
            value,
        ).strip()

        try:
            result = json.loads(value)
            if not isinstance(result, dict):
                raise RuntimeError(
                    "Qwen director output must be a JSON object."
                )
            return result
        except json.JSONDecodeError:
            pass

        # Scan balanced JSON objects so surrounding prose cannot invalidate
        # an otherwise usable structured response.
        for start_index, char in enumerate(value):
            if char != "{":
                continue

            depth = 0
            in_string = False
            escaped = False

            for index in range(start_index, len(value)):
                current = value[index]

                if in_string:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        in_string = False
                    continue

                if current == '"':
                    in_string = True
                    continue

                if current == "{":
                    depth += 1
                elif current == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = value[
                            start_index:index + 1
                        ]
                        try:
                            result = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if not isinstance(result, dict):
                            raise RuntimeError(
                                "Qwen director output must be a JSON object."
                            )
                        return result

        raise RuntimeError(
            "Qwen director returned invalid JSON."
        )

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
        json_mode: bool = True,
        disable_thinking: bool = True,
        response_schema: dict | None = None,
    ) -> dict:

        if self._llama is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        _, max_tokens = self._available_output_tokens(
            system_prompt,
            user_prompt,
            minimum_completion=minimum_completion,
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        if max_tokens <= 0:
            raise RuntimeError(
                f"No completion budget remains for {call_name}."
            )

        cache_key = None
        if self._cache_dir is not None:
            cache_key = self._cache_key(
                call_name,
                system_prompt,
                user_prompt,
                response_schema,
            )
            cached = self._cache_read(cache_key)
            if cached is not None:
                self._record_qwen_call(
                    call_name=call_name,
                    elapsed=0.0,
                    max_tokens=0,
                    temperature=temperature,
                    top_p=top_p,
                    response_format={"type": "json_schema"} if json_mode else None,
                    cache_hit=True,
                )
                return cached

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if disable_thinking:
            messages.append(
                {
                    "role": "assistant",
                    "content": "<think>\n\n</think>\n\n",
                }
            )

        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_mode:
            if response_schema is None:
                raise RuntimeError(
                    f"No JSON schema supplied for {call_name}."
                )
            kwargs["response_format"] = {
                "type": "json_schema",
                "schema": response_schema,
            }

        started = time.perf_counter()
        response = None
        error_text = ""

        try:
            response = self._llama.create_chat_completion(
                **kwargs
            )
        except Exception as exc:
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get("usage", {})
                if isinstance(response, dict)
                else {}
            )

            prompt_tokens = int(
                usage.get("prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens", 0) or 0
            )
            decode_tps = (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            )

            self._record_qwen_call(
                call_name=call_name,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=kwargs.get("response_format"),
                error=error_text,
            )

            self._trace_call(
                call_name,
                system_prompt,
                user_prompt,
                response,
                elapsed,
                error_text,
                kwargs.get("response_format"),
            )

        try:
            content = str(
                response["choices"][0]["message"]["content"] or ""
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Qwen director returned an unexpected completion structure."
            ) from exc

        if not content.strip():
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        parsed = self._extract_json(content)

        if not parsed:
            raise RuntimeError(
                f"Qwen returned an empty JSON object for {call_name}."
            )

        if cache_key is not None:
            self._cache_write(
                cache_key,
                parsed,
                call_name=call_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                elapsed=elapsed,
                raw_content=content,
            )

        return parsed

    def _chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
        disable_thinking: bool = True,
    ) -> str:

        if self._llama is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        _, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
                minimum_completion=minimum_completion,
            )
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if disable_thinking:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n\n</think>\n\n"
                    ),
                }
            )

        started = time.perf_counter()
        response = None
        error_text = ""

        try:
            response = (
                self._llama.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get(
                    "usage",
                    {},
                )
                if isinstance(
                    response,
                    dict,
                )
                else {}
            )

            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            decode_tps = (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            )
            self._record_qwen_call(
                call_name=call_name,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=None,
                error=error_text,
            )

        try:
            content = (
                response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Qwen director returned an "
                "unexpected completion structure."
            ) from exc

        content = str(
            content or ""
        ).strip()

        # Qwen3 may emit internal reasoning in a <think>...</think> block.
        # Keep narrative reasoning enabled, but never expose that block as
        # part of the story returned to the application.
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"<think>", content, flags=re.IGNORECASE):
            # A truncated reasoning block means the model spent its output
            # budget on hidden reasoning and never produced usable narrative.
            content = re.split(r"<think>", content, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return content

    # ========================================================
    # CHARACTER SANITIZATION
    # ========================================================

    def _valid_character_name(
        self,
        name: str,
    ) -> bool:

        value = str(
            name or ""
        ).strip()

        if not value:
            return False

        lowered = value.lower()

        if lowered in (
            self.VALID_GENERIC_ROLES
        ):
            return True

        if lowered in (
            self.FORBIDDEN_CHARACTER_NAMES
        ):
            return False

        if len(
            value.split()
        ) > 5:
            return False

        if any(
            token in value
            for token in (
                ":",
                ";",
                "|",
                "{",
                "}",
                "[",
                "]",
            )
        ):
            return False

        return True

    @staticmethod
    def _slug(
        value: str,
    ) -> str:

        cleaned = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(
                value or ""
            ).lower(),
        ).strip("_")

        return (
            cleaned
            or "character"
        )

    @staticmethod
    def _coerce_mapping(
        value,
    ) -> dict:
        """Safely normalize a model field that should be a JSON object."""
        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
                if isinstance(parsed, dict):
                    return parsed

        return {}

    @staticmethod
    def _coerce_list(
        value,
    ) -> list:
        """Safely normalize a model field that should be a JSON array."""
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, set):
            return list(value)

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return []
                return list(parsed) if isinstance(parsed, list) else []
            return [text] if text else []

        return []

    def _sanitize_characters(
        self,
        characters,
    ) -> list[dict]:

        result: list[dict] = []
        seen: set[str] = set()

        for value in (
            characters or []
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            name = str(
                value.get(
                    "name",
                    "",
                )
                or ""
            ).strip()

            if not self._valid_character_name(
                name
            ):
                continue

            key = name.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            character_id = str(
                value.get(
                    "character_id",
                    "",
                )
                or ""
            ).strip()

            if not character_id:

                character_id = (
                    f"char_{self._slug(name)}"
                )

            result.append(
                {
                    "character_id":
                        character_id,

                    "name":
                        name,

                    "role":
                        str(
                            value.get(
                                "role",
                                "story character",
                            )
                            or "story character"
                        ),

                    "description":
                        str(
                            value.get(
                                "description",
                                "",
                            )
                            or ""
                        ),

                    "personality":
                        str(
                            value.get(
                                "personality",
                                "",
                            )
                            or ""
                        ),

                    "appearance":
                        self._coerce_mapping(
                            value.get(
                                "appearance",
                                {},
                            )
                        ),

                    "clothing":
                        self._coerce_mapping(
                            value.get(
                                "clothing",
                                {},
                            )
                        ),

                    "distinctive_features":
                        self._coerce_list(
                            value.get(
                                "distinctive_features",
                                [],
                            )
                        ),

                    "character_state":
                        self._coerce_mapping(
                            value.get(
                                "character_state",
                                {},
                            )
                        ),

                    "continuity_rules":
                        self._coerce_list(
                            value.get(
                                "continuity_rules",
                                [],
                            )
                        ),
                }
            )

        return result

    @staticmethod
    def _clean_list(
        value,
        limit: int | None = None,
    ) -> list[str]:
        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            items = [
                item.strip()
                for item in re.split(
                    r"[\n,;]+",
                    value,
                )
                if item.strip()
            ]
        elif isinstance(
            value,
            (list, tuple, set),
        ):
            items = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        else:
            items = [
                str(value).strip()
            ] if str(value).strip() else []

        result: list[str] = []
        seen: set[str] = set()

        for item in items:
            key = item.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

            if (
                limit is not None
                and len(result) >= limit
            ):
                break

        return result

    # ========================================================
    # SCENE SANITIZATION
    # ========================================================

    @staticmethod
    def _character_alias_map(
        character_names: set[str],
    ) -> dict[str, str]:
        """Resolve safe, unambiguous aliases to canonical roster names."""
        canonical_names = sorted(
            {
                str(name or "").strip().lower()
                for name in character_names
                if str(name or "").strip()
            }
        )

        aliases: dict[str, str] = {}

        for canonical in canonical_names:
            aliases[canonical] = canonical

        first_candidates: dict[str, set[str]] = {}
        pair_candidates: dict[str, set[str]] = {}

        for canonical in canonical_names:

            tokens = re.findall(
                r"[a-z0-9']+",
                canonical,
            )

            if not tokens:
                continue

            first_candidates.setdefault(
                tokens[0],
                set(),
            ).add(canonical)

            if len(tokens) >= 2:

                pair = " ".join(
                    tokens[-2:]
                )

                pair_candidates.setdefault(
                    pair,
                    set(),
                ).add(canonical)

        for alias, matches in first_candidates.items():

            if len(matches) == 1:
                aliases[alias] = next(
                    iter(matches)
                )

        for alias, matches in pair_candidates.items():

            if len(matches) == 1:
                aliases[alias] = next(
                    iter(matches)
                )

        return aliases

    def _sanitize_scenes(
        self,
        scenes,
        character_names: set[str],
    ) -> list[dict]:

        result: list[dict] = []

        for index, value in enumerate(
            scenes or [],
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            description = str(
                value.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            if not description:

                description = str(
                    value.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ).strip()

            if not description:
                continue

            lower = description.lower()

            if lower.startswith(
                (
                    "tone:",
                    "visual priority:",
                    "visual priorities:",
                    "camera:",
                    "lighting:",
                    "mood:",
                    "sound:",
                )
            ):
                continue

            selected: list[str] = []

            alias_map = self._character_alias_map(
                character_names
            )

            for name in (
                value.get(
                    "characters",
                    [],
                )
                or []
            ):

                supplied = str(
                    name
                ).strip().lower()

                resolved = alias_map.get(
                    supplied
                )

                if resolved:
                    selected.append(
                        resolved
                    )

            if not selected and character_names:

                searchable = " ".join(
                    [
                        description,
                        str(value.get("title", "") or ""),
                        str(value.get("scene_objective", "") or ""),
                        str(value.get("continuity_notes", "") or ""),
                    ]
                ).lower()

                for alias, canonical in sorted(
                    alias_map.items(),
                    key=lambda item: (
                        -len(item[0]),
                        item[0],
                    ),
                ):

                    if re.search(
                        r"(?<![a-z0-9'])"
                        + re.escape(alias)
                        + r"(?![a-z0-9'])",
                        searchable,
                    ):

                        if canonical not in selected:
                            selected.append(
                                canonical
                            )

            selected = self._clean_list(
                selected,
                limit=6,
            )

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            title = str(
                value.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            if not title:

                title = (
                    str(
                        value.get(
                            "location",
                            "",
                        )
                        or ""
                    ).strip()
                    or f"Scene {index}"
                )

            result.append(
                {
                    "scene_id":
                        scene_id,

                    "title":
                        title,

                    "order":
                        len(result) + 1,

                    "location":
                        str(
                            value.get(
                                "location",
                                "",
                            )
                            or ""
                        ),

                    "time_of_day":
                        str(
                            value.get(
                                "time_of_day",
                                "",
                            )
                            or ""
                        ),

                    "weather":
                        str(
                            value.get(
                                "weather",
                                "",
                            )
                            or ""
                        ),

                    "atmosphere":
                        str(
                            value.get(
                                "atmosphere",
                                "",
                            )
                            or ""
                        ),

                    "description":
                        description,

                    "mood":
                        str(
                            value.get(
                                "mood",
                                "",
                            )
                            or ""
                        ),

                    "lighting":
                        str(
                            value.get(
                                "lighting",
                                "",
                            )
                            or ""
                        ),

                    "color_temperature":
                        str(
                            value.get(
                                "color_temperature",
                                "",
                            )
                            or ""
                        ),

                    "environment_details":
                        self._clean_list(
                            value.get(
                                "environment_details",
                                [],
                            ),
                            limit=12,
                        ),

                    "key_props":
                        self._clean_list(
                            value.get(
                                "key_props",
                                [],
                            ),
                            limit=8,
                        ),

                    "scene_objective":
                        str(
                            value.get(
                                "scene_objective",
                                "",
                            )
                            or ""
                        ),

                    "characters":
                        self._clean_list(
                        selected,
                    ),

                    "story_summary":
                        str(
                            value.get(
                                "story_summary",
                                description,
                            )
                            or description
                        ),

                    "continuity_notes":
                        str(
                            value.get(
                                "continuity_notes",
                                "",
                            )
                            or ""
                        ),

                    "shot_ids":
                        [],
                }
            )

        # Normalize duplicate scene IDs deterministically. Never silently allow
        # two scenes to share a checkpoint address. Fresh plans may be repaired;
        # resume plans are returned before this path so their stored IDs remain stable.
        used_ids: set[str] = set()
        for scene in result:
            base_id = str(scene.get("scene_id", "") or "").strip() or "scene"
            candidate = base_id
            suffix = 2
            while candidate.lower() in used_ids:
                candidate = f"{base_id}_{suffix}"
                suffix += 1
            scene["scene_id"] = candidate
            used_ids.add(candidate.lower())

        # Budget enforcement intentionally happens after sanitization.
        # Keeping the raw sanitized scene list here lets _compress_scenes_to_budget()
        # see every narrative beat instead of silently discarding over-segmented
        # scenes before the semantic compression pass can run.
        return result

    # ========================================================
    # SHOT SANITIZATION
    # ========================================================

    @staticmethod
    def _normalize_shot_response(
        response: dict,
    ) -> list[dict]:

        if not isinstance(
            response,
            dict,
        ):
            return []

        shots = response.get(
            "shots"
        )

        if isinstance(
            shots,
            list,
        ):
            return [
                item
                for item
                in shots
                if isinstance(
                    item,
                    dict,
                )
            ]

        # Qwen3 sometimes returns a single shot object even when the
        # prompt requests {"shots": [...]}. That response is still useful.
        shot_fields = {
            "shot_id",
            "camera_shot",
            "camera_movement",
            "lens_and_depth_of_field",
            "composition_notes",
            "lighting",
            "color_temperature",
            "mood",
            "visual_prompt",
        }

        if (
            shot_fields
            & set(
                response.keys()
            )
        ):
            return [
                response
            ]

        return []

    def _sanitize_shots(
        self,
        shots,
        scene: dict,
        character_names: set[str],
    ) -> list[dict]:

        scene_id = str(
            scene.get(
                "scene_id",
                "",
            )
            or ""
        ).strip()

        result: list[dict] = []

        for value in (
            shots
            or []
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            candidate = dict(
                value
            )

            candidate_scene_id = str(
                candidate.get(
                    "scene_id",
                    scene_id,
                )
                or scene_id
            ).strip()

            if candidate_scene_id != scene_id:
                continue

            candidate[
                "scene_id"
            ] = scene_id

            # Repair non-critical omissions deterministically instead of
            # discarding a useful shot. Use scene-owned values where available.
            scene_description = str(scene.get("description", "") or "").strip()
            scene_lighting = str(scene.get("lighting", "") or "").strip() or "soft natural light"
            scene_color = str(scene.get("color_temperature", "") or "").strip() or "neutral"
            scene_mood = str(scene.get("mood", "") or "").strip() or "cinematic"

            defaults = {
                "camera_shot": "medium wide",
                "camera_movement": "static",
                "lens_and_depth_of_field": "normal lens, moderate depth of field",
                "composition_notes": "Clear subject separation with readable spatial depth.",
                "lighting": scene_lighting,
                "color_temperature": scene_color,
                "mood": scene_mood,
                "visual_prompt": scene_description or str(candidate.get("action", "") or "").strip(),
            }

            for field, fallback in defaults.items():
                if not str(candidate.get(field, "") or "").strip():
                    candidate[field] = fallback

            if not str(candidate.get("visual_prompt", "") or "").strip():
                continue

            # Character binding is production-critical. Qwen may omit the
            # field or return an empty list even though the scene already
            # has approved characters. In that case deterministically inherit
            # the scene's character set. When Qwen does provide names, keep
            # only names already present in the approved character roster.
            scene_characters = self._clean_list(
                scene.get(
                    "characters",
                    [],
                ),
                limit=6,
            )

            supplied_characters = self._clean_list(
                candidate.get(
                    "characters",
                    [],
                ),
                limit=6,
            )

            selected_characters: list[str] = []

            for name in supplied_characters:

                lowered = name.lower()

                if lowered in character_names:
                    selected_characters.append(name)

            if not selected_characters:

                selected_characters = [
                    name
                    for name in scene_characters
                    if name.lower() in character_names
                ]

            candidate["characters"] = self._clean_list(
                selected_characters,
                limit=6,
            )

            dialogue = candidate.get("dialogue_events", [])
            if not isinstance(dialogue, list):
                dialogue = []
            normalized_dialogue = []
            for event in dialogue:
                if not isinstance(event, dict):
                    continue
                speaker = str(event.get("speaker", "") or "").strip()
                text = str(event.get("text", "") or "")
                if not speaker or not text.strip():
                    continue
                normalized_dialogue.append({
                    "speaker": speaker,
                    "text": text,
                    "continues_from_previous_shot": bool(event.get("continues_from_previous_shot", False)),
                    "continues_to_next_shot": bool(event.get("continues_to_next_shot", False)),
                })
            if not normalized_dialogue:
                legacy_speakers = candidate.get("speaking_characters", []) or []
                legacy_text = str(candidate.get("speech_text", "") or "")
                if legacy_text.strip() and legacy_speakers:
                    normalized_dialogue = [{
                        "speaker": str(legacy_speakers[0]).strip(),
                        "text": legacy_text,
                        "continues_from_previous_shot": False,
                        "continues_to_next_shot": False,
                    }]
            candidate["dialogue_events"] = normalized_dialogue

            def _normalize_continuity(value) -> dict:
                if isinstance(value, dict):
                    return dict(value)
                text = str(value or "").strip()
                if not text:
                    return {}
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return {"state_description": text}
                return parsed if isinstance(parsed, dict) else {"state_description": str(parsed)}

            candidate["continuity_start_state"] = _normalize_continuity(
                candidate.get("continuity_start_state", candidate.get("continuity_state_start"))
            )
            candidate["continuity_end_state"] = _normalize_continuity(
                candidate.get("continuity_end_state", candidate.get("continuity_state_end"))
            )
            candidate.pop("continuity_state_start", None)
            candidate.pop("continuity_state_end", None)
            candidate["is_scene_boundary"] = bool(candidate.get("is_scene_boundary", False))
            raw_bboxes = candidate.get("character_spatial_bboxes", {}) or {}
            normalized_bboxes = {}
            if isinstance(raw_bboxes, dict):
                for name, bbox in raw_bboxes.items():
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        values = [float(v) for v in bbox]
                        if all(0.0 <= v <= 1.0 for v in values) and values[2] >= values[0] and values[3] >= values[1]:
                            normalized_bboxes[str(name).strip()] = values
            candidate["character_spatial_bboxes"] = normalized_bboxes
            raw_regions = candidate.get("character_spatial_regions", {}) or {}
            candidate["character_spatial_regions"] = (
                {str(k).strip(): str(v).strip() for k, v in raw_regions.items() if str(k).strip() and str(v).strip()}
                if isinstance(raw_regions, dict) else {}
            )
            for spatial_key in ("character_spatial_bboxes_start", "character_spatial_bboxes_end"):
                raw = candidate.get(spatial_key, {}) or {}
                normalized = {}
                if isinstance(raw, dict):
                    for name, bbox in raw.items():
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            values = [float(v) for v in bbox]
                            if all(0.0 <= v <= 1.0 for v in values) and values[2] >= values[0] and values[3] >= values[1]:
                                normalized[str(name).strip()] = values
                candidate[spatial_key] = normalized
            for spatial_key in ("character_spatial_regions_start", "character_spatial_regions_end"):
                raw = candidate.get(spatial_key, {}) or {}
                candidate[spatial_key] = (
                    {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
                    if isinstance(raw, dict) else {}
                )

            result.append(
                candidate
            )

        return result

    @staticmethod
    def _normalize_ids(
        scenes: list[dict],
        shots: list[dict],
    ) -> None:

        old_to_new: dict[str, str] = {}

        for index, scene in enumerate(
            scenes,
            start=1,
        ):
            old_id = str(
                scene.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()
            canonical = f"scene_{index:03d}"
            if old_id:
                old_to_new.setdefault(
                    old_id.lower(),
                    canonical,
                )
            scene["scene_id"] = canonical
            scene["order"] = index

        scene_shot_counts: dict[str, int] = {}
        for shot in shots:
            old_scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            scene_id = old_to_new.get(
                old_scene_id.lower(),
                old_scene_id,
            )

            shot["scene_id"] = scene_id
            scene_shot_counts[scene_id] = (
                scene_shot_counts.get(
                    scene_id,
                    0,
                )
                + 1
            )
            shot_number = scene_shot_counts[scene_id]
            shot["shot_id"] = (
                f"{scene_id}_shot_{shot_number:03d}"
            )

    # ========================================================
    # MODE VALIDATION
    # ========================================================

    @staticmethod
    def _normalize_story(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(
                text or ""
            ).strip(),
        )

    @staticmethod
    def _meaningful_tokens(
        text: str,
    ) -> set[str]:

        words = re.findall(
            r"[A-Za-z][A-Za-z'-]{3,}",
            str(
                text or ""
            ).lower(),
        )

        stop = {
            "this",
            "that",
            "with",
            "from",
            "into",
            "about",
            "have",
            "will",
            "they",
            "their",
            "there",
            "which",
            "while",
            "where",
            "would",
            "could",
            "should",
            "story",
            "then",
            "than",
            "when",
        }

        return {
            word
            for word in words
            if word not in stop
        }

    @staticmethod
    def _preservation_anchors(text: str) -> set[str]:
        value = str(text or "")
        anchors: set[str] = set()

        # Explicitly named/called characters are high-confidence anchors.
        for match in re.finditer(
            r"\b(?:named|called)\s+([A-Z][A-Za-z'-]{1,}(?:\s+[A-Z][A-Za-z'-]{1,})*)",
            value,
        ):
            anchors.add(match.group(1).strip().lower())

        # Preserve dates/counts/measurements that can materially change plot facts.
        anchors.update(re.findall(r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?\b", value.lower()))
        return anchors

    def _preservation_coverage(
        self,
        source: str,
        result: str,
        minimum_sentence_overlap: float = 0.28,
    ) -> tuple[float, list[str]]:
        """Measure conservative sentence-level lexical preservation.

        This is deliberately lexical/structural rather than an LLM judge so
        acceptance remains deterministic, cheap, and available in CI.
        """
        source_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", source)
            if sentence.strip()
        ]
        result_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", result)
            if sentence.strip()
        ]
        if not source_sentences:
            return 1.0, []
        result_tokens = [
            self._meaningful_tokens(sentence)
            for sentence in result_sentences
        ]
        covered = 0
        missing: list[str] = []
        for sentence in source_sentences:
            tokens = self._meaningful_tokens(sentence)
            if not tokens:
                continue
            best = 0.0
            for candidate in result_tokens:
                if not candidate:
                    continue
                overlap = len(tokens & candidate) / max(1, len(tokens))
                best = max(best, overlap)
            if best >= minimum_sentence_overlap:
                covered += 1
            else:
                missing.append(sentence[:140])
        coverage = covered / max(1, len(source_sentences))
        return coverage, missing

    def _validate_mode_output(
        self,
        mode: str,
        user_input: str,
        story: str,
    ) -> None:

        source = self._normalize_story(
            user_input
        )

        result = self._normalize_story(
            story
        )

        if not result:

            raise RuntimeError(
                "Qwen director returned an empty story."
            )

        if mode == PRESERVE_USER_STORY_MODE:

            if source != result:

                raise RuntimeError(
                    "Preserve Story mode changed "
                    "the supplied story."
                )

            return

        if mode == AI_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "AI Story mode returned "
                    "the premise unchanged."
                )

            # Do not reject a valid one-sentence story using an arbitrary
            # sentence-count rule. Require deterministic source preservation
            # and some genuinely new meaningful content instead.
            source_anchors = self._preservation_anchors(source)
            result_lower = result.lower()
            missing_anchors = [
                anchor
                for anchor in source_anchors
                if anchor not in result_lower
            ]
            if missing_anchors:
                raise RuntimeError(
                    "AI Story mode dropped required source anchors: "
                    + ", ".join(missing_anchors[:8])
                )

            coverage, missing_sentences = self._preservation_coverage(
                source,
                result,
                minimum_sentence_overlap=0.20,
            )
            if source and coverage < 0.5:
                detail = "; ".join(missing_sentences[:3])
                raise RuntimeError(
                    "AI Story mode did not preserve enough of the supplied "
                    f"premise (coverage={coverage:.2f}). {detail}".strip()
                )

            source_tokens = self._meaningful_tokens(source)
            result_tokens = self._meaningful_tokens(result)
            if source_tokens and not (result_tokens - source_tokens):
                raise RuntimeError(
                    "AI Story mode did not add meaningful narrative content."
                )

            return

        if mode == EXPAND_USER_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "Expand Story mode returned "
                    "the supplied story unchanged."
                )

            source_words = (
                self._meaningful_tokens(
                    source
                )
            )

            if source_words:

                result_words = (
                    self._meaningful_tokens(
                        result
                    )
                )

                overlap = (
                    len(
                        source_words
                        & result_words
                    )
                    / max(
                        1,
                        len(source_words),
                    )
                )

                if overlap < 0.35:

                    raise RuntimeError(
                        "Expand Story mode changed too much of the supplied story "
                        f"(meaningful-token overlap={overlap:.3f}; minimum=0.350)."
                    )

            anchors = self._preservation_anchors(source)
            result_lower = result.lower()
            missing_anchors = [
                anchor
                for anchor in sorted(anchors)
                if anchor not in result_lower
            ]
            if missing_anchors:
                raise RuntimeError(
                    "Expand Story mode dropped source anchors: "
                    + ", ".join(missing_anchors)
                )

            coverage, missing_sentences = self._preservation_coverage(
                source,
                result,
                minimum_sentence_overlap=0.28,
            )
            if coverage < 0.60:
                details = "; ".join(missing_sentences[:3])
                raise RuntimeError(
                    "Expand Story mode did not preserve enough source-event coverage "
                    f"(coverage={coverage:.3f}; minimum=0.600). "
                    f"Unmatched source events: {details}"
                )

            sentences = [
                value
                for value
                in re.split(
                    r"[.!?]+",
                    result,
                )
                if value.strip()
            ]

            if len(sentences) < 3:

                raise RuntimeError(
                    "Expand Story mode did not "
                    "provide enough narrative development."
                )

            return

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def _validate_shot_character_contract(
        self,
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        if not characters:
            return

        allowed = {
            str(value.get("name", "")).strip().lower()
            for value in characters
            if isinstance(value, dict)
            and str(value.get("name", "")).strip()
        }

        for shot in shots:
            shot_characters = [
                str(name).strip()
                for name in (
                    shot.get("characters", [])
                    or []
                )
                if str(name).strip()
            ]

            if not shot_characters:
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has no character binding."
                )

            for field in (
                "characters",
                "speaking_characters",
            ):
                for name in (
                    shot.get(field, [])
                    or []
                ):
                    if (
                        str(name).strip().lower()
                        not in allowed
                    ):
                        raise RuntimeError(
                            f"Shot {shot.get('shot_id', '')} contains unknown character '{name}'."
                        )

            speakers = {
                str(name).strip().lower()
                for name in (
                    shot.get("speaking_characters", [])
                    or []
                )
                if str(name).strip()
            }
            if not speakers.issubset(
                {
                    name.lower()
                    for name in shot_characters
                }
            ):
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has a speaker not present in its character bindings."
                )

    @staticmethod
    def _baseline_visual_language() -> dict:
        return {
            "genre_tone": "cinematic, story-led, naturalistic with controlled contrast",
            "color_palette": "coherent palette derived from scene mood and environment",
            "lighting_philosophy": "motivated cinematic lighting consistent within each scene",
            "camera_philosophy": "deliberate composition with motivated movement and continuity-first coverage",
            "pacing": "clear escalation with varied cinematic rhythm",
        }

    @staticmethod
    def _sanitize_visual_language(
        value,
    ) -> dict:

        if not isinstance(value, dict):
            return {}

        fields = (
            "genre_tone",
            "color_palette",
            "lighting_philosophy",
            "camera_philosophy",
            "pacing",
        )

        return {
            field: str(
                value.get(
                    field,
                    "",
                )
                or ""
            ).strip()
            for field in fields
        }

    def _validate_production_quality(
        self,
        *,
        mode: str,
        story: str,
        scenes: list[dict],
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Deterministically reject structurally valid but production-poor plans."""
        if not story.strip():
            raise RuntimeError("Production plan has no story.")

        if not characters and mode != PRESERVE_USER_STORY_MODE:
            raise RuntimeError(
                "Production plan has no usable character roster for this story mode."
            )

        if not 4 <= len(scenes) <= self.MAX_SCENES:
            raise RuntimeError(
                f"Production plan must contain 4–{self.MAX_SCENES} scenes; got {len(scenes)}."
            )

        scene_ids = [str(scene.get("scene_id", "")).strip() for scene in scenes]
        if any(not scene_id for scene_id in scene_ids):
            raise RuntimeError("Production plan contains an empty scene ID.")
        if len(scene_ids) != len(set(scene_ids)):
            raise RuntimeError("Production plan contains duplicate scene IDs.")

        allowed = {
            str(item.get("name", "")).strip().lower()
            for item in characters
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }

        expected_shot_count = len(scenes) * self.SHOTS_PER_SCENE
        if len(shots) != expected_shot_count:
            raise RuntimeError(
                f"Production plan must contain exactly {expected_shot_count} shots; got {len(shots)}."
            )

        seen_shot_ids: set[str] = set()
        scene_counts = {scene_id: 0 for scene_id in scene_ids}
        for shot in shots:
            shot_id = str(shot.get("shot_id", "")).strip()
            scene_id = str(shot.get("scene_id", "")).strip()
            if not shot_id or shot_id in seen_shot_ids:
                raise RuntimeError("Production plan contains duplicate or empty shot IDs.")
            seen_shot_ids.add(shot_id)
            if scene_id not in scene_counts:
                raise RuntimeError(f"Shot {shot_id} references unknown scene {scene_id}.")
            scene_counts[scene_id] += 1
            if not str(shot.get("visual_prompt", "") or "").strip():
                raise RuntimeError(f"Shot {shot_id} has no visual prompt.")
            for field in ("camera_shot", "camera_movement", "lens_and_depth_of_field", "composition_notes"):
                if not str(shot.get(field, "") or "").strip():
                    raise RuntimeError(f"Shot {shot_id} is missing {field}.")
            shot_characters = shot.get("characters", []) or []
            speakers = shot.get("speaking_characters", []) or []
            if allowed and not shot_characters:
                raise RuntimeError(
                    f"Shot {shot_id} has no character binding."
                )
            for name in [*shot_characters, *speakers]:
                if allowed and str(name).strip().lower() not in allowed:
                    raise RuntimeError(f"Shot {shot_id} contains unknown character '{name}'.")

        if any(count != self.SHOTS_PER_SCENE for count in scene_counts.values()):
            raise RuntimeError("Every scene must contain exactly two production shots.")

    # ========================================================
    # CHECKPOINT / RESUME
    # ========================================================

    def _checkpoint_state(
        self,
        session_id: str,
        mode: str,
        user_input: str,
        base_plan: dict,
        director_plan: dict,
        status: str,
        stage: str,
        completed_scene_ids: list[str],
        current_scene_id: str = "",
        error: str = "",
    ) -> dict:

        checkpoint = ProductionCheckpoint(
            self.project_root
        )

        return {
            "mode": mode,
            "user_input": str(user_input or ""),
            "user_input_sha256": checkpoint.digest_text(
                user_input
            ),
            "director_sha256": checkpoint.digest_file(
                Path(__file__)
            ),
            "status": status,
            "stage": stage,
            "completed_scene_ids": list(
                completed_scene_ids
            ),
            "current_scene_id": current_scene_id,
            "error": error,
            "base_plan": deepcopy(
                base_plan
            ),
            "director_plan": deepcopy(
                director_plan
            ),
        }

    def _save_checkpoint(
        self,
        checkpoint_store: ProductionCheckpoint | None,
        session_id: str | None,
        state: dict,
    ) -> None:

        if checkpoint_store is None or not session_id:
            return

        checkpoint_store.save(
            session_id,
            state,
        )

    def _compress_scenes_to_budget(
        self,
        mode: str,
        story: str,
        characters: list[dict],
        scenes: list[dict],
        character_names: set[str],
    ) -> list[dict]:
        """Reduce an over-segmented scene plan while preserving narrative beats.

        This is only called when metadata violates the 4-6 scene contract.
        Prefer one controlled Qwen restructuring pass over silently discarding
        most of the story. If that repair fails, fall back to deterministic
        first/middle/last sampling so production can still continue.
        """
        if len(scenes) <= self.MAX_SCENES:
            return scenes

        scene_payload = []
        for scene in scenes:
            scene_payload.append(
                {
                    "scene_id": str(
                        scene.get("scene_id", "") or ""
                    ).strip(),
                    "order": int(
                        scene.get("order", len(scene_payload) + 1) or (
                            len(scene_payload) + 1
                        )
                    ),
                    "title": str(
                        scene.get("title", "") or ""
                    ).strip(),
                    "description": self._limit_text(
                        scene.get("description", ""),
                        600,
                    ),
                    "location": str(
                        scene.get("location", "") or ""
                    ).strip(),
                    "characters": self._clean_list(
                        scene.get("characters", []),
                        limit=6,
                    ),
                    "scene_objective": self._limit_text(
                        scene.get("scene_objective", ""),
                        180,
                    ),
                    "continuity_notes": self._limit_text(
                        scene.get("continuity_notes", ""),
                        180,
                    ),
                    "scene_function": str(
                        scene.get("scene_function", "development")
                        or "development"
                    ).strip(),
                    "obligatory_moment": self._limit_text(
                        scene.get("obligatory_moment", scene.get("description", "")),
                        220,
                    ),
                }
            )

        system_prompt = """
You are the STORYBOARD STRUCTURE EDITOR for MiniMax H3.

The supplied scene list is over-segmented.

Compress it into exactly 4–6 meaningful narrative scenes.

Do NOT delete important story events merely to reduce the count.
Instead MERGE adjacent or closely related beats into stronger scenes.

Every source scene has a load-bearing obligatory moment.
Treat that moment as protected narrative information.
When merging scenes, preserve the obligatory moments of all merged beats inside the resulting scene description/objective.
Prefer merging adjacent compatible beats rather than deleting beats.
Never solve the budget by silently truncating the source timeline.

Preserve:
- chronological order;
- protagonist goals;
- important character introductions;
- major discoveries;
- major conflict/escalation;
- climax;
- resolution;
- important locations and continuity.

Each resulting scene must represent a real dramatic beat.

Return JSON only:
{
  "scenes": [
    {
      "scene_id": "scene_001",
      "title": "...",
      "order": 1,
      "location": "...",
      "time_of_day": "...",
      "weather": "...",
      "atmosphere": "...",
      "description": "...",
      "mood": "...",
      "lighting": "...",
      "color_temperature": "...",
      "environment_details": [],
      "key_props": [],
      "characters": [],
      "scene_objective": "...",
      "continuity_notes": "..."
    }
  ]
}
""".strip()

        user_payload = json.dumps(
            {
                "mode": mode,
                "story": self._limit_text(story, 4500),
                "characters": [
                    {
                        "name": str(
                            item.get("name", "") or ""
                        ).strip(),
                        "role": str(
                            item.get("role", "") or ""
                        ).strip(),
                    }
                    for item in characters
                    if isinstance(item, dict)
                    and str(
                        item.get("name", "") or ""
                    ).strip()
                ],
                "scenes": scene_payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        try:
            repaired = self._chat_json(
                system_prompt,
                user_payload,
                minimum_completion=500,
                temperature=0.20,
                top_p=0.82,
                call_name="scene_budget_compression",
                max_completion=1600,
                json_mode=True,
                disable_thinking=True,
                response_schema=self._scene_compression_json_schema(),
            )

            compressed = self._sanitize_scenes(
                repaired.get("scenes", []),
                character_names,
            )

            compressed = self._annotate_scene_functions(
                compressed
            )

            if 4 <= len(compressed) <= self.MAX_SCENES:
                for order, scene in enumerate(compressed, start=1):
                    scene["order"] = order
                return compressed

        except Exception as exc:
            print(
                "[QWEN]",
                "scene_budget_compression_failed",
                str(exc),
                flush=True,
            )

        # Deterministic fallback preserves ALL source beats by merging
        # adjacent groups rather than dropping scenes.
        reduced = self._deterministic_compress_scenes(
            scenes,
            target_count=self.MAX_SCENES,
        )

        for order, scene in enumerate(
            reduced,
            start=1,
        ):
            scene["order"] = order

        return reduced

    def _shot_director_batch_system(
        self,
    ) -> str:
        return """
You are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.

Create exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.

The scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.
Keep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lighting <=12 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, camera_movement <=5 words.
Preserve:
- character identity;
- chronology;
- visual continuity;
- location continuity;
- emotional progression;
- visual-language consistency.
- exact dialogue text; never paraphrase or summarize supplied dialogue.
- stable speaker names from the supplied character roster.
- if dialogue is present, represent each line in dialogue_events; do not put timestamps in the response.
- describe the shot's required initial and ending continuity states in continuity_start_state and continuity_end_state.

Within each scene, the required shots must use meaningfully different
framing/composition while describing the SAME narrative beat.

SCENE-FUNCTION DIRECTING:
Each supplied scene includes scene_function and obligatory_moment.
Use them as directing constraints, not as new story events.
setup: establish geography and protagonist context.
catalyst: reveal the disruptive event, clue, or discovery.
development: show objective, movement, complication, or escalation.
midpoint: emphasize new information and changed understanding.
climax: emphasize danger, decisive action, choice, and consequence.
finale: emphasize aftermath, resolution, and the closing emotional image.
Every required shots must visibly serve the supplied obligatory_moment.

SHOT / FRAMING VOCABULARY:
framing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, POV, insert.

CAMERA MOVEMENT VOCABULARY:
movement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.

LENS / DEPTH OF FIELD:
wide-angle, normal, telephoto, shallow focus, deep focus, selective focus.

COMPOSITION VOCABULARY:
centered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.

LIGHTING VOCABULARY:
lighting: warm tungsten, cool daylight, golden-hour, blue-hour, moonlight, practical neon, hard chiaroscuro, soft overcast, mixed practical/ambient.

Return JSON only in exactly this structure:

{
  "scene_shots": [
    {
      "scene_id": "scene_001",
      "shots": [
        {
          "shot_id": "scene_001_shot_001",
          "scene_id": "scene_001",
          "duration_seconds": 5.2,
          "characters": [],
          "location": "...",
          "action": "...",
          "camera_shot": "...",
          "camera_movement": "...",
          "lens_and_depth_of_field": "...",
          "composition_notes": "...",
          "lighting": "...",
          "color_temperature": "...",
          "mood": "...",
          "visual_prompt": "...",
          "speaking_characters": [],
          "speech_text": "",
          "dialogue_events": [],
          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "is_scene_boundary": false,
          "character_spatial_bboxes": {},
          "character_spatial_regions": {},
          "character_spatial_bboxes_start": {},
          "character_spatial_bboxes_end": {},
          "character_spatial_regions_start": {},
          "character_spatial_regions_end": {}
        },
        {
          "shot_id": "scene_001_shot_002",
          "scene_id": "scene_001",
          "duration_seconds": 5.2,
          "characters": [],
          "location": "...",
          "action": "...",
          "camera_shot": "...",
          "camera_movement": "...",
          "lens_and_depth_of_field": "...",
          "composition_notes": "...",
          "lighting": "...",
          "color_temperature": "...",
          "mood": "...",
          "visual_prompt": "...",
          "speaking_characters": [],
          "speech_text": "",
          "dialogue_events": [],
          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "is_scene_boundary": false,
          "character_spatial_bboxes": {},
          "character_spatial_regions": {},
          "character_spatial_bboxes_start": {},
          "character_spatial_bboxes_end": {},
          "character_spatial_regions_start": {},
          "character_spatial_regions_end": {}
        }
      ]
    }
  ]
}

There must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and
exactly one entry for every supplied scene. Do not add prose outside JSON.

Do NOT output compiler-owned fields.
Do NOT add scenes.
Do NOT omit scenes.
Return JSON only.
""".strip().replace(
            "__SHOTS_PER_SCENE__",
            str(self.SHOTS_PER_SCENE),
        )

    @staticmethod
    def _compact_story_context(story: str, max_chars: int = 850) -> str:
        """Return a compact narrative spine for repeated shot-planning prompts."""
        value = str(story or "").strip()
        if len(value) <= max_chars:
            return value

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", value)
            if part.strip()
        ]
        if not sentences:
            return value[:max_chars].rstrip() + "…"

        if len(sentences) == 1:
            return value[:max_chars].rstrip() + "…"

        first = sentences[0]
        last = sentences[-1]
        if len(first) + len(last) + 1 <= max_chars:
            return f"{first} {last}"

        first_budget = max(220, int(max_chars * 0.60))
        last_budget = max_chars - first_budget - 1
        return (
            first[:first_budget].rstrip()
            + " "
            + last[:max(120, last_budget)].rstrip()
        ).strip()[:max_chars].rstrip() + "…"

    def _shot_director_batch_user(
        self,
        story: str,
        characters: list[dict],
        scenes: list[dict],
        visual_language: dict | None = None,
        reference_visual_context: dict[str, dict] | None = None,
    ) -> str:
        compact_characters = []

        for item in characters:
            if not isinstance(item, dict):
                continue

            name = str(
                item.get("name", "") or ""
            ).strip()

            if not name:
                continue

            compact_characters.append(
                {
                    "name": name,
                    "role": str(
                        item.get("role", "") or ""
                    ).strip(),
                }
            )

        language = {}

        if isinstance(visual_language, dict):
            for key in (
                "genre_tone",
                "color_palette",
                "lighting_philosophy",
                "camera_philosophy",
                "pacing",
            ):
                value = str(
                    visual_language.get(key, "") or ""
                ).strip()

                if value:
                    language[key] = value

        scene_payloads = []

        for scene in scenes:
            scene_payloads.append(
                {
                    "scene_id": str(
                        scene.get("scene_id", "") or ""
                    ).strip(),
                    "title": str(
                        scene.get("title", "") or ""
                    ).strip(),
                    "location": str(
                        scene.get("location", "") or ""
                    ).strip(),
                    "description": self._limit_text(
                        scene.get("description", ""),
                        900,
                    ),
                    "time_of_day": str(
                        scene.get("time_of_day", "") or ""
                    ).strip(),
                    "weather": str(
                        scene.get("weather", "") or ""
                    ).strip(),
                    "atmosphere": self._limit_text(
                        scene.get("atmosphere", ""),
                        180,
                    ),
                    "mood": str(
                        scene.get("mood", "") or ""
                    ).strip(),
                    "lighting": self._limit_text(
                        scene.get("lighting", ""),
                        180,
                    ),
                    "color_temperature": str(
                        scene.get("color_temperature", "") or ""
                    ).strip(),
                    "environment_details": self._clean_list(
                        scene.get("environment_details", []),
                        limit=4,
                    ),
                    "key_props": self._clean_list(
                        scene.get("key_props", []),
                        limit=4,
                    ),
                    "characters": self._clean_list(
                        scene.get("characters", []),
                        limit=6,
                    ),
                    "scene_objective": self._limit_text(
                        scene.get("scene_objective", ""),
                        220,
                    ),
                    "continuity_notes": self._limit_text(
                        scene.get("continuity_notes", ""),
                        180,
                    ),
                    "scene_function": str(
                        scene.get("scene_function", "development")
                        or "development"
                    ).strip(),
                    "obligatory_moment": self._limit_text(
                        scene.get("obligatory_moment", scene.get("description", "")),
                        220,
                    ),
                }
            )

        visual_context = {}
        context_source = reference_visual_context or self._reference_visual_context
        for path, analysis in context_source.items():
            if isinstance(analysis, dict):
                visual_context[str(path)] = {
                    "description": str(analysis.get("description", "") or "")[:500],
                    "identity_features": [str(v) for v in (analysis.get("identity_features", []) or [])][:6],
                    "wardrobe": [str(v) for v in (analysis.get("wardrobe", []) or [])][:6],
                    "environment": [str(v) for v in (analysis.get("environment", []) or [])][:6],
                    "lighting": str(analysis.get("lighting", "") or "")[:220],
                    "composition": str(analysis.get("composition", "") or "")[:220],
                }

        return json.dumps(
            {
                "story_context": self._compact_story_context(story, 850),
                "characters": compact_characters,
                "visual_language": language,
                "reference_visual_analysis": visual_context,
                "scenes": scene_payloads,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _normalize_batch_shot_response(
        response: dict,
    ) -> dict[str, list[dict]]:
        if not isinstance(response, dict):
            return {}

        entries = response.get("scene_shots")

        if not isinstance(entries, list):
            return {}

        result: dict[str, list[dict]] = {}

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            scene_id = str(
                entry.get("scene_id", "") or ""
            ).strip()

            shots = entry.get("shots")

            if not scene_id or not isinstance(shots, list):
                continue

            result[scene_id] = [
                item
                for item in shots
                if isinstance(item, dict)
            ]

        return result



    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
        checkpoint_session_id: str | None = None,
        resume_state: dict | None = None,
    ) -> dict:

        if not director_enabled():

            return {
                "enabled": False,
                "plan": deepcopy(
                    base_plan
                ),
                "director_notes": "",
            }

        if mode not in (
            self._MODE_LABELS
        ):

            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        if (resume_state or {}).get("stage") in {
            "director_complete",
            "production_plan",
            "rendering",
            "render_complete",
        }:
            prior = deepcopy(
                (resume_state or {}).get("director_plan", {}) or {}
            )
            if (
                prior.get("story")
                and prior.get("scenes")
                and prior.get("shots")
            ):
                return {
                    "enabled": True,
                    "plan": prior,
                    "director_notes": str(
                        prior.get("director_notes", "") or ""
                    ),
                }

        self.load()

        if self._llama is None:

            raise RuntimeError(
                "Qwen director model failed to load."
            )

        checkpoint_store = (
            ProductionCheckpoint(
                self.project_root
            )
            if checkpoint_session_id
            else None
        )

        prior_director_plan = (
            deepcopy(
                (resume_state or {}).get(
                    "director_plan",
                    {},
                )
                or {}
            )
        )

        prior_shots = list(
            prior_director_plan.get(
                "shots",
                [],
            )
            or []
        )

        resume_stage = str(
            (resume_state or {}).get(
                "stage",
                "",
            )
            or ""
        ).strip()

        resuming = bool(
            resume_state
            and prior_director_plan
            and resume_stage in {
                "initialized",
                "narrative",
                "metadata",
                "shots",
                "director_complete",
            }
        )

        temperature, top_p = (
            self._sampling_for_mode(
                mode
            )
        )

        # ----------------------------------------------------
        # PASS 1A: narrative
        # ----------------------------------------------------

        if resuming and prior_director_plan.get(
            "story"
        ):

            story = self._normalize_story(
                prior_director_plan.get(
                    "story",
                    "",
                )
            )

            story_plan = deepcopy(
                prior_director_plan
            )

            self._current_visual_language = (
                self._sanitize_visual_language(
                    story_plan.get(
                        "visual_language",
                        {},
                    )
                )
            )

        elif mode == PRESERVE_USER_STORY_MODE:

            story = self._normalize_story(
                user_input
            )

            story_plan = {
                "story": story,
            }

            self._current_visual_language = {}

        else:

            story = ""

            story_system = (
                self._story_text_system(
                    mode
                )
            )

            story_user = (
                self._story_text_user(
                    mode,
                    user_input,
                )
            )

            try:

                story = self._chat_text(
                    story_system,
                    story_user,
                    minimum_completion=350,
                    temperature=temperature,
                    top_p=top_p,
                    call_name=(
                        "ai_story_text_pass"
                        if mode == AI_STORY_MODE
                        else "expand_story_text_pass"
                    ),
                    max_completion=1600,
                    disable_thinking=True,
                )

                self._validate_mode_output(
                    mode,
                    user_input,
                    story,
                )

            except RuntimeError as first_error:

                if mode == EXPAND_USER_STORY_MODE:
                    # Expansion failure must not trigger another expensive
                    # Qwen call. The original user story is the deterministic
                    # correctness fallback; downstream planning can continue.
                    self._record_recovery(
                        "expand_story_source_fallback",
                        str(first_error),
                    )
                    story = self._normalize_story(
                        user_input
                    )
                    self._validate_mode_output(
                        PRESERVE_USER_STORY_MODE,
                        user_input,
                        story,
                    )
                else:
                    failure_text = str(first_error)
                    retry_user = (
                        story_user
                        + "\n\n"
                        "REPAIR REQUIRED.\n"
                        + f"Previous validation failure: {failure_text}\n"
                        + "Write the complete narrative again. "
                        "Return ONLY the story prose. "
                        "Do not output JSON or commentary."
                    )

                    story = self._chat_text(
                        story_system,
                        retry_user,
                        minimum_completion=350,
                        temperature=min(
                            0.85,
                            max(0.70, temperature + 0.10),
                        ),
                        top_p=0.92,
                        call_name="ai_story_text_retry",
                        max_completion=1600,
                        disable_thinking=True,
                    )

                    self._validate_mode_output(
                        mode,
                        user_input,
                        story,
                    )

            story_plan = {
                "story": story,
            }

        # ----------------------------------------------------
        # PASS 1B: deterministic production foundation
        # ----------------------------------------------------
        #
        # The ProductionPlanner has already created the canonical
        # characters and scene topology. Do not spend a Qwen call
        # regenerating deterministic metadata. Qwen is only the creative
        # enrichment layer from this point onward.
        #
        # This also guarantees that the Director always has canonical
        # entities/scene IDs even when the Director is enabled.
        base_characters = deepcopy(
            base_plan.get("characters", [])
            or []
        )

        base_scenes = deepcopy(
            base_plan.get("scenes", [])
            or []
        )

        story_plan = {
            "story": story,
            "characters": base_characters,
            "scenes": base_scenes,
        }

        story = self._normalize_story(
            story_plan.get(
                "story",
                story,
            )
            or story
        )

        director_notes = str(
            story_plan.get(
                "director_notes",
                "",
            )
            or ""
        ).strip()

        visual_language = (
            self._sanitize_visual_language(
                story_plan.get(
                    "visual_language",
                    {},
                )
            )
        )
        baseline_visual_language = self._baseline_visual_language()
        for key, value in baseline_visual_language.items():
            if not visual_language.get(key):
                visual_language[key] = value

        self._current_visual_language = (
            dict(
                visual_language
            )
        )

                # ----------------------------------------------------
        # CANONICAL CHARACTERS / SCENES
        # ----------------------------------------------------
        #
        # AI Story / Expand Story:
        # Qwen first creates the final narrative. The canonical production
        # roster and scene topology must then be derived from THAT final story.
        #
        # Preserve Story:
        # the supplied user story remains the source of truth.
        #
        # ProductionPlanner remains the canonical deterministic extractor;
        # Qwen does not directly own the final character/scene objects.

        planner = self._planner()

        if mode in (
            AI_STORY_MODE,
            EXPAND_USER_STORY_MODE,
        ):
            canonical_source_story = story
        else:
            canonical_source_story = user_input

        canonical_characters = planner.create_characters(
            canonical_source_story
        )

        characters = self._sanitize_characters(
            [
                character.to_dict()
                for character in canonical_characters
                if character is not None
            ]
        )

        if not characters:
            raise RuntimeError(
                "No canonical characters could be derived from the final story."
            )

        character_names = {
            str(character.get("name", "")).strip().lower()
            for character in characters
            if isinstance(character, dict)
            and str(character.get("name", "")).strip()
        }

        if not character_names:
            raise RuntimeError(
                "Canonical character extraction produced no usable names."
            )

        canonical_scenes = planner.create_scenes(
            canonical_source_story,
            canonical_characters,
        )

        scenes = self._sanitize_scenes(
            [
                scene.to_dict()
                for scene in canonical_scenes
                if scene is not None
            ],
            character_names,
        )
        
        if not scenes:
            raise RuntimeError(
                "Deterministic base plan contains no canonical scenes."
            )

        if len(scenes) < 4 or len(scenes) > self.MAX_SCENES:
            raise RuntimeError(
                "Deterministic base plan scene count is outside "
                f"the required 4-{self.MAX_SCENES} range: {len(scenes)}"
            )

        # Preserve deterministic scene topology and only add the
        # Director's creative scene annotations if they already exist.
        scenes = self._annotate_scene_functions(
            scenes
        )

        director_plan = {
            "story": story,
            "story_mode": mode,
            "director_notes": director_notes,
            "visual_language": visual_language,
            "characters": deepcopy(characters),
            "scenes": deepcopy(scenes),
            "shots": prior_shots,
        }

        completed_scene_ids = []
        prior_by_scene = {}

        for shot in prior_shots:
            if not isinstance(shot, dict):
                continue

            sid = str(
                shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                prior_by_scene.setdefault(
                    sid,
                    [],
                ).append(shot)

        for scene in scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            if len(prior_by_scene.get(sid, [])) >= self.SHOTS_PER_SCENE:
                completed_scene_ids.append(sid)

        if checkpoint_store and checkpoint_session_id:
            self._save_checkpoint(
                checkpoint_store,
                checkpoint_session_id,
                self._checkpoint_state(
                    checkpoint_session_id,
                    mode,
                    user_input,
                    base_plan,
                    director_plan,
                    "running",
                    "shots",
                    completed_scene_ids,
                    completed_scene_ids[-1]
                    if completed_scene_ids
                    else "",
                ),
            )

        # ----------------------------------------------------
        # PASS 2: cinematography
        #
        # Fresh scenes are planned in bounded creative batches of up to two.
        # Qwen supplies only
        # creative shot direction; deterministic compilation/rebinding
        # supplies all production identity and technical fields.
        # ----------------------------------------------------

        all_shots: list[dict] = []
        shot_temperature, shot_top_p = self._shot_sampling()

        try:

            scene_index = 0

            while scene_index < len(scenes):

                scene = scenes[
                    scene_index
                ]

                scene_id = str(
                    scene.get(
                        "scene_id",
                        "",
                    )
                    or ""
                ).strip()

                existing_scene_shots = [
                    deepcopy(item)
                    for item
                    in prior_by_scene.get(
                        scene_id,
                        [],
                    )[: self.SHOTS_PER_SCENE]
                    if isinstance(
                        item,
                        dict,
                    )
                ]

                # Resumed scene: never regenerate completed work.
                if len(existing_scene_shots) >= self.SHOTS_PER_SCENE:

                    existing_scene_shots = (
                        existing_scene_shots[
                            : self.SHOTS_PER_SCENE
                        ]
                    )

                    all_shots.extend(
                        existing_scene_shots
                    )

                    scene_index += 1
                    continue

                # Build the largest safe batch of fresh adjacent scenes.
                # A partially completed/resumed scene is deterministic-only:
                # its existing shots are preserved and any shortage is filled
                # later from the canonical base plan. Fresh scenes receive
                # exactly one creative Qwen batch call.
                if existing_scene_shots:
                    batch_scenes = [scene]
                else:
                    max_count = min(
                        self.MAX_SHOT_BATCH_SCENES,
                        len(scenes) - scene_index,
                    )

                    batch_scenes = []
                    for offset in range(max_count):
                        candidate_scene = scenes[scene_index + offset]
                        candidate_id = str(
                            candidate_scene.get("scene_id", "") or ""
                        ).strip()

                        # Do not cross a checkpoint/resume boundary.
                        if prior_by_scene.get(candidate_id):
                            break

                        batch_scenes.append(candidate_scene)

                generated_by_scene: dict[str, list[dict]] = {}

                # ------------------------------------------------
                # CREATIVE SHOT PASS
                # ------------------------------------------------
                # One fresh batch call for 1–2 scenes. No per-scene retry
                # or missing-shot Qwen recovery is performed.
                if existing_scene_shots:
                    generated_by_scene[scene_id] = list(
                        existing_scene_shots
                    )[: self.SHOTS_PER_SCENE]

                else:
                    # Start with the largest candidate and shrink only if the
                    # prompt would leave less than a useful completion reserve
                    # inside the fixed 8192-token context.
                    while True:
                        batch_user = self._shot_director_batch_user(
                            story,
                            characters,
                            batch_scenes,
                            visual_language,
                        )

                        desired_completion = min(
                            DIRECTOR_MAX_TOKENS,
                            max(
                                320,
                                1400 * len(batch_scenes),
                            ),
                        )

                        prompt_tokens = self._count_tokens(
                            self._shot_director_batch_system()
                            + "\n\n"
                            + batch_user
                        )

                        if (
                            prompt_tokens
                            <= int(DIRECTOR_N_CTX)
                            - 128
                            - desired_completion
                        ):
                            break

                        if len(batch_scenes) == 1:
                            break

                        batch_scenes = batch_scenes[:-1]

                    batch_ids = [
                        str(
                            item.get("scene_id", "") or ""
                        ).strip()
                        for item in batch_scenes
                    ]

                    try:
                        batch_response = self._chat_json(
                            self._shot_director_batch_system(),
                            batch_user,
                            minimum_completion=320,
                            temperature=shot_temperature,
                            top_p=shot_top_p,
                            call_name=(
                                "shot_batch:"
                                + "_".join(batch_ids)
                            ),
                            max_completion=min(
                                DIRECTOR_MAX_TOKENS,
                                1400 * len(batch_scenes),
                            ),
                            json_mode=True,
                            disable_thinking=True,
                            response_schema=self._shot_batch_json_schema(
                                scene_count=len(batch_scenes),
                            ),
                        )

                        batch_map = self._normalize_batch_shot_response(
                            batch_response
                        )

                        for target_scene in batch_scenes:
                            target_id = str(
                                target_scene.get("scene_id", "") or ""
                            ).strip()

                            candidate = self._sanitize_shots(
                                batch_map.get(
                                    target_id,
                                    [],
                                ),
                                target_scene,
                                character_names,
                            )

                            if candidate:
                                generated_by_scene[target_id] = candidate[
                                    : self.SHOTS_PER_SCENE
                                ]

                    except Exception as batch_error:
                        self._record_recovery(
                            "shot_batch_deterministic_fallback",
                            str(batch_error),
                        )

                # The deterministic repair pass after the loop owns missing
                # shots. Never launch another Qwen request here.
                # PERSIST THIS BATCH
                # ------------------------------------------------
                batch_added = []

                for target_scene in batch_scenes:

                    target_id = str(
                        target_scene.get(
                            "scene_id",
                            "",
                        )
                        or ""
                    ).strip()

                    scene_shots = list(
                        generated_by_scene.get(
                            target_id,
                            [],
                        )
                    )

                    # Missing shots are intentionally not regenerated with Qwen.
                    # The deterministic repair pass below fills them from base_plan.
                    if (
                        target_id == scene_id
                        and existing_scene_shots
                    ):

                        if not scene_shots:
                            scene_shots = existing_scene_shots

                        elif len(existing_scene_shots) == 1:
                            scene_shots = [
                                existing_scene_shots[0],
                                scene_shots[0],
                            ]

                    scene_shots = scene_shots[
                        : self.SHOTS_PER_SCENE
                    ]

                    batch_added.extend(
                        scene_shots
                    )

                    if len(scene_shots) >= self.SHOTS_PER_SCENE:
                        if target_id not in completed_scene_ids:
                            completed_scene_ids.append(
                                target_id
                            )

                all_shots.extend(
                    batch_added
                )

                if batch_added:

                    director_plan["shots"] = deepcopy(
                        all_shots
                    )

                    last_scene_id = str(
                        batch_scenes[-1].get(
                            "scene_id",
                            "",
                        )
                        or ""
                    ).strip()

                    self._save_checkpoint(
                        checkpoint_store,
                        checkpoint_session_id,
                        self._checkpoint_state(
                            checkpoint_session_id or "",
                            mode,
                            user_input,
                            base_plan,
                            director_plan,
                            "running",
                            "shots",
                            completed_scene_ids,
                            last_scene_id,
                        ),
                    )

                scene_index += len(
                    batch_scenes
                )

        except Exception as exc:

            director_plan["shots"] = deepcopy(
                all_shots
            )

            self._save_checkpoint(
                checkpoint_store,
                checkpoint_session_id,
                self._checkpoint_state(
                    checkpoint_session_id or "",
                    mode,
                    user_input,
                    base_plan,
                    director_plan,
                    "failed",
                    "shots",
                    completed_scene_ids,
                    scene_id if 'scene_id' in locals() else "",
                    str(exc),
                ),
            )

            raise

        # Deterministic shot fallback: if Qwen failed to provide enough
        # creative shots for a scene, reuse only the corresponding canonical
        # base-plan shots. This keeps the production structurally complete
        # without inventing entities or making another model call.
        base_shots_by_scene: dict[str, list[dict]] = {}

        for base_shot in (
            base_plan.get("shots", [])
            or []
        ):
            if not isinstance(base_shot, dict):
                continue

            sid = str(
                base_shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                base_shots_by_scene.setdefault(
                    sid,
                    [],
                ).append(
                    deepcopy(base_shot)
                )

        shots_by_scene: dict[str, list[dict]] = {}

        for shot in all_shots:
            if not isinstance(shot, dict):
                continue

            sid = str(
                shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                shots_by_scene.setdefault(
                    sid,
                    [],
                ).append(shot)

        repaired_shots: list[dict] = []

        for scene in scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            current = list(
                shots_by_scene.get(sid, [])
            )[: self.SHOTS_PER_SCENE]

            if len(current) < self.SHOTS_PER_SCENE:
                for fallback in base_shots_by_scene.get(sid, []):
                    if len(current) >= self.SHOTS_PER_SCENE:
                        break

                    candidate = deepcopy(fallback)

                    # Never let fallback structural identity conflict with
                    # the canonical scene.
                    candidate["scene_id"] = sid

                    existing_ids = {
                        str(
                            item.get("shot_id", "")
                        ).strip()
                        for item in current
                        if isinstance(item, dict)
                    }

                    candidate_id = str(
                        candidate.get("shot_id", "")
                    ).strip()

                    if candidate_id in existing_ids:
                        continue

                    current.append(candidate)

            # Final safety gate: deterministic fallback shots must traverse
            # the same sanitizer as Qwen-generated shots before compilation.
            # This prevents missing cinematography fields from reaching the
            # strict CinematicCompiler validator.
            current = self._sanitize_shots(
                current,
                scene,
                character_names,
            )[: self.SHOTS_PER_SCENE]

            if len(current) < self.SHOTS_PER_SCENE:
                self._record_recovery(
                    "shot_field_sanitization_incomplete",
                    f"scene={sid} count={len(current)} expected={self.SHOTS_PER_SCENE}",
                )

            repaired_shots.extend(current)

        all_shots = repaired_shots

        # No Qwen-shot failure is fatal here: the deterministic compiler
        # completes production fields while preserving every valid creative
        # shot Qwen produced.
        self._normalize_ids(
            scenes,
            all_shots,
        )

        all_shots = CinematicCompiler(
            character_names=character_names,
        ).compile_all(
            scenes,
            all_shots,
        )

        for scene in scenes:

            scene["shot_ids"] = [
                shot["shot_id"]
                for shot in all_shots
                if shot.get("scene_id") == scene.get("scene_id")
            ]

        self._validate_shot_character_contract(
            all_shots,
            characters,
        )

        self._validate_production_quality(
            mode=mode,
            story=story,
            scenes=scenes,
            shots=all_shots,
            characters=characters,
        )

        final_director_plan = {
            "story": story,
            "story_mode": mode,
            "director_notes": director_notes,
            "visual_language": visual_language,
            "characters": characters,
            "scenes": scenes,
            "shots": all_shots,
        }

        self._print_qwen_summary()

        self._save_checkpoint(
            checkpoint_store,
            checkpoint_session_id,
            self._checkpoint_state(
                checkpoint_session_id or "",
                mode,
                user_input,
                base_plan,
                final_director_plan,
                "director_completed",
                "director_complete",
                [
                    str(scene.get("scene_id", "")).strip()
                    for scene in scenes
                    if str(scene.get("scene_id", "")).strip()
                ],
                "",
                "",
            ),
        )

        return {
            "enabled": True,
            "plan": final_director_plan,
            "director_notes": director_notes,
        }

    # ========================================================
    # MERGE
    # ========================================================

    def critique_plan(self, *, mode: str, user_input: str, plan: dict) -> dict:
        """Run an optional read-only cinematic critique.

        The critic may identify problems but never mutates the canonical plan.
        """
        system_prompt = """
You are a conservative cinematic production critic.
Review the supplied production plan for narrative, shot-design, continuity,
reference-binding, and dialogue/action risks. Return a conservative critique.
When a fix is warranted, provide a minimal patch for an existing shot using
only the explicitly allowed creative fields in the schema. Never change scene
identity, shot identity, characters, reference bindings, timing, or continuity
state. Do not invent facts that are not present in the plan.
""".strip()
        compact = {
            "mode": mode,
            "story": str(user_input or "")[:5000],
            "visual_language": plan.get("visual_language", {}) or {},
            "scenes": plan.get("scenes", []) or [],
            "shots": plan.get("shots", []) or [],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_score": {"type": "number"},
                "status": {"type": "string", "enum": ["pass", "review"]},
                "findings": {"type": "array", "items": {"type": "string"}},
                "shot_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shot_id": {"type": "string"},
                            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                            "finding": {"type": "string"},
                        },
                        "required": ["shot_id", "severity", "finding"],
                    },
                },
                "recommended_focus": {"type": "array", "items": {"type": "string"}},
                "shot_patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shot_id": {"type": "string"},
                            "patch": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"type": "string"},
                                    "camera_shot": {"type": "string"},
                                    "camera_movement": {"type": "string"},
                                    "lens_and_depth_of_field": {"type": "string"},
                                    "composition_notes": {"type": "string"},
                                    "lighting": {"type": "string"},
                                    "color_temperature": {"type": "string"},
                                    "mood": {"type": "string"},
                                    "visual_prompt": {"type": "string"},
                                    "overall_soundscape": {"type": "string"},
                                    "non_diegetic_music": {"type": "string"},
                                },
                            },
                        },
                        "required": ["shot_id", "patch"],
                    },
                },
            },
            "required": ["overall_score", "status", "findings", "shot_findings", "recommended_focus", "shot_patches"],
        }
        return self._chat_json(
            system_prompt,
            json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
            minimum_completion=300,
            temperature=0.10,
            top_p=0.75,
            call_name="director_critique",
            max_completion=1000,
            json_mode=True,
            disable_thinking=True,
            response_schema=schema,
        )

    def enrich_plan(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
        checkpoint_session_id: str | None = None,
        resume_state: dict | None = None,
    ) -> dict:

        result = self.generate(
            mode=mode,
            user_input=user_input,
            base_plan=base_plan,
            checkpoint_session_id=checkpoint_session_id,
            resume_state=resume_state,
        )

        if not result.get(
            "enabled",
            False,
        ):

            return deepcopy(
                base_plan
            )

        creative = (
            result.get(
                "plan",
                {},
            )
            or {}
        )

        merged = deepcopy(
            base_plan
        )

        if mode == PRESERVE_USER_STORY_MODE:

            merged[
                "story"
            ] = user_input.strip()

        else:

            merged[
                "story"
            ] = str(
                creative.get(
                    "story",
                    merged.get(
                        "story",
                        user_input,
                    ),
                )
                or merged.get(
                    "story",
                    user_input,
                )
            ).strip()

        merged[
            "story_mode"
        ] = mode

        merged[
            "director_notes"
        ] = str(
            creative.get(
                "director_notes",
                "",
            )
            or ""
        )

        creative_visual_language = (
            creative.get(
                "visual_language",
                {},
            )
            or {}
        )

        # Visual language is creative metadata, not production identity.
        # Merge field-by-field so a partial Qwen response cannot erase
        # deterministic/default visual-language fields already present in
        # the base plan. Qwen never gets ownership of unrelated keys.
        base_visual_language = merged.get(
            "visual_language",
            {},
        )
        if not isinstance(base_visual_language, dict):
            base_visual_language = {}

        if isinstance(creative_visual_language, dict):
            for key in (
                "genre_tone",
                "color_palette",
                "lighting_philosophy",
                "camera_philosophy",
                "pacing",
            ):
                value = creative_visual_language.get(key)
                if value not in (None, "", [], {}):
                    base_visual_language[key] = deepcopy(value)

        merged["visual_language"] = base_visual_language

        # Canonical structure is never replaced by Qwen.
        # Characters and scene topology come from the deterministic base plan.
        base_characters = deepcopy(
            base_plan.get("characters", [])
            or []
        )
        
        creative_characters = deepcopy(
            creative.get("characters", [])
            or []
        )
        
        merged["characters"] = (
            base_characters
            if base_characters
            else creative_characters
        )

        canonical_scenes = deepcopy(
            base_plan.get("scenes", [])
            or []
        )

        creative_scenes = (
            creative.get("scenes", [])
            or []
        )

        creative_by_id = {
            str(scene.get("scene_id", "") or "").strip(): scene
            for scene in creative_scenes
            if isinstance(scene, dict)
            and str(scene.get("scene_id", "") or "").strip()
        }

        # Creative scene fields may enrich an existing scene, but structural
        # identity/topology remains deterministic.
        protected_scene_fields = {
            "scene_id",
            "order",
            "characters",
            "shot_ids",
        }

        for scene in canonical_scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            creative_scene = creative_by_id.get(sid)

            if not isinstance(creative_scene, dict):
                continue

            for key, value in creative_scene.items():
                if key in protected_scene_fields:
                    continue
                if value in (None, "", [], {}):
                    continue
                scene[key] = deepcopy(value)

        merged["scenes"] = canonical_scenes

        creative_shots = (
            creative.get("shots", [])
            or []
        )

        if creative_shots:
            valid_scene_ids = {
                str(scene.get("scene_id", "") or "").strip()
                for scene in canonical_scenes
            }

            merged["shots"] = [
                deepcopy(shot)
                for shot in creative_shots
                if isinstance(shot, dict)
                and str(
                    shot.get("scene_id", "") or ""
                ).strip() in valid_scene_ids
            ]
        else:
            merged["shots"] = deepcopy(
                base_plan.get("shots", [])
                or []
            )

        return merged
