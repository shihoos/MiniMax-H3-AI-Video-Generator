from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.identity_continuity import (
    IdentityContinuity,
)
from pipeline.reference_manager import (
    ReferenceManager,
)
from planner.entity_resolver import (
    EntityResolver,
)
from planner.config import (
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_STEPS,
    H3_WIDTH,
    PRESERVE_USER_STORY_MODE,
    TURBO_STEPS,
    VALID_STORY_MODES,
    WORKFLOW_AUTO,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
    director_enabled,
)
from schemas.character import Character
from schemas.scene import Scene
from schemas.shot import Shot


@dataclass
class StoryUnit:
    order: int
    text: str


class ProductionPlanner:
    """
    Dependency-free production planner.

    Important architectural rule:

    When the local Qwen director is enabled, Qwen is the
    creative authority for:
      - story development
      - character creation
      - scene design
      - cinematic shot planning

    This deterministic planner then acts only as the production
    safety/reference/H3 binding layer.

    When the Qwen director is disabled, this planner remains
    available as a deterministic fallback for CI and offline
    operation.

    Character creation therefore means:

        story
          -> deterministic canonical character profile
          -> identity locks
          -> H3 prompt

    When reference media exists, it is attached.

    When it does not exist, the generated character profile
    becomes the canonical identity source for the production.
    """

    ROLE_PATTERNS = (
        (
            r"\b(?:a|an|the)\s+(young\s+)?woman\b",
            "woman",
        ),
        (
            r"\b(?:a|an|the)\s+(young\s+)?man\b",
            "man",
        ),
        (
            r"\b(?:a|an|the)\s+girl\b",
            "girl",
        ),
        (
            r"\b(?:a|an|the)\s+boy\b",
            "boy",
        ),
        (
            r"\b(?:a|an|the)\s+child\b",
            "child",
        ),
        (
            r"\b(?:a|an|the)\s+person\b",
            "person",
        ),
        (
            r"\b(?:a|an|the)\s+hero\b",
            "hero",
        ),
        (
            r"\b(?:a|an|the)\s+heroine\b",
            "heroine",
        ),
        (
            r"\b(?:a|an|the)\s+explorer\b",
            "explorer",
        ),
        (
            r"\b(?:a|an|the)\s+detective\b",
            "detective",
        ),
        (
            r"\b(?:a|an|the)\s+scientist\b",
            "scientist",
        ),
        (
            r"\b(?:a|an|the)\s+soldier\b",
            "soldier",
        ),
        (
            r"\b(?:a|an|the)\s+warrior\b",
            "warrior",
        ),
        (
            r"\b(?:a|an|the)\s+king\b",
            "king",
        ),
        (
            r"\b(?:a|an|the)\s+queen\b",
            "queen",
        ),
        (
            r"\b(?:a|an|the)\s+child\b",
            "child",
        ),
        (
            r"\b(?:a|an|the)\s+robot\b",
            "robot",
        ),
        (
            r"\b(?:a|an|the)\s+android\b",
            "android",
        ),
        (
            r"\b(?:a|an|the)\s+pilot\b",
            "pilot",
        ),
    )

    COMMON_PROPER_WORDS = {
        "The",
        "A",
        "An",
        "Then",
        "When",
        "While",
        "After",
        "Before",
        "Suddenly",
        "Meanwhile",
        "Finally",
        "But",
        "And",
        "In",
        "On",
        "At",
        "As",
        "During",
    }

    ACTION_WORDS = (
        "walk",
        "runs",
        "run",
        "moves",
        "move",
        "looks",
        "look",
        "turns",
        "turn",
        "enters",
        "enter",
        "leaves",
        "leave",
        "fights",
        "fight",
        "talks",
        "talk",
        "speaks",
        "speak",
        "stands",
        "stand",
        "sits",
        "sit",
        "drives",
        "drive",
        "flies",
        "fly",
        "jumps",
        "jump",
        "opens",
        "open",
        "closes",
        "close",
        "reaches",
        "reach",
        "holds",
        "hold",
        "runs",
    )

    # Broad narrative-verb list (both present and past tense, plus
    # a few common auxiliaries) used to recognize a character name
    # used in ordinary sentence-subject position, e.g. "Eli walked
    # through the ruined city" or "Sara was hiding near the tower".
    # ACTION_WORDS above only covers present tense and is kept for
    # backward compatibility; this list is intentionally much wider
    # since most short-story prose is written in past tense.
    NARRATIVE_SUBJECT_VERBS = (
        "walked", "walks", "walk", "ran", "runs", "run",
        "moved", "moves", "move", "looked", "looks", "look",
        "turned", "turns", "turn", "entered", "enters", "enter",
        "left", "leaves", "leave", "fought", "fights", "fight",
        "talked", "talks", "talk", "spoke", "speaks", "speak",
        "stood", "stands", "stand", "sat", "sits", "sit",
        "drove", "drives", "drive", "flew", "flies", "fly",
        "jumped", "jumps", "jump", "opened", "opens", "open",
        "closed", "closes", "close", "reached", "reaches", "reach",
        "held", "holds", "hold", "watched", "watches", "watch",
        "waited", "waits", "wait", "hid", "hides", "hiding", "hide",
        "stared", "stares", "stare", "whispered", "whispers", "whisper",
        "shouted", "shouts", "shout", "cried", "cries", "cry",
        "smiled", "smiles", "smile", "frowned", "frowns", "frown",
        "nodded", "nods", "nod", "gasped", "gasps", "gasp",
        "sighed", "sighs", "sigh", "followed", "follows", "follow",
        "chased", "chases", "chase", "searched", "searches", "search",
        "found", "finds", "find", "saw", "sees", "see",
        "heard", "hears", "hear", "felt", "feels", "feel",
        "knew", "knows", "know", "remembered", "remembers", "remember",
        "thought", "thinks", "think", "wondered", "wonders", "wonder",
        "decided", "decides", "decide", "realized", "realizes", "realize",
        "climbed", "climbs", "climb", "carried", "carries", "carry",
        "pushed", "pushes", "push", "pulled", "pulls", "pull",
        "was", "is", "had", "has",
    )

    # Capitalized words that are pronouns or sentence-starting
    # function words rather than character names, so the
    # subject-verb heuristic below must never treat them as names.
    NARRATIVE_SUBJECT_EXCLUSIONS = {
        "He", "She", "It", "They", "We", "You", "I",
        "His", "Her", "Its", "Their", "Our", "Your",
        "This", "That", "These", "Those",
        "There", "Here", "Who", "What", "Which",
    }

    LOCATION_PATTERNS = (
        r"\bin\s+(?:the\s+)?([^,.!?]+)",
        r"\bat\s+(?:the\s+)?([^,.!?]+)",
        r"\bon\s+(?:the\s+)?([^,.!?]+)",
        r"\binside\s+(?:the\s+)?([^,.!?]+)",
        r"\bnear\s+(?:the\s+)?([^,.!?]+)",
        r"\bthrough\s+(?:the\s+)?([^,.!?]+)",
    )

    TIME_WORDS = {
        "sunrise": "sunrise",
        "dawn": "dawn",
        "morning": "morning",
        "noon": "midday",
        "midday": "midday",
        "afternoon": "afternoon",
        "sunset": "sunset",
        "evening": "evening",
        "dusk": "dusk",
        "night": "night",
        "midnight": "midnight",
    }

    MOOD_WORDS = (
        "tense",
        "dangerous",
        "joyful",
        "sad",
        "melancholic",
        "romantic",
        "hopeful",
        "mysterious",
        "dark",
        "peaceful",
        "urgent",
        "violent",
        "dramatic",
        "calm",
        "fearful",
        "exciting",
        "warm",
        "lonely",
        "eerie",
        "epic",
    )

    WEATHER_WORDS = (
        "rain",
        "rainy",
        "storm",
        "stormy",
        "fog",
        "foggy",
        "snow",
        "snowy",
        "wind",
        "windy",
        "clear",
        "cloudy",
    )

    def __init__(
        self,
        project_root: Path | str,
    ):
        self.project_root = Path(
            project_root
        )

        self.references = ReferenceManager(
            self.project_root
        )

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = str(
            text or ""
        )

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def _split_story(
        self,
        story: str,
    ) -> list[StoryUnit]:

        story = self._clean_text(
            story
        )

        if not story:
            return []

        paragraphs = [
            self._clean_text(
                paragraph
            )
            for paragraph
            in re.split(
                r"\n\s*\n+",
                story,
            )
            if self._clean_text(
                paragraph
            )
        ]

        if not paragraphs:
            paragraphs = [story]

        units: list[str] = []

        for paragraph in paragraphs:

            # Protect common honorifics/abbreviations before sentence
            # splitting so names such as "Dr. Elara Voss" remain intact.
            protected = paragraph
            abbreviation_patterns = (
                r"\bDr\.",
                r"\bMr\.",
                r"\bMrs\.",
                r"\bMs\.",
                r"\bMiss\.",
                r"\bProf\.",
                r"\bCapt\.",
                r"\bCmdr\.",
                r"\bLt\.",
                r"\bCol\.",
                r"\bGen\.",
                r"\bSgt\.",
                r"\bSt\.",
                r"\bJr\.",
                r"\bSr\.",
                r"\bVs\.",
                r"\bEtc\.",
                r"\bE\.g\.",
                r"\bI\.e\.",
            )
            for pattern in abbreviation_patterns:
                protected = re.sub(
                    pattern,
                    lambda match: match.group(0).replace(
                        ".",
                        "<dot>",
                    ),
                    protected,
                    flags=re.IGNORECASE,
                )

            sentences = []
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                protected,
            ):
                sentence = sentence.replace("<dot>", ".")
                sentence = self._clean_text(sentence)
                if sentence:
                    sentences.append(sentence)

            if len(sentences) <= 2:
                units.append(paragraph)
                continue

            # A long single paragraph is divided into small
            # narrative beats. Two sentences per beat gives the
            # deterministic fallback useful scene granularity
            # without pretending to be the creative director.
            current: list[str] = []

            for sentence in sentences:

                current.append(sentence)

                transition = re.search(
                    r"\b(?:"
                    r"then|suddenly|meanwhile|later|"
                    r"after|before|when|but|however|"
                    r"finally|eventually|soon|"
                    r"moments later|as soon as"
                    r")\b",
                    sentence,
                    flags=re.IGNORECASE,
                )

                if (
                    len(current) >= 2
                    and transition is not None
                ):
                    units.append(
                        " ".join(current)
                    )
                    current = []
                    continue

                if len(current) >= 2:
                    units.append(
                        " ".join(current)
                    )
                    current = []

            if current:
                units.append(
                    " ".join(current)
                )

        if not units:
            units = [story]

        return [
            StoryUnit(
                order=index,
                text=text,
            )
            for index, text
            in enumerate(
                units,
                start=1,
            )
        ]

    def _rebalance_story_units(
        self,
        units: list[StoryUnit],
    ) -> list[StoryUnit]:
        """Normalize source material into the deterministic 4–6 scene budget.

        This is deliberately non-creative: no new story events are invented.
        When the source is shorter than four narrative units, the original text
        is partitioned into four *structural planning beats* using sentence/word
        boundaries. If the source is a single sentence or otherwise too short to
        split semantically, the same source sentence is carried into the four
        planning beats rather than being mutilated or padded with invented prose.
        Qwen remains responsible for turning those structural beats into distinct
        cinematic scenes.
        """
        if not units:
            return []

        if 4 <= len(units) <= 6:
            return [
                StoryUnit(
                    order=index,
                    text=unit.text,
                )
                for index, unit in enumerate(
                    units,
                    start=1,
                )
            ]

        pieces: list[str] = []

        abbreviation_patterns = (
            r"\bDr\.",
            r"\bMr\.",
            r"\bMrs\.",
            r"\bMs\.",
            r"\bMiss\.",
            r"\bProf\.",
            r"\bCapt\.",
            r"\bCmdr\.",
            r"\bLt\.",
            r"\bCol\.",
            r"\bGen\.",
            r"\bSgt\.",
            r"\bSt\.",
            r"\bJr\.",
            r"\bSr\.",
            r"\bVs\.",
            r"\bEtc\.",
            r"\bE\.g\.",
            r"\bI\.e\.",
        )

        for unit in units:
            protected = str(unit.text or "").strip()

            for pattern in abbreviation_patterns:
                protected = re.sub(
                    pattern,
                    lambda match: match.group(0).replace(
                        ".",
                        "<dot>",
                    ),
                    protected,
                    flags=re.IGNORECASE,
                )

            for sentence in re.split(
                r"(?<=[.!?])\s+",
                protected,
            ):
                sentence = (
                    sentence
                    .replace("<dot>", ".")
                    .strip()
                )
                if sentence:
                    pieces.append(sentence)

        # Four-to-six real narrative pieces are preserved one-to-one.
        if 4 <= len(pieces) <= 6:
            return [
                StoryUnit(
                    order=index,
                    text=piece,
                )
                for index, piece in enumerate(
                    pieces,
                    start=1,
                )
            ]

        if len(pieces) < 4:
            # We need four deterministic topology slots for the production
            # contract, but we must not invent story events or cut words in half.
            full_text = " ".join(pieces).strip()
            if not full_text:
                full_text = " ".join(
                    str(unit.text or "").strip()
                    for unit in units
                    if str(unit.text or "").strip()
                ).strip()

            if not full_text:
                return []

            words = full_text.split()

            if len(words) >= 4:
                # Partition by word boundaries while guaranteeing four non-empty
                # structural units. This preserves every source word exactly once.
                boundaries: list[int] = []
                total = len(words)
                for index in range(1, 4):
                    target = round(total * index / 4)
                    minimum = boundaries[-1] + 1 if boundaries else 1
                    maximum = total - (4 - index)
                    boundary = min(max(target, minimum), maximum)
                    boundaries.append(boundary)

                groups: list[list[str]] = []
                start_word = 0
                for boundary in boundaries + [total]:
                    groups.append(words[start_word:boundary])
                    start_word = boundary

                return [
                    StoryUnit(
                        order=index,
                        text=" ".join(group).strip(),
                    )
                    for index, group in enumerate(
                        groups,
                        start=1,
                    )
                    if group
                ]

            # Extremely short inputs (1–3 words) cannot be split without either
            # losing information or fabricating content. Preserve the complete
            # source in each structural slot; the Director owns cinematic
            # expansion, while the source remains byte-for-byte recoverable here.
            return [
                StoryUnit(
                    order=index,
                    text=full_text,
                )
                for index in range(1, 5)
            ]

        # More than six source pieces: deterministically bucket contiguous source
        # material into exactly six groups, preserving order and all source text.
        buckets: list[list[str]] = [
            []
            for _ in range(6)
        ]

        total = len(pieces)
        for index, piece in enumerate(pieces):
            bucket = min(
                5,
                int(index * 6 / total),
            )
            buckets[bucket].append(piece)

        return [
            StoryUnit(
                order=index,
                text=" ".join(bucket).strip(),
            )
            for index, bucket in enumerate(
                buckets,
                start=1,
            )
            if bucket
        ]

    # ============================================================
    # STORY MODES
    # ============================================================

    def normalize_story(
        self,
        mode: str,
        user_input: str,
    ) -> str:

        story = self._clean_text(
            user_input
        )

        if not story:
            raise ValueError(
                "Story cannot be empty."
            )

        if mode not in VALID_STORY_MODES:
            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        # IMPORTANT:
        # Never inject instructions into the user's story.
        #
        # Qwen is responsible for:
        #   - developing AI stories
        #   - expanding supplied stories
        #   - cinematic interpretation
        #
        # The deterministic planner must not introduce words
        # such as "Treat", "Develop", "Clarify", etc.
        return story

    # ============================================================
    # CHARACTER DISCOVERY
    # ============================================================

    @staticmethod
    def _proper_names(
        story: str,
    ) -> list[str]:

        values = re.findall(
            r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}\b",
            story,
        )

        result = []
        seen = set()

        for value in values:
            value = value.strip()

            if value in ProductionPlanner.COMMON_PROPER_WORDS:
                continue

            words = value.split()

            if not words:
                continue

            if len(words) == 1 and value in {
                "City",
                "Street",
                "Road",
                "House",
                "Tower",
                "Castle",
                "Forest",
                "Mountain",
                "River",
                "Ocean",
            }:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(
                value
            )

        return result

    def detect_character_descriptors(
        self,
        story: str,
    ) -> list[str]:

        story = self._clean_text(
            story
        )

        if not story:
            return []

        role_names = {
            "woman",
            "man",
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

        # Explicit names are authoritative. A phrase such as
        # "a young man named Eli" represents ONE character:
        # Eli. The generic role "man" is metadata for Eli.
        explicit_names: list[str] = []

        named_pattern = re.compile(
            r"\b(?:named|called)\s+"
            r"([A-Z][A-Za-z0-9'_-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b"
        )

        for match in named_pattern.finditer(
            story
        ):

            name = match.group(
                1
            ).strip()

            if not name:
                continue

            if name in self.COMMON_PROPER_WORDS:
                continue

            explicit_names.append(
                name
            )

        # Subject-position names: ordinary narrative prose often uses a
        # character name directly as the grammatical subject (for example
        # "Marcus Chen arrived" or "Dr. Elara Voss stood").  The old
        # one-token expression silently reduced multi-word names to the
        # surname immediately before the verb.  Capture the complete
        # capitalized name, while treating an optional honorific as metadata.
        verb_alternation = "|".join(
            sorted(
                self.NARRATIVE_SUBJECT_VERBS,
                key=len,
                reverse=True,
            )
        )

        subject_verb_pattern = re.compile(
            r"\b(?:"
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+"
            r")?"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\s+"
            r"(?:" + verb_alternation + r")\b"
        )

        # Do not make the finite verb vocabulary a hard ceiling.  Short-film
        # prose routinely contains inflected verbs that are absent from any
        # hand-maintained list (for example "arrived" or "finished").  A
        # conservative morphological fallback catches common -ed/-ing/-s
        # finite/action forms, while the explicit vocabulary above remains
        # the higher-confidence path.
        subject_morphology_pattern = re.compile(
            r"\b(?:"
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+"
            r")?"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\s+"
            r"(?:[a-z]+(?:ed|ing|s))\b"
        )

        appositive_pattern = re.compile(
            r"\b([A-Z][a-z]+)\s*,\s*(?:who|which)\b"
        )

        for pattern in (
            subject_verb_pattern,
            subject_morphology_pattern,
            appositive_pattern,
        ):

            for match in pattern.finditer(
                story
            ):

                name = match.group(
                    1
                ).strip()

                # The regex intentionally permits sentence-leading
                # capitalized words. Remove known function words without
                # disturbing legitimate multi-word names.
                name_tokens = name.split()
                while (
                    len(name_tokens) > 1
                    and name_tokens[0] in self.COMMON_PROPER_WORDS
                ):
                    name_tokens.pop(0)
                name = " ".join(name_tokens).strip()

                if not name:
                    continue

                if name in self.COMMON_PROPER_WORDS:
                    continue

                if name in self.NARRATIVE_SUBJECT_EXCLUSIONS:
                    continue

                if name.lower() in role_names:
                    continue

                if (
                    name.lower()
                    in {
                        existing.lower()
                        for existing in explicit_names
                    }
                ):
                    continue

                explicit_names.append(
                    name
                )

        explicit_lower = {
            name.lower()
            for name in explicit_names
        }

        role_pattern = re.compile(
            r"\b(?:a|an|the)\s+"
            r"(?:[a-z][a-z'-]*\s+){0,3}"
            r"("
            + "|".join(
                sorted(
                    role_names,
                    key=len,
                    reverse=True,
                )
            )
            + r")\b",
            flags=re.IGNORECASE,
        )

        candidates: list[str] = []

        for match in role_pattern.finditer(
            story
        ):

            role = (
                match.group(1)
                .strip()
                .lower()
            )

            if role not in role_names:
                continue

            # Look immediately after the role first. This catches
            # the canonical forms "a man named Eli" and
            # "a woman called Sara" without producing duplicates.
            tail = story[
                match.end():
                min(
                    len(story),
                    match.end() + 48,
                )
            ]

            nearby_name = re.match(
                r"\s*,?\s*(?:named|called)\s+"
                r"([A-Z][A-Za-z0-9'_-]+"
                r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b",
                tail,
            )

            if (
                nearby_name is not None
                and nearby_name.group(1).strip().lower()
                in explicit_lower
            ):
                continue

            # Also check the surrounding phrase for a nearby
            # explicit name. This handles adjective-heavy forms
            # such as "the heavily wounded young man named Eli".
            start = max(
                0,
                match.start() - 80,
            )

            context = story[
                start:
                min(
                    len(story),
                    match.end() + 48,
                )
            ]

            surrounding_name = re.search(
                r"\b(?:named|called)\s+"
                r"([A-Z][A-Za-z0-9'_-]+"
                r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b",
                context,
            )

            if (
                surrounding_name is not None
                and surrounding_name.group(1).strip().lower()
                in explicit_lower
            ):
                continue

            candidates.append(
                role
            )

        # Explicit names always win over generic role descriptors.
        # Keep the named identity and let _make_character infer its
        # role from the surrounding story.
        candidates.extend(
            explicit_names
        )

        result: list[str] = []
        seen: set[str] = set()

        for value in candidates:

            value = str(
                value
            ).strip()

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return self._canonicalize_character_descriptors(
            result
        )

    @staticmethod
    def _appearance_from_story(
        name: str,
        story: str,
    ) -> dict:

        lower = story.lower()

        hair = "stable hairstyle inferred from the story"
        body_build = "natural consistent body structure"
        skin_tone = "preserve consistent skin tone"
        age_range = "consistent age appropriate to the story"

        if "long hair" in lower:
            hair = "long hair"
        elif "short hair" in lower:
            hair = "short hair"
        elif "curly hair" in lower:
            hair = "curly hair"
        elif "straight hair" in lower:
            hair = "straight hair"

        if "beard" in lower:
            stable_marks = [
                "stable beard/facial-hair appearance"
            ]
        else:
            stable_marks = []

        clothing = {}

        for token in (
            "black",
            "white",
            "red",
            "blue",
            "green",
            "leather",
            "armor",
            "jacket",
            "coat",
            "dress",
            "shirt",
            "uniform",
        ):
            if token in lower:
                clothing[token] = (
                    f"{token} clothing detail"
                )

        return {
            "facial_features": (
                f"stable facial identity for {name}"
            ),
            "hair": hair,
            "body_build": body_build,
            "body_proportions": (
                "consistent proportions across every shot"
            ),
            "skin_tone": skin_tone,
            "age_range": age_range,
            "stable_identity_marks": stable_marks,
            "story-derived": True,
            "clothing": clothing,
        }

    def _make_character(
        self,
        name: str,
        index: int,
        story: str,
    ) -> Character:

        appearance = (
            self._appearance_from_story(
                name,
                story,
            )
        )

        role = (
            name
            if name
            and name.lower()
            not in {
                "man",
                "woman",
                "girl",
                "boy",
                "child",
                "person",
            }
            else "story character"
        )

        character = Character(
            character_id=(
                f"character_{index:03d}"
            ),
            name=name,
            role=role,
            description=(
                f"Canonical character identity derived "
                f"from the supplied story for {name}."
            ),
            personality=(
                "Preserve personality and behavior "
                "consistently across the story."
            ),
            appearance=appearance,
            clothing=appearance[
                "clothing"
            ],
            distinctive_features=(
                appearance[
                    "stable_identity_marks"
                ]
            ),
            character_state={
                "origin": (
                    "story-derived canonical profile"
                ),
                "continuity": (
                    "preserve identity and state "
                    "across all relevant shots"
                ),
            },
            continuity_rules=[
                "preserve face geometry",
                "preserve hair and hairline",
                "preserve body proportions",
                "preserve stable identity features",
                "preserve story-state continuity",
            ],
        )

        character.build_identity_profile()
        character.build_story_state_profile()

        source = (
            self.references.get_character_source(
                name
            )
        )

        character.reference_mode = source[
            "mode"
        ]

        character.reference_paths = source[
            "reference_paths"
        ]

        character.reference_video_paths = source[
            "reference_video_paths"
        ]

        character.reference_audio_paths = source[
            "reference_audio_paths"
        ]

        character.reference_path = source[
            "path"
        ]

        character.reference_video_path = source[
            "reference_video_path"
        ]

        character.reference_audio_path = source[
            "reference_audio_path"
        ]

        # No supplied reference is not an error.
        # The story-derived identity profile becomes the
        # canonical identity source.
        if not (
            character.reference_paths
            or character.reference_video_paths
            or character.reference_audio_paths
        ):
            character.reference_mode = (
                "story_generated"
            )

        return character

    @staticmethod
    def _canonicalize_character_descriptors(
        descriptors: list[str],
    ) -> list[str]:
        """
        Remove deterministic short-name aliases when exactly one
        fuller canonical name owns that first-name token.

        Examples:
            Elara Voss + Elara -> Elara Voss
            Marcus Voss + Marcus -> Marcus Voss

        Ambiguous short names are retained rather than guessing.
        """
        cleaned: list[str] = []
        seen: set[str] = set()

        for raw in descriptors:
            value = str(
                raw or ""
            ).strip()

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            cleaned.append(value)

        honorific_pattern = re.compile(
            r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|"
            r"captain|commander|detective|agent)\.?(?:\s+)",
            re.IGNORECASE,
        )

        full_names_by_first: dict[
            str,
            list[str],
        ] = {}

        for value in cleaned:
            normalized = honorific_pattern.sub(
                "",
                value,
                count=1,
            ).strip()

            tokens = normalized.split()

            if len(tokens) >= 2:
                full_names_by_first.setdefault(
                    tokens[0].lower(),
                    [],
                ).append(value)

        result: list[str] = []

        for value in cleaned:
            normalized = honorific_pattern.sub(
                "",
                value,
                count=1,
            ).strip()

            tokens = normalized.split()

            if len(tokens) == 1:
                matches = full_names_by_first.get(
                    tokens[0].lower(),
                    [],
                )

                # A bare first-name token is not strong enough to create
                # another canonical person when a fuller name already exists
                # with that first token. This covers both unique aliases and
                # ambiguous first-name references safely.
                if matches:
                    continue

            result.append(value)

        return result

    def create_characters(
        self,
        story: str,
    ) -> list[Character]:

        descriptors = (
            self._canonicalize_character_descriptors(
                self.detect_character_descriptors(
                    story
                )
            )
        )

        if not descriptors:
            # A story with no explicit human/character descriptor
            # is allowed to remain character-free.
            return []

        characters = []

        for index, descriptor in enumerate(
            descriptors,
            start=1,
        ):

            characters.append(
                self._make_character(
                    descriptor,
                    index,
                    story,
                )
            )

        self.references.resolve_characters(
            characters
        )

        # References are optional now.
        self.references.validate(
            characters,
            require_images=False,
        )

        return characters

    # ============================================================
    # SCENE EXTRACTION
    # ============================================================

    @staticmethod
    def _location(
        text: str,
    ) -> str:

        for pattern in ProductionPlanner.LOCATION_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = (
                    match.group(1)
                    .strip(
                        " ,.;:"
                    )
                )

                if value:
                    return value

        return "cinematic environment"

    @staticmethod
    def _time_of_day(
        text: str,
    ) -> str:

        lower = text.lower()

        for key, value in ProductionPlanner.TIME_WORDS.items():
            if key in lower:
                return value

        return "unspecified time"

    @staticmethod
    def _mood(
        text: str,
    ) -> str:

        lower = text.lower()

        values = [
            word
            for word in ProductionPlanner.MOOD_WORDS
            if word in lower
        ]

        return (
            ", ".join(values)
            if values
            else "cinematic"
        )

    @staticmethod
    def _weather(
        text: str,
    ) -> str:

        lower = text.lower()

        values = [
            word
            for word in ProductionPlanner.WEATHER_WORDS
            if word in lower
        ]

        return (
            values[0]
            if values
            else "natural"
        )

    @staticmethod
    def _lighting(
        text: str,
    ) -> str:

        lower = text.lower()

        if "sunset" in lower:
            return "warm directional sunset light"

        if "sunrise" in lower:
            return "soft sunrise light"

        if "night" in lower:
            return "cinematic night illumination"

        if "neon" in lower:
            return "neon cinematic lighting"

        if "dark" in lower:
            return "low-key dramatic lighting"

        return "cinematic naturalistic lighting"

    def _characters_in_scene(
        self,
        text: str,
        characters: list[Character],
    ) -> list[str]:

        text = self._clean_text(
            text
        )

        if not text or not characters:
            return []

        canonical_names = {
            str(character.name).strip()
            for character in characters
            if str(character.name or "").strip()
        }

        if not canonical_names:
            return []

        # Resolve only references that are deterministically safe.
        # EntityResolver creates:
        #   "Elara Voss" -> "elara voss"
        #   "Elara"      -> "elara voss"
        # and refuses ambiguous aliases such as "Voss".
        aliases = EntityResolver.build_alias_map(
            canonical_names
        )

        lower = text.lower()
        resolved: list[str] = []
        seen: set[str] = set()

        # Longest aliases first so a fuller reference wins before a
        # shorter alias. Token boundaries prevent substring matches.
        for alias in sorted(
            aliases,
            key=len,
            reverse=True,
        ):
            alias = str(alias).strip()

            if not alias:
                continue

            if not re.search(
                rf"(?<![A-Za-z0-9'_-])"
                rf"{re.escape(alias)}"
                rf"(?![A-Za-z0-9'_-])",
                lower,
            ):
                continue

            canonical = aliases[alias]

            if canonical not in seen:
                seen.add(canonical)
                resolved.append(canonical)

        # Role descriptors remain a fallback only when no deterministic
        # name/alias was found. Never allow the generic "story character"
        # role to bind.
        if not resolved:
            for character in characters:
                role = str(
                    character.role or ""
                ).strip().lower()

                if (
                    role
                    and role != "story character"
                    and re.search(
                        rf"(?<![A-Za-z0-9'_-])"
                        rf"{re.escape(role)}"
                        rf"(?![A-Za-z0-9'_-])",
                        lower,
                    )
                ):
                    canonical = str(
                        character.name
                    ).strip().lower()

                    if canonical not in seen:
                        seen.add(canonical)
                        resolved.append(canonical)

        # A single canonical character can safely own an otherwise
        # unnamed scene. With multiple characters, do not guess.
        if (
            not resolved
            and len(characters) == 1
        ):
            resolved.append(
                str(
                    characters[0].name
                ).strip().lower()
            )

        return resolved

    def create_scenes(
        self,
        story: str,
        characters: list[Character],
    ) -> list[Scene]:

        units = self._split_story(
            story
        )

        units = self._rebalance_story_units(
            units
        )

        scenes = []

        for index, unit in enumerate(
            units,
            start=1,
        ):

            mood = self._mood(
                unit.text
            )

            location = self._location(
                unit.text
            )

            scenes.append(
                Scene(
                    scene_id=(
                        f"scene_{index:03d}"
                    ),
                    order=index,
                    location=location,
                    time_of_day=self._time_of_day(
                        unit.text
                    ),
                    weather=self._weather(
                        unit.text
                    ),
                    atmosphere=(
                        f"{mood}; natural cinematic "
                        "environmental ambience"
                    ),
                    description=unit.text,
                    mood=mood,
                    lighting=self._lighting(
                        unit.text
                    ),
                    environment_details=[
                        "cinematic depth",
                        "stable environmental continuity",
                    ],
                    key_props=[],
                    scene_objective=unit.text,
                    characters=(
                        self._characters_in_scene(
                            unit.text,
                            characters,
                        )
                    ),
                    story_summary=unit.text,
                    continuity_notes="",
                    shot_ids=[],
                )
            )

        return scenes

    # ============================================================
    # SHOT PLANNING
    # ============================================================

    @staticmethod
    def _camera(
        order: int,
    ) -> tuple[str, str]:

        choices = [
            (
                "wide establishing shot",
                "slow controlled push-in",
            ),
            (
                "medium cinematic shot",
                "subtle lateral tracking",
            ),
            (
                "close cinematic shot",
                "slow controlled forward movement",
            ),
            (
                "over-the-shoulder shot",
                "gentle cinematic follow",
            ),
        ]

        return choices[
            (order - 1)
            % len(choices)
        ]

    def _refs_for_characters(
        self,
        characters: list[Character],
        names: list[str],
    ):

        by_name = {
            character.name.lower():
                character
            for character in characters
        }

        images = []
        videos = []
        audio = []

        character_image_bindings = {}

        for name in names:

            character = by_name.get(
                name.lower()
            )

            if character is None:
                continue

            character_images = (
                character.normalized_reference_paths()
            )

            character_videos = (
                character.normalized_video_paths()
            )

            character_audio = (
                character.normalized_audio_paths()
            )

            character_image_bindings[
                character.name
            ] = character_images

            for path in character_images:
                if (
                    path not in images
                    and len(images)
                    < H3_MAX_REFERENCE_IMAGES
                ):
                    images.append(
                        path
                    )

            for path in character_videos:
                if (
                    path not in videos
                    and len(videos)
                    < H3_MAX_REFERENCE_VIDEOS
                ):
                    videos.append(
                        path
                    )

            for path in character_audio:
                if (
                    path not in audio
                    and len(audio)
                    < H3_MAX_REFERENCE_AUDIO
                ):
                    audio.append(
                        path
                    )

        return (
            images,
            videos,
            audio,
            character_image_bindings,
        )

    @staticmethod
    def _native_audio_instruction(
        scene: Scene,
    ) -> str:

        return (
            "Native H3 audio generation policy: "
            "generate suitable scene ambience natively. "
            f"Soundscape: {scene.atmosphere}. "
            "Do not require an external audio reference unless "
            "one is explicitly supplied."
        )

    def create_shots(
        self,
        story: str,
        characters: list[Character],
        scenes: list[Scene],
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> list[Shot]:

        shots = []

        by_name = {
            character.name.lower():
                character
            for character in characters
        }

        for scene in scenes:

            selected_characters = list(
                scene.characters
            )

            (
                images,
                videos,
                audio,
                bindings,
            ) = self._refs_for_characters(
                characters,
                selected_characters,
            )

            locks = (
                IdentityContinuity.build_locks(
                    characters,
                    selected_characters,
                )
            )

            reference_bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    images,
                    bindings,
                )
            )

            camera_shot, camera_movement = (
                self._camera(
                    len(shots) + 1
                )
            )

            detailed = (
                f"{scene.description} "
                f"Location: {scene.location}. "
                f"Time: {scene.time_of_day}. "
                f"Weather: {scene.weather}. "
                f"Lighting: {scene.lighting}. "
                f"Mood: {scene.mood}. "
                f"Camera: {camera_shot}; "
                f"{camera_movement}. "
                f"Environment: "
                f"{', '.join(scene.environment_details)}."
            )

            positive_prompt = (
                f"{detailed} "
                f"{self._native_audio_instruction(scene)}"
            )

            negative_prompt = (
                "identity drift, altered face geometry, "
                "different hairstyle, altered hairline, "
                "different body proportions, "
                "altered skin tone, facial deformation, "
                "extra limbs, duplicate person, "
                "inconsistent clothing, inconsistent props"
            )

            (
                positive_prompt,
                negative_prompt,
            ) = IdentityContinuity.merge(
                visual_prompt=positive_prompt,
                locks=locks,
                bindings=reference_bindings,
                negative_prompt=negative_prompt,
            )

            if (
                workflow_mode
                == WORKFLOW_TURBO_REF2V
                or profile == "turbo"
            ):
                selected_workflow = (
                    WORKFLOW_TURBO_REF2V
                )
                steps = TURBO_STEPS
            else:
                selected_workflow = (
                    WORKFLOW_REF2V
                )
                steps = H3_STEPS

            shot = Shot(
                shot_id=(
                    f"shot_{len(shots) + 1:03d}"
                ),
                scene_id=scene.scene_id,
                order=len(shots) + 1,
                duration_seconds=5.2,
                characters=selected_characters,
                location=scene.location,
                action=scene.scene_objective,
                camera_shot=camera_shot,
                camera_movement=camera_movement,
                lighting=scene.lighting,
                mood=scene.mood,
                visual_prompt=positive_prompt,
                retention_analysis=(
                    "Maintain visual continuity, "
                    "clear subject readability and "
                    "cinematic progression."
                ),
                detailed_description=detailed,
                overall_soundscape=(
                    self._native_audio_instruction(
                        scene
                    )
                ),
                non_diegetic_music=(
                    "Subtle cinematic score when "
                    "appropriate to the story."
                ),
                negative_prompt=negative_prompt,
                continuity_notes=(
                    scene.continuity_notes
                ),
                seed=(
                    100000
                    + len(shots)
                ),
                reference_images=images,
                reference_videos=videos,
                reference_audio=(
                    audio[0]
                    if audio
                    else None
                ),
                reference_audio_paths=audio,
                reference_audio_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_audio_paths()
                    )
                    for name in selected_characters
                    if name.lower()
                    in by_name
                },
                reference_video_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_video_paths()
                    )
                    for name in selected_characters
                    if name.lower()
                    in by_name
                },
                speaking_characters=selected_characters,
                speech_text="",
                dialogue_events=[],
                is_scene_boundary=(len(shots) == 0 or shots[-1].scene_id != scene.scene_id),
                character_spatial_bboxes={},
                character_spatial_regions={},
                character_spatial_bboxes_start={},
                character_spatial_bboxes_end={},
                character_spatial_regions_start={},
                character_spatial_regions_end={},
                reference_bindings=reference_bindings,
                identity_locks=locks,
                workflow_mode=selected_workflow,
                keyframe_images=[],
                keyframe_positions=[],
                extend_take_source_video=None,
                width=H3_WIDTH,
                height=H3_HEIGHT,
                fps=H3_FPS,
                frames_per_shot=H3_FRAMES_PER_SHOT,
                steps=steps,
            )

            shots.append(
                shot
            )

        return shots

    # ============================================================
    # PLAN VALIDATION
    # ============================================================

    def _validate_plan(
        self,
        characters,
        shots,
    ):

        for shot in shots:

            if (
                shot.width,
                shot.height,
            ) != (
                H3_WIDTH,
                H3_HEIGHT,
            ):
                raise RuntimeError(
                    f"{shot.shot_id}: invalid H3 resolution."
                )

            if shot.fps != H3_FPS:
                raise RuntimeError(
                    f"{shot.shot_id}: invalid FPS."
                )

            images = len(
                shot.reference_images
            )

            videos = len(
                shot.reference_videos
            )

            audio = len(
                shot.reference_audio_paths
            )

            if images > H3_MAX_REFERENCE_IMAGES:
                raise RuntimeError(
                    f"{shot.shot_id}: too many image refs."
                )

            if videos > H3_MAX_REFERENCE_VIDEOS:
                raise RuntimeError(
                    f"{shot.shot_id}: too many video refs."
                )

            if audio > H3_MAX_REFERENCE_AUDIO:
                raise RuntimeError(
                    f"{shot.shot_id}: too many audio refs."
                )

            if (
                images
                + videos
                + audio
                > H3_MAX_REFERENCE_FILES
            ):
                raise RuntimeError(
                    f"{shot.shot_id}: too many total refs."
                )

            # Audio reference is optional.
            # H3 may generate native audio from the prompt.
            if not audio:
                if (
                    "Native H3 audio generation policy"
                    not in shot.overall_soundscape
                ):
                    raise RuntimeError(
                        f"{shot.shot_id}: missing native audio policy."
                    )

            if shot.characters:
                if not shot.identity_locks:
                    raise RuntimeError(
                        f"{shot.shot_id}: "
                        "character identity locks missing."
                    )

    # ============================================================
    # COMPLETE PLAN
    # ============================================================

    def build(
        self,
        *,
        mode: str,
        user_input: str,
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> dict:

        story = self.normalize_story(
            mode,
            user_input,
        )

        # Always construct the deterministic production foundation.
        # The Director may enrich it, but it must never be responsible for
        # creating the canonical roster or scene topology.
        characters = self.create_characters(
            story
        )

        scenes = self.create_scenes(
            story,
            characters,
        )

        shots = self.create_shots(
            story,
            characters,
            scenes,
            workflow_mode=workflow_mode,
            profile=profile,
        )

        scene_shot_ids = {}

        for shot in shots:
            scene_shot_ids.setdefault(
                shot.scene_id,
                [],
            ).append(
                shot.shot_id
            )

        scene_dicts = []

        for scene in scenes:
            scene.shot_ids = list(
                scene_shot_ids.get(
                    scene.scene_id,
                    [],
                )
            )

            scene_dicts.append(
                scene.to_dict()
            )

        self._validate_plan(
            characters,
            shots,
        )

        character_dicts = [
            character.to_dict()
            for character in characters
        ]

        shot_dicts = [
            shot.to_dict()
            for shot in shots
        ]

        return {
            "story": story,
            "story_mode": mode,
            "profile": profile,
            "workflow_mode": workflow_mode,
            "preview_ready": not director_enabled(),
            "director_pending": director_enabled(),

            "character_count": len(
                character_dicts
            ),
            "scene_count": len(
                scene_dicts
            ),
            "shot_count": len(
                shot_dicts
            ),

            "characters": character_dicts,
            "scenes": scene_dicts,
            "shots": shot_dicts,
            "visual_language": {},

            "width": H3_WIDTH,
            "height": H3_HEIGHT,
            "fps": H3_FPS,
            "frames_per_shot": (
                H3_FRAMES_PER_SHOT
            ),
            "normal_steps": H3_STEPS,
            "turbo_steps": TURBO_STEPS,

            "audio_policy": (
                "Use supplied reference audio when present; "
                "otherwise request native H3 audio generation "
                "from the shot soundscape/dialogue prompt."
            ),
        }
