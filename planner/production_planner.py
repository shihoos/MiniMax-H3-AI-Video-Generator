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
    WORKFLOW_REF2VA,
    WORKFLOW_TURBO_REF2VA,
    director_enabled,
)
from schemas.character import Character
from schemas.scene import Scene
from schemas.shot import Shot
from pipeline.seed_lineage import semantic_content_digest


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
        "With",
        "Without",
        "From",
        "To",
        "By",
        "For",
        "Under",
        "Over",
        "Through",
        "Between",
        "Among",
        "Behind",
        "Beside",
        "Beyond",
        "Across",
        "Until",
        "Toward",
        "Towards",
        "Upon",
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
        """
        Deterministically normalize story units into the production
        scene budget when the source contains enough narrative material.

        This is a topology/budget step, not a creative rewrite:
        source text is preserved, no model call is made, and short
        stories are never padded with invented events.
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
            protected = str(
                unit.text or ""
            ).strip()

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

        if len(pieces) < 4:
            # The director contract requires at least four structural scene
            # inputs. For short source text, create four overlapping context
            # windows from the original material rather than inventing events
            # or cutting the source into meaningless word fragments.
            short_source = [
                piece for piece in pieces if piece
            ]
            if not short_source:
                short_source = [
                    str(unit.text or "").strip()
                    for unit in units
                    if str(unit.text or "").strip()
                ]

            if not short_source:
                return []

            if len(short_source) == 1:
                groups = [
                    list(short_source),
                    list(short_source),
                    list(short_source),
                    list(short_source),
                ]
            elif len(short_source) == 2:
                # Four structural views without inventing facts:
                # establish the first event, preserve the complete source,
                # focus the second event, then close on the complete source.
                groups = [
                    [short_source[0]],
                    [short_source[0], short_source[1]],
                    [short_source[1]],
                    [short_source[0], short_source[1]],
                ]
            else:
                groups = [
                    [short_source[0]],
                    [short_source[0], short_source[1]],
                    [short_source[1], short_source[2]],
                    [short_source[2]],
                ]

            return [
                StoryUnit(
                    order=index,
                    text=" ".join(group).strip(),
                )
                for index, group in enumerate(
                    groups,
                    start=1,
                )
                if any(group)
            ]

        # Four-to-six source sentences are preserved one-to-one.
        if len(pieces) <= 6:
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

        # More than six source sentences are compressed into exactly six
        # deterministic contiguous-ish groups without inventing content.
        buckets: list[list[str]] = [
            []
            for _ in range(6)
        ]

        for index, piece in enumerate(pieces):
            bucket = min(
                5,
                int(
                    index * 6 / len(pieces)
                ),
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

        # ========================================================
        # EVIDENCE MODEL
        # ========================================================
        #
        # Detectors produce evidence, not final identities.
        #
        #   explicit       100
        #   subject_verb    80
        #   appositive      70
        #   role            30
        #   morphology      20
        #
        # Weak evidence must survive occurrence-level validation
        # and the confidence gate before becoming canonical identity.
        evidence: dict[str, dict] = {}

        # Coordinated narrative subjects are a strong explicit signal. Keep this
        # detector inside the same evidence model so constructs such as
        # "Mira, a systems engineer, and Arun, her specialist, arrive..."
        # yield two canonical named entities.
        coordinated_name_pattern = re.compile(
            r"\b"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"\s*,\s*"
            r"[^.!?;]{0,120}?"
            r"\band\b"
            r"\s+"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"(?=\s*(?:,|\b(?:arrived|arrive|appears|appear|"
            r"enters|enter|stands|stand|walks|walk|runs|run|"
            r"leaves|leave|moves|move|descends|descend|"
            r"discovers|discover|chooses|choose|faces|face|"
            r"works|work|returns|return|waits|wait)\b))"
        )

        def add_evidence(
            raw_name: str,
            source: str,
            weight: int,
        ) -> None:

            name = str(
                raw_name or ""
            ).strip()

            if not name:
                return

            name_tokens = name.split()

            while (
                len(name_tokens) > 1
                and name_tokens[0]
                in self.COMMON_PROPER_WORDS
            ):
                name_tokens.pop(0)

            name = " ".join(
                name_tokens
            ).strip()

            if not name:
                return

            if name in self.COMMON_PROPER_WORDS:
                return

            if name in self.NARRATIVE_SUBJECT_EXCLUSIONS:
                return

            if (
                name.lower() in role_names
                and source != "role"
            ):
                return

            key = name.lower()

            item = evidence.setdefault(
                key,
                {
                    "name": name,
                    "score": 0,
                    "sources": set(),
                    "occurrences": 0,
                },
            )

            item["score"] += weight
            item["sources"].add(source)
            item["occurrences"] += 1

        for match in coordinated_name_pattern.finditer(story):
            for group in (1, 2):
                add_evidence(match.group(group), "explicit", 100)

        # ========================================================
        # OCCURRENCE-LEVEL CONTEXT
        # ========================================================

        def sentence_bounds(
            start: int,
            end: int,
        ) -> tuple[int, int]:

            left = max(
                story.rfind(".", 0, start),
                story.rfind("!", 0, start),
                story.rfind("?", 0, start),
                story.rfind("\n", 0, start),
            )

            right_candidates = [
                position
                for position in (
                    story.find(".", end),
                    story.find("!", end),
                    story.find("?", end),
                    story.find("\n", end),
                )
                if position >= 0
            ]

            right = (
                min(right_candidates)
                if right_candidates
                else len(story)
            )

            return (
                left + 1,
                right,
            )

        def validate_occurrence(
            match: re.Match,
            source: str,
        ) -> bool:

            candidate_start = match.start(1)
            candidate_end = match.end(1)

            sentence_start, sentence_end = (
                sentence_bounds(
                    candidate_start,
                    candidate_end,
                )
            )

            before = story[
                sentence_start:candidate_start
            ].strip()

            candidate = story[
                candidate_start:candidate_end
            ].strip()

            if not candidate:
                return False

            # ----------------------------------------------------
            # Explicit names are authoritative.
            # ----------------------------------------------------
            if source == "explicit":
                return True

            # ----------------------------------------------------
            # Do not allow sentence-leading function words to become
            # character identities merely because capitalization makes
            # them look like proper nouns.
            #
            # This uses the existing small structural vocabulary rather
            # than turning the extractor into an ever-growing blacklist.
            # ----------------------------------------------------
            candidate_tokens = candidate.split()

            if (
                len(candidate_tokens) == 1
                and candidate_tokens[0]
                in self.COMMON_PROPER_WORDS
            ):
                return False

            if (
                len(candidate_tokens) == 1
                and candidate_tokens[0]
                in self.NARRATIVE_SUBJECT_EXCLUSIONS
            ):
                return False

            # ----------------------------------------------------
            # Strong subject evidence has already matched a known
            # narrative verb. It is therefore trusted after the
            # basic structural exclusions above.
            # ----------------------------------------------------
            if source == "subject_verb":
                return True

            # ----------------------------------------------------
            # Appositive evidence is intentionally person-oriented:
            #
            #     Anton, who worked there...
            #
            # The "which" form is handled nowhere in the identity
            # path because it commonly describes non-person entities.
            # ----------------------------------------------------
            if source == "appositive":
                return True

            # ----------------------------------------------------
            # Morphology is weak evidence.
            #
            # Crucial protection:
            #
            #     "With trembling hands..."
            #     "After entering the chamber..."
            #     "Before leaving..."
            #
            # These are introductory participial/prepositional
            # constructions, not character subjects.
            #
            # Do NOT globally remove "-ing". Simply refuse to let a
            # sentence-leading "-ing" occurrence create identity by
            # itself.
            # ----------------------------------------------------
            if source == "morphology":

                verb = ""

                if match.lastindex and match.lastindex >= 2:
                    verb = match.group(
                        2
                    ).strip().lower()

                if not verb:
                    return False

            

                # A morphology candidate inside a prepositional phrase
                # is not subject evidence.
                previous_tokens = (
                    before.rstrip(
                        " ,;:-—"
                    ).split()
                    if before
                    else []
                )

                if previous_tokens:

                    previous = (
                        previous_tokens[-1]
                        .strip(
                            "\"'()[]{}"
                        )
                        .lower()
                    )

                    if previous in {
                        "with",
                        "without",
                        "from",
                        "to",
                        "by",
                        "for",
                        "on",
                        "in",
                        "at",
                        "under",
                        "over",
                        "through",
                        "between",
                        "among",
                        "behind",
                        "beside",
                        "beyond",
                        "across",
                        "during",
                        "before",
                        "after",
                        "until",
                        "toward",
                        "towards",
                        "upon",
                    }:
                        return False

                return True

            return False

        # ========================================================
        # 1. EXPLICIT NAMES
        # ========================================================

        named_pattern = re.compile(
            r"\b(?:named|called)\s+"
            r"([A-Z][A-Za-z0-9'_-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b"
        )

        for match in named_pattern.finditer(
            story
        ):
            if validate_occurrence(
                match,
                "explicit",
            ):
                add_evidence(
                    match.group(1),
                    "explicit",
                    100,
                )

        # ========================================================
        # 2. STRONG SUBJECT + KNOWN VERB
        # ========================================================

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
            r"([A-Z][A-Za-z0-9'_-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\s+"
            r"("
            + verb_alternation
            + r")\b"
        )

        for match in subject_verb_pattern.finditer(
            story
        ):
            if not validate_occurrence(
                match,
                "subject_verb",
            ):
                continue

            add_evidence(
                match.group(1),
                "subject_verb",
                80,
            )

        # ========================================================
        # 3. MORPHOLOGICAL FALLBACK
        # ========================================================
        #
        # Keep "-ing".
        #
        # It is useful as weak evidence for narrative constructions
        # that do not use a verb from NARRATIVE_SUBJECT_VERBS.
        #
        # It is NOT sufficient by itself to create identity.
        # ========================================================

        subject_morphology_pattern = re.compile(
            r"\b(?:"
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+"
            r")?"
            r"([A-Z][A-Za-z0-9'_-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\s+"
            r"([a-z]+(?:ed|ing|s))\b"
        )

        for match in subject_morphology_pattern.finditer(
            story
        ):
            if not validate_occurrence(
                match,
                "morphology",
            ):
                continue

            add_evidence(
                match.group(1),
                "morphology",
                20,
            )

        # ========================================================
        # 4. APPOSITIVE IDENTITY
        # ========================================================

        appositive_pattern = re.compile(
            r"\b([A-Z][a-z]+)\s*,\s*who\b"
        )

        for match in appositive_pattern.finditer(
            story
        ):
            if not validate_occurrence(
                match,
                "appositive",
            ):
                continue

            add_evidence(
                match.group(1),
                "appositive",
                70,
            )

        # ========================================================
        # 5. ROLE DESCRIPTORS
        # ========================================================

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

        explicit_keys = {
            key
            for key, item in evidence.items()
            if "explicit" in item["sources"]
        }

        for match in role_pattern.finditer(
            story
        ):
            role = (
                match.group(1)
                .strip()
                .lower()
            )

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
                in explicit_keys
            ):
                continue

            add_evidence(
                role,
                "role",
                30,
            )

        # ========================================================
        # 6. CONFIDENCE GATE
        # ========================================================

        accepted: list[str] = []

        for item in evidence.values():

            name = item["name"]
            sources = item["sources"]
            score = item["score"]
            occurrences = item["occurrences"]

            # Explicit identity is authoritative.
            if "explicit" in sources:
                accepted.append(
                    name
                )
                continue

            # Strong subject/appositive evidence is accepted when
            # the occurrence passed contextual validation.
            if (
                "subject_verb" in sources
                or "appositive" in sources
            ):
                accepted.append(
                    name
                )
                continue

            # Role descriptors retain the deterministic fallback.
            if (
                "role" in sources
                and score >= 30
            ):
                accepted.append(
                    name
                )
                continue

            # Morphological evidence is deliberately weak for ambiguous
            # single-token candidates, but a multi-word proper name in
            # subject morphology is strong enough to establish identity
            # from one validated occurrence.
            morphology_valid = False

            if "morphology" in sources:
                tokens = item["name"].split()

                # Example:
                #   "Elena Kovalenko stumbled ..."
                #
                # A multi-token proper name is a strong structural signal.
                if len(tokens) >= 2:
                    morphology_valid = True

                else:
                    # One-token morphology remains weak and therefore
                    # requires repeated validated occurrences.
                    morphology_valid = (
                        occurrences >= 2
                        and score >= 40
                    )

            if morphology_valid:
                accepted.append(
                    name
                )

        return self._canonicalize_character_descriptors(
            accepted
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
            # High-confidence fallback for ordinary narrative prose.
            # Example:
            #   "Mira, a polar systems engineer, ..."
            #   "Arun, her communications specialist, ..."
            #
            # Deliberately do NOT use _proper_names() because that
            # intentionally recognizes broad capitalized tokens and can
            # return locations/pronouns such as "Arctic" and "They".

            appositive_pattern = re.compile(
                r"\b(?:"
                r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|"
                r"Captain|Commander|Detective|Agent)\.?\s+"
                r")?"
                r"([A-Z][A-Za-z0-9'_-]+"
                r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
                r",\s*"
                r"(?=(?:a|an|the|his|her|their|my|our|whose)\b)",
                flags=0,
            )

            fallback_names = []
            seen_names = set()

            pronouns = {
                "they",
                "them",
                "he",
                "him",
                "she",
                "her",
                "it",
                "we",
                "us",
                "i",
                "you",
            }

            for match in appositive_pattern.finditer(story):
                name = match.group(1).strip()

                if not name:
                    continue

                if name in self.COMMON_PROPER_WORDS:
                    continue

                if name in self.NARRATIVE_SUBJECT_EXCLUSIONS:
                    continue

                if name.lower() in pronouns:
                    continue

                key = name.lower()

                if key in seen_names:
                    continue

                seen_names.add(key)
                fallback_names.append(name)

            descriptors = (
                self._canonicalize_character_descriptors(
                    fallback_names
                )
            )

        if not descriptors:
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
                == WORKFLOW_TURBO_REF2VA
                or profile == "turbo"
            ):
                selected_workflow = (
                    WORKFLOW_TURBO_REF2VA
                )
                steps = TURBO_STEPS
            else:
                selected_workflow = (
                    WORKFLOW_REF2VA
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
            shot.semantic_content_digest = semantic_content_digest(shot.to_dict())
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
