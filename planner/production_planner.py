from __future__ import annotations

import logging
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
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_STEPS,
    H3_WIDTH,
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


LOGGER = logging.getLogger(__name__)


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

    RELATIONSHIP_TERMS = {
        "father", "mother", "dad", "mom", "parent",
        "son", "daughter", "child", "brother", "sister",
        "husband", "wife", "partner", "fiance", "fiancee",
        "uncle", "aunt", "cousin", "grandfather", "grandmother",
        "grandson", "granddaughter", "nephew", "niece",
        "commander", "captain", "mentor", "teacher", "guardian",
        "assistant", "handler", "colleague", "friend",
    }

    GENERIC_PERSON_LABELS = {
        "man", "woman", "boy", "girl", "child", "person",
        "hero", "heroine", "explorer", "detective", "scientist",
        "soldier", "warrior", "king", "queen", "robot", "android",
        "pilot", "doctor", "guard", "officer", "stranger",
        # Bare honorifics/titles are lexical prefixes, never standalone
        # canonical identities (e.g. ``Dr. Elena Voss`` must not create ``Dr``).
        "dr", "prof", "professor", "mr", "mrs", "ms", "miss",
        "captain", "commander", "agent",
    }

    # A descriptive character is a recurring person who has no stable proper
    # name but DOES have a grounded distinguishing description.  Bare role words
    # such as ``man``/``woman`` are never canonical identities.
    DESCRIPTIVE_IDENTITY_ROLES = {
        "man", "woman", "boy", "girl", "person", "child",
        "detective", "scientist", "soldier", "warrior", "king", "queen",
        "robot", "android", "pilot", "doctor", "guard", "officer",
        "stranger", "captain", "commander", "engineer", "teacher",
        "nurse", "driver", "explorer", "hero", "heroine",
    }

    COMMON_PROPER_WORDS = {
        "The",
        "A",
        "An",
        "Then",
        "Now",
        "When",
        "While",
        "After",
        "Before",
        "Suddenly",
        "Meanwhile",
        "Finally",
        "Later",
        "Soon",
        "Still",
        "Yet",
        "Eventually",
        "Afterward",
        "Afterwards",
        "Immediately",
        "Instead",
        "Elsewhere",
        "Outside",
        "Inside",
        "Below",
        "Above",
        "Nearby",
        "Slowly",
        "Quietly",
        "Silently",
        "Perhaps",
        "Indeed",
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
    NARRATIVE_NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth",
    }

    NARRATIVE_SUBJECT_EXCLUSIONS = {
        "He", "She", "It", "They", "We", "You", "I",
        "His", "Her", "Its", "Their", "Our", "Your",
        "This", "That", "These", "Those",
        "There", "Here", "All", "Who", "What", "Which",
        "Why", "Where", "How", "Whom", "Whose", "Whether",
        "Someone", "Somebody", "Everyone", "Everybody", "Nobody", "Noone",
        "Anyone", "Anybody", "Anything", "Something", "Nothing", "Everything",
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November",
        "December",
    }

    # Auxiliary/copular verbs are grammatically useful for real character
    # subjects ("Sara was hiding", "Elena had left"), but they are dangerous
    # when capitalization is the only name signal ("Arctic station had...").
    # They therefore receive an additional noun-phrase / optional NER gate
    # rather than being trusted as ordinary lexical action verbs.
    AUXILIARY_SUBJECT_VERBS = {
        "am", "is", "are", "was", "were",
        "be", "been", "being",
        "have", "has", "had",
    }

    # Small fixed semantic head vocabulary for obvious non-person spans.
    NON_PERSON_HEAD_WORDS = {
        "airport", "arena", "base", "basin", "bay", "beach", "bridge",
        "building", "camp", "canyon", "castle", "cave", "center", "centre",
        "chamber", "channel", "city", "coast", "compound", "corridor",
        "country", "crater", "desert", "district", "facility", "field",
        "forest", "fort", "fortress", "garden", "glacier", "harbor", "harbour",
        "headquarters", "island", "lake", "library", "mine", "mountain",
        "museum", "ocean", "outpost", "park", "planet", "port", "prison",
        "project", "region", "research", "reservoir", "river", "road",
        "ruins", "school", "sea", "ship", "shore", "spaceport", "station",
        "temple", "terminal", "theater", "theatre", "tower", "town",
        "valley", "village", "warehouse", "world", "zone",
        "command", "corps", "division", "agency", "bureau", "council",
        "committee", "authority", "administration", "department",
        "ministry", "office", "organization", "organisation", "network",
        "alliance", "coalition", "federation", "union", "guild",
        "brigade", "battalion", "squadron", "fleet", "regiment",
    }

    # Prefixes that structurally identify a multi-token span as a project,
    # protocol, system, access state, or other non-person entity rather than
    # a human/sentient character name. These are deterministic safety guards;
    # explicit named-character evidence bypasses them.
    NON_PERSON_PREFIX_WORDS = {
        "protocol", "project", "class", "operation", "phase", "program",
        "system", "network", "initiative", "experiment", "sequence",
        "procedure", "authorization", "access", "containment", "category",
        "sector", "level", "file", "message", "warning", "terminal",
        "station", "facility", "mission", "module", "unit", "version",
    }

    # Qwen-owned roster safety filter. These are unambiguous production non-person
    # surfaces; this filter may reject a Qwen candidate, but it never creates one.
    QWEN_NON_PERSON_TOKENS = {
        "access", "warning", "danger", "caution", "error", "alert", "authorized",
        "restricted", "denied", "granted", "status", "protocol", "override",
        "locked", "unlocked", "confirmed", "initiated", "activated", "unknown",
        "station", "vault", "chamber", "corridor", "terminal", "system", "network",
        "project", "experiment", "operation", "mission", "module", "unit", "version",
        "static", "silence", "echo", "noise", "thunder", "hum", "frost", "ice",
        "snow", "dust", "smoke", "steam", "fog", "mist", "recording", "message",
        "signal", "broadcast", "transmission", "radio", "screen", "computer",
        "photograph", "photo", "portrait", "image", "map", "journal", "log", "entry",
    }



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

        # AI Story / Expand Story deliberately emit six scene-sized paragraphs.
        # Preserve those paragraph boundaries as production topology instead of
        # re-splitting them into sentence beats. This is deterministic and source-safe.
        if 4 <= len(paragraphs) <= 6:
            return [
                StoryUnit(
                    order=index,
                    text=paragraph,
                )
                for index, paragraph in enumerate(
                    paragraphs,
                    start=1,
                )
            ]

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
    def _identity_detection_text(story: str) -> str:
        """Normalize prose-only markup/punctuation for character evidence matching."""
        text = str(story or "")
        # Character identity detection is semantic; Markdown emphasis is not.
        text = re.sub(r"(?<!\w)[*_~`]+|[*_~`]+(?!\w)", "", text)
        # Treat em/en dashes as clause boundaries for descriptive-identity matching.
        text = re.sub(r"\s*(?:—|–|--)+\s*", ", ", text)
        return text

    @classmethod
    def _explicit_source_character_names(cls, story: str) -> list[str]:
        """Discover explicit source-grounded named identities independently of weak heuristics."""
        text = cls._identity_detection_text(story)
        found: list[str] = []
        title = (
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|"
            r"Commander|Detective|Agent)\.?\s+"
        )
        proper = (
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+(?!and\b|or\b|but\b)"
            r"[A-Z][A-Za-z0-9'_-]+){0,2})"
        )
        patterns = (
            rf"\b(?i:named|called)\s+(?:{title})?{proper}\b",
            rf"\b(?:nameplate|name tag|badge|plaque)\b"
            rf"[^:;.!?]{{0,90}}[:\-]\s*(?:{title})?{proper}\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = str(match.group(1) or "").strip()
                if not value:
                    continue
                value = re.sub(
                    r"^(?:dr|doctor|prof|professor|mr|mrs|ms|miss|captain|commander|detective|agent)\.?\s+",
                    "", value, count=1, flags=re.IGNORECASE,
                ).strip()
                if value and value not in cls.COMMON_PROPER_WORDS:
                    found.append(value)
        return cls._canonicalize_character_descriptors(found)

    def detect_character_descriptors(
        self,
        story: str,
    ) -> list[str]:

        story = self._clean_text(
            story
        )
        detection_story = self._identity_detection_text(story)

        if not story:
            return []

        role_names = set(self.DESCRIPTIVE_IDENTITY_ROLES)

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

        # Weak single-token capitalized nouns need a structural guard. These
        # are common locations/objects in narrative prose, not an exhaustive
        # blacklist of names. Explicitly named characters still bypass this
        # suppression.
        non_person_tokens = {
            "station", "city", "street", "road", "house", "tower", "castle",
            "forest", "mountain", "river", "ocean", "building", "room", "door",
            "platform", "corridor", "vault", "chamber", "world", "earth", "moon",
            "sun", "signal", "gate", "bridge", "ship", "train", "village",
            "kingdom", "planet", "island", "coast", "sky", "shadow", "darkness",
            "storm", "rain", "fire", "water", "wind", "light", "camera", "radio",
            "screen", "computer", "terminal", "vehicle", "car", "truck", "boat",
            "weapon", "map", "letter", "message", "phone", "alarm", "machine",
            "system", "engine", "device", "night", "morning", "dawn", "sunset",
            "evening", "afternoon", "midnight",
            # Sound, weather, and substance nouns that start sentences as
            # subjects ("Static answered.", "Silence fell.") but are never people.
            "static", "silence", "echo", "echoes", "noise", "thunder", "hum",
            "frost", "ice", "snow", "dust", "smoke", "steam", "fog", "mist",
            "metal", "steel", "glass", "concrete", "debris", "ash", "blood",
            "heat", "cold", "power", "electricity", "emergency", "red",
            "nothing", "darkness", "gas", "air", "sound", "voice", "feedback",
            "interference", "data", "footage", "recording",
            # On-screen / signage words that open a sentence ("Access granted...", "Warning: ...").
            "access", "warning", "danger", "caution", "error", "alert", "authorized",
            "restricted", "denied", "granted", "status", "protocol", "override",
            "locked", "unlocked", "confirmed", "initiated", "activated", "unknown",
        }

        # Common verbs that appear frequently in short-story prose but were not
        # part of the older closed narrative-verb vocabulary.
        additional_subject_verbs = {
            "said", "says", "told", "tells", "asked", "asks", "gave", "gives",
            "got", "gets", "made", "makes", "used", "uses", "took", "takes",
            "brought", "brings", "sent", "sends", "kept", "keeps", "lost", "loses",
            "met", "meets", "joined", "joins", "needed", "needs", "wanted", "wants",
            "tried", "tries", "started", "starts", "began", "begins", "continued",
            "continues", "stopped", "stops", "called", "calls", "answered", "answers",
            "returned", "returns", "came", "comes", "went", "goes", "arrived", "arrive", "arrives",
            "helped", "helps", "saved", "saves", "rescued", "rescues", "protected", "protects",
            "warned", "warns", "trusted", "trusts", "believed", "believes",
            "forgot", "forgets", "understood", "understands", "noticed", "notices",
            "appeared", "appears", "became", "becomes", "wore", "wears",
            # Irregular past tenses with no regular -ed/-ing/-s surface form,
            # so the morphology fallback can never reach them on its own.
            "led", "leads", "chose", "chooses", "grew", "grows", "drew", "draws",
            "threw", "throws", "wrote", "writes", "rose", "rises", "fell", "falls",
            "sank", "sinks", "shook", "shakes", "swept", "sweeps", "crept", "creeps",
            "sought", "seeks", "caught", "catches", "taught", "teaches", "bought", "buys",
            "dealt", "deals", "spent", "spends", "built", "builds", "sold", "sells",
            "hung", "hangs", "rang", "rings", "sang", "sings", "swam", "swims",
            "stole", "steals", "broke", "breaks", "woke", "wakes", "froze", "freezes",
            "tore", "tears", "swore", "swears", "rode", "rides", "bled", "bleeds",
            "fled", "flees", "shed", "sheds", "spread", "spreads", "bore", "bears",
            # Common narrative action verbs unlikely to double as ordinary nouns.
            "guided", "guides", "commanded", "commands", "signaled", "signals",
            "stepped", "steps", "stood", "stands", "walked", "walks",
            "gestured", "gestures", "muttered", "mutters", "murmured", "murmurs",
            "screamed", "screams", "flinched", "flinches", "hesitated", "hesitates",
            "glanced", "glances", "spotted", "spots", "shoved", "shoves",
            "dragged", "drags", "lifted", "lifts", "dropped", "drops",
            "crawled", "crawls", "crouched", "crouches", "ducked", "ducks",
            "dodged", "dodges", "blocked", "blocks", "aimed", "aims",
            "fired", "fires", "loaded", "loads", "activated", "activates",
            "scouted", "scouts", "deactivated", "deactivates", "triggered", "triggers",
        }

        subject_verbs = set(self.NARRATIVE_SUBJECT_VERBS) | additional_subject_verbs

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

        def has_non_person_prefix_context(
            candidate: str,
            match: re.Match,
            source: str,
        ) -> bool:
            """Reject a weak candidate only when its surrounding clause clearly frames it as non-person.

            Prefix words such as ``Protocol`` or ``Project`` are useful safety cues,
            but they are not globally forbidden: a sentient identity can legitimately
            be named ``Unit Seven`` or ``Project Orion``. The deterministic detector
            therefore uses these prefixes only when the local syntax also identifies
            the surface as a non-person entity.
            """
            tokens = [
                token.strip(" ,.;:!?()[]{}\\\"'")
                for token in str(candidate or "").split()
            ]
            if len(tokens) < 2:
                return False
            prefix = tokens[0].lower().rstrip(".,:;!?\"'")
            if prefix not in self.NON_PERSON_PREFIX_WORDS:
                return False

            if source == "object_reference":
                # Object-reference evidence is weak. A deterministic candidate
                # beginning with an explicit project/protocol/system prefix should
                # not become canonical solely because another verb introduced it.
                return True

            if not match.lastindex or match.lastindex < 2:
                return False
            verb = str(match.group(2) or "").strip().lower()
            if verb not in self.AUXILIARY_SUBJECT_VERBS:
                return False

            after = story[match.end(2):sentence_bounds(match.start(1), match.end(1))[1]]
            lowered_after = after.lower()

            # Copular predicates that explicitly identify a protocol/project/system
            # state or type are strong non-person context.
            if re.search(
                r"\b(?:a|an|the)\s+(?:[^,;.!?]{0,40}\b(?:sequence|protocol|project|system|program|operation|procedure|experiment|network|facility|module|message|warning|terminal|mission|station)\b)",
                lowered_after,
            ):
                return True
            if re.search(
                r"\b(?:active|inactive|enabled|disabled|armed|disarmed|operational|offline|online|stable|unstable|failing|failed|triggered|sealed|locked|unlocked|corrupted|expired)\b",
                lowered_after,
            ):
                return True
            return False

        def has_non_name_lowercase_token(candidate: str) -> bool:
            """Reject mixed-case noun phrases such as ``Unauthorized access``."""
            tokens = [
                token.strip(" ,.;:!?()[]{}\\\"'")
                for token in str(candidate or "").split()
            ]
            if len(tokens) < 2:
                return False
            allowed_particles = {
                "de", "da", "del", "della", "di", "du", "la", "le",
                "van", "von", "der", "den", "bin", "ibn",
            }
            for token in tokens[1:]:
                if token.lower() in allowed_particles:
                    continue
                if not token or not token[0].isupper():
                    return True
            return False

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
            common_proper_lower = {str(value).lower() for value in self.COMMON_PROPER_WORDS}
            narrative_exclusions_lower = {str(value).lower() for value in self.NARRATIVE_SUBJECT_EXCLUSIONS}

            while (
                len(name_tokens) > 1
                and name_tokens[0].lower() in common_proper_lower
            ):
                name_tokens.pop(0)

            name = " ".join(
                name_tokens
            ).strip()

            if not name:
                return

            if name.lower() in common_proper_lower:
                return

            if name.lower() in narrative_exclusions_lower:
                return

            if (
                len(name.split()) == 1
                and name.lower() in self.NARRATIVE_NUMBER_WORDS
                and source != "explicit"
            ):
                return

            if (
                source in {"subject_verb", "morphology"}
                and has_non_name_lowercase_token(name)
            ):
                return

            if (
                name.lower() in non_person_tokens
                and source != "explicit"
            ):
                return

            if (
                name.lower() in role_names
                and source not in {"role", "descriptive_role"}
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

        simple_coordinated_pattern = re.compile(
            r"\b"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"\s+and\s+"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"\s+([a-z][a-z'-]+)\b"
        )

        for match in simple_coordinated_pattern.finditer(story):
            verb = match.group(3).lower()
            if verb not in subject_verbs and not re.match(r"^[a-z]+(?:ed|ing|s)$", verb):
                continue
            for group in (1, 2):
                candidate = match.group(group).strip()
                candidate_tokens = candidate.split()
                if candidate_tokens and candidate_tokens[0].lower() in {str(value).lower() for value in self.COMMON_PROPER_WORDS}:
                    continue
                add_evidence(candidate, "coordinated", 100)

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

        def has_intervening_lowercase_token(
            match: re.Match,
        ) -> bool:
            """
            Detect a lowercase noun/adjective phrase between a candidate
            proper-name span and its verb.

            Example:
                "Arctic station had ..."
                         ^^^^^^^
            makes "Arctic" a modifier, not the grammatical character head.

            Real names such as:
                "Sara was hiding ..."
                "Elena Kovalenko had left ..."
            have no intervening lowercase token.
            """
            if not match.lastindex or match.lastindex < 2:
                return False

            gap = story[
                match.end(1):match.start(2)
            ]

            return re.search(
                r"(?<![A-Za-z0-9'_-])[a-z][a-z'-]*"
                r"(?![A-Za-z0-9'_-])",
                gap,
            ) is not None

        def has_non_person_semantic_head(
            candidate: str,
        ) -> bool:
            """Return True for multi-token spans ending in a common
            location/facility/project noun."""
            tokens = [
                token.strip(" ,.;:!?()[]{}\\\"'")
                for token in str(candidate or "").split()
            ]

            return (
                len(tokens) >= 2
                and tokens[-1].lower()
                in self.NON_PERSON_HEAD_WORDS
            )

        def candidate_has_definite_non_person_frame(
            match: re.Match,
        ) -> bool:
            """Catch explicit definite noun frames like 'The Frozen Lake was...'."""
            candidate = story[
                match.start(1):match.end(1)
            ].strip()

            if not candidate:
                return False

            sentence_start, _ = sentence_bounds(
                match.start(1),
                match.end(1),
            )

            before_candidate = story[
                sentence_start:match.start(1)
            ].strip()

            previous_word_match = re.search(
                r"([A-Za-z][A-Za-z'-]*)\s*$",
                before_candidate,
            )

            previous_word = (
                previous_word_match.group(1).lower()
                if previous_word_match is not None
                else ""
            )

            return (
                previous_word == "the"
                and has_non_person_semantic_head(candidate)
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

            common_proper_lower = {str(value).lower() for value in self.COMMON_PROPER_WORDS}
            narrative_exclusions_lower = {str(value).lower() for value in self.NARRATIVE_SUBJECT_EXCLUSIONS}
            if (
                len(candidate_tokens) == 1
                and candidate_tokens[0].lower() in common_proper_lower
            ):
                return False

            if (
                len(candidate_tokens) == 1
                and candidate_tokens[0].lower() in narrative_exclusions_lower
            ):
                return False

            # ----------------------------------------------------
            # Strong subject evidence is trusted only when the
            # proper-name span is actually the grammatical subject.
            #
            # This blocks:
            #     "Arctic station had ..."
            #     "Northern outpost was ..."
            # while preserving:
            #     "Sara was ..."
            #     "Elena Kovalenko had ..."
            #
            # Auxiliary/copular constructions receive an optional
            # local NER veto for obvious non-person entities such as
            # locations, facilities, and organizations.
            # ----------------------------------------------------
            if source == "subject_verb":
                if has_intervening_lowercase_token(match):
                    return False

                if has_non_person_prefix_context(candidate, match, source):
                    return False

                if candidate_has_definite_non_person_frame(match):
                    return False

                if has_non_person_semantic_head(candidate):
                    return False

                verb = (
                    match.group(2)
                    .strip()
                    .lower()
                    if match.lastindex and match.lastindex >= 2
                    else ""
                )

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

                if has_intervening_lowercase_token(match):
                    return False

                if has_non_person_prefix_context(candidate, match, source):
                    return False

                if candidate_has_definite_non_person_frame(match):
                    return False

                if has_non_person_semantic_head(candidate):
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

        # Explicit source labels are hard identity evidence even when the
        # surrounding sentence does not use a naming verb. Discover them before
        # Qwen so a model omission cannot erase a source-grounded identity.
        for source_name in self._explicit_source_character_names(story):
            add_evidence(source_name, "explicit", 100)

        nameplate_pattern = re.compile(
            r"\b(?:nameplate|name tag|badge|plaque)\b"
            r"[^:;.!?]{0,90}?[:\-]\s*"
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s*"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){1,3})\b"
        )
        for match in nameplate_pattern.finditer(detection_story):
            add_evidence(match.group(1), "explicit", 100)

        # ========================================================
        # 2. STRONG SUBJECT + KNOWN VERB
        # ========================================================

        verb_alternation = "|".join(
            sorted(
                subject_verbs,
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
            r"([a-z]+(?:ed|ing))\b"
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

        # A second, deliberately conservative subject pass catches ordinary
        # finite verbs not present in the maintained vocabulary. The candidate
        # must still have proper-name capitalization and a verb-like form.
        generic_subject_pattern = re.compile(
            r"\b(?:"
            r"(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+"
            r")?"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\s+"
            r"([a-z][a-z'-]*(?:ed|ing|s))\b"
        )

        for match in generic_subject_pattern.finditer(story):
            candidate = match.group(1).strip()
            verb = match.group(2).lower()
            if verb not in subject_verbs:
                # Regular morphology supplies weak support for unfamiliar verbs.
                if not re.match(r"^[a-z][a-z'-]*(?:ed|ing|s)$", verb):
                    continue
            if not candidate:
                continue
            if any(
                candidate.lower() == str(item["name"]).lower()
                for item in evidence.values()
                if "appositive" in item["sources"]
            ):
                continue

            # The generic pass uses its own permissive morphology rule.
            # Reuse the same noun-phrase boundary logic before allowing a
            # candidate into the evidence model.
            if (
                verb in self.AUXILIARY_SUBJECT_VERBS
                and re.search(
                    r"(?<![A-Za-z0-9'_-])[a-z][a-z'-]*"
                    r"(?![A-Za-z0-9'_-])",
                    story[match.end(1):match.start(2)],
                )
            ):
                continue

            if validate_occurrence(match, "morphology"):
                add_evidence(candidate, "morphology", 35 if len(candidate.split()) >= 2 else 25)

        # ========================================================
        # 4. APPOSITIVE IDENTITY
        # ========================================================

        appositive_pattern = re.compile(
            r"\b(?:(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+)?"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"\s*,\s*(?=(?:a|an|the|who|whose|his|her|their|my|our)\b)",
        )

        appositive_spans = []
        for match in appositive_pattern.finditer(
            story
        ):
            appositive_spans.append(
                (match.start(), match.end())
            )
            if not validate_occurrence(
                match,
                "appositive",
            ):
                continue

            add_evidence(
                match.group(1),
                "appositive",
                90,
            )

        # ========================================================
        # 5. PROPER-NAME OBJECT / COMPLEMENT REFERENCES
        # ========================================================
        # A character can first appear as the object of a verb:
        #   "Sara gives Eli a map."
        #   "Mira follows Arun."
        # These references are strong enough to preserve a canonical entity.
        object_name_pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"(?![A-Za-z0-9'_-])"
        )

        for match in object_name_pattern.finditer(story):
            candidate = match.group(1).strip()
            if candidate.lower() in {str(value).lower() for value in self.COMMON_PROPER_WORDS}:
                continue
            if candidate.lower() in {str(value).lower() for value in self.NARRATIVE_SUBJECT_EXCLUSIONS}:
                continue
            previous_text = story[:match.start(1)].rstrip(" ,;:!?\"'()[]{}")
            previous_tokens = previous_text.split()
            previous_word = previous_tokens[-1].lower() if previous_tokens else ""
            if previous_word not in subject_verbs:
                continue

            # A capitalized object can be followed by a normal determiner
            # ("Sara gives Eli a map"), so do not reject merely because some
            # lowercase word follows. Only reject when the immediately
            # following token is already recognized as a common non-person
            # noun ("entered Arctic station").
            tail = story[match.end(1):]
            next_word_match = re.match(
                r"\s+([a-z][a-z'-]*)\b",
                tail,
            )
            if (
                next_word_match is not None
                and next_word_match.group(1).lower()
                in non_person_tokens
            ):
                continue

            if has_non_person_semantic_head(candidate):
                continue

            if has_non_person_prefix_context(candidate, match, "object_reference"):
                continue

            add_evidence(
                candidate,
                "object_reference",
                75,
            )

        # ========================================================
        # 5. VOCATIVE / DIRECT ADDRESS
        # ========================================================
        # A quoted direct address such as 'Eli, run!' is strong identity
        # evidence even when the addressed character has no subject verb in
        # the same sentence.
        vocative_pattern = re.compile(
            r"(?:^|[\"'“”])\s*"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"\s*,\s*(?=[a-z][a-z'-]+\b)",
            flags=re.MULTILINE,
        )

        for match in vocative_pattern.finditer(story):
            if validate_occurrence(match, "vocative"):
                add_evidence(
                    match.group(1),
                    "vocative",
                    90,
                )

        # ========================================================
        # 6. GROUNDED DESCRIPTIVE CHARACTER IDENTITIES
        # ========================================================
        # Preserve a recurring unnamed person only when the story supplies a
        # distinguishing phrase. A bare ``man``/``woman`` remains non-canonical.
        descriptive_roles = "|".join(
            sorted(self.DESCRIPTIVE_IDENTITY_ROLES, key=len, reverse=True)
        )
        descriptive_pattern = re.compile(
            r"\b(?:a|an|the)\s+"
            r"((?:[a-z][a-z'-]+\s+){0,3}(?:" + descriptive_roles + r"))"
            r"\s+((?:with|wearing|in|holding|carrying|covered|marked|standing|sitting|"
            r"leaning|looking|whose))\s+"
            r"([^,;.!?—–]{1,70}?)"
            r"(?=\s*(?:,|—|–|--|\b(?:and|who|that|while|as)\b|[.!?;]|$))",
            flags=re.IGNORECASE,
        )
        for match in descriptive_pattern.finditer(detection_story):
            candidate_window_before = detection_story[max(0, match.start() - 60):match.start()]
            raw_candidate = str(match.group(0) or "").strip().lower()
            if any(token in raw_candidate.split() for token in ("photograph", "photo", "picture", "portrait", "image")):
                continue
            if re.search(
                r"\b(?:photograph|photo|picture|portrait|image)\s+of\s*$",
                candidate_window_before,
                flags=re.IGNORECASE,
            ):
                continue
            qualifier_tokens = match.group(3).strip().split()
            trimmed = []
            for token in qualifier_tokens:
                if token.lower() in subject_verbs or token.lower() in self.AUXILIARY_SUBJECT_VERBS:
                    break
                trimmed.append(token)
            if not trimmed:
                continue
            candidate = re.sub(
                r"\s+", " ",
                f"{match.group(1)} {match.group(2)} {' '.join(trimmed)}".strip(),
            )
            if candidate and candidate.lower() not in self.GENERIC_PERSON_LABELS:
                add_evidence(candidate, "descriptive_role", 90)

        # ========================================================
        # 7. ROLE DESCRIPTORS
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

            if any(
                start <= match.start() <= end + 8
                for start, end in appositive_spans
            ):
                continue

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
        # 7. CONFIDENCE GATE
        # ========================================================

        accepted: list[str] = []

        for item in evidence.values():

            name = item["name"]
            sources = item["sources"]
            score = item["score"]
            occurrences = item["occurrences"]

            # Bare generic person labels are never canonical identities. They
            # remain semantic surface evidence for later entity resolution.
            # This prevents strings such as ``man``/``woman`` from becoming
            # fake Character records and poisoning relational hints.
            if (
                name.lower() in self.GENERIC_PERSON_LABELS
                and "explicit" not in sources
            ):
                continue

            # Sentence-leading function words and interrogatives are prose, not character identities.
            if name.lower() in {str(v).lower() for v in self.NARRATIVE_SUBJECT_EXCLUSIONS}:
                continue
            if name.lower() in {str(v).lower() for v in self.COMMON_PROPER_WORDS}:
                continue

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
                or "coordinated" in sources
                or "vocative" in sources
                or "object_reference" in sources
            ):
                accepted.append(
                    name
                )
                continue

            # Role descriptors retain the deterministic fallback.
            if (
                "descriptive_role" in sources
                and score >= 70
                and len(name.split()) >= 2
            ):
                accepted.append(name)
                continue

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

        story_text = str(story or "")
        name_text = str(name or "").strip()

        stripped_name = re.sub(
            r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|commander|detective|agent)\.?\s+",
            "",
            name_text,
            flags=re.IGNORECASE,
        ).strip()
        name_tokens = stripped_name.split()
        appearance_aliases = [
            value
            for value in (
                name_text,
                stripped_name,
                name_tokens[0] if name_tokens else "",
            )
            if value
        ]

        sentence_pattern = re.compile(
            r"[^.!?\n]+(?:[.!?](?:\s+|$)|$)"
        )
        relevant_sentences = []
        for sentence_match in sentence_pattern.finditer(story_text):
            sentence = sentence_match.group(0).strip()
            lower_sentence = sentence.lower()
            if any(
                re.search(
                    rf"(?<![A-Za-z0-9'_-]){re.escape(alias.lower())}(?![A-Za-z0-9'_-])",
                    lower_sentence,
                )
                for alias in appearance_aliases
            ):
                relevant_sentences.append(sentence)

        local_text = " ".join(relevant_sentences)
        lower = local_text.lower()
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
            "story-derived": bool(local_text),
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

    @classmethod
    def _canonicalize_character_descriptors(
        cls,
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

        possessive_pattern = re.compile(r"^(.+?)'s$")

        for raw in descriptors:
            value = str(
                raw or ""
            ).strip()

            if not value:
                continue

            # Strip a trailing English possessive marker ("Elena's" ->
            # "Elena") before dedup/alias logic runs. Without this, a
            # possessive-only mention of an already-detected character
            # survives as its own malformed pseudo-character distinct
            # from the base name.
            possessive_match = possessive_pattern.match(value)
            if possessive_match:
                stripped = possessive_match.group(1).strip()
                if stripped:
                    value = stripped

            key = value.lower()

            if key in seen:
                continue

            if key in cls.GENERIC_PERSON_LABELS:
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

    @classmethod
    def character_detection_regression_cases(cls) -> tuple[tuple[str, set[str]], ...]:
        """
        Small, production-local regression corpus for capitalization/entity
        boundary failures. No model inference is required to run these cases.
        """
        return (
            (
                "The Arctic station had been abandoned for years.",
                set(),
            ),
            (
                "The Arctic station is empty.",
                set(),
            ),
            (
                "The Northern outpost was silent.",
                set(),
            ),
            (
                "Sara was hiding near the station.",
                {"Sara"},
            ),
            (
                "Sara had escaped before dawn.",
                {"Sara"},
            ),
            (
                "Elena Kovalenko was waiting outside.",
                {"Elena Kovalenko"},
            ),
            (
                "Someone called her instead. Dr. Kess entered the vault. Protocol Epsilon was active. Unauthorized access detected. Below, the air was cold.",
                {"Kess"},
            ),
            (
                "Eli emerged at dawn. A photograph of a child with his sister's eyes lay inside the crate.",
                {"Eli"},
            ),
            (
                "Unit Seven entered the chamber. Mara watched from the doorway.",
                {"Unit Seven", "Mara"},
            ),
            (
                "Father John waited by the door.",
                {"Father John"},
            ),
        )

    @staticmethod
    def _story_has_character_name(
        story: str,
        name: str,
    ) -> bool:
        """Return True only when a proposed canonical name is present in the story.

        This is a hard anti-hallucination boundary for semantic extraction: the
        language model may recover or classify names, but it may not invent a
        canonical identity that has no textual anchor in the source story.
        """
        story_norm = re.sub(r"[^a-z0-9]+", " ", str(story or "").lower()).strip()
        name_norm = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
        if not story_norm or not name_norm:
            return False
        story_tokens = story_norm.split()
        name_tokens = name_norm.split()
        if not name_tokens:
            return False
        width = len(name_tokens)
        for index in range(0, len(story_tokens) - width + 1):
            if story_tokens[index : index + width] == name_tokens:
                return True
        # Possessive mentions such as "Sara's notebook" normalize to the same
        # token, so they are intentionally covered by the normalization above.
        return False

    @classmethod
    def _semantic_named_surface_is_safe(cls, name: str) -> bool:
        """Reject obvious non-name noun phrases while leaving cast choice to Qwen.

        This is a structural safety boundary, not a character-count rule. Named
        characters remain model-selected when their source surface looks like an
        actual proper name; role/descriptive identities use their dedicated paths.
        """
        value = str(name or "").strip()
        if not value:
            return False
        tokens = [token.strip(" ,.;:!?()[]{}\\\"'") for token in value.split()]
        if not tokens:
            return False
        lowered = [token.lower().rstrip(".,:;!?\"'") for token in tokens]
        if lowered[0] in cls.NARRATIVE_NUMBER_WORDS:
            return False
        common_proper_lower = {str(value).lower() for value in cls.COMMON_PROPER_WORDS}
        narrative_exclusions_lower = {str(value).lower() for value in cls.NARRATIVE_SUBJECT_EXCLUSIONS}
        if lowered[0] in common_proper_lower or lowered[0] in narrative_exclusions_lower:
            return False
        if any(token in {"photo", "photograph", "picture", "portrait", "image"} for token in lowered):
            return False
        if len(tokens) >= 2:
            allowed_particles = {
                "de", "da", "del", "della", "di", "du", "la", "le",
                "van", "von", "der", "den", "bin", "ibn",
            }
            for token in tokens[1:]:
                if token.lower() in allowed_particles:
                    continue
                if not token or not token[0].isupper():
                    return False
        return True

    @classmethod
    def _single_token_name_has_case_proof(cls, story: str, name: str) -> bool:
        """True when a one-word name is distinguishable from a capitalized common word.

        English capitalizes the first word of every sentence, so a sentence-initial
        ``Dust swirled`` / ``"Access granted"`` looks identical to ``Eli stepped``.
        A single token is therefore only provably a proper name when it is
        capitalized somewhere capitalization is NOT forced by position, is
        possessive, is used as a vocative, or is introduced as ``named X``.
        Anything else is genuinely ambiguous and must be adjudicated semantically
        rather than trusted deterministically.
        """
        text = cls._identity_detection_text(story)
        core = str(name or "").strip()
        if not core or not text:
            return False
        if len(core.split()) != 1:
            return True  # multi-token proper names are not subject to this ambiguity
        if cls._hard_named_source_evidence(story, core):
            return True
        esc = re.escape(core)
        for m in re.finditer(r"(?<![A-Za-z0-9'_-])" + esc + r"(?![A-Za-z0-9_-])", text):
            tail = text[m.end():m.end() + 3]
            if re.match(r"(?:'s|\u2019s)", tail):
                return True  # possessive: "Eli's hands"
            head = text[:m.start()].rstrip()
            head = re.sub(r"[\"'\u201c\u2018(\[*_]+$", "", head).rstrip()
            if head and head[-1] not in ".!?\n":
                return True  # capitalized mid-sentence
            if re.match(r"\s*,\s*(?:a|an|the|who|whose|his|her|their|my|our)\b", text[m.end():m.end() + 40]):
                return True  # appositive: "Eli, a scientist, ..."
            if re.match(r"\s*[,!?]", text[m.end():m.end() + 3]) and re.search(
                r"[\"\u201c]\s*$", text[:m.start()]
            ):
                return True  # vocative inside dialogue: "Eli, run!"
        return False

    # ------------------------------------------------------------------
    # Mention-only identities: people who are named but never on screen
    # ------------------------------------------------------------------
    _ABSENT_DOC_NOUNS = (
        r"(?:last|final|first|old|own|earlier|previous)?\s*(?:transmissions?|journals?|logs?|notes?|messages?|"
        r"recordings?|diary|diaries|signature|research|files?|data|secrets?|work|reports?|voice|"
        r"memory|memories|legacy|body|grave|death|name|badge|initials|handwriting|findings|"
        r"experiments?|project|theory|warning|words|final entry)"
    )
    _ABSENT_BEFORE = re.compile(
        r"\b(?:name|names|labell?ed|marked|etched|engraved|stamped|inscribed|signed|printed|scrawled|"
        r"scratched|written|listed|tagged|badge|nameplate|plaque)\b[^.!?\n]{0,50}$"
        r"|\b(?:message|transmission|voice|recording|signal|broadcast|note|letter|call|log|entry|report)s?\b"
        r"[^.!?\n]{0,30}\bfrom\s+(?:(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Captain|Commander|Agent)\.?\s+)?$",
        flags=re.IGNORECASE,
    )
    _ABSENT_AFTER = re.compile(
        r"^\s*(?:had|has|have)\s+(?:\w+\s+)?(?:vanished|disappeared|died|perished|been\s+(?:missing|dead|lost|killed|gone)|"
        r"gone\s+missing|abandoned|left\s+behind|written|recorded|warned|noted|logged|signed|sent)\b"
        r"|^\s*,?\s*(?:was|were)\s+(?:missing|dead|killed|lost|gone|presumed|last\s+seen)\b"
        r"|^\s+(?:granted|denied|required|restricted|confirmed|accepted|rejected|detected|initiated|activated|"
        r"unlocked|locked|override|authorized|level|code|panel)\b",
        flags=re.IGNORECASE,
    )
    _SPEECH_AFTER = re.compile(
        r"^\s*[,.]?\s*(?:said|says|asked|replied|whispered|shouted|called|muttered|answered|warned|told|"
        r"snapped|murmured|cried|yelled|demanded|insisted)\b",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _is_mention_only_identity(cls, story: str, name: str) -> bool:
        """True when EVERY occurrence of a name is evidence, not presence.

        Evidence means: a label/etching/nameplate ("a name etched into the metal: Kovac"),
        a past-perfect disappearance ("Kovac had vanished"), a possessive of a document or
        secret ("Kovac's journal"), a message "from" the person, or on-screen UI text
        ("Access granted"). A name that ever speaks or acts in the present scene is kept.
        """
        text = str(story or "")
        name = str(name or "").strip()
        if not text or not name:
            return False
        pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])" + re.escape(name) + r"(?![A-Za-z0-9_-])",
            flags=re.IGNORECASE,
        )
        found = False
        for match in pattern.finditer(text):
            found = True
            before = text[max(0, match.start() - 80):match.start()]
            after = text[match.end():match.end() + 90]
            if cls._SPEECH_AFTER.search(after):
                return False
            possessive_doc = re.match(
                r"^[\u2019']s\s+" + cls._ABSENT_DOC_NOUNS + r"\b", after, flags=re.IGNORECASE
            )
            if (
                cls._ABSENT_BEFORE.search(before)
                or cls._ABSENT_AFTER.search(after)
                or possessive_doc
            ):
                continue
            return False
        return found

    @classmethod
    def _drop_mention_only_identities(
        cls,
        story: str,
        names: list[str],
        *,
        preserve_empty: bool = True,
    ) -> list[str]:
        """Remove identities that are only labels/backstory/document references.

        ``preserve_empty`` is retained for callers that intentionally need the
        original list when every identity is filtered. The Qwen-authoritative
        reconciliation path does not use this helper to restore rejected Qwen
        candidates; it fails closed when its own candidate set becomes empty.
        """
        kept = [name for name in names if not cls._is_mention_only_identity(story, name)]
        if kept or preserve_empty:
            return kept if kept else list(names)
        return []

    @classmethod
    def _high_confidence_deterministic_character(
        cls,
        story: str,
        name: str,
    ) -> bool:
        """Protect a strong deterministic named identity from one Qwen false-negative."""
        story = str(story or "")
        name = str(name or "").strip()
        if not story or not name:
            return False

        # A sentence-initial single word is not name evidence (capitalization is
        # positional), so it cannot bypass semantic adjudication.
        if not cls._single_token_name_has_case_proof(story, name):
            return False

        escaped_name = re.escape(name)
        name_tokens = name.split()

        subject_verbs = set(cls.NARRATIVE_SUBJECT_VERBS)
        verb_alt = "|".join(
            sorted(
                (re.escape(value) for value in subject_verbs),
                key=len,
                reverse=True,
            )
        )

        # Strong named subject with a maintained narrative verb.
        subject_pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])"
            + escaped_name
            + r"\s+(?:"
            + verb_alt
            + r")\b",
            flags=re.IGNORECASE,
        )
        if subject_pattern.search(story):
            return True

        # Multi-word proper name + unfamiliar regular narrative verb, e.g.
        # "Elena Kovalenko stumbled ...".
        generic_subject_pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])"
            + escaped_name
            + r"\s+[a-z][a-z'-]*(?:ed|ing|s)\b",
            flags=re.IGNORECASE,
        )
        if generic_subject_pattern.search(story):
            definite_non_person = re.compile(
                r"(?<![A-Za-z0-9'_-])the\s+"
                + escaped_name
                + r"\b",
                flags=re.IGNORECASE,
            )
            if not definite_non_person.search(story):
                return True

        # Strong appositive evidence.
        appositive_pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])"
            + escaped_name
            + r"\s*,\s*(?:a|an|the|who|whose|his|her|their|my|our)\b",
            flags=re.IGNORECASE,
        )
        if appositive_pattern.search(story):
            return True

        # Coordinated subject evidence.
        coordinated_pattern = re.compile(
            r"(?<![A-Za-z0-9'_-])"
            + escaped_name
            + r"\s+and\s+"
            r"[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}"
            r"\s+(?:"
            + verb_alt
            + r")\b"
            r"|"
            r"(?<![A-Za-z0-9'_-])"
            r"[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}"
            r"\s+and\s+"
            + escaped_name
            + r"\s+(?:"
            + verb_alt
            + r")\b",
            flags=re.IGNORECASE,
        )
        if coordinated_pattern.search(story):
            return True

        # Direct-address/vocative evidence.
        vocative_pattern = re.compile(
            r"(?:^|[\"'“”])\s*"
            + escaped_name
            + r"\s*,\s*(?=[a-z][a-z'-]+\b)",
            flags=re.MULTILINE,
        )
        return bool(vocative_pattern.search(story))

    @classmethod
    def _extract_relational_character_hints(
        cls,
        story: str,
        canonical_names: list[str],
    ) -> list[dict]:
        """Recover strongly grounded relational identities as semantic hints."""
        story = str(story or "")
        if not story:
            return []

        actual_by_norm = {
            EntityResolver.normalize(name): str(name).strip()
            for name in canonical_names
            if str(name or "").strip()
        }
        aliases = EntityResolver.build_alias_map(actual_by_norm.values())
        records: dict[str, dict] = {}
        relation_alt = "|".join(
            sorted(
                (re.escape(term) for term in cls.RELATIONSHIP_TERMS),
                key=len,
                reverse=True,
            )
        )

        def add(owner: str, relation: str, alias: str, strong: bool) -> None:
            owner_norm = aliases.get(EntityResolver.normalize(owner))
            owner_canonical = actual_by_norm.get(owner_norm or "")
            relation_norm = str(relation or "").strip().lower()
            if not owner_canonical or relation_norm not in cls.RELATIONSHIP_TERMS:
                return
            canonical = f"{owner_canonical}'s {relation_norm}"
            key = EntityResolver.normalize(canonical)
            entry = records.setdefault(
                key,
                {
                    "name": canonical,
                    "relationship_to": owner_canonical,
                    "relationship": relation_norm,
                    "aliases": [],
                    "strong": False,
                },
            )
            if alias and EntityResolver.normalize(alias) != key:
                entry["aliases"].append(str(alias).strip())
            entry["strong"] = bool(entry["strong"] or strong)

        possessive = re.compile(
            r"\b([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
            r"(?:'s|’s)\s+(" + relation_alt + r")\b"
        )
        for match in possessive.finditer(story):
            add(match.group(1), match.group(2), match.group(0), True)

        of_pattern = re.compile(
            r"\b(?:the|a|an)\s+(" + relation_alt + r")\s+of\s+"
            r"([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b"
        )
        for match in of_pattern.finditer(story):
            add(match.group(2), match.group(1), match.group(0), True)

        pronoun_pattern = re.compile(
            r"\b(his|her|their)\s+(" + relation_alt + r")\b",
            flags=re.IGNORECASE,
        )
        for match in pronoun_pattern.finditer(story):
            owners = cls._pronoun_relation_owner_candidates(
                story, match, list(actual_by_norm.values())
            )
            if len(owners) == 1:
                add(owners[0], match.group(2), match.group(0), True)

        for item in records.values():
            item["aliases"] = list(dict.fromkeys(item["aliases"]))
        return list(records.values())

    @classmethod
    def _pronoun_relation_owner_candidates(
        cls,
        story: str,
        match: re.Match,
        canonical_names: list[str],
    ) -> list[str]:
        """Resolve a possessive-pronoun relationship to a discourse owner.

        Do not choose the nearest noun blindly. In text such as
        ``a sentient AI his father had created``, ``AI`` is the object of the
        relative clause, while ``his`` refers back to the established protagonist.
        A canonical noun phrase immediately introduced by ``a/an/the`` before the
        pronoun is therefore excluded as the possessor candidate.
        """
        # Allow the antecedent to be established in the immediately preceding
        # sentence, but keep the window bounded so distant characters do not
        # become accidental owners.
        window_start = max(0, match.start() - 420)
        prefix = story[window_start:match.start()]
        candidates: list[tuple[int, str]] = []
        for canonical in canonical_names:
            canonical = str(canonical or "").strip()
            if not canonical:
                continue
            for occurrence in re.finditer(
                re.escape(canonical), prefix, flags=re.IGNORECASE
            ):
                occ_end = occurrence.end()
                local_before = prefix[max(0, occurrence.start() - 80):occurrence.start()]
                # Exclude the head of an indefinite/definite noun phrase directly
                # preceding the possessive pronoun, e.g. ``a sentient AI his``.
                if re.search(
                    r"\b(?:a|an|the)\s+(?:[a-z][a-z'-]+\s+){0,5}$",
                    local_before,
                    flags=re.IGNORECASE,
                ):
                    continue
                # Keep the discourse-owner window bounded, but allow a
                # character to be separated from a possessive relation by
                # several narrative sentences. Expand Story commonly plants
                # a relationship in memory/action beats before returning to
                # the protagonist (for example, ``Eli ... his father``).
                # The previous 10-word cap silently dropped those grounded
                # relational identities and could collapse a real two-person
                # story into a one-character roster.
                between = prefix[occ_end:]
                between_words = len(between.split())
                if between_words > 60:
                    continue
                candidates.append((occurrence.start(), canonical))
        candidates.sort(key=lambda item: item[0], reverse=True)
        unique = []
        seen = set()
        for _, name in candidates:
            key = EntityResolver.normalize(name)
            if key not in seen:
                unique.append(name)
                seen.add(key)
        return unique[:2]

    @classmethod
    def _descriptive_identity_is_grounded(
        cls,
        story: str,
        name: str,
    ) -> bool:
        """Return True when a qualified descriptive identity is grounded in the story.

        Qwen remains the semantic authority for selecting the identity. This
        helper only checks whether the selected descriptor is plausibly present
        in the source text. Articles and short function words may vary, so
        grounding uses the descriptor's content-token sequence rather than an
        exact phrase match.
        """
        normalized = EntityResolver.normalize(name).strip()
        if not normalized:
            return False

        core_tokens = normalized.split()
        if len(core_tokens) < 2:
            return False

        # Leading articles are semantic surface variation, not identity.
        if core_tokens and core_tokens[0] in {"the", "a", "an"}:
            core_tokens = core_tokens[1:]

        role_tokens = cls.DESCRIPTIVE_IDENTITY_ROLES
        if not any(token in role_tokens for token in core_tokens):
            return False
        if len(core_tokens) < 2:
            return False
        if any(token in {"photo", "photograph", "picture", "portrait", "image"} for token in core_tokens):
            return False

        stopwords = {
            "a", "an", "the", "in", "on", "at", "of", "with",
            "to", "from", "by", "for", "and", "or", "as", "into",
            "onto", "his", "her", "their",
        }
        content_tokens = [token for token in core_tokens if token not in stopwords]
        if len(content_tokens) < 2:
            return False

        story_text = str(story or "").lower()
        story_sentences = [
            re.sub(r"[^a-z0-9]+", " ", segment).strip().split()
            for segment in re.split(r"[.!?\n]+", story_text)
            if segment.strip()
        ]
        if not story_sentences:
            return False

        # Match content tokens in order within one sentence, tolerating
        # articles/prepositions and a small number of extra words. This handles
        # ``the man in the grey coverall`` vs ``the man in grey coverall`` or
        # ``the man wearing a grey coverall`` without combining unrelated
        # words from separate sentences.
        max_gap = 5
        for story_tokens in story_sentences:
            positions = {}
            for index, token in enumerate(story_tokens):
                positions.setdefault(token, []).append(index)

            match_start = None
            match_end = None
            for first_pos in positions.get(content_tokens[0], []):
                current = first_pos
                matched = True
                for token in content_tokens[1:]:
                    next_pos = next(
                        (pos for pos in positions.get(token, []) if current < pos <= current + max_gap + 1),
                        None,
                    )
                    if next_pos is None:
                        matched = False
                        break
                    current = next_pos
                if matched:
                    match_start = first_pos
                    match_end = current
                    break

            if match_start is None or match_end is None:
                continue

            # A descriptive person visible only inside a photograph/picture/portrait
            # is not automatically a recurring character. Keep this exclusion, but
            # apply it to the tolerant matched span rather than an exact phrase.
            before_tokens = story_tokens[max(0, match_start - 8):match_start]
            after_tokens = story_tokens[match_end + 1:match_end + 9]
            before_text = " ".join(before_tokens)
            after_text = " ".join(after_tokens)
            visual = r"(?:photograph|photo|picture|portrait|image)"
            if re.search(rf"{visual}\s+of(?:\s+(?:a|an|the))?\s*$", before_text):
                continue
            if re.search(rf"^(?:in|inside)\s+(?:a|an|the)?\s*{visual}\b", after_text):
                continue

            return True

        return False


    @classmethod
    def _hard_named_source_evidence(cls, story: str, name: str) -> bool:
        """Return True only for explicit source labels that should survive a Qwen negative."""
        story = cls._identity_detection_text(story)
        core = re.sub(
            r"^(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s+",
            "", str(name or "").strip(), flags=re.IGNORECASE
        ).strip()
        if not core:
            return False
        patterns = [
            rf"\b(?:named|called)\s+{re.escape(core)}\b",
            rf"\b(?:nameplate|name tag|badge|plaque)\b[^:;.!?]{{0,90}}[:\-]\s*(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Captain|Commander|Detective|Agent)\.?\s*{re.escape(core)}\b",
        ]
        return any(re.search(pattern, story, flags=re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _relational_candidate_is_grounded(
        cls,
        story: str,
        raw: dict,
        canonical_names: list[str],
    ) -> tuple[bool, str | None, dict]:
        if not isinstance(raw, dict):
            return False, None, {}
        identity_type = str(raw.get("identity_type", "") or "").strip().lower()
        if identity_type != "relational_character":
            return False, None, {}

        actual_by_norm = {
            EntityResolver.normalize(name): str(name).strip()
            for name in canonical_names
            if str(name or "").strip()
        }
        aliases = EntityResolver.build_alias_map(actual_by_norm.values())
        owner_norm = aliases.get(
            EntityResolver.normalize(str(raw.get("relationship_to", "") or ""))
        )
        owner = actual_by_norm.get(owner_norm or "")
        relationship = str(raw.get("relationship", "") or "").strip().lower()
        if not owner or relationship not in cls.RELATIONSHIP_TERMS:
            return False, None, {}

        canonical = f"{owner}'s {relationship}"
        supplied_name = str(raw.get("name", "") or "").strip()
        raw_aliases = [
            str(value or "").strip()
            for value in (raw.get("aliases", []) or [])
            if str(value or "").strip()
        ]
        surfaces = [supplied_name, *raw_aliases]
        story_lower = story.lower()

        exact_pair = bool(re.search(
            re.escape(owner.lower()) + r"(?:'s|’s)\s+" + re.escape(relationship) + r"\b",
            story_lower,
        ))
        of_form = bool(re.search(
            r"\b(?:the|a|an)\s+" + re.escape(relationship) + r"\s+of\s+"
            + re.escape(owner.lower()) + r"\b",
            story_lower,
        ))
        alias_anchor = any(
            cls._story_has_character_name(story, surface)
            for surface in surfaces
            if surface
        )

        pronoun_relation = False
        pronoun_pattern = re.compile(
            r"\b(his|her|their)\s+" + re.escape(relationship) + r"\b",
            flags=re.IGNORECASE,
        )
        for pronoun_match in pronoun_pattern.finditer(story):
            owners = cls._pronoun_relation_owner_candidates(
                story, pronoun_match, [owner]
            )
            if owners and EntityResolver.normalize(owners[0]) == EntityResolver.normalize(owner):
                pronoun_relation = True
                break

        grounded = bool(exact_pair or of_form or alias_anchor or pronoun_relation)
        if not grounded:
            return False, None, {}

        semantic_aliases = []
        for surface in raw_aliases:
            if EntityResolver.normalize(surface) == EntityResolver.normalize(canonical):
                continue
            text = str(surface or "").strip()
            generic_surface = EntityResolver.generic_role_surface(text)
            # Preserve the grounded relational role as a non-canonical semantic
            # alias in its stable bare form. This lets a later dialogue speaker
            # such as ``man`` resolve to the already-approved relational identity
            # without ever promoting ``man`` into the canonical character roster.
            if generic_surface:
                semantic_aliases.append(generic_surface)
            else:
                semantic_aliases.append(text)
        if supplied_name and EntityResolver.normalize(supplied_name) != EntityResolver.normalize(canonical):
            text = str(supplied_name).strip()
            generic_surface = EntityResolver.generic_role_surface(text)
            semantic_aliases.append(generic_surface or text)

        return True, canonical, {
            "identity_type": "relational_character",
            "relationship_to": owner,
            "relationship": relationship,
            "semantic_aliases": list(dict.fromkeys(semantic_aliases)),
        }

    @classmethod
    def _semantic_character_metadata(
        cls,
        story: str,
        canonical_names: list[str],
        semantic_result,
        relational_hints: list[dict] | None = None,
        descriptive_hints: list[str] | None = None,
    ) -> dict[str, dict]:
        """Return validated semantic identity metadata keyed by canonical name."""
        result: dict[str, dict] = {}
        candidates = []
        if isinstance(semantic_result, dict):
            candidates = list(semantic_result.get("candidates", []) or [])
            if not candidates:
                candidates = list(semantic_result.get("characters", []) or [])

        for raw in candidates:
            if isinstance(raw, str):
                raw = {
                    "name": raw,
                    "entity_type": "CHARACTER",
                    "is_character": True,
                    "aliases": [],
                    "identity_type": "named_character",
                }
            if not isinstance(raw, dict) or not bool(raw.get("is_character", False)):
                continue
            identity_type = str(raw.get("identity_type", "named_character") or "named_character").strip().lower()
            if identity_type == "relational_character":
                valid, canonical, metadata = cls._relational_candidate_is_grounded(
                    story, raw, canonical_names
                )
                if valid and canonical:
                    result[EntityResolver.normalize(canonical)] = metadata
                continue

            if identity_type == "descriptive_character":
                name = str(raw.get("name", "") or "").strip()
                aliases = [
                    str(alias or "").strip()
                    for alias in (raw.get("aliases", []) or [])
                    if str(alias or "").strip()
                ]
                candidate_surfaces = [name, *aliases]
                canonical = next(
                    (surface for surface in candidate_surfaces
                     if surface and cls._descriptive_identity_is_grounded(story, surface)),
                    None,
                )
                if not canonical:
                    continue
                safe_aliases = []
                for alias in aliases:
                    if EntityResolver.is_safe_semantic_reference(alias) or EntityResolver.generic_role_surface(alias):
                        safe_aliases.append(alias)
                result[EntityResolver.normalize(canonical)] = {
                    "identity_type": "descriptive_character",
                    "relationship_to": None,
                    "relationship": None,
                    "semantic_aliases": list(dict.fromkeys(safe_aliases)),
                }
                continue

            entity_type = str(raw.get("entity_type", "") or "").strip().upper()
            name = str(raw.get("name", "") or "").strip()
            aliases = [
                str(alias or "").strip()
                for alias in (raw.get("aliases", []) or [])
                if str(alias or "").strip()
            ]
            if (
                entity_type not in {"PERSON", "CHARACTER", "SENTIENT"}
                or not name
                or not cls._semantic_named_surface_is_safe(name)
                or name.lower() in cls.GENERIC_PERSON_LABELS
                or name.lower() in cls.RELATIONSHIP_TERMS
                or not any(cls._story_has_character_name(story, surface) for surface in [name, *aliases])
            ):
                continue
            canonical = re.sub(
                r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|commander|detective|agent)\.?\s+",
                "",
                name,
                count=1,
                flags=re.IGNORECASE,
            ).strip() or name
            filtered_aliases = []
            for alias in aliases:
                if EntityResolver.generic_role_surface(alias):
                    continue
                if EntityResolver.is_safe_semantic_reference(alias):
                    filtered_aliases.append(alias)
            result[EntityResolver.normalize(canonical)] = {
                "identity_type": "named_character",
                "relationship_to": None,
                "relationship": None,
                "semantic_aliases": list(dict.fromkeys(filtered_aliases)),
            }

        # Descriptive hints are metadata-only fallback support. They may be
        # consumed when no semantic result exists, but they must never override
        # a Qwen-declared identity type in the creative path.
        if semantic_result is None:
            for hint in descriptive_hints or []:
                canonical = str(hint or "").strip()
                if not canonical or not cls._descriptive_identity_is_grounded(story, canonical):
                    continue
                key = EntityResolver.normalize(canonical)
                result.setdefault(key, {
                    "identity_type": "descriptive_character",
                    "relationship_to": None,
                    "relationship": None,
                    "semantic_aliases": [],
                })

        for hint in relational_hints or []:
            if not hint.get("strong"):
                continue
            canonical = str(hint.get("name", "") or "").strip()
            if not canonical:
                continue
            key = EntityResolver.normalize(canonical)
            existing = result.get(key)
            if not existing or existing.get("identity_type") != "relational_character":
                result[key] = {
                    "identity_type": "relational_character",
                    "relationship_to": str(hint.get("relationship_to", "") or "").strip() or None,
                    "relationship": str(hint.get("relationship", "") or "").strip() or None,
                    "semantic_aliases": list(dict.fromkeys(hint.get("aliases", []) or [])),
                }
            else:
                result[key]["semantic_aliases"] = list(dict.fromkeys(
                    (result[key].get("semantic_aliases", []) or []) + list(hint.get("aliases", []) or [])
                ))
        return result

    @classmethod
    def _reconcile_semantic_characters(
        cls,
        story: str,
        semantic_result,
    ) -> list[str]:
        """Validate Qwen's roster without adding or protecting characters.

        Qwen owns character identity, count, aliases, and identity type. The
        planner only applies bounded safety checks: grounded-in-story evidence,
        mention-only filtering, obvious non-person surfaces, and relationship
        ownership against the *Qwen-approved* named roster. Deterministic
        detection is intentionally not a cast floor.
        """
        if not isinstance(semantic_result, dict):
            return []

        candidates = list(semantic_result.get("candidates", []) or [])
        if not candidates:
            candidates = list(semantic_result.get("characters", []) or [])

        def candidate_obj(raw):
            if isinstance(raw, str) and raw.strip():
                return {
                    "name": raw.strip(),
                    "entity_type": "CHARACTER",
                    "is_character": True,
                    "aliases": [],
                    "identity_type": "named_character",
                }
            return raw if isinstance(raw, dict) else None

        def safe_non_person(name: str) -> bool:
            normalized = EntityResolver.normalize(name)
            if normalized in cls.QWEN_NON_PERSON_TOKENS:
                return True
            tokens = normalized.split()
            if not tokens:
                return True
            if tokens[-1] in cls.NON_PERSON_HEAD_WORDS:
                return True
            if tokens[0] in cls.NON_PERSON_PREFIX_WORDS:
                return True
            return False

        named_candidates: list[tuple[dict, str]] = []
        descriptive_candidates: list[tuple[dict, str]] = []
        relational_candidates: list[dict] = []

        # Pass 1: accept only Qwen-declared named/descriptive identities.
        for raw_value in candidates:
            raw = candidate_obj(raw_value)
            if not raw or not bool(raw.get("is_character", False)):
                continue
            identity_type = str(raw.get("identity_type", "named_character") or "named_character").strip().lower()
            entity_type = str(raw.get("entity_type", "") or "").strip().upper()
            name = str(raw.get("name", "") or "").strip()
            aliases = [
                str(alias or "").strip()
                for alias in (raw.get("aliases", []) or [])
                if str(alias or "").strip()
            ]

            if identity_type == "relational_character":
                relational_candidates.append(raw)
                continue

            if identity_type == "descriptive_character":
                surfaces = [name, *aliases]
                canonical = next(
                    (surface for surface in surfaces
                     if surface
                     and not safe_non_person(surface)
                     and cls._descriptive_identity_is_grounded(story, surface)),
                    None,
                )
                if canonical and not cls._is_mention_only_identity(story, canonical):
                    descriptive_candidates.append((raw, canonical))
                continue

            if entity_type not in {"PERSON", "CHARACTER", "SENTIENT"} or not name:
                continue
            surfaces = [name, *aliases]
            grounded_surface = next(
                (surface for surface in surfaces
                 if surface
                 and not safe_non_person(surface)
                 and cls._semantic_named_surface_is_safe(surface)
                 and cls._story_has_character_name(story, surface)),
                None,
            )
            if not grounded_surface:
                continue
            canonical = re.sub(
                r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|commander|detective|agent)\.?\s+",
                "",
                name,
                count=1,
                flags=re.IGNORECASE,
            ).strip() or name
            if safe_non_person(canonical) or cls._is_mention_only_identity(story, canonical):
                continue
            named_candidates.append((raw, canonical))

        # Deduplicate named identities while preserving Qwen's order.
        named_names: list[str] = []
        seen_named: set[str] = set()
        for _raw, name in named_candidates:
            key = EntityResolver.normalize(name)
            if key and key not in seen_named:
                seen_named.add(key)
                named_names.append(name)

        # Pass 2: relational identities can only attach to a Qwen-approved named
        # character. Deterministic names cannot become relationship owners.
        normalized: list[str] = list(named_names)
        seen: set[str] = {EntityResolver.normalize(name) for name in normalized}
        for raw in relational_candidates:
            valid, canonical, _metadata = cls._relational_candidate_is_grounded(
                story, raw, named_names,
            )
            if not valid or not canonical:
                continue
            if cls._is_mention_only_identity(story, canonical):
                continue
            if EntityResolver.normalize(canonical) not in seen:
                normalized.append(canonical)
                seen.add(EntityResolver.normalize(canonical))

        for _raw, canonical in descriptive_candidates:
            key = EntityResolver.normalize(canonical)
            if key and key not in seen:
                normalized.append(canonical)
                seen.add(key)

        return cls._canonicalize_character_descriptors(normalized)

    def create_characters(
        self,
        story: str,
        *,
        qwen_character_extractor=None,
        qwen_character_adjudicator=None,
    ) -> list[Character]:
        """Create the canonical production roster.

        When the Qwen extractor is supplied, Qwen is the sole semantic authority
        for cast membership, count, aliases, and identity type. Deterministic
        detection is used only as extraction hints and bounded safety evidence;
        it never adds a missing Qwen character or overrides ``is_character=false``.

        ``qwen_character_adjudicator`` is a deprecated compatibility parameter.
        It is intentionally ignored in the creative path: one semantic Qwen call
        is the contract. The deterministic fallback remains available only when
        no extractor is supplied.
        """
        self._semantic_character_metadata_cache = {}
        story = self._clean_text(story)
        if not story:
            return []

        hints = self._canonicalize_character_descriptors(
            [
                *self.detect_character_descriptors(story),
                *self._explicit_source_character_names(story),
            ]
        )
        hints = self._drop_mention_only_identities(story, hints, preserve_empty=True)

        if qwen_character_extractor is not None:
            semantic_result = qwen_character_extractor(story, hints[:32])
            reconciled = self._reconcile_semantic_characters(story, semantic_result)
            if not reconciled:
                raise RuntimeError(
                    "Qwen character extraction produced no valid production characters after bounded planner validation."
                )
            descriptors = reconciled
            semantic_result_final = semantic_result
        else:
            # Non-Qwen fallback for disabled/legacy callers only. This path may use
            # deterministic evidence because there is no semantic model to defer to.
            descriptors = self._drop_mention_only_identities(story, hints, preserve_empty=False)
            if not descriptors:
                return []
            semantic_result_final = None

        metadata = self._semantic_character_metadata(
            story,
            descriptors,
            semantic_result_final,
            relational_hints=[],
            descriptive_hints=[
                name for name in descriptors
                if self._descriptive_identity_is_grounded(story, name)
            ],
        )
        self._semantic_character_metadata_cache = metadata

        characters = []
        for index, descriptor in enumerate(descriptors, start=1):
            character = self._make_character(descriptor, index, story)
            info = metadata.get(EntityResolver.normalize(descriptor))
            if info:
                identity_type = str(info.get("identity_type", "named_character") or "named_character").strip().lower()
                character.identity_type = identity_type if identity_type in {
                    "named_character", "relational_character", "descriptive_character"
                } else "named_character"
                character.relationship_to = str(info.get("relationship_to", "") or "").strip() or None
                character.relationship = str(info.get("relationship", "") or "").strip() or None
                character.semantic_aliases = list(dict.fromkeys(
                    str(value).strip() for value in (info.get("semantic_aliases", []) or []) if str(value).strip()
                ))
            if character.identity_type == "relational_character" and (
                not character.relationship_to or not character.relationship
            ):
                character.identity_type = "named_character"
                character.relationship_to = None
                character.relationship = None
                character.semantic_aliases = []
            if character.identity_type == "descriptive_character" and not self._descriptive_identity_is_grounded(
                story, descriptor
            ):
                raise RuntimeError(
                    f"Qwen descriptive character failed grounding validation: {descriptor}"
                )
            character.build_identity_profile()
            characters.append(character)

        self.references.resolve_characters(characters)
        self.references.validate(characters, require_images=False)
        return characters

    # ============================================================
    # SCENE EXTRACTION
    # ============================================================

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
        aliases = EntityResolver.build_character_alias_map(
            characters
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

            # Physical scene location is semantic metadata. The deterministic
            # planner must not infer it from arbitrary prepositional phrases.
            # The existing Qwen cinematography pass supplies this field.
            location = ""

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
