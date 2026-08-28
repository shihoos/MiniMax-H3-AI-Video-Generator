from __future__ import annotations

import re
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
}


class ReferenceManager:

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3
    MAX_TOTAL_FILES = 12

    def __init__(
        self,
        project_root,
    ):
        self.project_root = Path(
            project_root
        )

        self.assets = (
            self.project_root
            / "assets"
        )

        self.characters_dir = (
            self.assets
            / "characters"
        )

        self.references_dir = (
            self.assets
            / "references"
        )

        self.audio_dir = (
            self.assets
            / "audio"
        )

        self.video_dir = (
            self.assets
            / "videos"
        )

        for directory in (
            self.characters_dir,
            self.references_dir,
            self.audio_dir,
            self.video_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _key(
        value,
    ) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "_",
            str(value)
            .strip()
            .lower(),
        ).strip("_")

    @staticmethod
    def _files(
        directory,
        extensions,
    ):
        directory = Path(
            directory
        )

        if not directory.is_dir():
            return []

        return sorted(
            [
                path
                for path in directory.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in extensions
                )
            ],
            key=lambda path:
                path.name.lower(),
        )

    def character_asset_names(
        self,
    ) -> list[str]:

        names = set()

        for path in self._files(
            self.characters_dir,
            IMAGE_EXTENSIONS,
        ):
            names.add(
                path.stem
            )

        for path in self._files(
            self.references_dir,
            IMAGE_EXTENSIONS
            | VIDEO_EXTENSIONS,
        ):
            names.add(
                path.stem
            )

        if self.references_dir.is_dir():

            for directory in (
                self.references_dir.iterdir()
            ):
                if directory.is_dir():
                    names.add(
                        directory.name
                    )

        return sorted(
            names,
            key=lambda value:
                self._key(value),
        )

    def resolve_character(
        self,
        character_name,
    ):

        key = self._key(
            character_name
        )

        images = []
        videos = []
        audio = []

        root = (
            self.references_dir
            / key
        )

        images.extend(
            self._files(
                root / "images",
                IMAGE_EXTENSIONS,
            )
        )

        videos.extend(
            self._files(
                root / "videos",
                VIDEO_EXTENSIONS,
            )
        )

        audio.extend(
            self._files(
                root / "audio",
                AUDIO_EXTENSIONS,
            )
        )

        for path in self._files(
            self.characters_dir,
            IMAGE_EXTENSIONS,
        ):
            if (
                self._key(path.stem)
                == key
            ):
                if path not in images:
                    images.append(
                        path
                    )

        for path in self._files(
            self.references_dir,
            IMAGE_EXTENSIONS
            | VIDEO_EXTENSIONS,
        ):
            if (
                self._key(path.stem)
                != key
            ):
                continue

            if (
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                if path not in images:
                    images.append(
                        path
                    )

            else:
                if path not in videos:
                    videos.append(
                        path
                    )

        for path in self._files(
            self.audio_dir,
            AUDIO_EXTENSIONS,
        ):
            if (
                self._key(path.stem)
                == key
            ):
                if path not in audio:
                    audio.append(
                        path
                    )

        for path in self._files(
            self.video_dir,
            VIDEO_EXTENSIONS,
        ):
            if (
                self._key(path.stem)
                == key
            ):
                if path not in videos:
                    videos.append(
                        path
                    )

        return {
            "reference_paths": [
                str(path)
                for path in images[
                    : self.MAX_IMAGES
                ]
            ],
            "reference_video_paths": [
                str(path)
                for path in videos[
                    : self.MAX_VIDEOS
                ]
            ],
            "reference_audio_paths": [
                str(path)
                for path in audio[
                    : self.MAX_AUDIO
                ]
            ],
        }

    def get_character_source(
        self,
        character_name,
    ):

        resolved = (
            self.resolve_character(
                character_name
            )
        )

        images = resolved[
            "reference_paths"
        ]

        videos = resolved[
            "reference_video_paths"
        ]

        audio = resolved[
            "reference_audio_paths"
        ]

        return {
            "mode": (
                "provided"
                if images
                or videos
                or audio
                else "missing"
            ),
            "path": (
                images[0]
                if images
                else None
            ),
            "reference_paths": images,
            "reference_video_paths": videos,
            "reference_audio_paths": audio,
            "reference_video_path": (
                videos[0]
                if videos
                else None
            ),
            "reference_audio_path": (
                audio[0]
                if audio
                else None
            ),
        }

    def resolve_characters(
        self,
        characters,
    ):

        for character in characters:

            source = (
                self.resolve_character(
                    character.name
                )
            )

            character.reference_paths = (
                source[
                    "reference_paths"
                ]
            )

            character.reference_video_paths = (
                source[
                    "reference_video_paths"
                ]
            )

            character.reference_audio_paths = (
                source[
                    "reference_audio_paths"
                ]
            )

            character.reference_path = (
                character.reference_paths[0]
                if character.reference_paths
                else None
            )

            character.reference_video_path = (
                character.reference_video_paths[0]
                if character.reference_video_paths
                else None
            )

            character.reference_audio_path = (
                character.reference_audio_paths[0]
                if character.reference_audio_paths
                else None
            )

            character.reference_mode = (
                "provided"
                if (
                    character.reference_paths
                    or character.reference_video_paths
                    or character.reference_audio_paths
                )
                else "story_generated"
            )

            character.build_identity_profile()

        return characters

    def validate(
        self,
        characters,
        require_images: bool = False,
    ) -> bool:

        errors = []

        for character in characters:

            images = (
                character.normalized_reference_paths()
            )

            videos = (
                character.normalized_video_paths()
            )

            audio = (
                character.normalized_audio_paths()
            )

            if (
                require_images
                and not images
            ):
                errors.append(
                    f"{character.name}: "
                    "reference image required"
                )

            if len(images) > self.MAX_IMAGES:
                errors.append(
                    f"{character.name}: "
                    "too many images"
                )

            if len(videos) > self.MAX_VIDEOS:
                errors.append(
                    f"{character.name}: "
                    "too many videos"
                )

            if len(audio) > self.MAX_AUDIO:
                errors.append(
                    f"{character.name}: "
                    "too many audio files"
                )

            total = (
                len(images)
                + len(videos)
                + len(audio)
            )

            if total > self.MAX_TOTAL_FILES:
                errors.append(
                    f"{character.name}: "
                    f"{total} reference files; "
                    f"maximum is "
                    f"{self.MAX_TOTAL_FILES}"
                )

            for path in (
                images
                + videos
                + audio
            ):
                if not Path(
                    path
                ).is_file():
                    errors.append(
                        f"{character.name}: "
                        f"missing media {path}"
                    )

        if errors:
            raise RuntimeError(
                "Reference validation failed:\n- "
                + "\n- ".join(errors)
            )

        return True
