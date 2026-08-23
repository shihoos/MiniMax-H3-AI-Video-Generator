from __future__ import annotations

import subprocess
from pathlib import Path


class H3ReferenceBinding:

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3

    @staticmethod
    def video_has_audio(
        path: str,
    ) -> bool:

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )

            return bool(
                result.stdout.strip()
            )

        except (
            OSError,
            subprocess.SubprocessError,
        ):
            # Do not break reference compilation if ffprobe is
            # unavailable. H3 itself will determine whether the
            # video contains an audio stream.
            return False

    @classmethod
    def collect(
        cls,
        characters: list,
        shot_character_names: list[str],
    ) -> dict:

        lookup = {
            character.name.lower():
                character
            for character in characters
        }

        images = []
        videos = []
        audio = []

        image_owners = {}
        video_owners = {}
        audio_owners = {}

        for name in shot_character_names:

            character = lookup.get(
                str(name).lower()
            )

            if character is None:
                continue

            for path in (
                character
                .normalized_reference_paths()
            ):
                if (
                    path not in images
                    and len(images) < cls.MAX_IMAGES
                ):
                    images.append(path)

                image_owners.setdefault(
                    path,
                    [],
                ).append(
                    character.name
                )

            for path in (
                character
                .normalized_video_paths()
            ):
                if (
                    path not in videos
                    and len(videos) < cls.MAX_VIDEOS
                ):
                    videos.append(path)

                video_owners.setdefault(
                    path,
                    [],
                ).append(
                    character.name
                )

            for path in (
                character
                .normalized_audio_paths()
            ):
                if (
                    path not in audio
                    and len(audio) < cls.MAX_AUDIO
                ):
                    audio.append(path)

                audio_owners.setdefault(
                    path,
                    [],
                ).append(
                    character.name
                )

        paired_video_audio_count = sum(
            1
            for video in videos
            if cls.video_has_audio(video)
        )

        picture_lines = [
            (
                f"<Picture {index}> is the canonical "
                f"visual identity reference for "
                f"{', '.join(image_owners.get(path, []))}. "
                "Use it for face, hair, hairline, body "
                "structure, body proportions and stable "
                "identity features."
            )
            for index, path in enumerate(
                images,
                start=1,
            )
        ]

        video_lines = [
            (
                f"<Video {index}> is the motion/reference "
                f"video for "
                f"{', '.join(video_owners.get(path, []))}."
            )
            for index, path in enumerate(
                videos,
                start=1,
            )
        ]

        # H3 places paired video soundtracks before standalone
        # reference audio in its audio ordinal sequence.
        audio_start = (
            paired_video_audio_count + 1
        )

        audio_lines = [
            (
                f"<Audio {audio_start + index}> is the "
                f"standalone voice/audio identity reference "
                f"for "
                f"{', '.join(audio_owners.get(path, []))}."
            )
            for index, path in enumerate(
                audio
            )
        ]

        return {
            "images": images,
            "videos": videos,
            "audio": audio,
            "paired_video_audio_count": (
                paired_video_audio_count
            ),
            "prompt_lines": (
                picture_lines
                + video_lines
                + audio_lines
            ),
        }

    @classmethod
    def prompt_block(
        cls,
        characters: list,
        shot_character_names: list[str],
    ):

        data = cls.collect(
            characters=characters,
            shot_character_names=shot_character_names,
        )

        return (
            "\n".join(
                data["prompt_lines"]
            ),
            data,
        )
