from __future__ import annotations

import subprocess
from pathlib import Path


class AssemblyManager:

    def __init__(
        self,
        output_dir: Path,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def check_ffmpeg() -> None:

        result = subprocess.run(
            [
                "ffmpeg",
                "-version",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg is not installed."
            )

    def assemble(
        self,
        videos: list[Path],
        final_name: str = "final_video.mp4",
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> Path:

        self.check_ffmpeg()

        if not videos:
            raise ValueError(
                "No videos supplied."
            )

        if (
            width <= 0
            or height <= 0
            or fps <= 0
        ):
            raise ValueError(
                "Invalid final delivery parameters."
            )

        concat_file = (
            self.output_dir
            / "concat.txt"
        )

        destination = (
            self.output_dir
            / final_name
        )

        temp = (
            self.output_dir
            / ".final.tmp.mp4"
        )

        lines: list[str] = []

        for video in videos:

            path = Path(
                video
            ).resolve()

            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            escaped = str(
                path
            ).replace(
                "'",
                "'\\''",
            )

            lines.append(
                f"file '{escaped}'"
            )

        concat_file.write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),

            "-vf",
            (
                f"fps={fps},"
                f"scale={width}:{height}:"
                "force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:"
                f"({width}-iw)/2:"
                f"({height}-ih)/2"
            ),

            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",
            "-b:a",
            "192k",

            "-movflags",
            "+faststart",

            str(temp),
        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "FFmpeg assembly failed:\n"
                    + result.stderr[-5000:]
                )

            temp.replace(
                destination
            )

            return destination

        finally:

            concat_file.unlink(
                missing_ok=True
            )

            if (
                temp.exists()
                and not destination.exists()
            ):
                temp.unlink(
                    missing_ok=True
                )
