from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.retake_manager import RetakeManager
from execution.retake_executor import RetakeExecutor


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120.0)
    if result.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + result.stderr[-4000:])


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    return json.loads(result.stdout)


def main() -> None:
    assert RetakeExecutor.__name__ == "RetakeExecutor"
    with tempfile.TemporaryDirectory(prefix="h3-media-contract-") as tmp:
        root = Path(tmp)
        base = root / "base.mp4"
        retake = root / "retake.mp4"
        out = root / "out.mp4"

        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000",
            "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "32000", "-ac", "2", "-shortest", str(base),
        ])
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100",
            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-ac", "1", "-shortest", str(retake),
        ])

        RetakeManager(root).stitch(
            base,
            retake,
            out,
            start_seconds=1.0,
            end_seconds=5.0,
            preserve_audio=True,
        )

        base_probe = probe(base)
        out_probe = probe(out)
        base_video = next(s for s in base_probe["streams"] if s["codec_type"] == "video")
        out_video = next(s for s in out_probe["streams"] if s["codec_type"] == "video")
        out_audio = next((s for s in out_probe["streams"] if s["codec_type"] == "audio"), None)

        assert abs(float(out_video["duration"]) - float(base_video["duration"])) <= 1 / 24 + 0.01
        assert out_audio is not None
        assert out_audio.get("sample_rate") == "32000"
        assert int(out_audio.get("channels") or 0) == 2

        out_segmented = root / "out_segmented.mp4"
        RetakeManager(root).stitch(
            base, retake, out_segmented,
            start_seconds=1.0, end_seconds=5.0, preserve_audio=False,
        )
        segmented_probe = probe(out_segmented)
        segmented_video = next(s for s in segmented_probe["streams"] if s["codec_type"] == "video")
        segmented_audio = next((s for s in segmented_probe["streams"] if s["codec_type"] == "audio"), None)
        assert abs(float(segmented_video["duration"]) - float(base_video["duration"])) <= 1 / 24 + 0.01
        assert segmented_audio is not None
        assert segmented_audio.get("sample_rate") == "32000"
        assert int(segmented_audio.get("channels") or 0) == 2
        assert abs(float(segmented_audio.get("duration") or 0.0) - float(base_video["duration"])) <= 0.10

    print("PASS media/retake contracts")


if __name__ == "__main__":
    main()
