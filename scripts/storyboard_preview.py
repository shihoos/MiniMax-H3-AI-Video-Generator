from __future__ import annotations

import argparse
from pathlib import Path

from ui.storyboard_server import (
    serve_storyboard,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 interactive storyboard preview"
        )
    )

    parser.add_argument(
        "--plan",
        default=str(
            ROOT
            / "data"
            / "production"
            / "story_preview.json"
        ),
        help=(
            "Production plan JSON."
        ),
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help=(
            "Wait until the storyboard is approved."
        ),
    )

    args = parser.parse_args()

    result = serve_storyboard(
        Path(
            args.plan
        ),
        host=args.host,
        port=args.port,
        wait_for_approval=args.wait,
    )

    print(
        "Storyboard result:",
        result,
    )


if __name__ == "__main__":
    main()
