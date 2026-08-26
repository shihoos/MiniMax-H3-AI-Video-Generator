from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from datetime import datetime
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from urllib.parse import (
    parse_qs,
    urlparse,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

STATIC_HTML = (
    ROOT
    / "ui"
    / "storyboard.html"
)

DEFAULT_PORT = 8765


def _safe_path(
    value: str,
) -> Path:

    candidate = Path(
        value
    )

    if not candidate.is_absolute():
        candidate = (
            ROOT
            / candidate
        )

    candidate = candidate.resolve()

    allowed_roots = [
        (
            ROOT
            / "assets"
        ).resolve(),
        (
            ROOT
            / "data"
        ).resolve(),
        (
            ROOT
            / "ComfyUI"
            / "input"
        ).resolve(),
    ]

    if candidate == ROOT.resolve():
        return candidate

    for allowed in allowed_roots:

        if (
            candidate == allowed
            or allowed in candidate.parents
        ):
            return candidate

    raise PermissionError(
        "Media path is outside allowed project directories."
    )


class StoryboardServer:

    def __init__(
        self,
        plan_path: Path,
        host: str,
        port: int,
    ):

        self.plan_path = Path(
            plan_path
        ).resolve()

        self.approved_path = (
            self.plan_path.parent
            / "story_preview_approved.json"
        )

        self.host = host
        self.port = int(
            port
        )

        self.server = None

        self.approved_event = (
            threading.Event()
        )

    def load_plan(self) -> dict:

        if not self.plan_path.is_file():
            raise FileNotFoundError(
                self.plan_path
            )

        data = json.loads(
            self.plan_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "Storyboard plan must be a JSON object."
            )

        data.setdefault(
            "approval",
            {
                "status": "draft",
                "approved_at": None,
            },
        )

        return data

    def save_plan(
        self,
        data: dict,
    ) -> None:

        data["updated_at"] = (
            datetime.now().isoformat()
        )

        self.plan_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def approve(
        self,
        data: dict,
    ) -> None:

        data[
            "approval"
        ] = {
            "status": "approved",
            "approved_at": (
                datetime.now().isoformat()
            ),
        }

        self.save_plan(
            data
        )

        self.approved_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.approved_event.set()

    def make_handler(self):

        outer = self

        class Handler(
            BaseHTTPRequestHandler
        ):

            def _send_json(
                self,
                payload,
                status=200,
            ):

                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                ).encode(
                    "utf-8"
                )

                self.send_response(
                    status
                )

                self.send_header(
                    "Content-Type",
                    "application/json; charset=utf-8",
                )

                self.send_header(
                    "Content-Length",
                    str(
                        len(encoded)
                    ),
                )

                self.send_header(
                    "Cache-Control",
                    "no-store",
                )

                self.end_headers()

                self.wfile.write(
                    encoded
                )

            def _send_bytes(
                self,
                payload,
                content_type,
            ):

                self.send_response(
                    200
                )

                self.send_header(
                    "Content-Type",
                    content_type
                )

                self.send_header(
                    "Content-Length",
                    str(
                        len(payload)
                    ),
                )

                self.end_headers()

                self.wfile.write(
                    payload
                )

            def do_GET(self):

                parsed = urlparse(
                    self.path
                )

                if parsed.path == "/":
                    self._send_bytes(
                        STATIC_HTML.read_bytes(),
                        "text/html; charset=utf-8",
                    )
                    return

                if parsed.path == "/api/plan":
                    self._send_json(
                        outer.load_plan()
                    )
                    return

                if parsed.path == "/api/status":
                    plan = (
                        outer.load_plan()
                    )

                    self._send_json(
                        plan.get(
                            "approval",
                            {
                                "status":
                                    "draft"
                            },
                        )
                    )
                    return

                if parsed.path == "/media":

                    query = parse_qs(
                        parsed.query
                    )

                    values = query.get(
                        "path",
                        [],
                    )

                    if not values:
                        self._send_json(
                            {
                                "error":
                                    "missing path"
                            },
                            400,
                        )
                        return

                    try:
                        path = _safe_path(
                            values[0]
                        )

                        if (
                            not path.is_file()
                        ):
                            self._send_json(
                                {
                                    "error":
                                        "file not found"
                                },
                                404,
                            )
                            return

                        content_type = (
                            mimetypes.guess_type(
                                path.name
                            )[0]
                            or "application/octet-stream"
                        )

                        self._send_bytes(
                            path.read_bytes(),
                            content_type,
                        )

                    except (
                        OSError,
                        PermissionError,
                    ) as exc:

                        self._send_json(
                            {
                                "error":
                                    str(exc)
                            },
                            403,
                        )

                    return

                self._send_json(
                    {
                        "error":
                            "not found"
                    },
                    404,
                )

            def do_POST(self):

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )

                body = self.rfile.read(
                    length
                )

                try:
                    payload = json.loads(
                        body.decode(
                            "utf-8"
                        )
                    )

                except Exception:
                    self._send_json(
                        {
                            "error":
                                "invalid JSON"
                        },
                        400,
                    )
                    return

                if self.path == "/api/save":

                    if not isinstance(
                        payload,
                        dict,
                    ):
                        self._send_json(
                            {
                                "error":
                                    "plan must be an object"
                            },
                            400,
                        )
                        return

                    outer.save_plan(
                        payload
                    )

                    self._send_json(
                        {
                            "status":
                                "saved"
                        }
                    )

                    return

                if self.path == "/api/approve":

                    if not isinstance(
                        payload,
                        dict,
                    ):
                        self._send_json(
                            {
                                "error":
                                    "plan must be an object"
                            },
                            400,
                        )
                        return

                    outer.approve(
                        payload
                    )

                    self._send_json(
                        {
                            "status":
                                "approved",
                            "approved_plan":
                                str(
                                    outer.approved_path
                                ),
                        }
                    )

                    return

                self._send_json(
                    {
                        "error":
                            "not found"
                    },
                    404,
                )

            def log_message(
                self,
                format,
                *args,
            ):
                print(
                    "[STORYBOARD]",
                    format % args,
                )

        return Handler

    def start(
        self,
        wait_for_approval: bool = False,
    ) -> Path:

        if not STATIC_HTML.is_file():
            raise FileNotFoundError(
                STATIC_HTML
            )

        self.server = (
            ThreadingHTTPServer(
                (
                    self.host,
                    self.port,
                ),
                self.make_handler(),
            )
        )

        thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

        thread.start()

        print(
            "=" * 80
        )

        print(
            "MINIMAX H3 STORYBOARD UI"
        )

        print(
            "=" * 80
        )

        print(
            f"URL: http://127.0.0.1:{self.port}/"
        )

        print(
            f"Plan: {self.plan_path}"
        )

        print(
            "Edit and review the storyboard in the browser."
        )

        if not wait_for_approval:
            return self.plan_path

        print(
            "Waiting for storyboard approval..."
        )

        try:

            while not self.approved_event.wait(
                timeout=1
            ):

                if (
                    self.approved_path.is_file()
                ):
                    self.approved_event.set()

            return self.approved_path

        finally:

            self.server.shutdown()
            self.server.server_close()


def serve_storyboard(
    plan_path: Path,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    wait_for_approval: bool = False,
) -> Path:

    server = StoryboardServer(
        plan_path=plan_path,
        host=host,
        port=port,
    )

    return server.start(
        wait_for_approval=
            wait_for_approval
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 interactive storyboard preview"
        )
    )

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )

    parser.add_argument(
        "--wait",
        action="store_true",
    )

    args = parser.parse_args()

    result = serve_storyboard(
        Path(args.plan),
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
