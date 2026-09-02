from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


class ComfyPreviewStreamer:
    """Best-effort ComfyUI WebSocket progress/preview streamer.

    The renderer does not depend on this feature. When websocket-client is
    present, ComfyUI's binary preview frames and progress events are mirrored
    to a small session directory for the Gradio UI. A disconnect never fails a
    production render.
    """

    def __init__(self, base_url: str, output_dir: Path, callback: Callable[[dict[str, Any]], None] | None = None):
        self.base_url = str(base_url).strip().rstrip("/")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _ws_url(base_url: str, client_id: str) -> str:
        parsed = urlparse(base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/ws?clientId={client_id}"

    @staticmethod
    def _extract_image(data: bytes) -> bytes | None:
        jpeg = data.find(b"\xff\xd8\xff")
        png = data.find(b"\x89PNG\r\n\x1a\n")
        offsets = [offset for offset in (jpeg, png) if offset >= 0]
        if not offsets:
            return None
        payload = data[min(offsets):]
        return payload

    def start(self, prompt_id: str, client_id: str) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(str(prompt_id), str(client_id)),
            name="h3-comfy-preview",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _emit(self, payload: dict[str, Any]) -> None:
        try:
            self.callback(payload) if self.callback else None
        except Exception:
            pass
        try:
            target = self.output_dir / "progress.json"
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)

            preview_path = payload.get("preview_path")
            if preview_path:
                pointer = self.output_dir.parent / "current_preview.json"
                pointer_tmp = pointer.with_name(pointer.name + ".tmp")
                pointer_tmp.write_text(
                    json.dumps({"preview_path": str(preview_path)}, ensure_ascii=False),
                    encoding="utf-8",
                )
                pointer_tmp.replace(pointer)
        except Exception:
            pass

    def _run(self, prompt_id: str, client_id: str) -> None:
        try:
            import websocket
        except Exception as exc:
            self._emit({"status": "unavailable", "reason": f"websocket-client unavailable: {exc}"})
            return
        ws = None
        try:
            ws = websocket.create_connection(self._ws_url(self.base_url, client_id), timeout=3)
            ws.settimeout(2.0)
            self._emit({"status": "connected", "prompt_id": prompt_id, "progress": 0.0})
            while not self._stop.is_set():
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except websocket.WebSocketConnectionClosedException:
                    self._emit({"status": "disconnected", "prompt_id": prompt_id})
                    break
                except Exception as recv_error:
                    self._emit({"status": "warning", "prompt_id": prompt_id, "reason": str(recv_error)})
                    break
                if isinstance(message, str):
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") not in {"progress", "executing", "execution_start", "executed"}:
                        continue
                    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                    if payload.get("type") == "progress":
                        value = float(data.get("value", 0) or 0)
                        maximum = float(data.get("max", 1) or 1)
                        self._emit({"status": "sampling", "prompt_id": prompt_id, "progress": max(0.0, min(1.0, value / maximum)), "node": data.get("node")})
                    elif payload.get("type") == "executing":
                        self._emit({"status": "executing", "prompt_id": prompt_id, "node": data.get("node")})
                    continue
                if isinstance(message, (bytes, bytearray)):
                    image = self._extract_image(bytes(message))
                    if image:
                        suffix = ".jpg" if image.startswith(b"\xff\xd8\xff") else ".png"
                        path = self.output_dir / f"latest{suffix}"
                        temporary = path.with_name(path.name + ".tmp")
                        temporary.write_bytes(image)
                        temporary.replace(path)
                        self._emit({"status": "sampling", "prompt_id": prompt_id, "preview_path": str(path), "progress": None})
        except Exception as exc:
            self._emit({"status": "warning", "prompt_id": prompt_id, "reason": str(exc)})
        finally:
            try:
                if ws:
                    ws.close()
            except Exception:
                pass
