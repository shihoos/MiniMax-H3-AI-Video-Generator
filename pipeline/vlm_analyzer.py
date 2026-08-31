from __future__ import annotations

import base64
import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from planner.config import H3_VLM_ENABLED


LOGGER = logging.getLogger(__name__)


class VLMAnalyzer:
    """Optional OpenAI-compatible local/remote VLM adapter.

    It is disabled unless H3_VLM_ENABLED=1 and an endpoint is configured.
    The adapter is intentionally schema-constrained and never owns production
    identity or execution decisions.
    """

    def __init__(self):
        self.enabled = self._bool(os.getenv("H3_VLM_ENABLED", "1" if H3_VLM_ENABLED else "0"))
        self.endpoint = os.getenv("H3_VLM_ENDPOINT", "").strip().rstrip("/")
        self.model = os.getenv("H3_VLM_MODEL", "").strip()
        self.api_key = os.getenv("H3_VLM_API_KEY", "").strip()
        self.timeout = max(5.0, float(os.getenv("H3_VLM_TIMEOUT", "90")))
        self.max_image_bytes = max(1_000_000, int(os.getenv("H3_VLM_MAX_IMAGE_BYTES", str(12 * 1024 * 1024))))
        cache_value = os.getenv("H3_VLM_CACHE_DIR", "").strip()
        self.cache_dir = Path(cache_value).expanduser().resolve() if cache_value else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_namespace = "minimax-h3-vlm-v1"
        # The feature may be enabled in the production manifest even when the
        # optional endpoint is not configured yet. In that case the pipeline
        # remains usable and simply reports VLM as unavailable.
        self.configuration_warning = ""
        if self.enabled and (not self.endpoint or not self.model):
            self.configuration_warning = (
                "VLM is enabled but H3_VLM_ENDPOINT/H3_VLM_MODEL are not configured; "
                "reference analysis and semantic QA will remain inactive."
            )
            LOGGER.warning(
                "[VLM] %s",
                self.configuration_warning,
            )
        elif self.enabled and not self._looks_like_http_endpoint(self.endpoint):
            self.configuration_warning = (
                "VLM is enabled but H3_VLM_ENDPOINT does not look like an HTTP(S) URL; "
                "reference analysis and semantic QA may fail until the endpoint is corrected."
            )
            LOGGER.warning(
                "[VLM] %s endpoint=%r",
                self.configuration_warning,
                self.endpoint,
            )

    @staticmethod
    def _bool(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _looks_like_http_endpoint(value: str) -> bool:
        endpoint = str(value or "").strip().lower()
        return endpoint.startswith("http://") or endpoint.startswith("https://")

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.endpoint and self.model)

    @staticmethod
    def _mime(path: Path) -> str:
        mapping = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        return mapping.get(path.suffix.lower(), "application/octet-stream")

    def _encode_image(self, path: Path) -> str:
        path = Path(path).resolve()
        data = path.read_bytes()
        if len(data) > self.max_image_bytes:
            raise ValueError(f"VLM image exceeds configured size limit: {path}")
        return f"data:{self._mime(path)};base64,{base64.b64encode(data).decode('ascii')}"

    def _request(self, messages: list[dict[str, Any]], *, json_schema: dict[str, Any] | None = None) -> Any:
        if not self.available:
            raise RuntimeError("VLM analyzer is not enabled/configured.")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.endpoint
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/")
            endpoint += "/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions"
        request = Request(
            endpoint,
            method="POST",
            data=payload,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            raise RuntimeError(f"VLM returned no choices: {result!r}")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        content = str(content or "").strip()
        if not content:
            raise RuntimeError("VLM returned empty content.")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"description": content}


    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cache_key(self, operation: str, image_path: Path, payload: Any) -> str:
        material = {
            "namespace": self._cache_namespace,
            "operation": operation,
            "model": self.model,
            "image_sha256": self._file_digest(image_path),
            "payload": payload,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        temp = path.with_name(path.name + ".tmp")
        try:
            temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def analyze_image(self, image_path: Path, instruction: str) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        cache_key = self._cache_key("analyze_image", image_path, {"instruction": instruction})
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached
        image_data = self._encode_image(image_path)
        schema = {
            "name": "visual_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "subjects": {"type": "array", "items": {"type": "string"}},
                    "wardrobe": {"type": "array", "items": {"type": "string"}},
                    "environment": {"type": "array", "items": {"type": "string"}},
                    "lighting": {"type": "string"},
                    "composition": {"type": "string"},
                    "identity_features": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["description", "subjects", "wardrobe", "environment", "lighting", "composition", "identity_features", "confidence"],
            },
        }
        result = self._request([
            {
                "role": "system",
                "content": "Analyze the image only. Be concrete and conservative. Do not invent unseen identity attributes.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ], json_schema=schema)
        if not isinstance(result, dict):
            raise RuntimeError("VLM image analysis was not a JSON object.")
        self._cache_write(cache_key, result)
        return result


    def score_frames(self, image_paths: list[Path], expected_state: dict[str, Any]) -> dict[str, Any]:
        """Score multiple representative frames and aggregate conservatively."""
        valid = [Path(p) for p in image_paths if Path(p).is_file()]
        if not valid:
            raise FileNotFoundError("No valid VLM review frames were supplied.")
        results = [self.score_frame(path, expected_state) for path in valid]
        numeric = ("identity_score", "continuity_score", "prompt_compliance_score")
        aggregate = {
            key: round(sum(float(r.get(key, 0.0) or 0.0) for r in results) / len(results), 2)
            for key in numeric
        }
        findings = []
        for result in results:
            findings.extend(str(v) for v in (result.get("findings", []) or []))
        actions = [str(r.get("recommended_action", "accept") or "accept") for r in results]
        status = "fail" if any(r.get("status") == "fail" for r in results) else "warning" if any(r.get("status") == "warning" for r in results) else "pass"
        return {
            "status": status,
            **aggregate,
            "findings": list(dict.fromkeys(findings)),
            "recommended_action": "retake" if "retake" in actions else "review" if "review" in actions else "accept",
            "frames": [str(p.resolve()) for p in valid],
        }

    def describe_references(self, paths: list[str]) -> dict[str, dict[str, Any]]:
        if not self.available:
            return {}
        output: dict[str, dict[str, Any]] = {}
        for raw in paths:
            path = Path(str(raw)).expanduser()
            if not path.is_file():
                continue
            try:
                output[str(path.resolve())] = self.analyze_image(
                    path,
                    "Describe visual information useful for cinematic prompting and reference conditioning. Focus on visible character identity cues, wardrobe, environment, lighting and composition.",
                )
            except Exception:
                continue
        return output

    def score_frame(self, image_path: Path, expected_state: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        cache_key = self._cache_key("score_frame", image_path, {"expected_state": expected_state or {}})
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached
        instruction = (
            "Compare the rendered frame against this expected production state. "
            "Return only observations about visible evidence; never invent hidden facts.\n\n"
            + json.dumps(expected_state or {}, ensure_ascii=False, sort_keys=True)
        )
        schema = {
            "name": "shot_qa",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["pass", "warning", "fail"]},
                    "identity_score": {"type": "number"},
                    "continuity_score": {"type": "number"},
                    "prompt_compliance_score": {"type": "number"},
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "recommended_action": {"type": "string", "enum": ["accept", "review", "retake"]},
                },
                "required": ["status", "identity_score", "continuity_score", "prompt_compliance_score", "findings", "recommended_action"],
            },
        }
        result = self._request([
            {"role": "system", "content": "You are a conservative visual QA inspector for a video production pipeline."},
            {"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image_url", "image_url": {"url": self._encode_image(Path(image_path))}}]},
        ], json_schema=schema)
        if not isinstance(result, dict):
            raise RuntimeError("VLM QA result was not a JSON object.")
        self._cache_write(cache_key, result)
        return result
