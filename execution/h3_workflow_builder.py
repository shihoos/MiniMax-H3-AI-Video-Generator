from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:

    BASE_WORKFLOWS = {
        "hard_r2v": (
            "base",
            "H3_HardMode_R2V.json",
        ),
        "hard_chained": (
            "base",
            "H3_HardMode_Chained.json",
        ),
        "seamless_v2": (
            "base",
            "H3_Seamless_Chain_v2.json",
        ),
        "seamless_core": (
            "base",
            "H3_Seamless_Chain_CORE.json",
        ),
        "keyframes": (
            "base",
            "H3_Keyframes.json",
        ),
        "extend_take": (
            "base",
            "H3_Extend_Take.json",
        ),
    }

    TURBO_WORKFLOWS = {
        "turbo_i2v": (
            "turbo",
            "H3_Turbo_I2V.json",
        ),
        "turbo_ref2v": (
            "turbo",
            "H3_Turbo_Ref2V.json",
        ),
        "turbo_t2v": (
            "turbo",
            "H3_Turbo_T2V.json",
        ),
    }

    MANUAL_WORKFLOW = (
        "H3_Ref2VA_Memory_API.json"
    )

    def __init__(
        self,
        project_root: Path,
        comfy_client,
    ):
        self.project_root = Path(
            project_root
        )

        self.client = comfy_client

        self.workflow_root = (
            self.project_root
            / "workflows"
            / "MiniMax-H3"
        )

    # =========================================================
    # WORKFLOW REGISTRY
    # =========================================================

    @classmethod
    def all_modes(cls):
        return set(
            cls.BASE_WORKFLOWS
        ) | set(
            cls.TURBO_WORKFLOWS
        )

    @classmethod
    def is_turbo_mode(
        cls,
        mode: str,
    ):
        return mode in cls.TURBO_WORKFLOWS

    def path_for_mode(
        self,
        mode: str,
    ) -> Path:

        registry = (
            self.TURBO_WORKFLOWS
            if self.is_turbo_mode(mode)
            else self.BASE_WORKFLOWS
        )

        try:
            directory, filename = (
                registry[mode]
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown H3 workflow mode: {mode}"
            ) from error

        path = (
            self.workflow_root
            / directory
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Workflow missing for "
                f"{mode}: {path}"
            )

        return path

    # =========================================================
    # LOAD
    # =========================================================

    def load_ui(
        self,
        mode: str,
    ) -> dict:

        path = self.path_for_mode(
            mode
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            workflow = json.load(
                handle
            )

        if not isinstance(
            workflow,
            dict,
        ):
            raise ValueError(
                f"Invalid workflow JSON: {path}"
            )

        return workflow

    # =========================================================
    # WIDGET HELPERS
    # =========================================================

    @staticmethod
    def _widgets(
        node: dict,
    ) -> list:

        values = node.get(
            "widgets_values"
        )

        if not isinstance(
            values,
            list,
        ):
            values = []

        node[
            "widgets_values"
        ] = values

        return values

    @classmethod
    def _set_widget(
        cls,
        node: dict,
        index: int,
        value,
    ):

        values = cls._widgets(
            node
        )

        while len(values) <= index:
            values.append(
                None
            )

        values[index] = value

    @staticmethod
    def _title(
        node: dict,
    ) -> str:

        return str(
            node.get(
                "title",
                "",
            )
        ).lower()

    @staticmethod
    def _type(
        node: dict,
    ) -> str:

        return str(
            node.get(
                "type",
                "",
            )
        )

    @staticmethod
    def _number_in_title(
        node: dict,
    ) -> int:

        match = re.search(
            r"(?:image|picture|video|audio|"
            r"keyframe)\s*([0-9]+)",
            H3WorkflowBuilder._title(
                node
            ),
            flags=re.IGNORECASE,
        )

        if not match:
            return 9999

        return int(
            match.group(1)
        )

    # =========================================================
    # UI MODEL PATCHING
    # =========================================================

    def _patch_models_ui(
        self,
        workflow: dict,
        mode: str,
    ):

        turbo = (
            self.is_turbo_mode(mode)
        )

        h3_model_nodes = [
            node
            for node in workflow.get(
                "nodes",
                []
            )
            if node.get("type")
            == "H3ModelLoaderAny"
        ]

        for index, node in enumerate(
            h3_model_nodes
        ):

            widgets = self._widgets(
                node
            )

            current = str(
                widgets[0]
                if widgets
                else ""
            ).lower()

            if turbo:

                if mode == "turbo_ref2v":
                    model_name = (
                        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
                    )
                else:
                    model_name = (
                        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
                    )

            elif mode == "hard_r2v":

                model_name = (
                    "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
                )

            elif mode == "hard_chained":

                if (
                    "ref2va"
                    in current
                ):
                    model_name = (
                        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
                    )

                elif (
                    "fl2va"
                    in current
                ):
                    model_name = (
                        "minimax_h3_fl2va_pruned-Q4_K_M.gguf"
                    )

                elif index == 0:

                    model_name = (
                        "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
                    )

                else:

                    model_name = (
                        "minimax_h3_fl2va_pruned-Q4_K_M.gguf"
                    )

            else:

                model_name = (
                    "minimax_h3_fl2va_pruned-Q4_K_M.gguf"
                )

            self._set_widget(
                node,
                0,
                model_name,
            )

        # Turbo workflow JSONs normally use UNETLoader.
        for node in workflow.get(
            "nodes",
            []
        ):

            if node.get(
                "type"
            ) not in {
                "UNETLoader",
                "CheckpointLoaderSimple",
            }:
                continue

            if not turbo:
                continue

            if mode == "turbo_ref2v":
                model_name = (
                    "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
                )
            else:
                model_name = (
                    "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
                )

            self._set_widget(
                node,
                0,
                model_name,
            )

    def _patch_text_encoders_ui(
        self,
        workflow: dict,
        mode: str,
    ):

        turbo = (
            self.is_turbo_mode(mode)
        )

        for node in workflow.get(
            "nodes",
            []
        ):

            node_type = (
                node.get(
                    "type"
                )
            )

            if node_type == "H3ClipLoaderAny":

                encoder = (
                    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
                    if turbo
                    else
                    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
                )

                self._set_widget(
                    node,
                    0,
                    encoder,
                )

                widgets = self._widgets(
                    node
                )

                if len(widgets) > 1:
                    self._set_widget(
                        node,
                        1,
                        "minimax",
                    )

            elif node_type in {
                "DualCLIPLoader",
                "CLIPLoader",
            }:

                values = self._widgets(
                    node
                )

                text = " ".join(
                    str(value)
                    for value in values
                ).lower()

                if "qwen" not in text:
                    continue

                encoder = (
                    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
                    if turbo
                    else
                    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
                )

                self._set_widget(
                    node,
                    0,
                    encoder,
                )

    def _patch_vaes_ui(
        self,
        workflow: dict,
    ):

        for node in workflow.get(
            "nodes",
            []
        ):

            if node.get(
                "type"
            ) != "VAELoader":
                continue

            title = self._title(
                node
            )

            if "audio" in title:

                self._set_widget(
                    node,
                    0,
                    "minimax_h3_audio_vae_fp32.safetensors",
                )

            elif "video" in title:

                self._set_widget(
                    node,
                    0,
                    "minimax_h3_video_vae_fp16.safetensors",
                )

    # =========================================================
    # SAMPLING / RESOLUTION
    # =========================================================

    def _patch_controls_ui(
        self,
        workflow,
        mode,
        width,
        height,
        frames_per_shot,
        steps,
        seed,
        duration_seconds,
    ):

        turbo = (
            self.is_turbo_mode(mode)
        )

        if turbo:
            if mode == "turbo_ref2v":
                steps_to_use = 4
            else:
                steps_to_use = 8
        else:
            steps_to_use = int(
                steps
            )

        megapixels = (
            (width * height)
            / 1_000_000
        )

        for node in workflow.get(
            "nodes",
            []
        ):

            node_type = node.get(
                "type"
            )

            if node_type == "ResolutionSelector":

                self._set_widget(
                    node,
                    0,
                    "16:9 (Widescreen)",
                )

                self._set_widget(
                    node,
                    1,
                    float(
                        megapixels
                    ),
                )

                self._set_widget(
                    node,
                    2,
                    32,
                )

            elif node_type == "BasicScheduler":

                self._set_widget(
                    node,
                    1,
                    steps_to_use,
                )

            elif node_type == "KSamplerSelect":

                if turbo:
                    self._set_widget(
                        node,
                        0,
                        "euler",
                    )
                else:
                    self._set_widget(
                        node,
                        0,
                        "res_multistep",
                    )

            elif node_type == "RandomNoise":

                self._set_widget(
                    node,
                    0,
                    int(seed),
                )

            elif (
                node_type
                == "PrimitiveFloat"
                and "duration" in self._title(node)
            ):

                self._set_widget(
                    node,
                    0,
                    float(
                        duration_seconds
                    ),
                )

    # =========================================================
    # PROMPT
    # =========================================================

    def _patch_prompt_ui(
        self,
        workflow,
        script,
    ):

        candidates = []

        for node in workflow.get(
            "nodes",
            []
        ):

            node_type = node.get(
                "type"
            )

            if node_type not in {
                "PrimitiveStringMultiline",
                "PrimitiveString",
            }:
                continue

            title = self._title(
                node
            )

            score = 0

            if (
                "input text"
                in title
            ):
                score += 100

            if "prompt" in title:
                score += 70

            if "script" in title:
                score += 60

            if (
                "script + bindings"
                in title
            ):
                score += 110

            if score:
                candidates.append(
                    (
                        score,
                        node,
                    )
                )

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        if candidates:

            self._set_widget(
                candidates[0][1],
                0,
                script,
            )

    # =========================================================
    # REFERENCES
    # =========================================================

    @staticmethod
    def _matching_nodes(
        workflow,
        *,
        kind,
    ):

        result = []

        for node in workflow.get(
            "nodes",
            []
        ):

            node_type = str(
                node.get(
                    "type",
                    ""
                )
            ).lower()

            title = str(
                node.get(
                    "title",
                    ""
                )
            ).lower()

            score = 0

            if kind == "image":

                if (
                    node_type == "loadimage"
                ):
                    score += 10

                if (
                    "<picture"
                    in title
                ):
                    score += 100

                if (
                    "reference image"
                    in title
                ):
                    score += 90

            elif kind == "video":

                if (
                    "loadvideo"
                    in node_type
                ):
                    score += 10

                if (
                    "<video"
                    in title
                ):
                    score += 100

                if (
                    "reference video"
                    in title
                ):
                    score += 90

            elif kind == "audio":

                if (
                    node_type == "loadaudio"
                ):
                    score += 10

                if (
                    "<audio"
                    in title
                ):
                    score += 100

                if (
                    "reference audio"
                    in title
                ):
                    score += 90

                if (
                    "voice reference"
                    in title
                ):
                    score += 95

            elif kind == "keyframe":

                if (
                    node_type == "loadimage"
                ):
                    score += 10

                if "keyframe" in title:
                    score += 100

            if score:
                result.append(
                    (
                        score,
                        H3WorkflowBuilder._number_in_title(node),
                        node,
                    )
                )

        result.sort(
            key=lambda item:
                (
                    -item[0],
                    item[1],
                )
        )

        return [
            item[2]
            for item in result
        ]

    def _patch_reference_loaders_ui(
        self,
        workflow,
        image_files,
        video_files,
        audio_files,
        keyframe_files,
        mode,
    ):

        if image_files:

            nodes = self._matching_nodes(
                workflow,
                kind="image",
            )

            if len(nodes) < len(
                image_files
            ):

                raise RuntimeError(
                    f"{mode}: workflow exposes "
                    f"{len(nodes)} image reference "
                    f"slots but {len(image_files)} "
                    "images were requested."
                )

            for node, filename in zip(
                nodes,
                image_files,
            ):

                self._set_widget(
                    node,
                    0,
                    filename,
                )

        if video_files:

            nodes = self._matching_nodes(
                workflow,
                kind="video",
            )

            if len(nodes) < len(
                video_files
            ):

                raise RuntimeError(
                    f"{mode}: workflow exposes "
                    f"{len(nodes)} video reference "
                    f"slots but {len(video_files)} "
                    "videos were requested."
                )

            for node, filename in zip(
                nodes,
                video_files,
            ):

                self._set_widget(
                    node,
                    0,
                    filename,
                )

        if audio_files:

            nodes = self._matching_nodes(
                workflow,
                kind="audio",
            )

            if len(nodes) < len(
                audio_files
            ):

                raise RuntimeError(
                    f"{mode}: workflow exposes "
                    f"{len(nodes)} audio slots but "
                    f"{len(audio_files)} standalone "
                    "audio references were requested."
                )

            for node, filename in zip(
                nodes,
                audio_files,
            ):

                self._set_widget(
                    node,
                    0,
                    filename,
                )

        if keyframe_files:

            nodes = self._matching_nodes(
                workflow,
                kind="keyframe",
            )

            if len(nodes) < len(
                keyframe_files
            ):

                raise RuntimeError(
                    f"{mode}: workflow exposes "
                    f"{len(nodes)} keyframe slots but "
                    f"{len(keyframe_files)} were requested."
                )

            for node, filename in zip(
                nodes,
                keyframe_files,
            ):

                self._set_widget(
                    node,
                    0,
                    filename,
                )

    # =========================================================
    # API PATCHES AFTER CONVERSION
    # =========================================================

    @staticmethod
    def _set_api(
        node,
        key,
        value,
    ):

        inputs = node.setdefault(
            "inputs",
            {}
        )

        if (
            key in inputs
            or key in {
                "script",
                "shot_count",
                "width",
                "height",
                "length",
                "frames_per_shot",
                "steps",
                "seed",
                "seed_per_shot",
                "prompt",
                "sampler_name",
                "filename_prefix",
                "scheduler",
            }
        ):

            inputs[key] = value

    def _patch_api(
        self,
        workflow,
        *,
        mode,
        script,
        shot_count,
        width,
        height,
        frames_per_shot,
        steps,
        seed,
        output_prefix,
        refs_root,
    ):

        turbo = self.is_turbo_mode(
            mode
        )

        sampler_steps = (
            4
            if mode == "turbo_ref2v"
            else 8
            if turbo
            else steps
        )

        for _, node in workflow.items():

            if not isinstance(
                node,
                dict,
            ):
                continue

            class_type = node.get(
                "class_type"
            )

            if class_type in {
                "H3MultishotMemorySampler",
                "H3MultishotSampler",
            }:

                self._set_api(
                    node,
                    "script",
                    script,
                )

                self._set_api(
                    node,
                    "shot_count",
                    shot_count,
                )

                self._set_api(
                    node,
                    "width",
                    width,
                )

                self._set_api(
                    node,
                    "height",
                    height,
                )

                self._set_api(
                    node,
                    "frames_per_shot",
                    frames_per_shot,
                )

                self._set_api(
                    node,
                    "steps",
                    sampler_steps,
                )

                self._set_api(
                    node,
                    "seed",
                    seed,
                )

                self._set_api(
                    node,
                    "seed_per_shot",
                    True,
                )

            elif class_type == (
                "MiniMaxH3ReferenceToVideo"
            ):

                self._set_api(
                    node,
                    "prompt",
                    script,
                )

                self._set_api(
                    node,
                    "width",
                    width,
                )

                self._set_api(
                    node,
                    "height",
                    height,
                )

                self._set_api(
                    node,
                    "length",
                    frames_per_shot,
                )

                self._set_api(
                    node,
                    "ref_image_size",
                    "match",
                )

            elif class_type == (
                "BasicScheduler"
            ):

                self._set_api(
                    node,
                    "steps",
                    sampler_steps,
                )

            elif class_type == (
                "KSamplerSelect"
            ):

                self._set_api(
                    node,
                    "sampler_name",
                    (
                        "euler"
                        if turbo
                        else "res_multistep"
                    ),
                )

            elif class_type == (
                "RandomNoise"
            ):

                self._set_api(
                    node,
                    "noise_seed",
                    seed,
                )

            elif class_type == (
                "H3AutoRefs"
            ):

                self._set_api(
                    node,
                    "refs_root",
                    refs_root,
                )

                self._set_api(
                    node,
                    "max_per_character",
                    3,
                )

                self._set_api(
                    node,
                    "script",
                    script,
                )

            elif class_type == (
                "H3EpisodeSplit"
            ):

                self._set_api(
                    node,
                    "script",
                    script,
                )

                self._set_api(
                    node,
                    "shot_count",
                    shot_count,
                )

            elif class_type in {
                "SaveVideo",
                "VHS_VideoCombine",
            }:

                self._set_api(
                    node,
                    "filename_prefix",
                    output_prefix,
                )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _is_api(
        workflow,
    ):

        return (
            isinstance(
                workflow,
                dict,
            )
            and bool(workflow)
            and all(
                isinstance(node, dict)
                and "class_type" in node
                and "inputs" in node
                for node in workflow.values()
            )
        )

    def _validate_api(
        self,
        workflow,
        mode,
    ):

        if not self._is_api(
            workflow
        ):
            raise RuntimeError(
                f"{mode}: converted workflow "
                "is not valid ComfyUI API format."
            )

        classes = {
            node.get(
                "class_type"
            )
            for node in workflow.values()
            if isinstance(
                node,
                dict,
            )
        }

        if mode == "hard_r2v":
            required = {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
                "MiniMaxH3ReferenceToVideo",
                "H3FreeTextEncoder",
                "VAEDecode",
                "VAEDecodeAudio",
                "CreateVideo",
                "SaveVideo",
            }

        elif mode in {
            "hard_chained",
            "seamless_v2",
            "seamless_core",
        }:
            required = {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
            }

            if not (
                {
                    "H3MultishotSampler",
                    "H3MultishotMemorySampler",
                }
                & classes
            ):
                raise RuntimeError(
                    f"{mode}: no H3 multishot sampler "
                    "exists in the selected workflow."
                )

        elif mode == "keyframes":
            required = {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
            }

        elif mode == "extend_take":
            required = {
                "H3ModelLoaderAny",
                "H3ClipLoaderAny",
            }

        elif mode.startswith(
            "turbo_"
        ):
            required = set()

        else:
            raise ValueError(
                f"Unsupported workflow: {mode}"
            )

        missing = sorted(
            required
            - classes
        )

        if missing:
            raise RuntimeError(
                f"{mode}: missing required nodes: "
                + ", ".join(missing)
            )

    # =========================================================
    # PUBLIC
    # =========================================================

    def build(
        self,
        *,
        mode,
        profile,
        script,
        shot_count,
        width,
        height,
        frames_per_shot,
        steps,
        seed,
        image_files=None,
        video_files=None,
        audio_files=None,
        keyframe_files=None,
        output_prefix="h3/output",
        refs_root="",
    ):

        if mode not in self.all_modes():
            raise ValueError(
                f"Unknown workflow mode: {mode}"
            )

        if (
            mode.startswith(
                "turbo_"
            )
            and profile != "turbo"
        ):
            raise ValueError(
                f"{mode} requires profile='turbo'."
            )

        if (
            profile == "turbo"
            and not self.is_turbo_mode(mode)
        ):
            raise ValueError(
                "Turbo profile requires one of the "
                "H3_Turbo_* workflows."
            )

        workflow = self.load_ui(
            mode
        )

        workflow = copy.deepcopy(
            workflow
        )

        self._patch_models_ui(
            workflow,
            mode,
        )

        self._patch_text_encoders_ui(
            workflow,
            mode,
        )

        self._patch_vaes_ui(
            workflow
        )

        self._patch_controls_ui(
            workflow,
            mode,
            width,
            height,
            frames_per_shot,
            steps,
            seed,
            duration_seconds=(
                frames_per_shot
                / 24.0
            ),
        )

        self._patch_prompt_ui(
            workflow,
            script,
        )

        self._patch_reference_loaders_ui(
            workflow,
            image_files or [],
            video_files or [],
            audio_files or [],
            keyframe_files or [],
            mode,
        )

        api_workflow = (
            self.client.convert_workflow(
                workflow
            )
        )

        api_workflow = copy.deepcopy(
            api_workflow
        )

        self._patch_api(
            api_workflow,
            mode=mode,
            script=script,
            shot_count=shot_count,
            width=width,
            height=height,
            frames_per_shot=frames_per_shot,
            steps=steps,
            seed=seed,
            output_prefix=output_prefix,
            refs_root=refs_root,
        )

        self._validate_api(
            api_workflow,
            mode,
        )

        return api_workflow
