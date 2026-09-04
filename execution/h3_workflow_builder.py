from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from planner.config import (
    H3_AUDIO_VAE,
    H3_FPS,
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_REF2VA_MODEL,
    H3_REF_IMAGE_SIZE,
    H3_TEXT_ENCODER,
    H3_TURBO_LORA,
    H3_VIDEO_VAE,
    H3_LATENT_UPSCALER_3D,
    H3_WIDTH,
    TURBO_STEPS,
    UPSCALE_HEIGHT,
    UPSCALE_WIDTH,
)


class H3WorkflowBuilder:

    MODES = {
        "ref2va": (
            Path("generation"),
            "H3_Ref2VA_Production.json",
        ),
        "turbo_ref2va": (
            Path("generation"),
            "H3_Turbo_Ref2VA_Production.json",
        ),
        "upscale": (
            Path("postprocess"),
            "H3_Ref2VA_UltimateUpscale_Production.json",
        ),
    }

    def __init__(
        self,
        project_root: Path,
        comfy_client,
    ):
        self.project_root = Path(project_root)
        self.client = comfy_client
        self.workflow_root = (
            self.project_root / "workflows"
        )

    @classmethod
    def _repair_turbo_model_chain(cls, workflow: dict) -> None:
        """Enforce the canonical T4 Turbo MODEL topology.

        Production topology is deliberately explicit and does not infer
        downstream consumers from a damaged graph:

            UNETLoader
                -> MiniMaxH3TurboLoRA
                -> H3MemoryOptimization
                -> BasicScheduler
                -> BasicGuider

        The FP16Safe experiment is intentionally not part of the production
        workflow. Its instance-level DiT/MLP patches conflict with the public
        H3MemoryOptimization block patch: the H3 optimizer preserves foreign
        block-forward patches by disabling its bounded block path. The final
        production path therefore uses the H3 optimizer as the sole block/MLP
        execution owner after the Turbo LoRA merge.
        """
        unet = cls._one(workflow, "UNETLoader")
        turbo = cls._one(workflow, "MiniMaxH3TurboLoRA")
        optimizer = cls._one(workflow, "H3MemoryOptimization")
        scheduler = cls._one(workflow, "BasicScheduler")
        guider = cls._one(workflow, "BasicGuider")

        nodes = cls._nodes(workflow)
        links = workflow.setdefault("links", [])

        def model_input_slot(node):
            slots = [
                i for i, item in enumerate(node.get("inputs", []))
                if (
                    isinstance(item, dict)
                    and str(item.get("name", "")).lower() == "model"
                )
            ]
            if len(slots) != 1:
                raise RuntimeError(
                    f"Expected exactly one MODEL input on {node.get('type')}; found {slots}"
                )
            return slots[0]

        def is_model_link(row):
            return (
                isinstance(row, list)
                and len(row) >= 6
                and str(row[5]).upper() == "MODEL"
            )

        ids = {
            "unet": cls._node_id(unet),
            "turbo": cls._node_id(turbo),
            "optimizer": cls._node_id(optimizer),
            "scheduler": cls._node_id(scheduler),
            "guider": cls._node_id(guider),
        }
        chain_ids = set(ids.values())

        # A stale FP16Safe node is not a production dependency. Remove it from
        # the graph entirely so bootstrap can reproduce the runtime without
        # an extra unpinned/unused custom node.
        stale_safe = [
            node for node in nodes
            if node.get("type") == "MiniMaxH3FP16Safe"
        ]
        stale_safe_ids = {cls._node_id(node) for node in stale_safe}
        if stale_safe_ids:
            links[:] = [
                row for row in links
                if not (
                    is_model_link(row)
                    and (
                        int(row[1]) in stale_safe_ids
                        or int(row[3]) in stale_safe_ids
                    )
                )
            ]
            workflow["nodes"] = [
                node for node in nodes
                if cls._node_id(node) not in stale_safe_ids
            ]
            nodes = cls._nodes(workflow)

        model_targets = {
            ids["scheduler"],
            ids["guider"],
        }

        # Remove every MODEL edge touching any canonical chain node, and every
        # MODEL edge into the two terminal consumers. Non-MODEL graph links are
        # preserved byte-for-byte.
        links[:] = [
            row for row in links
            if not (
                is_model_link(row)
                and (
                    int(row[1]) in chain_ids
                    or int(row[3]) in chain_ids
                    or int(row[3]) in model_targets
                )
            )
        ]

        model_nodes = (unet, turbo, optimizer, scheduler, guider)
        for node in model_nodes:
            for inp in node.get("inputs", []):
                if (
                    isinstance(inp, dict)
                    and str(inp.get("name", "")).lower() == "model"
                ):
                    inp["link"] = None
            for output in node.get("outputs", []):
                if (
                    isinstance(output, dict)
                    and str(output.get("name", "")).upper() == "MODEL"
                ):
                    output["links"] = []

        turbo_slot = model_input_slot(turbo)
        optimizer_slot = model_input_slot(optimizer)
        scheduler_slot = model_input_slot(scheduler)
        guider_slot = model_input_slot(guider)

        existing_ids = [
            int(row[0])
            for row in links
            if isinstance(row, list) and row
        ]
        next_link = max(existing_ids, default=0) + 1

        def add_link(src_id, dst_id, dst_slot):
            nonlocal next_link
            lid = next_link
            next_link += 1
            links.append([
                lid,
                src_id,
                0,
                dst_id,
                dst_slot,
                "MODEL",
            ])
            return lid

        l1 = add_link(ids["unet"], ids["turbo"], turbo_slot)
        l2 = add_link(ids["turbo"], ids["optimizer"], optimizer_slot)
        l3 = add_link(ids["optimizer"], ids["scheduler"], scheduler_slot)
        l4 = add_link(ids["optimizer"], ids["guider"], guider_slot)

        turbo["inputs"][turbo_slot]["link"] = l1
        optimizer["inputs"][optimizer_slot]["link"] = l2
        scheduler["inputs"][scheduler_slot]["link"] = l3
        guider["inputs"][guider_slot]["link"] = l4

        def add_output_link(node, link_id):
            for output in node.get("outputs", []):
                if (
                    isinstance(output, dict)
                    and str(output.get("name", "")).upper() == "MODEL"
                ):
                    output.setdefault("links", []).append(link_id)

        add_output_link(unet, l1)
        add_output_link(turbo, l2)
        add_output_link(optimizer, l3)
        add_output_link(optimizer, l4)

        widgets = cls._widgets(turbo)
        if len(widgets) < 3:
            raise RuntimeError(
                "MiniMaxH3TurboLoRA requires at least three widgets."
            )
        widgets[0] = H3_TURBO_LORA
        widgets[1] = 1.0
        widgets[2] = True

        cls._apply_memory_optimization_config(optimizer)

        # Final structural assertion: exactly the four canonical MODEL edges.
        expected = {
            (ids["unet"], ids["turbo"]),
            (ids["turbo"], ids["optimizer"]),
            (ids["optimizer"], ids["scheduler"]),
            (ids["optimizer"], ids["guider"]),
        }
        actual = {
            (int(row[1]), int(row[3]))
            for row in links
            if is_model_link(row)
            and int(row[1]) in chain_ids
            and int(row[3]) in chain_ids
        }
        if actual != expected:
            raise RuntimeError(
                f"Turbo MODEL topology mismatch. expected={sorted(expected)} actual={sorted(actual)}"
            )

    # ============================================================
    # BASIC GRAPH HELPERS
    # ============================================================

    @staticmethod
    def _nodes(
        workflow: dict,
    ):
        return workflow.get(
            "nodes",
            [],
        )

    @classmethod
    def _find(
        cls,
        workflow: dict,
        node_type: str,
    ):
        return [
            node
            for node in cls._nodes(workflow)
            if node.get("type") == node_type
        ]

    @classmethod
    def _one(
        cls,
        workflow: dict,
        node_type: str,
    ):
        found = cls._find(
            workflow,
            node_type,
        )

        if len(found) != 1:
            raise RuntimeError(
                f"Expected exactly one "
                f"{node_type}; found {len(found)}."
            )

        return found[0]

    @staticmethod
    def _widgets(
        node: dict,
    ):
        values = node.setdefault(
            "widgets_values",
            [],
        )

        if not isinstance(values, list):
            raise RuntimeError(
                f"Unsupported widgets_values format "
                f"for node {node.get('id')}."
            )

        return values

    @classmethod
    def _set_widget(
        cls,
        node: dict,
        index: int,
        value,
    ):
        widgets = cls._widgets(node)

        while len(widgets) <= index:
            widgets.append(None)

        widgets[index] = value

    @staticmethod
    def _node_id(
        node: dict,
    ) -> int:
        return int(node["id"])


    # ============================================================
    # H3 MEMORY OPTIMIZATION
    # ============================================================

    @staticmethod
    def _apply_memory_optimization_config(
        node: dict,
    ) -> None:
        """Synchronize an existing H3MemoryOptimization node with runtime config."""
        try:
            from planner.config import RUNTIME
            cfg = dict(RUNTIME.get("h3_optimization", {}) or {})
        except Exception:
            cfg = {}

        chunk_rows = int(cfg.get("chunk_rows", 2560) or 2560)
        precision_mode = str(cfg.get("precision_mode", "Auto") or "Auto")
        qkv_streaming_mode = str(
            cfg.get("qkv_streaming_mode", "Auto") or "Auto"
        )
        attention_memory_mode = str(
            cfg.get("attention_memory_mode", "Standard") or "Standard"
        )
        version = str(cfg.get("version", "0.2.41") or "0.2.41")

        node["properties"] = dict(node.get("properties") or {})
        node["properties"]["cnr_id"] = "h3-optimizations"
        node["properties"]["ver"] = version
        node["properties"]["Node name for S&R"] = "H3MemoryOptimization"

        values = [
            "auto",
            "auto",
            chunk_rows,
            True,
            precision_mode,
            qkv_streaming_mode,
            "Auto",
            attention_memory_mode,
        ]
        node["widgets_values"] = values
        node["widgets_values_named"] = {
            "fused_qkv": "auto",
            "mlp_memory": "auto",
            "chunk_rows": chunk_rows,
            "preserve_precision": True,
            "precision_mode": precision_mode,
            "qkv_streaming_mode": qkv_streaming_mode,
            "embedding_memory_mode": "Auto",
            "kitchen_v_memory_mode": attention_memory_mode,
        }

    def _inject_memory_optimization(
        self,
        workflow: dict,
        *,
        source_node_type: str = "UNETLoader",
    ) -> None:
        """Ensure one H3MemoryOptimization node owns the H3 MODEL fan-out."""
        existing = self._find(workflow, "H3MemoryOptimization")
        if existing:
            if len(existing) != 1:
                raise RuntimeError(
                    "Workflow contains multiple H3MemoryOptimization nodes."
                )
            self._assert_optimizer_ownership(
                workflow,
                source_node_type,
            )
            self._apply_memory_optimization_config(existing[0])
            return

        source = self._one(workflow, source_node_type)
        source_id = self._node_id(source)
        model_edges = [
            list(row)
            for row in workflow.get("links", [])
            if (
                isinstance(row, list)
                and len(row) >= 6
                and int(row[1]) == source_id
                and str(row[5]).upper() == "MODEL"
            )
        ]
        if not model_edges:
            raise RuntimeError(
                f"{source_node_type} has no MODEL consumers to optimize."
            )

        node_id = self._next_id(workflow)
        pos = source.get("pos", [0, 0])
        node = {
            "id": node_id,
            "type": "H3MemoryOptimization",
            "pos": [float(pos[0]) + 720.0, float(pos[1])],
            "size": [390, 270],
            "flags": {},
            "order": int(source.get("order", 0)) + 1,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": None},
            ],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": []},
            ],
            "properties": {},
            "widgets_values": [],
            "widgets_values_named": {},
        }
        self._apply_memory_optimization_config(node)

        # Connect producer -> optimizer.
        input_link = self._next_link_id(workflow)
        node["inputs"][0]["link"] = input_link
        workflow.setdefault("links", []).append(
            [input_link, source_id, 0, node_id, 0, "MODEL"]
        )

        old_ids = {int(row[0]) for row in model_edges}
        workflow["links"] = [
            row
            for row in workflow.get("links", [])
            if not (
                isinstance(row, list)
                and len(row) >= 1
                and int(row[0]) in old_ids
            )
        ]
        for output in source.get("outputs", []):
            if isinstance(output.get("links"), list):
                output["links"] = [
                    x for x in output["links"] if int(x) not in old_ids
                ]

        for row in model_edges:
            new_link = self._next_link_id(workflow)
            target_id = int(row[3])
            target_slot = int(row[4])
            workflow["links"].append(
                [new_link, node_id, 0, target_id, target_slot, "MODEL"]
            )
            target = next(
                (
                    n
                    for n in self._nodes(workflow)
                    if self._node_id(n) == target_id
                ),
                None,
            )
            if target is None:
                raise RuntimeError(
                    f"Missing MODEL consumer node {target_id} after optimizer insertion."
                )
            replaced = False
            for inp in target.get("inputs", []):
                if isinstance(inp, dict) and inp.get("link") == int(row[0]):
                    inp["link"] = new_link
                    replaced = True
                    break
            if not replaced:
                raise RuntimeError(
                    f"Could not replace MODEL link {row[0]} on node {target_id}."
                )
            node["outputs"][0]["links"].append(new_link)

        workflow.setdefault("nodes", []).append(node)
        if not node["outputs"][0]["links"]:
            raise RuntimeError(
                "H3MemoryOptimization produced no downstream MODEL links."
            )

        self._assert_optimizer_ownership(
            workflow,
            source_node_type,
        )

    @classmethod
    def _assert_optimizer_ownership(
        cls,
        workflow: dict,
        source_node_type: str,
    ) -> None:
        """Verify that H3MemoryOptimization is the sole MODEL consumer of the H3 source."""
        source = cls._one(workflow, source_node_type)
        optimizer = cls._one(workflow, "H3MemoryOptimization")
        source_id = cls._node_id(source)
        optimizer_id = cls._node_id(optimizer)

        def is_model_link(row) -> bool:
            return (
                isinstance(row, list)
                and len(row) >= 6
                and str(row[5]).upper() == "MODEL"
            )

        model_links = [
            row for row in workflow.get("links", [])
            if is_model_link(row)
        ]

        source_outgoing = [
            row for row in model_links
            if int(row[1]) == source_id
        ]
        if len(source_outgoing) != 1 or int(source_outgoing[0][3]) != optimizer_id:
            raise RuntimeError(
                f"{source_node_type} must have exactly one MODEL consumer "
                "and it must be H3MemoryOptimization."
            )

        optimizer_incoming = [
            row for row in model_links
            if int(row[3]) == optimizer_id
        ]
        if len(optimizer_incoming) != 1 or int(optimizer_incoming[0][1]) != source_id:
            raise RuntimeError(
                "H3MemoryOptimization must have exactly one MODEL input "
                f"owned by {source_node_type}."
            )

        optimizer_outgoing = [
            row for row in model_links
            if int(row[1]) == optimizer_id
        ]
        if not optimizer_outgoing:
            raise RuntimeError(
                "H3MemoryOptimization has no downstream MODEL consumers."
            )

        source_model_outputs = [
            output for output in source.get("outputs", [])
            if (
                isinstance(output, dict)
                and str(output.get("name", "")).upper() == "MODEL"
            )
        ]
        if len(source_model_outputs) != 1:
            raise RuntimeError(
                f"{source_node_type} must expose exactly one MODEL output."
            )

        source_output_links = {
            int(link_id)
            for link_id in source_model_outputs[0].get("links", []) or []
        }
        if source_output_links != {int(source_outgoing[0][0])}:
            raise RuntimeError(
                f"{source_node_type} MODEL output metadata disagrees with the graph link."
            )

        optimizer_model_inputs = [
            item for item in optimizer.get("inputs", [])
            if (
                isinstance(item, dict)
                and str(item.get("name", "")).lower() == "model"
            )
        ]
        optimizer_link = (
            optimizer_model_inputs[0].get("link")
            if len(optimizer_model_inputs) == 1
            else None
        )
        if len(optimizer_model_inputs) != 1 or optimizer_link is None:
            raise RuntimeError(
                "H3MemoryOptimization must expose exactly one linked MODEL input."
            )
        if int(optimizer_link) != int(optimizer_incoming[0][0]):
            raise RuntimeError(
                "H3MemoryOptimization MODEL input metadata disagrees with the graph link."
            )

    # ============================================================
    # LOAD
    # ============================================================

    def load(
        self,
        mode: str,
    ) -> dict:

        if mode not in self.MODES:
            raise ValueError(
                f"Unknown H3 workflow mode: {mode}"
            )

        directory, filename = self.MODES[
            mode
        ]

        path = (
            self.workflow_root
            / directory
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing H3 production workflow:\n{path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        if mode == "turbo_ref2va":
            data = copy.deepcopy(data)
            self._repair_turbo_model_chain(data)
            return data

        return copy.deepcopy(data)

    # ============================================================
    # H3 FRAME GRID
    # ============================================================

    @staticmethod
    def _legal_frames(
        duration_seconds: float,
    ) -> int:

        seconds = float(
            duration_seconds
        )

        if seconds < 4.0:
            seconds = 4.0

        if seconds > 15.0:
            seconds = 15.0

        requested = round(
            seconds * H3_FPS
        )

        # H3 VAE frame grid = 17*n + 5.
        n = max(
            0,
            (requested - 5 + 16) // 17,
        )

        frames = 17 * n + 5

        frames = max(
            124,
            frames,
        )

        frames = min(
            362,
            frames,
        )

        return frames

    # ============================================================
    # PROMPT / SEED
    # ============================================================

    def _set_prompt(
        self,
        workflow: dict,
        prompt: str,
    ):
        nodes = self._find(
            workflow,
            "PrimitiveStringMultiline",
        )

        if not nodes:
            raise RuntimeError(
                "H3 workflow has no prompt node."
            )

        labelled = [
            node
            for node in nodes
            if (
                "prompt"
                in str(
                    node.get(
                        "title",
                        "",
                    )
                ).lower()
            )
        ]

        target = (
            labelled[0]
            if labelled
            else nodes[0]
        )

        self._set_widget(
            target,
            0,
            prompt,
        )

    def _set_seed(
        self,
        workflow: dict,
        seed: int,
    ):
        nodes = self._find(
            workflow,
            "RandomNoise",
        )

        if not nodes:
            raise RuntimeError(
                "H3 workflow has no RandomNoise node."
            )

        for node in nodes:
            self._set_widget(
                node,
                0,
                int(seed),
            )

    # ============================================================
    # 16:9
    # ============================================================

    def _set_resolution(
        self,
        workflow: dict,
        width: int,
        height: int,
    ):
        key = (int(width), int(height))
        supported = {
            (608, 352): 0.20,
            (736, 416): 0.30,
            (864, 480): 0.40,
            (960, 544): 0.50,
            (1056, 608): 0.60,
            (1152, 640): 0.70,
            (1216, 672): 0.80,
            (1280, 736): 0.90,
            (1344, 768): 0.98,
            (1376, 768): 1.00,
            (1504, 832): 1.20,
            (1664, 928): 1.50,
            (1824, 1024): 1.80,
            (1920, 1088): 2.00,
        }

        if key not in supported:
            ratio = float(width) / float(height)
            if abs(ratio - (16.0 / 9.0)) > 0.03:
                raise ValueError(
                    "MiniMax H3 production is locked to 16:9 in this repository."
                )

        selectors = self._find(
            workflow,
            "ResolutionSelector",
        )

        if len(selectors) != 1:
            raise RuntimeError(
                "Expected one ResolutionSelector."
            )

        selector = selectors[0]

        widgets = self._widgets(
            selector
        )

        while len(widgets) < 3:
            widgets.append(None)

        if key not in supported:
            raise ValueError(
                "Unsupported H3 16:9 generation resolution: "
                f"{width}x{height}. Supported production sizes are: "
                + ", ".join(
                    f"{w}x{h}"
                    for w, h in supported
                )
            )

        widgets[0] = "16:9 (Widescreen)"
        widgets[1] = supported[key]
        widgets[2] = 32

        named = selector.get(
            "widgets_values_named",
        )

        if isinstance(
            named,
            dict,
        ):
            named[
                "aspect_ratio"
            ] = "16:9 (Widescreen)"
            named[
                "megapixels"
            ] = widgets[1]
            named[
                "multiple"
            ] = 32

    # ============================================================
    # REFERENCE IMAGE SIZE POLICY
    # ============================================================

    def _set_ref_image_size(
        self,
        workflow: dict,
        ref_image_size: str | None,
    ) -> None:
        if ref_image_size is None:
            return

        value = str(ref_image_size).strip().lower()
        if value not in {"match", "max"}:
            raise ValueError(
                "ref_image_size must be 'match' or 'max'."
            )

        nodes = self._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )
        if len(nodes) != 1:
            raise RuntimeError(
                "Expected exactly one MiniMaxH3ReferenceToVideo node."
            )

        node = nodes[0]
        widgets = self._widgets(node)
        while len(widgets) <= 4:
            widgets.append(None)
        widgets[4] = value

        named = node.get("widgets_values_named")
        if isinstance(named, dict):
            named["ref_image_size"] = value

    # ============================================================
    # LINKED DURATION
    # ============================================================

    def _set_duration_source(
        self,
        workflow: dict,
        duration_seconds: float,
    ) -> bool:
        """Set the PrimitiveFloat feeding the H3 frame-grid expression.

        The production workflow intentionally keeps the ComfyMathExpression
        formula intact. The duration input is the source value consumed by
        that expression, while the expression itself converts seconds into
        the H3-legal 17*n+5 frame count.
        """
        ref_node = self._one(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        length_link = next(
            (
                item.get("link")
                for item in ref_node.get("inputs", [])
                if item.get("name") == "length"
                and item.get("link") is not None
            ),
            None,
        )

        if length_link is None:
            return False

        links = workflow.get("links", [])
        expression_node = None

        for row in links:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                if int(row[0]) != int(length_link):
                    continue
                expression_node_id = int(row[1])
            except (TypeError, ValueError):
                continue
            expression_node = next(
                (
                    node
                    for node in self._nodes(workflow)
                    if self._node_id(node) == expression_node_id
                ),
                None,
            )
            break

        if expression_node is None or expression_node.get("type") != "ComfyMathExpression":
            raise RuntimeError(
                "H3 length input must be driven by ComfyMathExpression."
            )

        duration_input = next(
            (
                item
                for item in expression_node.get("inputs", [])
                if item.get("name") == "values.a"
                and item.get("link") is not None
            ),
            None,
        )

        if duration_input is None:
            raise RuntimeError(
                "H3 ComfyMathExpression is missing its duration input link."
            )

        duration_link = duration_input.get("link")
        duration_source = None

        for row in links:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                if int(row[0]) != int(duration_link):
                    continue
                duration_source_id = int(row[1])
            except (TypeError, ValueError):
                continue
            duration_source = next(
                (
                    node
                    for node in self._nodes(workflow)
                    if self._node_id(node) == duration_source_id
                ),
                None,
            )
            break

        if duration_source is None:
            raise RuntimeError(
                "H3 duration source node could not be resolved."
            )

        if duration_source.get("type") != "PrimitiveFloat":
            raise RuntimeError(
                "H3 ComfyMathExpression duration input must be fed by PrimitiveFloat; "
                f"found {duration_source.get('type')!r}."
            )

        value = float(duration_seconds)
        self._set_widget(
            duration_source,
            0,
            value,
        )

        named = duration_source.get("widgets_values_named")
        if isinstance(named, dict) and "value" in named:
            named["value"] = value

        return True

    def _set_duration(
        self,
        workflow: dict,
        duration_seconds: float,
    ) -> None:
        """Set seconds on the workflow's duration source without changing its formula."""
        frames = self._legal_frames(
            duration_seconds
        )

        ref_node = self._one(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        if self._set_duration_source(
            workflow,
            float(duration_seconds),
        ):
            # Keep the serialized reference-node widget coherent for tooling
            # that inspects the saved graph, while the connected PrimitiveFloat
            # remains the authoritative runtime input.
            widgets = self._widgets(ref_node)
            while len(widgets) <= 3:
                widgets.append(None)
            widgets[3] = frames

            named = ref_node.get("widgets_values_named")
            if isinstance(named, dict) and "length" in named:
                named["length"] = frames
            return

        # A production H3 graph is expected to have the linked duration source.
        # Do not silently replace the frame-grid expression with a constant.
        raise RuntimeError(
            "H3 production workflow has no valid PrimitiveFloat duration source "
            "feeding MiniMaxH3ReferenceToVideo.length."
        )


    # ============================================================
    # ULTIMATE UPSCALE TARGET
    # ============================================================

    def _set_upscale_target(
        self,
        workflow: dict,
        width: int,
        height: int,
    ):
        params = self._find(
            workflow,
            "MMH3LatentUpscaleWithModelParams",
        )

        if len(params) != 1:
            raise RuntimeError(
                "Expected one "
                "MMH3LatentUpscaleWithModelParams."
            )

        param = params[0]

        input_links = {
            item.get("name"): item.get("link")
            for item in param.get(
                "inputs",
                [],
            )
        }

        for name, value in (
            ("width", width),
            ("height", height),
        ):
            link_id = input_links.get(
                name
            )

            if link_id is None:
                continue

            link = next(
                (
                    row
                    for row in workflow.get(
                        "links",
                        [],
                    )
                    if (
                        isinstance(row, list)
                        and row
                        and row[0] == link_id
                    )
                ),
                None,
            )

            if link is None:
                continue

            source_id = int(
                link[1]
            )

            source = next(
                (
                    node
                    for node in self._nodes(
                        workflow
                    )
                    if self._node_id(node)
                    == source_id
                ),
                None,
            )

            if (
                source is not None
                and source.get("type")
                == "ComfyMathExpression"
            ):
                widgets = self._widgets(
                    source
                )

                if widgets:
                    widgets[0] = str(
                        int(value)
                    )

        # Keep widget fallback coherent.
        widgets = self._widgets(
            param
        )

        if len(widgets) >= 3:
            widgets[1] = int(width)
            widgets[2] = int(height)

    # ============================================================
    # MEDIA GRAPH HELPERS
    # ============================================================

    def _next_id(
        self,
        workflow: dict,
    ) -> int:
        value = int(
            workflow.get(
                "last_node_id",
                0,
            )
        ) + 1

        workflow[
            "last_node_id"
        ] = value

        return value

    def _next_link_id(
        self,
        workflow: dict,
    ) -> int:
        value = int(
            workflow.get(
                "last_link_id",
                0,
            )
        ) + 1

        workflow[
            "last_link_id"
        ] = value

        return value

    def _append_input(
        self,
        node: dict,
        name: str,
        type_name: str,
    ) -> int:

        inputs = node.setdefault(
            "inputs",
            [],
        )

        for index, item in enumerate(
            inputs
        ):
            if item.get(
                "name"
            ) == name:
                return index

        inputs.append(
            {
                "name": name,
                "type": type_name,
                "link": None,
            }
        )

        return len(inputs) - 1

    def _add_load_image(
        self,
        workflow: dict,
        filename: str,
    ) -> tuple[int, int]:

        node_id = self._next_id(
            workflow
        )

        node = {
            "id": node_id,
            "type": "LoadImage",
            "pos": [
                -2200,
                6500 + node_id * 20,
            ],
            "size": [
                360,
                320,
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "IMAGE",
                    "type": "IMAGE",
                    "links": [],
                },
                {
                    "name": "MASK",
                    "type": "MASK",
                    "links": None,
                },
            ],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.33.0",
            },
            "widgets_values": [
                filename,
                "image",
            ],
        }

        workflow[
            "nodes"
        ].append(node)

        return (
            node_id,
            0,
        )

    def _add_load_video(
        self,
        workflow: dict,
        filename: str,
    ) -> tuple[int, int, int]:

        node_id = self._next_id(
            workflow
        )

        node = {
            "id": node_id,
            "type": "VHS_LoadVideoPath",
            "pos": [
                -2200,
                6500 + node_id * 20,
            ],
            "size": [
                400,
                500,
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "IMAGE",
                    "type": "IMAGE",
                    "links": [],
                },
                {
                    "name": "frame_count",
                    "type": "INT",
                    "links": None,
                },
                {
                    "name": "audio",
                    "type": "AUDIO",
                    "links": [],
                },
                {
                    "name": "video_info",
                    "type": "VHS_VIDEOINFO",
                    "links": None,
                },
            ],
            "properties": {
                "Node name for S&R":
                    "VHS_LoadVideoPath",
            },
            "widgets_values": {
                "video": filename,
                "force_rate": 24,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        }

        workflow[
            "nodes"
        ].append(node)

        return (
            node_id,
            0,
            2,
        )

    def _add_load_audio(
        self,
        workflow: dict,
        filename: str,
    ) -> tuple[int, int]:

        node_id = self._next_id(
            workflow
        )

        node = {
            "id": node_id,
            "type": "VHS_LoadAudio",
            "pos": [
                -2200,
                6500 + node_id * 20,
            ],
            "size": [
                400,
                300,
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "audio",
                    "type": "AUDIO",
                    "links": [],
                }
            ],
            "properties": {
                "Node name for S&R":
                    "VHS_LoadAudio",
            },
            "widgets_values": {
                "audio_file": filename,
                "seek_seconds": 0,
            },
        }

        workflow[
            "nodes"
        ].append(node)

        return (
            node_id,
            0,
        )

    def _connect(
        self,
        workflow: dict,
        source_id: int,
        source_slot: int,
        target_id: int,
        target_slot: int,
        type_name: str,
    ):
        link_id = self._next_link_id(
            workflow
        )

        workflow.setdefault(
            "links",
            [],
        ).append(
            [
                link_id,
                source_id,
                source_slot,
                target_id,
                target_slot,
                type_name,
            ]
        )

        source = next(
            node
            for node in self._nodes(
                workflow
            )
            if self._node_id(node)
            == source_id
        )

        outputs = source.get(
            "outputs",
            [],
        )

        if (
            0 <= source_slot
            < len(outputs)
        ):
            links = outputs[source_slot].get("links")
            if not isinstance(links, list):
                links = []
                outputs[source_slot]["links"] = links
            links.append(link_id)

        return link_id

    def _clear_template_reference_inputs(
        self,
        workflow: dict,
    ) -> None:
        """Remove static reference-image placeholders from production templates.

        Production reference media is attached at runtime. The checked-in
        workflow templates may contain placeholder LoadImage nodes for visual
        editing, but those nodes must never remain connected in an API graph
        when no matching runtime files are supplied.
        """
        ref_node = self._one(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )
        ref_id = self._node_id(ref_node)

        optional_prefixes = (
            "ref_images.",
            "ref_videos.",
            "ref_video_audios.",
            "ref_audios.",
        )

        placeholder_node_ids: set[int] = set()
        placeholder_link_ids: set[int] = set()

        for node in list(self._nodes(workflow)):
            if node.get("type") != "LoadImage":
                continue

            widgets = node.get("widgets_values", [])
            filename = widgets[0] if isinstance(widgets, list) and widgets else ""
            if not (
                isinstance(filename, str)
                and filename.startswith("__H3_REF_IMAGE_")
            ):
                continue

            node_id = self._node_id(node)
            placeholder_node_ids.add(node_id)

        if placeholder_node_ids:
            for row in workflow.get("links", []):
                if not isinstance(row, list) or len(row) < 6:
                    continue
                if int(row[1]) in placeholder_node_ids:
                    placeholder_link_ids.add(int(row[0]))
                if int(row[3]) == ref_id:
                    input_name = None
                    try:
                        input_name = ref_node.get("inputs", [])[int(row[4])].get("name")
                    except (IndexError, TypeError, ValueError, AttributeError):
                        pass
                    if isinstance(input_name, str) and input_name.startswith(optional_prefixes):
                        placeholder_link_ids.add(int(row[0]))

            workflow["nodes"] = [
                node
                for node in self._nodes(workflow)
                if self._node_id(node) not in placeholder_node_ids
            ]

        if placeholder_link_ids:
            workflow["links"] = [
                row
                for row in workflow.get("links", [])
                if not (
                    isinstance(row, list)
                    and row
                    and int(row[0]) in placeholder_link_ids
                )
            ]

            for node in self._nodes(workflow):
                for output in node.get("outputs", []) or []:
                    links = output.get("links") if isinstance(output, dict) else None
                    if isinstance(links, list):
                        output["links"] = [
                            int(link)
                            for link in links
                            if int(link) not in placeholder_link_ids
                        ]

        # Any runtime-optional inputs left in the template start unbound.
        for node in self._nodes(workflow):
            if self._node_id(node) != ref_id:
                continue
            for item in node.get("inputs", []) or []:
                name = item.get("name") if isinstance(item, dict) else None
                if isinstance(name, str) and name.startswith(optional_prefixes):
                    item["link"] = None

    @staticmethod
    def _remove_placeholder_references(
        workflow: dict,
    ) -> None:
        """Remove template-only H3 reference LoadImage nodes and links."""
        placeholder_ids: set[int] = set()

        for node in workflow.get("nodes", []):
            if node.get("type") != "LoadImage":
                continue

            widgets = node.get("widgets_values", [])
            filename = (
                widgets[0]
                if isinstance(widgets, list) and widgets
                else ""
            )

            if (
                isinstance(filename, str)
                and filename.startswith("__H3_REF_IMAGE_")
            ):
                placeholder_ids.add(int(node["id"]))

        if not placeholder_ids:
            return

        link_ids: set[int] = set()

        for row in workflow.get("links", []):
            if (
                isinstance(row, list)
                and len(row) >= 6
                and int(row[1]) in placeholder_ids
            ):
                link_ids.add(int(row[0]))

        workflow["links"] = [
            row
            for row in workflow.get("links", [])
            if not (
                isinstance(row, list)
                and len(row) >= 6
                and (
                    int(row[0]) in link_ids
                    or int(row[1]) in placeholder_ids
                    or int(row[3]) in placeholder_ids
                )
            )
        ]

        for node in workflow.get("nodes", []):
            for item in node.get("inputs", []):
                link = item.get("link")
                if link is not None and int(link) in link_ids:
                    item["link"] = None

            for output in node.get("outputs", []):
                links = output.get("links")
                if isinstance(links, list):
                    output["links"] = [
                        link
                        for link in links
                        if int(link) not in link_ids
                    ]

        workflow["nodes"] = [
            node
            for node in workflow.get("nodes", [])
            if int(node.get("id", -1)) not in placeholder_ids
        ]

    def _connect_media(
        self,
        workflow: dict,
        reference_images: list[str],
        reference_videos: list[str],
        reference_audio: list[str],
    ):
        self._remove_placeholder_references(
            workflow
        )
        self._clear_template_reference_inputs(
            workflow
        )

        if (
            len(reference_images)
            + len(reference_videos)
            + len(reference_audio)
            > H3_MAX_REFERENCE_FILES
        ):
            raise ValueError(
                "H3 allows at most 12 total references."
            )

        if len(reference_images) > H3_MAX_REFERENCE_IMAGES:
            raise ValueError(
                "H3 allows at most 9 image references."
            )

        if len(reference_videos) > H3_MAX_REFERENCE_VIDEOS:
            raise ValueError(
                "H3 allows at most 3 video references."
            )

        if len(reference_audio) > H3_MAX_REFERENCE_AUDIO:
            raise ValueError(
                "H3 allows at most 3 audio references."
            )

        ref_node = self._one(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        ref_id = self._node_id(
            ref_node
        )

        # Images
        for index, filename in enumerate(
            reference_images
        ):
            node_id, output_slot = (
                self._add_load_image(
                    workflow,
                    filename,
                )
            )

            target_slot = self._append_input(
                ref_node,
                f"ref_images.ref_image_{index}",
                "IMAGE",
            )

            link_id = self._connect(
                workflow,
                node_id,
                output_slot,
                ref_id,
                target_slot,
                "IMAGE",
            )

            ref_node[
                "inputs"
            ][target_slot][
                "link"
            ] = link_id

        # Videos + paired audio.
        for index, filename in enumerate(
            reference_videos
        ):
            (
                node_id,
                image_slot,
                audio_slot,
            ) = self._add_load_video(
                workflow,
                filename,
            )

            video_slot = self._append_input(
                ref_node,
                f"ref_videos.ref_video_{index}",
                "IMAGE",
            )

            video_link = self._connect(
                workflow,
                node_id,
                image_slot,
                ref_id,
                video_slot,
                "IMAGE",
            )

            ref_node[
                "inputs"
            ][video_slot][
                "link"
            ] = video_link

            audio_input = self._append_input(
                ref_node,
                f"ref_video_audios.ref_video_audio_{index}",
                "AUDIO",
            )

            audio_link = self._connect(
                workflow,
                node_id,
                audio_slot,
                ref_id,
                audio_input,
                "AUDIO",
            )

            ref_node[
                "inputs"
            ][audio_input][
                "link"
            ] = audio_link

        # Standalone audio.
        for index, filename in enumerate(
            reference_audio
        ):
            node_id, output_slot = (
                self._add_load_audio(
                    workflow,
                    filename,
                )
            )

            audio_slot = self._append_input(
                ref_node,
                f"ref_audios.ref_audio_{index}",
                "AUDIO",
            )

            audio_link = self._connect(
                workflow,
                node_id,
                output_slot,
                ref_id,
                audio_slot,
                "AUDIO",
            )

            ref_node[
                "inputs"
            ][audio_slot][
                "link"
            ] = audio_link

        return (
            len(reference_images),
            len(reference_videos),
            len(reference_audio),
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_models(
        self,
        workflow: dict,
        mode: str,
    ):
        unets = self._find(
            workflow,
            "UNETLoader",
        )

        if len(unets) != 1:
            raise RuntimeError(
                f"{mode}: invalid UNETLoader count."
            )

        unet_widgets = self._widgets(
            unets[0]
        )

        if (
            not unet_widgets
            or unet_widgets[0]
            != H3_REF2VA_MODEL
        ):
            raise RuntimeError(
                f"{mode}: wrong Ref2VA model."
            )

        clips = self._find(
            workflow,
            "CLIPLoader",
        )

        if len(clips) != 1:
            raise RuntimeError(
                f"{mode}: invalid CLIPLoader count."
            )

        clip_widgets = self._widgets(
            clips[0]
        )

        if (
            not clip_widgets
            or clip_widgets[0]
            != H3_TEXT_ENCODER
        ):
            raise RuntimeError(
                f"{mode}: wrong Qwen3-VL encoder."
            )

        vae_values = [
            self._widgets(node)[0]
            for node in self._find(
                workflow,
                "VAELoader",
            )
            if self._widgets(node)
        ]

        if H3_VIDEO_VAE not in vae_values:
            raise RuntimeError(
                f"{mode}: video VAE missing."
            )

        if H3_AUDIO_VAE not in vae_values:
            raise RuntimeError(
                f"{mode}: audio VAE missing."
            )

    def _validate_mode(
        self,
        workflow: dict,
        mode: str,
    ):
        self._validate_models(
            workflow,
            mode,
        )

        if mode == "turbo_ref2va":
            lora = self._one(
                workflow,
                "MiniMaxH3TurboLoRA",
            )

            widgets = self._widgets(
                lora
            )

            if (
                not widgets
                or widgets[0]
                != H3_TURBO_LORA
            ):
                raise RuntimeError(
                    "Turbo is not using the "
                    "locked Step600 LoRA."
                )

        elif mode == "upscale":
            node = self._one(
                workflow,
                "MMH3LatentUpscaleWithModelParams",
            )

            widgets = self._widgets(
                node
            )

            if (
                not widgets
                or widgets[0]
                != H3_LATENT_UPSCALER_3D
            ):
                raise RuntimeError(
                    "Upscale workflow is not using "
                    "the locked H3 3D upscaler."
                )

    # ============================================================
    # PUBLIC BUILD
    # ============================================================

    def build(
        self,
        *,
        mode: str,
        prompt: str,
        seed: int,
        turbo_steps: int = TURBO_STEPS,
        reference_images: list[str] | None = None,
        reference_videos: list[str] | None = None,
        reference_audio: list[str] | None = None,
        width: int = H3_WIDTH,
        height: int = H3_HEIGHT,
        duration_seconds: float = 5.2,
        ref_image_size: str | None = None,
        context_ir: dict | None = None,
    ):
        workflow = self.load(
            mode
        )

        reference_images = list(
            reference_images or []
        )

        reference_videos = list(
            reference_videos or []
        )

        reference_audio = list(
            reference_audio or []
        )

        self._validate_mode(
            workflow,
            mode,
        )

        # Turbo on the T4 uses the LoRA merge path explicitly.
        # Do not rely on the workflow UI default.
        if mode == "turbo_ref2va":
            turbo_lora = self._one(
                workflow,
                "MiniMaxH3TurboLoRA",
            )
            widgets = self._widgets(turbo_lora)
            if len(widgets) < 3:
                raise RuntimeError(
                    "Turbo LoRA node is missing the low_vram control."
                )
            widgets[2] = True

        if context_ir is not None:
            from pipeline.context_ir import H3ContextIRCompiler
            H3ContextIRCompiler.validate(context_ir)
            if str(context_ir.get("mode", "")).strip().lower() != "ref2va":
                raise ValueError("Production H3 builder accepts Ref2VA Context-IR only.")
            prompt = H3ContextIRCompiler.prompt(context_ir)

        if mode not in {"ref2va", "turbo_ref2va", "upscale"}:
            raise ValueError(f"Unsupported production workflow mode: {mode}")

        if mode == "turbo_ref2va":
            if int(turbo_steps) != 8:
                raise ValueError(
                    "Production Turbo is locked to 8 steps."
                )

        self._set_prompt(
            workflow,
            prompt,
        )

        self._set_seed(
            workflow,
            seed,
        )

        self._set_resolution(
            workflow,
            width,
            height,
        )

        self._set_duration(
            workflow,
            duration_seconds,
        )

        self._set_ref_image_size(
            workflow,
            ref_image_size,
        )

        if mode == "upscale":
            self._set_upscale_target(
                workflow,
                UPSCALE_WIDTH,
                UPSCALE_HEIGHT,
            )

        image_count, video_count, audio_count = (
            self._connect_media(
                workflow,
                reference_images,
                reference_videos,
                reference_audio,
            )
        )

        # Tell H3 which reference categories it must associate
        # with the prompt. H3's Ref2VA path is multimodal and
        # uses ordered reference tags.
        tags = []

        for index in range(
            image_count
        ):
            tags.append(
                f"<Picture {index + 1}>"
            )

        for index in range(
            video_count
        ):
            tags.append(
                f"<Video {index + 1}>"
            )

        for index in range(
            audio_count
        ):
            tags.append(
                f"<Audio {index + 1}>"
            )

        if tags:
            prompt_prefix = (
                "REFERENCE INPUTS: "
                + ", ".join(tags)
                + ". Use each reference only for "
                "the role described in the prompt.\n\n"
            )

            self._set_prompt(
                workflow,
                prompt_prefix + prompt,
            )

        from planner.config import RUNTIME

        if bool(
            RUNTIME.get("h3_optimization", {}).get(
                "memory_optimization",
                True,
            )
        ):
            source_type = (
                "MiniMaxH3TurboLoRA"
                if mode == "turbo_ref2va"
                else "UNETLoader"
            )
            self._inject_memory_optimization(
                workflow,
                source_node_type=source_type,
            )

        return self.client.convert_workflow(
            workflow
        )
