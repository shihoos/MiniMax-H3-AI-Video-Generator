from __future__ import annotations

import copy
import json
from pathlib import Path

from execution.h3_workflow_builder import (
    H3WorkflowBuilder,
)
from planner.config import (
    H3_HEIGHT,
    H3_WIDTH,
    UPSCALE_HEIGHT,
    UPSCALE_WIDTH,
)


class H3UpscaledWorkflowBuilder(
    H3WorkflowBuilder
):

    UPSCALE_NODE_TYPES = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }

    def _disconnect_target(
        self,
        workflow: dict,
        target_id: int,
        target_slot: int,
    ) -> None:

        removed = set()

        new_links = []

        for row in workflow.get(
            "links",
            [],
        ):

            if (
                isinstance(row, list)
                and len(row) >= 6
                and int(row[3]) == target_id
                and int(row[4]) == target_slot
            ):
                removed.add(
                    int(row[0])
                )
                continue

            new_links.append(
                row
            )

        workflow[
            "links"
        ] = new_links

        for node in self._nodes(
            workflow
        ):

            for output in node.get(
                "outputs",
                [],
            ):

                links = output.get(
                    "links"
                )

                if not isinstance(
                    links,
                    list,
                ):
                    continue

                output[
                    "links"
                ] = [
                    link
                    for link in links
                    if int(link)
                    not in removed
                ]

    def _connect_graph(
        self,
        workflow: dict,
        source_id: int,
        source_slot: int,
        target_id: int,
        target_name: str,
        type_name: str,
    ) -> int:

        target = next(
            node
            for node in self._nodes(
                workflow
            )
            if self._node_id(node)
            == target_id
        )

        target_slot = self._append_input(
            target,
            target_name,
            type_name,
        )

        self._disconnect_target(
            workflow,
            target_id,
            target_slot,
        )

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

        outputs = source.setdefault(
            "outputs",
            [],
        )

        while (
            len(outputs)
            <= source_slot
        ):
            outputs.append(
                {
                    "name": f"output_{len(outputs)}",
                    "type": type_name,
                    "links": [],
                }
            )

        output_links = (
            outputs[source_slot].get(
                "links"
            )
        )

        if isinstance(
            output_links,
            list,
        ):
            output_links.append(
                link_id
            )

        target[
            "inputs"
        ][target_slot][
            "link"
        ] = link_id

        return link_id

    def _clone_upscale_nodes(
        self,
        workflow: dict,
    ) -> dict[str, int]:

        upscale_workflow = self.load(
            "upscale"
        )

        source_nodes = {
            node.get("type"): node
            for node
            in self._nodes(
                upscale_workflow
            )
            if node.get("type")
            in self.UPSCALE_NODE_TYPES
        }

        missing = (
            self.UPSCALE_NODE_TYPES
            - set(
                source_nodes
            )
        )

        if missing:
            raise RuntimeError(
                "Upscale workflow is missing nodes: "
                + ", ".join(
                    sorted(missing)
                )
            )

        cloned = {}

        for node_type in (
            "MMH3LatentUpscaleWithModelParams",
            "MMH3TemporalSplitParams",
            "MMH3SpatialSplitParams",
            "MMH3UltimateUpscale",
        ):

            node = copy.deepcopy(
                source_nodes[
                    node_type
                ]
            )

            node_id = self._next_id(
                workflow
            )

            node[
                "id"
            ] = node_id

            node[
                "order"
            ] = max(
                [
                    int(
                        value.get(
                            "order",
                            0,
                        )
                    )
                    for value
                    in self._nodes(
                        workflow
                    )
                    if isinstance(
                        value,
                        dict,
                    )
                ]
                or [0]
            ) + 1

            for item in node.get(
                "inputs",
                [],
            ):
                item[
                    "link"
                ] = None

            for output in node.get(
                "outputs",
                [],
            ):
                links = output.get(
                    "links"
                )

                if isinstance(
                    links,
                    list,
                ):
                    output[
                        "links"
                    ] = []

            workflow[
                "nodes"
            ].append(
                node
            )

            cloned[
                node_type
            ] = node_id

        # Width / height are direct widgets in this
        # parameter node. Remove stale linked expressions.
        param = next(
            node
            for node in self._nodes(
                workflow
            )
            if self._node_id(node)
            == cloned[
                "MMH3LatentUpscaleWithModelParams"
            ]
        )

        widgets = self._widgets(
            param
        )

        if len(widgets) >= 3:
            widgets[1] = int(
                UPSCALE_WIDTH
            )
            widgets[2] = int(
                UPSCALE_HEIGHT
            )

        return cloned

    def _find_generation_nodes(
        self,
        workflow: dict,
    ) -> dict[str, dict]:

        required = {
            "UNETLoader",
            "MiniMaxH3ReferenceToVideo",
            "RandomNoise",
            "KSamplerSelect",
            "BasicScheduler",
            "VAEDecode",
        }

        found = {}

        for node in self._nodes(
            workflow
        ):
            node_type = node.get(
                "type"
            )

            if node_type in required:
                found[
                    node_type
                ] = node

        missing = (
            required
            - set(found)
        )

        if missing:
            raise RuntimeError(
                "Cannot build combined H3 upscale graph; "
                "generation workflow is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        sampler_advanced = self._find(
            workflow,
            "SamplerCustomAdvanced",
        )

        turbo_sampler = self._find(
            workflow,
            "MiniMaxH3TurboSampler",
        )

        if sampler_advanced:
            found[
                "LATENT_SOURCE"
            ] = sampler_advanced[0]

        elif turbo_sampler:
            found[
                "LATENT_SOURCE"
            ] = turbo_sampler[0]

        else:
            raise RuntimeError(
                "Combined H3 upscale graph has no "
                "SamplerCustomAdvanced or "
                "MiniMaxH3TurboSampler."
            )

        return found

    def build_upscaled(
        self,
        *,
        generation_mode: str,
        prompt: str,
        seed: int,
        turbo_steps: int = 8,
        reference_images: list[str] | None = None,
        reference_videos: list[str] | None = None,
        reference_audio: list[str] | None = None,
        width: int = H3_WIDTH,
        height: int = H3_HEIGHT,
        duration_seconds: float = 5.2,
        ref_image_size: str | None = None,
        context_ir: dict | None = None,
    ):

        if generation_mode not in {
            "ref2v",
            "turbo_ref2v",
        }:
            raise ValueError(
                "Combined upscale requires "
                "ref2v or turbo_ref2v generation mode."
            )

        workflow = self.load(
            generation_mode
        )

        self._validate_mode(
            workflow,
            generation_mode,
        )

        if context_ir is not None:
            from pipeline.context_ir import H3ContextIRCompiler
            H3ContextIRCompiler.validate(context_ir)
            from planner.config import H3_CONTEXT_IR_VERSION
            if int(context_ir.get("version", 0) or 0) != int(H3_CONTEXT_IR_VERSION):
                raise ValueError(
                    f"Context-IR version mismatch: got {context_ir.get('version')}, "
                    f"configured {H3_CONTEXT_IR_VERSION}."
                )
            if str(context_ir.get("mode", "")).strip().lower() != "ref2va":
                raise ValueError("This production inventory is Ref2VA-only; Context-IR mode must be ref2va.")
            prompt = H3ContextIRCompiler.prompt(context_ir)

        if (
            generation_mode
            == "turbo_ref2v"
            and int(turbo_steps) != 8
        ):
            raise ValueError(
                "Turbo is locked to 8 steps."
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

        (
            image_count,
            video_count,
            audio_count,
        ) = self._connect_media(
            workflow,
            reference_images,
            reference_videos,
            reference_audio,
        )

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

            prompt = (
                "REFERENCE INPUTS: "
                + ", ".join(tags)
                + ". Use each reference only for "
                  "the role described in the prompt.\n\n"
                + prompt
            )

            self._set_prompt(
                workflow,
                prompt,
            )

        generation_nodes = (
            self._find_generation_nodes(
                workflow
            )
        )

        cloned = (
            self._clone_upscale_nodes(
                workflow
            )
        )

        ultimate_id = cloned[
            "MMH3UltimateUpscale"
        ]

        latent_param_id = cloned[
            "MMH3LatentUpscaleWithModelParams"
        ]

        temporal_param_id = cloned[
            "MMH3TemporalSplitParams"
        ]

        spatial_param_id = cloned[
            "MMH3SpatialSplitParams"
        ]

        # Existing generation model.
        self._connect_graph(
            workflow,
            self._node_id(
                generation_nodes[
                    "UNETLoader"
                ]
            ),
            0,
            ultimate_id,
            "model",
            "MODEL",
        )

        # H3 positive conditioning.
        self._connect_graph(
            workflow,
            self._node_id(
                generation_nodes[
                    "MiniMaxH3ReferenceToVideo"
                ]
            ),
            0,
            ultimate_id,
            "conditioning",
            "CONDITIONING",
        )

        # Original denoised H3 AV latent enters the
        # two-stage upscale node.
        latent_source = generation_nodes[
            "LATENT_SOURCE"
        ]

        self._connect_graph(
            workflow,
            self._node_id(
                latent_source
            ),
            0,
            ultimate_id,
            "latent",
            "LATENT",
        )

        # Reuse the same noise/sampler/sigma schedule so the
        # upscale refinement remains tied to the original H3
        # generation configuration.
        self._connect_graph(
            workflow,
            self._node_id(
                generation_nodes[
                    "RandomNoise"
                ]
            ),
            0,
            ultimate_id,
            "noise",
            "NOISE",
        )

        self._connect_graph(
            workflow,
            self._node_id(
                generation_nodes[
                    "KSamplerSelect"
                ]
            ),
            0,
            ultimate_id,
            "sampler",
            "SAMPLER",
        )

        self._connect_graph(
            workflow,
            self._node_id(
                generation_nodes[
                    "BasicScheduler"
                ]
            ),
            0,
            ultimate_id,
            "sigmas",
            "SIGMAS",
        )

        self._connect_graph(
            workflow,
            latent_param_id,
            0,
            ultimate_id,
            "latent_upscale_param",
            "H3_UPSCALE_PARAM",
        )

        self._connect_graph(
            workflow,
            temporal_param_id,
            0,
            ultimate_id,
            "temporal_split_param",
            "H3_TEMPORAL_PARAM",
        )

        self._connect_graph(
            workflow,
            spatial_param_id,
            0,
            ultimate_id,
            "spatial_split_param",
            "H3_SPATIAL_PARAM",
        )

        # The original SamplerCustomAdvanced output remains the
        # source for VAEDecodeAudio, preserving native H3 audio.
        #
        # Only video decode is redirected through Ultimate Upscale.
        video_decode = generation_nodes[
            "VAEDecode"
        ]

        video_decode_slot = next(
            (
                index
                for index, item
                in enumerate(
                    video_decode.get(
                        "inputs",
                        [],
                    )
                )
                if item.get(
                    "name"
                ) == "samples"
            ),
            None,
        )

        if video_decode_slot is None:
            raise RuntimeError(
                "VAEDecode samples input not found."
            )

        self._connect_graph(
            workflow,
            ultimate_id,
            0,
            self._node_id(
                video_decode
            ),
            "samples",
            "LATENT",
        )

        # CRITICAL: VAEDecodeAudio must consume the original H3
        # audio-bearing latent emitted by the generation sampler. The
        # UltimateUpscale node produces a video latent and is not a valid
        # source for VAEDecodeAudio. The source workflow historically had
        # an accidental 150 -> 121 connection, so enforce the correct edge
        # in the generated graph rather than relying on the static JSON.
        audio_decode = next(
            (
                node
                for node in self._nodes(workflow)
                if node.get("type") == "VAEDecodeAudio"
            ),
            None,
        )
        if audio_decode is None:
            raise RuntimeError(
                "Combined H3 upscale graph is missing VAEDecodeAudio."
            )

        audio_input = next(
            (
                item
                for item in audio_decode.get("inputs", [])
                if item.get("name") == "samples"
            ),
            None,
        )
        if audio_input is None:
            raise RuntimeError(
                "VAEDecodeAudio samples input not found."
            )

        self._connect_graph(
            workflow,
            self._node_id(latent_source),
            0,
            self._node_id(audio_decode),
            "samples",
            "LATENT",
        )

        # Defensive postcondition: video and audio decoders must now have
        # different sources. If this ever regresses, fail before submitting
        # an invalid graph to ComfyUI.
        video_samples_link = next(
            item.get("link")
            for item in video_decode.get("inputs", [])
            if item.get("name") == "samples"
        )
        audio_samples_link = next(
            item.get("link")
            for item in audio_decode.get("inputs", [])
            if item.get("name") == "samples"
        )
        links_by_id = {
            int(row[0]): row
            for row in workflow.get("links", [])
            if isinstance(row, list) and len(row) >= 6
        }
        video_edge = links_by_id.get(int(video_samples_link)) if video_samples_link is not None else None
        audio_edge = links_by_id.get(int(audio_samples_link)) if audio_samples_link is not None else None
        if not video_edge or int(video_edge[1]) != int(ultimate_id):
            raise RuntimeError(
                "Upscale postcondition failed: VAEDecode is not connected to MMH3UltimateUpscale."
            )
        if not audio_edge or int(audio_edge[1]) != self._node_id(latent_source):
            raise RuntimeError(
                "Upscale postcondition failed: VAEDecodeAudio is not connected to the original latent source."
            )

        # Store explicit metadata for the runtime and UI.
        workflow[
            "_h3_combined_upscale"
        ] = {
            "generation_mode": generation_mode,
            "generation_width": int(width),
            "generation_height": int(height),
            "upscale_width": int(
                UPSCALE_WIDTH
            ),
            "upscale_height": int(
                UPSCALE_HEIGHT
            ),
            "final_delivery_width": 1280,
            "final_delivery_height": 720,
            "audio_source": (
                "original_generation_latent"
            ),
            "video_source": (
                "MMH3UltimateUpscale_output"
            ),
        }

        return self.client.convert_workflow(
            workflow
        )
