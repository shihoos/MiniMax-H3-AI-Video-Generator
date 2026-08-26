from __future__ import annotations

import tempfile
from pathlib import Path

from execution.shot_executor import (
    ShotExecutor,
)
from pipeline.h3_scene_continuity import (
    H3SceneContinuity,
)
from pipeline.identity_anchor_store import (
    IdentityAnchorStore,
)


class FakeClient:

    @staticmethod
    def convert_workflow(
        workflow,
    ):
        return workflow


def main() -> None:

    with tempfile.TemporaryDirectory(
        prefix="h3_reference_test_"
    ) as raw_root:

        root = Path(
            raw_root
        ).resolve()

        comfy_input = (
            root
            / "ComfyUI"
            / "input"
        )

        source_dir = (
            root
            / "source"
        )

        source_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source = (
            source_dir
            / "reference.png"
        )

        source.write_bytes(
            b"fake-png-data"
        )

        # ----------------------------------------------------
        # REFERENCE FILE PATH CONTRACT
        # ----------------------------------------------------

        executor = ShotExecutor(
            comfy_client=FakeClient(),
            project_root=root,
            comfy_input_dir=(
                comfy_input
                / "production_test"
                / "gpu_0"
                / "scene_001"
            ),
        )

        returned = (
            executor.copy_input(
                source,
                "shot_001_image_1",
            )
        )

        expected = (
            "production_test/"
            "gpu_0/"
            "scene_001/"
            "shot_001_image_1_reference.png"
        )

        if returned != expected:

            raise RuntimeError(
                "Reference path contract failed.\n"
                f"Expected: {expected!r}\n"
                f"Received: {returned!r}"
            )

        copied = (
            comfy_input
            / returned
        )

        if not copied.is_file():

            raise RuntimeError(
                "Reference file was not copied "
                "to the expected ComfyUI input path.\n"
                f"Expected file: {copied}"
            )

        if copied.stat().st_size <= 0:

            raise RuntimeError(
                "Copied reference file is empty.\n"
                f"File: {copied}"
            )

        # ----------------------------------------------------
        # WORKFLOW LOADER PATH CONTRACT
        # ----------------------------------------------------

        workflow = {
            "nodes": [],
            "links": [],
            "last_node_id": 0,
            "last_link_id": 0,
        }

        node_id, output_slot = (
            executor.builder._add_load_image(
                workflow,
                returned,
            )
        )

        if node_id != 1:

            raise RuntimeError(
                "Unexpected LoadImage node ID.\n"
                f"Expected: 1\n"
                f"Received: {node_id}"
            )

        if output_slot != 0:

            raise RuntimeError(
                "Unexpected LoadImage output slot.\n"
                f"Expected: 0\n"
                f"Received: {output_slot}"
            )

        nodes = workflow.get(
            "nodes",
            [],
        )

        if len(nodes) != 1:

            raise RuntimeError(
                "Expected exactly one LoadImage node "
                "in the test workflow."
            )

        node = nodes[0]

        widgets = node.get(
            "widgets_values",
            [],
        )

        if not widgets:

            raise RuntimeError(
                "LoadImage node has no widgets_values."
            )

        workflow_path = (
            widgets[0]
        )

        if workflow_path != expected:

            raise RuntimeError(
                "Workflow LoadImage path does not "
                "match the copied ComfyUI-relative path.\n"
                f"Expected: {expected!r}\n"
                f"Received: {workflow_path!r}"
            )

        # ----------------------------------------------------
        # PRODUCTION ISOLATION CONTRACT
        # ----------------------------------------------------

        continuity_a = H3SceneContinuity(
            root,
            production_id="production_a",
        )

        continuity_b = H3SceneContinuity(
            root,
            production_id="production_b",
        )

        identity_a = IdentityAnchorStore(
            root,
            production_id="production_a",
        )

        identity_b = IdentityAnchorStore(
            root,
            production_id="production_b",
        )

        if (
            continuity_a.root
            == continuity_b.root
        ):

            raise RuntimeError(
                "Continuity storage is not "
                "production-isolated."
            )

        if (
            identity_a.root
            == identity_b.root
        ):

            raise RuntimeError(
                "Identity anchor storage is not "
                "production-isolated."
            )

        if (
            "production_a"
            not in str(
                continuity_a.root
            )
        ):

            raise RuntimeError(
                "Production A continuity path does not "
                "contain production_a."
            )

        if (
            "production_b"
            not in str(
                continuity_b.root
            )
        ):

            raise RuntimeError(
                "Production B continuity path does not "
                "contain production_b."
            )

        if (
            "production_a"
            not in str(
                identity_a.root
            )
        ):

            raise RuntimeError(
                "Production A identity path does not "
                "contain production_a."
            )

        if (
            "production_b"
            not in str(
                identity_b.root
            )
        ):

            raise RuntimeError(
                "Production B identity path does not "
                "contain production_b."
            )

        print(
            "Reference media path wiring PASSED."
        )

        print(
            "Production isolation wiring PASSED."
        )

        print(
            "Reference and production wiring validation PASSED."
        )


if __name__ == "__main__":
    main()
