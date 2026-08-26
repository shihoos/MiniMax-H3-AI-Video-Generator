from __future__ import annotations

import tempfile
from pathlib import Path

from execution.shot_executor import (
    ShotExecutor,
)


class FakeClient:

    @staticmethod
    def convert_workflow(
        workflow,
    ):
        return workflow


def main():

    with tempfile.TemporaryDirectory(
        prefix="h3_reference_test_"
    ) as raw_root:

        root = Path(
            raw_root
        )

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

        assert returned == expected, (
            f"Expected relative ComfyUI path "
            f"{expected!r}, got {returned!r}"
        )

        copied = (
            comfy_input
            / returned
        )

        assert copied.is_file(), (
            f"Copied reference does not exist: "
            f"{copied}"
        )

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

        assert node_id == 1
        assert output_slot == 0

        node = workflow[
            "nodes"
        ][0]

        assert (
            node[
                "widgets_values"
            ][0]
            == expected
        ), (
            "Workflow LoadImage path is not "
            "the same relative path returned "
            "by ShotExecutor.copy_input()."
        )

        print(
            "Reference media path wiring PASSED."
        )


if __name__ == "__main__":
    main()
