from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

WORKFLOWS = (
    ROOT
    / "workflows"
    / "MiniMax-H3"
)

BASE = {
    "H3_Extend_Take.json",
    "H3_HardMode_Chained.json",
    "H3_HardMode_R2V.json",
    "H3_Keyframes.json",
    "H3_Seamless_Chain_CORE.json",
    "H3_Seamless_Chain_v2.json",
}

TURBO = {
    "H3_Turbo_I2V.json",
    "H3_Turbo_Ref2V.json",
    "H3_Turbo_T2V.json",
}

MANUAL = (
    WORKFLOWS
    / "base"
    / "H3_Ref2VA_Memory_API.json"
)


def validate_python():

    for path in ROOT.rglob(
        "*.py"
    ):

        if ".git" in path.parts:
            continue

        if "ComfyUI" in path.parts:
            continue

        if "__pycache__" in path.parts:
            continue

        source = path.read_text(
            encoding="utf-8"
        )

        ast.parse(
            source,
            filename=str(path),
        )


def validate_json_file(
    path,
):

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Workflow root is not object: {path}"
        )


def validate_workflows():

    if MANUAL.exists():

        raise RuntimeError(
            "Remove manual workflow: "
            f"{MANUAL}"
        )

    actual_base = {
        path.name
        for path in (
            WORKFLOWS
            / "base"
        ).glob(
            "*.json"
        )
    }

    actual_turbo = {
        path.name
        for path in (
            WORKFLOWS
            / "turbo"
        ).glob(
            "*.json"
        )
    }

    if actual_base != BASE:

        raise RuntimeError(
            "Base workflow set does not match expected "
            "production set.\n"
            f"Expected: {sorted(BASE)}\n"
            f"Actual: {sorted(actual_base)}"
        )

    if actual_turbo != TURBO:

        raise RuntimeError(
            "Turbo workflow set does not match expected "
            "production set.\n"
            f"Expected: {sorted(TURBO)}\n"
            f"Actual: {sorted(actual_turbo)}"
        )

    for name in BASE:
        validate_json_file(
            WORKFLOWS
            / "base"
            / name
        )

    for name in TURBO:
        validate_json_file(
            WORKFLOWS
            / "turbo"
            / name
        )


def validate_no_wrong_workflow_paths():

    builder = (
        ROOT
        / "execution"
        / "h3_workflow_builder.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = [
        "canonical/H3_Multishot_AIO.json",
        "canonical/H3_Multishot_MEMORY.json",
        "H3_Ref2VA_Memory_API.json",
    ]

    for value in forbidden:

        if value in builder:

            raise RuntimeError(
                "Production builder still references "
                f"forbidden workflow: {value}"
            )


def main():

    validate_python()
    validate_workflows()
    validate_no_wrong_workflow_paths()

    print(
        "MiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()
