#!/usr/bin/env python3
"""
Prove that ``data_convert.normalize_quadrant`` never touches box geometry --
only remaps ``category_id`` by name.

table:quadrant_box_geometry (added for the tier-0 boxes) shows the *converted*
boxes are tight tooth-row boxes, not literal quarter-image regions. It does
not by itself rule out the concern that the conversion step could have picked
a convention different from the one the original paper's own tier-0
evaluation used. This script closes that gap directly, on the code rather
than on the numbers: it feeds ``normalize_quadrant`` a synthetic quadrant.json
built with the exact real trap this project already found (quadrant's own
category id 0 is named "2", while ``categories_1`` id 0 is named "1" -- see
the module docstring), and asserts every annotation's ``bbox`` field is
byte-identical before and after conversion, in original list order.

If this ever fails, the converter has started inventing or altering box
geometry and every downstream tier-0 number is suspect. If it passes, the
tier-0 boxes visible in ``table:quadrant_box_geometry`` are exactly the boxes
in the public release -- there is no separate "flattening convention" for
geometry to disagree with the original paper about, because geometry is
never touched; only which integer category id a box's label maps to is
remapped, and by name, not by position.

No dataset download needed -- this is a pure property of the conversion
code, checked with a minimal synthetic fixture matching the real schema.

Run with:

    python tools/verify_quadrant_geometry_passthrough.py
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import data_convert  # noqa: E402


def build_synthetic_quadrant_json() -> dict:
    """
    Mirrors the real ``quadrant/train_quadrant.json`` schema: flat
    ``categories``/``category_id``, and the real id-swap trap where
    quadrant's own id 0 is named "2" (not "1").
    """
    return {
        "images": [{"id": 1, "file_name": "synthetic_0.png", "width": 1000, "height": 800}],
        "categories": [
            {"id": 0, "name": "2"},
            {"id": 1, "name": "1"},
            {"id": 2, "name": "4"},
            {"id": 3, "name": "3"},
        ],
        "annotations": [
            {"id": 100, "image_id": 1, "category_id": 0,
             "bbox": [10.5, 20.25, 300.75, 250.125], "area": 75262.6, "iscrowd": 0},
            {"id": 101, "image_id": 1, "category_id": 1,
             "bbox": [500.0, 0.0, 480.333, 260.667], "area": 125198.4, "iscrowd": 0},
            {"id": 102, "image_id": 1, "category_id": 2,
             "bbox": [520.1, 400.9, 470.0, 390.0], "area": 183300.0, "iscrowd": 0},
            {"id": 103, "image_id": 1, "category_id": 3,
             "bbox": [0.0, 410.0, 290.0, 380.0], "area": 110200.0, "iscrowd": 0},
        ],
    }


def build_canonical_categories() -> dict:
    return {
        "categories_1": [{"id": 0, "name": "1"}, {"id": 1, "name": "2"},
                          {"id": 2, "name": "3"}, {"id": 3, "name": "4"}],
        "categories_2": [{"id": i, "name": str(i)} for i in range(8)],
        "categories_3": [{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"},
                          {"id": 2, "name": "Periapical Lesion"}, {"id": 3, "name": "Deep Caries"}],
    }


def main() -> int:
    raw = build_synthetic_quadrant_json()
    canonical = build_canonical_categories()
    before = copy.deepcopy(raw["annotations"])

    converted = data_convert.normalize_quadrant(
        raw, canonical["categories_1"], canonical["categories_2"], canonical["categories_3"])

    failures = []

    if len(converted["annotations"]) != len(before):
        failures.append("annotation count changed: {} -> {}".format(
            len(before), len(converted["annotations"])))

    for original, out in zip(before, converted["annotations"]):
        if original["bbox"] != out["bbox"]:
            failures.append("bbox changed for annotation {}: {} -> {}".format(
                original["id"], original["bbox"], out["bbox"]))
        if original.get("area") != out.get("area"):
            failures.append("area changed for annotation {}: {} -> {}".format(
                original["id"], original.get("area"), out.get("area")))

    # The id-remap-by-name trap: quadrant's own id 0 is named "2", so it must
    # land on categories_1's id for name "2", which is 1 -- NOT id 0.
    name_by_new_id = {c["id"]: c["name"] for c in canonical["categories_1"]}
    for original, out in zip(before, converted["annotations"]):
        old_name = {c["id"]: c["name"] for c in raw["categories"]}[original["category_id"]]
        new_name = name_by_new_id[out["category_id_1"]]
        if old_name != new_name:
            failures.append(
                "category remap broke name preservation for annotation {}: "
                "old name {!r} -> new name {!r}".format(original["id"], old_name, new_name))
        if out["category_id_2"] is not None or out["category_id_3"] is not None:
            failures.append(
                "tier-0 annotation {} got a non-null tier-2/3 id, expected null for "
                "both since quadrant has no enumeration/diagnosis supervision".format(
                    original["id"]))

    if failures:
        print("FAILED -- normalize_quadrant is not a pure category-id-by-name "
              "passthrough:")
        for failure in failures:
            print("  - " + failure)
        return 1

    print("PASSED: normalize_quadrant left every bbox and area byte-identical "
          "across {} synthetic annotations, and remapped category_id_1 by name "
          "correctly through the real id-swap trap (quadrant id 0 named \"2\" -> "
          "categories_1 id 1). Tier-0 box geometry in this project's converted "
          "data is exactly the public release's geometry, unmodified -- there is "
          "no separate flattening convention for it to disagree with the "
          "original paper about.".format(len(before)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
