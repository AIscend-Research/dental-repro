"""
Table construction for ``paper_assets/tables/``.

Every table is emitted twice — CSV (auditable, diffable) and booktabs LaTeX
(paste-ready) — and every emission registers itself in the manifest.

The original paper's Table 1 is embedded verbatim below. It is the *only*
place in this repository where a number that was not produced by an executed
run is allowed to live, and it is labelled as such everywhere it is used. Cells
we did not run say ``not run``; they are never interpolated, extrapolated or
inferred from a neighbouring mode.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from typing import Dict, List, Optional, Sequence

from . import manifest, setup_env

TABLES_DIR = os.path.join(setup_env.PAPER_ASSETS, "tables")

#: Sentinel for a cell whose experiment did not run in this mode.
NOT_RUN = "not run"

# --------------------------------------------------------------------------
# Reference numbers — HierarchicalDet, MICCAI 2023, Table 1 (test split).
# Columns: AR, AP[0.5:0.95], AP50, AP75, APm, APl.
# --------------------------------------------------------------------------
REFERENCE_TABLE_CSV = """\
tier,method,AR,AP,AP50,AP75,APm,APl
quadrant,RetinaNet,0.604,25.1,41.7,28.8,32.9,25.1
quadrant,FasterRCNN,0.588,29.5,48.6,33.0,39.9,29.5
quadrant,DETR,0.659,39.1,60.5,47.6,55.0,39.1
quadrant,DiffusionDet_base,0.677,38.8,60.7,46.1,39.1,39.0
quadrant,Ours_wo_Transfer,0.699,42.7,64.7,52.4,50.5,42.8
quadrant,Ours_wo_Manipulation,0.727,40.0,60.7,48.2,59.3,40.0
quadrant,Ours_wo_Manip_Transfer,0.658,38.1,60.1,45.3,45.1,38.1
quadrant,Ours_full,0.717,43.2,65.1,51.0,68.3,43.1
enumeration,RetinaNet,0.560,25.4,41.5,28.5,55.1,25.2
enumeration,FasterRCNN,0.496,25.6,43.7,27.0,53.3,25.2
enumeration,DETR,0.440,23.1,37.3,26.6,43.4,23.0
enumeration,DiffusionDet_base,0.617,29.9,47.4,34.2,48.6,29.7
enumeration,Ours_wo_Transfer,0.648,32.8,49.4,39.4,60.1,32.9
enumeration,Ours_wo_Manipulation,0.662,30.4,46.5,36.6,58.4,30.5
enumeration,Ours_wo_Manip_Transfer,0.557,26.8,42.4,29.5,51.4,26.5
enumeration,Ours_full,0.668,30.5,47.6,37.1,51.8,30.4
diagnosis,RetinaNet,0.587,32.5,54.2,35.6,41.7,32.5
diagnosis,FasterRCNN,0.533,33.2,54.3,38.0,24.2,33.3
diagnosis,DETR,0.514,33.4,52.8,41.7,48.3,33.4
diagnosis,DiffusionDet_base,0.644,37.0,58.1,42.6,31.8,37.2
diagnosis,Ours_wo_Transfer,0.669,39.4,61.3,47.9,49.7,39.5
diagnosis,Ours_wo_Manipulation,0.688,36.3,55.5,43.1,45.6,37.4
diagnosis,Ours_wo_Manip_Transfer,0.648,37.3,59.5,42.8,33.6,36.4
diagnosis,Ours_full,0.691,37.6,60.2,44.0,36.0,37.7
"""

#: Metric column order used everywhere a per-tier result is tabulated.
METRIC_COLUMNS = ("AR", "AP", "AP50", "AP75", "APm", "APl")
TIER_ORDER = ("quadrant", "enumeration", "diagnosis")


def reference_rows() -> List[Dict[str, object]]:
    """The paper's Table 1 as records, with numeric metric values."""
    rows = []
    for row in csv.DictReader(io.StringIO(REFERENCE_TABLE_CSV)):
        parsed = {"tier": row["tier"], "method": row["method"]}
        for metric in METRIC_COLUMNS:
            parsed[metric] = float(row[metric])
        rows.append(parsed)
    return rows


def reference_lookup() -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{tier: {method: {metric: value}}}``."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in reference_rows():
        out.setdefault(row["tier"], {})[row["method"]] = {
            metric: row[metric] for metric in METRIC_COLUMNS
        }
    return out


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------
def _format_cell(value) -> str:
    if value is None:
        return NOT_RUN
    # NaN is a real, meaningful outcome here, distinct from "not run": COCOeval
    # returns -1 for a metric with no ground truth in that bucket (e.g. APs when
    # the split has no small objects), which the evaluator turns into NaN.
    # Printing a bare "nan" in a paper table would read as a bug.
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        return "{:.3f}".format(value) if abs(value) < 10 else "{:.2f}".format(value)
    return str(value)


def _latex_escape(text: str) -> str:
    for character, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                                   ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                                   ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                                   ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(character, replacement)
    return text


def write_table(name: str, rows: Sequence[Dict[str, object]], columns: Sequence[str],
                caption: str, notebook: str, run_mode: str, asset_class: str,
                inputs: Sequence[str] = (), label: Optional[str] = None,
                note: str = "") -> Dict[str, str]:
    """Write ``<name>.csv`` and ``<name>.tex`` and register both."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, name + ".csv")
    tex_path = os.path.join(TABLES_DIR, name + ".tex")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})

    # Right-align a column only if every cell in it reads as a number; text
    # columns (tier, method, metric, ...) stay left-aligned.
    def numeric(column: str) -> bool:
        values = [_format_cell(row.get(column)) for row in rows]
        values = [v for v in values if v not in ("", NOT_RUN)]
        if not values:
            return False
        for value in values:
            try:
                float(value.split(" ")[0])
            except ValueError:
                return False
        return True

    label = label or "tab:" + name
    lines = [
        "% Generated by src/tables.py — do not edit by hand.",
        "% Source rows: {}".format(csv_path),
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + "".join("r" if numeric(c) else "l" for c in columns) + "}",
        r"\toprule",
        " & ".join(_latex_escape(str(c)) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(c))) for c in columns)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(tex_path, "w") as handle:
        handle.write("\n".join(lines))

    for path in (csv_path, tex_path):
        manifest.record_asset(path, asset_class, notebook, run_mode,
                              inputs=inputs, note=note)
    return {"csv": csv_path, "tex": tex_path}


def record_not_run(asset_class: str, notebook: str, run_mode: str, reason: str) -> None:
    """Account for an asset class that this run mode deliberately skips."""
    placeholder = os.path.join(TABLES_DIR, "_not_run_{}.txt".format(
        asset_class.replace(":", "_")))
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(placeholder, "w") as handle:
        handle.write("{} was not produced in RUN_MODE={}.\nReason: {}\n".format(
            asset_class, run_mode, reason))
    manifest.drop_missing(asset_class)
    manifest.record_asset(placeholder, asset_class, notebook, run_mode,
                          status="not run", note=reason)


# --------------------------------------------------------------------------
# Builders — each takes already-loaded raw results and returns table rows
# --------------------------------------------------------------------------
def _agg(payload: Dict[str, object], tier: str, metric: str) -> Optional[Dict[str, float]]:
    try:
        cell = payload["aggregate"][tier][metric]
    except (KeyError, TypeError):
        return None
    if cell.get("mean") is None:
        return None
    return cell


def _mean_std(cell: Optional[Dict[str, float]], digits: int = 2) -> object:
    if cell is None:
        return None
    if cell.get("n", 0) <= 1:
        return round(cell["mean"], digits)
    return "{:.{d}f} ± {:.{d}f}".format(cell["mean"], cell["std"], d=digits)


def main_results_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    ``{model_label: multi-seed payload}`` -> one row per (tier, model) with
    mean ± std across inference seeds.
    """
    rows = []
    for tier in TIER_ORDER:
        for label, payload in results.items():
            row: Dict[str, object] = {"tier": tier, "method": label}
            any_value = False
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                digits = 3 if metric == "AR" else 2
                row[metric] = _mean_std(cell, digits)
                any_value = any_value or cell is not None
            row["seeds"] = len(payload.get("seeds", []) or [])
            if any_value:
                rows.append(row)
    return rows


def comparison_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Reproduced vs original, with absolute and relative deltas.

    A method the original reports but we did not run appears with the original
    value and ``not run`` reproduced — never the other way round.
    """
    reference = reference_lookup()
    rows = []
    for tier in TIER_ORDER:
        for method, original in reference.get(tier, {}).items():
            payload = results.get(method)
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric) if payload else None
                ours = cell["mean"] if cell else None
                delta = None if ours is None else round(ours - original[metric], 3)
                relative = (None if ours is None or original[metric] == 0
                            else round(100.0 * (ours - original[metric]) / original[metric], 1))
                rows.append({
                    "tier": tier,
                    "method": method,
                    "metric": metric,
                    "original": original[metric],
                    "reproduced": None if cell is None else round(cell["mean"], 3),
                    "reproduced_std": None if cell is None else round(cell.get("std", 0.0), 3),
                    "delta": delta,
                    "delta_pct": relative,
                })
    return rows


def seed_variance_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for label, payload in results.items():
        for tier in TIER_ORDER:
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                if cell is None:
                    continue
                rows.append({
                    "method": label, "tier": tier, "metric": metric,
                    "n_seeds": cell.get("n"),
                    "mean": round(cell["mean"], 4),
                    "std": round(cell.get("std", 0.0), 4),
                    "min": round(min(cell.get("values", [cell["mean"]])), 4),
                    "max": round(max(cell.get("values", [cell["mean"]])), 4),
                    "values": ";".join("{:.3f}".format(v) for v in cell.get("values", [])),
                })
    return rows


def sweep_rows(payloads: Dict[str, Dict[str, object]], axis_name: str,
               extra: Optional[Dict[str, Dict[str, object]]] = None
               ) -> List[Dict[str, object]]:
    """
    Generic one-axis sweep table (steps, degradation, fault injection,
    clean-vs-stress): one row per (axis value, tier), metrics mean ± std.
    """
    rows = []
    for axis_value, payload in payloads.items():
        for tier in TIER_ORDER:
            row: Dict[str, object] = {axis_name: axis_value, "tier": tier}
            any_value = False
            for metric in METRIC_COLUMNS:
                cell = _agg(payload, tier, metric)
                row[metric] = _mean_std(cell, 3 if metric == "AR" else 2)
                any_value = any_value or cell is not None
            for key, value in (extra or {}).get(axis_value, {}).items():
                row[key] = value
            if any_value:
                rows.append(row)
    return rows


def degradation_seed_consistency_rows(
        degradation_results: Dict[str, Dict[str, object]], clean_key: str = "clean",
        tier: str = "diagnosis", metric: str = "AP") -> List[Dict[str, object]]:
    """
    Per-seed, paired condition-minus-clean deltas at one tier/metric.

    An averaged degradation-vs-clean gap can look like noise even when every
    individual inference seed agrees on its sign -- averaging is exactly what
    hides that agreement. Pairing by seed (same weights, same seed, only the
    input image differs) exposes whether a condition beating clean is a
    reproducible effect or a coin flip that happened to land the same way on
    average. ``consistent_direction`` is True only when every paired seed
    agrees in sign; that is the property "noise" does not have.
    """
    def by_seed(payload):
        return {r.get("inference_seed"): r["tiers"].get(tier, {}).get("metrics", {}).get(metric)
                for r in (payload or {}).get("runs") or []}

    clean_by_seed = by_seed(degradation_results.get(clean_key))
    rows = []
    for condition, payload in degradation_results.items():
        if condition == clean_key:
            continue
        cond_by_seed = by_seed(payload)
        diffs = {}
        for seed, clean_value in clean_by_seed.items():
            cond_value = cond_by_seed.get(seed)
            if (cond_value is None or clean_value is None
                    or not math.isfinite(cond_value) or not math.isfinite(clean_value)):
                continue
            diffs[seed] = cond_value - clean_value
        if not diffs:
            continue
        signs = {1 if d > 0 else -1 if d < 0 else 0 for d in diffs.values()}
        rows.append({
            "condition": condition,
            "n_seeds_paired": len(diffs),
            "mean_delta_vs_clean": round(sum(diffs.values()) / len(diffs), 3),
            "min_delta": round(min(diffs.values()), 3),
            "max_delta": round(max(diffs.values()), 3),
            "consistent_direction": len(signs - {0}) <= 1,
        })
    return rows


def label_scheme_rows() -> List[Dict[str, object]]:
    """
    The test split's 9-code Turkish labelling scheme, spelled out as a table.

    Static -- derived from ``data_convert.EXPECTED_WORD_CODES`` and
    ``OUT_OF_TASK_LABELS``, not from any experiment run -- so it needs no
    dataset, no GPU, and cannot drift out of date with the converter itself.
    Exists so a reader can check "does the test release have a Deep Caries
    code" without reading Python: it does not, which is why that row is
    listed with no code at all.
    """
    from . import data_convert

    task_words = {"curuk": "Caries", "gomulu": "Impacted", "lezyon": "Periapical Lesion"}
    rows = []
    for word, code in sorted(data_convert.EXPECTED_WORD_CODES.items(), key=lambda kv: kv[1]):
        if word in task_words:
            gloss, task_class = task_words[word], True
        else:
            gloss, task_class = data_convert.OUT_OF_TASK_LABELS.get(word, "?"), False
            gloss = gloss.split(" (code")[0]
        rows.append({"code": code, "turkish_word": word, "gloss": gloss,
                     "task_class": task_class, "note": ""})
    rows.append({"code": "(none)", "turkish_word": "derin curuk / deep caries",
                 "gloss": "Deep Caries", "task_class": False,
                 "note": "no code in this scheme -- see table:diagnosis_label_histogram"})
    return rows


def quadrant_geometry_rows(geometry: Dict[str, object]) -> List[Dict[str, object]]:
    """
    One row per tier-0 quadrant category: box count, mean area fraction of
    the image, mean box center. A box averaging ~25% of the image would read
    as a literal quarter-image region; anything much smaller is a tight
    tooth-row box -- see ``data_convert.quadrant_box_geometry``.
    """
    rows = []
    for quadrant, stats in sorted(geometry.get("per_quadrant", {}).items()):
        rows.append({
            "quadrant_category_id": quadrant,
            "n_boxes": stats["n"],
            "mean_area_fraction_of_image": stats["mean_area_fraction"],
            "mean_center_x": stats["mean_center_xy"][0],
            "mean_center_y": stats["mean_center_xy"][1],
        })
    return rows


def clean_stress_confound_rows(confound: Dict[str, object]) -> List[Dict[str, object]]:
    """One row per subset: is it matched with the other on anything besides
    the annotation count that defines the split? See
    ``data_convert.clean_stress_confound_check``."""
    rows = []
    for name, stats in confound.items():
        rows.append({"subset": name, **stats})
    return rows


def audit_rows(audits: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for name, audit in audits.items():
        coverage = audit.get("tier_coverage", {})
        rows.append({
            "split": name,
            "images": audit.get("num_images"),
            "annotations": audit.get("num_annotations"),
            "images_without_annotations": audit.get("images_with_zero_annotations"),
            "quadrant_labels": coverage.get("tier0_quadrant"),
            "enumeration_labels": coverage.get("tier1_enumeration"),
            "diagnosis_labels": coverage.get("tier2_diagnosis"),
            "distinct_resolutions": audit.get("num_distinct_resolutions"),
            "unreadable_images": len(audit.get("unreadable_images", [])),
            "missing_image_files": len(audit.get("missing_image_files", [])),
            "malformed_boxes": len(audit.get("malformed_boxes", [])),
        })
    return rows


def diagnosis_label_histogram_rows(audits: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Raw diagnosis-tier class counts per split, independent of any training run.

    Exists to make one claim independently checkable by a reader: which
    diagnosis classes actually have ground truth in each split, by name and
    count, straight out of the converted annotations. A class absent from a
    split's histogram had zero occurrences in that split's converted JSON --
    it is not implied or inferred, it is what ``Counter`` returned.
    """
    all_classes = sorted({
        name for audit in audits.values()
        for name in audit.get("class_histograms", {}).get("diagnosis", {})
    })
    rows = []
    for split, audit in audits.items():
        if not audit.get("tier_coverage", {}).get("tier2_diagnosis"):
            continue                      # split carries no diagnosis-tier labels at all
        histogram = audit.get("class_histograms", {}).get("diagnosis", {})
        row = {"split": split}
        for class_name in all_classes:
            row[class_name] = histogram.get(class_name, 0)
        rows.append(row)
    return rows


def failure_rows(runtimes: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for label, summary in runtimes.items():
        counts = summary.get("failure_counts", {}) or {}
        scores = summary.get("score_stats") or {}
        rows.append({
            "method": label,
            "images": summary.get("images"),
            "detections": summary.get("total_detections"),
            "images_with_no_detections": summary.get("images_with_no_detections"),
            "degenerate_box": counts.get("degenerate_box", 0),
            "box_out_of_image": counts.get("box_out_of_image", 0),
            "box_covers_whole_image": counts.get("box_covers_whole_image", 0),
            "extreme_aspect_ratio": counts.get("extreme_aspect_ratio", 0),
            "crashes": summary.get("crashes", 0),
            "median_score": scores.get("median"),
            "frac_score_above_0.5": scores.get("frac_above_0.5"),
        })
    return rows


def low_resource_rows(entries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for entry in entries:
        rows.append({
            "method": entry.get("method"),
            "device": entry.get("device"),
            "sample_step": entry.get("sample_step"),
            "seconds_per_image": entry.get("mean_seconds_per_image"),
            "images_per_minute": entry.get("images_per_minute"),
            "peak_gpu_memory_mb": entry.get("peak_gpu_memory_mb"),
            "parameters_millions": entry.get("parameters_millions"),
            "checkpoint_mb": entry.get("checkpoint_mb"),
            "images": entry.get("images"),
        })
    return rows


def per_class_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Per-class AP for every tier. Explicitly OUR extension: the paper reports
    only tier-level aggregates, so these numbers have no reference column.
    """
    rows = []
    for label, payload in results.items():
        runs = payload.get("runs") or []
        if not runs:
            continue
        for tier in TIER_ORDER:
            per_class_by_name: Dict[str, List[float]] = {}
            for run in runs:
                for name, value in (run["tiers"].get(tier, {})
                                    .get("per_class_AP", {}) or {}).items():
                    if isinstance(value, (int, float)):
                        per_class_by_name.setdefault(name, []).append(value)
            for name, values in sorted(per_class_by_name.items()):
                rows.append({
                    "method": label, "tier": tier, "class": name,
                    "AP_mean": round(sum(values) / len(values), 2),
                    "n_seeds": len(values),
                    "note": "OUR EXTENSION — not reported in the original paper",
                })
    return rows


def manipulation_margin_rows(results: Dict[str, Dict[str, object]], tier: str = "diagnosis",
                             full: str = "Ours_full",
                             ablated: str = "Ours_wo_Manipulation") -> List[Dict[str, object]]:
    """
    Per-class decomposition of the noisy-box-manipulation margin at one tier.

    A tier-level AP gap between the full model and the ablation can hide a
    reversal -- one class carrying the entire margin while another moves the
    opposite way, which a reader cannot see from the aggregate number or a bar
    chart alone. This makes that check explicit and exact instead of visual.
    """
    def per_class_means(label: str) -> Dict[str, float]:
        runs = (results.get(label) or {}).get("runs") or []
        by_class: Dict[str, List[float]] = {}
        for run in runs:
            for name, value in (run["tiers"].get(tier, {})
                                .get("per_class_AP", {}) or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    by_class.setdefault(name, []).append(value)
        return {name: sum(values) / len(values) for name, values in by_class.items()
                if values}

    full_by_class = per_class_means(full)
    ablated_by_class = per_class_means(ablated)
    deltas = {}
    for name in sorted(set(full_by_class) & set(ablated_by_class)):
        deltas[name] = full_by_class[name] - ablated_by_class[name]
    total_margin = sum(deltas.values())

    rows = []
    for name, delta in deltas.items():
        rows.append({
            "class": name,
            "{}_AP".format(full): round(full_by_class[name], 3),
            "{}_AP".format(ablated): round(ablated_by_class[name], 3),
            "delta": round(delta, 3),
            "share_of_tier_margin_pct": (round(100 * delta / total_margin, 1)
                                        if total_margin else None),
            "direction": ("favors full" if delta > 0 else
                         "favors ablated" if delta < 0 else "tie"),
        })
    return rows


# --------------------------------------------------------------------------
# Claim-coverage matrix (consumed by paper_assets/scope_and_claims.md)
# --------------------------------------------------------------------------
#: Every claim the original paper makes, mapped to how this study treats it.
#: ``requires_models`` downgrades a claim to "cited, untested" automatically
#: when the run mode did not train the models the claim needs — so the scope
#: statement can never overstate what was actually tested.
PAPER_CLAIMS = (
    {
        "claim": "The hierarchical multi-label model beats base DiffusionDet at every tier.",
        "default_status": "tested",
        "evidence": "notebook 05, tables/original_vs_reproduced",
        "requires_models": ["Ours_full", "DiffusionDet_base_tier2"],
    },
    {
        "claim": "Noisy-box manipulation is the component that contributes the accuracy gain.",
        "default_status": "tested",
        "evidence": "notebook 03 (matched budgets) + notebook 05; "
                    "Ours_full vs Ours_wo_Manipulation",
        "requires_models": ["Ours_full", "Ours_wo_Manipulation"],
    },
    {
        "claim": "Weight transfer alone does not improve accuracy.",
        "default_status": "tested",
        "evidence": "Ours_wo_Transfer vs Ours_wo_Manip_Transfer at matched budgets",
        "requires_models": ["Ours_wo_Transfer", "Ours_wo_Manip_Transfer"],
    },
    {
        "claim": "The full model outperforms RetinaNet, Faster R-CNN and DETR at every tier.",
        "default_status": "tested",
        "evidence": "notebook opt-10 baselines",
        "requires_models": ["RetinaNet", "FasterRCNN", "DETR"],
    },
    {
        "claim": "SimMIM pretraining on 1,571 unlabelled X-rays contributes to the result.",
        "default_status": "out of scope",
        "evidence": "the authors' SimMIM checkpoint is not published; notebook opt-11 "
                    "exists for anyone with spare quota",
    },
    {
        "claim": "The reported numbers are obtained on the DENTEX test split (250 images).",
        "default_status": "tested",
        "evidence": "notebook 01 converts and verifies the test ground truth; every "
                    "number in main_results is computed on that split",
    },
    {
        "claim": "The approach is robust enough for panoramic X-ray analysis in practice.",
        "default_status": "extended beyond the paper",
        "evidence": "notebook 06: degradation grid, clean/stress subsets and hierarchy "
                    "fault injection - none of which the original paper tests",
        # Cited evidence that must actually exist. A claim whose supporting
        # figure was recorded "not run" is not supported by it, and saying so
        # here is cheaper than hoping a reader cross-checks the manifest.
        "requires_assets": ["figure:degradation", "figure:fault_injection"],
    },
    {
        "claim": "Diffusion sampling steps trade accuracy against latency.",
        "default_status": "extended beyond the paper",
        "evidence": "notebook 06 step sweep with both axes measured",
        "requires_assets": ["figure:ap_vs_steps"],
    },
)


def load_all(kind: str) -> Dict[str, Dict[str, object]]:
    """Load every raw result of a kind, keyed by its distinguishing part."""
    out = {}
    for name in manifest.list_results(kind):
        path = os.path.join(setup_env.RESULTS_RAW, name + ".json")
        with open(path) as handle:
            out[name] = json.load(handle)
    return out
