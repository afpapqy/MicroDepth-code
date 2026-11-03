"""Re-evaluate exactly three calibrated scan strategies at one common ray budget.

Unchanged first-hit ray caches are reused. All new projection, dropout, budget
selection, filtering and surface-provenance counts are recomputed here.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
WORK = REVISION.parent.parent
SIM = WORK / "Blender背景污染仿真"
STAGE = SIM / "source_reuse/stage2_occlusion"
METHODS = {"fixed_sL": ("recorded_L", "recorded_L"),
           "fixed_sR": ("recorded_R", "recorded_R"),
           "STEPS": ("recorded_L", "recorded_R")}
NAMES = {"fixed_sL": "Fixed sL", "fixed_sR": "Fixed sR", "STEPS": "STEPS"}
MISSING = ("none", "tip_drop", "all_tool_drop")
VIEWS = ("L", "R")
SEED = 20261025
F_REFERENCE = 1316.3658753527188
INPUTS: dict[str, str] = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def track(path):
    path = Path(path)
    INPUTS[str(path)] = sha(path)
    return path


def js(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(track(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csvwrite(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    return {"n": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def main():
    start = time.monotonic()
    if (HERE / "complete.json").exists():
        raise RuntimeError("Completed run already exists; preserve its result and provenance")
    (HERE / "maps").mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    old_config = js(STAGE / "config/historical_scaled.json")
    design = js(STAGE / "config/stage2_design.json")
    acquisition = js(STAGE / "acquisition_complete.json")
    assert design["seed"] == acquisition["seed"] == SEED
    assert acquisition["status"] == "complete"
    config = copy.deepcopy(old_config)
    config["scanners"] = {name: config["scanners"][name] for name in ("recorded_L", "recorded_R")}
    config["methods"] = {name: {"scans": scans, "acquisitions": len(set(scans)),
                                "checkerboard_half_per_scan": name == "STEPS"}
                         for name, scans in METHODS.items()}
    full_rays = int(np.prod(config["scanners"]["recorded_L"]["size"]))
    assert full_rays == int(np.prod(config["scanners"]["recorded_R"]["size"])) == 199176
    assert config["scanners"]["recorded_L"]["Rcw"] == config["scanners"]["recorded_R"]["Rcw"]
    protocol = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "methods": config["methods"], "target_rgb_size": config["rgb"]["L"]["size"],
        "attempted_rays_total_per_method": full_rays,
        "fixed_budget": "One complete 386 x 516 grid at the chosen calibrated scan pose, reused for both RGB projections.",
        "steps_budget": "Two scans; even checkerboard cells (u+v)%2=0 on each native simulation grid, 99,588 attempted rays per pose.",
        "budget_order": "Choose attempted scanner rays independently of hits or labels; remove specified first-hit tool returns without restoring occluded background.",
        "scope": "The frozen brain/closed-forceps scene, four gaps x three dropout conditions, exactly three calibrated strategies. No constructed scanner reorientation.",
        "fixed_meaning": "One depth acquisition position. The initial travel from s0 to that position is permitted; this is not a claim of zero stage movement throughout RGB and depth acquisition.",
        "dropout": design["missing_definitions"],
        "main_estimator": "Pure-background assignments divided by all complete-scene foreground pixel centres, pooled across both views by counts. Mixed, correct foreground and no output are retained separately.",
        "processing": "Unchanged legacy_mean rasterizer and legacy left-right score/mask. Unfiltered provenance counts are also saved.",
        "parallax": "All complete-scene foreground centres in the predeclared 5 mm case, without method-output filtering. Common RGB orientation; f_reference=1316.3658753527188 px. delta=f*((p-d)_xy/(p-d)_z-p_xy/p_z). Also report native distorted 512x384 chart pixels.",
        "cache_reuse": "Reuse unchanged BVH first-hit caches for the two real calibrated positions and complete-scene RGB truth; recompute all strategy-dependent processing and counts.",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }
    dump(HERE / "protocol.json", protocol)
    dump(HERE / "apparatus.json", config)
    sys.path.insert(0, str(SIM / "src"))
    from common import rays
    source = track(Path(design["reuse_analysis"]))
    assert INPUTS[str(source)] == design["reuse_analysis_sha256"]
    track(SIM / "src/common.py")
    track(SIM / "evidence/microdepth_tools.py")
    track(SIM / "evidence/calibration_20260114_114459.json")
    spec = importlib.util.spec_from_file_location("equal_budget_frozen_engine", source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.CFG = config
    engine.W, engine.H = config["rgb"]["L"]["size"]
    yy, xx = np.indices((engine.H, engine.W))
    engine.UV = np.column_stack((xx.ravel(), yy.ravel()))
    engine.RAYS = {side: rays(engine.UV, config["rgb"][side]) for side in VIEWS}
    engine.CAL = replace(engine.CAL, K_L=np.asarray(config["rgb"]["L"]["K"]),
                         K_R=np.asarray(config["rgb"]["R"]["K"]), image_size=(engine.W, engine.H))
    static = {name: load(STAGE / "data/static" / (name + ".npz")) for name in config["scanners"]}
    axis, tip = np.asarray(acquisition["tool_axis_xy"]), np.asarray(acquisition["tool_tip_xy_mm"])
    rows, budgets, checks, selected_truth = [], [], [], None
    count_fields = ("roi_pixels", "returned", "pure_background", "mixed", "other_wrong_surface", "correct_foreground", "missing_pixels")
    for case in acquisition["cases"]:
        case_name, gap = case["case"], case["requested_min_axial_gap_mm"]
        truth = {side: load(STAGE / "data/cases" / case_name / ("gt_" + side + ".npz")) for side in VIEWS}
        if gap == 5.0:
            selected_truth = truth
        clouds = {}
        shadowed = {}
        for scan in config["scanners"]:
            base = static[scan]
            delta = load(STAGE / "data/cases" / case_name / (scan + "_tool_firsthits.npz"))
            points = base["points"].copy()
            labels = (base["faces"] >= 0).astype(np.uint8)
            points[delta["indices"]] = delta["points"]
            labels[delta["indices"]] = 2
            ids = np.flatnonzero(labels > 0)
            clouds[scan] = {"points": points[ids], "labels": labels[ids], "sample_ids": ids}
            shadowed[scan] = delta["indices"]
            assert int(np.sum(labels == 2)) == len(delta["indices"])
        for missing in MISSING:
            selected = {}
            for method, scans in METHODS.items():
                chosen, total_attempted = {}, 0
                for scan in dict.fromkeys(scans):
                    cloud = clouds[scan]
                    width, height = config["scanners"][scan]["size"]
                    all_ids = np.arange(width * height)
                    budget_mask = ((all_ids % width + all_ids // width) % 2 == 0) if method == "STEPS" else np.ones(width * height, bool)
                    attempted = int(budget_mask.sum())
                    total_attempted += attempted
                    is_tool = cloud["labels"] == 2
                    remove = np.zeros(len(is_tool), bool)
                    if missing == "tip_drop":
                        remove = is_tool & (((cloud["points"][:, :2] - tip) @ axis) <= 22.)
                    elif missing == "all_tool_drop":
                        remove = is_tool
                    keep = budget_mask[cloud["sample_ids"]] & ~remove
                    chosen[scan] = {key: value[keep] for key, value in cloud.items()}
                    assert not np.any(np.isin(chosen[scan]["sample_ids"][chosen[scan]["labels"] == 1], shadowed[scan]))
                    budgets.append({"case": case_name, "gap_mm": gap, "missing": missing, "method": method, "scan": scan,
                                    "attempted_rays": attempted, "retained_points": int(keep.sum()),
                                    "retained_tool_points": int(np.sum(is_tool & keep)),
                                    "deleted_tool_points_on_attempted_rays": int(np.sum(remove & budget_mask[cloud["sample_ids"]]))})
                assert total_attempted == full_rays, (method, total_attempted)
                maps = {side: engine.rasterize(chosen[scan], config["rgb"][side], "legacy_mean")
                        for side, scan in zip(VIEWS, scans)}
                filters = engine.filter_maps(maps["L"], maps["R"])
                saved = {}
                for side in VIEWS:
                    for filter_name in ("raw", "legacy"):
                        result = engine.metrics(maps[side], filters[side][filter_name], truth[side], "all_centers")
                        assert result["other_wrong_surface"] == 0
                        assert result["pure_background"] + result["mixed"] + result["correct_foreground"] + result["missing_pixels"] == result["roi_pixels"]
                        rows.append({"case": case_name, "gap_mm": gap, "missing": missing, "method": method,
                                     "view": side, "filter": filter_name, "attempted_rays_total": total_attempted,
                                     **{key: result[key] for key in count_fields},
                                     "background_over_all_foreground": result["pure_background_per_roi"],
                                     "mixed_over_all_foreground": result["mixed"] / result["roi_pixels"],
                                     "correct_foreground_over_all_foreground": result["correct_coverage"],
                                     "no_output_over_all_foreground": result["missing_pixels"] / result["roi_pixels"]})
                    for key, value in maps[side].items():
                        saved[side + "_" + key] = value
                    saved[side + "_legacy"] = filters[side]["legacy"]
                    saved[side + "_truth_labels"] = truth[side]["labels"]
                    saved[side + "_truth_z"] = truth[side]["z"]
                for filter_name in ("raw", "legacy"):
                    pair = [r for r in rows if r["case"] == case_name and r["missing"] == missing and r["method"] == method and r["filter"] == filter_name and r["view"] in VIEWS]
                    assert len(pair) == 2
                    pooled = {key: sum(r[key] for r in pair) for key in count_fields}
                    denom = pooled["roi_pixels"]
                    rows.append({"case": case_name, "gap_mm": gap, "missing": missing, "method": method,
                                 "view": "L+R", "filter": filter_name, "attempted_rays_total": total_attempted,
                                 **pooled, "background_over_all_foreground": pooled["pure_background"] / denom,
                                 "mixed_over_all_foreground": pooled["mixed"] / denom,
                                 "correct_foreground_over_all_foreground": pooled["correct_foreground"] / denom,
                                 "no_output_over_all_foreground": pooled["missing_pixels"] / denom})
                if gap == 5.0:
                    np.savez_compressed(HERE / "maps" / (method + "__" + missing + ".npz"), **saved)
                selected[method] = maps
            # A fixed pose changes only the source scan in the cross-associated eye;
            # both fixed clouds are reused for two target projections, never merged.
            checks.append({"case": case_name, "missing": missing, "equal_total_attempted_rays": True,
                           "no_shadowed_background_restored": True, "methods": list(METHODS)})
        print(json.dumps({"case_complete": case_name, "seconds": round(time.monotonic() - start, 2)}), flush=True)
    assert selected_truth is not None
    primary = [r for r in rows if r["view"] == "L+R" and r["filter"] == "legacy"]
    assert len(primary) == 36
    selected_rows = [r for r in primary if r["gap_mm"] == 5.0 and r["missing"] == "tip_drop"]
    assert {r["roi_pixels"] for r in selected_rows} == {6732}
    # The unchanged equal-budget STEPS arm must reproduce its archived count vector.
    steps = next(r for r in selected_rows if r["method"] == "STEPS")
    assert [steps[k] for k in ("pure_background", "mixed", "correct_foreground", "missing_pixels")] == [269, 221, 4352, 1890]
    csvwrite(HERE / "all_metrics.csv", rows)
    csvwrite(HERE / "foreground_outcomes.csv", primary)
    csvwrite(HERE / "ray_budgets.csv", budgets)
    parallax, fields, reprojection_max = [], {}, 0.
    chart = lambda points: F_REFERENCE * points[:, :2] / points[:, 2, None]
    control_points = np.array([[1., 2., 230.], [-5., 3., 210.]])
    assert np.max(np.abs(chart(control_points) - chart(control_points - np.zeros(3)))) == 0
    dxy = np.array([.8, .2, 0.])
    assert np.max(np.abs(chart(control_points - dxy) - chart(control_points) + F_REFERENCE * dxy[:2] / control_points[:, 2, None])) < 1e-10
    dz = 27.1
    assert np.max(np.abs(chart(control_points - [0., 0., dz]) - chart(control_points) - (control_points[:, 2] / (control_points[:, 2] - dz) - 1)[:, None] * chart(control_points))) < 1e-10
    for side in VIEWS:
        camera = config["rgb"][side]
        K, D, R, C = (np.asarray(camera[key], float) for key in ("K", "D", "Rcw", "C"))
        mask = selected_truth[side]["labels"] == 2
        v, u = np.nonzero(mask)
        pixels = np.column_stack((u, v)).astype(float)
        normalized = cv2.undistortPointsIter(pixels.reshape(-1, 1, 2), K, D, None, None,
                                            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)).reshape(-1, 2)
        depth = selected_truth[side]["z"][mask]
        points = np.column_stack((normalized * depth[:, None], depth))
        projected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        reprojection_max = max(reprojection_max, float(np.max(np.abs(projected - pixels))))
        for method, scans in METHODS.items():
            scan = scans[VIEWS.index(side)]
            centre = np.asarray(config["scanners"][scan]["C"])
            displacement = (centre - C) @ R
            shifted = points - displacement
            assert np.all(shifted[:, 2] > 0)
            vectors = chart(shifted) - chart(points)
            norms = np.linalg.norm(vectors, axis=1)
            native = cv2.projectPoints(shifted, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
            parallax.append({"method": method, "view": side, "scan": scan,
                             "centre_displacement_camera_mm": displacement.tolist(),
                             "reference_px": stats(norms), "native_simulation_px": stats(np.linalg.norm(native - projected, axis=1))})
            image = np.full(mask.shape, np.nan)
            image[mask] = norms
            fields[method + "__" + side] = image
    assert reprojection_max < 1.1e-12
    np.savez_compressed(HERE / "parallax_fields.npz", **fields)
    dump(HERE / "parallax.json", {"definition": protocol["parallax"], "reference_focal_px": F_REFERENCE,
                                   "records": parallax, "reprojection_max_px": reprojection_max,
                                   "analytic_controls_pass": True})
    assert all(sha(Path(path)) == expected for path, expected in INPUTS.items())
    dump(HERE / "input_manifest.json", INPUTS)
    result = {"status": "complete", "seed": SEED, "protocol_sha256": sha(HERE / "protocol.json"),
              "exactly_three_methods": list(METHODS), "primary_rows": len(primary), "all_metric_rows": len(rows),
              "all_source_inputs_unchanged": True, "equal_total_ray_budget": full_rays,
              "case_checks": checks, "primary_condition": "5 mm clearance; distal 22 mm tool-return deletion",
              "primary_outcomes": selected_rows, "parallax": parallax,
              "archived_equal_budget_steps_reproduced": True, "seconds": time.monotonic() - start}
    dump(HERE / "complete.json", result)
    print(json.dumps({"complete": True, "seconds": round(result["seconds"], 2),
                      "primary": [{"method": r["method"], "B": r["pure_background"], "M": r["mixed"],
                                   "F": r["correct_foreground"], "O": r["missing_pixels"],
                                   "leakage_pct": 100 * r["background_over_all_foreground"]} for r in selected_rows],
                      "parallax": parallax}, ensure_ascii=False).encode("utf-8")[:5500].decode("utf-8", errors="ignore"), flush=True)


if __name__ == "__main__":
    main()
