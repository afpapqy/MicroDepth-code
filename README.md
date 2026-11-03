# MicroDepth — processing and evaluation code

Code accompanying the *Scientific Data* descriptor **A stereo-depth dataset for microsurgical scenes of soft tissues**.

The dataset is on Zenodo (version 1.0.0, DOI [10.5281/zenodo.19680077](https://doi.org/10.5281/zenodo.19680077), concept DOI 10.5281/zenodo.19680076). This repository holds the code that reads the released data and reproduces the processing and evaluation reported in the descriptor. It also holds the session partition and the machine-readable validation tables that the descriptor cites. A Photoneo scanner, the motion stage and the acquisition application are not needed.

## Contents

| Path | Contents |
|---|---|
| `code/` | Frame loading, calibration parsing, coordinate conversion, rasterization, left–right consistency score, depth–disparity conversion, metric accumulation and the fixed SGBM baseline. These scripts produce the numbers reported in Technical Validation. |
| `splits/` | `train.txt`, `validation.txt`, `test.txt` and `split_manifest.json`. See "Session partition". |
| `frame_paths.csv` | Mapping from release frame paths to the retained prediction paths (which use the earlier category names Brain, Chicken, Eye, NeuroBot, OtherInst and UsedInst), used by `code/evaluate_cached.py`. |
| `calibration/` | `calibration_20260114_114459.json`, `axis_calibration_20260114_172609.json` and `acquisition_settings.json` (logged rotation, empirical offset, apparent-motion axis and stage commands). |
| `validation/` | Machine-readable validation files. See "Validation files". |
| `simulation/` | `run_equal_budget.py`, the equal-total-ray comparison of fixed-viewpoint and STEPS scan-position allocation. Its inputs are not included. |

## Install

```bash
python -m pip install -r requirements.txt
```

The recorded environment is in `environment.json` (Python 3.12, NumPy 2.3.5, OpenCV 5.0.0, Pillow 12.3.0). The scripts contain no platform-specific code.

## Released data and two conventions

Download and unpack `MicroDepth.zip` from Zenodo. The unpacked folder contains `index.json`, `calibration/` and `data/` with one directory per scene category.

* `depth_gt.npy` and `depth_gt_right.npy` in version 1.0 hold **original-camera axial Z in millimetres on rectified image grids**. Nearest-neighbour remapping did not change the axial coordinate. The legacy `disparity_gt.npy` used that scalar in `f·B/Z`. `corrected_rectified_z` and `depth_to_disparity` in `code/microdepth_tools.py` give the rectified pixel-centre axial convention used in the descriptor. The conversion cannot undo floor binning, nearest-remap quantization, mean collisions, incomplete foreground or wrong-surface visibility. Zero remains zero.
* `confidence_rect.png` is an eight-bit left–right depth-agreement score after grayscale morphological closing. It is not the scanner's native confidence and it is not a probability of true error. The left ray used for the score is not undistorted, and a threshold of 128 does not guarantee an error below 5%.

## Quick start

Run these commands from the repository directory. Substitute your own paths. Every output directory must be new and separate from the original data.

```text
python code/load_example.py --data PATH_TO_MicroDepth --frame data/CATEGORY/SESSION/FRAME
python code/evaluate_cached.py --data PATH_TO_MicroDepth --stereo PATH_TO_FFS --mono PATH_TO_DA3 --partition test --out NEW_OUTPUT_DIR
python code/run_sgbm.py --help
```

Keep `code/`, `frame_paths.csv` and `splits/` in this relative arrangement. By default the evaluator reads the mapping and the split lists from the parent of `code/`, and `--mapping` overrides the location of the mapping file.

* `code/microdepth_tools.py` holds the calibration parser, the frame loader, the coordinate conversion, forward and inverse disparity conversion, the native-to-reference transforms, mean or nearest-Z rasterization, labelled consistency-score variants and unrounded metric accumulators.
* `code/evaluate_cached.py` re-evaluates retained stereo and monocular prediction arrays. Prediction failures are not filtered silently. Depth Anything 3 uses per-frame ground-truth median scaling, so its result is not an unscaled metric-depth score.
* `code/run_sgbm.py` runs the fixed classical method on the test sessions. Unmatched predictions receive zero disparity for the full-mask errors, and the returned-prediction coverage and the conditional EPE are reported separately.

The single SGBM configuration is minDisparity 0, numDisparities 256, blockSize 5, P1 200, P2 800, disp12MaxDiff 1, preFilterCap 63, uniquenessRatio 10, no speckle removal, SGBM_3WAY, grayscale 1280 × 960 inputs and two CPU threads. No parameter was tuned on the test results, and no model was trained.

### Reference results

All 3,586 retained prediction pairs were re-evaluated on the published valid mask. Overall Fast-FoundationStereo EPE is 0.9205 px under the legacy convention and 0.7941 px under the converted convention, and the prediction arrays are identical. Depth Anything 3 RMSE after reference-median scaling is 18.4431 mm under the converted convention. On the test sessions (525 frames, 33,707,574 published valid pixels), Fast-FoundationStereo EPE is 0.8968 px and SGBM full-mask EPE is 14.0116 px. SGBM returns a prediction for 93.26% of the valid pixels, and its EPE over the returned pixels is 2.4096 px. The smaller conditional value excludes unmatched predictions and must not replace the full-mask value.

## Session partition

`splits/train.txt`, `splits/validation.txt` and `splits/test.txt` list release frame paths (`data/CATEGORY/SESSION/FRAME`), one per line. They hold 2,656, 405 and 525 frames from 23, 5 and 6 acquisition sessions. All frames of a session, and therefore both views of every frame, are in the same list. The lists were frozen before the SGBM evaluation. They are not part of the version 1.0 data archive, which has no train, validation or test lists.

Categories are taken in the order recorded in `splits/split_manifest.json`. Within a category the sessions are sorted by name, and one `numpy.random.default_rng(20261025)` stream, shared by all categories, draws `rng.permutation(n_sessions)`. The first permuted session is the test session. The second is the validation session if the category has at least three sessions. The remaining sessions are training sessions. RobotEffector has two sessions and therefore no validation session. Sessions of one category can share a specimen, so the partition separates acquisition sessions and does not certify specimen independence.

To regenerate the lists and check them against the recorded hashes, run this with a new output directory:

```text
python code/make_splits.py --frames frame_paths.csv --manifest splits/split_manifest.json --out NEW_OUTPUT_DIR
```

The script exits with status 1 if a list differs from the manifest. The lists use CRLF line endings, and `.gitattributes` keeps them unchanged when the repository is cloned.

## Validation files

| Path | Contents |
|---|---|
| `validation/roi/region_definitions.json` | Image identifiers, orientation and vertices of the hand-drawn regions of the tissue-plus-instrument repeatability series (main Table 5). The native Boolean region and common-valid masks, the first-scan textures, display images, `reproduce_regions.py` and the drawing tool are in the `validation/roi/` folder of the Zenodo archive. |
| `validation/scanner_repeatability_settings.json` | Fixed scanner settings of the ten-scan series. |
| `validation/repeated_scan_counts.csv`, `repeated_scan_pairs.csv`, `repeated_scan_summary.csv` | Valid-point counts of every scan, all 45 scan pairs of each series and the summary of the effect of keeping the higher-count scan. |
| `validation/edge_configuration.json`, `validation/edge_proximity.csv` | Settings and category-level results of the unmatched edge-proximity diagnostic over all 3,586 rectified frames. |
| `validation/heldout_stereo.csv`, `validation/heldout_monocular.csv`, `validation/sgbm_configuration.json` | Category-level results on the test sessions and the fixed SGBM configuration. |
| `validation/simulation/` | Outputs of the equal-budget comparison. It holds `foreground_outcomes.csv` (36 pooled rows), `all_metrics.csv`, `ray_budgets.csv`, `parallax.json`, `parallax_fields.npz`, `protocol.json`, `complete.json`, `input_manifest.json` and a README. |

## What is not included

* The raw scanner point clouds and native depth arrays of the ten-scan series, and the scans behind the gauge-block measurements. The ROI masks and `reproduce_regions.py` that verify the polygon rasterization are in the Zenodo archive; recomputing the standard-deviation summaries needs the raw arrays.
* The inputs of the simulation, which are the cached first-hit rays and the scene assets. `simulation/run_equal_budget.py` documents the computation and `validation/simulation/` holds its outputs.
* The acquisition application. Re-acquiring data needs the Photoneo runtime and the motion stage.
* The original point clouds, the discarded trigger attempts and the complete historical tuning and rejection logs, which are not available.

The code checks above test implementation consistency. They do not measure absolute tissue accuracy, surface-identity contamination or independent sub-pixel registration.

## Licence

Code: MIT. Dataset: CC BY 4.0.
