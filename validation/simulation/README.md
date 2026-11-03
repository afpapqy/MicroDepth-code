# Equal-budget comparison of three calibrated scan configurations

This folder holds the outputs of `simulation/run_equal_budget.py`. Only fixed sL, fixed sR and STEPS are compared. The point clouds are simulated first-hit points in a frozen scene. The scanner poses come from the retained calibration, so no additional real acquisition is involved.

`recorded_L` and `recorded_R` are internal names in the cache. They denote the calibrated sL and sR scanner rail poses and do not denote an additional eye-specific orientation. Both poses keep the same recorded rotation.

The scanner grid is 386 x 516 and the RGB projection grid is 512 x 384. Each fixed arm uses one complete scan of 199,176 rays and projects it to both RGB cameras. Rays are not counted twice because of the double projection. STEPS selects 99,588 cells of a deterministic even checkerboard at each of sL and sR, so 199,176 rays are attempted in total. The budget does not depend on hits or on surface identity.

The unchanged first-hit geometry cache is reused. The configuration allocation, budget sampling, tool-return deletion, projection, mean rasterization, left-right score and mask, and surface-identity counts are recomputed. Deletion is applied after visibility is determined, so background that the tool occluded is not restored. The seed is 20261025.

The primary condition is the predeclared 5 mm clearance with deletion of the distal 22 mm of tool returns. The complete results cover four clearances, three missing-return conditions and three strategies. `foreground_outcomes.csv` holds the 36 pooled rows after the legacy score mask. `all_metrics.csv` also keeps the per-view and unfiltered results. `ray_budgets.csv` records the attempted rays of every scan.

`parallax.json` and `parallax_fields.npz` give the calibrated-centre parallax over all complete foreground support, independently of output, deletion and budget selection. Reference-focal-length pixels and native simulation RGB pixels are reported separately. Neither is a registration error after correct three-dimensional reprojection.

`input_manifest.json` records the SHA-256 of every reused input. The local workspace prefix of the original paths is replaced by `<workspace>`. `protocol.json` records the rules frozen for this round and `complete.json` records the completion evidence. The script protects a completed result and refuses to overwrite it, so a rerun needs a separate working directory.

The input caches are not part of this repository. They are the cached first-hit rays and the scene assets in the authors' working area (`source_reuse/stage2_occlusion` of the background-contamination simulation project).
