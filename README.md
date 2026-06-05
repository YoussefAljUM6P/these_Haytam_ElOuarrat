# SERVIS Stop Reasons

- `ibvs_rms_below_threshold`: IBVS converged; feature reprojection error is low enough.
- `photometric_ssd_below_threshold`: Photometric servo converged; ViSP-style feature-error SSD is low enough.
- `max_iterations`: The run used all planned iterations without another stop.
- `callback`: The runner asked the servo loop to stop.
- `measurement_invalid_not_enough_features`: IBVS could not get enough valid feature matches.
- `measurement_invalid_not_enough_pixels`: Photometric servo had too few usable pixels.
- `measurement_invalid_nonfinite`: A measurement contained `NaN` or `inf`.
- `measurement_invalid_interaction_shape`: Torch photometric interaction matrix had the wrong shape.
- `measurement_invalid_length_mismatch`: Torch photometric residual and interaction matrix sizes did not match.
- `measurement_invalid_rank_deficient`: Interaction matrix does not observe enough DOF.
- `measurement_invalid_ill_conditioned`: Interaction matrix is too unstable to trust.
- `measurement_invalid_svd_failed`: Interaction matrix rank check failed.
- `velocity_invalid_shape`: Controller returned a velocity that was not 6-DOF.
- `velocity_invalid_nonfinite`: Controller returned `NaN` or `inf` velocity.
- `velocity_invalid_pose_update`: Velocity could not be converted into a valid next pose.
- `velocity_invalid_exceeds_hard_limit`: Velocity was too large to clip safely.

`step_accepted=false` means the pose was not advanced for that iteration.
`velocity_limited=true` means the command was clipped before applying it.
`raw_velocity` is the controller output; `velocity` is what was applied.

# SERVIS Session Tracker

- Stop/fault checks happen before pose updates.
- Rejected steps keep the pose unchanged.
- Velocity has soft clipping and hard rejection.
- History logs `raw_velocity` and applied `velocity`.
- Photometric stop uses ViSP-style feature-error SSD.
- Raw image MSE/SSD is diagnostic only.
- Interaction matrices must pass rank/condition checks before solving.
- IBVS stop uses pixel RMS, not normalized camera-coordinate error.
- Console `t_gap` is `||target translation - current translation||` in scene scale, no physical unit.
- Compact console output shows `closed=<pct>` instead of translation step size.
- Compact console output hides rank/condition; CSV/JSON still keep those diagnostics.
- Saved outputs use `translation_gap`/`translation_step` scene-scale names, not meter suffixes.
- Live servo/controller diagnostics use scene-scale names internally; old meter-suffix diagnostic keys were removed.
- Plots, validations, matrix reports, and comparisons label translation as scene-scale gap, not meters/mm.
- Depth comparison reports use scene-scale labels instead of meter suffixes for COLMAP-derived values.
- Photometric visualizations save one combined render, desired, and intensity-error image; final/task views are labelled final.
- Torch photometric validation now asserts accepted servo steps and nonzero pose updates; unusable real-depth fixtures skip explicitly.
- Main servo and trajectory loaders use COLMAP poses for mesh, GS, and NeRF; no ScanNet pose fallback.
- Photometric intrinsic-depth runs preflight target render depth and fail before servo if the COLMAP frame has too few valid pixels.
- Mesh diagnostics, smoke tests, and depth comparison CLI now use COLMAP records only.
- Added `python cli.py mesh-check --scene <name>` to verify mesh.ply projects into COLMAP cameras before trusting mesh runs.
- CLI wizard now has a comparative-table option that runs trajectory on all renderable datasets and writes `trajectory_summary.md`.
- New trajectory and servo-frame runs use `RUNS/<task>/<run_id>/` with root `README.md`, `summary.md`, `summary.json`, `config.resolved.json`, and `command.txt`.
- Unused ScanNet loader path was removed from project code; third-party references are untouched.
- Config keys: `stop_ssd`, `min_interaction_rank`, `max_interaction_condition`.
