# SERVIS — Agent Notes

## What this is
ViSERVO: virtual visual servoing for 6-DOF camera pose estimation. Given a target image + 3D scene, iteratively move a virtual camera until its render matches the target. Loop: render → image error → LM-on-SE(3) update → repeat. Convergence = pose.

## Comparison matrix (the research goal)
- **Error signal**: photometric (pixel-wise) vs feature-based (keypoints).
- **Renderer**: 3DGS vs mesh vs NeRF.
- **Metrics**: pose error (rot deg, trans m), convergence rate, iters / wall-clock.
- Working hypothesis: photometric + 3DGS wins (appearance fidelity closes the domain gap).

## Layout
- `SRC/` — project code. Single entry point: `cli.py`.
- `SRC/runners/` — runnable experiments (`smoke`, `servo_frames`, `trajectory`, `matrix`). Not executed directly; dispatched by `cli.py`.
- `SRC/scenes/` — renderers (`mesh.py`, `gs.py`, `nerf.py`).
- `SRC/depth.py` — single entry point for depth (see policy below).
- `SRC/third_party/` — vendored upstream. Don't edit; wrap.
- `DATA/` — scenes (e.g. `DATA/kitchen`). Not committed.
- `RUNS/` — experiment outputs. Not committed.
- `CONFIGS/` — experiment JSONs.

## Run
```bash
cd SRC
python cli.py                                              # interactive wizard
python cli.py smoke                                        # mesh + GS smoke test
python cli.py servo-frames --config ../CONFIGS/servo_kitchen_mesh.json
python cli.py trajectory   --config ../CONFIGS/trajectory_kitchen_mesh.json
python cli.py matrix       --dataset kitchen --iterations 30
```

## Conventions
- Python: 4-space, `snake_case` funcs, `CapWords` classes. Comment only where math/frames are non-obvious.
- Camera: OpenCV convention, `T_world_cam` stored, float32 on device.
- Optimizer: LM on SE(3).
- Be explicit with frame names (`T_world_cam`, `T_cam_world`) — never ambiguous.

## Depth policy (load-bearing)
`SRC/depth.py` is the only depth entry point. Default = MoGe2 (`get_depth(..., use_intrinsic=False)` / `estimate_depth_moge`). MoGe2 is metric — no scale alignment. Scene-intrinsic depth (`scene.render_depth()`) is opt-in via `use_intrinsic=True`. If the requested estimator fails, raise — never silently fall back, never swap the default without explicit instruction.

## Don't commit
Generated images, datasets (`DATA/`), runs (`RUNS/`), `__pycache__`, third-party edits.

## When unsure
Ask before architectural changes.
