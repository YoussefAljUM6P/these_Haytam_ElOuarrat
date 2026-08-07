# SERVIS (ViSERVO)

Virtual visual servoing for 6-DoF camera-pose estimation with mesh, 3D
Gaussian Splatting (3DGS), and NeRF scene representations.

SERVIS estimates the pose of a registered target image by moving a virtual
camera until its rendered view agrees with that image. Each iteration renders
the scene, measures feature or photometric error, computes a guarded 6-DoF
update, applies it on SE(3), and repeats.

> **Research-code status:** datasets, trained models, renderer assets, and run
> outputs are intentionally not stored in Git. A fresh clone must receive a
> prepared COLMAP scene before an experiment can run. Start with
> [Preparing a dataset](#preparing-a-dataset).

## Contents

- [Research objective](#research-objective)
- [How the system works](#how-the-system-works)
- [Repository layout](#repository-layout)
- [Requirements and backend support](#requirements-and-backend-support)
- [Installation](#installation)
- [Preparing a dataset](#preparing-a-dataset)
- [First run](#first-run)
- [Command-line workflows](#command-line-workflows)
- [Configuration](#configuration)
- [Outputs and metrics](#outputs-and-metrics)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Reproducibility and GitHub handoff](#reproducibility-and-github-handoff)
- [Known limitations](#known-limitations)

## Research objective

The project provides one experiment interface for the following comparison:

| Axis | Alternatives |
|---|---|
| Error signal | Feature-based IBVS (SIFT or XFeat) vs dense photometric servoing |
| Renderer | Mesh vs 3DGS vs NeRF |
| Depth source | Renderer/intrinsic depth vs learned MoGe2 depth |
| Evaluation | Rotation error, translation error, convergence rate, iterations, and wall-clock time |

The working hypothesis is that dense photometric servoing combined with 3DGS
can reduce the appearance gap between the real target and the virtual view.
This repository supplies the experiment machinery; it does not assume that
hypothesis is already proven.

## How the system works

~~~mermaid
flowchart LR
    A["COLMAP camera and real target frame"] --> B["Initial T_world_cam"]
    B --> C["Mesh / 3DGS / NeRF render"]
    C --> D["Intrinsic depth or MoGe2 depth"]
    C --> E["Feature error or photometric error"]
    D --> F["Interaction matrix"]
    E --> F
    F --> G["Guarded LM / damped 6-DoF solve"]
    G --> H["SE(3) camera-pose update"]
    H --> C
    H --> I["History, trajectories, renders, evo metrics"]
~~~

The important implementation boundaries are:

- <code>SRC/cli.py</code> is the single public entry point.
- <code>SRC/runners/</code> contains runnable workflows.
- <code>SRC/scenes/</code> adapts mesh, 3DGS, and NeRF to one camera interface.
- <code>SRC/controllers.py</code> implements feature-based IBVS.
- <code>SRC/photometric/</code> implements dense photometric optimization.
- <code>SRC/depth.py</code> is the only depth entry point.
- <code>SRC/servo.py</code> owns the render/control/update loop and safety guards.

### Camera and unit conventions

- Poses are stored as <code>T_world_cam</code>; the inverse is
  <code>T_cam_world</code>.
- Cameras use the OpenCV convention: x right, y down, z forward.
- Pose tensors are float32 on the active device.
- COLMAP supplies registered camera poses and intrinsics for every backend.
- Rotations are reported in degrees.
- COLMAP translations are normally in arbitrary **scene scale**, not
  automatically in metres. Results are metric only when the reconstruction was
  explicitly metric-scaled before the experiment.

## Repository layout

~~~text
SERVIS/
├── SRC/
│   ├── cli.py                    # unified entry point
│   ├── config_schema.py          # defaults and config validation
│   ├── dataset.py                # COLMAP + RGB loader
│   ├── camera.py                 # camera/frame conventions
│   ├── depth.py                  # sole depth gateway
│   ├── servo.py                  # iterative servo loop and guards
│   ├── controllers.py            # feature-based IBVS
│   ├── photometric/              # dense photometric controller
│   ├── runners/                  # experiments, sweeps, plots
│   ├── scenes/                   # mesh, 3DGS, NeRF adapters
│   └── third_party/              # Git submodules; do not edit in place
├── CONFIGS/                      # stable JSON configurations
├── DATA/                         # local scenes; ignored by Git
├── RUNS/                         # generated experiments; ignored by Git
├── scripts/                      # publishing, profiling, cluster helpers
├── requirements.txt              # audited first-party dependencies
└── .gitmodules                   # MoGe, XFeat, and 3DGS dependencies
~~~

<code>DATA/</code>, <code>RUNS/</code>, model weights, PLY files, COLMAP
binaries, checkpoints, and generated media are excluded by
<code>.gitignore</code>. Transfer them separately from the source repository.

## Requirements and backend support

The current environment was audited with Python 3.12.3; the lint configuration
targets Python 3.10 syntax. Linux is the supported deployment environment
because mesh rendering forces EGL and the GPU backends depend on CUDA.

| Capability | Required software/hardware | Notes |
|---|---|---|
| Core CLI and evaluation | Packages in <code>requirements.txt</code> | PyCOLMAP loads cameras; evo evaluates trajectories. |
| Mesh | EGL/OpenGL, Trimesh, Pyrender | Headless EGL rendering; no CUDA requirement in the mesh adapter itself. |
| SIFT IBVS | OpenCV | Recommended first matcher because it needs no learned weights. |
| XFeat IBVS | Accelerated-features submodule and <code>tqdm</code> | Implementation and weights are vendored as a submodule. |
| 3DGS | NVIDIA GPU, CUDA/nvcc, compatible PyTorch, compiled rasterizer | CUDA-only adapter; Graphdeco-format binary PLY required. |
| Learned depth | NVIDIA GPU and MoGe2 dependencies | First use downloads <code>Ruicheng/moge-2-vitl-normal</code>; no CPU fallback. |
| NeRF | NVIDIA GPU and checkpoint-compatible Nerfstudio | Keep config, dataparser transform, and checkpoint together. |

For GPU paths, the NVIDIA driver, CUDA toolkit, compiler, and PyTorch CUDA build
must be mutually compatible:

~~~bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

## Installation

All commands below run from the repository root unless stated otherwise.

### 1. Clone source and submodules

~~~bash
git clone --recurse-submodules <repository-url> SERVIS
cd SERVIS
git submodule update --init --recursive
~~~

The repository declares MoGe, XFeat, the Graphdeco rasterizer, and Gaussian
Splatting as submodules. Gaussian Splatting also has nested dependencies, so the
recursive update is mandatory even after a non-recursive clone.

### 2. Create the core environment

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
~~~

The root manifest covers the first-party import surface. Optional backends are
installed separately because they are lazy-loaded only when selected.

On Debian/Ubuntu, a typical software EGL runtime is:

~~~bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libgl1-mesa-dri
~~~

On an NVIDIA server, use the EGL/OpenGL packages supplied for the active driver.

### 3. Install only the optional backends you need

XFeat:

~~~bash
python -m pip install tqdm
~~~

MoGe2 learned depth:

~~~bash
python -m pip install -e SRC/third_party/MoGe
~~~

3DGS rendering:

~~~bash
python -m pip install --no-build-isolation \
  SRC/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
~~~

Use this nested rasterizer. The SERVIS adapter passes the
<code>antialiasing</code> argument, which the older top-level rasterizer
checkout does not support.

NeRF rendering:

~~~bash
python -m pip install nerfstudio
~~~

Nerfstudio serialization is version-sensitive. Prefer the version used to
train the stored checkpoint.

### 4. Verify the environment

~~~bash
python SRC/cli.py --help
python -c "import cv2, numpy, pycolmap, torch; print('core imports: OK')"
~~~

Optional checks:

~~~bash
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('3DGS extension: OK')"
python -c "import nerfstudio; print('Nerfstudio: OK')"
~~~

The root is intentionally not installed as a Python package. Invoke
<code>python SRC/cli.py ...</code> from the root, or change to
<code>SRC/</code> and invoke <code>python cli.py ...</code>, so flat imports
resolve correctly.

### Cluster setup

<code>scripts/setup_toubkal.sh</code> is a site-specific template, not a
portable installer. It contains cluster-module, CUDA, environment, and path
assumptions and currently expects a top-level <code>environment.yml</code>.
Audit it for the target cluster before use.

## Preparing a dataset

### Required scene contract

Every experiment starts from a COLMAP-registered scene under
<code>DATA/&lt;scene-name&gt;/</code>:

~~~text
DATA/<scene-name>/
├── images/                         # preferred RGB root
│   ├── frame-000001.jpg
│   ├── frame-000002.jpg
│   └── ...
│   # data/ may be used instead of images/
├── sparse/
│   └── 0/
│       ├── cameras.bin             # or cameras.txt
│       ├── images.bin              # or images.txt
│       └── points3D.bin            # or points3D.txt
├── mesh.ply                        # optional: mesh backend
├── gs.ply                          # optional: standard 3DGS
├── gs_moge.ply                     # optional: depth-supervised GS
├── split.txt                       # optional capture/take ranges
├── takes.json                      # optional/generated take cache
└── nerf/
    └── <training-run>/             # optional: NeRF backend
        ├── config.yml
        ├── dataparser_transforms.json
        └── nerfstudio_models/
            └── step-XXXXXXXXX.ckpt
~~~

These rules are load-bearing:

1. The sparse-model path is fixed at <code>sparse/0</code>.
2. The exact <code>image.name</code> stored by COLMAP must exist below either
   <code>images/</code> or <code>data/</code>. Preserve subdirectories when
   COLMAP registered a path such as
   <code>sequence_01/frame-000001.jpg</code>.
3. Use unique numeric names, preferably <code>frame-000001.jpg</code>,
   <code>frame-000002.jpg</code>, and so on. Several workflows extract the
   **first** number from the basename; duplicate extracted IDs can overwrite
   one another.
4. Prefer an undistorted <code>PINHOLE</code> or
   <code>SIMPLE_PINHOLE</code> reconstruction. The loader accepts several
   COLMAP camera models but does not apply their distortion coefficients when
   comparing real images with renders.
5. <code>mesh.ply</code>, <code>gs.ply</code>, and
   <code>gs_moge.ply</code> must already use the same COLMAP world frame and
   scale. SERVIS does not estimate an alignment at runtime.
6. Camera poses always come from COLMAP. Per-frame pose text files, captured
   depth PNGs, and a generic <code>transforms.json</code> are not inputs to
   the main servo loop.

### Copy an already prepared scene

Create the ignored data root and copy a complete scene without flattening it:

~~~bash
mkdir -p DATA
rsync -a --info=progress2 /external/storage/my_scene/ DATA/my_scene/
~~~

An equivalent local copy is:

~~~bash
mkdir -p DATA
cp -a /external/storage/my_scene DATA/my_scene
~~~

Verify the essential files:

~~~bash
find DATA/my_scene/sparse/0 -maxdepth 1 -type f -print
find DATA/my_scene/images -type f | head
~~~

Change the second path to <code>data/</code> if that is the scene's RGB root.
Do not rename registered images after reconstruction unless the names inside
the COLMAP model are updated too.

For supervisor handoff, keep the archive outside Git—for example in managed
institutional storage—and publish its download location, archive size,
checksum, scene-scale convention, frame count, and available renderer assets:

~~~bash
sha256sum my_scene.tar.zst
tar --zstd -xf my_scene.tar.zst -C DATA/
~~~

Replace placeholder storage instructions in release notes with the real,
authorized dataset URL. This repository has no dataset downloader.

### Assemble a scene from separate outputs

If reconstruction and renderer training happened in separate workspaces:

~~~bash
mkdir -p DATA/my_scene/sparse/0
rsync -a /path/to/registered/images/ DATA/my_scene/images/
rsync -a /path/to/colmap/sparse/0/ DATA/my_scene/sparse/0/
cp /path/to/mesh.ply DATA/my_scene/mesh.ply
cp /path/to/point_cloud/iteration_30000/point_cloud.ply DATA/my_scene/gs.ply
~~~

Copy only assets that exist and retain their original COLMAP transform and
scale. A Graphdeco <code>gs.ply</code> must be binary and keep the position,
opacity, scale, rotation, and spherical-harmonic fields.

<code>scripts/publish_gs.sh</code> can publish the latest Graphdeco point cloud
into a scene directory. Review it and use its dry-run mode first because
publishing can replace the destination PLY.

For NeRF, copy the run as a unit:

~~~bash
mkdir -p DATA/my_scene/nerf
cp -a /path/to/nerfstudio/run DATA/my_scene/nerf/my_run
~~~

Do not separate <code>config.yml</code>,
<code>dataparser_transforms.json</code>, and
<code>nerfstudio_models/</code>. If several valid configs exist, the adapter
selects the most recently modified one. Keeping one intended run avoids
ambiguity.

### Starting from raw images

SERVIS is not a reconstruction or training pipeline. There is no generic
importer that turns an arbitrary image folder into a runnable scene. Prepare
these products externally:

1. Rename frames with stable, unique numeric IDs.
2. Reconstruct and preferably undistort the sequence with COLMAP.
3. Put the final sparse model at <code>sparse/0</code> and the matching RGB
   files at <code>images/</code> or <code>data/</code>.
4. Optionally create a mesh, train Graphdeco 3DGS, and/or train Nerfstudio from
   that same reconstruction.
5. Export every renderer asset without losing the COLMAP world transform.
6. Run the checks below before launching a long experiment.

If a training pipeline normalizes, recentres, or rescales the scene, undo that
transform in the published model or implement an explicit, documented adapter.
Do not expect the servo loop to discover the alignment.

### Capture boundaries and takes.json

Trajectory runs avoid chaining across camera jumps. By default, takes are
loaded from <code>takes.json</code>, generated from an optional
<code>split.txt</code>, or inferred from jumps in COLMAP camera centres.

An explicit <code>split.txt</code> uses inclusive ranges:

~~~text
start=1 ; end=320
start=321 ; end=610
~~~

Changing <code>split.txt</code> may not invalidate every old take cache. Move
the cache aside to force regeneration while keeping it recoverable:

~~~bash
mv DATA/my_scene/takes.json DATA/my_scene/takes.json.bak
~~~

The scene directory must be writable when trajectory needs to generate
<code>takes.json</code>. Supply an existing valid cache for a read-only mount.

## First run

Start with a mesh, intrinsic renderer depth, SIFT, one frame pair, and five
iterations. This isolates dataset/pose problems before CUDA extensions or
learned depth.

### 1. Check mesh visibility in registered cameras

~~~bash
python SRC/cli.py mesh-check --scene my_scene
~~~

### 2. Render one registered pose beside the real image

~~~bash
python SRC/cli.py inspect \
  --scene my_scene \
  --index 1 \
  --renderer mesh
~~~

### 3. Run one small trajectory task

~~~bash
python SRC/cli.py trajectory \
  --datasets my_scene \
  --renderer mesh \
  --controller ibvs \
  --depth-mode intrinsic \
  --feature-method sift \
  --max-pairs 1 \
  --mini-iterations 5 \
  --no-save-task-viz
~~~

Inspect the new directory under <code>RUNS/</code>. Increase
<code>mini_iterations</code> and <code>max_pairs</code> only after real and
rendered views are geometrically consistent.

## Command-line workflows

Run <code>python SRC/cli.py &lt;command&gt; --help</code> for the
authoritative option list.

| Command | Purpose |
|---|---|
| no command / <code>wizard</code> | Interactive experiment builder using Questionary. |
| <code>inspect</code> | Compare one registered real frame with a rendered pose and optional features. |
| <code>mesh-check</code> | Diagnose whether <code>mesh.ply</code> projects into COLMAP cameras. |
| <code>servo-frames</code> | Estimate one target frame from another initial frame. |
| <code>trajectory</code> | Chain frame-to-frame servo tasks and evaluate the trajectory. |
| <code>matrix</code> | Legacy eight-condition IBVS sweep: mesh/GS × intrinsic/learned × SIFT/XFeat. |
| <code>compare</code> | General variant-by-dataset sweep over renderer, depth, matcher, or custom settings. |
| <code>plot</code> | Regenerate evo plots from a completed run. |
| <code>video</code> | Compile saved visualizations into an MP4. |
| <code>smoke</code> | Fixed mesh + GS smoke run; requires both assets and XFeat. |
| <code>bot</code> | Optional Telegram launcher for remote experiment control. |

### Interactive wizard

~~~bash
python SRC/cli.py
# equivalent:
python SRC/cli.py wizard
~~~

The wizard discovers renderer assets, writes a generated config, and dispatches
the selected runner in-process. Timestamped wizard configs are local run
records and are ignored by Git.

### Single frame-to-frame servo

Feature-based IBVS:

~~~bash
python SRC/cli.py servo-frames \
  --scene-dir my_scene \
  --renderer mesh \
  --controller ibvs \
  --depth-mode intrinsic \
  --feature-method sift \
  --start-index 1 \
  --index-away 1 \
  --iterations 100 \
  --viz-iter 5
~~~

Dense photometric servoing:

~~~bash
python SRC/cli.py servo-frames \
  --scene-dir my_scene \
  --renderer gs \
  --controller photometric \
  --depth-mode intrinsic \
  --start-index 1 \
  --index-away 1 \
  --iterations 100 \
  --gain-photo 0.1
~~~

When <code>target_index</code> is omitted, the target is
<code>start_index + index_away</code>. <code>feature_method</code> matters
only for IBVS; photometric runs use dense intensity residuals, image gradients,
depth, and their photometric solver settings.

For cross-backend work, use contiguous canonical filenames. In
<code>servo-frames</code>/<code>inspect</code>, mesh selection treats N as the
numeric <code>frame-NNNNNN</code> ID, whereas GS and NeRF may treat it as the
Nth frame in numeric order. Trajectory selection is numerically sorted for all
backends.

### Chained trajectory

~~~bash
python SRC/cli.py trajectory \
  --config CONFIGS/trajectory.example.json \
  --datasets my_scene \
  --renderer mesh \
  --controller ibvs \
  --depth-mode intrinsic \
  --feature-method sift \
  --start-index 1 \
  --stride 1 \
  --mini-iterations 100 \
  --max-pairs 20
~~~

<code>start_index</code> is the one-based position in the numerically sorted
registered-frame list. Each target is <code>stride</code> frames ahead; one
task's estimate initializes the next task. A take boundary resets the chain.

Resume an exact run explicitly:

~~~bash
python SRC/cli.py trajectory \
  --config CONFIGS/trajectory.example.json \
  --resume RUNS/<exact-run-directory>
~~~

Prefer an explicit path. Bare <code>--resume</code> matches only a subset of
identity fields and can select a run whose stride, gain, iterations, tag, or
even workflow differs.

### Research sweeps

The fixed legacy IBVS matrix contains eight conditions:

~~~text
2 renderers (mesh, GS)
× 2 depth modes (intrinsic, learned)
× 2 feature methods (SIFT, XFeat)
= 8 conditions
~~~

Run it with an explicit base config:

~~~bash
python SRC/cli.py matrix \
  --config CONFIGS/trajectory.example.json \
  --dataset my_scene \
  --iterations 30 \
  --max-pairs 10
~~~

Always pass a config in a clean clone. The built-in legacy default refers to a
project-local kitchen config that is not distributed, and its progress text
currently says 12 even though eight conditions are generated.

Use <code>compare</code> for NeRF, photometric control, or deliberately named
variants:

~~~bash
python SRC/cli.py compare \
  --config CONFIGS/trajectory.example.json \
  --datasets my_scene \
  --variant 'mesh_sift:renderer=mesh,controller=ibvs,depth_mode=intrinsic,feature_method=sift' \
  --variant 'gs_sift:renderer=gs,controller=ibvs,depth_mode=intrinsic,feature_method=sift' \
  --variant 'gs_photo:renderer=gs,controller=photometric,depth_mode=intrinsic' \
  --max-pairs 10
~~~

Preset axes are also available:

~~~bash
python SRC/cli.py compare \
  --config CONFIGS/trajectory.example.json \
  --datasets my_scene \
  --axis renderer
~~~

Valid axes are <code>renderer</code>, <code>depth</code>, and
<code>feature</code>. Repeated axes append presets; they do not create a
Cartesian product. Use custom variants for an exact factorial design. A feature
axis has no experimental effect with a photometric controller.

### Inspect, plot, and video

Inspect a renderer and feature correspondences:

~~~bash
python SRC/cli.py inspect \
  --scene my_scene \
  --index 1 \
  --renderer gs \
  --features sift
~~~

Regenerate plots or build a video from a known run:

~~~bash
python SRC/cli.py plot --run RUNS/<exact-run-directory>
python SRC/cli.py video --run RUNS/<exact-run-directory> --fps 8
~~~

Explicit paths are safer than automatic latest discovery while old and new run
layouts coexist. Avoid <code>video --render-missing</code> for current
trajectory summaries.

### Smoke test

~~~bash
python SRC/cli.py smoke --scene my_scene
~~~

This is not the recommended first command. It unconditionally expects both
<code>mesh.ply</code> and <code>gs.ply</code>, a working CUDA rasterizer, and
XFeat. It writes <code>output_mesh.png</code>, <code>output_gs.png</code>,
<code>servo_mesh/</code>, and <code>servo_gs/</code> outside the normal run
layout and overwrites earlier smoke artifacts.

## Configuration

Two stable starting points are included:

- <code>CONFIGS/servo_frames.example.json</code>
- <code>CONFIGS/trajectory.example.json</code>

Copy one, give it a descriptive name, and replace
<code>example_scene</code>:

~~~bash
cp CONFIGS/trajectory.example.json CONFIGS/my_scene_mesh_sift.json
~~~

Configuration precedence is:

~~~text
schema defaults < JSON config < named CLI flags < repeated --set overrides
~~~

Example:

~~~bash
python SRC/cli.py trajectory \
  --config CONFIGS/my_scene_mesh_sift.json \
  --renderer gs \
  --set gain_ibvs=0.5 \
  --set max_pairs=5
~~~

The last value wins. Values passed through <code>--set</code> are parsed as
JSON where possible. Unknown keys fail validation. Keys are normalized to
lowercase and hyphens become underscores. If JSON contains <code>kind</code>,
it must match <code>trajectory</code> or <code>servo_frames</code>.

Relative config paths resolve from the project root and then
<code>CONFIGS/</code>. Bare scene names resolve below <code>DATA/</code>.
Dataset/take lists may be comma-separated; take indices are one-based.

Do not feed a generated <code>config.resolved.json</code> back to
<code>--config</code>: resolved snapshots include runner metadata such as
<code>run_root</code>, and strict config loading rejects unknown keys.

Do not rely on the built-in scene defaults in a fresh clone. Trajectory
currently defaults to dataset <code>living</code>, renderer <code>gs</code>,
and 500 iterations per pair; servo-frames defaults to
<code>DATA/kitchen</code>, renderer <code>mesh</code>, and 100 iterations.
Both default to IBVS, intrinsic depth, SIFT, and controller gain 0.75. The
named scenes are not distributed, so set a scene/config explicitly.

### Main selectors

| Key / flag | Values | Meaning |
|---|---|---|
| <code>renderer</code> | <code>mesh</code>, <code>gs</code>, <code>nerf</code> | Virtual scene representation. |
| <code>controller</code> | <code>ibvs</code>, <code>photometric</code> | Sparse feature or dense intensity error. |
| <code>depth_mode</code> | <code>intrinsic</code>, <code>learned</code> | Renderer depth or MoGe2 depth. |
| <code>feature_method</code> | <code>sift</code>, <code>xfeat</code> | IBVS matcher; ignored by photometric control. |
| <code>gs_model</code> | <code>standard</code>, <code>moge</code> | Selects <code>gs.ply</code> or <code>gs_moge.ply</code>. |
| <code>datasets</code> | names/list | Scenes below <code>DATA/</code> for trajectory/compare. |
| <code>scene_dir</code> | name/path | Scene used by <code>servo-frames</code>. |

The experiment CLI currently defaults to
<code>depth_mode=intrinsic</code>. At the Python API,
<code>get_depth(..., use_intrinsic=False)</code> defaults to MoGe2. This
distinction is intentional:

- MoGe2 is metric and SERVIS never scale-aligns it.
- Intrinsic depth comes explicitly from <code>scene.render_depth()</code>.
- A requested estimator failure raises; there is no silent fallback.
- Captured depth PNGs are not substituted for either mode.
- <code>gs_model=moge</code> chooses a separately trained GS asset and is
  independent of runtime <code>depth_mode=learned</code>.

### Frequently tuned parameters

| Group | Keys / flags | Purpose |
|---|---|---|
| Trajectory | <code>start_index</code>, <code>stride</code>, <code>mini_iterations</code>, <code>max_pairs</code>, <code>rpe_delta</code> | Pair selection and evaluation horizon. |
| Feature IBVS | <code>gain_ibvs</code>, <code>ratio</code>, <code>min_features</code>, <code>dynamic_ibvs_iters</code> | Matching and feature updates. |
| Photometric | <code>gain_photo</code>, <code>sigma_blur</code>, <code>use_gzn</code>, <code>grad_percentile</code>, <code>photometric_max_pixels</code>, <code>use_huber</code>, <code>huber_k</code> | Dense sampling, smoothing, and robust solve. |
| Rendering | <code>mesh_render_scale</code>, <code>gs_render_scale</code>, <code>nerf_render_scale</code>, <code>gs_model</code> | Backend resolution/model choice. |
| Convergence | <code>stop_residual_px</code>, <code>stop_mse_per_px</code>, <code>stop_ssd</code> | Successful stopping thresholds. |
| Divergence | <code>diverge_residual_px</code>, <code>diverge_mse_per_px</code>, <code>continue_on_task_failure</code> | Failure/cut behavior. |
| Safety | Translation/rotation soft and hard limits, <code>min_interaction_rank</code>, <code>max_interaction_condition</code> | Reject unobservable, unstable, or unsafe updates. |
| Outputs | <code>save_task_viz</code>, <code>task_viz_every</code>, <code>viz_iter</code>, <code>run_tag</code> | Visualization and provenance settings. The tag does not currently name the output folder. |

Schema plateau options are parsed but not connected to the runners, so do not
use them as a claimed convergence criterion. Verify critical fields in the
saved command/config because some resolved snapshots omit
<code>gs_model</code>.

Use help for exact types and defaults:

~~~bash
python SRC/cli.py servo-frames --help
python SRC/cli.py trajectory --help
python SRC/cli.py matrix --help
python SRC/cli.py compare --help
~~~

## Outputs and metrics

New <code>servo-frames</code> and <code>trajectory</code> runs use
collision-safe flat names:

~~~text
RUNS/my_scene-MESH-IBVS-INTRINSIC-SIFT/
RUNS/my_scene-MESH-IBVS-INTRINSIC-SIFT-01/
~~~

The numeric suffix prevents overwriting. A typical run contains:

~~~text
<run>/
├── README.md
├── summary.md
├── summary.json
├── config.json                    # servo-frames
├── config.resolved.json
├── command.txt
├── sim_traj.tum
├── gt_traj.tum
├── per_task_errors.csv
├── history.csv                    # servo-frames
├── target.png                     # servo-frames
├── initial_render.png             # servo-frames
├── final_render.png               # servo-frames
├── error_evolution.png            # servo-frames
├── trajectory_xyz.png             # trajectory
├── trajectory_xy.png              # trajectory
├── ape_translation.png            # trajectory
├── ape_rotation.png               # trajectory
└── visualizations/
~~~

Exact artifacts depend on workflow and visualization/evo settings. Servo runs
also append <code>RUNS/servo_frames/_index.csv</code>. Batch outputs use:

~~~text
RUNS/servo_matrix/<batch-id>/       # per-condition logs + matrices/manifests
RUNS/compare/<batch-id>/            # per-variant/dataset logs + combined CSVs
RUNS/inspect/                       # inspection images
~~~

The main trajectory metrics are synchronized, unaligned evo APE/RPE, plus task
convergence and timing:

- absolute/relative translation error in raw COLMAP scene scale;
- absolute/relative rotation error in degrees;
- completed and converged tasks plus convergence rate;
- iteration, render, controller, and wall-clock timing;
- per-task initial/final error, pose gaps, and stop reason.

Do not interpret translation as metres unless the scene was explicitly scaled.
Some legacy plots or helpers still label it <code>m</code>; the main runner
does not automatically apply hidden Sim(3) alignment. Optional standardized
outputs are separate from raw results.

### Servo stop reasons and guards

| Stop reason | Meaning |
|---|---|
| <code>ibvs_rms_below_threshold</code> | Feature reprojection RMS reached its threshold. |
| <code>photometric_ssd_below_threshold</code> | Photometric feature-error SSD reached its threshold. |
| <code>max_iterations</code> | The task used all planned iterations. |
| <code>callback</code> | The runner requested a stop. |
| <code>measurement_invalid_not_enough_features</code> | Too few valid IBVS matches. |
| <code>measurement_invalid_not_enough_pixels</code> | Too few usable photometric pixels. |
| <code>measurement_invalid_nonfinite</code> | Measurement contained NaN or infinity. |
| <code>measurement_invalid_interaction_shape</code> | Interaction matrix had the wrong shape. |
| <code>measurement_invalid_length_mismatch</code> | Residual/matrix lengths differed. |
| <code>measurement_invalid_rank_deficient</code> | Matrix did not observe enough DoF. |
| <code>measurement_invalid_ill_conditioned</code> | Matrix was too unstable. |
| <code>measurement_invalid_svd_failed</code> | Rank/condition analysis failed. |
| <code>velocity_invalid_shape</code> | Controller output was not a 6-vector. |
| <code>velocity_invalid_nonfinite</code> | Controller output contained NaN or infinity. |
| <code>velocity_invalid_hard_limit</code> | Raw update exceeded a configured hard limit. |
| <code>velocity_invalid_pose_update</code> | Update did not produce a valid pose. |
| <code>render_exception</code> / <code>controller_exception</code> | Rendering or measurement raised. |

<code>step_accepted=false</code> means the camera did not advance in that
iteration. <code>velocity_limited=true</code> means the raw command was clipped
to a soft limit. History records both <code>raw_velocity</code> and applied
<code>velocity</code>.

## Validation

There is no consolidated pytest suite or CI workflow yet. Validate in proportion
to the selected backend.

### Dataset and renderer checks

~~~bash
python SRC/cli.py mesh-check --scene my_scene
python SRC/cli.py inspect --scene my_scene --index 1 --renderer mesh
python SRC/cli.py inspect --scene my_scene --index 1 --renderer gs
python SRC/cli.py inspect --scene my_scene --index 1 --renderer nerf
~~~

Run only backends for which the scene has assets and the environment has
dependencies.

### Executable numerical checks

~~~bash
python SRC/validate_controller.py
python SRC/validate_servo.py
python SRC/validate_photometric.py
~~~

Backend-specific scripts include <code>SRC/validate_gs_depth.py</code> and
<code>SRC/validate_nerf_pose.py</code>. They require GPU environments/assets;
inspect them first because some research helpers retain machine-specific
examples.

If Ruff is installed:

~~~bash
python -m pip install ruff
ruff check SRC
~~~

The strongest handoff test is a fresh recursive clone, one small copied scene,
and the mesh-only first run.

## Troubleshooting

### The scene loads no frames

- Confirm <code>DATA/&lt;scene&gt;/sparse/0</code> exists.
- Confirm the RGB root is exactly <code>images/</code> or <code>data/</code>.
- Compare every COLMAP <code>image.name</code> with the relative RGB path,
  including case and nested directories.
- Use unique numeric basenames.

### The real image and render have different geometry

- Confirm images were undistorted; distortion coefficients are not applied.
- Confirm the model uses the RGB intrinsics and resolution.
- Confirm the renderer asset remains in the COLMAP world frame.
- Run <code>mesh-check</code> and <code>inspect</code> before servoing.

### Mesh fails on a headless server

Install a working EGL/OpenGL runtime matching the active GPU driver. The mesh
module selects EGL before importing Pyrender. Remote X11 is not required, but
EGL device access is.

### 3DGS import fails or rejects antialiasing

Initialize recursive submodules, remove a conflicting rasterizer installation
from the environment, and compile the nested Gaussian Splatting rasterizer
shown in [Installation](#3-install-only-the-optional-backends-you-need).
PyTorch and nvcc must target compatible CUDA versions.

### A 3DGS PLY fails to load

Export a binary Graphdeco point cloud, not an arbitrary XYZ/color PLY. Keep its
opacity, scale, quaternion, and spherical-harmonic properties.

### Learned depth fails

Install vendored MoGe, ensure CUDA is available, and allow the first model
download on a network-enabled machine. SERVIS raises on failure by design; it
never silently switches depth estimators.

### Learned depth destabilizes updates

MoGe2 depth is metric while an unscaled COLMAP scene usually is not. SERVIS does
not estimate a scale factor. Metric-scale the scene explicitly or use intrinsic
renderer depth for a scale-consistent baseline.

### Too few features or pixels

- For IBVS, inspect correspondences, reduce the frame gap, and review
  <code>ratio</code> and <code>min_features</code>.
- For photometric control, verify exposure/overlap, target intrinsic-depth
  coverage, gradient percentile, and usable pixel count.
- A rank/condition fault indicates geometry or observability trouble; it is not
  automatically a reason to disable every safety guard.

### Tasks unexpectedly cross or split captures

Add inclusive ranges to <code>split.txt</code> and move the old
<code>takes.json</code> aside so it can be rebuilt.

### CUDA runs out of memory

Lower <code>gs_render_scale</code> or <code>nerf_render_scale</code> and the
photometric sample budget <code>photometric_max_pixels</code>. Run one
high-resolution backend per process.

## Reproducibility and GitHub handoff

### What belongs in Git

Commit source, stable configs, small documentation, the dependency manifest,
and submodule pointers. Keep these outside Git:

- <code>DATA/</code> and <code>RUNS/</code>;
- PLY models, COLMAP binaries, NeRF checkpoints, and learned weights;
- generated images, videos, NumPy dumps, logs, caches, and backups;
- credentials, bot tokens, local environment files, and machine paths;
- paper libraries or benchmark exports unless they are an intentional,
  documented part of the repository.

Do not use <code>git add .</code> blindly in a research worktree. A safer
handoff sequence is:

~~~bash
git status --short
git diff --stat
git submodule status --recursive

# Stage only files reviewed for the handoff.
git add README.md requirements.txt ruff.toml \
  CONFIGS/servo_frames.example.json \
  CONFIGS/trajectory.example.json
git add -p SRC scripts

git diff --cached --check
git diff --cached --stat
git status --short
~~~

Before committing:

1. Verify <code>requirements.txt</code>, stable configs, and every required
   first-party module are actually tracked. The development worktree contains
   files that a tracked-only clone would otherwise omit, so this check matters.
2. Confirm no dataset, run, model, checkpoint, backup, credential, paper dump,
   or generated benchmark artifact is staged.
3. Secret-scan and inspect every match, including notification URLs and bot
   configuration.
4. Confirm every submodule pointer resolves to a commit available from its
   remote. Parent commits do not contain modifications made inside submodules;
   either leave them clean or publish authorized forks and update the gitlinks.
5. Record the external dataset URL and SHA-256.
6. Record Python, PyTorch/CUDA, driver, GPU, renderer checkpoint/iteration, and
   the exact command/config with each reported result.
7. Test a fresh recursive clone.

Useful checks:

~~~bash
git ls-files DATA RUNS
git grep -n -I -E 'TOKEN|PASSWORD|SECRET|API_KEY' -- . ':!SRC/third_party'
python SRC/cli.py --help
~~~

The first command should print nothing. For the optional bot, ensure
<code>SRC/phone_bot.py</code> is deliberately tracked and provide
<code>SERVIS_BOT_TOKEN</code> and <code>SERVIS_BOT_ALLOWED_USERS</code> only
through the environment—never a committed config.

A private supervisor repository may not need a public-release license
immediately. Before public release, add a top-level license and preferred
citation.

### Suggested experiment provenance

Preserve with every result:

- Git commit SHA and submodule SHAs;
- resolved JSON config and command saved by the runner;
- dataset archive checksum and scene-scale convention;
- Python, package, PyTorch, CUDA, driver, and GPU versions;
- renderer training method/checkpoint/iteration;
- random seeds where applicable;
- untouched raw CSV/JSON outputs, not only summary figures.

## Known limitations

- Targets are registered COLMAP frames. The CLI does not yet take an arbitrary
  unseen image and estimate its pose.
- <code>sparse/0</code> and the <code>images/</code>/<code>data/</code> roots
  are fixed by the loader.
- Lens distortion is not applied in the comparison loop.
- Reconstruction, mesh generation, 3DGS training, and NeRF training are
  external workflows.
- Renderer assets are assumed aligned; there is no runtime Sim(3)
  registration.
- Translation remains raw COLMAP scene scale unless metric-standardized.
- Learned depth and 3DGS are CUDA-only in the current adapters.
- NeRF config selection is automatic when several runs exist.
- Multi-dataset trajectory creates separate runs rather than one combined
  summary.
- Automatic resume/latest-run discovery, some plot labels, default config
  names, and cluster helpers retain older assumptions; use explicit paths.
- <code>video --render-missing</code> is not compatible with the current
  trajectory summary layout.
- Plateau schema values currently have no runner effect.
- Bot relaunch inference is unsafe for servo resolved configs that omit
  <code>kind</code>; launch them directly instead.
- The fixed matrix covers legacy IBVS conditions only. Use custom
  <code>compare</code> variants for photometric or NeRF experiments.
- There is no portable installer, raw-dataset importer, consolidated automated
  test suite, CI workflow, or top-level public-release license yet.

---

When extending the project, keep frame names explicit
(<code>T_world_cam</code> and <code>T_cam_world</code>), route all depth
access through <code>SRC/depth.py</code>, and wrap vendored dependencies
instead of editing <code>SRC/third_party/</code>.
