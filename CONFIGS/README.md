# SERVIS Experiment Configs

Experiment scripts now read JSON config files and accept one-off overrides.

Run these commands from the same Python environment you use for SERVIS. For
example, if that environment is `viservo`, prefix commands with
`conda run -n viservo`.

All execution goes through the unified CLI at `SRC/cli.py`. Running it with no
arguments opens an interactive wizard; passing a subcommand runs a specific
experiment.

Run a trajectory from the repo root:

```bash
python SRC/cli.py trajectory --config trajectory_kitchen_mesh.json
python SRC/cli.py trajectory --config trajectory_kitchen_mesh.json --set renderer=mesh --set datasets=kitchen --set iterations=30
```

Run a single frame-to-frame servo trial:

```bash
python SRC/cli.py servo-frames --config servo_kitchen_mesh.json
python SRC/cli.py servo-frames --config servo_kitchen_mesh.json --set scene=kitchen --set start=1 --set target=2
```

Run the full servo matrix sweep:

```bash
python SRC/cli.py matrix --dataset kitchen --iterations 30
```
