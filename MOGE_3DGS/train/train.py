"""Standalone 3DGS trainer with switchable MoGe-depth supervision.

Reuses the gaussian-splatting submodule's Scene / GaussianModel / renderer.
The only thing this script owns is the training loop + the depth-loss
selection (see depth_losses.py).

Usage (minimal):
    python MOGE_3DGS/train/train.py \
        -s /path/to/colmap_scene \
        -m /path/to/output_dir \
        --depth-loss l1_inv_aligned --lambda-d 0.1 \
        -d depths            # required for *_aligned variants; folder under -s

The scene directory layout must match the gaussian-splatting convention:
    <source>/images/...                  RGB images
    <source>/sparse/0/{cameras,images,points3D}.{bin,txt}
    <source>/<depths>/...                inverse-depth 16-bit PNGs (optional)
    <source>/sparse/0/depth_params.json  per-image scale/offset (optional)

Run the preprocessing scripts (MOGE_3DGS/preprocess/) to populate the depth
folder + depth_params.json from MoGe.
"""
from __future__ import annotations

import os
import sys
import uuid
import json
from argparse import ArgumentParser, Namespace
from random import randint

import torch

# --- Make the gaussian-splatting submodule importable. ---------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GS_DIR = os.path.join(_REPO_ROOT, "SRC", "third_party", "gaussian-splatting")
if _GS_DIR not in sys.path:
    sys.path.insert(0, _GS_DIR)
# Also expose `utils.*`, `scene.*`, `arguments.*` packages that live under it.
for _sub in ("utils", "scene", "gaussian_renderer", "arguments", "lpipsPyTorch"):
    _p = os.path.join(_GS_DIR, _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.general_utils import safe_state, get_expon_lr_func  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from gaussian_renderer import render, network_gui  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    from fused_ssim import fused_ssim
    _HAS_FUSED_SSIM = True
except ImportError:
    _HAS_FUSED_SSIM = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    _HAS_SPARSE_ADAM = True
except ImportError:
    _HAS_SPARSE_ADAM = False

# --- Our own depth-loss module. -------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from depth_losses import depth_loss as compute_depth_loss, VARIANTS as DEPTH_VARIANTS  # noqa: E402


# ---------------------------------------------------------------------------
# Output / logging
# ---------------------------------------------------------------------------

def prepare_output(args) -> SummaryWriter | None:
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    print(f"[MOGE_3DGS] output: {args.model_path}")
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "moge_3dgs_cfg.json"), "w") as f:
        json.dump(
            {
                "depth_loss": args.depth_loss,
                "lambda_d_init": args.lambda_d,
                "lambda_d_final": args.lambda_d_final,
                "depth_loss_until": args.depth_loss_until,
            },
            f,
            indent=2,
        )
    if _HAS_TB:
        return SummaryWriter(args.model_path)
    print("[MOGE_3DGS] tensorboard not available; logging to stdout only.")
    return None


def eval_and_log(
    tb_writer,
    iteration,
    scene: Scene,
    pipe,
    background,
    train_test_exp,
    test_iterations,
):
    if iteration not in test_iterations:
        return
    torch.cuda.empty_cache()
    configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[i % len(scene.getTrainCameras())]
                for i in range(5, 30, 5)
            ],
        },
    )
    for cfg in configs:
        if not cfg["cameras"]:
            continue
        l1_test, psnr_test, n = 0.0, 0.0, 0
        for vp in cfg["cameras"]:
            out = render(
                vp,
                scene.gaussians,
                pipe,
                background,
                use_trained_exp=train_test_exp,
                separate_sh=_HAS_SPARSE_ADAM,
            )
            img = torch.clamp(out["render"], 0.0, 1.0)
            gt = torch.clamp(vp.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                img = img[..., img.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]
            l1_test += l1_loss(img, gt).mean().double().item()
            psnr_test += psnr(img, gt).mean().double().item()
            n += 1
        l1_test /= max(n, 1)
        psnr_test /= max(n, 1)
        print(f"[ITER {iteration}] {cfg['name']}  L1={l1_test:.4f}  PSNR={psnr_test:.3f}")
        if tb_writer:
            tb_writer.add_scalar(f"{cfg['name']}/l1", l1_test, iteration)
            tb_writer.add_scalar(f"{cfg['name']}/psnr", psnr_test, iteration)
    if tb_writer:
        tb_writer.add_scalar("scene/n_points", scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram(
            "scene/opacity", scene.gaussians.get_opacity, iteration
        )
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training(
    dataset,
    opt,
    pipe,
    args,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    if opt.optimizer_type == "sparse_adam" and not _HAS_SPARSE_ADAM:
        sys.exit("sparse_adam requested but diff-gaussian-rasterization does not export it.")

    tb_writer = prepare_output(args)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    first_iter = 0
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and _HAS_SPARSE_ADAM

    # Exponential decay for lambda_d (lambda_d -> lambda_d_final by iter `depth_loss_until`).
    lambda_d_sched = get_expon_lr_func(
        args.lambda_d,
        args.lambda_d_final,
        max_steps=max(args.depth_loss_until, 1),
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    if not viewpoint_stack:
        sys.exit("No training cameras found — check --source_path / data layout.")

    ema_total = 0.0
    ema_depth = 0.0

    from tqdm import tqdm
    pbar = tqdm(range(first_iter + 1, opt.iterations + 1), desc=f"train [{args.depth_loss}]")
    for iteration in pbar:
        # --- Live-viewer poke (kept verbatim from upstream) --------------------
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (custom_cam, do_training, pipe.convert_SHs_python,
                 pipe.compute_cov3D_python, keep_alive, scaling_modifer) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=_HAS_SPARSE_ADAM,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (iteration < int(opt.iterations) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --- Pick a random training camera --------------------------------
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand(3, device="cuda") if opt.random_background else background

        # --- Render -------------------------------------------------------
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=_HAS_SPARSE_ADAM,
        )
        image = render_pkg["render"]
        viewspace_pts = render_pkg["viewspace_points"]
        visibility = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        rendered_invdepth = render_pkg["depth"]  # convention in this fork

        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        # --- Photometric loss --------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        l1_val = l1_loss(image, gt_image)
        if _HAS_FUSED_SSIM:
            ssim_val = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_val = ssim(image, gt_image)
        loss_photo = (1.0 - opt.lambda_dssim) * l1_val + opt.lambda_dssim * (1.0 - ssim_val)

        # --- Depth-supervision loss --------------------------------------
        lam_d = lambda_d_sched(iteration) if iteration <= args.depth_loss_until else 0.0
        depth_term = rendered_invdepth.new_zeros(())
        if lam_d > 0.0 and args.depth_loss != "none" and getattr(viewpoint_cam, "depth_reliable", False):
            gt_invd = viewpoint_cam.invdepthmap.cuda()
            d_mask = viewpoint_cam.depth_mask.cuda() if viewpoint_cam.depth_mask is not None else None
            depth_term = compute_depth_loss(
                args.depth_loss, rendered_invdepth, gt_invd, d_mask
            )

        loss = loss_photo + lam_d * depth_term
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_total = 0.4 * loss.item() + 0.6 * ema_total
            ema_depth = 0.4 * float(depth_term.detach().item() if depth_term.requires_grad else depth_term.item()) + 0.6 * ema_depth

            if iteration % 10 == 0:
                pbar.set_postfix(
                    loss=f"{ema_total:.5f}",
                    d=f"{ema_depth:.5f}",
                    lam=f"{lam_d:.3f}",
                    n=int(gaussians.get_xyz.shape[0]),
                )

            if tb_writer and iteration % 50 == 0:
                tb_writer.add_scalar("train/loss_total", loss.item(), iteration)
                tb_writer.add_scalar("train/loss_photo", loss_photo.item(), iteration)
                tb_writer.add_scalar("train/loss_depth_raw", float(depth_term.item()), iteration)
                tb_writer.add_scalar("train/lambda_d", lam_d, iteration)
                tb_writer.add_scalar("train/n_points", gaussians.get_xyz.shape[0], iteration)

            eval_and_log(
                tb_writer, iteration, scene, pipe, background,
                dataset.train_test_exp, testing_iterations,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] saving gaussians")
                scene.save(iteration)

            # --- Densification (verbatim from upstream) -------------------
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility] = torch.max(
                    gaussians.max_radii2D[visibility], radii[visibility]
                )
                gaussians.add_densification_stats(viewspace_pts, visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold, radii,
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # --- Optimizer step ------------------------------------------
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
                )

    print("\n[MOGE_3DGS] training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="3DGS + MoGe depth-supervision trainer")
    lp = ModelParams(p)
    op = OptimizationParams(p)
    pp = PipelineParams(p)
    p.add_argument("--ip", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6009)
    p.add_argument("--debug-from", type=int, default=-1)
    p.add_argument("--detect-anomaly", action="store_true")
    p.add_argument("--test-iterations", nargs="+", type=int, default=[7000, 30000])
    p.add_argument("--save-iterations", nargs="+", type=int, default=[7000, 30000])
    p.add_argument("--checkpoint-iterations", nargs="+", type=int, default=[])
    p.add_argument("--start-checkpoint", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--disable-viewer", action="store_true", default=True)
    # --- MoGe-specific knobs ----------------------------------------------
    p.add_argument(
        "--depth-loss",
        choices=DEPTH_VARIANTS,
        default="none",
        help="Which depth-supervision loss to use (default: none = vanilla 3DGS).",
    )
    p.add_argument(
        "--lambda-d",
        type=float,
        default=0.1,
        help="Initial weight of depth loss in the total objective.",
    )
    p.add_argument(
        "--lambda-d-final",
        type=float,
        default=0.01,
        help="Final weight of depth loss after exponential decay.",
    )
    p.add_argument(
        "--depth-loss-until",
        type=int,
        default=30000,
        help="Iteration at which depth-loss weight reaches lambda_d_final and then becomes 0.",
    )
    # `lp` exposes -d / --depths and -s / --source_path / -m / --model_path / etc.
    return p, lp, op, pp


def main():
    parser, lp, op, pp = build_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print(f"[MOGE_3DGS] depth_loss={args.depth_loss}  lambda_d={args.lambda_d} -> {args.lambda_d_final} (until iter {args.depth_loss_until})")
    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )


if __name__ == "__main__":
    main()
