"""Telegram control bot for launching SERVIS experiments from your phone.

Instead of building a config from scratch, the bot lists your most **recent
runs** (every ``RUNS/*`` dir with a ``config.resolved.json``); you tap one to
reuse its exact settings, optionally tweak a few knobs, then launch. Trajectory
runs can also be *resumed* in place. The chosen config is written under
``CONFIGS/`` and run as ``python cli.py <command> --config <path>`` — a
subprocess *on this machine* whose output is streamed back to the chat.

Why reuse a recent run: a resolved config is a complete, known-good snapshot
(scene, renderer, controller, every gain/threshold), so relaunching or tweaking
one is both faster and safer on a phone than re-picking everything.

Why a subprocess and not an in-process call: it keeps the bot process light (no
torch import) and responsive, isolates a crashing run from the bot, and lets
``/stop`` kill a run cleanly.

Why Telegram: it long-polls Telegram's servers, so it works from anywhere behind
your home NAT with zero port-forwarding, and inline keyboards give real tap-able
menu buttons on mobile.

Setup
-----
1. In Telegram, message @BotFather -> ``/newbot`` -> copy the HTTP API token.
2. Find your own numeric user id (message @userinfobot, or run this bot once and
   it will tell you the id of anyone who is denied).
3. Export the two required env vars and run:

       export SERVIS_BOT_TOKEN="123456:ABC-..."
       export SERVIS_BOT_ALLOWED_USERS="<your-numeric-id>"   # comma-separated
       python cli.py bot          # or: python phone_bot.py

Commands
--------
    /start, /run   list recent runs to relaunch
    /status        show whether a run is in progress and its recent output
    /stop          terminate the current run
    /cancel        discard the in-progress selection
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config_schema as S
from experiment_config import (
    SERVO_FRAMES_CONFIG_KEYS,
    TRAJECTORY_CONFIG_KEYS,
    parse_cli_overrides,
)

# telegram is only needed to actually serve; imported at module top so a missing
# dependency fails loudly with a clear message rather than deep in a handler.
try:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "python-telegram-bot is not installed. Install it with:\n"
        "    pip install 'python-telegram-bot>=21,<22'"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "SRC"
CONFIG_ROOT = PROJECT_ROOT / "CONFIGS"
RUNS_ROOT = PROJECT_ROOT / "RUNS"

# Telegram messages cap at 4096 chars; we keep the streamed tail well under that.
_MAX_TAIL_CHARS = 3500
_STREAM_INTERVAL_S = 4.0
# How many recent runs to list as buttons.
_RUN_LIST_LIMIT = 12

# config kind -> cli.py subcommand.
_COMMAND = {S.TRAJECTORY: "trajectory", S.SERVO_FRAMES: "servo-frames"}

# Only the render-scale field matching the renderer is worth showing in Tweak.
_RENDER_SCALE_FIELD = {
    "nerf": "nerf_render_scale", "gs": "gs_render_scale", "mesh": "mesh_render_scale",
}


# ---------------------------------------------------------------------------
# Run discovery + config extraction (light; no torch import)
# ---------------------------------------------------------------------------


def _run_scene(cfg) -> str:
    ds = cfg.get("datasets")
    if isinstance(ds, list) and ds:
        return str(ds[0])
    if isinstance(ds, str) and ds.strip():
        return ds.split(",")[0].strip()
    sd = cfg.get("scene_dir")
    if sd:
        return Path(str(sd)).name
    return "?"


def _run_method(cfg) -> str:
    parts = [cfg.get("renderer"), cfg.get("controller"), cfg.get("depth_mode")]
    if cfg.get("controller") == "ibvs" and cfg.get("feature_method"):
        parts.append(cfg["feature_method"])
    return "/".join(str(p) for p in parts if p)


def _tasks_done(run_dir: Path) -> int:
    csv_path = run_dir / "per_task_errors.csv"
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)  # minus header row
    except OSError:
        return 0


def _resumable(run_dir: Path) -> bool:
    """A trajectory run can be resumed only if it saved the resume artifacts."""
    p = Path(run_dir)
    return all((p / name).exists()
               for name in ("per_task_errors.csv", "sim_traj.tum", "gt_traj.tum"))


def discover_runs(limit: int = _RUN_LIST_LIMIT):
    """Recent runs (newest first) that carry a resolved config we can relaunch."""
    if not RUNS_ROOT.is_dir():
        return []
    runs = []
    for entry in RUNS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        cfg_path = entry / "config.resolved.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, ValueError):
            continue
        runs.append({
            "path": entry,
            "cfg": cfg,
            "mtime": entry.stat().st_mtime,
            "tasks_done": _tasks_done(entry),
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs[:limit]


def _key_map(kind):
    return TRAJECTORY_CONFIG_KEYS if kind == S.TRAJECTORY else SERVO_FRAMES_CONFIG_KEYS


def write_config(cfg, scene_name):
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    controller = cfg.get("controller", "ibvs")
    kind = cfg.get("kind", "trajectory")
    name = f"bot_{kind}_{scene_name}_{controller}_{stamp}.json"
    path = CONFIG_ROOT / name
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Tweak: which knobs the override screen lists (same set the wizard prompts)
# ---------------------------------------------------------------------------

# Fields that describe *what/where* rather than a tunable knob; not offered.
_STRUCTURAL = {"renderer", "controller", "depth_mode", "feature_method",
               "datasets", "scene_dir", "use_takes", "take_indices"}


def _tunable_fields(kind, controller, renderer=None):
    keep_scale = _RENDER_SCALE_FIELD.get(renderer)
    out = []
    for f in S.fields_for(kind):
        if not f.ask or f.name in _STRUCTURAL:
            continue
        if f.applies not in (None, controller):
            continue
        if f.section in (None, "advanced"):
            continue
        if f.section == "render" and renderer is not None and f.name != keep_scale:
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Job runner: at most one subprocess at a time, output streamed to the chat.
# ---------------------------------------------------------------------------


class JobManager:
    """Owns the single running experiment subprocess and its captured output."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.command: str | None = None
        self.started_at: float | None = None
        self.log_lines: list[str] = []
        self._lock = asyncio.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def start(self, command: str, config_path: Path,
                    resume_dir: str | None = None) -> bool:
        """Launch ``cli.py <command> --config … [--resume …]``. False if busy."""
        async with self._lock:
            if self.is_running():
                return False
            argv = [sys.executable, "cli.py", command, "--config", str(config_path)]
            if resume_dir:
                argv += ["--resume", str(resume_dir)]
            self.command = command + (" (resume)" if resume_dir else "")
            self.started_at = time.time()
            self.log_lines = []
            self.proc = subprocess.Popen(
                argv,
                cwd=str(SRC_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return True

    def stop(self) -> bool:
        """SIGINT the run (graceful). False if nothing is running."""
        if not self.is_running():
            return False
        assert self.proc is not None
        try:
            self.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return False
        return True

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.time() - self.started_at

    def tail(self, max_chars: int = _MAX_TAIL_CHARS) -> str:
        text = "".join(self.log_lines)
        if len(text) > max_chars:
            text = "…\n" + text[-max_chars:]
        return text or "(no output yet)"


JOBS = JobManager()


async def _stream_job(app: Application, chat_id: int) -> None:
    """Read the subprocess stdout to EOF, posting a periodic tail to the chat."""
    proc = JOBS.proc
    if proc is None or proc.stdout is None:
        return

    loop = asyncio.get_running_loop()
    last_sent = 0.0
    status_msg = await app.bot.send_message(
        chat_id, "⏳ starting run…", parse_mode=ParseMode.HTML
    )

    async def push(final: bool = False) -> None:
        nonlocal last_sent
        elapsed = int(JOBS.elapsed())
        icon = "✅" if final and proc.returncode == 0 else "❌" if final else "⏳"
        header = f"{icon} <b>{html.escape(JOBS.command or '')}</b> · {elapsed}s"
        if final:
            header += f" · exit {proc.returncode}"
        try:
            await app.bot.edit_message_text(
                f"{header}\n<pre>{html.escape(JOBS.tail())}</pre>",
                chat_id=chat_id,
                message_id=status_msg.message_id,
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            # Telegram rejects a no-op edit ("message is not modified"); ignore.
            pass
        last_sent = time.time()

    def readline() -> str:
        return proc.stdout.readline()

    while True:
        line = await loop.run_in_executor(None, readline)
        if line:
            JOBS.log_lines.append(line)
        elif proc.poll() is not None:
            break
        if time.time() - last_sent >= _STREAM_INTERVAL_S:
            await push()

    proc.wait()
    await push(final=True)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


def _allowed_users() -> set[int]:
    raw = os.environ.get("SERVIS_BOT_ALLOWED_USERS", "").strip()
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                ids.add(int(part))
            except ValueError:
                pass
    return ids


async def _deny_if_unauthorized(update: Update) -> bool:
    """Return True (and reply) when the sender is not on the allowlist."""
    user = update.effective_user
    if user is not None and user.id in _allowed_users():
        return False
    who = user.id if user else "unknown"
    if update.effective_message:
        await update.effective_message.reply_text(
            f"⛔ Not authorized. Your Telegram user id is: {who}\n"
            f"Add it to SERVIS_BOT_ALLOWED_USERS to grant access."
        )
    return True


# ---------------------------------------------------------------------------
# Conversation: pick a recent run -> confirm (tweak) -> run / resume
# ---------------------------------------------------------------------------


def _kb(rows):
    return InlineKeyboardMarkup(rows)


def _buttons(step, options):
    """options: list[(label, value)] -> one button per row (callback 'step:value')."""
    return [[InlineKeyboardButton(label, callback_data=f"{step}:{value}")]
            for label, value in options]


def _run_button_label(idx, r) -> str:
    when = datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")
    scene = _run_scene(r["cfg"])
    method = _run_method(r["cfg"])
    return f"{idx + 1}. {when} · {scene} · {method} · {r['tasks_done']}t"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_unauthorized(update):
        return
    context.user_data.clear()
    runs = discover_runs()
    if not runs:
        await update.effective_message.reply_text(
            f"No runs with a resolved config found under {RUNS_ROOT}.\n"
            f"Launch one from the terminal wizard first, then relaunch it here."
        )
        return
    context.user_data["_runs"] = runs
    rows = _buttons("pick", [(_run_button_label(i, r), str(i))
                             for i, r in enumerate(runs)])
    await update.effective_message.reply_text(
        "🛰️ <b>SERVIS launcher</b>\nPick a recent run to relaunch:",
        reply_markup=_kb(rows),
        parse_mode=ParseMode.HTML,
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_unauthorized(update):
        return
    context.user_data.clear()
    await update.effective_message.reply_text("Selection discarded. /start to begin again.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_unauthorized(update):
        return
    if JOBS.is_running():
        await update.effective_message.reply_text(
            f"⏳ <b>{html.escape(JOBS.command or '')}</b> running · {int(JOBS.elapsed())}s\n"
            f"<pre>{html.escape(JOBS.tail())}</pre>",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.effective_message.reply_text("Idle. /start to relaunch a run.")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_unauthorized(update):
        return
    if JOBS.stop():
        await update.effective_message.reply_text("🛑 sent stop signal to the running job.")
    else:
        await update.effective_message.reply_text("Nothing is running.")


def _base_config(ud) -> dict:
    """The picked run's resolved config, filtered to valid keys for its kind."""
    kind = ud["_kind"]
    key_map = _key_map(kind)
    cfg = {k: v for k, v in ud["_run"]["cfg"].items() if k in key_map}
    cfg["kind"] = kind
    return cfg


def _build_config(ud) -> dict:
    """Base config from the run + any typed tweaks (tweaks win)."""
    cfg = _base_config(ud)
    cfg.update(ud.get("_overrides") or {})
    return cfg


def _summary(ud, cfg) -> str:
    lines = [
        f"from run: {Path(ud['_run']['path']).name}",
        f"scene: {_run_scene(cfg)}",
        f"renderer: {cfg.get('renderer')}",
        f"controller: {cfg.get('controller')}",
        f"depth: {cfg.get('depth_mode')}",
    ]
    if cfg.get("controller") == "ibvs" and cfg.get("feature_method"):
        lines.append(f"feature: {cfg['feature_method']}")
    tweaks = ud.get("_overrides") or {}
    if tweaks:
        lines.append("tweaks: " + ", ".join(f"{k}={v}" for k, v in sorted(tweaks.items())))
    return "\n".join(lines)


def _confirm_message(ud):
    """(text, keyboard) for the launch-confirm screen; caches ud['_cfg']."""
    cfg = _build_config(ud)
    ud["_cfg"] = cfg
    opts = [("▶️ Run fresh", "run"), ("⚙️ Tweak settings", "tweak")]
    if ud["_kind"] == S.TRAJECTORY and _resumable(Path(ud["_run"]["path"])):
        opts.insert(1, ("🔁 Resume this run", "resume"))
    opts.append(("✖️ Abort", "abort"))
    text = f"<b>Ready to launch</b>\n<pre>{html.escape(_summary(ud, cfg))}</pre>"
    return text, _kb(_buttons("go", opts))


async def _show_confirm(query, ud) -> None:
    text, kb = _confirm_message(ud)
    await query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)


async def _prompt_overrides(query, ud) -> None:
    """Show the run's current knob values and wait for typed key=value overrides."""
    ud["_awaiting_overrides"] = True
    kind = ud["_kind"]
    cfg = _build_config(ud)
    controller = cfg.get("controller")
    renderer = cfg.get("renderer")
    lines = [
        f"{f.name} = {cfg.get(f.name, f.default_for(kind))}"
        for f in _tunable_fields(kind, controller, renderer)
    ]
    gain_field = "gain_photo" if controller == "photometric" else "gain"
    await query.edit_message_text(
        "⚙️ <b>Current settings</b> (from the run unless tweaked):\n"
        f"<pre>{html.escape(chr(10).join(lines))}</pre>\n"
        "Send overrides as <code>key=value</code>, space/comma separated — e.g.\n"
        f"<code>{gain_field}=0.5 mini_iterations=300</code>\n"
        "Any schema field works by name (advanced knobs too). /cancel to abort.",
        parse_mode=ParseMode.HTML,
    )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _deny_if_unauthorized(update):
        return
    query = update.callback_query
    await query.answer()
    step, _, value = (query.data or "").partition(":")
    ud = context.user_data

    if step == "pick":
        runs = ud.get("_runs") or []
        idx = int(value) if value.isdigit() else -1
        if not (0 <= idx < len(runs)):
            await query.edit_message_text("That run list expired. Send /start again.")
            return
        r = runs[idx]
        ud["_run"] = {"path": str(r["path"]), "cfg": r["cfg"]}
        ud["_kind"] = r["cfg"].get("kind", S.TRAJECTORY)
        ud["_overrides"] = {}
        ud["_awaiting_overrides"] = False
        await _show_confirm(query, ud)

    elif step == "go":
        if "_run" not in ud:
            await query.edit_message_text("Nothing selected. Send /start.")
            return
        if value == "run":
            await _launch(update, context, resume=False)
        elif value == "resume":
            await _launch(update, context, resume=True)
        elif value == "tweak":
            await _prompt_overrides(query, ud)
        else:  # abort
            ud.clear()
            await query.edit_message_text("Aborted. /start to begin again.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Free-text handler: applies typed key=value overrides on the Tweak screen."""
    if await _deny_if_unauthorized(update):
        return
    ud = context.user_data
    msg = update.effective_message
    if not ud.get("_awaiting_overrides") or "_run" not in ud:
        await msg.reply_text("Send /start to relaunch a run.")
        return

    kind = ud["_kind"]
    tokens = [t for t in (msg.text or "").replace(",", " ").split() if t]
    try:
        overrides = parse_cli_overrides(tokens, _key_map(kind), kind)
    except (KeyError, ValueError) as exc:
        await msg.reply_text(f"⚠️ {html.escape(str(exc))}\n\nTry again, or /cancel.")
        return

    ud.setdefault("_overrides", {}).update(overrides)
    ud["_awaiting_overrides"] = False
    applied = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "(nothing)"
    text, kb = _confirm_message(ud)
    await msg.reply_text(
        f"✅ applied: <code>{html.escape(applied)}</code>\n\n{text}",
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
    )


async def _launch(update: Update, context: ContextTypes.DEFAULT_TYPE, resume: bool) -> None:
    ud = context.user_data
    query = update.callback_query
    cfg = ud.get("_cfg") or _build_config(ud)
    kind = ud["_kind"]
    command = _COMMAND.get(kind, "trajectory")
    run_path = ud["_run"]["path"]

    if JOBS.is_running():
        await query.edit_message_text(
            "⚠️ A run is already in progress. /status to watch it, /stop to cancel it."
        )
        return

    config_path = write_config(cfg, _run_scene(cfg))
    started = await JOBS.start(command, config_path,
                               resume_dir=run_path if resume else None)
    if not started:
        await query.edit_message_text("⚠️ Could not start — another run is active.")
        return

    rel = config_path.relative_to(PROJECT_ROOT)
    verb = "resuming" if resume else "launched"
    detail = f"resume: <code>{html.escape(Path(run_path).name)}</code>\n" if resume else ""
    await query.edit_message_text(
        f"🚀 {verb} <b>{html.escape(command)}</b>\n{detail}"
        f"config: <code>{html.escape(str(rel))}</code>\n"
        f"streaming output below… (/status, /stop)",
        parse_mode=ParseMode.HTML,
    )
    chat_id = update.effective_chat.id
    context.application.create_task(_stream_job(context.application, chat_id))
    ud.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_application() -> Application:
    token = os.environ.get("SERVIS_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "SERVIS_BOT_TOKEN is not set. Get a token from @BotFather and export it:\n"
            "    export SERVIS_BOT_TOKEN='123456:ABC-...'"
        )
    if not _allowed_users():
        raise SystemExit(
            "SERVIS_BOT_ALLOWED_USERS is not set. Export your numeric Telegram user id:\n"
            "    export SERVIS_BOT_ALLOWED_USERS='123456789'"
        )

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler(["start", "run"], cmd_start))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app


def run(args=None):
    """CLI entry point (also usable as ``python phone_bot.py``)."""
    app = build_application()
    print("SERVIS phone bot: polling Telegram (Ctrl-C to stop)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
