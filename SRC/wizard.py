"""Schema-driven config builders for the interactive CLI wizard.

The wizard's state machine, scene tables and run-discovery sub-wizards live in
``cli.py``. This module owns the part that used to hardcode every knob three
times over (``common_questions`` / ``build_servo_frames_config`` /
``build_trajectory_config``): it walks :data:`config_schema.FIELDS` and turns
each :class:`~config_schema.Field` into the right questionary prompt, so the
wizard's parameters, defaults and validation come from the same single source
of truth as the CLI flags and the runners.

``cli.py`` passes in a small ``ctx`` namespace holding the prompt helpers that
are closures over the live questionary session (``q_text`` / ``q_select`` /
``q_confirm`` / ``run_prompt`` / ``console`` / ``Choice``), so the existing
copy-settings and fast-forward behaviour is preserved unchanged.
"""

from __future__ import annotations

import config_schema as S
from config_schema import (
    BOOL,
    CHOICE,
    FEATURE,
    OPT_FLOAT,
    OPT_INT,
    OPT_STR,
)


# Cosmetic labels for a couple of choice values; every other choice renders as
# its bare schema value + the field help.
_CHOICE_LABELS = {
    ("depth_mode", "intrinsic"): "intrinsic  — scene.render_depth()",
    ("depth_mode", "learned"): "learned    — MoGe2",
}

_RENDER_SCALE_FIELD = {
    "nerf": "nerf_render_scale",
    "gs": "gs_render_scale",
    "mesh": "mesh_render_scale",
}

_CASTERS = {"int": int, "opt_int": int, "float": float, "opt_float": float}
_OPTIONAL_TYPES = {OPT_INT, OPT_FLOAT, OPT_STR}


def _label(field):
    return (field.help or field.name).rstrip(".")


def _field_question(field, kind, ctx, renderers=None, default_override=None):
    """Map one Field to a questionary question dict via the ctx helpers."""
    Choice = ctx.Choice
    default = field.default_for(kind) if default_override is None else default_override
    t = field.type

    if field.name == "renderer":
        opts = list(renderers) if renderers else list(field.choices)
        return ctx.q_select(
            "renderer", "Renderer:", [Choice(r, value=r) for r in opts], default=opts[0]
        )
    if t == CHOICE:
        choices = [
            Choice(_CHOICE_LABELS.get((field.name, c), c), value=c)
            for c in field.choices
        ]
        return ctx.q_select(field.name, f"{_label(field)}:", choices, default=default)
    if t == FEATURE:
        return ctx.q_select(
            field.name,
            f"{_label(field)}:",
            [Choice(m, value=m) for m in S.FEATURE_METHODS],
            default=default,
        )
    if t == BOOL:
        return ctx.q_confirm(
            field.name, f"{_label(field)}?", default=bool(default) if default is not None else True
        )
    caster = _CASTERS.get(t, str)
    # index_away only matters when no explicit target frame was given.
    when = (lambda a: a.get("target_index") is None) if field.name == "index_away" else None
    return ctx.q_text(
        field.name,
        f"{_label(field)}:",
        default,
        caster,
        optional=t in _OPTIONAL_TYPES,
        when=when,
    )


def _section_fields(kind, controller, sections):
    """Wizard-eligible fields for a kind in the given sections, in registry
    order, gated by controller applicability. Advanced knobs and the
    ask=False selectors (scene / renderer / controller) are excluded here;
    renderer is asked explicitly by :func:`build_config`."""
    out = []
    for f in S.fields_for(kind):
        if not f.ask or f.section not in sections:
            continue
        if f.applies not in (None, controller):
            continue
        out.append(f)
    return out


# Section groupings that reproduce the historical wizard's rule-separated
# prompt batches. Ordering within a batch follows the registry.
_SERVO_GROUPS = (
    ("Frame selection", ("frame",)),
    ("Servo loop", ("loop", "loop_shared", "stopping")),
)
_TRAJECTORY_GROUPS = (
    ("Trajectory pacing", ("pacing", "loop_shared", "frame")),
    ("Stopping + viz", ("stopping",)),
)


def build_config(kind, controller, scene_selector, renderers, ctx, run_tag_default=None):
    """Build an experiment config dict by prompting through the schema.

    kind            config_schema.TRAJECTORY or SERVO_FRAMES
    controller      "ibvs" or "photometric" (chosen in an earlier wizard state)
    scene_selector  {"datasets": [...]} or {"scene_dir": name}
    renderers       renderers detected for the chosen scene(s)
    ctx             namespace with q_text/q_select/q_confirm/run_prompt/console/Choice
    run_tag_default seed for the trajectory run_tag prompt (compare sweeps)
    """
    cfg = {"kind": kind, "controller": controller}
    cfg.update(scene_selector)

    if controller == "photometric":
        ctx.console.rule("[bold magenta]Photometric controller knobs[/]")

    # Common block: renderer (scene-specific choices) + depth + controller knobs.
    common = [_field_question(S.field_by_name("renderer"), kind, ctx, renderers)]
    common += [
        _field_question(f, kind, ctx, renderers)
        for f in _section_fields(kind, controller, ("common",))
    ]
    ctrl_section = "controller_photo" if controller == "photometric" else "controller_ibvs"
    common += [
        _field_question(f, kind, ctx, renderers)
        for f in _section_fields(kind, controller, (ctrl_section,))
    ]
    cfg.update(ctx.run_prompt(common))

    # Trajectory: the render-scale prompt depends on the chosen renderer.
    if kind == S.TRAJECTORY:
        scale_name = _RENDER_SCALE_FIELD.get(cfg.get("renderer"))
        if scale_name:
            field = S.field_by_name(scale_name)
            cfg.update(ctx.run_prompt([_field_question(field, kind, ctx, renderers)]))

    groups = _TRAJECTORY_GROUPS if kind == S.TRAJECTORY else _SERVO_GROUPS
    for title, sections in groups:
        fields = _section_fields(kind, controller, sections)
        if not fields:
            continue
        ctx.console.rule(f"[bold magenta]{title}[/]")
        questions = []
        for f in fields:
            override = run_tag_default if (f.name == "run_tag" and run_tag_default) else None
            questions.append(_field_question(f, kind, ctx, renderers, default_override=override))
        cfg.update(ctx.run_prompt(questions))

    return cfg
