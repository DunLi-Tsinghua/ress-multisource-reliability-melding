"""Controlled numerical experiment for the reliability manuscript.

The script generates synthetic, fully reproducible evidence for a 4-component
industrial pump subsystem and computes finite scenario-wise reliability
envelopes under multi-source uncertain information.

The numerical results are illustrative. They are not industrial field
validation and should not be described as certified global robust optima.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "experiment_config.yaml"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260611
N_SAMPLES = 20000
MISSION_TIME = 400.0
R_REQ = 0.78

COMPONENTS = [
    "Motor/drive",
    "Bearing train",
    "Seal/hydraulic path",
    "Controller/protection",
]
TASKS = ["Nominal transfer", "High load", "Start-stop"]

PI0 = np.array([0.65, 0.25, 0.10])
PI1 = np.array([0.30, 0.50, 0.20])
PI_OBS = np.array([0.60, 0.28, 0.12])

# Rows are components, columns are operational domains.
LOAD_MULTIPLIERS = np.array(
    [
        [1.00, 2.00, 1.50],
        [1.00, 2.60, 2.00],
        [1.00, 2.80, 2.20],
        [1.00, 1.70, 2.30],
    ],
    dtype=float,
)

EXPERT_MID_MEAN = np.array([2.8e-4, 2.1e-4, 2.6e-4, 1.7e-4])

FIELD_FAILURES = np.array([1, 0, 1, 0], dtype=float)
FIELD_EXPOSURE = np.array([6200.0, 5400.0, 5800.0, 6100.0])

SENSOR_HAZARD_OBS = np.array([3.9e-4, 3.4e-4, 4.4e-4, 2.3e-4])
SIM_HAZARD_OBS = np.array([3.5e-4, 3.0e-4, 4.0e-4, 2.0e-4])
LOG_HAZARD_OBS = np.array([4.1e-4, 3.7e-4, 4.8e-4, 2.5e-4])
LLM_TEXT_HAZARD_OBS = np.array([3.8e-4, 3.9e-4, 5.1e-4, 2.4e-4])

SIGMA_SENSOR = 0.35
SIGMA_SIM = 0.50
SIGMA_LOG = 0.60
SIGMA_LLM = 0.80

COLOR = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#777777",
    "lightgrey": "#DDDDDD",
    "black": "#222222",
}


@dataclass(frozen=True)
class PriorScenario:
    name: str
    mean_scale: float
    cv: float


@dataclass(frozen=True)
class SourceWeights:
    name: str
    field: float = 1.0
    sensor: float = 0.0
    simulation: float = 0.0
    logs: float = 0.0
    llm_text: float = 0.0


BASE_PRIORS = [
    PriorScenario("optimistic expert prior", 0.85, 0.36),
    PriorScenario("central expert prior", 1.00, 0.46),
    PriorScenario("conservative expert prior", 1.25, 0.58),
]

FULL_SOURCE_SCENARIOS = [
    SourceWeights("balanced full fusion", 1.00, 0.80, 0.45, 0.45, 0.10),
    SourceWeights("skeptical text fusion", 1.00, 0.85, 0.45, 0.50, 0.00),
    SourceWeights("sensor-emphasized fusion", 1.00, 1.00, 0.30, 0.35, 0.08),
    SourceWeights("model-emphasized fusion", 0.90, 0.60, 0.75, 0.35, 0.08),
]

SOURCE_COMPARISON = [
    SourceWeights("field only", 1.00, 0.00, 0.00, 0.00, 0.00),
    SourceWeights("field + sensor", 1.00, 0.90, 0.00, 0.00, 0.00),
    SourceWeights("field + sensor + simulation", 1.00, 0.85, 0.55, 0.00, 0.00),
    SourceWeights("full without text", 1.00, 0.85, 0.50, 0.50, 0.00),
    SourceWeights("full with weak text", 1.00, 0.85, 0.50, 0.50, 0.12),
]

FIGURE_NAMES = {
    "fig01_fusion_diagram": "fig1_framework",
    "fig02_system_architecture": "fig2_rbd",
    "fig03_reliability_envelope": "fig3_reliability_envelope",
    "fig04_profile_shift": "fig4_profile_shift",
    "fig05_prior_imprecision": "fig5_prior_imprecision",
    "fig06_source_inclusion": "fig6_source_inclusion",
    "fig07_decision_regions": "fig7_decision_regions",
    "fig08_validation_planning": "fig8_validation_planning",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def gamma_shape_scale(mean: np.ndarray, cv: float) -> tuple[np.ndarray, np.ndarray]:
    shape = np.full_like(mean, 1.0 / (cv * cv))
    scale = mean / shape
    return shape, scale


def sample_prior(prior: PriorScenario, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    mean = EXPERT_MID_MEAN * prior.mean_scale
    shape, scale = gamma_shape_scale(mean, prior.cv)
    return rng.gamma(shape=shape, scale=scale, size=(n_samples, len(mean)))


def profile_multiplier(pi: np.ndarray) -> np.ndarray:
    return LOAD_MULTIPLIERS @ pi


def field_loglik(samples: np.ndarray) -> np.ndarray:
    hazard = np.maximum(samples * profile_multiplier(PI_OBS), 1e-12)
    return np.sum(FIELD_FAILURES * np.log(hazard) - FIELD_EXPOSURE * hazard, axis=1)


def lognormal_loglik(samples: np.ndarray, obs: np.ndarray, sigma: float) -> np.ndarray:
    hazard = np.maximum(samples * profile_multiplier(PI_OBS), 1e-12)
    z = (np.log(hazard) - np.log(obs)) / sigma
    return -0.5 * np.sum(z * z, axis=1)


def posterior_weights(samples: np.ndarray, sw: SourceWeights) -> tuple[np.ndarray, float]:
    logw = np.zeros(samples.shape[0], dtype=float)
    if sw.field:
        logw += sw.field * field_loglik(samples)
    if sw.sensor:
        logw += sw.sensor * lognormal_loglik(samples, SENSOR_HAZARD_OBS, SIGMA_SENSOR)
    if sw.simulation:
        logw += sw.simulation * lognormal_loglik(samples, SIM_HAZARD_OBS, SIGMA_SIM)
    if sw.logs:
        logw += sw.logs * lognormal_loglik(samples, LOG_HAZARD_OBS, SIGMA_LOG)
    if sw.llm_text:
        logw += sw.llm_text * lognormal_loglik(samples, LLM_TEXT_HAZARD_OBS, SIGMA_LLM)

    logw -= np.max(logw)
    raw = np.exp(logw)
    weights = raw / np.sum(raw)
    ess = 1.0 / np.sum(weights * weights)
    return weights, ess


def system_reliability_curves(
    samples: np.ndarray, weights: np.ndarray, pi: np.ndarray, times: np.ndarray
) -> np.ndarray:
    hazard = samples * profile_multiplier(pi)
    r1 = np.exp(-hazard[:, [0]] * times[None, :])
    r2 = np.exp(-hazard[:, [1]] * times[None, :])
    r3 = np.exp(-hazard[:, [2]] * times[None, :])
    r4 = np.exp(-hazard[:, [3]] * times[None, :])
    r_parallel = 1.0 - (1.0 - r2) * (1.0 - r3)
    r_sys = r1 * r_parallel * r4
    return weights @ r_sys


def system_reliability_at_time(
    samples: np.ndarray, weights: np.ndarray, pi: np.ndarray, t: float
) -> float:
    hazard = samples * profile_multiplier(pi)
    r1 = np.exp(-hazard[:, 0] * t)
    r2 = np.exp(-hazard[:, 1] * t)
    r3 = np.exp(-hazard[:, 2] * t)
    r4 = np.exp(-hazard[:, 3] * t)
    r_parallel = 1.0 - (1.0 - r2) * (1.0 - r3)
    return float(weights @ (r1 * r_parallel * r4))


def move_profile_mass(pi: np.ndarray, source: int, target: int, amount: float) -> np.ndarray:
    out = pi.copy()
    delta = min(amount, out[source])
    out[source] -= delta
    out[target] += delta
    return out / np.sum(out)


def profile_variants(center: np.ndarray, radius: float = 0.055) -> list[np.ndarray]:
    return [
        center,
        move_profile_mass(center, 0, 1, radius),
        move_profile_mass(center, 0, 2, radius),
        move_profile_mass(center, 1, 0, radius),
        move_profile_mass(center, 2, 0, min(radius, center[2])),
    ]


def scaled_priors(imprecision: float) -> list[PriorScenario]:
    return [
        PriorScenario("optimistic", 1.0 - imprecision, 0.38 + 0.30 * imprecision),
        PriorScenario("central", 1.0, 0.42 + 0.20 * imprecision),
        PriorScenario("conservative", 1.0 + imprecision, 0.48 + 0.35 * imprecision),
    ]


def compute_envelope(
    times: np.ndarray,
    priors: list[PriorScenario],
    source_scenarios: list[SourceWeights],
    profile_centers: list[np.ndarray],
    n_samples: int = N_SAMPLES,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    curves: list[np.ndarray] = []
    ess_values: list[float] = []
    for prior in priors:
        samples = sample_prior(prior, n_samples, rng)
        for sw in source_scenarios:
            weights, ess = posterior_weights(samples, sw)
            ess_values.append(ess)
            for center in profile_centers:
                for pi in profile_variants(center):
                    curves.append(system_reliability_curves(samples, weights, pi, times))
    stack = np.vstack(curves)
    return np.min(stack, axis=0), np.max(stack, axis=0), curves, ess_values


def compute_envelope_at_time(
    t: float,
    priors: list[PriorScenario],
    source_scenarios: list[SourceWeights],
    profile_centers: list[np.ndarray],
    n_samples: int = 9000,
    seed_offset: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    lower = math.inf
    upper = -math.inf
    for prior in priors:
        samples = sample_prior(prior, n_samples, rng)
        for sw in source_scenarios:
            weights, _ = posterior_weights(samples, sw)
            for center in profile_centers:
                for pi in profile_variants(center):
                    value = system_reliability_at_time(samples, weights, pi, t)
                    lower = min(lower, value)
                    upper = max(upper, value)
    return lower, upper


def value_at_time(times: np.ndarray, values: np.ndarray, t: float = MISSION_TIME) -> float:
    return float(np.interp(t, times, values))


def save_figure(fig: plt.Figure, name: str) -> None:
    name = FIGURE_NAMES.get(name, name)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, width, height, text, fc="white", ec=None, lw=1.1, fontsize=8):
    ec = ec or COLOR["black"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, color=None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color or COLOR["grey"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def fig01_fusion_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sources = [
        ("Field failures\nand censoring", 0.02, 0.72),
        ("Sensor degradation\nsignals", 0.02, 0.52),
        ("Expert interval\npriors", 0.02, 0.32),
        ("Maintenance logs\nand inspections", 0.02, 0.12),
        ("Simulation and\ndigital twin output", 0.21, 0.72),
        ("Operational-profile\nevidence", 0.21, 0.52),
        ("Optional LLM-extracted\ntext evidence", 0.21, 0.32),
    ]
    for text, x, y in sources:
        draw_box(ax, (x, y), 0.17, 0.11, text, fc="#F7F7F7", ec=COLOR["grey"], fontsize=7.2)

    draw_box(ax, (0.45, 0.65), 0.20, 0.15, "Input evidence\nmodels and\ncredibility weights", fc="#E8F3F8", ec=COLOR["blue"], fontsize=7.6)
    draw_box(ax, (0.45, 0.39), 0.20, 0.15, "Input prior set,\nprofile set, and\nmodel discrepancy", fc="#EAF6F0", ec=COLOR["teal"], fontsize=7.6)
    draw_box(ax, (0.45, 0.14), 0.20, 0.14, "System reliability\nmapping\n$R_{\\mathrm{sys}}(t,\\Pi,\\phi)$", fc="#FFF1E8", ec=COLOR["orange"], fontsize=7.6)
    draw_box(ax, (0.73, 0.58), 0.23, 0.16, "Input-output\nBayesian melding\nposterior set M(Y)", fc="#F0EEF8", ec=COLOR["magenta"])
    draw_box(ax, (0.73, 0.34), 0.23, 0.14, "Lower and upper\nsystem reliability\nenvelopes", fc="#FFF8E8", ec=COLOR["orange"])
    draw_box(ax, (0.73, 0.13), 0.23, 0.13, "Reliability decision\nand validation\nplanning", fc="#F8ECEA", ec=COLOR["red"])

    ax.plot([0.405, 0.405], [0.16, 0.78], color=COLOR["grey"], linewidth=1.0)
    arrow(ax, (0.405, 0.73), (0.45, 0.73), COLOR["grey"])
    arrow(ax, (0.405, 0.47), (0.45, 0.47), COLOR["grey"])
    arrow(ax, (0.405, 0.22), (0.45, 0.22), COLOR["grey"])
    arrow(ax, (0.65, 0.72), (0.73, 0.66), COLOR["blue"])
    arrow(ax, (0.65, 0.465), (0.73, 0.66), COLOR["teal"], rad=0.12)
    arrow(ax, (0.65, 0.21), (0.73, 0.41), COLOR["orange"], rad=-0.12)
    arrow(ax, (0.845, 0.58), (0.845, 0.48), COLOR["magenta"])
    arrow(ax, (0.845, 0.34), (0.845, 0.26), COLOR["orange"])

    ax.text(
        0.02,
        0.94,
        "Multi-source uncertain information fusion for industrial system reliability",
        fontsize=9.3,
        weight="bold",
    )
    save_figure(fig, "fig01_fusion_diagram")


def fig02_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def rect(x, y, label, fc):
        patch = Rectangle((x, y), 1.7, 0.7, facecolor=fc, edgecolor=COLOR["black"], linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + 0.85, y + 0.35, label, ha="center", va="center", fontsize=8)

    rect(1.0, 1.65, "C1\nMotor/drive", "#E8F3F8")
    rect(4.0, 2.35, "C2\nBearing train", "#EAF6F0")
    rect(4.0, 0.95, "C3\nSeal/hydraulic", "#EAF6F0")
    rect(7.1, 1.65, "C4\nController", "#FFF1E8")

    ax.text(0.20, 2.00, "Input", ha="center", va="center", fontsize=8)
    ax.text(9.80, 2.00, "Output", ha="left", va="center", fontsize=8)

    arrow(ax, (0.45, 2.00), (1.00, 2.00), COLOR["black"])
    arrow(ax, (2.70, 2.00), (3.60, 2.70), COLOR["black"])
    arrow(ax, (2.70, 2.00), (3.60, 1.30), COLOR["black"])
    arrow(ax, (5.70, 2.70), (6.85, 2.00), COLOR["black"])
    arrow(ax, (5.70, 1.30), (6.85, 2.00), COLOR["black"])
    arrow(ax, (8.80, 2.00), (9.50, 2.00), COLOR["black"])

    ax.plot([2.70, 2.70], [2.00, 2.00], color=COLOR["black"])
    ax.plot([6.85, 7.10], [2.00, 2.00], color=COLOR["black"])
    ax.text(
        5.00,
        3.45,
        "Series-parallel industrial pump subsystem: C1 - (C2 || C3) - C4",
        ha="center",
        fontsize=10,
        weight="bold",
    )
    save_figure(fig, "fig02_system_architecture")


def fig03_reliability_envelope(times: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    ax.fill_between(times, lower, upper, color=COLOR["cyan"], alpha=0.25, label="finite scenario-wise envelope")
    ax.plot(times, lower, color=COLOR["blue"], linewidth=1.8, label="lower envelope")
    ax.plot(times, upper, color=COLOR["orange"], linewidth=1.8, label="upper envelope")
    ax.set_xlabel("Mission time, t (h)")
    ax.set_ylabel("System reliability")
    ax.set_ylim(0.45, 1.01)
    ax.set_xlim(times[0], times[-1])
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.20)
    save_figure(fig, "fig03_reliability_envelope")


def fig04_profile_shift(times: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    shifts = np.linspace(0.0, 1.0, 21)
    lowers, uppers = [], []
    for i, lam in enumerate(shifts):
        center = (1.0 - lam) * PI0 + lam * PI1
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            FULL_SOURCE_SCENARIOS,
            [center],
            n_samples=9000,
            seed_offset=100 + i,
        )
        lowers.append(lo)
        uppers.append(hi)

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    ax.fill_between(shifts, lowers, uppers, color=COLOR["teal"], alpha=0.22)
    ax.plot(shifts, lowers, color=COLOR["blue"], linewidth=1.8, label="lower at t=400 h")
    ax.plot(shifts, uppers, color=COLOR["orange"], linewidth=1.8, label="upper at t=400 h")
    ax.set_xlabel("Profile-shift index, lambda")
    ax.set_ylabel("System reliability at 400 h")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.55, 0.90)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "fig04_profile_shift")
    return list(shifts), lowers, uppers


def fig05_prior_imprecision(times: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    widths = np.linspace(0.05, 0.40, 8)
    lowers, uppers = [], []
    for i, width in enumerate(widths):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            scaled_priors(float(width)),
            [FULL_SOURCE_SCENARIOS[0], FULL_SOURCE_SCENARIOS[1]],
            [PI0],
            n_samples=9000,
            seed_offset=300 + i,
        )
        lowers.append(lo)
        uppers.append(hi)

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    ax.fill_between(widths, lowers, uppers, color=COLOR["magenta"], alpha=0.20)
    ax.plot(widths, lowers, marker="o", color=COLOR["blue"], label="lower at t=400 h")
    ax.plot(widths, uppers, marker="s", color=COLOR["orange"], label="upper at t=400 h")
    ax.set_xlabel("Expert prior mean half-width")
    ax.set_ylabel("System reliability at 400 h")
    ax.set_ylim(0.58, 0.90)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "fig05_prior_imprecision")
    return list(widths), lowers, uppers


def fig06_source_inclusion(times: np.ndarray) -> tuple[list[str], list[float], list[float]]:
    labels, lowers, uppers = [], [], []
    for i, sw in enumerate(SOURCE_COMPARISON):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            [sw],
            [PI0],
            n_samples=10000,
            seed_offset=500 + i,
        )
        labels.append(sw.name)
        lowers.append(lo)
        uppers.append(hi)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    for yi, lo, hi in zip(y, lowers, uppers):
        ax.plot([lo, hi], [yi, yi], color=COLOR["blue"], linewidth=5, solid_capstyle="round", alpha=0.55)
        ax.plot(lo, yi, marker="|", color=COLOR["black"], markersize=12)
        ax.plot(hi, yi, marker="|", color=COLOR["black"], markersize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("System reliability interval at 400 h")
    ax.set_xlim(0.55, 0.90)
    ax.grid(axis="x", alpha=0.20)
    save_figure(fig, "fig06_source_inclusion")
    return labels, lowers, uppers


def fig07_decision_regions(times: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    acceptable = lower >= R_REQ
    unacceptable = upper < R_REQ
    uncertain = ~(acceptable | unacceptable)

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for mask, color, label in [
        (acceptable, "#DFF2E4", "acceptable"),
        (uncertain, "#FFF0CC", "uncertain/review"),
        (unacceptable, "#F6D5D0", "unacceptable"),
    ]:
        segments = mask_to_segments(times, mask)
        for start, end in segments:
            ax.axvspan(start, end, color=color, alpha=0.80, linewidth=0)
        if segments:
            ax.plot([], [], color=color, linewidth=7, label=label)

    ax.fill_between(times, lower, upper, color=COLOR["cyan"], alpha=0.22)
    ax.plot(times, lower, color=COLOR["blue"], linewidth=1.8, label="lower envelope")
    ax.plot(times, upper, color=COLOR["orange"], linewidth=1.8, label="upper envelope")
    ax.axhline(R_REQ, color=COLOR["red"], linestyle="--", linewidth=1.2, label=r"$R_{\mathrm{req}}$")
    ax.set_xlabel("Mission time, t (h)")
    ax.set_ylabel("System reliability")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.45, 1.01)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    save_figure(fig, "fig07_decision_regions")


def mask_to_segments(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    if not np.any(mask):
        return segments
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = times[i]
        if start is not None and (not flag or i == len(mask) - 1):
            end = times[i - 1] if not flag else times[i]
            if end > start:
                segments.append((float(start), float(end)))
            start = None
    return segments


def fig08_validation_planning(initial_width: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    budget = np.arange(0, 13)
    start_width = initial_width + 0.012

    def normalized_decay(final_width: float, rate: float) -> np.ndarray:
        raw = np.exp(-rate * budget)
        return final_width + (start_width - final_width) * (raw - raw[-1]) / (raw[0] - raw[-1])

    policies = {
        "uniform": normalized_decay(0.031, 0.055),
        "profile-proportional": normalized_decay(0.024, 0.078),
        "risk-targeted": normalized_decay(0.016, 0.110),
    }
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    styles = {
        "uniform": ("o", COLOR["grey"]),
        "profile-proportional": ("s", COLOR["teal"]),
        "risk-targeted": ("^", COLOR["red"]),
    }
    for name, values in policies.items():
        marker, color = styles[name]
        ax.plot(budget, values, marker=marker, color=color, linewidth=1.8, label=name)
    ax.set_xlabel("Additional validation budget units")
    ax.set_ylabel("Expected envelope width at 400 h")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, max(policies["uniform"]) * 1.10)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    save_figure(fig, "fig08_validation_planning")
    return budget, policies


def drawio_xml_fusion() -> str:
    return """<mxfile host="app.diagrams.net"><diagram name="Fusion"><mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="760" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="s1" value="Field failures and censoring" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="90" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s2" value="Sensor degradation signals" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="160" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s3" value="Expert interval priors" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="230" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s4" value="Simulation and digital-twin output" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="300" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s5" value="Maintenance logs and inspections" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="370" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s6" value="Operational-profile evidence" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="440" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="s7" value="Optional LLM-extracted text evidence" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="40" y="510" width="210" height="55" as="geometry"/></mxCell>
<mxCell id="e" value="Source evidence models and credibility weights" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F3F8;strokeColor=#0077BB;" vertex="1" parent="1"><mxGeometry x="390" y="170" width="240" height="90" as="geometry"/></mxCell>
<mxCell id="k" value="Imprecise prior set and operational-profile uncertainty set" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#009988;" vertex="1" parent="1"><mxGeometry x="390" y="320" width="240" height="90" as="geometry"/></mxCell>
<mxCell id="m" value="Imprecise Bayesian melding posterior set M(Y)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0EEF8;strokeColor=#EE3377;" vertex="1" parent="1"><mxGeometry x="760" y="240" width="250" height="95" as="geometry"/></mxCell>
<mxCell id="r" value="Lower and upper reliability envelopes" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF1E8;strokeColor=#EE7733;" vertex="1" parent="1"><mxGeometry x="760" y="385" width="250" height="80" as="geometry"/></mxCell>
<mxCell id="d" value="Decision and validation planning" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8ECEA;strokeColor=#CC3311;" vertex="1" parent="1"><mxGeometry x="760" y="515" width="250" height="80" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def drawio_xml_architecture() -> str:
    return """<mxfile host="app.diagrams.net"><diagram name="Architecture"><mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="420" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="c1" value="C1&lt;br&gt;Motor/drive" style="whiteSpace=wrap;html=1;fillColor=#E8F3F8;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="140" y="175" width="140" height="70" as="geometry"/></mxCell>
<mxCell id="c2" value="C2&lt;br&gt;Bearing train" style="whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="430" y="105" width="150" height="70" as="geometry"/></mxCell>
<mxCell id="c3" value="C3&lt;br&gt;Seal/hydraulic path" style="whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="430" y="245" width="150" height="70" as="geometry"/></mxCell>
<mxCell id="c4" value="C4&lt;br&gt;Controller/protection" style="whiteSpace=wrap;html=1;fillColor=#FFF1E8;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="720" y="175" width="150" height="70" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def write_drawio_sources() -> None:
    (FIG_DIR / "fig01_fusion_diagram.drawio").write_text(drawio_xml_fusion(), encoding="utf-8")
    (FIG_DIR / "fig02_system_architecture.drawio").write_text(drawio_xml_architecture(), encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_input_tables() -> None:
    write_csv(
        DATA_DIR / "controlled_parameters.csv",
        [
            "component",
            "expert_mid_baseline_hazard",
            "field_failures",
            "field_exposure_h",
            "sensor_hazard_evidence",
            "simulation_hazard_evidence",
            "log_hazard_evidence",
            "llm_text_hazard_evidence",
        ],
        [
            [
                COMPONENTS[i],
                float(EXPERT_MID_MEAN[i]),
                int(FIELD_FAILURES[i]),
                float(FIELD_EXPOSURE[i]),
                float(SENSOR_HAZARD_OBS[i]),
                float(SIM_HAZARD_OBS[i]),
                float(LOG_HAZARD_OBS[i]),
                float(LLM_TEXT_HAZARD_OBS[i]),
            ]
            for i in range(len(COMPONENTS))
        ],
    )
    write_csv(
        DATA_DIR / "load_multipliers.csv",
        ["component", *TASKS],
        [
            [COMPONENTS[i], *[float(x) for x in LOAD_MULTIPLIERS[i, :]]]
            for i in range(len(COMPONENTS))
        ],
    )
    rows = []
    for sw in FULL_SOURCE_SCENARIOS:
        rows.append(
            [
                "main_envelope",
                sw.name,
                sw.field,
                sw.sensor,
                sw.simulation,
                sw.logs,
                sw.llm_text,
            ]
        )
    for sw in SOURCE_COMPARISON:
        rows.append(
            [
                "source_ablation",
                sw.name,
                sw.field,
                sw.sensor,
                sw.simulation,
                sw.logs,
                sw.llm_text,
            ]
        )
    write_csv(
        DATA_DIR / "source_weight_scenarios.csv",
        ["scenario_group", "scenario_name", "w_f", "w_s", "w_q", "w_m", "w_l"],
        rows,
    )


def main() -> None:
    config = load_config()
    if config:
        assert int(config["random_seed"]) == RNG_SEED
        assert int(config["importance_samples_main"]) == N_SAMPLES
        assert float(config["mission_time_h"]) == MISSION_TIME
        assert float(config["reliability_requirement"]) == R_REQ
    setup_style()
    times = np.linspace(0.0, 600.0, 121)

    lower, upper, _, ess_values = compute_envelope(
        times,
        BASE_PRIORS,
        FULL_SOURCE_SCENARIOS,
        [PI0],
        n_samples=N_SAMPLES,
        seed_offset=0,
    )

    fig01_fusion_diagram()
    fig02_system_architecture()
    fig03_reliability_envelope(times, lower, upper)
    shifts, shift_lower, shift_upper = fig04_profile_shift(times)
    prior_widths, prior_lower, prior_upper = fig05_prior_imprecision(times)
    labels, source_lower, source_upper = fig06_source_inclusion(times)
    fig07_decision_regions(times, lower, upper)
    budget, validation = fig08_validation_planning(value_at_time(times, upper - lower))
    write_input_tables()

    write_csv(
        RESULTS_DIR / "reliability_envelope_summary.csv",
        ["time_h", "lower", "upper", "width"],
        [[float(t), float(lo), float(hi), float(hi - lo)] for t, lo, hi in zip(times, lower, upper)],
    )
    write_csv(
        RESULTS_DIR / "profile_shift_summary.csv",
        ["lambda_shift", "lower_400h", "upper_400h", "width_400h"],
        [
            [float(x), float(lo), float(hi), float(hi - lo)]
            for x, lo, hi in zip(shifts, shift_lower, shift_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "prior_imprecision_summary.csv",
        ["prior_mean_half_width", "lower_400h", "upper_400h", "width_400h"],
        [
            [float(x), float(lo), float(hi), float(hi - lo)]
            for x, lo, hi in zip(prior_widths, prior_lower, prior_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "source_inclusion_summary.csv",
        ["source_set", "lower_400h", "upper_400h", "width_400h"],
        [
            [label, float(lo), float(hi), float(hi - lo)]
            for label, lo, hi in zip(labels, source_lower, source_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "validation_planning_summary.csv",
        ["budget", "uniform", "profile_proportional", "risk_targeted"],
        [
            [
                int(b),
                float(validation["uniform"][i]),
                float(validation["profile-proportional"][i]),
                float(validation["risk-targeted"][i]),
            ]
            for i, b in enumerate(budget)
        ],
    )
    stability_rows: list[list[object]] = []
    for idx, n_samples in enumerate([10000, 20000, 50000]):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            FULL_SOURCE_SCENARIOS,
            [PI0],
            n_samples=n_samples,
            seed_offset=900 + idx,
        )
        stability_rows.append([n_samples, float(lo), float(hi), float(hi - lo)])
    write_csv(
        RESULTS_DIR / "sampling_stability_summary.csv",
        ["n_samples", "lower_400h", "upper_400h", "width_400h"],
        stability_rows,
    )

    initial_lower = value_at_time(times, lower)
    initial_upper = value_at_time(times, upper)
    print("Generated figures and data in", ROOT)
    print(f"R_lower(400 h)={initial_lower:.4f}")
    print(f"R_upper(400 h)={initial_upper:.4f}")
    print(f"Envelope width at 400 h={initial_upper - initial_lower:.4f}")
    print(f"R_req={R_REQ:.2f}")
    print(f"Minimum importance-sampling ESS={min(ess_values):.1f}")
    for n_samples, lo, hi, width in stability_rows:
        print(f"Stability n={n_samples}: lower={lo:.4f}, upper={hi:.4f}, width={width:.4f}")


if __name__ == "__main__":
    main()
