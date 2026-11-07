"""Proposal Figure 1: overview of the three stress-test axes for UltraBreak.

Run from the repo root:  python3 figures/make_f0_study_design.py
Writes figures/f0_study_design.png and figures/f0_study_design.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "images"
OUT = ROOT / "figures"

W_IN, H_IN = 12.0, 5.4
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, FAINT = "#222222", "#666666", "#F4F4F4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
fig = plt.figure(figsize=(W_IN, H_IN))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.axis("off")


def square_axes(x_center, y_bottom, w_frac):
    """Add an axes that is square in inches, centred at x_center (figure fraction)."""
    h_frac = w_frac * W_IN / H_IN
    ax = fig.add_axes([x_center - w_frac / 2, y_bottom, w_frac, h_frac])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")
        s.set_linewidth(0.8)
    return ax, h_frac


def thumb(path, x_center, y_bottom, w_frac, caption):
    ax, h = square_axes(x_center, y_bottom, w_frac)
    ax.imshow(Image.open(path).convert("RGB"))
    bg.text(x_center, y_bottom - 0.02, caption, ha="center", va="top", fontsize=8.5, color=MUTED)
    return h


def pill(x_center, y, text, color):
    bg.text(x_center, y, text, ha="center", va="center", fontsize=8.3, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.2))


def panel(x0, w, color, title, cost):
    y0, y1, head = 0.20, 0.88, 0.085
    bg.add_patch(FancyBboxPatch((x0, y0), w, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=FAINT, ec=color, lw=1.4))
    bg.add_patch(FancyBboxPatch((x0, y1 - head), w, head, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=color, ec=color, lw=1.4))
    bg.text(x0 + w / 2, y1 - head / 2, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    bg.text(x0 + w / 2, y1 - head - 0.035, cost, ha="center", va="center",
            fontsize=8.5, style="italic", color=MUTED)


def compare(x0, w, color, text):
    bg.text(x0 + 0.018, 0.315, "Compare", ha="left", va="center", fontsize=9,
            fontweight="bold", color=color)
    bg.text(x0 + 0.018, 0.29, text, ha="left", va="top", fontsize=8.8, color=INK, linespacing=1.35)


fig.suptitle("Three stress tests of UltraBreak", fontsize=14, fontweight="bold", y=0.975)

gap, left, right = 0.02, 0.02, 0.98
pw = (right - left - 2 * gap) / 3
xs = [left + i * (pw + gap) for i in range(3)]

# Axis 1: image or text?
x = xs[0]
panel(x, pw, BLUE, "Axis 1: Image or text?", "Inference only on an existing attack image")
thumb(IMG / "ultrabreak_v2.png", x + pw * 0.30, 0.50, 0.085, "UltraBreak image")
tax, _ = square_axes(x + pw * 0.70, 0.50, 0.085)
tax.set_facecolor("white")
tax.text(0.5, 0.5, "[Jailbroken\nMode]", ha="center", va="center", fontsize=8,
         family="DejaVu Sans Mono", color="black")
bg.text(x + pw * 0.70, 0.48, "Typographic baseline\n(plain text, no optimization)", ha="center", va="top",
        fontsize=8.5, color=MUTED)
bg.text(x + pw / 2, 0.40, "Each image is run with the jailbreak\nprompt and with a neutral question",
        ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.35)
compare(x, pw, BLUE, "Drop in ASR when the jailbreak text is removed\nUltraBreak vs. the typographic baseline")

# Axis 2: newer models?
x = xs[1]
panel(x, pw, ORANGE, "Axis 2: Newer models?", "Inference only on recent open-weight VLMs")
bg.text(x + pw / 2, 0.70, "Matched pairs that differ in one property", ha="center", va="center",
        fontsize=8.8, color=INK)
bg.text(x + pw / 2, 0.635, "Thinking vs. instruct", ha="center", va="center", fontsize=9,
        fontweight="bold", color=ORANGE)
pill(x + pw * 0.27, 0.58, "Qwen3-VL-8B-Instruct", ORANGE)
pill(x + pw * 0.73, 0.58, "Qwen3-VL-8B-Thinking", ORANGE)
bg.text(x + pw / 2, 0.505, "Preference-aligned vs. not", ha="center", va="center", fontsize=9,
        fontweight="bold", color=ORANGE)
pill(x + pw * 0.27, 0.45, "InternVL3.5-8B-Instruct", ORANGE)
pill(x + pw * 0.73, 0.45, "InternVL3.5-8B-MPO", ORANGE)
bg.text(x + pw / 2, 0.40, "Proposed pairs, to be confirmed", ha="center", va="center",
        fontsize=8.3, style="italic", color=MUTED)
compare(x, pw, ORANGE, "ASR difference within each pair\n(thinking minus instruct, aligned minus unaligned)")

# Axis 3: stable across runs?
x = xs[2]
panel(x, pw, GREEN, "Axis 3: Stable across runs?", "Needs GPU time to re-optimize the image")
for i, frac in enumerate([0.20, 0.50, 0.80]):
    thumb(IMG / f"rigor_ub_s{i}.png", x + pw * frac, 0.51, 0.075, f"seed {i}")
bg.text(x + pw / 2, 0.40, "Re-optimize the image with several random\nseeds (and AdvBench if time allows)",
        ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.35)
compare(x, pw, GREEN, "Mean and standard deviation of ASR\nacross seeds")

# Shared evaluation strip
bg.add_patch(FancyBboxPatch((left, 0.03), right - left, 0.12, boxstyle="round,pad=0,rounding_size=0.01",
                            fc="white", ec="#BBBBBB", lw=1.0))
bg.text(left + 0.015, 0.118, "Every axis", ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
bg.text(left + 0.105, 0.118,
        "ASR on held-out prompts, judged by a keyword check and an LLM judge, compared with a no-attack "
        "baseline, reported with confidence intervals.",
        ha="left", va="center", fontsize=8.8, color=INK)
bg.text(left + 0.015, 0.062, "Benign track", ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
bg.text(left + 0.105, 0.062,
        "The image targets a harmless string. A response that begins with it means the model was steered; "
        "one that only contains it means the model read it.",
        ha="left", va="center", fontsize=8.8, color=INK)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"f0_study_design.{ext}", dpi=220)
print("wrote", OUT / "f0_study_design.png")
