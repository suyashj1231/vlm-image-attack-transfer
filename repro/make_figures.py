"""Build the E7-E10 paper figures from results/grid5_*.csv.

Story spine (content-vs-control reconciliation):
  F1  dissociation    -- content leaks where control fails, across attacks (clean transfer)
  F2  deployment ladder-- preprocessing destroys control; content partially survives
  F3  quant axis       -- quantization is NOT a defense (3-bit cliff only)
  F4  stress two-regime-- strongest attack: control deployable within-arch, 0% cross-arch
  F5  EOT-patch ladder -- even an EOT patch loses control under preprocessing

Control = starts_with (forces the exact target prefix).  Content = leak50 (>=50%
OCR content leakage).  Okabe-Ito colourblind-safe palette; one axis per chart.

    python3 repro/make_figures.py        # writes figures/*.{png,pdf}
"""
import csv, re, os, glob, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
CTRL, CONT = "#0072B2", "#D55E00"          # Okabe-Ito blue / vermillion (CVD-safe)
LEAK = "#999999"                           # partial-OCR leak50: muted (unreliable)
GRID = "#B0B0B0"
# CONTENT is measured VERBATIM (contains). leak50 (one contiguous word, e.g.
# "hijacked") over-fires on generic glitch-screen descriptions -> shown only as a
# muted secondary line labelled "partial OCR legibility", never as the headline.
PIPE_ORDER = ["none", "jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_q30", "webp_q75",
              "resize_512", "resize_384", "resize_256", "blur_r1", "crop_90"]
PIPE_LBL = {p: p.replace("jpeg_q", "jpg").replace("resize_", "rs").replace("webp_q", "webp")
               .replace("blur_r", "blur").replace("crop_", "crop") for p in PIPE_ORDER}
PIPE_LBL["none"] = "none"
BITS_ORDER = ["3bit", "4bit", "5bit", "6bit", "8bit", "bf16"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.25,
    "axes.axisbelow": True, "legend.frameon": False,
})

def rows(path, split="heldout"):
    if not os.path.exists(path): return []
    return [r for r in csv.DictReader(open(path)) if r["split"] == split]

def rate(rs, metric):
    rs = [r for r in rs if r.get(metric) not in (None, "")]
    return 100 * sum(int(r[metric]) for r in rs) / len(rs) if rs else float("nan")

def scale(v):  # mlx-community/Qwen3-VL-4B-Instruct-8bit -> 4B
    m = re.search(r"-(\d+B)-", v); return m.group(1) if m else v
def bits(v):
    m = re.search(r"Instruct-(\w+)$", v); return m.group(1) if m else "?"

def barlabels(ax, bars):
    for b in bars:
        h = b.get_height()
        if h == h:  # not NaN
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8.5, color="#333")

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig); print(f"  wrote {OUT}/{name}.png / .pdf")

# ---------------------------------------------------------------- F1 dissociation
def f1():
    attacks = [("mitidi", "results/grid5_mitidi_deploy.csv"),
               ("UltraBreak", "results/grid5_ub_deploy.csv"),
               ("stress (ensemble)", "results/grid5_stress_deploy.csv")]
    scales = ["2B", "4B", "8B"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (name, path) in zip(axes, attacks):
        rs = [r for r in rows(path) if r["pipeline"] == "none"]
        byv = defaultdict(list)
        for r in rs: byv[scale(r["victim"])].append(r)
        ctrl = [rate(byv.get(s, []), "starts_with") for s in scales]
        cont = [rate(byv.get(s, []), "contains") for s in scales]
        x = range(len(scales)); w = 0.38
        b1 = ax.bar([i - w/2 for i in x], ctrl, w, label="Control (exact prefix)", color=CTRL)
        b2 = ax.bar([i + w/2 for i in x], cont, w, label="Content (verbatim target)", color=CONT)
        barlabels(ax, b1); barlabels(ax, b2)
        ax.set_title(name); ax.set_xticks(list(x)); ax.set_xticklabels(scales)
        ax.set_ylim(0, 108); ax.set_xlabel("victim scale")
    axes[0].set_ylabel("held-out success (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.suptitle("Content leaks where control fails — clean transfer, held-out prompts",
                 y=1.02, fontsize=12.5, fontweight="bold")
    save(fig, "f1_dissociation")

# ------------------------------------------------------------ F2 deployment ladder
def ladder(path, victim_scale, title, name, note=None):
    rs = [r for r in rows(path) if scale(r["victim"]) == victim_scale]
    byp = defaultdict(list)
    for r in rs: byp[r["pipeline"]].append(r)
    xs = [p for p in PIPE_ORDER if p in byp]
    ctrl = [rate(byp[p], "starts_with") for p in xs]
    cont = [rate(byp[p], "contains") for p in xs]
    leak = [rate(byp[p], "leak50") for p in xs]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(range(len(xs)), leak, "--^", color=LEAK, lw=1.6, ms=5,
            label="Partial OCR legibility (leak50)", zorder=1)
    ax.plot(range(len(xs)), ctrl, "-o", color=CTRL, lw=2, ms=6, label="Control (exact prefix)", zorder=3)
    ax.plot(range(len(xs)), cont, "-s", color=CONT, lw=2, ms=6, label="Content (verbatim target)", zorder=3)
    ax.axvline(0.5, color=GRID, ls=":", lw=1)
    ax.text(0.02, 0.06, "no\npreproc", transform=ax.transAxes, fontsize=8, color="#666")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([PIPE_LBL[p] for p in xs], rotation=45, ha="right")
    ax.set_ylim(-3, 108); ax.set_ylabel("held-out success (%)"); ax.set_xlabel("deployment preprocessing")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    if note: ax.text(0.01, 0.04, note, transform=ax.transAxes, ha="left", fontsize=8, color="#666")
    save(fig, name)

# ---------------------------------------------------------------- F3 quant axis
def f3():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for path, mark, tag in [("results/grid5_stress_quant.csv", "o", "stress"),
                            ("results/grid5_mitidi_quant.csv", "^", "mitidi")]:
        rs = [r for r in rows(path) if scale(r["victim"]) == "2B" and r["pipeline"] == "none"]
        byb = defaultdict(list)
        for r in rs: byb[bits(r["victim"])].append(r)
        xs = [b for b in BITS_ORDER if b in byb]
        ctrl = [rate(byb[b], "starts_with") for b in xs]
        cont = [rate(byb[b], "contains") for b in xs]
        ax.plot(range(len(xs)), ctrl, "-"+mark, color=CTRL, lw=2, ms=6,
                label=f"Control ({tag})", alpha=1 if tag == "stress" else 0.5)
        ax.plot(range(len(xs)), cont, "--"+mark, color=CONT, lw=2, ms=6,
                label=f"Content ({tag})", alpha=1 if tag == "stress" else 0.5)
    ax.set_xticks(range(len(BITS_ORDER))); ax.set_xticklabels(BITS_ORDER)
    ax.set_ylim(-3, 108); ax.set_ylabel("held-out success (%)")
    ax.set_xlabel("victim quantization (2B, no preprocessing)")
    ax.set_title("Quantization is not a defense — only 3-bit (model-breaking) suppresses")
    ax.legend(loc="center right", fontsize=9)
    save(fig, "f3_quant_axis")

# -------------------------------------------------------- F4 stress two-regime
def f4():
    rs = rows("results/grid5_stress_deploy.csv")  # mean over all pipelines
    scales = ["2B", "4B", "8B"]
    byv = defaultdict(list)
    for r in rs: byv[scale(r["victim"])].append(r)
    ctrl = [rate(byv.get(s, []), "starts_with") for s in scales]
    cont = [rate(byv.get(s, []), "contains") for s in scales]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = range(len(scales)); w = 0.38
    b1 = ax.bar([i - w/2 for i in x], ctrl, w, color=CTRL, label="Control (exact prefix)")
    b2 = ax.bar([i + w/2 for i in x], cont, w, color=CONT, label="Content (verbatim target)")
    barlabels(ax, b1); barlabels(ax, b2)
    ax.axvspan(-0.5, 1.5, color="#0072B2", alpha=0.05)
    ax.axvspan(1.5, 2.5, color="#D55E00", alpha=0.05)
    ax.text(0.5, 112, "in surrogate ensemble\n(deployment gap)", ha="center", fontsize=8.5, color="#444")
    ax.text(2.0, 30, "held-out arch\n(true transfer)", ha="center", fontsize=8.5, color="#444")
    ax.set_xticks(list(x)); ax.set_xticklabels(scales); ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 122); ax.set_ylabel("held-out success (%)")
    ax.set_xlabel("victim scale  (mean over 11 deployment pipelines)")
    ax.set_title("Strongest attack: within-arch deployable, ~0% to held-out 8B", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "f4_stress_two_regime")

print("building figures ...")
f1()
ladder("results/grid5_stress_deploy.csv", "2B",
       "Deployment ladder (stress attack, 2B): preprocessing degrades both",
       "f2_deployment_ladder",
       note="leak50 rises as JPEG/resize make embedded glyphs more OCR-legible")
f3()
f4()
ladder("results/grid5_patch_eot_deploy.csv", "2B",
       "EOT-trained patch (2B): control still dies under preprocessing",
       "f5_eot_patch_ladder", note="light EOT (x2) did not confer robust control")
print("done ->", OUT)
