"""Transfer leaderboard: the one number the 4B/8B claim rests on.

eval.summarize prints the full victim x pipeline grid. This is the compressed
view for the stage-4 sweep: for each attack method (one grid CSV, or several
comma-separated seed CSVs) it pulls the HELD-OUT `starts_with` rate on the big
victims, so "did method X finally move 8B?" is one sorted table instead of nine
CSVs eyeballed by hand.

    python -m eval.summarize_transfer \
        eot=results/grid3_attack_eot.csv \
        ens=results/grid4_ens2b4b.csv \
        mitidi=results/grid4_mitidi.csv \
        embed=results/grid4_embed.csv \
        apgd=results/grid4_apgd.csv

Bare paths work too (the method name is taken from the filename):
    python -m eval.summarize_transfer results/grid4_*.csv

Three columns per victim:
  none        -- starts_with on the clean image (the headline hijack number)
  base        -- mean starts_with over anticipated (base-tier) pipelines
  none(succ)  -- substring success on the clean image, for context (inflated on
                 4B/8B when the model OCRs the text instead of obeying it --
                 RESULTS.md finding 4)
Rows are sorted by the first victim's `none` starts_with, best first.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tier(r: dict) -> str:
    if "tier" in r and r["tier"]:
        return r["tier"]
    return "base" if r.get("seen_by_eot") == "1" else "wide"


def _rate(rows: list[dict], metric: str, pred) -> float | None:
    sel = [r for r in rows if pred(r)]
    if not sel:
        return None
    return sum(int(r[metric]) for r in sel) / len(sel)


def _fmt(vals: list[float | None]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  -  "
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.0%}"
    return f"{m:.0%} ({min(vals):.0%}-{max(vals):.0%})"


def _cell_vals(seed_rows: list[list[dict]], metric: str, pred) -> list[float | None]:
    return [_rate(rows, metric, pred) for rows in seed_rows]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    # parse: name=path[,path...] OR bare path (name = filename stem)
    groups: dict[str, list[list[dict]]] = {}
    order: list[str] = []
    for arg in args:
        if "=" in arg and not arg.lower().endswith(".csv"):
            name, paths = arg.split("=", 1)
        elif "=" in arg:
            name, paths = arg.split("=", 1)
        else:
            paths = arg
            name = Path(arg).stem.replace("grid4_", "").replace("grid3_", "")
        groups[name] = [load(p) for p in paths.split(",")]
        order.append(name)

    # victims present (keep file order), then pick the big ones to show
    first = next(iter(groups.values()))[0]
    all_victims = list(dict.fromkeys(r["victim"] for r in first))
    want = [v for v in all_victims if any(s in v for s in ("4B", "8B"))] or all_victims

    def short(v: str) -> str:
        return v.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")

    heldout = lambda r: r.get("split") == "heldout"  # noqa: E731

    def preds(victim: str):
        vmatch = lambda r: r["victim"] == victim and heldout(r)  # noqa: E731
        return {
            "none": lambda r: vmatch(r) and r["pipeline"] == "none",
            "base": lambda r: vmatch(r) and tier(r) == "base",
        }

    # build rows
    table = []
    for name in order:
        seed_rows = groups[name]
        row = {"method": name}
        for v in want:
            p = preds(v)
            row[f"{short(v)}|none"] = _cell_vals(seed_rows, "starts_with", p["none"])
            row[f"{short(v)}|base"] = _cell_vals(seed_rows, "starts_with", p["base"])
            row[f"{short(v)}|succ"] = _cell_vals(seed_rows, "success", p["none"])
        table.append(row)

    # sort by first victim's clean starts_with (best first)
    key_col = f"{short(want[0])}|none"

    def sort_key(row):
        vals = [v for v in row.get(key_col, []) if v is not None]
        return -(sum(vals) / len(vals)) if vals else 1.0
    table.sort(key=sort_key)

    # render markdown
    cols = ["method"]
    for v in want:
        s = short(v)
        cols += [f"{s} none", f"{s} base", f"{s} none(succ)"]
    print("\nHeld-out `starts_with` on the big victims (hijack rate), best first.")
    print("`base` = mean over anticipated (base-tier) pipelines. "
          "`none(succ)` = substring success on clean image (OCR-inflated).\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in table:
        cells = [row["method"]]
        for v in want:
            s = short(v)
            cells.append(_fmt(row[f"{s}|none"]))
            cells.append(_fmt(row[f"{s}|base"]))
            cells.append(_fmt(row[f"{s}|succ"]))
        print("| " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
