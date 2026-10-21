"""Summarise grid CSVs into per-cell success rates, averaged over seeds.

    python -m eval.summarize v1=results/grid2_attack.csv \
        eot=results/grid2_attack_eot.csv,results/grid2_attack_eot_s1.csv,results/grid2_attack_eot_s2.csv

Each group is method=comma-separated CSVs (one per seed/image). Prints markdown
tables: success (substring) and starts_with, as mean over seeds with the
per-seed min-max range, for every victim x pipeline, plus totals per EOT tier.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def short(victim: str) -> str:
    return victim.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")


def tier(r: dict) -> str:
    # grid2_* CSVs predate the tier column and only had seen_by_eot
    if "tier" in r:
        return r["tier"]
    return "base" if r.get("seen_by_eot") == "1" else "wide"


def rates(rows: list[dict], key, metric: str) -> dict:
    hit, n = defaultdict(int), defaultdict(int)
    for r in rows:
        k = key(r)
        hit[k] += int(r[metric])
        n[k] += 1
    return {k: hit[k] / n[k] for k in n}


def fmt(vals: list[float]) -> str:
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.0%}"
    return f"{m:.0%} ({min(vals):.0%}-{max(vals):.0%})"


def main():
    groups = {}
    for arg in sys.argv[1:]:
        name, paths = arg.split("=", 1)
        groups[name] = [load(p) for p in paths.split(",")]
    first = next(iter(groups.values()))[0]
    victims = list(dict.fromkeys(r["victim"] for r in first))
    pipes = list(dict.fromkeys(r["pipeline"] for r in first))
    tiers = list(dict.fromkeys(tier(r) for r in first))
    mark = {"base": "", "wide": " *", "never": " **"}

    for metric in ("success", "starts_with"):
        print(f"\n### {metric} (mean over seeds, min-max in brackets)\n")
        print("| victim | method | " + " | ".join(
            p + mark[next(tier(r) for r in first if r["pipeline"] == p)]
            for p in pipes) + " |")
        print("|---|---|" + "---|" * len(pipes))
        for v in victims:
            for name, runs in groups.items():
                per = [rates(rs, lambda r: (r["victim"], r["pipeline"]), metric) for rs in runs]
                cells = [fmt([p[(v, pipe)] for p in per]) for pipe in pipes]
                print(f"| {short(v)} | {name} | " + " | ".join(cells) + " |")

        print(f"\n{metric} totals by victim and EOT tier "
              f"(held-out prompts only in brackets):\n")
        for v in victims:
            for name, runs in groups.items():
                def tot(pred):
                    return fmt([
                        sum(int(r[metric]) for r in rs if r["victim"] == v and pred(r))
                        / max(1, sum(1 for r in rs if r["victim"] == v and pred(r)))
                        for rs in runs])
                parts = [f"{t} {tot(lambda r, t=t: tier(r) == t)} "
                         f"[{tot(lambda r, t=t: tier(r) == t and r['split'] == 'heldout')}]"
                         for t in tiers]
                print(f"- {short(v)} {name}: " + " / ".join(parts))
    print("\n`*` = only in the wide EOT preset; `**` = in no EOT preset")


if __name__ == "__main__":
    main()
