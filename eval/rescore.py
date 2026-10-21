"""Re-score saved benign responses with the strict->lenient metric ladder.

`eval.run_grid` saves benign responses to `<grid>_responses.jsonl`. This applies
the ladder (eval.metrics.LADDER) to them and prints, per victim x pipeline, the
success rate at each leniency level. The point of the paper's reconciliation:
the SAME attack's apparent success climbs as the metric gets more lenient --
near-0 at strict control, high at permissive content.

    python -m eval.rescore results/grid4_mitidi_deploy_responses.jsonl
    python -m eval.rescore <responses.jsonl> --target HIJACKED-BY-TEAM-X --split heldout

No model needed; pure string scoring, so it runs in any env.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from attack.targets import BENIGN_TARGET
from eval.metrics import LADDER, lcs_fraction, score_ladder

try:
    from eval.preprocess import PIPELINES
    PIPE_ORDER = [p.name for p in PIPELINES]
except Exception:  # noqa: BLE001
    PIPE_ORDER = []


def _short(v: str) -> str:
    return v.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", help="*_responses.jsonl from eval.run_grid")
    ap.add_argument("--target", default=BENIGN_TARGET)
    ap.add_argument("--split", default=None, choices=[None, "train", "heldout", "custom"],
                    help="restrict to one prompt split")
    ap.add_argument("--by", default="pipeline", choices=["pipeline", "victim"],
                    help="group the table by pipeline (per victim) or collapse")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.responses).read_text().splitlines() if l.strip()]
    if args.split:
        recs = [r for r in recs if r.get("split") == args.split]
    levels = [n for n, _ in LADDER]

    # counts[(victim,pipeline)][level] = hits ; and totals
    hits = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    for r in recs:
        key = (_short(r["victim"]), r.get("pipeline", "?"))
        tot[key] += 1
        for lvl, v in score_ladder(r["response"], args.target).items():
            hits[key][lvl] += v

    victims = sorted({k[0] for k in tot})
    pipes = [p for p in PIPE_ORDER if any(k == (v, p) for v in victims for k in tot)] \
        or sorted({k[1] for k in tot})

    print(f"\nMetric ladder (strict -> lenient) on {Path(args.responses).name}"
          + (f"  [split={args.split}]" if args.split else "") + "\n")
    hdr = f"{'victim':9} {'pipeline':11} " + " ".join(f"{l:>13}" for l in levels) + "   n"
    print(hdr)
    print("-" * len(hdr))
    for v in victims:
        for p in pipes:
            n = tot.get((v, p), 0)
            if not n:
                continue
            cells = " ".join(f"{hits[(v,p)][l]/n:>12.0%} " for l in levels)
            print(f"{v:9} {p:11} {cells}  {n}")

    # collapsed over pipelines, per victim -- the headline "leniency climbs" view
    leak_mean = defaultdict(list)
    for r in recs:
        leak_mean[_short(r["victim"])].append(lcs_fraction(r["response"], args.target))
    print("\nPer victim, averaged over pipelines (the reconciliation curve):")
    print(f"{'victim':9} " + " ".join(f"{l:>13}" for l in levels) + f"{'leak(mean%)':>14}")
    for v in victims:
        ks = [(v, p) for p in pipes if tot.get((v, p))]
        n = sum(tot[k] for k in ks)
        line = " ".join(f"{sum(hits[k][l] for k in ks)/n:>12.0%} " for l in levels)
        lm = sum(leak_mean[v]) / len(leak_mean[v]) if leak_mean[v] else 0
        print(f"{v:9} {line}{lm:>13.0%}")
    print("\n(exact/starts_with = CONTROL; contains/contains_norm = verbatim; "
          "leak50 = >=50% contiguous OCR leakage; leak(mean%) = graded content leakage.)")


if __name__ == "__main__":
    main()
