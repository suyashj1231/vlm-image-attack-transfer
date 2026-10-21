import csv, sys
from collections import defaultdict

def short(v): return v.split("/")[-1].replace("Qwen3-VL-","").replace("-Instruct","")

def load(f):
    return list(csv.DictReader(open(f)))

def rate(rows, metric):
    rows=[r for r in rows if r.get(metric) not in (None,"")]
    return 100*sum(int(r[metric]) for r in rows)/len(rows) if rows else float("nan")

def section(title, f, split="heldout"):
    print(f"\n===== {title} =====  ({f.split('/')[-1]})")
    rows=[r for r in load(f) if r["split"]==split]
    if not rows: print("  (no", split, "rows)"); return
    # per victim x pipeline table for control/leak50
    byv=defaultdict(list)
    for r in rows: byv[short(r["victim"])].append(r)
    print(f"  held-out, per victim: mean over {len(set(r['pipeline'] for r in rows))} pipelines")
    print(f"  {'victim':8} {'CONTROL(sw)':>12} {'contains':>10} {'leak50':>8}   | clean(none): ctrl/leak")
    for v in sorted(byv):
        rs=byv[v]
        none=[r for r in rs if r["pipeline"]=="none"]
        print(f"  {v:8} {rate(rs,'starts_with'):11.0f}% {rate(rs,'contains'):9.0f}% {rate(rs,'leak50'):7.0f}%"
              f"   | {rate(none,'starts_with'):.0f}% / {rate(none,'leak50'):.0f}%")

def quant(title,f,split="heldout"):
    print(f"\n===== {title} =====  ({f.split('/')[-1]})")
    rows=[r for r in load(f) if r["split"]==split and r["pipeline"]=="none"]
    byv=defaultdict(list)
    for r in rows: byv[short(r["victim"])].append(r)
    print(f"  {'victim(bits)':16} {'CONTROL(sw)':>12} {'contains':>10} {'leak50':>8}")
    for v in sorted(byv):
        rs=byv[v]
        print(f"  {v:16} {rate(rs,'starts_with'):11.0f}% {rate(rs,'contains'):9.0f}% {rate(rs,'leak50'):7.0f}%")

section("STRESS deploy (strongest transfer attack)", "results/grid5_stress_deploy.csv")
quant("STRESS quant axis", "results/grid5_stress_quant.csv")
section("EOT-PATCH deploy (valid robustness test)", "results/grid5_patch_eot_deploy.csv")

# pipeline breakdown for EOT patch: does it survive compression?
print("\n===== EOT-PATCH per-pipeline (held-out, 2B victim if present else all) =====")
rows=[r for r in load("results/grid5_patch_eot_deploy.csv") if r["split"]=="heldout"]
bypipe=defaultdict(list)
for r in rows: bypipe[r["pipeline"]].append(r)
print(f"  {'pipeline':12} {'CONTROL(sw)':>12} {'leak50':>8}")
for p in sorted(bypipe):
    rs=bypipe[p]
    print(f"  {p:12} {rate(rs,'starts_with'):11.0f}% {rate(rs,'leak50'):7.0f}%")
