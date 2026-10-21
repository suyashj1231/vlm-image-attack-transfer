"""Aggregate the 3-seed rigor grids into mean +/- std, held-out split only.
Reads results/grid5_rigor_{attack}_s{0,1,2}.csv (ladder columns present)."""
import csv, glob, re, statistics as st
from collections import defaultdict

def short(v): return v.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")

# data[(attack,victim)][metric] = {seed: rate}
data = defaultdict(lambda: defaultdict(dict))
for f in sorted(glob.glob("results/grid5_rigor_*.csv")):
    m = re.search(r"rigor_(\w+?)_s(\d)\.csv", f)
    if not m: continue
    attack, seed = m.group(1), int(m.group(2))
    rows = list(csv.DictReader(open(f)))
    # held-out only
    ho = [r for r in rows if r.get("split") == "heldout"]
    byv = defaultdict(list)
    for r in ho: byv[short(r["victim"])].append(r)
    for v, rs in byv.items():
        n = len(rs)
        for met in ("starts_with", "contains", "leak50"):
            data[(attack, v)][met][seed] = sum(int(r[met]) for r in rs) / n

print("\n3-SEED RIGOR (held-out, pipeline=none)  mean%  (sd)  over seeds 0/1/2")
print(f"{'attack':11} {'victim':8} {'CONTROL(sw)':>14} {'contains':>14} {'leak50':>14}")
print("-"*65)
for (attack, v) in sorted(data):
    cells = []
    for met in ("starts_with", "contains", "leak50"):
        vals = [data[(attack,v)][met][s] for s in sorted(data[(attack,v)][met])]
        mean = sum(vals)/len(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"{mean*100:5.0f} ({sd*100:3.0f})")
    print(f"{attack:11} {v:8} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
print("\n(sw=starts_with=CONTROL; contains=verbatim; leak50=>=50% OCR content. "
      "n seeds per cell shown as sd; held-out prompts only.)")
