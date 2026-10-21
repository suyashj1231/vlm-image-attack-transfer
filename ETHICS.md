# Ethics & Scope

This project studies the *robustness* of known vision-language-model image
attacks under realistic deployment conditions. It is defensive security
research: the goal is to tell defenders how much protection quantisation and
input preprocessing actually provide.

## Rules we hold ourselves to
1. **Benign-first.** The headline metric forces a harmless string
   (`HIJACKED-BY-TEAM-X`). It demonstrates the control mechanism without
   producing any harmful content. This is what we show in class and in the demo.
2. **Aggregate rates only for benchmarks.** When we run standard benchmarks
   (AdvBench / SafeBench) for comparability with prior work, we record and report
   only a *refusal-bypass rate*. The model's actual responses are never written
   to disk, committed, screenshotted, or put on a slide.
3. **Our own models only.** All victims are models we downloaded and run locally
   (MLX / Ollama). We do not attack third-party or commercial API endpoints.
4. **No harmful artefacts in the repo.** No harmful target strings, no harmful
   generations, no weaponised images beyond the benign demonstrator.
5. **Responsible disclosure.** Any finding that a specific deployment path is
   unexpectedly vulnerable is framed as a defense recommendation, and noted in a
   disclosure section of the report — matching how the three cited papers publish.

## Why this is acceptable coursework
The attacks themselves are already published (Image Hijacks ICLR'24, Schaeffer
et al. ICLR'25, UltraBreak ICLR'26) with public code. Our contribution is a
*measurement study* of their durability, plus the mitigations that blunt them.
We add no new offensive capability aimed at real targets.

## Benchmark prompt files
AdvBench/SafeBench prompt files are **not** included here. A team member must
download them from their official sources and place them locally; `targets.py`
reads only the prompt column.
