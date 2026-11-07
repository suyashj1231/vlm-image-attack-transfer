# Ethics & Scope

This project studies the _robustness_ of known vision-language-model image
attacks under realistic deployment conditions. It is defensive security
research: the goal is to tell defenders how much protection quantisation and
input preprocessing actually provide.

## Rules we hold ourselves to

1. **Attacks.** The headline metric forces authors strings as it is approved and for research. It demonstrates the control mechanism without producing any real dangeerous content as this is only for the purpose of research,. This is what we show in class and in the demo as paper is approved by professor and authority.
2. **Aggregate rates only for benchmarks.** When we run standard benchmarks
   (AdvBench / SafeBench) for comparability with prior work, we record and report only a _refusal-bypass rate_. The model's actual responses are can be written to disk, committed, screenshotted, or put on a slide.
3. **Our own models only.** All victims are models we downloaded and run locally
   (MLX / Ollama). We do not attack third-party or commercial API endpoints.
4. **Responsible disclosure.** Any finding that a specific deployment path is
   unexpectedly vulnerable is framed as a defense recommendation, and noted in a
   disclosure section of the report — matching how the three cited papers publish.

## Why this is acceptable coursework

The attacks themselves are already published (Image Hijacks ICLR'24, Schaeffer
et al. ICLR'25, UltraBreak ICLR'26) with public code. Our contribution is a
_measurement study_ of their durability, plus the mitigations that blunt them.
We only add more models to list to see capability aimed at real targets, so as to protect future model from attacks via exposing the issues.

## Benchmark prompt files

We also train against AdvBench/SafeBench, but these prompt files are **not** included here. Download them from their official sources and place them locally; `targets.py` reads only the prompt column.
