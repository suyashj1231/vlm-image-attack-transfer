# Content vs. Control: What VLM Image Attacks Actually Transfer — and Whether It Survives Deployment

**CS 205 research project → ICLR 2027 submission plan.** Working draft.

## 1. The puzzle (two recent papers disagree)

Three papers frame the field of adversarial image attacks on vision-language
models (VLMs):

- **Image Hijacks (ICLR'24, Bailey et al.)** — white-box image attacks reliably
  *control* a VLM's output (force an exact string), but **cross-model transfer is
  ~0%**, and ensembling surrogates does not fix it. Patch attacks survive
  JPEG/noise where L∞ perturbations do not.
- **"Failures to Find Transferable Image Jailbreaks" (ICLR'25, Schaeffer et al.)**
  — a large negative result over 40+ VLMs: image jailbreaks **do not transfer**
  across *any* factor (shared vision backbone, shared LLM, safety training,
  init, dataset). Transfer appears only between near-identical models.
- **UltraBreak (ICLR'26, Cui et al.)** — **claims to overcome this** with a
  single surrogate, via (a) a *semantic embedding-space loss* that smooths the
  landscape and (b) *input-space constraints* (random transforms + TV) that make
  the perturbation into **OCR-able letters/shapes**. Reports strong black-box ASR.
  But it is scored by a **lenient judge-LLM ASR**, its transferring patterns are
  literally *readable text*, and it **fails on large/frontier models** (GPT-5:
  26% vs 24% no-attack; Claude: 20% vs **30%** no-attack).

**The unresolved question:** how can Schaeffer find *no* transfer while UltraBreak
finds transfer solved — one year apart, same model class?

## 2. Our thesis

> **The transfer that recent attacks achieve is typographic *content injection*,
> not adversarial *control* — and it does not survive ordinary deployment
> preprocessing.**

Concretely, we claim:
1. **Content ≠ control.** What transfers across models is the target *appearing*
   in the output (the model OCRs text baked into the image), not the model being
   *hijacked* into emitting it as the controlled response. A strict control
   metric makes UltraBreak's "transfer" largely vanish.
2. **Lenient judges hide this.** Judge-LLM ASR counts content as success;
   a control metric (does the response *begin with* the target) does not.
3. **Deployment kills what remains.** The surviving content-transfer is
   non-robust to the resize / JPEG / WebP / quantization a real service applies —
   especially outside the attacker's assumed transform distribution.

If true, this *reconciles* ICLR'25 and ICLR'26 (both are right, about different
phenomena) and yields a cheap, deployable defense. Both papers list exactly this
as future work (Schaeffer: detection + robust VLMs; UltraBreak: the scale gap).

## 3. Why we are positioned to prove it

- **The metric already exists in our harness.** `starts_with` (control) vs
  `success`/substring (content) in `eval/metrics.py` *is* the control-vs-content
  axis. We formalize it as the paper's core measurement.
- **The deployment axis is our existing contribution.** Quantization (3–8 bit) ×
  resize/JPEG/WebP (`eval/preprocess.py`) × MLX runtime × model scale — the
  stress test no prior VLM-attack paper applies.
- **Fully benign.** We run the entire study with the benign forced string
  `HIJACKED-BY-TEAM-X`; UltraBreak's "affirming phrase" becomes "begin your
  response with: <string>". We measure whether that string *controls* vs merely
  *appears*. No harmful content is produced (see `ETHICS.md`).

## 4. Method

**Attacks compared** (all re-implemented in-repo, single 2B/4B surrogate):
- CE-PGD baseline (`attack/optimizer.py`, `--loss lm`).
- EOT-PGD (preprocessing-aware).
- MI/TI/DI transfer-boosted PGD.
- **UltraBreak, faithful** (`attack/ultrabreak.py`): semantic embedding loss with
  attention weighting (Eqs 3–8), random-transform patch, affine projection, TV,
  Adam/1300 steps/224², TPG. Matches paper Table 4.
- Adversarial patch (Image-Hijacks style, robustness control).

**Dual metric, always reported together:**
- *Control* = `starts_with` (response begins with the target).
- *Content* = substring presence (+ an OCR read of the image, to confirm the
  mechanism is literally readable text).

**Axes:** {attack} × {victim 2B/4B/8B} × {quantization} × {preprocessing pipeline}
× {train/held-out prompts}. 8B is deferred; 2B/4B is the current focus.

**Headline experiment:** show UltraBreak clean → high content, **low control**,
and that **both collapse** under preprocessing outside its transform distribution,
while quantization alone does not defend. Contrast with white-box (both high),
isolating transfer as the failure point.

## 5. Expected contributions

1. A **reconciliation** of the ICLR'25 negative and ICLR'26 positive results via
   the content/control distinction — a conceptual contribution.
2. A **measurement study** of VLM attack transfer under realistic deployment
   (quantization + preprocessing + runtime + scale), the first of its kind.
3. A **metric**: control-vs-content reporting that we argue should be standard
   for VLM-attack evaluation (judge-ASR over-credits content).
4. A **cheap defense**: unpredictable preprocessing + quantization, evaluated
   against UltraBreak.

## 6. Status / plan

- [x] Harness: differentiable Qwen surrogate, EOT, deployment grid, dual metric.
- [x] Replicated the no-transfer wall (2B→4B/8B control = 0%) and the
  content-vs-control phenomenon (MI/TI/DI: 100% content, 0% control on 4B).
- [x] White-box ceiling (4B hijacks white-box 75%) → wall is transfer, not capacity.
- [x] **Faithful UltraBreak baseline** (`attack/ultrabreak.py`), white-box verified.
- [ ] UltraBreak 2B/4B surrogate → transfer grid under BOTH metrics.
- [ ] Deployment sweep on UltraBreak image (quant × preprocessing).
- [ ] Patch-attack robustness comparison.
- [ ] Defense evaluation; writeup.

## 7. Faithfulness notes (UltraBreak reimplementation)

Implemented per paper Sec 3 / Table 4. Two deliberate deviations, both benign and
documented: (i) target is the project's benign string, not a harmful phrase;
(ii) the affine projection uses the surrogate's own normalization stats rather
than CLIP's, since our surrogate/victims are Qwen-VL. All else — semantic
attention loss, random rotate/scale/translate on blank canvas, TV, Adam,
1300 steps, 224², random init, TPG — matches.
