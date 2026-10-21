"""White-box PGD optimiser for a Qwen3-VL *reference* model (PyTorch / MPS).

This is the attacker's side. It optimises a single image so the model emits a
chosen target string for every prompt in a set (universal-over-prompts, in the
spirit of Image Hijacks). The result is a PNG we then ship to the quantised MLX
victims to measure how much of the effect survives (see eval/run_grid.py).

The only subtle part is keeping the path to the pixels differentiable. Qwen VL
does not take a plain [B,3,H,W] tensor; its processor resizes to a 28-multiple
grid, normalises, and flattens the image into a patch sequence `pixel_values`
with an `image_grid_thw`. We reimplement that flattening in torch so autograd
reaches the raw pixels, and we NUMERICALLY VERIFY our reimplementation against
the HF processor's own output (verify_patchify) before trusting any gradient.

Nothing here targets harmful behaviour by itself; the default target is the
benign string in targets.py.

Transferability extensions (all off by default, so existing runs reproduce):
  * ensemble      -- optimise against several surrogate models at once (run_pgd
                     takes a list of QwenVLAttack members). The single biggest
                     lever for cross-model / cross-scale transfer.
  * momentum      -- MI-FGSM gradient accumulation.
  * TI            -- translation-invariant gradient smoothing (Gaussian blur).
  * DI            -- input-diversity random resize+pad each step.
  * embedding     -- feature-space loss on the projected image embeddings,
                     bypassing the LLM (often transfers across scale better).
  * optimizer     -- sign-PGD (default), Adam, or an APGD-style schedule.
  * TV penalty    -- suppress high-frequency "glyph" structure (see Finding 4).
See AttackConfig and run_pgd().
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


# ----------------------------------------------------------------------------
# Qwen image-processing reimplementation (differentiable)
# ----------------------------------------------------------------------------
def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen's smart_resize: round H,W to multiples of `factor`, keep aspect, and
    clamp total pixels into [min_pixels, max_pixels]. Mirrors the HF processor."""
    import math
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio too extreme for smart_resize")
    hbar = max(factor, round(h / factor) * factor)
    wbar = max(factor, round(w / factor) * factor)
    if hbar * wbar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hbar = max(factor, math.floor(h / beta / factor) * factor)
        wbar = max(factor, math.floor(w / beta / factor) * factor)
    elif hbar * wbar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hbar = math.ceil(h * beta / factor) * factor
        wbar = math.ceil(w * beta / factor) * factor
    return hbar, wbar


def torch_patchify(
    img01: torch.Tensor,      # [3, H, W] in [0,1], H & W already grid-aligned
    mean: torch.Tensor,       # [3]
    std: torch.Tensor,        # [3]
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Differentiable version of Qwen2/Qwen3-VL image processing for one image.

    Returns (pixel_values, (grid_t, grid_h, grid_w)) with pixel_values shaped
    [grid_t*grid_h*grid_w, 3*temporal_patch_size*patch_size*patch_size], matching
    the HF processor layout so it can be fed straight to the model.
    """
    c, h, w = img01.shape
    # normalise
    x = (img01 - mean[:, None, None]) / std[:, None, None]
    # temporal: a still image is repeated temporal_patch_size times
    x = x.unsqueeze(0)                                  # [1, 3, H, W]
    x = x.repeat(temporal_patch_size, 1, 1, 1)          # [T, 3, H, W]
    grid_t = 1
    grid_h = h // patch_size
    grid_w = w // patch_size
    x = x.reshape(
        grid_t, temporal_patch_size, c,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pixel_values = x.reshape(
        grid_t * grid_h * grid_w,
        c * temporal_patch_size * patch_size * patch_size,
    )
    return pixel_values, (grid_t, grid_h, grid_w)


def _linearise_patch_embed(model) -> None:
    """The vision patch embed is a Conv3d with kernel == stride, i.e. a plain
    matmul over flattened patches. Conv3d *backward* on MPS takes ~11.6 s here
    (vs 7 ms forward) and dominated every PGD step, so swap in the equivalent
    linear map. Rows of pixel_values are already (C, T, P, P)-flattened, which
    matches proj.weight.flatten(1)."""
    pe = model.model.visual.patch_embed
    proj = pe.proj
    if not isinstance(proj, torch.nn.Conv3d) or proj.kernel_size != proj.stride:
        return
    w = proj.weight.flatten(1)  # [embed_dim, C*T*P*P]
    b = proj.bias

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, w.shape[1]).to(w.dtype)
        return torch.nn.functional.linear(x, w, b)

    pe.forward = forward


# ----------------------------------------------------------------------------
# EOT: expectation over deployment transforms (resize, JPEG)
# ----------------------------------------------------------------------------
def _to_pil(img01: torch.Tensor) -> Image.Image:
    import numpy as np
    arr = (img01.detach().clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255)
    return Image.fromarray(arr.round().astype(np.uint8))


def _from_pil(img: Image.Image, like: torch.Tensor) -> torch.Tensor:
    import numpy as np
    t = torch.from_numpy(np.array(img.convert("RGB"))).float().permute(2, 0, 1) / 255.0
    return t.to(like.device, like.dtype)


@dataclass
class EOTConfig:
    """Random deployment transforms sampled each PGD step. Mirrors the axes in
    eval/preprocess.py (resize then JPEG) but samples a wider range than the
    grid tests, so the grid's exact settings are not simply memorised. Sizes
    are multiples of 32 (patch 16 x merge 2) so no further resize happens."""
    p_resize: float = 0.5
    sizes: tuple[int, ...] = (320, 352, 384, 416, 480, 512)
    p_jpeg: float = 0.7
    jpeg_quality: tuple[int, int] = (40, 95)   # inclusive range
    p_webp: float = 0.0                        # applied only when JPEG is not
    webp_quality: tuple[int, int] = (40, 95)

    @classmethod
    def preset(cls, name: str) -> "EOTConfig":
        if name == "base":
            return cls()
        if name == "wide":
            # an attacker who anticipates harsher/different preprocessing:
            # covers the grid's "wide" tier (JPEG q30, resize 256, WebP q75)
            return cls(sizes=tuple(range(224, 513, 32)), jpeg_quality=(20, 95),
                       p_jpeg=0.5, p_webp=0.4)
        raise ValueError(f"unknown EOT preset {name!r}")


def eot_transform(img01: torch.Tensor, eot: EOTConfig, rng) -> tuple[torch.Tensor, str]:
    """Apply one random resize/JPEG draw. Forward values are the *exact* PIL
    outputs the victims will see; gradients use BPDA (Athalye et al. 2018):
    through a differentiable bicubic resize, and straight-through (identity)
    for JPEG, whose quantisation step has no useful gradient."""
    import io
    desc = []
    if rng.random() < eot.p_resize:
        s = rng.choice(eot.sizes)
        approx = torch.nn.functional.interpolate(
            img01[None], size=(s, s), mode="bicubic", align_corners=False,
            antialias=True)[0]
        exact = _from_pil(_to_pil(img01).resize((s, s), Image.BICUBIC), img01)
        img01 = approx + (exact - approx).detach()
        desc.append(f"r{s}")
    if rng.random() < eot.p_jpeg:
        q = rng.randint(*eot.jpeg_quality)
        buf = io.BytesIO()
        _to_pil(img01).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        exact = _from_pil(Image.open(buf), img01)
        img01 = img01 + (exact - img01).detach()
        desc.append(f"q{q}")
    # guard keeps the RNG stream (and so the base preset's seeds) unchanged
    elif eot.p_webp > 0 and rng.random() < eot.p_webp:
        q = rng.randint(*eot.webp_quality)
        buf = io.BytesIO()
        _to_pil(img01).save(buf, format="WEBP", quality=q)
        buf.seek(0)
        exact = _from_pil(Image.open(buf), img01)
        img01 = img01 + (exact - img01).detach()
        desc.append(f"w{q}")
    return img01, "+".join(desc) or "id"


# ----------------------------------------------------------------------------
# DI-FGSM: input diversity (random resize + pad), differentiable
# ----------------------------------------------------------------------------
def di_transform(img01: torch.Tensor, cfg: "AttackConfig", rng) -> torch.Tensor:
    """Xie et al. 2019 input diversity: with prob di_prob, shrink the image by
    a random factor and random-place it back on a canvas of the original size,
    padding with edge replication. Improves transfer to models with different
    spatial handling (receptive fields, position encodings). Differentiable via
    bicubic interpolate; the canvas side is unchanged so the patch grid is too."""
    if cfg.di_prob <= 0 or rng.random() >= cfg.di_prob:
        return img01
    import torch.nn.functional as F
    c, h, w = img01.shape
    scale = rng.uniform(1.0 - cfg.di_frac, 1.0)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    small = F.interpolate(img01[None], size=(nh, nw), mode="bicubic",
                          align_corners=False, antialias=True)[0]
    top = rng.randint(0, h - nh)
    left = rng.randint(0, w - nw)
    # F.pad order is (left, right, top, bottom); replicate avoids a hard border
    out = F.pad(small[None], (left, w - nw - left, top, h - nh - top),
                mode="replicate")[0]
    return out


def _gaussian_kernel(ksize: int, sigma: float, device, dtype) -> torch.Tensor:
    """Depthwise [3,1,k,k] Gaussian kernel for TI gradient smoothing."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1)
    k2 = k2 / k2.sum()
    return k2[None, None].repeat(3, 1, 1, 1).to(device=device, dtype=dtype)


def _ti_smooth(grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Translation-invariant attack (Dong et al. 2019): convolve the gradient
    with a Gaussian so the step is an average over small input shifts."""
    import torch.nn.functional as F
    k = kernel.shape[-1]
    return F.conv2d(grad[None], kernel, padding=k // 2, groups=3)[0]


def _as_feature_tensor(feats) -> torch.Tensor:
    """Coerce whatever a model's image-feature call returns into a flat
    [n_tokens, hidden] tensor. Qwen3-VL's get_image_features / visual tower
    returns a ModelOutput (BaseModelOutputWithDeepstackFeatures) whose image
    embeddings live in .last_hidden_state; older paths return a tensor, list,
    or tuple. Keep this tolerant so the embed loss survives version drift."""
    if hasattr(feats, "last_hidden_state"):
        feats = feats.last_hidden_state
    elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
        feats = feats.image_embeds
    if isinstance(feats, (list, tuple)):
        feats = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], 0)
    return feats.reshape(-1, feats.shape[-1])


def _tv(delta: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of the perturbation. Penalising it suppresses
    the high-frequency, glyph-like texture that larger victims OCR instead of
    obeying (RESULTS.md finding 4) -- i.e. keep it a control signal, not text."""
    dh = (delta[:, 1:, :] - delta[:, :-1, :]).abs().mean()
    dw = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
    return dh + dw


@dataclass
class AttackConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    canvas: int = 448          # image side (multiple of 28); no smart-resize if aligned
    epsilon: float = 16 / 255  # L-inf budget around the base image; None = unconstrained
    steps: int = 300
    lr: float = 1 / 255
    device: str = "mps"
    dtype: str = "float32"     # fp32 is safest for gradients on MPS
    seed: int = 0
    eot: EOTConfig | None = None   # None = plain PGD on the untransformed image
    eot_samples: int = 1           # transform draws averaged per step

    # -- transferability knobs (all default to the original behaviour) --------
    loss_mode: str = "lm"          # "lm" cross-entropy | "embed" feature-space
    embed_layer: str = "image_features"  # which features the embed loss targets
    optimizer: str = "sign"        # "sign" (PGD) | "adam" | "apgd"
    momentum: float = 0.0          # MI-FGSM decay on the (L1-normalised) grad
    ti_kernel: int = 0             # TI-FGSM Gaussian kernel side (odd); 0 = off
    ti_sigma: float = 1.0
    di_prob: float = 0.0           # DI-FGSM apply probability; 0 = off
    di_frac: float = 0.15          # max relative shrink for DI
    tv_weight: float = 0.0         # total-variation penalty weight; 0 = off
    prompt_batch: bool = False     # average loss over ALL prompts each step
    patch_frac: float = 0.0        # >0 = adversarial PATCH: confine the
    # perturbation to a centred square of this fraction of the canvas side,
    # unbounded within it (ignores `epsilon`). This is the Image-Hijacks
    # "sticker" control: a small region of arbitrary pixels, expected to survive
    # JPEG/resize where a full-image L-inf perturbation does not.


class QwenVLAttack:
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        from transformers import AutoProcessor
        # model class name differs across versions; resolve lazily in _load_model
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = self._load_model(cfg)
        self.device = torch.device(cfg.device)
        ip = self.processor.image_processor
        self.patch_size = ip.patch_size
        self.temporal_patch_size = ip.temporal_patch_size
        self.merge_size = ip.merge_size
        self.min_pixels = getattr(ip, "min_pixels", 4 * 28 * 28)
        self.max_pixels = getattr(ip, "max_pixels", 1280 * 28 * 28)
        mean = torch.tensor(ip.image_mean).float()
        std = torch.tensor(ip.image_std).float()
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.factor = self.patch_size * self.merge_size  # 28 for Qwen

    def _load_model(self, cfg: AttackConfig):
        import transformers
        dtype = getattr(torch, cfg.dtype)
        # Qwen3-VL exposes a dedicated conditional-generation class; fall back to
        # the generic AutoModelForImageTextToText if the name changes.
        for cls_name in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                try:
                    model = cls.from_pretrained(
                        cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                    ).to(cfg.device).eval()
                except Exception:
                    continue
                # we only need d(loss)/d(pixels); without this, backward also
                # builds a full fp32 gradient for every weight (~2x model memory)
                model.requires_grad_(False)
                _linearise_patch_embed(model)
                return model
        raise RuntimeError("Could not load a Qwen3-VL model class from transformers")

    # -- differentiable input assembly ---------------------------------------
    def verify_patchify(self, tol: float = 1e-3) -> float:
        """Sanity check: our torch_patchify must match the HF processor's
        pixel_values on a random image. Returns max abs diff; raises if > tol."""
        import numpy as np
        side = self.cfg.canvas
        arr = (np.random.rand(side, side, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        proc = self.processor.image_processor(images=img, return_tensors="pt")
        ref = proc["pixel_values"].float()
        img01 = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        img01 = img01.to(self.device)
        ours, thw = torch_patchify(
            img01, self.mean, self.std, self.patch_size,
            self.temporal_patch_size, self.merge_size,
        )
        ours = ours.float().cpu()
        if ours.shape != ref.shape:
            raise AssertionError(f"shape mismatch ours={ours.shape} ref={ref.shape}")
        diff = (ours - ref).abs().max().item()
        if diff > tol:
            raise AssertionError(f"patchify mismatch max|diff|={diff:.4g} > {tol}")
        return diff

    def _target_ids(self, target_text: str) -> torch.Tensor:
        return self.processor.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

    def _assemble(self, img01: torch.Tensor, prompt: str, target_text: str):
        """Build input_ids (prompt + image placeholders + target) and a
        differentiable pixel_values. Returns dict for a single teacher-forced
        forward pass, plus the index range of target tokens for the loss."""
        pixel_values, (gt, gh, gw) = torch_patchify(
            img01, self.mean, self.std, self.patch_size,
            self.temporal_patch_size, self.merge_size,
        )
        # weights may be bf16/fp16; delta stays fp32, so cast only at the boundary
        pixel_values = pixel_values.to(self.model.dtype)
        input_ids, mm_token_type_ids, grid_thw, tgt_start, target_ids = \
            self._text_inputs(prompt, target_text, (gt, gh, gw))
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }, tgt_start, target_ids

    def _text_inputs(self, prompt: str, target_text: str, thw: tuple[int, int, int]):
        """Tokenised (prompt + image placeholders + target). Pixel-independent,
        so cached per prompt instead of re-templating every PGD step."""
        cache = self.__dict__.setdefault("_text_cache", {})
        key = (prompt, target_text, thw)
        if key in cache:
            return cache[key]
        gt, gh, gw = thw
        grid_thw = torch.tensor([[gt, gh, gw]], device=self.device)
        n_merge = self.merge_size ** 2
        n_image_tokens = (gt * gh * gw) // n_merge

        # chat template with an image placeholder, then append the target text
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_str = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        tok = self.processor.tokenizer
        # expand the single <|image_pad|> into n_image_tokens copies
        img_pad = tok.convert_ids_to_tokens(
            self.model.config.image_token_id) if hasattr(self.model.config, "image_token_id") else "<|image_pad|>"
        prompt_str = prompt_str.replace(img_pad, img_pad * n_image_tokens)
        prompt_ids = tok(prompt_str, return_tensors="pt").input_ids.to(self.device)
        target_ids = self._target_ids(target_text)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        tgt_start = prompt_ids.shape[1]
        # newer transformers needs per-token modality ids for M-RoPE
        # (0 = text, 1 = image), mirroring what the HF processor returns
        image_token_id = tok.convert_tokens_to_ids(img_pad)
        mm_token_type_ids = (input_ids == image_token_id).long()
        cache[key] = (input_ids, mm_token_type_ids, grid_thw, tgt_start, target_ids)
        return cache[key]

    def loss_for(self, img01: torch.Tensor, prompt: str, target_text: str) -> torch.Tensor:
        shift_logits, target_ids = self.target_logits(img01, prompt, target_text)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            target_ids.reshape(-1),
        )
        return loss

    def target_logits(self, img01: torch.Tensor, prompt: str, target_text: str):
        """Teacher-forced next-token logits at the target positions, plus the
        target ids. shift_logits[:, t] predicts target token t (from position
        t-1). Shape [1, T, vocab]. Shared by the CE loss and the UltraBreak
        semantic loss (attack/ultrabreak.py)."""
        inputs, tgt_start, target_ids = self._assemble(img01, prompt, target_text)
        out = self.model(**inputs)
        logits = out.logits  # [1, seq, vocab]
        shift_logits = logits[:, tgt_start - 1:-1, :].float()
        return shift_logits, target_ids

    # -- embedding / feature-space loss --------------------------------------
    def _image_features(self, img01: torch.Tensor) -> torch.Tensor:
        """Projected image embeddings (what the LLM actually reads for the image
        tokens), [n_image_tokens, hidden]. Prefer the model's own
        get_image_features; fall back to a forward hook on the vision stack."""
        pixel_values, (gt, gh, gw) = torch_patchify(
            img01, self.mean, self.std, self.patch_size,
            self.temporal_patch_size, self.merge_size,
        )
        pixel_values = pixel_values.to(self.model.dtype)
        grid_thw = torch.tensor([[gt, gh, gw]], device=self.device)
        getf = getattr(self.model, "get_image_features", None)
        if getf is not None:
            feats = getf(pixel_values=pixel_values, image_grid_thw=grid_thw)
            return _as_feature_tensor(feats).float()
        # fallback: run the visual tower directly
        visual = self.model.model.visual
        feats = visual(pixel_values, grid_thw=grid_thw)
        return _as_feature_tensor(feats).float()

    def _target_embed(self, target_text: str) -> torch.Tensor:
        """Mean input embedding of the target tokens -- the direction in the
        LLM's embedding space we want the image to look like."""
        tid = self._target_ids(target_text)
        emb = self.model.get_input_embeddings()(tid).float()  # [1, n, hidden]
        return emb[0].mean(0)

    def loss_embed(self, img01: torch.Tensor, target_text: str) -> torch.Tensor:
        """Feature-space objective: make the pooled image embedding point in the
        same direction as the target-token embedding (prompt-independent). This
        bypasses the LLM's reasoning, so it tends to transfer across model
        *scale* better than the full teacher-forced LM loss -- provided the
        surrogate and victim share a vision encoder (check inspect_models.py)."""
        feats = self._image_features(img01)              # [n_img, hidden]
        tgt = self._target_embed(target_text)            # [hidden]
        pooled = feats.mean(0)
        cos = torch.nn.functional.cosine_similarity(pooled, tgt, dim=0)
        return 1.0 - cos                                 # minimise -> align

    def _mem_note(self) -> str:
        if self.device.type != "mps":
            return ""
        return f"  mem={torch.mps.driver_allocated_memory() / 2**30:.1f}G"

    # -- PGD ------------------------------------------------------------------
    def run(self, base_img: Image.Image, prompts: list[str], target_text: str):
        """Single-surrogate entry point (back-compatible). Delegates to the
        ensemble loop with just this model."""
        return run_pgd([self], base_img, prompts, target_text, self.cfg)


# ----------------------------------------------------------------------------
# Ensemble-capable PGD loop (works for a single member too)
# ----------------------------------------------------------------------------
def run_pgd(members: list["QwenVLAttack"], base_img: Image.Image,
            prompts: list[str], target_text: str, cfg: "AttackConfig") -> Image.Image:
    """Optimise one image against one or more surrogate models.

    With len(members) > 1 the per-step loss is averaged over all members (an
    ensemble transfer attack, Liu et al. 2017): the perturbation is pushed to
    work on every surrogate at once, which is what makes it survive to victims
    none of the surrogates exactly match. `cfg` controls momentum / TI / DI /
    optimizer / loss_mode / TV; members all share one `canvas` and device."""
    import random
    import numpy as np

    lead = members[0]
    device = lead.device
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    recent: list[float] = []

    # -- base image: center-crop to square, resize to canvas (as before) ------
    base = base_img.convert("RGB")
    side = min(base.size)
    left, top = (base.width - side) // 2, (base.height - side) // 2
    base = base.crop((left, top, left + side, top + side)).resize(
        (cfg.canvas, cfg.canvas), Image.BICUBIC)
    base01 = torch.from_numpy(np.array(base)).float().permute(2, 0, 1) / 255.0
    base01 = base01.to(device)
    x = base01.clone()
    delta = torch.zeros_like(x, requires_grad=True)

    # -- adversarial patch: confine delta to a centred square region ----------
    pmask = None
    if cfg.patch_frac > 0:
        side_px = max(1, int(round(cfg.patch_frac * cfg.canvas)))
        y0 = (x.shape[1] - side_px) // 2
        x0 = (x.shape[2] - side_px) // 2
        pmask = torch.zeros_like(x)
        pmask[:, y0:y0 + side_px, x0:x0 + side_px] = 1.0
        # a sticker of arbitrary pixels: random-init inside the patch, base
        # image everywhere else (so x + delta = noise in-patch, unchanged out)
        delta.data = (torch.rand_like(x) - x) * pmask
        print(f"patch mode: {side_px}x{side_px}px centred "
              f"({cfg.patch_frac:.0%} of canvas {cfg.canvas}), "
              f"{int(pmask[0].sum().item())} px, epsilon ignored")

    # -- optimiser state ------------------------------------------------------
    g_mom = torch.zeros_like(x)                 # MI-FGSM momentum buffer
    adam_m = torch.zeros_like(x)
    adam_v = torch.zeros_like(x)
    ti_kernel = (_gaussian_kernel(cfg.ti_kernel, cfg.ti_sigma, device, x.dtype)
                 if cfg.ti_kernel else None)
    # APGD-style schedule state
    lr = cfg.lr
    best_loss = float("inf")
    best_delta = delta.detach().clone()
    last_check_loss = float("inf")
    check_every = max(10, cfg.steps // 10)

    step_prompts = prompts if cfg.prompt_batch else None

    for step in range(cfg.steps):
        if delta.grad is not None:
            delta.grad = None
        this_prompts = step_prompts or [prompts[step % len(prompts)]]

        tot, descs = 0.0, []
        n_draws = max(1, cfg.eot_samples)
        # average over members x eot draws x prompts; backward per piece so only
        # one graph is alive at a time (MPS memory). Each piece rebuilds its own
        # transformed image -- reusing one img_t across the prompt loop would
        # backward through the same graph twice (autograd crash) and pin several
        # graphs live at once, defeating the per-piece memory discipline.
        denom = len(members) * n_draws * (len(this_prompts) if cfg.loss_mode == "lm" else 1)

        def _draw():
            img_t = torch.clamp(x + delta, 0, 1)
            img_t = di_transform(img_t, cfg, rng)
            if cfg.eot is not None:
                return eot_transform(img_t, cfg.eot, rng)
            return img_t, "id"

        for member in members:
            for _ in range(n_draws):
                if cfg.loss_mode == "embed":
                    img_t, d = _draw()
                    loss = member.loss_embed(img_t, target_text) / denom
                    loss.backward()
                    tot += loss.item()
                    descs.append(d)
                else:
                    for p in this_prompts:
                        img_t, d = _draw()
                        loss = member.loss_for(img_t, p, target_text) / denom
                        loss.backward()
                        tot += loss.item()
                    descs.append(d)
                if device.type == "mps":
                    torch.mps.empty_cache()

        # total-variation penalty (its own small backward adds to delta.grad)
        if cfg.tv_weight > 0:
            tv = cfg.tv_weight * _tv(delta)
            tv.backward()
            tot += tv.item()

        loss_val = tot
        recent = (recent + [loss_val])[-20:]

        with torch.no_grad():
            grad = delta.grad
            if ti_kernel is not None:
                grad = _ti_smooth(grad, ti_kernel)
            if cfg.momentum > 0:
                grad = grad / grad.abs().mean().clamp_min(1e-12)
                g_mom.mul_(cfg.momentum).add_(grad)
                step_grad = g_mom
            else:
                step_grad = grad

            if cfg.optimizer == "adam":
                adam_m.mul_(0.9).add_(step_grad, alpha=0.1)
                adam_v.mul_(0.999).addcmul_(step_grad, step_grad, value=0.001)
                mhat = adam_m / (1 - 0.9 ** (step + 1))
                vhat = adam_v / (1 - 0.999 ** (step + 1))
                delta -= lr * mhat / (vhat.sqrt() + 1e-8)
            else:
                # sign-PGD and apgd both take a signed step; apgd adapts lr below
                delta -= lr * step_grad.sign()

            if cfg.epsilon is not None and pmask is None:
                delta.clamp_(-cfg.epsilon, cfg.epsilon)
            if pmask is not None:
                delta.mul_(pmask)        # zero any step outside the patch
            delta.data = torch.clamp(x + delta, 0, 1) - x

            # track best iterate; APGD halves the step on a stalled checkpoint
            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.detach().clone()
            if cfg.optimizer == "apgd" and (step + 1) % check_every == 0:
                if last_check_loss - best_loss < 1e-3 * abs(last_check_loss):
                    lr *= 0.5
                    delta.data = best_delta.clone()   # restart from best
                last_check_loss = best_loss

        if step % 20 == 0 or step == cfg.steps - 1:
            tdesc = f"  [{','.join(descs)}]" if (cfg.eot or cfg.di_prob) else ""
            print(f"step {step:4d}  loss={loss_val:.4f}"
                  f"  avg20={sum(recent) / len(recent):.4f}  lr={lr:.5f}"
                  f"{tdesc}{lead._mem_note()}")

    # APGD / any run: ship the best iterate seen, not just the last one
    final_delta = best_delta if cfg.optimizer == "apgd" else delta.detach()
    final01 = torch.clamp(x + final_delta, 0, 1).detach().cpu()
    arr = (final01.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)
