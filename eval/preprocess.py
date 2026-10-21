"""Input-pipeline transforms -- the 'real deployment' preprocessing axis.

A deployed service rarely feeds the attacker's exact PNG to the model. It often
resizes and re-encodes uploads (JPEG). These transforms are applied to the
adversarial image BEFORE it reaches the victim model, so we can measure how much
of the attack survives each everyday step. Each returns a new PIL.Image.

All transforms are pure PIL/numpy -- no model, no torch -- so this module runs in
either virtualenv.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Pipeline:
    name: str
    jpeg_quality: int | None = None   # None = no JPEG re-encode
    resize_to: int | None = None      # None = keep size; else longest side -> this
    webp_quality: int | None = None   # None = no WebP re-encode
    blur_radius: float | None = None  # Gaussian blur, applied first
    crop_frac: float | None = None    # center crop to this fraction of each side, applied first
    # which EOT distribution covers it: "base" (both presets), "wide" (only the
    # wide preset), "never" (neither) -- see attack.optimizer.EOTConfig.preset
    tier: str = "base"


def apply_pipeline(img: Image.Image, pipe: Pipeline) -> Image.Image:
    out = img.convert("RGB")
    if pipe.crop_frac is not None:
        w, h = out.size
        cw, ch = round(w * pipe.crop_frac), round(h * pipe.crop_frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        out = out.crop((left, top, left + cw, top + ch))
    if pipe.blur_radius is not None:
        from PIL import ImageFilter
        out = out.filter(ImageFilter.GaussianBlur(pipe.blur_radius))
    if pipe.resize_to is not None:
        w, h = out.size
        scale = pipe.resize_to / max(w, h)
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        out = out.resize(new, Image.BICUBIC)
    if pipe.jpeg_quality is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=pipe.jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
    if pipe.webp_quality is not None:
        buf = io.BytesIO()
        out.save(buf, format="WEBP", quality=pipe.webp_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
    return out


# The preprocessing axis used in the grid.
PIPELINES: list[Pipeline] = [
    Pipeline("none"),
    Pipeline("resize_384", resize_to=384),   # downscale (448 -> 384, 32-aligned)
    Pipeline("resize_512", resize_to=512),   # upscale
    Pipeline("jpeg_q90", jpeg_quality=90),
    Pipeline("jpeg_q75", jpeg_quality=75),
    Pipeline("jpeg_q50", jpeg_quality=50),
    # outside the base EOT ranges, inside the wide preset
    Pipeline("jpeg_q30", jpeg_quality=30, tier="wide"),
    Pipeline("resize_256", resize_to=256, tier="wide"),
    Pipeline("webp_q75", webp_quality=75, tier="wide"),
    # outside every EOT preset: truly unanticipated
    Pipeline("blur_r1", blur_radius=1.0, tier="never"),
    Pipeline("crop_90", crop_frac=0.9, tier="never"),
]
