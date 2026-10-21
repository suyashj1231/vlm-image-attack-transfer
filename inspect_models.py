"""Experiment 0b: do the 2B/4B/8B victims share a vision encoder?

Pixel-level transfer across model *scale* is only expected to work when the
surrogate and victim share (or closely match) their vision tower. The
embedding-space attack (--loss embed) relies on it directly. This script loads
only the HF *configs* (no weights, so it is fast and needs no GPU) and prints
the vision-encoder shape for each model, so we can see at a glance whether the
2B encoder is the same stack the 4B/8B use.

    source ~/venvs/vlm-torch/bin/activate   # (any env with transformers)
    python inspect_models.py

If the vision sections match (depth, hidden size, patch/merge, head count) the
encoders are architecturally identical and a feature-space attack has a real
chance of crossing scale; if they differ, that difference is itself a clean
explanation for why plain PGD stalls on 4B/8B (RESULTS.md finding 5).
"""
from __future__ import annotations

MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

# config keys that describe the vision tower across Qwen-VL versions
VISION_KEYS = [
    "depth", "num_hidden_layers", "hidden_size", "embed_dim", "out_hidden_size",
    "intermediate_size", "num_heads", "patch_size", "temporal_patch_size",
    "spatial_merge_size", "in_chans", "in_channels", "hidden_act",
]


def _vision_cfg(cfg):
    """Pull the vision sub-config out of whatever attribute this version uses."""
    for attr in ("vision_config", "visual_config", "vision"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return v
    return None


def summarise(model_id: str) -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    vc = _vision_cfg(cfg)
    row = {"model": model_id}
    if vc is None:
        row["vision"] = "<no vision_config found>"
        return row
    vd = vc.to_dict() if hasattr(vc, "to_dict") else dict(vars(vc))
    for k in VISION_KEYS:
        if k in vd and vd[k] is not None:
            row[k] = vd[k]
    # text/LLM side, for context on where the models actually differ
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        td = tc.to_dict() if hasattr(tc, "to_dict") else dict(vars(tc))
        row["llm_layers"] = td.get("num_hidden_layers")
        row["llm_hidden"] = td.get("hidden_size")
    return row


def main():
    rows = []
    for mid in MODELS:
        try:
            rows.append(summarise(mid))
        except Exception as e:  # noqa: BLE001 - diagnostic script, report and go on
            rows.append({"model": mid, "error": repr(e)})

    # union of keys, model first
    keys = ["model"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    print("\nvision-encoder / llm config comparison\n" + "=" * 60)
    for r in rows:
        print(f"\n{r.get('model')}")
        for k in keys[1:]:
            if k in r:
                print(f"  {k:22s} {r[k]}")

    # quick verdict on the vision tower
    vis_sigs = []
    for r in rows:
        sig = tuple((k, r.get(k)) for k in VISION_KEYS if k in r)
        vis_sigs.append(sig)
    if vis_sigs and all(s == vis_sigs[0] and s for s in vis_sigs):
        print("\n=> vision encoders are IDENTICAL across sizes: a feature-space "
              "(--loss embed) attack is well motivated.")
    else:
        print("\n=> vision encoders DIFFER across sizes: pixel/feature transfer "
              "across scale is expected to be hard; this supports finding 5.")


if __name__ == "__main__":
    main()
