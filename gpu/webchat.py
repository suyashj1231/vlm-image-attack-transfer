"""Gradio manual-test chat for the UltraBreak adversarial image on Qwen-VL.

Run inside a Kaggle notebook (Accelerator: GPU T4 x2, Internet: ON) with the
cs205-ub-code dataset attached. In a cell:

    !pip install -q gradio
    import glob, os, sys
    code = os.path.dirname(os.path.dirname(
        glob.glob("/kaggle/input/**/gpu/webchat.py", recursive=True)[0]))
    sys.path.insert(0, code)
    from gpu.webchat import launch
    launch()

It prints a public https://<id>.gradio.live URL. Open it, type a prompt, and
toggle the adversarial image to compare jailbroken (on) vs clean baseline (off).
Uses the exact generate_patched eval path that produced the 100% result, so the
manual test is faithful to the measured attack. Local verification only; do not
log or redistribute generations (see ETHICS.md).
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image


def _find(pattern, fallback=None):
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else fallback


def launch(model_id="Qwen/Qwen3-VL-8B-Instruct",
           adv_name="ub_adv_full_qwen3_8b.png", base_name="gray.png",
           dtype=torch.float16):
    import gradio as gr

    code_hit = _find("/kaggle/input/**/attack/ultrabreak_official.py")
    if code_hit:
        code = os.path.dirname(os.path.dirname(code_hit))
        if code not in sys.path:
            sys.path.insert(0, code)
    from attack.ultrabreak_official import QwenVLOfficial

    adv_path = _find(f"/kaggle/input/**/{adv_name}", f"images/{adv_name}")
    base_path = _find(f"/kaggle/input/**/{base_name}", f"images/{base_name}")
    print(f"adversarial image: {adv_path}\nbase image: {base_path}\nloading {model_id} ...",
          flush=True)

    model = QwenVLOfficial(model_id, device="cuda", dtype=dtype,
                           full_image=True, normalize_patch=True)
    ph, pw = model.patch_shape[1], model.patch_shape[2]
    adv = (torch.from_numpy(np.array(Image.open(adv_path).convert("RGB").resize((pw, ph))))
           .float().permute(2, 0, 1).div(255).to(model.device))
    print("model ready", flush=True)

    def chat(prompt, use_adv, max_tokens):
        if not prompt or not prompt.strip():
            return ""
        patch = adv if use_adv else None   # None = clean-base baseline
        return model.generate_patched(base_path, prompt, patch, max_new_tokens=int(max_tokens))

    with gr.Blocks(title="UltraBreak manual test") as demo:
        gr.Markdown(
            "## UltraBreak manual test — Qwen3-VL-8B\n"
            "Type a prompt and press **Send**. Toggle the adversarial image to compare "
            "**jailbroken** (on) vs **clean baseline** (off) on the *same* model.")
        with gr.Row():
            use_adv = gr.Checkbox(value=True,
                                  label="Adversarial image ON  (uncheck = clean baseline)")
            max_tokens = gr.Slider(16, 256, value=128, step=16, label="max new tokens")
        prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Ask the model something...")
        out = gr.Textbox(label="Model response", lines=12)
        gr.Button("Send", variant="primary").click(chat, [prompt, use_adv, max_tokens], out)

    demo.launch(share=True)


if __name__ == "__main__":
    launch()
