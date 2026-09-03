from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"
CUSTOM = COMFY / "custom_nodes"
MODELS = COMFY / "models"

_configured_input_root = os.getenv("H3_INPUT_ROOT", "").strip()
if _configured_input_root:
    KAGGLE_INPUT = Path(_configured_input_root).expanduser().resolve()
elif Path("/kaggle/input").is_dir():
    KAGGLE_INPUT = Path("/kaggle/input").resolve()
else:
    KAGGLE_INPUT = (ROOT / "input").resolve()

MODEL_MANIFEST = (
    ROOT
    / "configs"
    / "model_inventory.yaml"
)

NODE_MANIFEST = (
    ROOT
    / "configs"
    / "custom_nodes.yaml"
)

RUNTIME_MANIFEST = (
    ROOT
    / "configs"
    / "runtime_versions.yaml"
)



def apply_embedded_h3_runtime_overlay() -> None:
    """Apply the project-owned H3 runtime overrides from one file-free overlay.

    ComfyUI is never stored in the Git repository. It is cloned by bootstrap and
    these four runtime-owned files are written directly into that temporary
    ComfyUI checkout.
    """
    import hashlib

    patches = {
        'comfy/ldm/minimax/model.py': '"""MiniMax H3 audio-video DiT.\n\nSingle-stream packed-token transformer denoising video (24ch, patch 1x2x2) and\nstereo audio (32ch, 40 Hz) latents jointly, conditioned on Qwen3-VL layer-50 hidden states.\nThe packed sequence is:\n[text | cond rows | audio | video] for t2va/fl2va\n[text | reference blocks | audio | video] for ref2va\n\nTimestep domain: the model receives the *video* sigma from the sampler and\nderives per-token timesteps t = 1 - sigma internally; the audio stream runs on\nits own shifted schedule (sigma_shift video 12.0 / audio 3.0), mapped from the\nvideo sigma in closed form. The sampler carries the audio latent scaled onto the\nvideo schedule (ModelSamplingAV); forward() undoes that scale and converts the\nvelocity back, so _forward only ever sees the stream\'s own latent.\n"""\n\nimport math\n\nimport torch\nimport torch.nn as nn\n\nimport comfy.ldm.common_dit\nimport comfy.model_management\nimport comfy.model_prefetch\nimport comfy.ops\nimport comfy.patcher_extension\nimport comfy.quant_ops\nfrom comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention\n\nFRAME_PER_TOKEN = (1, 4, 4, 4, 4)\nFRAME_RESCALE = 5.0 / 3.0\nVISUAL_COND_TIMESTEP = 0.999\nAUDIO_COND_TIMESTEP = 1.0\n\n\ndef time_shift_sigma(sigma, from_shift, to_shift):\n    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift\n    base = sigma / (from_shift + sigma * (1.0 - from_shift))\n    return to_shift * base / (1.0 + (to_shift - 1.0) * base)\n\n\ndef patchify_video(latent, patch_size=(1, 2, 2)):\n    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]\n    b, c, t_full, h_full, w_full = latent.shape\n    pt, ph, pw = patch_size\n    t, h, w = t_full // pt, h_full // ph, w_full // pw\n    x = latent.reshape(b, c, t, pt, h, ph, w, pw)\n    x = torch.einsum("nctrhpwq->nthwcrpq", x)\n    return x.reshape(b * t * h * w, c * pt * ph * pw)\n\n\ndef unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):\n    pt, ph, pw = patch_size\n    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)\n    x = torch.einsum("nthwcrpq->nctrhpwq", x)\n    return x.reshape(-1, c, t * pt, h * ph, w * pw)\n\n\ndef pack_audio(latent):\n    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)\n    b, c, ch, t = latent.shape\n    return latent[0].permute(1, 2, 0).reshape(ch * t, c)\n\n\ndef unpack_audio(rows, ch=2):\n    t = rows.shape[0] // ch\n    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)\n\n\ndef _axis_from_sqrt_area(dim, patch, sqrt_area):\n    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32\n    ratio = dim / sqrt_area\n    n = dim // patch\n    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0\n\n\ndef mask_row_values(mask, latent_t, lat_h, lat_w):\n    # [T, H, W] denoise mask (1 = generate) -> per-2x2-patch-row float in [0, 1],\n    # None when every row fully generates\n    m = torch.nn.functional.pad(mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]), mode="replicate")\n    m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))\n    values = m.reshape(-1)\n    if bool((values >= 1.0 - 1e-3).all()):\n        return None\n    return values\n\n\ndef _frame_grid(h, w):\n    # area-normalized (h, w) coordinates of one latent frame\'s 2x2-patch rows\n    area = math.sqrt(h * w)\n    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")\n    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)\n\n\ndef _video_t_spans(n):\n    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]\n\n\ndef _video_t_grid(n, origin):\n    # origin + exclusive cumsum\n    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)\n    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])\n\n\ndef _ref_t_span(blk):\n    # time-axis span a reference block occupies ahead of the target streams\n    kind = blk["kind"]\n    if kind == "image":\n        return 1.0\n    if kind == "audio":\n        return float(blk["ref_audio_t"])\n    if kind in ("video", "video_audio"):\n        return max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))\n    return 0.0\n\n\ndef _audio_grid(cursor, t, w_low, w_high):\n    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0\n    g = torch.zeros(t * 2, 3, dtype=torch.float64)\n    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)\n    g[:t, 2] = w_low\n    g[t:, 2] = w_high\n    return g\n\n\ndef _video_grid(vt, frame, cursor):\n    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)\n    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]\n    g[:, :, 1:] = frame[None]\n    return g.reshape(-1, 3)\n\n\nclass TimeEmbedder(nn.Module):\n    def __init__(self, freq_dim, hidden, out, dtype=None, device=None, operations=None):\n        super().__init__()\n        self.freq_dim = freq_dim\n        self.proj_in = operations.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)\n        self.proj_out = operations.Linear(hidden, out, bias=True, dtype=dtype, device=device)\n\n    def forward(self, t):\n        # t: [M] in [0, 1]; fp32 throughout, cos before sin\n        half = self.freq_dim // 2\n        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)\n        args = t.to(torch.float32)[:, None] * freqs[None]\n        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)\n        return self.proj_out(nn.functional.silu(self.proj_in(emb)))\n\n\ndef rope_rotation_table(angles, dtype):\n    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices."""\n    half = angles.shape[-1] // 2\n    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]\n    c, s = torch.cos(ang), torch.sin(ang)\n    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)\n    return table.to(dtype)\n\n\nclass Attention(nn.Module):\n    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None):\n        super().__init__()\n        self.heads = heads\n        self.head_dim = head_dim\n        inner = heads * head_dim\n        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)\n        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)\n        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)\n        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)\n\n    def forward(self, x, rope_freqs=None, transformer_options={}):\n        s = x.shape[0]\n        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)\n        v = v.view(s, self.heads, self.head_dim)\n        if rope_freqs is not None:\n            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer\n            q = q.view(1, s, self.heads, self.head_dim)\n            k = k.view(1, s, self.heads, self.head_dim)\n            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)\n            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)\n            rot = rope_freqs.shape[-3] * 2\n            if comfy.model_management.in_training:\n                q, k = comfy.quant_ops.ck.rms_rope_split_half(\n                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)\n            else:\n                comfy.quant_ops.ck.rms_rope_split_half_(\n                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)\n            q = q[0]\n            k = k[0]\n        else:\n            q = self.q_norm(q.view(s, self.heads, self.head_dim))\n            k = self.k_norm(k.view(s, self.heads, self.head_dim))\n        # H3-T4-WORKAROUND: removed redundant V clone for SM75\n        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))\n        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))\n        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))\n        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)\n        out = out.squeeze(0)\n\n        # H3 T4 FP16: protect the attention output projection from FP16 overflow.\n        # The surrounding residual stream remains FP32; only this matrix multiply\n        # is range-scaled into FP16 and restored to FP32 afterwards.\n        proj_weight = getattr(self.out_proj, "weight", None)\n        proj_dtype = getattr(proj_weight, "dtype", None)\n        if x.dtype == torch.float32 and proj_dtype == torch.float16:\n            return (\n                self.out_proj((out / 64.0).to(torch.float16))\n                .to(torch.float32)\n                .mul_(64.0)\n            )\n\n        return self.out_proj(out)\n\n\nclass MLP(nn.Module):\n    def __init__(self, hidden, ffn, dtype=None, device=None, operations=None):\n        super().__init__()\n        self.fc1 = operations.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)\n        self.fc2 = operations.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)\n\n    def forward(self, x):\n        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")\n\n\nclass AdalnProj(nn.Module):\n    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True,\n                 dtype=None, device=None, operations=None):\n        super().__init__()\n        self.expand = expand\n        self.modalities = modalities\n        self.hidden = hidden\n        self.apply_silu = apply_silu\n        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)\n\n    def forward(self, t_emb):\n        # [M, t_dim] -> expand tensors of [M*modalities, hidden]\n        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)\n        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)\n        return x.chunk(self.expand, dim=-1)\n\n\ndef _mod_row(vecs, row, dtype):\n    # row is a mod-row index, or a per-token LongTensor of mod-row indices\n    return vecs[row].to(dtype)\n\n\ndef _mod_scale_shift(h, shift, scale, segments):\n    # segments: [(start, stop, mod_row)] covering h contiguously.\n    for a, b, row in segments:\n        h[a:b].mul_(1.0 + _mod_row(scale, row, h.dtype)).add_(_mod_row(shift, row, h.dtype))\n    return h\n\n\ndef _mod_gate(x, gate, other, segments):\n    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place, one fused kernel per segment\n    for a, b, row in segments:\n        x[a:b].addcmul_(other[a:b], _mod_row(gate, row, x.dtype))\n    return x\n\n\nclass RefinerBlock(nn.Module):\n    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None, operations=None):\n        super().__init__()\n        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)\n        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)\n        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)\n        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)\n\n    def forward(self, x, transformer_options={}):\n        # attn/mlp outputs are fresh: accumulate residuals in place\n        x = self.attn(self.norm1(x), transformer_options=transformer_options).add_(x)\n        return self.mlp(self.norm2(x)).add_(x)\n\n\nclass TokenRefiner(nn.Module):\n    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps,\n                 dtype=None, device=None, operations=None):\n        super().__init__()\n        self.blocks = nn.ModuleList([\n            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device, operations=operations)\n            for _ in range(num_layers)])\n        self.final_norm = operations.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)\n\n    def forward(self, x, transformer_options={}):\n        for block in self.blocks:\n            x = block(x, transformer_options=transformer_options)\n        return self.final_norm(x)\n\n\nclass DiTBlock(nn.Module):\n    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,\n                 apply_silu=True, adaln_dtype=None, dtype=None, device=None, operations=None):\n        super().__init__()\n        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)\n        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)\n        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)\n        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)\n        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,\n                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,\n                                    device=device, operations=operations)\n\n    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):\n        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)\n        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)\n        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)\n        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)\n        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)\n\n\nclass FinalLayer(nn.Module):\n    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None,\n                 dtype=None, device=None, operations=None):\n        super().__init__()\n        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)\n        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,\n                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,\n                                    device=device, operations=operations)\n        # output heads are the checkpoint\'s fp32 island; norm/adaln are stored at model dtype\n        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)\n        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)\n\n    def forward(self, x, t_emb, video_seg, audio_seg):\n        # video_seg / audio_seg: (start, stop, row) of the target streams, where row\n        # is a mod-row index or a per-token blend (see _mod_row)\n        shift, scale = self.adaln_proj(t_emb)\n\n        def mod(seg):\n            a, b, row = seg\n            return (self.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype)) + _mod_row(shift, row, shift.dtype)).to(torch.float32)\n\n        return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))\n\n\nclass PackedLayout:\n    """Static packed-sequence structure for one shape/conditioning signature."""\n\n    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):\n        frame, w_grid = _frame_grid(latent_h, latent_w)\n        frame_rows = frame.shape[0]\n\n        segments = [("text", text_len)]  # (kind, n_rows)\n        g = torch.zeros(text_len, 3, dtype=torch.float64)\n        g[:, 0] = torch.arange(text_len, dtype=torch.float64)\n        pos = [g]  # per segment: [n, 3] float64 (t, h, w)\n\n        img_pos, img_update = [], []\n        audio_pos, audio_update = [], []\n        row = text_len\n\n        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))\n        # refs pack between text and the targets, so the target timeline starts after their spans\n        cursor = float(text_len)\n        for blk in refs or ():\n            cursor += _ref_t_span(blk)\n\n        if keyframes:\n            # fl2va: keyframe cond rows right after text, sharing the target spatial grid;\n            # anchors count from the target timeline origin, FRAME_RESCALE per pixel frame, 1.0 per audio latent frame\n            for kf in keyframes:\n                cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]\n                video_latent = kf.get("latent")\n                if video_latent is not None:\n                    vt = video_latent.shape[2]\n                    n = vt * frame_rows\n                    segments.append(("cond", n))\n                    pos.append(_video_grid(vt, frame, cond_t))\n                    img_pos.append(torch.arange(row, row + n))\n                    img_update.append(torch.zeros(n, dtype=torch.bool))\n                    row += n\n                audio_latent = kf.get("audio_latent")\n                if audio_latent is not None:\n                    rt = audio_latent.shape[-1]\n                    segments.append(("cond_audio", rt * 2))\n                    pos.append(_audio_grid(cond_t, rt, *target_audio_w))\n                    audio_pos.append(torch.arange(row, row + rt * 2))\n                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))\n                    row += rt * 2\n\n        if refs:\n            cursor = float(text_len)\n            for blk in refs:\n                kind = blk["kind"]\n                if kind == "image":\n                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])\n                    n = r_frame.shape[0]\n                    g = torch.empty(n, 3, dtype=torch.float64)\n                    g[:, 0] = cursor\n                    g[:, 1:] = r_frame\n                    segments.append(("ref_img", n))\n                    pos.append(g)\n                    img_pos.append(torch.arange(row, row + n))\n                    img_update.append(torch.zeros(n, dtype=torch.bool))\n                    row += n\n                    cursor += 1.0\n                elif kind == "audio":\n                    rt = blk["ref_audio_t"]\n                    if rt > 0:\n                        segments.append(("ref_audio", rt * 2))\n                        pos.append(_audio_grid(cursor, rt, *target_audio_w))\n                        audio_pos.append(torch.arange(row, row + rt * 2))\n                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))\n                        row += rt * 2\n                    cursor += float(rt)\n                elif kind in ("video", "video_audio"):\n                    # the block\'s audio rows pack immediately before its video\n                    # rows, both sharing the cursor origin\n                    rt = blk["ref_audio_t"]\n                    vt = blk["latent_t"]\n                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])\n                    if rt > 0:\n                        segments.append(("ref_audio", rt * 2))\n                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))\n                        audio_pos.append(torch.arange(row, row + rt * 2))\n                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))\n                        row += rt * 2\n                    n = vt * r_frame.shape[0]\n                    segments.append(("ref_img", n))\n                    pos.append(_video_grid(vt, r_frame, cursor))\n                    img_pos.append(torch.arange(row, row + n))\n                    img_update.append(torch.zeros(n, dtype=torch.bool))\n                    row += n\n                    cursor += max(float(rt), sum(_video_t_spans(vt)))\n\n        # target audio then target video, always the last two segments\n        segments.append(("audio", audio_t * 2))\n        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))\n        audio_pos.append(torch.arange(row, row + audio_t * 2))\n        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))\n        row += audio_t * 2\n\n        n_video = latent_t * frame_rows\n        segments.append(("video", n_video))\n        pos.append(_video_grid(latent_t, frame, cursor))\n        img_pos.append(torch.arange(row, row + n_video))\n        img_update.append(torch.ones(n_video, dtype=torch.bool))\n        row += n_video\n\n        self.seq_len = row\n        self.position_ids = torch.cat(pos)  # [S, 3] float64\n        self.img_pos = torch.cat(img_pos)\n        self.img_update = torch.cat(img_update)\n        self.audio_pos = torch.cat(audio_pos)\n        self.audio_update = torch.cat(audio_update)\n        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)\n        # contiguous segment table (start, stop, kind)\n        # kinds: text / cond / cond_audio / ref_img / ref_audio / audio / video\n        # the packed sequence is uniform per segment in (modality tag, timestep class),\n        # except the text span (tag runs resolved at forward time from the presentation tags)\n        seg_abs = []\n        off = 0\n        for kind, n in segments:\n            seg_abs.append((off, off + n, kind))\n            off += n\n        self.segments = seg_abs\n\n\nclass MiniMaxH3Model(nn.Module):\n    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,\n                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,\n                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,\n                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,\n                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,\n                 sigma_shift_video=12.0, sigma_shift_audio=3.0,\n                 adaln_curve_grid=None,\n                 image_model=None, dtype=None, device=None, operations=None, **kwargs):\n        super().__init__()\n        self.dtype = dtype\n        self.hidden_size = hidden_size\n        self.patch_size = tuple(patch_size)\n        self.latents_dim = latents_dim\n        self.audio_latents_dim = audio_latents_dim\n        self.sigma_shift_video = sigma_shift_video\n        self.sigma_shift_audio = sigma_shift_audio\n        self.use_adaln_curves = adaln_curve_grid is not None\n        # curve-form checkpoints replace the time embedder and full-width adaln weights with a small shared basis of the time-embedding curve\n        curve = {"apply_silu": not self.use_adaln_curves,\n                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}\n        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]\n\n        self.video_patch_proj = operations.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)\n        self.audio_patch_proj = operations.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)\n        self.condition_proj = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)\n        if self.use_adaln_curves:\n            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32))\n        else:\n            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,\n                                              dtype=torch.float32, device=device, operations=operations)\n        self.rope = nn.Module()\n        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))\n        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,\n                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,\n                                          final_norm_eps, dtype=dtype, device=device, operations=operations)\n        self.blocks = nn.ModuleList([\n            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,\n                     time_embed_dim, norm_eps, qk_norm_eps, **curve, dtype=dtype, device=device, operations=operations)\n            for _ in range(num_layers)])\n        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,\n                                      final_norm_eps, **curve, dtype=dtype, device=device, operations=operations)\n\n    def preprocess_text_embeds(self, text_states):\n        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds."""\n        if text_states.shape[-1] == self.hidden_size:\n            return text_states\n        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)\n\n    def rope_freqs(self, position_ids, device):\n        # [S, 3] float64 -> [S, 96] fp32\n        pos = position_ids.to(torch.float32).to(device)\n        inv = comfy.model_management.cast_to(self.rope.inv_freq, device=device)\n        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]\n        t_f, h_f, w_f = per_axis.unbind(dim=1)\n        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]\n        return torch.cat((half, half), dim=-1)                 # [S, 96]\n\n    def _cond_video_rows(self, payload, device):\n        """Concatenated visual condition rows (normalized latents -> patchified), with condition noise augmentation."""\n        rows = []\n        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)\n        seed = int(payload.get("seed", 0))\n        # every condition intentionally restarts the same RNG stream\n        for z in payload.get("cond_video_latents", []):\n            r = patchify_video(z.to(torch.float32), self.patch_size)\n            if aug < 1.0:\n                gen = torch.Generator("cpu").manual_seed(seed)\n                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)\n                r = aug * r + (1.0 - aug) * noise.to(r.device)\n            rows.append(r.to(device))\n        return torch.cat(rows, dim=0) if rows else None\n\n    def _cond_audio_rows(self, payload, device):\n        rows = []\n        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)\n        seed = int(payload.get("seed", 0)) + 1\n        for z in payload.get("cond_audio_latents", []):\n            r = pack_audio(z.to(torch.float32))\n            if aug < 1.0:\n                gen = torch.Generator("cpu").manual_seed(seed)\n                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)\n                r = aug * r + (1.0 - aug) * noise.to(r.device)\n            rows.append(r.to(device))\n        return torch.cat(rows, dim=0) if rows else None\n\n    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):\n        # the sampler carries the audio as (sigma_v / sigma_a) * x_audio; undo it outside\n        # the wrappers so they and the network see the stream\'s own latent and velocity\n        scale = float((minimax_payload or {}).get("audio_scale", 1.0))\n        audio_src = x[1]\n        if scale != 1.0:\n            shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))\n            shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))\n            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)\n            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)\n            carry = (sigma_a / sigma_v).to(audio_src.dtype)\n            x = [x[0], audio_src * carry]\n\n        out = comfy.patcher_extension.WrapperExecutor.new_class_executor(\n            self._forward,\n            self,\n            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)\n        ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,\n                  denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)\n\n        if scale != 1.0:\n            # d/d(sigma_v) of the carried variable\n            out[1] = ((1.0 - scale) * (audio_src * carry)\n                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])\n        return out\n\n    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):\n        video_x, audio_x = x[0], x[1]\n        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]\n        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)\n        if video_x.shape[0] != 1:\n            raise ValueError("MiniMax H3 supports batch size 1")\n        payload = minimax_payload or {}\n        device = video_x.device\n        dtype = context.dtype  # compute dtype\n\n        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]\n        audio_t = audio_x.shape[-1]\n        text_len = context.shape[1]\n        # extra_conds prebuilds the layout once per sampling run\n        layout = payload.get("layout")\n        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):\n            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,\n                                  keyframes=payload.get("keyframes"),\n                                  refs=payload.get("refs"))\n\n        # model_base passes model_sampling.timestep(sigma) = sigma * 1000\n        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))\n        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))\n        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)\n        t_v = float(1.0 - sigma_v)\n        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))\n\n        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1\n        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))\n        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))\n        seg_t = {"text": t_v, "video": t_v, "audio": t_a,\n                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),\n                 "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}\n\n        # masked rows run at their own strength: mask value m puts a row at sigma = m * sigma_stream,\n        # so its label is 1 - m * sigma, clamped at the cond timestep for fully preserved rows\n        t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)\n        t_pin_a = max(t_a, AUDIO_COND_TIMESTEP)\n        video_rows_t = None\n        audio_rows_t = None\n        if denoise_mask is not None:\n            m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)\n            if m is not None:\n                rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)\n                if rows_t.unique().numel() == 1:\n                    seg_t["video"] = float(rows_t[0])\n                else:\n                    video_rows_t = rows_t\n        if audio_denoise_mask is not None:\n            m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)\n            if not bool((m >= 1.0 - 1e-3).all()):\n                sigma_a = 1.0 - t_a\n                rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)\n                if rows_t.unique().numel() == 1:\n                    seg_t["audio"] = float(rows_t[0])\n                else:\n                    audio_rows_t = rows_t\n\n        unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}\n                          | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())\n                          | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))\n        t_row = {t: i for i, t in enumerate(unique_t)}\n        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}\n\n        def rows_to_mod_index(rows_t, tag):\n            # per-row timestep values -> per-row mod-row indices into the t_emb table\n            levels = rows_t.unique()\n            base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],\n                                dtype=torch.long, device=rows_t.device)\n            return base[torch.searchsorted(levels, rows_t)]\n\n        text_tags = payload.get("text_token_tags")\n        mod_segments = []\n        for a, b, kind in layout.segments:\n            row_base = t_row[seg_t[kind]] * 3\n            if kind == "text" and text_tags is not None:\n                # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs\n                tags = text_tags.view(-1).tolist()\n                run_start = 0\n                for i in range(1, b - a + 1):\n                    if i == b - a or tags[i] != tags[run_start]:\n                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))\n                        run_start = i\n            elif kind == "video" and video_rows_t is not None:\n                mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))\n            elif kind == "audio" and audio_rows_t is not None:\n                mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))\n            else:\n                mod_segments.append((a, b, row_base + seg_tag[kind]))\n\n        # embed\n        img_update = layout.img_update.to(device)\n        audio_update = layout.audio_update.to(device)\n        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)\n        audio_rows = pack_audio(audio_x.to(torch.float32))\n        cond_video_rows = self._cond_video_rows(payload, device)\n        cond_audio_rows = self._cond_audio_rows(payload, device)\n\n        all_video_rows = video_rows\n        if cond_video_rows is not None:\n            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)\n            all_video_rows[~img_update] = cond_video_rows\n            all_video_rows[img_update] = video_rows\n        all_audio_rows = audio_rows\n        if cond_audio_rows is not None:\n            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)\n            all_audio_rows[~audio_update] = cond_audio_rows\n            all_audio_rows[audio_update] = audio_rows\n\n        # H3 T4 FP16: retain FP32 embeddings at the residual boundary.\n        # On non-FP16 model paths preserve the model\'s existing compute dtype.\n        embed_dtype = torch.float32 if dtype == torch.float16 else dtype\n        video_embed = self.video_patch_proj(all_video_rows).to(embed_dtype)\n        audio_embed = self.audio_patch_proj(all_audio_rows).to(embed_dtype)\n        text_states = context[0]\n        if text_states.shape[-1] != self.hidden_size:\n            text_states = self.token_refiner(self.condition_proj(text_states),\n                                             transformer_options=transformer_options)\n\n        # segments are contiguous: assemble by slices, embed rows follow segment order\n        # H3 T4 FP16: the residual stream must stay FP32 because repeated block\n        # accumulation can exceed the finite range of FP16. Expensive branch work\n        # remains FP16 inside the model/optimization path.\n        residual_dtype = torch.float32 if dtype == torch.float16 else dtype\n        h = torch.empty(layout.seq_len, self.hidden_size, dtype=residual_dtype, device=device)\n        voff = aoff = 0\n        for a, b, kind in layout.segments:\n            n = b - a\n            if kind == "text":\n                h[a:b] = text_states\n            elif kind in ("cond", "ref_img", "video"):\n                h[a:b] = video_embed[voff:voff + n]\n                voff += n\n            else:  # ref_audio / audio\n                h[a:b] = audio_embed[aoff:aoff + n]\n                aoff += n\n\n        del video_embed, audio_embed\n\n        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)\n        if self.use_adaln_curves:\n            # adaln projections consume interpolated coordinates of the time-embedding curve\n            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)\n            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends\n            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table\n            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part\n        else:\n            t_emb = self.time_embedder(t_vals).to(dtype)\n\n        # rotation table computed once per forward, consumed by the kitchen split-half rope\n        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)\n\n        # blocks\n        patches_replace = transformer_options.get("patches_replace", {})\n        blocks_replace = patches_replace.get("dit", {})\n        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)\n        for i, block in enumerate(self.blocks):\n            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)\n            if ("double_block", i) in blocks_replace:\n                def block_wrap(args):\n                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],\n                                         transformer_options=args["transformer_options"])}\n                h = blocks_replace[("double_block", i)](\n                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,\n                     "transformer_options": transformer_options},\n                    {"original_block": block_wrap})["img"]\n            else:\n                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)\n        if prefetch_queue is not None:\n            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)\n\n        # target streams are single contiguous segments (audio then video, last two)\n        va, vb, _ = next(s for s in layout.segments if s[2] == "video")\n        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")\n        if video_rows_t is not None:\n            video_seg = (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)\n        else:\n            video_seg = (va, vb, t_row[seg_t["video"]])\n        if audio_rows_t is not None:\n            audio_seg = (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)\n        else:\n            audio_seg = (aa, ab, t_row[seg_t["audio"]])\n        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)\n\n        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)\n        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]\n        audio_out = unpack_audio(a)\n\n        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]\n',
        'comfy/ldm/minimax/vae.py': '# MiniMax H3 video VAE: 3D causal CNN encoder + ViT3D decoder.\n\nimport math\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nimport comfy.model_management\nimport comfy.ops\nimport comfy.quant_ops\nimport comfy.rmsnorm\nfrom comfy.ldm.modules.attention import optimized_attention\n\nops = comfy.ops.disable_weight_init\n\nIMAGENET_MEAN = (0.485, 0.456, 0.406)\nIMAGENET_STD = (0.229, 0.224, 0.225)\n\nLATENTS_MEAN = [\n    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,\n    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,\n    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,\n    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,\n    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,\n    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,\n]\n\nLATENTS_STD = [\n    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,\n    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,\n    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,\n    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,\n    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,\n    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,\n]\n\n\n# 3D causal CNN encoder\n\nclass CausalConv3d(ops.Conv3d):\n    # Reflect spatial padding, causal (zeros, front-only) temporal padding.\n    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):\n        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)\n        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)\n\n    def forward(self, x):\n        if sum(self.causal_padding) == 0:\n            return super().forward(x)\n\n        x = F.pad(x, (self.causal_padding[2], self.causal_padding[2], self.causal_padding[1], self.causal_padding[1], 0, 0),  mode="reflect")\n        if x.shape[2] == 1:\n            # single frame: the causal front padding is all zeros truncate the temporal taps instead of convolving zero frames\n            return super().forward(x, autopad="causal_zero")\n        x = F.pad(x, (0, 0, 0, 0, self.causal_padding[0] * 2, 0), mode="constant")\n        return super().forward(x)\n\n\nclass TemporalIsolatedGroupNorm(ops.GroupNorm):\n    # GroupNorm with statistics computed per frame (time merged into batch).\n    def forward(self, x):\n        if x.dim() == 5:\n            b, c, t, h, w = x.shape\n            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)\n            x = super().forward(x)\n            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()\n        return super().forward(x)\n\n\ndef group_norm_3d(num_channels):\n    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)\n\n\nclass Downsample3D(nn.Module):\n    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):\n        super().__init__()\n        self.space_stride = space_stride\n        self.conv = CausalConv3d(\n            in_channels,\n            out_channels,\n            kernel_size=3,\n            padding=(1, 0, 0),\n            stride=(time_stride, space_stride, space_stride),\n        )\n\n    def forward(self, x):\n        if self.space_stride == 2:\n            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")\n        return self.conv(x)\n\n\nclass ResnetBlock3D(nn.Module):\n    def __init__(self, in_channels, out_channels=None):\n        super().__init__()\n        self.in_channels = in_channels\n        out_channels = in_channels if out_channels is None else out_channels\n        self.out_channels = out_channels\n\n        self.norm1 = group_norm_3d(in_channels)\n        self.norm2 = group_norm_3d(out_channels)\n        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)\n        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)\n        if in_channels != out_channels:\n            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)\n\n    def forward(self, x):\n        h = self.conv1(F.silu(self.norm1(x), inplace=True))\n        h = self.conv2(F.silu(self.norm2(h), inplace=True))\n        if self.in_channels != self.out_channels:\n            x = self.nin_shortcut(x)\n        return h.add_(x)\n\n\nclass EncoderFCN3D(nn.Module):\n    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):\n        super().__init__()\n        self.num_levels = len(ch_mult)\n        if isinstance(num_res_blocks, int):\n            num_res_blocks = [num_res_blocks] * self.num_levels\n        self.num_res_blocks = num_res_blocks\n\n        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]\n        block_in = [block_mid[0]] + block_mid[:-1]\n        block_out = block_mid\n\n        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)\n\n        self.down = nn.ModuleList()\n        for i_level in range(self.num_levels):\n            down = nn.Module()\n            down.block = nn.ModuleList()\n            for i in range(self.num_res_blocks[i_level]):\n                down.block.append(\n                    ResnetBlock3D(\n                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],\n                        out_channels=block_mid[i_level],\n                    )\n                )\n            if space_down[i_level] * time_down[i_level] > 1:\n                down.downsample = Downsample3D(\n                    block_mid[i_level],\n                    block_out[i_level],\n                    time_stride=time_down[i_level],\n                    space_stride=space_down[i_level],\n                )\n            self.down.append(down)\n\n        self.norm_out = group_norm_3d(block_out[-1])\n        self.conv_out = CausalConv3d(\n            block_out[-1],\n            2 * z_channels if double_z else z_channels,\n            kernel_size=3,\n            padding=1,\n        )\n\n    def forward(self, x):\n        h = self.conv_in(x)\n        for i_level in range(self.num_levels):\n            for i_block in range(self.num_res_blocks[i_level]):\n                h = self.down[i_level].block[i_block](h)\n            if hasattr(self.down[i_level], "downsample"):\n                h = self.down[i_level].downsample(h)\n        h = F.silu(self.norm_out(h))\n        return self.conv_out(h)\n\n\n# ViT3D decoder\n\ndef create_token_ids(patch_dims, device, dtype):\n    coords_list = []\n    for dim_size in patch_dims:\n        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)\n        coords = coords / dim_size\n        coords = 2.0 * coords - 1.0\n        coords_list.append(coords)\n    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)\n    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)\n\n\nclass RotaryEmbeddingND(nn.Module):\n    def __init__(self, dim, rotary_base=100.0, n_dim=3):\n        super().__init__()\n        self.n_dim = n_dim\n        self.angle_scale = 2.0 * math.pi\n        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)\n        self.register_buffer("inv_freq", inv_freq, persistent=False)\n\n    def forward(self, img_ids):\n        # [B, S, n_dim] -> [B, S, 1, pairs, 2, 2] rotation table for the kitchen split-half rope\n        angles = (\n            self.angle_scale\n            * img_ids[:, :, :, None].float()\n            * self.inv_freq.to(img_ids.device)[None, None, None, :]\n        )\n        angles = angles.flatten(2, 3)\n        c, s = torch.cos(angles), torch.sin(angles)\n        table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)\n        return table.to(img_ids.dtype)\n\n\nclass FeedForward(nn.Module):\n    # Gated SiLU FFN.\n    def __init__(self, dim, mult=4, bias=True, operations=ops):\n        super().__init__()\n        inner_dim = dim * mult\n        self.w1 = operations.Linear(dim, inner_dim * 2, bias=bias)\n        self.w2 = operations.Linear(inner_dim, dim, bias=bias)\n\n    def forward(self, x):\n        gate, x = self.w1(x).chunk(2, dim=-1)\n        return self.w2(F.silu(gate).mul_(x))\n\n\nclass Attention(nn.Module):\n    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):\n        super().__init__()\n        self.dim_head = dim_head\n        self.heads = heads\n        inner_dim = dim_head * heads\n        self.norm_q = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)\n        self.norm_k = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)\n        self.to_qkv = operations.Linear(inner_dim, inner_dim * 3, bias=bias)\n        self.to_out = operations.Linear(inner_dim, inner_dim, bias=bias)\n\n    def forward(self, x, rotary_pos_emb=None):\n        batch_size, seq_len, _ = x.shape\n\n        qkv = self.to_qkv(x)\n        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)\n        query, key, value = torch.chunk(qkv, 3, dim=-1)\n\n        query = comfy.rmsnorm.rms_norm(query, self.norm_q.weight, self.norm_q.eps)\n        key = comfy.rmsnorm.rms_norm(key, self.norm_k.weight, self.norm_k.eps)\n\n        if rotary_pos_emb is not None:\n            rot = rotary_pos_emb.shape[-3] * 2\n            query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(\n                query[..., :rot], key[..., :rot], rotary_pos_emb)\n\n        out = optimized_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),\n                                  self.heads, skip_reshape=True).nan_to_num_(0.0)\n        return self.to_out(out)\n\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):\n        super().__init__()\n        dim = heads * dim_head\n        self.norm1 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)\n        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)\n        self.scale1 = nn.Parameter(torch.empty(dim))\n        self.norm2 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)\n        self.ff = FeedForward(dim=dim, bias=bias, operations=operations)\n        self.scale2 = nn.Parameter(torch.empty(dim))\n\n    def forward(self, x, rotary_pos_emb=None):\n        x = x.addcmul_(self.attn(comfy.rmsnorm.rms_norm(x, self.norm1.weight, self.norm1.eps), rotary_pos_emb), comfy.ops.cast_to_input(self.scale1, x))\n        return x.addcmul_(self.ff(comfy.rmsnorm.rms_norm(x, self.norm2.weight, self.norm2.eps)), comfy.ops.cast_to_input(self.scale2, x))\n\n\nclass ViT3DDecoder(nn.Module):\n    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36, heads=32, dim_head=64, rope_theta=100.0,\n                 rope_dim_ratio=0.75, bias=True, eps=1e-5, num_register_tokens=4, operations=ops):\n        super().__init__()\n        dim = heads * dim_head\n        self.patch_size = patch_size\n        self.patch_size_t = patch_size_t\n        self.out_channels = out_channels\n        self.num_register_tokens = num_register_tokens\n\n        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)\n        self.x_embedder = ops.Linear(in_channels, dim)\n        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))\n        # unused at inference; kept so the checkpoint loads without leftover keys\n        self.register_buffer("mask_token", torch.empty(1, 1, dim))\n\n        self.transformer_blocks = nn.ModuleList(\n            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)\n             for _ in range(num_layers)]\n        )\n\n        self.norm_out = ops.LayerNorm(dim, elementwise_affine=True, eps=eps)\n        self.proj_out = ops.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)\n\n    def forward(self, x):\n        B, C, latent_T, latent_H, latent_W = x.shape\n\n        h = self.x_embedder(x.flatten(2).transpose(1, 2))  # [B, T*H*W, C]\n\n        num_patches = h.shape[1]\n        num_suffix = 1 + self.num_register_tokens\n\n        h = torch.cat([h, comfy.ops.cast_to_input(self.register_tokens, h).expand(B, -1, -1), torch.zeros_like(h[:, 0:1, :])], dim=1)\n\n        img_ids = create_token_ids((latent_T, latent_H, latent_W), x.device, x.dtype).expand(B, -1, -1)\n        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)\n        img_ids = torch.cat([img_ids, suffix_ids], dim=1)\n\n        rotary_pos_emb = self.pos_embed(img_ids)\n\n        for block in self.transformer_blocks:\n            h = block(h, rotary_pos_emb)\n\n        output = self.proj_out(self.norm_out(h))\n\n        output = output[:, :num_patches, :]\n\n        output = output.view(\n            B, latent_T, latent_H, latent_W,\n            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,\n        )\n        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()\n        output = output.reshape(\n            B, self.out_channels,\n            latent_T * self.patch_size_t,\n            latent_H * self.patch_size,\n            latent_W * self.patch_size,\n        )\n        return output\n\n\n# Full VAE\n\nclass MiniMaxH3VideoVAE(nn.Module):\n    comfy_has_chunked_io = True\n\n    def __init__(\n        self,\n        in_channels=3,\n        out_ch=3,\n        ch=128,\n        embed_dim=24,\n        z_channels=24,\n        ch_mult=(1, 2, 2, 4, 4, 8),\n        num_res_blocks=2,\n        space_down=(2, 2, 2, 2, 1, 1),\n        time_down=(1, 2, 2, 1, 1, 1),\n        clip_length=17,\n        token_drop=3,\n        tile_size=256,\n        tile_overlap_min=64,\n        tiling=True,\n        operations=ops,\n    ):\n        super().__init__()\n        self.vae_ratio = int(math.prod(space_down))\n        self.vae_ratio_t = int(math.prod(time_down))\n\n        # temporal chunking parameters\n        self.clip_length = clip_length\n        self.token_drop = token_drop\n        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t\n        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)\n        self.token_overlap = (-token_drop) % self.tokens_chunk_size\n        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)\n\n        # spatial tiling parameters\n        self.tiling = tiling\n        self.tile_size = tile_size\n        self.tile_overlap_min = tile_overlap_min\n\n        self.encoder = EncoderFCN3D(\n            ch=ch,\n            ch_mult=list(ch_mult),\n            space_down=list(space_down),\n            time_down=list(time_down),\n            num_res_blocks=num_res_blocks,\n            in_channels=in_channels,\n            z_channels=z_channels,\n            double_z=True,\n        )\n        self.quant_conv = ops.Conv3d(z_channels * 2, 2 * embed_dim, 1)\n        self.post_quant_conv = ops.Conv3d(embed_dim, z_channels, 1)\n        self.decoder = ViT3DDecoder(\n            patch_size=self.vae_ratio,\n            patch_size_t=self.vae_ratio_t,\n            in_channels=z_channels,\n            out_channels=out_ch,\n            operations=operations,\n        )\n\n        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))\n        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))\n        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)\n        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)\n\n    # single-shot forward\n\n    def _encode_moments(self, x):\n        return self.quant_conv(self.encoder(x))\n\n    def _decode_pixels(self, z):\n        z = self.post_quant_conv(z)\n\n        # H3 FP16 T4: match decoder input to the actual decoder weight dtype.\n        decoder_dtype = next(self.decoder.parameters()).dtype\n        if z.dtype != decoder_dtype:\n            z = z.to(decoder_dtype)\n\n        return self.decoder(z)\n\n\n    def _normalize_pixels(self, x):\n        return x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))\n\n    def _finalize_pixels(self, part):\n        # raw decoder output -> float32 pixels in [0, 1] (the VAE wrapper\'s process_output is identity)\n        part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)\n        return part.add_(self.pixel_mean.to(device=part.device, dtype=torch.float32)).clamp_(0.0, 1.0)\n\n    def decode_output_shape(self, input_shape):\n        b, c, t, h, w = input_shape\n        if t == 1:\n            frames = 1\n        else:\n            pad_tokens, num_chunks = self._decode_temporal_chunks(t)\n            frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)\n        return (b, self.decoder.out_channels, frames, h * self.vae_ratio, w * self.vae_ratio)\n\n    def _adaptive_encode(self, x):\n        if self.tiling:\n            return self.tiled_encode(x)\n        return self._encode_moments(x)\n\n    def _adaptive_decode(self, z):\n        if self.tiling:\n            return self.tiled_decode(z)\n        return self._decode_pixels(z)\n\n    # spatial tiling\n\n    def split_tiles(self, input_len):\n        tile_size = self.tile_size\n        if tile_size >= input_len:\n            return [0], [input_len], []\n\n        N = math.ceil(input_len / tile_size)\n        while True:\n            overlaps = [self.tile_overlap_min] * (N - 1)\n            remaining = tile_size * N - sum(overlaps) - input_len\n            if remaining < 0:\n                N += 1\n            else:\n                break\n\n        remaining_units = remaining // self.vae_ratio\n        for i in range(remaining_units):\n            overlaps[i % (N - 1)] += self.vae_ratio\n\n        tile_start_idx = [0]\n        for i in range(N - 1):\n            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])\n\n        return tile_start_idx, [tile_size] * N, overlaps\n\n    def blend(self, a, b, blend_extent, dim):\n        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)\n\n        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)\n        weight_a = 1 - positions / blend_extent\n        weight_b = positions / blend_extent\n\n        shape = [1] * a.ndim\n        shape[dim] = blend_extent\n        weight_a = weight_a.view(shape)\n        weight_b = weight_b.view(shape)\n\n        slice_a = [slice(None)] * a.ndim\n        slice_a[dim] = slice(-blend_extent, None)\n        slice_b = [slice(None)] * b.ndim\n        slice_b[dim] = slice(0, blend_extent)\n\n        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b\n\n        if blend_extent < b.shape[dim]:\n            slice_b_rest = [slice(None)] * b.ndim\n            slice_b_rest[dim] = slice(blend_extent, None)\n            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)\n        return blended\n\n    def tiled_encode(self, x):\n        height, width = x.shape[-2], x.shape[-1]\n        y_idx, y_len, y_overlap = self.split_tiles(height)\n        x_idx, x_len, x_overlap = self.split_tiles(width)\n\n        rows = []\n        for i_pos, i_len in zip(y_idx, y_len):\n            row = []\n            for j_pos, j_len in zip(x_idx, x_len):\n                tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]\n                row.append(self._encode_moments(tile))\n            rows.append(row)\n\n        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]\n        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]\n\n        result_rows = []\n        for i, row in enumerate(rows):\n            result_row = []\n            for j, tile in enumerate(row):\n                if i > 0:\n                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)\n                if j > 0:\n                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)\n                if i < len(rows) - 1:\n                    tile = tile[..., :-latent_y_overlap[i], :]\n                if j < len(row) - 1:\n                    tile = tile[..., :, :-latent_x_overlap[j]]\n                result_row.append(tile)\n            result_rows.append(torch.cat(result_row, dim=-1))\n        return torch.cat(result_rows, dim=-2)\n\n    def tiled_decode(self, z):\n        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio\n        y_idx, y_len, y_overlap = self.split_tiles(height)\n        x_idx, x_len, x_overlap = self.split_tiles(width)\n\n        # Blended tiles are written straight into a pre-allocated canvas.\n        canvas = None\n        row_tails = []\n        out_y = 0\n        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):\n            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio\n            new_tails = []\n            left_tail = None\n            out_x = 0\n            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):\n                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio\n                tile = self._decode_pixels(z[..., zi:zi + zl, zj:zj + zw])\n                if i < len(y_idx) - 1:\n                    new_tails.append(tile[..., -y_overlap[i]:, :].clone())\n                next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None\n                if i > 0:\n                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)\n                if j > 0:\n                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)\n                left_tail = next_left_tail\n                if i < len(y_idx) - 1:\n                    tile = tile[..., :-y_overlap[i], :]\n                if j < len(x_idx) - 1:\n                    tile = tile[..., :, :-x_overlap[j]]\n                if canvas is None:\n                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)\n                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)\n                out_x += tile.shape[-1]\n            row_tails = new_tails\n            out_y += tile.shape[-2]\n        return canvas\n\n    # temporal chunking\n\n    def encode_temporal(self, x, device):\n        # chunked input io: x may live on the CPU, clips move to the device as they encode\n        z_list = []\n        for i in range(math.ceil(x.shape[2] / self.clip_length)):\n            clip_x = x[:, :, i * self.clip_length:(i + 1) * self.clip_length, :, :].to(device)\n            if clip_x.shape[2] < self.clip_length:\n                pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)\n                clip_x = torch.cat([clip_x, pad_frames], dim=2)\n            z_list.append(self._adaptive_encode(self._normalize_pixels(clip_x)))\n\n        z = torch.cat(z_list, dim=2)\n        if self.token_drop > 0:\n            z = z[:, :, :-self.token_drop]\n        return z\n\n    def _decode_temporal_pad_frames(self, z_len, pad_tokens):\n        if pad_tokens <= 0:\n            return 0\n        intra_tail = self.clip_length % self.vae_ratio_t\n        if intra_tail == 0:\n            return pad_tokens * self.vae_ratio_t\n\n        z_len_before_pad = z_len - pad_tokens\n        return sum(\n            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0\n             else self.vae_ratio_t)\n            for k in range(pad_tokens)\n        )\n\n    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):\n        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t\n        split_count = int(self.token_drop > 0) + 1\n        total_frames = 0\n        final_overlap_frames = 0\n\n        for i in range(num_chunks):\n            t_start_idx = i * self.tokens_chunk_size\n            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap\n            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))\n            clip_frame_len = clip_token_len * self.vae_ratio_t\n\n            for j in range(split_count):\n                f_start_idx = j * chunk_dec\n                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)\n                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)\n                if j == 0:\n                    total_frames += chunk_frames\n                else:\n                    final_overlap_frames = chunk_frames\n\n        total_frames += final_overlap_frames\n        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)\n\n    def _decode_temporal_chunks(self, z_len):\n        pseudo_total_tokens = z_len + self.token_drop\n        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size\n        pseudo_total_tokens += pad_tokens\n\n        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)\n        if num_chunks < 1:\n            # too few tokens for one chunk (e.g. T_lat == 2): pad one extra chunk\n            pad_tokens += self.tokens_chunk_size\n            num_chunks += 1\n        return pad_tokens, num_chunks\n\n    def decode_temporal(self, z, output_buffer=None):\n        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t\n        split_count = int(self.token_drop > 0) + 1\n\n        if output_buffer is None:\n            # finalized chunks stream out of VRAM so the full video never sits on the GPU\n            output_buffer = torch.empty(self.decode_output_shape(z.shape), dtype=torch.float32,\n                                        device=comfy.model_management.intermediate_device())\n\n        pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])\n        if pad_tokens > 0:\n            pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)\n            z = torch.cat([z, pad_z], dim=2)\n\n        dec = output_buffer\n        dec_overlap = None\n        write_pos = 0\n\n        def write_part(part):\n            nonlocal write_pos\n            part_frames = part.shape[2]\n            if part_frames <= 0:\n                return\n            part = self._finalize_pixels(part)\n            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))\n            if copy_frames > 0:\n                dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(\n                    part[:, :, :copy_frames, :, :]\n                )\n                write_pos += copy_frames\n\n        for i in range(num_chunks):\n            t_start_idx = i * self.tokens_chunk_size\n            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap\n            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]\n\n            clip_dec = self._adaptive_decode(clip_z)\n\n            for j in range(split_count):\n                f_start_idx = j * chunk_dec\n                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])\n                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]\n                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]\n\n                if j == 0:\n                    if dec_overlap is not None:\n                        clip_dec_chunk = self.blend(\n                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3\n                        )\n                        dec_overlap = None\n                    write_part(clip_dec_chunk)\n                else:\n                    dec_overlap = clip_dec_chunk.contiguous()\n\n            if i == num_chunks - 1 and dec_overlap is not None:\n                write_part(dec_overlap)\n                dec_overlap = None\n\n            del clip_dec, clip_z\n\n        return dec\n\n\n    def encode(self, x, device=None):\n        # x: [B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 24, T_lat, H/16, W/16]\n        if x.ndim == 4:\n            x = x.unsqueeze(2)\n        if device is None:\n            device = x.device\n\n        if x.shape[2] == 1:\n            moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))\n            moments = moments[:, :, -1:, :, :]\n        else:\n            moments = self.encode_temporal(x, device)\n\n        mean = torch.chunk(moments.float(), 2, dim=1)[0]\n\n        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)\n        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)\n        return (mean - latents_mean) / latents_std\n\n    def encode_tiled(self, x, **kwargs):\n        # tiling is always on internally with the reference\'s semantic tile sizes, ignore tiling fallbacks\n        return self.encode(x)\n\n    def decode_tiled(self, z, **kwargs):\n        return self.decode(z)\n\n    def decode(self, z, output_buffer=None):\n        # z: [B, 24, T_lat, H_lat, W_lat] normalized latents -> float32 pixels [B, 3, T, H, W] in [0, 1]\n        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)\n        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)\n        z = z * latents_std + latents_mean\n\n        if z.shape[2] == 1:\n            dec = self._finalize_pixels(self._adaptive_decode(z)[:, :, -1:, :, :])\n            if output_buffer is None:\n                return dec\n            output_buffer.copy_(dec)\n            return output_buffer\n        return self.decode_temporal(z, output_buffer)\n',
        'comfy/supported_models.py': 'import torch\nfrom . import model_base\nfrom . import utils\n\nfrom . import sd1_clip\nfrom . import sdxl_clip\nimport comfy.text_encoders.sd2_clip\nimport comfy.text_encoders.sd3_clip\nimport comfy.text_encoders.sa_t5\nimport comfy.text_encoders.sa3\nimport comfy.text_encoders.aura_t5\nimport comfy.text_encoders.pixart_t5\nimport comfy.text_encoders.hydit\nimport comfy.text_encoders.flux\nimport comfy.text_encoders.genmo\nimport comfy.text_encoders.lt\nimport comfy.text_encoders.hunyuan_video\nimport comfy.text_encoders.minimax\nimport comfy.text_encoders.minimax_music\nimport comfy.text_encoders.cosmos\nimport comfy.text_encoders.lumina2\nimport comfy.text_encoders.wan\nimport comfy.text_encoders.ace\nimport comfy.text_encoders.omnigen2\nimport comfy.text_encoders.qwen_image\nimport comfy.text_encoders.hunyuan_image\nimport comfy.text_encoders.kandinsky5\nimport comfy.text_encoders.z_image\nimport comfy.text_encoders.ideogram4\nimport comfy.text_encoders.boogu\nimport comfy.text_encoders.krea2\nimport comfy.text_encoders.mage_flow\nimport comfy.text_encoders.joyimage\nimport comfy.text_encoders.anima\nimport comfy.text_encoders.ace15\nimport comfy.text_encoders.longcat_image\nimport comfy.text_encoders.ernie\nimport comfy.text_encoders.cogvideo\nimport comfy.text_encoders.hidream_o1\nimport comfy.text_encoders.pixeldit\n\nfrom . import supported_models_base\nfrom . import latent_formats\n\nfrom . import diffusers_convert\nimport comfy.model_management\n\nclass SD15(supported_models_base.BASE):\n    unet_config = {\n        "context_dim": 768,\n        "model_channels": 320,\n        "use_linear_in_transformer": False,\n        "adm_in_channels": None,\n        "use_temporal_attention": False,\n    }\n\n    unet_extra_config = {\n        "num_heads": 8,\n        "num_head_channels": -1,\n    }\n\n    latent_format = latent_formats.SD15\n    memory_usage_factor = 1.0\n\n    def process_clip_state_dict(self, state_dict):\n        k = list(state_dict.keys())\n        for x in k:\n            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):\n                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")\n                state_dict[y] = state_dict.pop(x)\n\n        if \'cond_stage_model.transformer.text_model.embeddings.position_ids\' in state_dict:\n            ids = state_dict[\'cond_stage_model.transformer.text_model.embeddings.position_ids\']\n            if ids.dtype == torch.float32:\n                state_dict[\'cond_stage_model.transformer.text_model.embeddings.position_ids\'] = ids.round()\n\n        replace_prefix = {}\n        replace_prefix["cond_stage_model."] = "clip_l."\n        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)\n        return state_dict\n\n    def process_clip_state_dict_for_saving(self, state_dict):\n        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]\n        for p in pop_keys:\n            if p in state_dict:\n                state_dict.pop(p)\n\n        replace_prefix = {"clip_l.": "cond_stage_model."}\n        return utils.state_dict_prefix_replace(state_dict, replace_prefix)\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)\n\nclass SD20(supported_models_base.BASE):\n    unet_config = {\n        "context_dim": 1024,\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "adm_in_channels": None,\n        "use_temporal_attention": False,\n    }\n\n    unet_extra_config = {\n        "num_heads": -1,\n        "num_head_channels": 64,\n        "attn_precision": torch.float32,\n    }\n\n    latent_format = latent_formats.SD15\n    memory_usage_factor = 1.0\n\n    def model_type(self, state_dict, prefix=""):\n        if self.unet_config["in_channels"] == 4: #SD2.0 inpainting models are not v prediction\n            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)\n            out = state_dict.get(k, None)\n            if out is not None and torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.\n                return model_base.ModelType.V_PREDICTION\n        return model_base.ModelType.EPS\n\n    def process_clip_state_dict(self, state_dict):\n        replace_prefix = {}\n        replace_prefix["conditioner.embedders.0.model."] = "clip_h." #SD2 in sgm format\n        replace_prefix["cond_stage_model.model."] = "clip_h."\n        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)\n        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_h.", "clip_h.transformer.")\n        return state_dict\n\n    def process_clip_state_dict_for_saving(self, state_dict):\n        replace_prefix = {}\n        replace_prefix["clip_h"] = "cond_stage_model.model"\n        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)\n        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)\n        return state_dict\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.sd2_clip.SD2Tokenizer, comfy.text_encoders.sd2_clip.SD2ClipModel)\n\nclass SD21UnclipL(SD20):\n    unet_config = {\n        "context_dim": 1024,\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "adm_in_channels": 1536,\n        "use_temporal_attention": False,\n    }\n\n    clip_vision_prefix = "embedder.model.visual."\n    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 768}\n\n\nclass SD21UnclipH(SD20):\n    unet_config = {\n        "context_dim": 1024,\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "adm_in_channels": 2048,\n        "use_temporal_attention": False,\n    }\n\n    clip_vision_prefix = "embedder.model.visual."\n    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1024}\n\nclass SDXLRefiner(supported_models_base.BASE):\n    unet_config = {\n        "model_channels": 384,\n        "use_linear_in_transformer": True,\n        "context_dim": 1280,\n        "adm_in_channels": 2560,\n        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],\n        "use_temporal_attention": False,\n    }\n\n    latent_format = latent_formats.SDXL\n    memory_usage_factor = 1.0\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.SDXLRefiner(self, device=device)\n\n    def process_clip_state_dict(self, state_dict):\n        keys_to_replace = {}\n        replace_prefix = {}\n        replace_prefix["conditioner.embedders.0.model."] = "clip_g."\n        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)\n\n        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")\n        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)\n        return state_dict\n\n    def process_clip_state_dict_for_saving(self, state_dict):\n        replace_prefix = {}\n        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")\n        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:\n            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")\n        replace_prefix["clip_g"] = "conditioner.embedders.0.model"\n        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)\n        return state_dict_g\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLRefinerClipModel)\n\nclass SDXL(supported_models_base.BASE):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 0, 2, 2, 10, 10],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n    }\n\n    latent_format = latent_formats.SDXL\n\n    memory_usage_factor = 0.8\n\n    def model_type(self, state_dict, prefix=""):\n        if \'edm_mean\' in state_dict and \'edm_std\' in state_dict: #Playground V2.5\n            self.latent_format = latent_formats.SDXL_Playground_2_5()\n            self.sampling_settings["sigma_data"] = 0.5\n            self.sampling_settings["sigma_max"] = 80.0\n            self.sampling_settings["sigma_min"] = 0.002\n            return model_base.ModelType.EDM\n        elif "edm_vpred.sigma_max" in state_dict:\n            self.sampling_settings["sigma_max"] = float(state_dict["edm_vpred.sigma_max"].item())\n            if "edm_vpred.sigma_min" in state_dict:\n                self.sampling_settings["sigma_min"] = float(state_dict["edm_vpred.sigma_min"].item())\n            return model_base.ModelType.V_PREDICTION_EDM\n        elif "v_pred" in state_dict:\n            if "ztsnr" in state_dict: #Some zsnr anime checkpoints\n                self.sampling_settings["zsnr"] = True\n            return model_base.ModelType.V_PREDICTION\n        else:\n            return model_base.ModelType.EPS\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)\n        if self.inpaint_model():\n            out.set_inpaint()\n        return out\n\n    def process_clip_state_dict(self, state_dict):\n        keys_to_replace = {}\n        replace_prefix = {}\n\n        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"\n        replace_prefix["conditioner.embedders.1.model."] = "clip_g."\n        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)\n\n        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)\n        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")\n        return state_dict\n\n    def process_clip_state_dict_for_saving(self, state_dict):\n        replace_prefix = {}\n        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")\n        for k in state_dict:\n            if k.startswith("clip_l"):\n                state_dict_g[k] = state_dict[k]\n\n        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))\n        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]\n        for p in pop_keys:\n            if p in state_dict_g:\n                state_dict_g.pop(p)\n\n        replace_prefix["clip_g"] = "conditioner.embedders.1.model"\n        replace_prefix["clip_l"] = "conditioner.embedders.0"\n        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)\n        return state_dict_g\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLClipModel)\n\nclass SSD1B(SDXL):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 0, 2, 2, 4, 4],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n    }\n\nclass Segmind_Vega(SDXL):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 0, 1, 1, 2, 2],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n    }\n\nclass KOALA_700M(SDXL):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 2, 5],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n    }\n\nclass KOALA_1B(SDXL):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 2, 6],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n    }\n\nclass SVD_img2vid(supported_models_base.BASE):\n    unet_config = {\n        "model_channels": 320,\n        "in_channels": 8,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],\n        "context_dim": 1024,\n        "adm_in_channels": 768,\n        "use_temporal_attention": True,\n        "use_temporal_resblock": True\n    }\n\n    unet_extra_config = {\n        "num_heads": -1,\n        "num_head_channels": 64,\n        "attn_precision": torch.float32,\n    }\n\n    clip_vision_prefix = "conditioner.embedders.0.open_clip.model.visual."\n\n    latent_format = latent_formats.SD15\n\n    sampling_settings = {"sigma_max": 700.0, "sigma_min": 0.002}\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SVD_img2vid(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass SV3D_u(SVD_img2vid):\n    unet_config = {\n        "model_channels": 320,\n        "in_channels": 8,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],\n        "context_dim": 1024,\n        "adm_in_channels": 256,\n        "use_temporal_attention": True,\n        "use_temporal_resblock": True\n    }\n\n    vae_key_prefix = ["conditioner.embedders.1.encoder."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SV3D_u(self, device=device)\n        return out\n\nclass SV3D_p(SV3D_u):\n    unet_config = {\n        "model_channels": 320,\n        "in_channels": 8,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],\n        "context_dim": 1024,\n        "adm_in_channels": 1280,\n        "use_temporal_attention": True,\n        "use_temporal_resblock": True\n    }\n\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SV3D_p(self, device=device)\n        return out\n\nclass Stable_Zero123(supported_models_base.BASE):\n    unet_config = {\n        "context_dim": 768,\n        "model_channels": 320,\n        "use_linear_in_transformer": False,\n        "adm_in_channels": None,\n        "use_temporal_attention": False,\n        "in_channels": 8,\n    }\n\n    unet_extra_config = {\n        "num_heads": 8,\n        "num_head_channels": -1,\n    }\n\n    required_keys = {\n        "cc_projection.weight": None,\n        "cc_projection.bias": None,\n    }\n\n    clip_vision_prefix = "cond_stage_model.model.visual."\n\n    latent_format = latent_formats.SD15\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Stable_Zero123(self, device=device, cc_projection_weight=state_dict["cc_projection.weight"], cc_projection_bias=state_dict["cc_projection.bias"])\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass SD_X4Upscaler(SD20):\n    unet_config = {\n        "context_dim": 1024,\n        "model_channels": 256,\n        \'in_channels\': 7,\n        "use_linear_in_transformer": True,\n        "adm_in_channels": None,\n        "use_temporal_attention": False,\n    }\n\n    unet_extra_config = {\n        "disable_self_attentions": [True, True, True, False],\n        "num_classes": 1000,\n        "num_heads": 8,\n        "num_head_channels": -1,\n    }\n\n    latent_format = latent_formats.SD_X4\n\n    sampling_settings = {\n        "linear_start": 0.0001,\n        "linear_end": 0.02,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SD_X4Upscaler(self, device=device)\n        return out\n\nclass Stable_Cascade_C(supported_models_base.BASE):\n    unet_config = {\n        "stable_cascade_stage": \'c\',\n    }\n\n    unet_extra_config = {}\n\n    latent_format = latent_formats.SC_Prior\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    sampling_settings = {\n        "shift": 2.0,\n    }\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoder."]\n    clip_vision_prefix = "clip_l_vision."\n\n    def process_unet_state_dict(self, state_dict):\n        key_list = list(state_dict.keys())\n        for y in ["weight", "bias"]:\n            suffix = "in_proj_{}".format(y)\n            keys = filter(lambda a: a.endswith(suffix), key_list)\n            for k_from in keys:\n                weights = state_dict.pop(k_from)\n                prefix = k_from[:-(len(suffix) + 1)]\n                shape_from = weights.shape[0] // 3\n                for x in range(3):\n                    p = ["to_q", "to_k", "to_v"]\n                    k_to = "{}.{}.{}".format(prefix, p[x], y)\n                    state_dict[k_to] = weights[shape_from*x:shape_from*(x + 1)]\n        return state_dict\n\n    def process_clip_state_dict(self, state_dict):\n        state_dict = utils.state_dict_prefix_replace(state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True)\n        if "clip_g.text_projection" in state_dict:\n            state_dict["clip_g.transformer.text_projection.weight"] = state_dict.pop("clip_g.text_projection").transpose(0, 1)\n        return state_dict\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.StableCascade_C(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(sdxl_clip.StableCascadeTokenizer, sdxl_clip.StableCascadeClipModel)\n\nclass Stable_Cascade_B(Stable_Cascade_C):\n    unet_config = {\n        "stable_cascade_stage": \'b\',\n    }\n\n    unet_extra_config = {}\n\n    latent_format = latent_formats.SC_B\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    sampling_settings = {\n        "shift": 1.0,\n    }\n\n    clip_vision_prefix = None\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.StableCascade_B(self, device=device)\n        return out\n\nclass SD15_instructpix2pix(SD15):\n    unet_config = {\n        "context_dim": 768,\n        "model_channels": 320,\n        "use_linear_in_transformer": False,\n        "adm_in_channels": None,\n        "use_temporal_attention": False,\n        "in_channels": 8,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.SD15_instructpix2pix(self, device=device)\n\nclass SDXL_instructpix2pix(SDXL):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "transformer_depth": [0, 0, 2, 2, 10, 10],\n        "context_dim": 2048,\n        "adm_in_channels": 2816,\n        "use_temporal_attention": False,\n        "in_channels": 8,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.SDXL_instructpix2pix(self, model_type=self.model_type(state_dict, prefix), device=device)\n\nclass LotusD(SD20):\n    unet_config = {\n        "model_channels": 320,\n        "use_linear_in_transformer": True,\n        "use_temporal_attention": False,\n        "adm_in_channels": 4,\n        "in_channels": 4,\n    }\n\n    unet_extra_config = {\n        "num_classes": \'sequential\',\n        "num_head_channels": 64,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.Lotus(self, device=device)\n\nclass SD3(supported_models_base.BASE):\n    unet_config = {\n        "in_channels": 16,\n        "pos_embed_scaling_factor": None,\n    }\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.SD3\n\n    memory_usage_factor = 1.6\n\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SD3(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        clip_l = False\n        clip_g = False\n        t5 = False\n        pref = self.text_encoder_key_prefix[0]\n        if "{}clip_l.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:\n            clip_l = True\n        if "{}clip_g.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:\n            clip_g = True\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        if "dtype_t5" in t5_detect:\n            t5 = True\n\n        return supported_models_base.ClipTarget(comfy.text_encoders.sd3_clip.SD3Tokenizer, comfy.text_encoders.sd3_clip.sd3_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, **t5_detect))\n\nclass StableAudio(supported_models_base.BASE):\n    unet_config = {\n        "audio_model": "dit1.0",\n    }\n\n    sampling_settings = {"sigma_max": 500.0, "sigma_min": 0.03}\n\n    unet_extra_config = {}\n    latent_format = latent_formats.StableAudio1\n\n    text_encoder_key_prefix = ["text_encoders."]\n    vae_key_prefix = ["pretransform.model."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        seconds_start_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_start.": ""}, filter_keys=True)\n        seconds_total_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_total.": ""}, filter_keys=True)\n        return model_base.StableAudio1(self, seconds_start_embedder_weights=seconds_start_sd, seconds_total_embedder_weights=seconds_total_sd, device=device)\n\n    def process_unet_state_dict(self, state_dict):\n        for k in list(state_dict.keys()):\n            if k.endswith(".cross_attend_norm.beta") or k.endswith(".ff_norm.beta") or k.endswith(".pre_norm.beta"): #These weights are all zero\n                state_dict.pop(k)\n        return state_dict\n\n    def process_unet_state_dict_for_saving(self, state_dict):\n        replace_prefix = {"": "model.model."}\n        return utils.state_dict_prefix_replace(state_dict, replace_prefix)\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.sa_t5.SAT5Tokenizer, comfy.text_encoders.sa_t5.SAT5Model)\n\nclass StableAudio3(StableAudio):\n    unet_config = {\n        "audio_model": "dit1.0",\n        "global_cond_shared_embed": True,\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 2.0,\n    }\n\n    latent_format = latent_formats.StableAudio3\n\n    memory_usage_factor = 7\n\n    def get_model(self, state_dict, prefix="", device=None):\n        seconds_total_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_total.": ""}, filter_keys=True)\n        padding_embedding = state_dict.get("conditioner.conditioners.prompt.padding_embedding", None)\n        return model_base.StableAudio3(self, seconds_total_embedder_weights=seconds_total_sd, padding_embedding=padding_embedding, device=device)\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.sa3.SAT5GemmaTokenizer, comfy.text_encoders.sa3.SAT5GemmaModel)\n\nclass AuraFlow(supported_models_base.BASE):\n    unet_config = {\n        "cond_seq_dim": 2048,\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.73,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.SDXL\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.AuraFlow(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.aura_t5.AuraT5Tokenizer, comfy.text_encoders.aura_t5.AuraT5Model)\n\nclass PixArtAlpha(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "pixart_alpha",\n    }\n\n    sampling_settings = {\n        "beta_schedule" : "sqrt_linear",\n        "linear_start"  : 0.0001,\n        "linear_end"    : 0.02,\n        "timesteps"     : 1000,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.SD15\n\n    memory_usage_factor = 0.5\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.PixArt(self, device=device)\n        return out.eval()\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.pixart_t5.PixArtTokenizer, comfy.text_encoders.pixart_t5.PixArtT5XXL)\n\nclass PixArtSigma(PixArtAlpha):\n    unet_config = {\n        "image_model": "pixart_sigma",\n    }\n    latent_format = latent_formats.SDXL\n\nclass HunyuanDiT(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "hydit",\n    }\n\n    unet_extra_config = {\n        "attn_precision": torch.float32,\n    }\n\n    sampling_settings = {\n        "linear_start": 0.00085,\n        "linear_end": 0.018,\n    }\n\n    latent_format = latent_formats.SDXL\n\n    memory_usage_factor = 1.3\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanDiT(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.hydit.HyditTokenizer, comfy.text_encoders.hydit.HyditModel)\n\nclass HunyuanDiT1(HunyuanDiT):\n    unet_config = {\n        "image_model": "hydit1",\n    }\n\n    unet_extra_config = {}\n\n    sampling_settings = {\n        "linear_start" : 0.00085,\n        "linear_end" : 0.03,\n    }\n\nclass Flux(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "flux",\n        "guidance_embed": True,\n    }\n\n    sampling_settings = {\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux\n\n    memory_usage_factor = 3.1 # TODO: debug why flux mem usage is so weird on windows.\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    def process_unet_state_dict(self, state_dict):\n        out_sd = {}\n        for k in list(state_dict.keys()):\n            key_out = k\n            if key_out.endswith("_norm.scale"):\n                key_out = "{}.weight".format(key_out[:-len(".scale")])\n            out_sd[key_out] = state_dict[k]\n        return out_sd\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Flux(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))\n\nclass FluxInpaint(Flux):\n    unet_config = {\n        "image_model": "flux",\n        "guidance_embed": True,\n        "in_channels": 96,\n    }\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\nclass FluxSchnell(Flux):\n    unet_config = {\n        "image_model": "flux",\n        "guidance_embed": False,\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.0,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Flux(self, model_type=model_base.ModelType.FLOW, device=device)\n        return out\n\nclass Flux2(Flux):\n    unet_config = {\n        "image_model": "flux2",\n    }\n\n    sampling_settings = {\n        "shift": 2.02,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux2\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = self.memory_usage_factor * (2.0 * 2.0) * (unet_config[\'hidden_size\'] / 2604)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Flux2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))\n        if len(detect) > 0:\n            detect["model_type"] = "qwen3_4b"\n            return supported_models_base.ClipTarget(comfy.text_encoders.flux.KleinTokenizer, comfy.text_encoders.flux.klein_te(**detect))\n\n        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_8b.transformer.".format(pref))\n        if len(detect) > 0:\n            detect["model_type"] = "qwen3_8b"\n            return supported_models_base.ClipTarget(comfy.text_encoders.flux.KleinTokenizer8B, comfy.text_encoders.flux.klein_te(**detect))\n\n        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}mistral3_24b.transformer.".format(pref))\n        if len(detect) > 0:\n            if "{}mistral3_24b.transformer.model.layers.39.post_attention_layernorm.weight".format(pref) not in state_dict:\n                detect["pruned"] = True\n            return supported_models_base.ClipTarget(comfy.text_encoders.flux.Flux2Tokenizer, comfy.text_encoders.flux.flux2_te(**detect))\n\n        return None\n\n\nclass Lens(supported_models_base.BASE):\n    """Microsoft Lens (3.8B dual-stream MMDiT, GPT-OSS-20B text features, Flux2 VAE)."""\n\n    unet_config = {\n        "image_model": "lens",\n    }\n\n    sampling_settings = {\n        "shift": 1.829, # Default mu for 1440x1440 (and any seq_len > 4300\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux2\n\n    memory_usage_factor = 4.0\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32] # fp16 causes NaNs\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.Lens(self, model_type=model_base.ModelType.FLUX, device=device)\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        for hint in ("gpt_oss.transformer.", ""):\n            full_prefix = "{}{}".format(pref, hint)\n            if "{}layers.0.self_attn.sinks".format(full_prefix) in state_dict:\n                detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, full_prefix)\n                return supported_models_base.ClipTarget(\n                    comfy.text_encoders.gpt_oss.LensTokenizer,\n                    comfy.text_encoders.gpt_oss.lens_te(**detect),\n                )\n        return supported_models_base.ClipTarget(\n            comfy.text_encoders.gpt_oss.LensTokenizer,\n            comfy.text_encoders.gpt_oss.lens_te(),\n        )\n\n\nclass GenmoMochi(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "mochi_preview",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 6.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Mochi\n\n    memory_usage_factor = 2.0 #TODO\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.GenmoMochi(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.genmo.MochiT5Tokenizer, comfy.text_encoders.genmo.mochi_te(**t5_detect))\n\nclass LTXV(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "ltxv",\n    }\n\n    sampling_settings = {\n        "shift": 2.37,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.LTXV\n\n    memory_usage_factor = 5.5 # TODO: img2vid is about 2x vs txt2vid\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = (unet_config.get("cross_attention_dim", 2048) / 2048) * 5.5\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.LTXV(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.lt.LTXVT5Tokenizer, comfy.text_encoders.lt.ltxv_te(**t5_detect))\n\nclass LTXAV(LTXV):\n    unet_config = {\n        "image_model": "ltxav",\n    }\n\n    latent_format = latent_formats.LTXAV\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = 0.077  # TODO\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.LTXAV(self, device=device)\n        return out\n\nclass MiniMaxH3(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "minimax_h3",\n    }\n\n    sampling_settings = {\n        "shift": 12.0,\n        "audio_shift": 3.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.MiniMaxH3AV\n\n    memory_usage_factor = 0.17\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.MiniMaxH3(self, device=device)\n\n    def clip_target(self, state_dict={}, prefix=""):\n        pref = self.text_encoder_key_prefix[0]\n        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_32b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.minimax.MiniMaxH3Tokenizer, comfy.text_encoders.minimax.te(**detect))\n\nclass HunyuanVideo(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "hunyuan_video",\n    }\n\n    sampling_settings = {\n        "shift": 7.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.HunyuanVideo\n\n    memory_usage_factor = 1.8 #TODO\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanVideo(self, device=device)\n        return out\n\n    def process_unet_state_dict(self, state_dict):\n        out_sd = {}\n        for k in list(state_dict.keys()):\n            key_out = k\n            key_out = key_out.replace("txt_in.t_embedder.mlp.0.", "txt_in.t_embedder.in_layer.").replace("txt_in.t_embedder.mlp.2.", "txt_in.t_embedder.out_layer.")\n            key_out = key_out.replace("txt_in.c_embedder.linear_1.", "txt_in.c_embedder.in_layer.").replace("txt_in.c_embedder.linear_2.", "txt_in.c_embedder.out_layer.")\n            key_out = key_out.replace("_mod.linear.", "_mod.lin.").replace("_attn_qkv.", "_attn.qkv.")\n            key_out = key_out.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")\n            key_out = key_out.replace("_attn_q_norm.weight", "_attn.norm.query_norm.weight").replace("_attn_k_norm.weight", "_attn.norm.key_norm.weight")\n            key_out = key_out.replace(".q_norm.weight", ".norm.query_norm.weight").replace(".k_norm.weight", ".norm.key_norm.weight")\n            key_out = key_out.replace("_attn_proj.", "_attn.proj.")\n            key_out = key_out.replace(".modulation.linear.", ".modulation.lin.")\n            key_out = key_out.replace("_in.mlp.2.", "_in.out_layer.").replace("_in.mlp.0.", "_in.in_layer.")\n            if key_out.endswith(".scale"):\n                key_out = "{}.weight".format(key_out[:-len(".scale")])\n            out_sd[key_out] = state_dict[k]\n        return out_sd\n\n    def process_unet_state_dict_for_saving(self, state_dict):\n        replace_prefix = {"": "model.model."}\n        return utils.state_dict_prefix_replace(state_dict, replace_prefix)\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}llama.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideoTokenizer, comfy.text_encoders.hunyuan_video.hunyuan_video_clip(**hunyuan_detect))\n\nclass HunyuanVideoI2V(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "in_channels": 33,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanVideoI2V(self, device=device)\n        return out\n\nclass HunyuanVideoSkyreelsI2V(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "in_channels": 32,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanVideoSkyreelsI2V(self, device=device)\n        return out\n\nclass CosmosT2V(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "cosmos",\n        "in_channels": 16,\n    }\n\n    sampling_settings = {\n        "sigma_data": 0.5,\n        "sigma_max": 80.0,\n        "sigma_min": 0.002,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Cosmos1CV8x8x8\n\n    memory_usage_factor = 1.6 #TODO\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32] #TODO\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.CosmosVideo(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.cosmos.CosmosT5Tokenizer, comfy.text_encoders.cosmos.te(**t5_detect))\n\nclass CosmosI2V(CosmosT2V):\n    unet_config = {\n        "image_model": "cosmos",\n        "in_channels": 17,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.CosmosVideo(self, image_to_video=True, device=device)\n        return out\n\nclass CosmosT2IPredict2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "cosmos_predict2",\n        "in_channels": 16,\n    }\n\n    sampling_settings = {\n        "sigma_data": 1.0,\n        "sigma_max": 80.0,\n        "sigma_min": 0.002,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Wan21\n\n    memory_usage_factor = 1.0\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = (unet_config.get("model_channels", 2048) / 2048) * 0.95\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.CosmosPredict2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.cosmos.CosmosT5Tokenizer, comfy.text_encoders.cosmos.te(**t5_detect))\n\nclass Anima(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "anima",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 3.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Wan21\n\n    memory_usage_factor = 1.0\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Anima(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_06b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.anima.AnimaTokenizer, comfy.text_encoders.anima.te(**detect))\n\n    def set_inference_dtype(self, dtype, manual_cast_dtype, **kwargs):\n        self.memory_usage_factor = (self.unet_config.get("model_channels", 2048) / 2048) * 0.95\n        if dtype is torch.float16:\n            self.memory_usage_factor *= 1.4\n        return super().set_inference_dtype(dtype, manual_cast_dtype, **kwargs)\n\nclass CosmosI2VPredict2(CosmosT2IPredict2):\n    unet_config = {\n        "image_model": "cosmos_predict2",\n        "in_channels": 17,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.CosmosPredict2(self, image_to_video=True, device=device)\n        return out\n\nclass Lumina2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "lumina2",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 6.0,\n    }\n\n    memory_usage_factor = 1.4\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Lumina2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}gemma2_2b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.lumina2.LuminaTokenizer, comfy.text_encoders.lumina2.te(**hunyuan_detect))\n\nclass ZImage(Lumina2):\n    unet_config = {\n        "image_model": "lumina2",\n        "dim": 3840,\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 3.0,\n    }\n\n    memory_usage_factor = 2.8\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        if comfy.model_management.extended_fp16_support() and unet_config.get("allow_fp16", False):\n            self.supported_inference_dtypes = self.supported_inference_dtypes.copy()\n            self.supported_inference_dtypes.insert(1, torch.float16)\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.z_image.ZImageTokenizer, comfy.text_encoders.z_image.te(**hunyuan_detect))\n\nclass ZImagePixelSpace(ZImage):\n    unet_config = {\n        "image_model": "zimage_pixel",\n    }\n\n    # Pixel-space model: no spatial compression, operates on raw RGB patches.\n    latent_format = latent_formats.ZImagePixelSpace\n\n    # Much lower memory than latent-space models (no VAE, small patches).\n    memory_usage_factor = 0.03 # TODO: figure out the optimal value for this.\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.ZImagePixelSpace(self, device=device)\n\nclass PixelDiTT2I(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "pixeldit_t2i",\n    }\n\n    unet_extra_config = {}\n\n    sampling_settings = {\n        "shift": 4.0,  # 1024px stage 3 default; 2.0 for 512px\n    }\n\n    latent_format = latent_formats.PixelDiTPixel\n    memory_usage_factor = 0.04\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.PixelDiTT2I(self, device=device)\n\n    def process_unet_state_dict(self, state_dict):\n        # pixel_dim from pixel_embedder.proj.weight = (pixel_dim, in_channels); p2 derived per-weight from total // (6 * pixel_dim).\n        pixel_dim = next(v for k, v in state_dict.items() if k.endswith("pixel_embedder.proj.weight")).shape[0]\n\n        out = {}\n        marker = ".adaLN_modulation.0."\n        for k, v in state_dict.items():\n            if k.startswith("_repa_projector") or k.startswith("net_ema."):\n                continue\n            if k.startswith("core."):\n                k = k[len("core."):]\n            elif k.startswith("net."):\n                k = k[len("net."):]\n            if "pixel_blocks." in k and marker in k:\n                # Split into msa (chunks 0-2) and mlp (chunks 3-5) for the two-Linear PiTBlock to reduce peak VRAM\n                p2 = v.shape[0] // (6 * pixel_dim)\n                trail = v.shape[1:]  # () for bias, (in_dim,) for weight\n                vv = v.view(p2, 6, pixel_dim, *trail)\n                base, suffix = k.split(marker)\n                out[f"{base}.adaLN_modulation_msa.{suffix}"] = vv[:, 0:3].reshape(3 * p2 * pixel_dim, *trail).contiguous()\n                out[f"{base}.adaLN_modulation_mlp.{suffix}"] = vv[:, 3:6].reshape(3 * p2 * pixel_dim, *trail).contiguous()\n            else:\n                out[k] = v\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(\n            comfy.text_encoders.pixeldit.PixelDiTGemma2Tokenizer,\n            comfy.text_encoders.pixeldit.PixelDiTGemma2TE,\n        )\n\nclass PiD(PixelDiTT2I):\n    unet_config = {\n        "image_model": "pid",\n    }\n\n    sampling_settings = {\n        "shift": 1.5, # close approximation of the original distill 4 steps [0.999, 0.866, 0.634, 0.342, 0]\n    }\n\n    memory_usage_factor = 0.04\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.PiD(self, device=device)\n\nclass WAN21_T2V(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "t2v",\n    }\n\n    sampling_settings = {\n        "shift": 8.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Wan21\n\n    memory_usage_factor = 0.9\n\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2222\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.wan.WanT5Tokenizer, comfy.text_encoders.wan.te(**t5_detect))\n\nclass WAN21_CausalAR_T2V(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "t2v",\n        "causal_ar": True,\n    }\n\n    sampling_settings = {\n        "shift": 5.0,\n    }\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.unet_config.pop("causal_ar", None)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.WAN21_CausalAR(self, device=device)\n\n\nclass WAN21_I2V(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "i2v",\n        "in_dim": 36,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21(self, image_to_video=True, device=device)\n        return out\n\nclass WAN21_FunControl2V(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "i2v",\n        "in_dim": 48,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21(self, image_to_video=False, device=device)\n        return out\n\nclass WAN21_Camera(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "camera",\n        "in_dim": 32,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)\n        return out\n\nclass WAN22_Camera(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "camera_2.2",\n        "in_dim": 36,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)\n        return out\n\nclass WAN21_Vace(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "vace",\n    }\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = 1.2 * self.memory_usage_factor\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_Vace(self, image_to_video=False, device=device)\n        return out\n\nclass WAN21_HuMo(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "humo",\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_HuMo(self, image_to_video=False, device=device)\n        return out\n\nclass WAN22_S2V(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "s2v",\n    }\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN22_S2V(self, device=device)\n        return out\n\nclass WAN22_Animate(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "animate",\n    }\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN22_Animate(self, device=device)\n        return out\n\nclass WAN_Animate2(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "animate2",\n    }\n\n    sampling_settings = {\n        "shift": 5.0,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN_Animate2(self, device=device)\n        return out\n\nclass WAN22_T2V(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "t2v",\n        "out_dim": 48,\n    }\n\n    latent_format = latent_formats.Wan22\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN22(self, image_to_video=True, device=device)\n        return out\n\nclass Trellis2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "trellis2"\n    }\n\n    unet_extra_config = {"num_heads": 12}\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    memory_usage_factor = 6\n\n    latent_format = latent_formats.Trellis2\n    vae_key_prefix = ["vae."]\n    clip_vision_prefix = "conditioner.main_image_encoder.model."\n    # this is only needed for the texture model\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.Trellis2(self, device=device)\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass WAN21_FlowRVS(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "flow_rvs",\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_FlowRVS(self, image_to_video=True, device=device)\n        return out\n\nclass WAN21_SCAIL(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "scail",\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_SCAIL(self, image_to_video=False, device=device)\n        return out\n\n\nclass WAN21_SCAIL2(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "scail2",\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN21_SCAIL2(self, image_to_video=False, device=device)\n        return out\n\nclass WAN22_WanDancer(WAN21_T2V):\n    unet_config = {\n        "image_model": "wan2.1",\n        "model_type": "wandancer",\n        "in_dim": 36,\n    }\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        self.memory_usage_factor = 1.8\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.WAN22_WanDancer(self, image_to_video=True, device=device)\n        return out\n\n    def process_unet_state_dict(self, state_dict):\n        out_sd = {}\n        for k in list(state_dict.keys()):\n            # split music_encoder in_proj into q_proj, k_proj, v_proj\n            if "music_encoder" in k and "self_attn.in_proj" in k:\n                suffix = "weight" if k.endswith("weight") else "bias"\n                tensor = state_dict[k]\n                d = tensor.shape[0] // 3\n                prefix = k.replace(f"in_proj_{suffix}", "")\n                out_sd[f"{prefix}q_proj.{suffix}"] = tensor[:d]\n                out_sd[f"{prefix}k_proj.{suffix}"] = tensor[d:2*d]\n                out_sd[f"{prefix}v_proj.{suffix}"] = tensor[2*d:]\n            else:\n                out_sd[k] = state_dict[k]\n        return out_sd\n\nclass Hunyuan3Dv2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "hunyuan3d2",\n    }\n\n    unet_extra_config = {}\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.0,\n    }\n\n    memory_usage_factor = 3.5\n\n    clip_vision_prefix = "conditioner.main_image_encoder.model."\n    vae_key_prefix = ["vae."]\n\n    latent_format = latent_formats.Hunyuan3Dv2\n\n    def process_unet_state_dict(self, state_dict):\n        out_sd = {}\n        for k in list(state_dict.keys()):\n            key_out = k\n            if key_out.endswith(".scale"):\n                key_out = "{}.weight".format(key_out[:-len(".scale")])\n            out_sd[key_out] = state_dict[k]\n        return out_sd\n\n    def process_unet_state_dict_for_saving(self, state_dict):\n        replace_prefix = {"": "model."}\n        return utils.state_dict_prefix_replace(state_dict, replace_prefix)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Hunyuan3Dv2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass Hunyuan3Dv2_1(Hunyuan3Dv2):\n    unet_config = {\n        "image_model": "hunyuan3d2_1",\n    }\n\n    latent_format = latent_formats.Hunyuan3Dv2_1\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Hunyuan3Dv2_1(self, device = device)\n        return out\n\nclass Hunyuan3Dv2mini(Hunyuan3Dv2):\n    unet_config = {\n        "image_model": "hunyuan3d2",\n        "depth": 8,\n    }\n\n    latent_format = latent_formats.Hunyuan3Dv2mini\n\nclass TripoSplat(supported_models_base.BASE):\n    # Image -> 3D gaussian splat flow denoiser\n    unet_config = {\n        "image_model": "triposplat",\n    }\n\n    unet_extra_config = {}\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    memory_usage_factor = 0.6\n\n    latent_format = latent_formats.TripoSplat\n\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.TripoSplat(self, device=device)\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass HiDream(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "hidream",\n    }\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    sampling_settings = {\n    }\n\n    # memory_usage_factor = 1.2 # TODO\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HiDream(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None #  TODO\n\nclass HiDreamO1(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "hidream_o1",\n    }\n\n    sampling_settings = {\n        "shift": 3.0,\n        "noise_scale": 8.0,\n    }\n\n    latent_format = latent_formats.HiDreamO1Pixel\n    memory_usage_factor = 0.033\n    # fp16 not supported: LM MLP down_proj activations fp16 overflow, causing NaNs\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    optimizations = {"fp8": False}\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.HiDreamO1(self, device=device)\n\n    def process_unet_state_dict(self, state_dict):\n        # Drop unused Qwen3-VL deepstack merger weights; upstream discards them at inference.\n        for key in list(state_dict.keys()):\n            if "visual.deepstack_merger_list" in key:\n                del state_dict[key]\n        return state_dict\n\n    def process_vae_state_dict(self, state_dict):\n        # Pixel-space model: inject sentinel so VAE construction picks PixelspaceConversionVAE.\n        return {"pixel_space_vae": torch.tensor(1.0)}\n\n    def process_clip_state_dict(self, state_dict):\n        # Tokenizer-only TE: inject sentinel so load_state_dict_guess_config triggers CLIP init.\n        return {"_hidream_o1_te_sentinel": torch.zeros(1)}\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(\n            comfy.text_encoders.hidream_o1.HiDreamO1Tokenizer,\n            comfy.text_encoders.hidream_o1.HiDreamO1TE,\n        )\n\nclass Chroma(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "chroma",\n    }\n\n    unet_extra_config = {\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n    }\n\n    latent_format = comfy.latent_formats.Flux\n\n    memory_usage_factor = 3.2\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    def process_unet_state_dict(self, state_dict):\n        out_sd = {}\n        for k in list(state_dict.keys()):\n            key_out = k\n            if key_out.endswith(".scale"):\n                key_out = "{}.weight".format(key_out[:-len(".scale")])\n            out_sd[key_out] = state_dict[k]\n        return out_sd\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Chroma(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.pixart_t5.PixArtTokenizer, comfy.text_encoders.pixart_t5.pixart_te(**t5_detect))\n\nclass SeedVR2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "seedvr2"\n    }\n    unet_extra_config = {}\n    required_keys = {\n        "{}positive_conditioning",\n        "{}negative_conditioning",\n    }\n    latent_format = comfy.latent_formats.SeedVR2\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n    sampling_settings = {\n        "shift": 1.0,\n    }\n\n    def set_inference_dtype(self, dtype, manual_cast_dtype, device=None):\n        if (\n            dtype == torch.float16\n            and manual_cast_dtype is None\n            and comfy.model_management.should_use_bf16(device)\n        ):\n            manual_cast_dtype = torch.bfloat16\n        super().set_inference_dtype(dtype, manual_cast_dtype, device=device)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.SeedVR2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None\n\nclass ChromaRadiance(Chroma):\n    unet_config = {\n        "image_model": "chroma_radiance",\n    }\n\n    latent_format = comfy.latent_formats.ChromaRadiance\n\n    # Pixel-space model, no spatial compression for model input.\n    memory_usage_factor = 0.044\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.ChromaRadiance(self, device=device)\n\nclass ACEStep(supported_models_base.BASE):\n    unet_config = {\n        "audio_model": "ace",\n    }\n\n    unet_extra_config = {\n    }\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    latent_format = comfy.latent_formats.ACEAudio\n\n    memory_usage_factor = 0.5\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.ACEStep(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.ace.AceT5Tokenizer, comfy.text_encoders.ace.AceT5Model)\n\nclass Omnigen2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "omnigen2",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 2.6,\n    }\n\n    memory_usage_factor = 1.95 #TODO\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        super().__init__(unet_config)\n        if comfy.model_management.extended_fp16_support():\n            self.supported_inference_dtypes = [torch.float16] + self.supported_inference_dtypes\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Omnigen2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_3b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.omnigen2.Omnigen2Tokenizer, comfy.text_encoders.omnigen2.te(**hunyuan_detect))\n\nclass Boogu(Omnigen2):\n    unet_config = {\n        "image_model": "boogu",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 3.16,\n    }\n\n    memory_usage_factor = 2.15\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Boogu(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_8b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.boogu.BooguTokenizer, comfy.text_encoders.boogu.te(**hunyuan_detect))\n\nclass Ideogram4(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "ideogram4",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.0,\n    }\n\n    memory_usage_factor = 11.6\n\n    unet_extra_config = {\n        "num_attention_heads": 18,\n        "attention_head_dim": 256,\n        "intermediate_size": 12288,\n        "adaln_dim": 512,\n        "llm_features_dim": 53248,\n        "rope_theta": 5000000,\n        "mrope_section": [24, 20, 20],\n        "norm_eps": 1e-5,\n    }\n    latent_format = latent_formats.Flux2\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Ideogram4(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_8b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.ideogram4.Ideogram4Tokenizer, comfy.text_encoders.ideogram4.te(**hunyuan_detect))\n\n\nclass Krea2(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "krea2",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.15,\n    }\n\n    memory_usage_factor = 2.2\n\n    latent_format = latent_formats.Wan21\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Krea2(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_4b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.krea2.Krea2Tokenizer, comfy.text_encoders.krea2.te(**hunyuan_detect))\n\nclass MageFlow(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "mage_flow",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 6.0,\n    }\n\n    memory_usage_factor = 6.5\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux2\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.MageFlow(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_4b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.mage_flow.MageFlowTokenizer, comfy.text_encoders.mage_flow.te(**hunyuan_detect))\n\nclass QwenImage(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "qwen_image",\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 1.15,\n    }\n\n    memory_usage_factor = 1.8 #TODO\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Wan21\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.QwenImage(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.qwen_image.QwenImageTokenizer, comfy.text_encoders.qwen_image.te(**hunyuan_detect))\n\nclass JoyImage(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "joyimage",\n    }\n\n    sampling_settings = {\n        "multiplier": 1000,\n        "shift": 1.5,\n    }\n\n    memory_usage_factor = 1.8\n\n    unet_extra_config = {\n        "theta": 10000,\n        "rope_dim_list": [16, 56, 56],\n    }\n\n    latent_format = latent_formats.Wan21\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.JoyImage(self, device=device)\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        qwen3vl_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.joyimage.JoyImageTokenizer, comfy.text_encoders.joyimage.te(**qwen3vl_detect))\n\nclass HunyuanImage21(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "vec_in_dim": None,\n    }\n\n    sampling_settings = {\n        "shift": 5.0,\n    }\n\n    latent_format = latent_formats.HunyuanImage21\n\n    memory_usage_factor = 8.7\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanImage21(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_image.HunyuanImageTokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))\n\nclass HunyuanImage21Refiner(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "patch_size": [1, 1, 1],\n        "vec_in_dim": None,\n    }\n\n    sampling_settings = {\n        "shift": 4.0,\n    }\n\n    latent_format = latent_formats.HunyuanImage21Refiner\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanImage21Refiner(self, device=device)\n        return out\n\nclass HunyuanVideo15(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "vision_in_dim": 1152,\n    }\n\n    sampling_settings = {\n        "shift": 7.0,\n    }\n    memory_usage_factor = 4.0 #TODO\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    latent_format = latent_formats.HunyuanVideo15\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanVideo15(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))\n\n\nclass HunyuanVideo15_SR_Distilled(HunyuanVideo):\n    unet_config = {\n        "image_model": "hunyuan_video",\n        "vision_in_dim": 1152,\n        "in_channels": 98,\n    }\n\n    sampling_settings = {\n        "shift": 2.0,\n    }\n    memory_usage_factor = 4.0 #TODO\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    latent_format = latent_formats.HunyuanVideo15\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.HunyuanVideo15_SR_Distilled(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))\n\n\nclass Kandinsky5(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "kandinsky5",\n    }\n\n    sampling_settings = {\n        "shift": 10.0,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.HunyuanVideo\n\n    memory_usage_factor = 1.25 #TODO\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Kandinsky5(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.kandinsky5.Kandinsky5Tokenizer, comfy.text_encoders.kandinsky5.te(**hunyuan_detect))\n\n\nclass Kandinsky5Image(Kandinsky5):\n    unet_config = {\n        "image_model": "kandinsky5",\n        "model_dim": 2560,\n        "visual_embed_dim": 64,\n    }\n\n    sampling_settings = {\n        "shift": 3.0,\n    }\n\n    latent_format = latent_formats.Flux\n    memory_usage_factor = 1.25 #TODO\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.Kandinsky5Image(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.kandinsky5.Kandinsky5TokenizerImage, comfy.text_encoders.kandinsky5.te(**hunyuan_detect))\n\n\nclass ACEStep15(supported_models_base.BASE):\n    unet_config = {\n        "audio_model": "ace1.5",\n    }\n\n    unet_extra_config = {\n    }\n\n    sampling_settings = {\n        "multiplier": 1.0,\n        "shift": 3.0,\n    }\n\n    latent_format = comfy.latent_formats.ACEAudio15\n\n    memory_usage_factor = 4.7\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.ACEStep15(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        detect_2b = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_2b.transformer.".format(pref))\n        detect_4b = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))\n        if "dtype_llama" in detect_2b:\n            detect = detect_2b\n            detect["lm_model"] = "qwen3_2b"\n        elif "dtype_llama" in detect_4b:\n            detect = detect_4b\n            detect["lm_model"] = "qwen3_4b"\n\n        return supported_models_base.ClipTarget(comfy.text_encoders.ace15.ACE15Tokenizer, comfy.text_encoders.ace15.te(**detect))\n\nclass MiniMaxMusic3(supported_models_base.BASE):\n    unet_config = {\n        "audio_model": "minimax_music3",\n    }\n\n    latent_format = comfy.latent_formats.MiniMaxMusic3\n    memory_usage_factor = 2.0\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n    sampling_settings = {"multiplier": 1.0}\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.MiniMaxMusic3(self, device=device)\n\n    def model_type(self, state_dict, prefix=""):\n        return model_base.ModelType.FLOW\n\n    def clip_target(self, state_dict={}):\n        detect = comfy.text_encoders.minimax_music.detect_merged_config(state_dict, self.text_encoder_key_prefix[0])\n        target = supported_models_base.ClipTarget(comfy.text_encoders.minimax_music.MiniMaxMusic3Tokenizer, comfy.text_encoders.minimax_music.MiniMaxMusic3TEModel)\n        target.params["projection_config"] = detect\n        return target\n\n\nclass LongCatImage(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "flux",\n        "guidance_embed": False,\n        "vec_in_dim": None,\n        "context_in_dim": 3584,\n        "txt_ids_dims": [1, 2],\n    }\n\n    sampling_settings = {\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux\n\n    memory_usage_factor = 2.5\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.LongCatImage(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.longcat_image.LongCatImageTokenizer, comfy.text_encoders.longcat_image.te(**hunyuan_detect))\n\n\nclass RT_DETR_v4(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "RT_DETR_v4",\n    }\n\n    supported_inference_dtypes = [torch.float16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.RT_DETR_v4(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return None\n\n\nclass DepthAnything3(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "DepthAnything3",\n    }\n\n    # Mono path: no num_heads / num_head_channels needed.\n    unet_extra_config = {}\n\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.DepthAnything3(self, device=device)\n\n    def clip_target(self, state_dict={}):\n        return None\n\n\nclass ErnieImage(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "ernie",\n    }\n\n    sampling_settings = {\n        "multiplier": 1000.0,\n        "shift": 3.0,\n    }\n\n    memory_usage_factor = 10.0\n\n    unet_extra_config = {}\n    latent_format = latent_formats.Flux2\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def get_model(self, state_dict, prefix="", device=None):\n        out = model_base.ErnieImage(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        pref = self.text_encoder_key_prefix[0]\n        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}ministral3_3b.transformer.".format(pref))\n        return supported_models_base.ClipTarget(comfy.text_encoders.ernie.ErnieTokenizer, comfy.text_encoders.ernie.te(**hunyuan_detect))\n\n\nclass SAM3(supported_models_base.BASE):\n    unet_config = {"image_model": "SAM3"}\n    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]\n    text_encoder_key_prefix = ["detector.backbone.language_backbone."]\n    unet_extra_prefix = ""\n\n    def process_clip_state_dict(self, state_dict):\n        clip_keys = getattr(self, "_clip_stash", {})\n        clip_keys = utils.state_dict_prefix_replace(clip_keys, {"detector.backbone.language_backbone.": "", "backbone.language_backbone.": ""}, filter_keys=True)\n        clip_keys = utils.clip_text_transformers_convert(clip_keys, "encoder.", "sam3_clip.transformer.")\n        return {k: v for k, v in clip_keys.items() if not k.startswith("resizer.")}\n\n    def process_unet_state_dict(self, state_dict):\n        self._clip_stash = {k: state_dict.pop(k) for k in list(state_dict.keys()) if "language_backbone" in k and "resizer" not in k}\n        # SAM3.1: remap tracker.model.* -> tracker.*\n        for k in list(state_dict.keys()):\n            if k.startswith("tracker.model."):\n                state_dict["tracker." + k[len("tracker.model."):]] = state_dict.pop(k)\n        # SAM3.1: remove per-block freqs_cis buffers (computed dynamically)\n        for k in [k for k in list(state_dict.keys()) if ".attn.freqs_cis" in k]:\n            state_dict.pop(k)\n        # Split fused QKV projections\n        for k in [k for k in list(state_dict.keys()) if k.endswith((".in_proj_weight", ".in_proj_bias"))]:\n            t = state_dict.pop(k)\n            base, suffix = k.rsplit(".in_proj_", 1)\n            s = ".weight" if suffix == "weight" else ".bias"\n            d = t.shape[0] // 3\n            state_dict[base + ".q_proj" + s] = t[:d]\n            state_dict[base + ".k_proj" + s] = t[d:2*d]\n            state_dict[base + ".v_proj" + s] = t[2*d:]\n        # Remap tracker SAM decoder transformer key names to match sam.py TwoWayTransformer\n        for k in list(state_dict.keys()):\n            if "sam_mask_decoder.transformer." not in k:\n                continue\n            new_k = k.replace(".mlp.lin1.", ".mlp.0.").replace(".mlp.lin2.", ".mlp.2.").replace(".norm_final_attn.", ".norm_final.")\n            if new_k != k:\n                state_dict[new_k] = state_dict.pop(k)\n        return state_dict\n\n    def get_model(self, state_dict, prefix="", device=None):\n        return model_base.SAM3(self, device=device)\n\n    def clip_target(self, state_dict={}):\n        import comfy.text_encoders.sam3_clip\n        return supported_models_base.ClipTarget(comfy.text_encoders.sam3_clip.SAM3TokenizerWrapper, comfy.text_encoders.sam3_clip.SAM3ClipModelWrapper)\n\n\nclass SAM31(SAM3):\n    unet_config = {"image_model": "SAM31"}\n\n\nclass CogVideoX_T2V(supported_models_base.BASE):\n    unet_config = {\n        "image_model": "cogvideox",\n    }\n\n    sampling_settings = {\n        "linear_start": 0.00085,\n        "linear_end": 0.012,\n        "beta_schedule": "linear",\n        "zsnr": True,\n    }\n\n    unet_extra_config = {}\n    latent_format = latent_formats.CogVideoX\n\n    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]\n\n    vae_key_prefix = ["vae."]\n    text_encoder_key_prefix = ["text_encoders."]\n\n    def __init__(self, unet_config):\n        # 2b-class (dim=1920, heads=30) uses scale_factor=1.15258426.\n        # 5b-class (dim=3072, heads=48) — incl. CogVideoX-5b, 1.5-5B, and\n        # Fun-V1.5 inpainting — uses scale_factor=0.7 per vae/config.json.\n        if unet_config.get("num_attention_heads", 0) >= 48:\n            self.latent_format = latent_formats.CogVideoX1_5\n        super().__init__(unet_config)\n\n    def get_model(self, state_dict, prefix="", device=None):\n        # CogVideoX 1.5 (patch_size_t=2) has different training base dimensions for RoPE\n        if self.unet_config.get("patch_size_t") is not None:\n            self.unet_config.setdefault("sample_height", 96)\n            self.unet_config.setdefault("sample_width", 170)\n            self.unet_config.setdefault("sample_frames", 81)\n        out = model_base.CogVideoX(self, device=device)\n        return out\n\n    def clip_target(self, state_dict={}):\n        return supported_models_base.ClipTarget(comfy.text_encoders.cogvideo.CogVideoXT5Tokenizer, comfy.text_encoders.sd3_clip.T5XXLModel)\n\nclass CogVideoX_I2V(CogVideoX_T2V):\n    unet_config = {\n        "image_model": "cogvideox",\n        "in_channels": 32,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        if self.unet_config.get("patch_size_t") is not None:\n            self.unet_config.setdefault("sample_height", 96)\n            self.unet_config.setdefault("sample_width", 170)\n            self.unet_config.setdefault("sample_frames", 81)\n        out = model_base.CogVideoX(self, image_to_video=True, device=device)\n        return out\n\nclass CogVideoX_Inpaint(CogVideoX_T2V):\n    unet_config = {\n        "image_model": "cogvideox",\n        "in_channels": 48,\n    }\n\n    def get_model(self, state_dict, prefix="", device=None):\n        if self.unet_config.get("patch_size_t") is not None:\n            self.unet_config.setdefault("sample_height", 96)\n            self.unet_config.setdefault("sample_width", 170)\n            self.unet_config.setdefault("sample_frames", 81)\n        out = model_base.CogVideoX(self, image_to_video=True, device=device)\n        return out\n\n\nmodels = [\n    LotusD,\n    Stable_Zero123,\n    SD15_instructpix2pix,\n    SD15,\n    SD20,\n    SD21UnclipL,\n    SD21UnclipH,\n    SDXL_instructpix2pix,\n    SDXLRefiner,\n    SDXL,\n    SSD1B,\n    KOALA_700M,\n    KOALA_1B,\n    Segmind_Vega,\n    SD_X4Upscaler,\n    Stable_Cascade_C,\n    Stable_Cascade_B,\n    SV3D_u,\n    SV3D_p,\n    SD3,\n    StableAudio3,\n    StableAudio,\n    AuraFlow,\n    PixArtAlpha,\n    PixArtSigma,\n    HunyuanDiT,\n    HunyuanDiT1,\n    FluxInpaint,\n    Flux,\n    LongCatImage,\n    FluxSchnell,\n    GenmoMochi,\n    LTXV,\n    LTXAV,\n    MiniMaxH3,\n    HunyuanVideo15_SR_Distilled,\n    HunyuanVideo15,\n    HunyuanImage21Refiner,\n    HunyuanImage21,\n    HunyuanVideoSkyreelsI2V,\n    HunyuanVideoI2V,\n    HunyuanVideo,\n    CosmosT2V,\n    CosmosI2V,\n    CosmosT2IPredict2,\n    CosmosI2VPredict2,\n    ZImagePixelSpace,\n    ZImage,\n    PiD,\n    PixelDiTT2I,\n    Lumina2,\n    WAN22_T2V,\n    WAN21_CausalAR_T2V,\n    WAN21_T2V,\n    WAN21_I2V,\n    WAN21_FunControl2V,\n    WAN21_Vace,\n    WAN21_Camera,\n    WAN22_Camera,\n    WAN22_S2V,\n    WAN21_HuMo,\n    WAN22_Animate,\n    WAN_Animate2,\n    WAN21_FlowRVS,\n    WAN21_SCAIL,\n    WAN21_SCAIL2,\n    WAN22_WanDancer,\n    Hunyuan3Dv2mini,\n    Hunyuan3Dv2,\n    Hunyuan3Dv2_1,\n    TripoSplat,\n    HiDream,\n    HiDreamO1,\n    Chroma,\n    SeedVR2,\n    ChromaRadiance,\n    ACEStep,\n    ACEStep15,\n    MiniMaxMusic3,\n    Omnigen2,\n    Boogu,\n    MageFlow,\n    QwenImage,\n    JoyImage,\n    Ideogram4,\n    Krea2,\n    Flux2,\n    Lens,\n    Kandinsky5Image,\n    Kandinsky5,\n    Anima,\n    RT_DETR_v4,\n    ErnieImage,\n    SAM3,\n    SAM31,\n    CogVideoX_Inpaint,\n    CogVideoX_I2V,\n    CogVideoX_T2V,\n    SVD_img2vid,\n    Trellis2,\n    DepthAnything3,\n]\n',
        'custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py': '"""ComfyUI nodes for the MiniMax-H3 Turbo LoRA (4-step audio-video).\n\nDrops into the stock MiniMax-H3 workflow (t2v and i2v):\n\n  MiniMaxH3TurboLoRA    MODEL -> MODEL   applies the turbo LoRA\n  MiniMaxH3TurboSampler       -> SAMPLER 4-step sampler for SamplerCustomAdvanced\n\nOn older ComfyUI the sampler steps the video and audio streams on their own flow\nschedules (video shift 12, audio shift 3), because a stock single-schedule sampler\nover-steps the audio at 4 steps and it breaks. Recent ComfyUI resolves that dual\nschedule natively (ModelSamplingAV), so this sampler auto-detects it and falls back\nto a plain single-schedule step, avoiding a double-shift that would corrupt the\naudio. Either way it drops into the same workflow slot.\n"""\n\nimport math\nimport os\n\nimport torch\nimport torch.nn.functional as F\nfrom tqdm.auto import trange\n\nimport comfy.samplers\nimport comfy.model_sampling\nimport comfy.lora\nimport comfy.weight_adapter\nimport comfy.utils\nimport comfy.patcher_extension\nimport folder_paths\n\nSHIFT_V, SHIFT_A = 12.0, 3.0\n\n\ndef _time_shift_sigma(sigma, fr, to):\n    base = sigma / (fr + sigma * (1.0 - fr))\n    return to * base / (1.0 + (to - 1.0) * base)\n\n\ndef _time_shift_slope(sigma, fr, to):\n    base = sigma / (fr + sigma * (1.0 - fr))\n    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)\n\n\ndef _audio_sigma(sv):\n    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)\n\n\ndef _audio_slope(sv):\n    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)\n\n\ndef _latent_shapes(model):\n    """[video_shape, audio_shape] the sampler is packing over — video latent is\n    flattened first, then audio, so we need the split point."""\n    guider = getattr(model, "inner_model", model)\n    conds = getattr(guider, "conds", None)\n    if conds:\n        for cond_list in conds.values():\n            for c in (cond_list or []):\n                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}\n                if "latent_shapes" in mc:\n                    return mc["latent_shapes"].cond\n    return None\n\n\ndef _model_sampling(model):\n    """The model\'s model_sampling instance, reached from the object a KSAMPLER\n    hands the sampler function: KSamplerX0Inpaint -> CFGGuider -> predictor, where\n    the predictor carries .model_sampling (comfy/samplers.py accesses exactly\n    model_wrap.inner_model.model_sampling)."""\n    for chain in (("inner_model", "inner_model", "model_sampling"),\n                  ("inner_model", "model_sampling"),\n                  ("model_sampling",)):\n        o = model\n        try:\n            for a in chain:\n                o = getattr(o, a)\n        except AttributeError:\n            continue\n        if o is not None:\n            return o\n    return None\n\n\ndef _native_av_schedule(model):\n    """True when this ComfyUI resolves the MiniMax-H3 audio/video dual flow\n    schedule natively via ModelSamplingAV.\n\n    Recent ComfyUI carries the audio latent scaled onto the video schedule\n    (ModelSamplingAV), so the packed latent is an ordinary single-schedule flow\n    latent and a plain flow step is correct. Re-applying the audio shift here, as\n    older ComfyUI required, would double-shift and corrupt the audio (node issues\n    #6 / #18 / #19, HF discussions #17 / #19). Older ComfyUI has no ModelSamplingAV\n    and still needs the manual dual-schedule step, so this sampler adapts to\n    whichever ComfyUI it runs under."""\n    ms = _model_sampling(model)\n    if ms is None:\n        return False\n    if getattr(ms, "audio_shift", None) is not None:\n        return True\n    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)\n    return av is not None and isinstance(ms, av)\n\n\n@torch.no_grad()\ndef _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,\n                   **kwargs):\n    extra_args = {} if extra_args is None else extra_args\n    s_in = x.new_ones([x.shape[0]])\n    _rms = lambda t: float(t.float().pow(2).mean().sqrt())\n\n    if _native_av_schedule(model):\n        # Recent ComfyUI: ModelSamplingAV already carries the audio stream scaled\n        # onto the video schedule, so the pack is an ordinary single-schedule flow\n        # latent. Step the whole pack with a plain flow (Euler) update — the model\n        # and ModelSamplingAV handle the audio clock. Manually re-shifting the audio\n        # here (the legacy path below) would double-apply and corrupt the audio.\n        print(f"[H3TURBO sampler] native ModelSamplingAV -> single-schedule Euler  "\n              f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "\n              f"dtype={x.dtype}", flush=True)\n        for i in trange(len(sigmas) - 1, disable=disable):\n            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])\n            denoised = model(x, sigmas[i] * s_in, **extra_args)\n            d = (x - denoised) / sigmas[i]\n            x = x + (sv_n - sv) * d\n            print(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  "\n                  f"denoised_rms={_rms(denoised):.4f} x_rms={_rms(x):.4f} d_rms={_rms(d):.4f}",\n                  flush=True)\n            if callback is not None:\n                callback({"i": i, "denoised": denoised, "x": x,\n                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})\n        return x\n\n    # Legacy ComfyUI without ModelSamplingAV: video and audio ride separate flow\n    # schedules (video shift 12, audio shift 3); step each on its own clock. A stock\n    # single-schedule sampler over-steps the audio at 4 steps and breaks it — that\n    # is the reason this node\'s sampler exists on older ComfyUI.\n    shapes = _latent_shapes(model)\n    if not shapes or len(shapes) < 2:\n        raise RuntimeError(\n            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "\n            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")\n    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]\n    a_numel = (x.shape[-1] - v_numel)\n    print(f"[H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  "\n          f"sigmas={[round(float(s),4) for s in sigmas]}  x.shape={tuple(x.shape)} "\n          f"dtype={x.dtype}  v_numel={v_numel} a_numel={a_numel}  shapes={shapes}", flush=True)\n    for i in trange(len(sigmas) - 1, disable=disable):   # tqdm it/s bar, like stock\n        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])\n        denoised = model(x, sigmas[i] * s_in, **extra_args)\n        out = (x - denoised) / sigmas[i]\n        xv, ov = x[..., :v_numel], out[..., :v_numel]\n        xa, oa = x[..., v_numel:], out[..., v_numel:]\n        xv = xv + (sv_n - sv) * ov               # video on its own sigma\n        sl = _audio_slope(max(sv, 1e-6))\n        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock\n        x = torch.cat([xv, xa], dim=-1)\n        print(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  denoised_rms={_rms(denoised):.4f}  "\n              f"video: x_rms={_rms(xv):.4f} v_rms={_rms(ov):.4f}  "\n              f"audio: x_rms={_rms(xa):.4f} v_rms={_rms(oa):.4f} slope={sl:.4f}", flush=True)\n        if callback is not None:\n            callback({"i": i, "denoised": denoised, "x": x,\n                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})\n    return x\n\n\n# --- pruned / curve-mode base support -------------------------------------\n# Pruned H3 checkpoints replace the time embedder + full-width adaln with a small\n# 8-dim curve (adaln_t_table), so the LoRA\'s adaln update (which lives in the\n# 2688-dim silu(t_emb) space) can\'t be applied as a weight patch. Instead we\n# re-inject it at run time: a shared silu(t_emb) is interpolated from a bundled\n# grid each forward, and each adaln projection adds B @ A @ silu(t_emb) to its\n# output. The backbone (attn/mlp/refiner) is patched normally.\n\n_EGRID = None\n\n\ndef _egrid():\n    global _EGRID\n    if _EGRID is None:\n        p = os.path.join(os.path.dirname(__file__), "h3_silu_temb_grid.safetensors")\n        _EGRID = comfy.utils.load_torch_file(p)["silu_t_emb_grid"]   # [1025, 2688]\n    return _EGRID\n\n\ndef _unique_t(timestep, shift_v, shift_a, payload):\n    """Mirror of the model\'s unique-timestep row computation (model.py forward):\n    the injected adaln delta carries one row per unique t, so the set built here\n    must dedup and sort to exactly the rows the model builds — same float32\n    tensor arithmetic (a float64 recompute can disagree on t_v == t_a collapse),\n    same conditioning rows: visual cond (keyframes / image refs) pins near 1,\n    and an audio ref adds its own max(t_a, aug) row."""\n    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)\n    t_v = float(1.0 - sigma_v)\n    t_a = float(1.0 - _time_shift_sigma(sigma_v, shift_v, shift_a))\n    s = {t_v, t_a}\n    refs = payload.get("refs") or ()\n    if payload.get("keyframes") or any(r.get("kind") == "image" for r in refs):\n        s.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))\n    if any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):\n        s.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))\n    return sorted(s)\n\n\ndef _interp_egrid(unique_t, E, device, dtype):\n    E = E.to(device)\n    n = E.shape[0]\n    rows = []\n    for t in unique_t:\n        pos = min(max(t, 0.0), 1.0) * (n - 1)\n        i0 = min(int(math.floor(pos)), n - 2)\n        rows.append(torch.lerp(E[i0].float(), E[i0 + 1].float(), pos - i0))\n    return torch.stack(rows).to(dtype)                               # [M, 2688]\n\n\ndef _make_adaln_forward(base, a, b, shared, table=None, egrid=None):\n    """Curve-mode adaln injection as a *forward-attribute* patch: returns a\n    replacement AdalnProj.forward that adds B @ A @ silu(t_emb) to the projection\n    before the reference view/chunk. Installed via add_object_patch on the\n    "<adaln_proj>.forward" attribute, so the module tree is left untouched.\n\n    Why not a wrapper module: replacing the whole AdalnProj with an nn.Module that\n    holds the original under .base injects a `.base` submodule (and its\n    `.base.linear.weight`) into the model\'s parameter/buffer tree. ComfyUI\'s\n    dynamic-VRAM streaming loader records every such path in its backup and, on\n    unload, restores it by that path via set_attr_param/set_attr_buffer — but the\n    object-patch has by then reverted AdalnProj to the plain module, so `.base`\n    no longer resolves and it crashes with\n    `AttributeError: \'AdalnProj\' object has no attribute \'base\'` (issue #4).\n    Patching only the .forward attribute keeps adaln_proj.linear.weight at its\n    natural path, so streaming backup/restore behaves exactly as unpatched.\n\n    a/b are held as plain captured tensors (never registered), so they never enter\n    the tree; they\'re cast to x\'s device/dtype per call, which also covers the\n    VRAM-offload case where the projection runs on GPU while a/b sit on CPU."""\n\n    def forward(t_emb):\n        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)\n\n        st = None\n        if table is not None and egrid is not None and not base.apply_silu:\n            try:\n                tb = table.to(t_emb.device, torch.float32)\n                idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)\n                st = egrid.to(t_emb.device)[idx]          # [M, 2688], M from the model\n            except Exception:\n                st = None\n        if st is None:\n            st = shared.get("silu_temb")                  # legacy _unique_t path\n\n        if st is not None and st.shape[0] == x.shape[0]:\n            av = a.to(x.device, x.dtype)\n            bv = b.to(x.device, x.dtype)\n            sv = st.to(x.device, x.dtype)\n            x = x + (bv @ (av @ sv.T)).T                              # [M, out]\n        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)\n        return x.chunk(base.expand, dim=-1)\n\n    return forward\n\n\nclass MiniMaxH3TurboSampler:\n    @classmethod\n    def INPUT_TYPES(cls):\n        return {"required": {}}\n\n    RETURN_TYPES = ("SAMPLER",)\n    FUNCTION = "get_sampler"\n    CATEGORY = "MiniMaxH3Turbo"\n    DESCRIPTION = ("4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "\n                   "SamplerCustomAdvanced and set the scheduler to 4 steps. "\n                   "Auto-adapts to the ComfyUI version: on recent builds that "\n                   "handle the audio schedule natively (ModelSamplingAV) it steps "\n                   "as a plain single-schedule sampler; on older builds it steps "\n                   "video and audio on their separate clocks.")\n\n    def get_sampler(self):\n        return (comfy.samplers.KSAMPLER(_turbo_sampler),)\n\n\nclass _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):\n    """LoRA bypass adapter with a memory-frugal additive path.\n\n    ComfyUI\'s default bypass is g(base_out + h(x)); for LoRA, h(x) allocates the\n    full-size projection twice (`out` and `out * scale`) and the outer add\n    allocates a third, so each bypassed layer holds ~3× its output activation\n    transiently. On the DiT\'s MLP down-projection (fc2, out = hidden, over a\n    ~46k-token sequence) that is ~1.5 GB of avoidable peak per block and is what\n    OOMs low-VRAM / pruned-fp8 runs that used to fit under the old merge path\n    (issue #4). Overriding bypass_forward to accumulate up(down(x))*scale straight\n    into base_out in place keeps one temporary instead of three. base_out is the\n    module\'s fresh output, so the in-place add is safe. Numerically identical to\n    the stock LoRA bypass. Linear-only (all H3 lora modules are Linear); anything\n    else falls back to the stock path."""\n\n    def bypass_forward(self, org_forward, x, *args, **kwargs):\n        base_out = org_forward(x, *args, **kwargs)\n        if getattr(self, "is_conv", False):\n            return super().bypass_forward(org_forward, x, *args, **kwargs)\n        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]\n        rank = down.shape[0]\n        scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)\n        down = down.to(dtype=x.dtype)\n        up = up.to(dtype=x.dtype)\n        return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)\n\n\ndef _apply_bypass_lora(new_model, lora, modules, strength):\n    """Apply the low-rank update at RUN TIME (output = base(x) + lora(x)) via\n    ComfyUI\'s bypass injection, so it is never folded into the weights. The delta\n    is tiny relative to the base weight — merging rounds it away in bf16 and\n    requantizes it away in int8/fp8 — whereas bypass runs the base\'s own\n    (possibly quantized) forward and adds the bf16 update in activation space,\n    exactly like the standalone generate.py reference. The stock\n    model_lora_keys_unet does not recognise the H3 lora naming, so build the key\n    map directly (module -> diffusion_model.<module>.weight). Adapters are wrapped\n    in _FrugalLoRA for the in-place additive path (see its docstring)."""\n    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}\n    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)\n    manager = comfy.weight_adapter.BypassInjectionManager()\n    sd_keys = set(new_model.model.state_dict().keys())\n    n = 0\n    for key, adapter in loaded.items():\n        if key not in sd_keys:\n            continue\n        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):\n            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)\n        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):\n            continue\n        manager.add_adapter(key, adapter, strength=strength)\n        n += 1\n    injections = manager.create_injections(new_model.model)\n    if manager.get_hook_count() > 0:\n        new_model.set_injections("bypass_lora", injections)\n    return n\n\n\ndef _apply_merge_lora(new_model, lora, modules, strength):\n    """Low-VRAM path: fold the low-rank update into the weights (add_patches, the\n    same call ComfyUI\'s own load_lora_for_models makes), so nothing extra is\n    computed at forward time. This is the cheapest on peak VRAM and lets small\n    GPUs run, but on a quantized base the delta is partly rounded away when it is\n    merged back into int8/fp8 (and on bf16 it sits near the ULP), i.e. softer than\n    the bypass path — the sharpness/VRAM trade the low_vram switch exposes."""\n    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}\n    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)\n    return len(new_model.add_patches(loaded, strength))\n\n\ndef _int8_fused_fc2(dm, modules):\n    """MLP fc2 modules whose base weight rides ComfyUI\'s fused int8 matmul.\n\n    comfy.ops.linear_input_act (minimax MLP.forward) folds the swiglu activation\n    into an INT8 activation quantizer and calls the fused int8 kernel on\n    linear.weight DIRECTLY — it never calls the module\'s forward, so a\n    BypassForwardHook installed on fc2.forward never fires and that fc2\'s LoRA is\n    silently dropped (measured: on int8_convrot the 50 DiT-block fc2 hooks fire 0\n    times). Those fc2 must instead go through the merge/weight-function path, where\n    ComfyUI dequantizes the int8 weight and applies the LoRA during the weight cast\n    (delta preserved in fp32; ~one fc2 weight dequantized transiently per call, no\n    resident cost). fc2 on bf16 / fp8 bases is left on bypass — there the fused int8\n    path isn\'t taken (the eager `linear(swiglu(x))` fallback runs) so the hook fires\n    normally."""\n    fused = []\n    for m in modules:\n        if not m.endswith(".mlp.fc2"):\n            continue\n        try:\n            w = comfy.utils.get_attr(dm, m + ".weight")\n        except Exception:\n            continue\n        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"\n                and not getattr(getattr(w, "_params", None), "transposed", False)):\n            fused.append(m)\n    return fused\n\n\ndef _inject_adaln_egrid(new_model, dm, lora, adaln, strength):\n    """Pruned/curve base only: the adaln update lives in the 2688-dim silu(t_emb)\n    space, which the pruned base has collapsed into a small curve, so it can be\n    neither a bypass adapter nor a merged weight patch. Re-inject it at run time —\n    a shared silu(t_emb) interpolated from the bundled E-grid each forward, plus a\n    forward-attribute patch on each adaln projection that adds B @ A @ silu(t_emb)\n    (see _make_adaln_forward). Peak memory is negligible (M <= 3 rows), so this is\n    identical in both the bypass and low_vram modes."""\n    E = _egrid()\n    shared = {"silu_temb": None}\n    shift_v = float(getattr(dm, "sigma_shift_video", SHIFT_V))\n    shift_a = float(getattr(dm, "sigma_shift_audio", SHIFT_A))\n\n    tt = None\n    for _n, _t in list(dm.named_buffers()) + list(dm.named_parameters()):\n        if _n.endswith("adaln_t_table"):\n            tt = _t\n            break\n    if tt is not None and tt.shape[0] != E.shape[0]:\n        tt = None\n\n    def wrap(executor, *args, **kwargs):\n        ts = args[1] if len(args) > 1 else kwargs.get("timestep")\n        ctx = args[2] if len(args) > 2 else kwargs.get("context")\n        payload = kwargs.get("minimax_payload") or {}\n        us = _unique_t(ts, shift_v, shift_a, payload)\n        shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)\n        return executor(*args, **kwargs)\n\n    new_model.add_wrapper_with_key(\n        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)\n    for name in adaln:                       # name = "....adaln_proj.linear"\n        a = lora[name + ".lora_A.weight"]\n        b = lora[name + ".lora_B.weight"] * strength\n        key = "diffusion_model." + name.rsplit(".linear", 1)[0]\n        new_model.add_object_patch(\n            key + ".forward",\n            _make_adaln_forward(new_model.get_model_object(key), a, b, shared, tt, E))\n\n\ndef _add_dbg_wrapper(new_model, dm, tag, mode):\n    """Observability: at diffusion-model forward time, log that the lora is\n    actually active this forward, plus the timestep and video/audio input rms.\n    Only the first few calls are printed to avoid flooding.\n\n    The activity canary depends on `mode`. In bypass mode a lora\'d module\'s\n    forward is taken over by BypassForwardHook, so qkv_proj.forward_owner reads\n    `BypassForwardHook` iff the lora is live. In merge mode the delta is folded\n    into the weights and the forward stays the base Linear, so the owner is\n    expected to be the base module — activity is instead reflected by the layer\n    carrying a patch (weight_function), which we report separately."""\n    st = {"n": 0}\n\n    def wrap(executor, *args, **kwargs):\n        if st["n"] < 6:\n            st["n"] += 1\n            try:\n                m0 = dm.blocks[0].attn.qkv_proj\n                owner = type(getattr(m0.forward, "__self__", None)).__name__\n                has_wf = bool(getattr(m0, "weight_function", None)) or \\\n                    getattr(m0, "weight_lowvram_function", None) is not None\n            except Exception as e:                       # noqa\n                owner, has_wf = "err:%s" % e, "?"\n            ts = args[1] if len(args) > 1 else kwargs.get("timestep")\n            xx = args[0] if args else kwargs.get("x")\n            try:\n                vr = float(xx[0].float().pow(2).mean().sqrt())\n                ar = float(xx[1].float().pow(2).mean().sqrt())\n                dt = str(xx[0].dtype)\n            except Exception:\n                vr = ar = -1.0\n                dt = "?"\n            tsv = float(ts.flatten()[0]) if ts is not None else -1\n            if mode == "merge":\n                canary = (f"qkv_proj.forward_owner={owner} weight_patched={has_wf} "\n                          f"(merge: delta folded into weights; owner is the base "\n                          f"Linear, patch presence => lora ACTIVE)")\n            else:\n                canary = (f"qkv_proj.forward_owner={owner} "\n                          f"(BypassForwardHook => lora ACTIVE; else => BASE ONLY!)")\n            print(f"[H3TURBO fwd {tag}/{mode}] call#{st[\'n\']}  {canary}  is_injected="\n                  f"{getattr(new_model, \'is_injected\', \'?\')}  timestep={tsv:.2f}  "\n                  f"video_rms={vr:.4f} audio_rms={ar:.4f} dtype={dt}", flush=True)\n        return executor(*args, **kwargs)\n\n    new_model.add_wrapper_with_key(\n        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo_dbg", wrap)\n\n\nclass MiniMaxH3TurboLoRA:\n    @classmethod\n    def INPUT_TYPES(cls):\n        return {"required": {\n            "model": ("MODEL",),\n            "lora_name": (folder_paths.get_filename_list("loras"),),\n            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,\n                                   "step": 0.01}),\n            "low_vram": ("BOOLEAN", {\n                "default": True,\n                "label_on": "merge (low VRAM, softer)",\n                "label_off": "bypass (sharp, more VRAM)",\n                "tooltip": "OFF (default): apply the LoRA at run time (bypass) — "\n                           "sharpest, but costs extra peak VRAM. ON: merge the "\n                           "LoRA into the weights — lowest VRAM so small GPUs can "\n                           "run, but softer on quantized bases (the delta is "\n                           "partly rounded away). Turn ON only if you OOM."}),\n        }}\n\n    RETURN_TYPES = ("MODEL",)\n    FUNCTION = "apply_lora"\n    CATEGORY = "MiniMaxH3Turbo"\n    DESCRIPTION = "Apply the MiniMax-H3 Turbo LoRA to the H3 diffusion model."\n\n    def apply_lora(self, model, lora_name, strength, low_vram=False):\n        path = folder_paths.get_full_path("loras", lora_name)\n        lora = comfy.utils.load_torch_file(path, safe_load=True)\n        dm = model.model.diffusion_model\n        pruned = getattr(dm, "use_adaln_curves", False)\n        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})\n        new_model = model.clone()\n        mode = "merge" if low_vram else "bypass"\n\n        # On the pruned base the adaln update can\'t be a weight patch (it lives in\n        # the collapsed silu(t_emb) curve), so it is always re-injected at run\n        # time regardless of mode; everything else is the "backbone", which takes\n        # the bypass or merge path per low_vram.\n        if pruned:\n            backbone = [m for m in modules if "adaln_proj" not in m]\n            adaln = [m for m in modules if "adaln_proj" in m]\n        else:\n            backbone, adaln = modules, []\n\n        n_fc2 = 0\n        if low_vram:\n            n = _apply_merge_lora(new_model, lora, backbone, strength)\n        else:\n            # int8-fused fc2 is invisible to the bypass hook — apply those via merge,\n            # the rest via bypass (see _int8_fused_fc2).\n            fc2_fused = set(_int8_fused_fc2(dm, backbone))\n            bypass_mods = [m for m in backbone if m not in fc2_fused]\n            n = _apply_bypass_lora(new_model, lora, bypass_mods, strength)\n            if fc2_fused:\n                n_fc2 = _apply_merge_lora(new_model, lora, sorted(fc2_fused), strength)\n                n += n_fc2\n        if pruned and adaln:\n            _inject_adaln_egrid(new_model, dm, lora, adaln, strength)\n\n        try:\n            p0 = dm.blocks[0].attn.qkv_proj.weight\n            wdt, wdev = str(p0.dtype), str(p0.device)\n        except Exception:\n            wdt, wdev = "?", "?"\n        if low_vram:\n            detail = f"{n} weights patched (merged)"\n        else:\n            injs = new_model.injections.get("bypass_lora", [])\n            detail = f"{n - n_fc2} bypass adapters, {len(injs)} injections"\n            if n_fc2:\n                detail += f", {n_fc2} int8 fc2 via merge"\n        extra = f" + {len(adaln)} adaln injected at run time" if adaln else ""\n        print(f"[MiniMaxH3TurboLoRA] {\'pruned\' if pruned else \'full\'} base [{mode}]: "\n              f"lora={lora_name} strength={strength} | {len(backbone)} backbone "\n              f"modules, {detail}{extra} | model={type(new_model.model).__name__} "\n              f"weight_dtype={wdt} weight_dev={wdev}", flush=True)\n        _add_dbg_wrapper(new_model, dm, "pruned" if pruned else "full", mode)\n        return (new_model,)\n\n\nNODE_CLASS_MAPPINGS = {\n    "MiniMaxH3TurboLoRA": MiniMaxH3TurboLoRA,\n    "MiniMaxH3TurboSampler": MiniMaxH3TurboSampler,\n}\nNODE_DISPLAY_NAME_MAPPINGS = {\n    "MiniMaxH3TurboLoRA": "MiniMax-H3 Turbo LoRA",\n    "MiniMaxH3TurboSampler": "MiniMax-H3 Turbo Sampler (4-step)",\n}\n',
    }

    for relative, content in patches.items():
        target = COMFY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_suffix(target.suffix + ".h3_pre_overlay")
        if target.exists() and not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(content, encoding="utf-8")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"H3 runtime overlay verification failed: {target}")
        print(f"[H3 COMFY PATCH] {relative} sha256={actual[:16]}")

    print("[H3 COMFY PATCH] embedded runtime overlay applied: 4 files")

def run(
    *args,
    env=None,
) -> None:

    print(
        "+",
        " ".join(
            str(value)
            for value in args
        ),
    )

    subprocess.run(
        [
            str(value)
            for value in args
        ],
        check=True,
        env=env,
    )


def load_yaml(
    path: Path,
) -> dict:

    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Invalid YAML mapping: {path}"
        )

    return value


def find_kaggle_file(
    filename: str,
) -> Path:

    matches = []

    for path in KAGGLE_INPUT.rglob(
        "*"
    ):

        if (
            path.is_file()
            and path.name.lower()
            == filename.lower()
        ):
            matches.append(
                path
            )

    if not matches:

        raise FileNotFoundError(
            "Required Kaggle asset not found: "
            f"{filename}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple copies found for {filename}:\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    return matches[0]


def link_model(
    source: Path,
    destination: Path,
) -> None:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        or destination.is_symlink()
    ):
        destination.unlink()

    try:
        destination.symlink_to(
            source
        )
    except OSError:
        shutil.copy2(
            source,
            destination,
        )


def _site_packages() -> list[Path]:

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import site; "
                "print('\\n'.join(site.getsitepackages()))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return [
        Path(
            line.strip()
        )
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _cuda_library_dirs() -> list[Path]:

    directories = []

    for site_root in _site_packages():

        nvidia_root = (
            site_root
            / "nvidia"
        )

        if not nvidia_root.is_dir():
            continue

        for pattern in (
            "libcudart.so.13*",
            "libcublas.so.13*",
        ):

            for library in nvidia_root.rglob(
                pattern
            ):

                if not library.is_file():
                    continue

                directory = (
                    library.parent
                )

                if directory not in directories:
                    directories.append(
                        directory
                    )

    return directories


def _configure_cuda_environment(
    directories: list[Path],
) -> dict[str, str]:

    environment = dict(
        os.environ
    )

    existing = environment.get(
        "LD_LIBRARY_PATH",
        "",
    )

    values = [
        str(path)
        for path in directories
    ]

    if existing:
        values.append(
            existing
        )

    environment[
        "LD_LIBRARY_PATH"
    ] = ":".join(
        values
    )

    return environment


def install_base_requirements() -> None:
    requirements = ROOT / "requirements.txt"

    if not requirements.is_file():
        raise RuntimeError(
            f"Repository dependency manifest is missing: {requirements}"
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "-r",
        requirements,
    )



def install_pytorch_runtime(runtime: dict) -> None:
    """Install and verify the project-locked PyTorch CUDA build last.

    ComfyUI and custom-node requirements are allowed to install their own
    compatible dependencies first. PyTorch is re-asserted only after all of
    those dependency installs so a transitive requirement cannot silently
    leave the worker on a different CUDA build.
    """
    config = dict(runtime.get("pytorch", {}) or {})
    version = str(config.get("version", "") or "").strip()
    cuda = str(config.get("cuda", "") or "").strip().lower()
    index = str(config.get("index", "") or "").strip()
    torchvision_version = str(config.get("torchvision_version", "") or "").strip()
    torchaudio_version = str(config.get("torchaudio_version", "") or "").strip()

    if not all((version, cuda, index, torchvision_version, torchaudio_version)):
        raise RuntimeError("runtime_versions.yaml pytorch configuration is incomplete.")
    if cuda != "cu130":
        raise RuntimeError(f"This Ref2VA project is locked to cu130, got {cuda!r}.")

    print("=" * 80)
    print("INSTALLING LOCKED PYTORCH RUNTIME")
    print("=" * 80)

    run(
        sys.executable,
        "-m", "pip", "install", "-q",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--force-reinstall",
        "--index-url", index,
        f"torch=={version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
    )

    verify = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import torch; "
                f"assert torch.__version__ == '2.10.0+{cuda}'; "
                "assert torch.version.cuda == '13.0'; "
                "print(torch.__version__); "
                "print(torch.version.cuda)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise RuntimeError(
            "Locked PyTorch runtime verification failed.\n"
            + (verify.stdout or "")
            + (verify.stderr or "")
        )
    print("[PYTORCH]", (verify.stdout or "").strip().replace("\n", " | "))


def patch_t4_h3_value_clone(runtime: dict) -> None:
    """Apply the narrowly-scoped T4 H3 v-clone workaround to the locked ComfyUI.

    ComfyUI 0.34.0's MiniMax H3 Attention copies the large V tensor before
    wrapping it in AttentionTensorContainer. On 16-GB-class GPUs that can
    add about a gigabyte of peak memory and severely degrade throughput.
    The workaround is applied only when the exact upstream 0.34.0 source
    pattern is present and only on SM75 GPUs. If the source changes, fail
    loudly instead of silently patching the wrong code.
    """
    enabled = bool(
        runtime.get("comfyui", {}).get("h3_t4_value_clone_workaround", True)
    )
    if not enabled:
        print("[H3 T4 PATCH] disabled by runtime configuration")
        return

    try:
        import torch
        if not torch.cuda.is_available():
            print("[H3 T4 PATCH] skipped: CUDA unavailable")
            return
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (7, 5):
            print(f"[H3 T4 PATCH] skipped: GPU SM{major}{minor} is not SM75")
            return
    except Exception as exc:
        raise RuntimeError(f"Cannot determine GPU capability for H3 T4 patch: {exc}") from exc

    target = COMFY / "comfy" / "ldm" / "minimax" / "model.py"
    if not target.is_file():
        raise RuntimeError(f"H3 model source not found: {target}")

    text = target.read_text(encoding="utf-8")
    marker = "# H3-T4-WORKAROUND: removed redundant V clone for SM75"
    if marker in text:
        print("[H3 T4 PATCH] already applied")
        return

    exact = "        v = v.clone()\n        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))"
    replacement = "        " + marker + "\n        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))"
    if exact not in text:
        raise RuntimeError(
            "Refusing to apply the H3 T4 workaround because ComfyUI's expected "
            "0.34.0 Attention pattern was not found."
        )
    target.write_text(text.replace(exact, replacement, 1), encoding="utf-8")
    print("[H3 T4 PATCH] applied to", target)

def patch_h3_vae_decoder_dtype(runtime: dict) -> None:
    """Keep MiniMax H3 video VAE decoder input on the decoder's dtype.

    This is deliberately scoped to the locked H3 VAE implementation. It is
    idempotent and fails closed if the expected 0.34.0 source pattern changes.
    """
    enabled = bool(runtime.get("comfyui", {}).get("h3_vae_decoder_dtype_patch", True))
    if not enabled:
        print("[H3 VAE PATCH] disabled by runtime configuration")
        return

    target = COMFY / "comfy" / "ldm" / "minimax" / "vae.py"
    if not target.is_file():
        raise RuntimeError(f"H3 VAE source not found: {target}")

    text = target.read_text(encoding="utf-8")
    marker = "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters"
    existing_patch = (
        "        z = self.post_quant_conv(z)\n"
        "        decoder_dtype = next(self.decoder.parameters()).dtype\n"
        "        if z.dtype != decoder_dtype:\n"
        "            z = z.to(decoder_dtype)\n"
        "        return self.decoder(z)"
    )
    if marker in text or existing_patch in text:
        print("[H3 VAE PATCH] already applied")
        return

    exact = "        z = self.post_quant_conv(z)\n        return self.decoder(z)"
    replacement = (
        "        z = self.post_quant_conv(z)\n"
        f"        {marker}\n"
        "        decoder_dtype = next(self.decoder.parameters()).dtype\n"
        "        if z.dtype != decoder_dtype:\n"
        "            z = z.to(decoder_dtype)\n"
        "        return self.decoder(z)"
    )
    if exact not in text:
        raise RuntimeError(
            "Refusing to apply the H3 VAE dtype patch because the expected "
            "locked ComfyUI 0.34.0 source pattern was not found."
        )
    target.write_text(text.replace(exact, replacement, 1), encoding="utf-8")
    print("[H3 VAE PATCH] applied to", target)


def install_director_runtime(
    runtime: dict,
) -> None:

    llama_config = runtime[
        "llama_cpp"
    ]

    cuda_config = runtime[
        "cuda_runtime"
    ]

    llama_package = llama_config[
        "package"
    ]

    llama_version = llama_config[
        "version"
    ]

    cuda_index = llama_config[
        "cuda_index"
    ]

    cuda_runtime_package = cuda_config[
        "runtime_package"
    ]

    cuda_runtime_version = cuda_config[
        "runtime_version"
    ]

    cublas_package = cuda_config[
        "cublas_package"
    ]

    cublas_version = cuda_config[
        "cublas_version"
    ]

    print(
        "=" * 80
    )

    print(
        "INSTALLING QWEN DIRECTOR RUNTIME"
    )

    print(
        "=" * 80
    )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        f"{cuda_runtime_package}=={cuda_runtime_version}",
        f"{cublas_package}=={cublas_version}",
    )

    library_dirs = (
        _cuda_library_dirs()
    )

    if not library_dirs:

        raise RuntimeError(
            "NVIDIA CUDA runtime packages installed, "
            "but no native CUDA library directories were found."
        )

    has_cudart = any(
        any(
            path.name.startswith(
                "libcudart.so.13"
            )
            for path in directory.iterdir()
            if path.is_file()
        )
        for directory in library_dirs
    )

    has_cublas = any(
        any(
            path.name.startswith(
                "libcublas.so.13"
            )
            for path in directory.iterdir()
            if path.is_file()
        )
        for directory in library_dirs
    )

    if not has_cudart:

        raise RuntimeError(
            "libcudart.so.13 was not found."
        )

    if not has_cublas:

        raise RuntimeError(
            "libcublas.so.13 was not found."
        )

    environment = (
        _configure_cuda_environment(
            library_dirs
        )
    )

    print(
        "[CUDA RUNTIME LIBRARIES]"
    )

    for directory in library_dirs:

        print(
            " ",
            directory,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--only-binary=:all:",
        "--disable-pip-version-check",
        f"{llama_package}=={llama_version}",
        "--extra-index-url",
        cuda_index,
    )

    verification = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from llama_cpp import Llama; "
                "print('llama-cpp-python CUDA import: PASS')"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if verification.stdout:
        print(
            verification.stdout
        )

    if verification.stderr:
        print(
            verification.stderr
        )

    if verification.returncode != 0:

        raise RuntimeError(
            "llama-cpp-python CUDA import failed."
        )




def install_comfyui(runtime: dict) -> None:
    """Install the exact ComfyUI revision required by the project."""
    config = runtime.get("comfyui", {})
    repository = str(config.get("repository", "") or "").strip()
    revision = str(config.get("revision", "") or "").strip()
    if not repository or not revision:
        raise RuntimeError("runtime_versions.yaml must define comfyui.repository and comfyui.revision.")

    print("=" * 80)
    print("INSTALLING COMFYUI")
    print("=" * 80)

    if COMFY.exists() and not (COMFY / ".git").exists():
        raise RuntimeError(
            f"ComfyUI path exists but is not a git checkout: {COMFY}. "
            "Move it away and rerun bootstrap."
        )

    if not COMFY.exists():
        run("git", "clone", repository, COMFY)

    run("git", "-C", COMFY, "fetch", "--tags", "--prune", "origin")
    run("git", "-C", COMFY, "checkout", "--detach", revision)

    requirements = COMFY / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError(f"ComfyUI requirements.txt is missing: {requirements}")

    run(
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "-r", requirements,
    )

    expected_version = str(config.get("expected_version", "") or "").strip()
    version_check = subprocess.run(
        [sys.executable, "-c", "import importlib.metadata as m; print(m.version('comfyui'))"],
        cwd=str(COMFY),
        capture_output=True,
        text=True,
        check=False,
    )
    installed_version = (version_check.stdout or "").strip()
    # The git checkout is the authority. If a package distribution is not
    # installed under the same name, verify the git revision directly below.
    rev = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_rev = subprocess.run(
        ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if rev != expected_rev:
        raise RuntimeError(
            f"ComfyUI revision mismatch: checked out {rev}, expected {expected_rev}."
        )
    print(f"[COMFYUI] revision={rev}")
    if expected_version:
        print(f"[COMFYUI] expected release={expected_version}")
    if installed_version:
        print(f"[COMFYUI] package version={installed_version}")

def install_storyboard_runtime(
    runtime: dict,
) -> None:

    storyboard = runtime["storyboard"]
    packages = []

    gradio_version = str(
        storyboard.get("gradio_version", "")
    ).strip()
    pillow_version = str(
        storyboard.get("pillow_version", "")
    ).strip()

    if gradio_version:
        packages.append(
            f"gradio=={gradio_version}"
        )
    if pillow_version:
        packages.append(
            f"Pillow=={pillow_version}"
        )

    if not packages:
        return

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        *packages,
    )


def install_nodes() -> None:

    manifest = load_yaml(
        NODE_MANIFEST
    )

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = (
        manifest[
            "custom_nodes"
        ][
            "required"
        ],
        manifest[
            "custom_nodes"
        ][
            "supporting"
        ],
    )

    for group in groups:

        for node in group:

            destination = (
                CUSTOM
                / node[
                    "name"
                ]
            )

            if not destination.exists():

                run(
                    "git",
                    "clone",
                    node[
                        "repository"
                    ],
                    destination,
                )

            run(
                "git",
                "-C",
                destination,
                "fetch",
                "--all",
                "--tags",
                "--prune",
            )

            run(
                "git",
                "-C",
                destination,
                "checkout",
                "--detach",
                node[
                    "revision"
                ],
            )

            requirements = destination / "requirements.txt"

            if requirements.is_file():
                run(
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--disable-pip-version-check",
                    "-r",
                    requirements,
                )

                print(
                    "[NODE DEPS]",
                    node[
                        "name"
                    ],
                )

            print(
                "[NODE]",
                node[
                    "name"
                ],
            )


def install_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        filename = model[
            "filename"
        ]

        source = find_kaggle_file(
            filename
        )

        destination = (
            MODELS
            / model[
                "directory"
            ]
            / filename
        )

        link_model(
            source,
            destination,
        )

        print(
            "[MODEL]",
            filename,
        )


def verify_inventory() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    expected = {
        (
            model["directory"],
            model["filename"].lower(),
        )
        for model in manifest["models"].values()
    }

    # These files are created by ComfyUI as empty placeholder markers.
    # They are not model assets and must not be treated as unexpected
    # production models.
    placeholder_names = {
        "put_diffusion_model_files_here",
        "put_latent_upscale_models_here",
        "put_loras_here",
        "put_text_encoder_files_here",
        "put_vae_here",
    }

    actual = set()

    production_directories = {
        "diffusion_models",
        "text_encoders",
        "loras",
        "vae",
        "latent_upscale_models",
    }

    for directory_name in production_directories:

        directory = (
            MODELS
            / directory_name
        )

        if not directory.is_dir():
            continue

        for item in directory.iterdir():

            if not item.is_file():
                continue

            # Ignore only ComfyUI's known empty placeholder markers.
            if (
                item.name.lower()
                in placeholder_names
            ):
                continue

            actual.add(
                (
                    directory_name,
                    item.name.lower(),
                )
            )

    missing = (
        expected
        - actual
    )

    unexpected = (
        actual
        - expected
    )

    if missing:

        raise RuntimeError(
            "Missing H3 models:\n"
            + "\n".join(
                f"{directory}/{filename}"
                for directory, filename
                in sorted(
                    missing
                )
            )
        )

    if unexpected:

        raise RuntimeError(
            "Unexpected H3 production models:\n"
            + "\n".join(
                f"{directory}/{filename}"
                for directory, filename
                in sorted(
                    unexpected
                )
            )
        )



def verify_runtime_files(runtime: dict) -> None:
    if not (COMFY / "main.py").is_file():
        raise RuntimeError(f"ComfyUI main.py is missing: {COMFY / 'main.py'}")
    if not CUSTOM.is_dir():
        raise RuntimeError(f"ComfyUI custom_nodes directory is missing: {CUSTOM}")
    if not MODELS.is_dir():
        raise RuntimeError(f"ComfyUI models directory is missing: {MODELS}")

    expected_version = str(runtime.get("comfyui", {}).get("expected_version", "") or "").strip()
    revision = str(runtime.get("comfyui", {}).get("revision", "") or "").strip()
    head = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    tagged = subprocess.run(
        ["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if revision:
        expected_head = subprocess.run(
            ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if head != expected_head:
            raise RuntimeError("ComfyUI checkout is not at the locked revision.")
    if expected_version and tagged not in {expected_version, f"v{expected_version}"}:
        raise RuntimeError(
            f"ComfyUI checkout is not the expected release tag. "
            f"HEAD={tagged or 'untagged'}, expected={expected_version}."
        )


def main():

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    director_filename = (
        runtime[
            "director"
        ][
            "model_filename"
        ]
    )

    director_model = (
        find_kaggle_file(
            director_filename
        )
    )

    print(
        "[DIRECTOR MODEL]",
        director_model,
    )

    install_base_requirements()

    install_comfyui(runtime)

    install_director_runtime(
        runtime
    )

    install_storyboard_runtime(
        runtime
    )

    install_nodes()

    # Re-assert the locked PyTorch CUDA build after every package that can
    # mutate the Python runtime has been installed.
    install_pytorch_runtime(runtime)

    # ComfyUI is a runtime dependency, not repository content.
    # Apply the project-owned H3 overrides only after the locked upstream
    # ComfyUI checkout and custom nodes are installed.
    apply_embedded_h3_runtime_overlay()

    install_models()

    verify_inventory()
    verify_runtime_files(runtime)

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 Kaggle bootstrap PASSED."
    )


if __name__ == "__main__":
    main()
