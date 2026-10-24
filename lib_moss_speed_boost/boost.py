# MOSS-Image Speed Boost for Forge Neo
# Runs the MOSS-Image DiT blocks of the moss-image-forge-neo extension on comfy_kitchen's fused kernels, the way
# ComfyUI's native MOSS-Image model does (comfy/ldm/moss_image/model.py):
#   - RMSNorm + adaLN modulation in one kernel (ck.rms_adaln, the norm weight folded into the scale);
#   - per-head q/k RMSNorm + split-half RoPE in one kernel (ck.rms_rope_split_half);
#   - INT8 attention (ck.int8_attention) instead of bf16 SDPA (optional);
#   - Fused SiTU: SiTU folded into ffn_down's INT8 activation quantization (optional, comfy_kitchen >= 0.2.39);
#   - the gated residuals with fewer temporaries.
# Only Block.forward and rope_table of forge_moss.native.transformer are replaced; everything else (the text blocks,
# the block-22 skip of the negative pass, LoRA, offload, the INT8 ConvRot linears) stays the extension's own code.
# Every patched function hands over to the original when the boost is off or a kernel is unavailable.

import torch

TAG = "[MOSS-Image Speed Boost]"

# "enabled", "int8_attention" and "fused_situ" are set per generation by the script (configure)
_state = {"applied": False, "ck": None, "int8_attn": None, "situ": False, "pe": None, "pe_text": None,
          "enabled": True, "int8_attention": True, "fused_situ": True}
_orig = {}


def configure(enabled: bool, int8_attention: bool, fused_situ: bool) -> None:
    _state.update(enabled=bool(enabled), int8_attention=bool(int8_attention), fused_situ=bool(fused_situ))


def enabled() -> bool:
    return _state["ck"] is not None and _state["enabled"]


def fused_situ_available() -> bool:
    return _state["situ"]


def fused_situ() -> bool:
    return _state["situ"] and _state["fused_situ"]


def int8_attention_available(device=None) -> bool:
    if _state["int8_attn"] is None:
        try:
            _state["int8_attn"] = bool(_state["ck"].int8_attention_is_available(device))
        except Exception:
            _state["int8_attn"] = False
    return _state["int8_attn"]


def _use_int8_attention(device) -> bool:
    return _state["int8_attention"] and int8_attention_available(device)


# ---- RoPE: the extension's own phases, kept in float32, in comfy_kitchen's freqs_cis layout ----

def _rope_table(height, width, text_length, head_dim, base, device):
    table = _orig["rope_table"](height, width, text_length, head_dim, base, device)
    if _state["ck"] is not None:
        # refreshed even while the boost is off, so a table never outlives the size it was built for
        # table (L, head_dim) = sin then cos of the 4x(head_dim/8) phases, float32; freqs_cis wants, per frequency,
        # the 2x2 rotation [[cos, -sin], [sin, cos]] applied to the pair (x[i], x[i + head_dim/2]) - the split-half
        # pairs of the extension's rotate()
        sin, cos = table.float().chunk(2, -1)
        pe = torch.stack((cos, -sin, sin, cos), -1).unflatten(-1, (2, 2))
        _state["pe"] = pe[None, :, None].contiguous()  # (1, L, 1, head_dim/2, 2, 2)
        _state["pe_text"] = None
    return table


def _freqs_for(rope):
    pe = _state["pe"]
    if pe is None or pe.device != rope.device:
        return None
    length = rope.shape[0]
    if pe.shape[1] == length:
        return pe
    if length < pe.shape[1]:  # the text slice rope[image_tokens:] is the tail
        text = _state["pe_text"]
        if text is None or text.shape[1] != length:
            text = pe[:, -length:].contiguous()
            _state["pe_text"] = text
        return text
    return None


def _weight(module, dtype, device):
    from forge_moss.native import kernels

    return kernels.cast_to(module.weight, dtype, device)


def _modulated_norm(norm, x, scale, shift):
    # rmsnorm(x) * w * (1 + scale) + shift == rms_adaln(x, w * (1 + scale) - 1, shift); folded in float32
    w = _weight(norm, torch.float32, x.device)
    folded = (w * (1 + scale.float()) - 1).to(x.dtype)
    return _state["ck"].rms_adaln(x, folded, shift.to(x.dtype), 1e-6)


def _attention(q, k, v, q_norm, k_norm, pe, heads, transformer_options):
    """q, k, v (B, L, H*D) -> (B, L, H*D): fused head norm + RoPE, then INT8 or Forge attention."""
    from forge_moss.native import kernels

    b, n, _ = q.shape
    q, k, v = q.view(b, n, heads, -1), k.view(b, n, heads, -1), v.view(b, n, heads, -1)
    q, k = _state["ck"].rms_rope_split_half(q, k, pe, _weight(q_norm, q.dtype, q.device),
                                            _weight(k_norm, k.dtype, k.device), 1e-6)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if _use_int8_attention(q.device):
        out = _state["ck"].int8_attention(q, k.to(q.dtype), v.to(q.dtype))
        return out.transpose(1, 2).reshape(b, n, -1)
    return kernels.attention(q, k.to(v.dtype), v, transformer_options=transformer_options)


def _situ(hidden, gate):
    # (4 tanh(g/4) sigmoid(g)) * (25 tanh(h/25)) with one output buffer
    out = torch.sigmoid(gate)
    out.mul_(torch.tanh(gate * 0.25))
    out.mul_(torch.tanh(hidden * 0.04))
    return out.mul_(100.0)


def _situ_halves(x):
    return _situ(*x.chunk(2, -1))


def _ffn_down(linear, hidden_gate):
    """ffn_down(situ(hidden, gate)) for the [hidden | gate] halves of ffn_up_gate's output.

    comfy_kitchen >= 0.2.39 folds SiTU into the INT8 ConvRot activation quantization of ffn_down (what ComfyUI does
    through comfy.ops.linear_input_act), so the 2x8192-wide intermediate never goes through eager elementwise ops.
    Forge's own linear_input_act picks the fused kernel only for an INT8 weight and computes the activation eagerly
    otherwise (online LoRA, fp16 weights), so every path gives the same result."""
    if fused_situ():
        from backend.operations_mixed_precision import linear_input_act

        return linear_input_act(linear, hidden_gate, "situ")
    return linear(_situ_halves(hidden_gate))


def _block_forward(self, x, rope, mod, dtype, transformer_options=None):
    pe = _freqs_for(rope) if enabled() and not _state.get("failed") else None
    if pe is None or x.dtype != dtype:
        # boost off, no table, or the float32 residual stream of fp16 GPUs: the extension's own path
        return _orig["block_forward"](self, x, rope, mod, dtype, transformer_options)
    try:
        return _fused_block(self, x, rope, mod, dtype, pe, transformer_options)
    except Exception as e:
        # a kernel or the extension changed under us: never break generation, fall back for the rest of the session
        _state["failed"] = True
        print(f"{TAG} fused path failed ({type(e).__name__}: {e}); using the extension's own path")
        return _orig["block_forward"](self, x, rope, mod, dtype, transformer_options)


def _fused_block(self, x, rope, mod, dtype, pe, transformer_options):

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
    y = _modulated_norm(self.norm1, x, scale_a, shift_a)
    if self.fused_qkv:
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
    else:
        q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
    out = self.proj["o"](_attention(q, k, v, self.q_norm, self.k_norm, pe, self.heads, transformer_options).to(dtype))
    # out of place: the caller keeps the input tokens for the samples whose block is skipped (negative pass)
    x = torch.addcmul(x, out, gate_a.to(x.dtype))
    y = _modulated_norm(self.norm2, x, scale_f, shift_f)
    # x is this block's own tensor now, so the second residual can be in place
    return x.addcmul_(_ffn_down(self.proj["ffn_down"], self.proj["ffn_up_gate"](y)), gate_f.to(x.dtype))


def _situ_supported() -> bool:
    """SiTU as an int8_linear input activation arrived in comfy_kitchen 0.2.39 (Forge 2.30 ships 0.2.37)."""
    try:
        import re
        from importlib.metadata import version

        if tuple(int(n) for n in re.findall(r"\d+", version("comfy-kitchen"))[:3]) < (0, 2, 39):
            return False
        from backend import operations_mixed_precision as ops

        # the eager fallback of Forge's linear_input_act (non-INT8 weights) needs to know the activation
        ops.INPUT_ACT_EAGER.setdefault("situ", _situ_halves)
        return hasattr(ops, "linear_input_act")
    except Exception:
        return False


def apply() -> bool:
    if _state["applied"]:
        return True
    try:
        import comfy_kitchen as ck

        for name in ("rms_adaln", "rms_rope_split_half", "int8_attention", "int8_attention_is_available"):
            if not hasattr(ck, name):
                raise ImportError(f"comfy_kitchen has no {name}")
    except Exception as e:
        print(f"{TAG} comfy_kitchen is missing what it needs ({e}); MOSS-Image keeps its own path")
        return False

    import inspect

    from forge_moss.native import transformer as T

    expected = ["self", "x", "rope", "mod", "dtype", "transformer_options"]
    if list(inspect.signature(T.Block.forward).parameters) != expected or not hasattr(T, "rope_table"):
        print(f"{TAG} this moss-image-forge-neo version has a different DiT block; boost disabled")
        return False

    _orig["rope_table"] = T.rope_table
    _orig["block_forward"] = T.Block.forward
    T.rope_table = _rope_table
    T.Block.forward = _block_forward
    _state["ck"] = ck
    _state["situ"] = _situ_supported()
    _state["applied"] = True
    return True


def remove() -> None:
    """Restores the extension's functions (tests)."""
    if not _state["applied"]:
        return
    from forge_moss.native import transformer as T

    T.rope_table = _orig["rope_table"]
    T.Block.forward = _orig["block_forward"]
    _state.update(applied=False, ck=None, pe=None, pe_text=None)
