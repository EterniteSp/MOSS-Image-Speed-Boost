# MOSS-Image Speed Boost for Forge Neo
# Entry point: patches the moss-image-forge-neo extension's DiT blocks onto comfy_kitchen's fused kernels (see
# lib_moss_speed_boost/boost.py) and adds a "MOSS-Image Speed Boost" accordion to txt2img and img2img. The accordion's
# checkbox turns the boost on; INT8 attention and Fused SiTU are two optional extra optimizations. All three are
# per-generation settings, written to the infotext and restored by PNG Info / "Read generation parameters".
# Forge Neo and the MOSS-Image extension are not modified on disk; delete this folder and everything is as before.

import sys
import traceback
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]


def _find_moss_extension():
    """The moss-image-forge-neo folder, found by its forge_moss package so a renamed clone works too."""
    extensions = ROOT.parent
    preferred = extensions / "moss-image-forge-neo"
    others = sorted(p for p in extensions.iterdir() if p.is_dir() and p != preferred)
    for path in (preferred, *others):
        if (path / "forge_moss" / "native" / "transformer.py").is_file():
            return path
    return None


MOSS = _find_moss_extension()
for path in (ROOT, MOSS):
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules import scripts  # noqa: E402
from modules.infotext_utils import PasteField  # noqa: E402
from modules.ui_components import InputAccordion  # noqa: E402

TITLE = "MOSS-Image Speed Boost"
INFOTEXT_KEY = "MOSS Speed Boost"
LEGACY_KEY = "MOSS kitchen"  # written by the earlier "MOSS Kitchen Boost" builds of this extension
INT8_ATTENTION = "INT8 attention"
FUSED_SITU = "Fused SiTU"

ENABLED = False
try:
    from lib_moss_speed_boost import boost

    if MOSS is None:
        print(f"{boost.TAG} moss-image-forge-neo is not installed; nothing to speed up")
    else:
        ENABLED = boost.apply()
        if ENABLED:
            situ = "Fused SiTU available" if boost.fused_situ_available() else "Fused SiTU needs comfy-kitchen >= 0.2.39"
            print(f"{boost.TAG} MOSS-Image blocks can run on comfy_kitchen kernels ({situ})")
except Exception:
    print(f"[{TITLE}] failed to load, MOSS-Image keeps its own path:")
    traceback.print_exc()


def _infotext_value(int8_attention: bool, fused_situ: bool) -> str:
    extras = [name for name, on in ((INT8_ATTENTION, int8_attention), (FUSED_SITU, fused_situ)) if on]
    return ", ".join(["on", *extras])


def _parse(params: dict):
    """(enabled, int8_attention, fused_situ) from an infotext, or None when it says nothing about the boost."""
    value = params.get(INFOTEXT_KEY)
    if value is not None:
        value = str(value).lower()
        if value.strip() == "off":
            return False, None, None
        return True, INT8_ATTENTION.lower() in value, FUSED_SITU.lower() in value
    value = params.get(LEGACY_KEY)
    if value is not None:
        value = str(value).lower()
        return True, "int8" in value, "fused situ" in value
    return None


def _field(index: int):
    def read(params: dict):
        parsed = _parse(params)
        return None if parsed is None else parsed[index]

    return read


class MossImageSpeedBoost(scripts.Script):
    sorting_priority = 99

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def ui(self, is_img2img):
        situ_ok = boost.fused_situ_available()
        with InputAccordion(True, label=TITLE) as enable:
            gr.Markdown("Runs MOSS-Image on comfy_kitchen's fused kernels, about 2x faster. Has no effect on other "
                        "models. Each option changes fine details at the same seed, so the combinations also give "
                        "close alternatives of one image.")
            with gr.Row():
                int8_attention = gr.Checkbox(True, label=INT8_ATTENTION,
                                             info="Quantized attention, about 1.25x faster; off = bf16 SDPA")
                fused_situ = gr.Checkbox(situ_ok, label=FUSED_SITU, interactive=situ_ok,
                                         info="SiTU folded into the ffn_down quantization, about 1.25x faster"
                                         if situ_ok else "Needs comfy-kitchen 0.2.39 or newer")

        self.infotext_fields = [
            PasteField(enable, _field(0)),
            PasteField(int8_attention, _field(1)),
            PasteField(fused_situ, _field(2)),
        ]
        self.paste_field_names = [INFOTEXT_KEY]
        return [enable, int8_attention, fused_situ]

    def process(self, p, enable=True, int8_attention=True, fused_situ=True, *args):
        boost.configure(enable, int8_attention, fused_situ)
        if not getattr(getattr(p, "sd_model", None), "is_moss_image", False):
            return
        if not enable:
            p.extra_generation_params[INFOTEXT_KEY] = "off"
            return
        p.extra_generation_params[INFOTEXT_KEY] = _infotext_value(
            boost.int8_attention_available() and int8_attention, boost.fused_situ() and fused_situ)
