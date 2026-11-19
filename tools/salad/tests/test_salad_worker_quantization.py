"""Fail-closed tests for --load-in-4bit (no GPU or bitsandbytes required)."""
import builtins
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import salad_worker as sw

BF16 = object()


@pytest.fixture(autouse=True)
def _bf16(monkeypatch):
    # Other test modules stub torch in sys.modules; pin a sentinel either way.
    monkeypatch.setattr(sw.torch, "bfloat16", BF16, raising=False)


def test_off_keeps_previous_bf16_path():
    kwargs, optim, fields = sw.quantized_load_kwargs(False)
    assert kwargs == {"torch_dtype": BF16}
    assert optim == "adamw_torch"
    assert fields == {"quantization": "none"}


def test_on_without_bitsandbytes_is_a_hard_stop(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match="requires bitsandbytes"):
        sw.quantized_load_kwargs(True)


def test_on_builds_nf4_double_quant_and_paged_optimizer(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    seen = {}

    class FakeBnb:
        def __init__(self, **kw):
            seen.update(kw)

    fake_tf = types.ModuleType("transformers")
    fake_tf.BitsAndBytesConfig = FakeBnb
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    kwargs, optim, fields = sw.quantized_load_kwargs(True)
    assert isinstance(kwargs["quantization_config"], FakeBnb)
    assert kwargs["torch_dtype"] is BF16
    assert seen == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": BF16,
                    "bnb_4bit_use_double_quant": True}
    assert optim == "paged_adamw_8bit"
    assert fields == {"quantization": "nf4-4bit-double-quant", "optim": "paged_adamw_8bit"}


def test_flag_is_parsed_and_defaults_off():
    src = Path(sw.__file__).read_text()
    assert '"--load-in-4bit", action="store_true"' in src
    assert "optim=optim_name" in src
    assert 'optim="adamw_torch"' not in src.split("cfg = SFTConfig(")[1].split(")")[0]
