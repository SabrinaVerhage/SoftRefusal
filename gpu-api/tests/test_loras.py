"""Synthetic adapter isolation tests: no model files, downloads or GPU."""
import hashlib
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from soft_refusal import loras
from soft_refusal.schemas import ImageRequest


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    data = b"synthetic adapter fixture"
    (tmp_path / "fixture.safetensors").write_bytes(data)
    monkeypatch.setenv("SOFT_REFUSAL_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("SOFT_REFUSAL_ENABLE_LORAS", "1")
    monkeypatch.setattr(loras, "CATALOG", {"body-shape-v1": {
        "filename": "fixture.safetensors", "sha256": hashlib.sha256(data).hexdigest(),
        "name": "Test adapter", "version": "test"}})
    monkeypatch.setattr(loras, "adapter_state", lambda *args: {"synthetic": "weights"})
    monkeypatch.setattr(loras, "verify_loaded", lambda *args: None)
    return tmp_path


def test_adapter_unloads_before_next_original_request(adapter):
    pipe = Mock()
    with loras.image_adapter(pipe, "body-shape-v1", .6) as metadata:
        assert metadata["strength"] == .6
        pipe.unload_lora_weights.assert_not_called()
    pipe.unload_lora_weights.assert_called_once_with()
    pipe.load_lora_weights.assert_called_once_with({"synthetic": "weights"}, adapter_name="visitor")
    pipe.set_adapters.assert_called_once_with(["visitor"], adapter_weights=[.6])
    pipe.reset_mock()
    with loras.image_adapter(pipe) as metadata:
        assert metadata is None
    assert not pipe.mock_calls


@pytest.mark.parametrize("stage", ["load", "generate"])
def test_failure_always_unloads(adapter, stage):
    pipe = Mock()
    if stage == "load":
        pipe.load_lora_weights.side_effect = RuntimeError("partial load")
    with pytest.raises(RuntimeError):
        with loras.image_adapter(pipe, "body-shape-v1"):
            raise RuntimeError("generation failed")
    pipe.unload_lora_weights.assert_called_once_with()


def test_corrupt_disabled_and_unknown_adapters_do_not_load(adapter, monkeypatch):
    pipe = Mock()
    (adapter / "fixture.safetensors").write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="checksum"):
        with loras.image_adapter(pipe, "body-shape-v1"):
            pass
    monkeypatch.setenv("SOFT_REFUSAL_ENABLE_LORAS", "0")
    for name in ["body-shape-v1", "../../other", "https://example.com/model"]:
        with pytest.raises(ValueError):
            with loras.image_adapter(pipe, name):
                pass
    assert not pipe.mock_calls


@pytest.mark.parametrize("fields", [
    {"lora": "unknown"}, {"lora_strength": True}, {"lora_strength": "0.7"},
    {"lora_strength": 0}, {"lora_strength": 1.1}, {"lora_strength": float("nan")},
])
def test_request_rejects_unlisted_or_invalid_adapter_settings(fields):
    with pytest.raises(ValidationError):
        ImageRequest(prompt="A fully clothed adult portrait", **fields)


def test_real_tensor_export_normalization_and_load_verification(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    transformer = torch.nn.Module()
    transformer.linear = torch.nn.Linear(3, 4, bias=False)
    a, b = torch.randn(2, 3), torch.randn(4, 2)
    path = tmp_path / "fixture.safetensors"
    safetensors.save_file({"linear.lora_A.default.weight": a, "linear.lora_B.default.weight": b}, str(path))
    state = loras.adapter_state(path, transformer)
    assert set(state) == {"transformer.linear.lora_A.weight", "transformer.linear.lora_B.weight"}
    transformer.linear.lora_A = torch.nn.ModuleDict({"visitor": torch.nn.Linear(3, 2, bias=False)})
    transformer.linear.lora_B = torch.nn.ModuleDict({"visitor": torch.nn.Linear(2, 4, bias=False)})
    with torch.no_grad():
        transformer.linear.lora_A["visitor"].weight.copy_(a)
        transformer.linear.lora_B["visitor"].weight.copy_(b)
    loras.verify_loaded(state, transformer)
    with torch.no_grad():
        transformer.linear.lora_B["visitor"].weight.zero_()
    with pytest.raises(ValueError, match="not loaded exactly"):
        loras.verify_loaded(state, transformer)
    path = tmp_path / "wrong-shape.safetensors"
    safetensors.save_file({"linear.lora_A.default.weight": torch.randn(2, 9)}, str(path))
    with pytest.raises(ValueError, match="shape"):
        loras.adapter_state(path, transformer)
    path = tmp_path / "missing-pair.safetensors"
    safetensors.save_file({"linear.lora_A.default.weight": a}, str(path))
    with pytest.raises(ValueError, match="pair"):
        loras.adapter_state(path, transformer)
