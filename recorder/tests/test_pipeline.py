import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avoidance.collect import collect, validate_conversations
from avoidance.io import load_dataset, read_json, read_jsonl
from avoidance.model import HFAdapter, residual_hooks
from avoidance.probe import read_annotations, score, train_probe
from avoidance.reduce import reduce
from avoidance.scope import encode, features

ROOT = Path(__file__).parents[1]


@pytest.fixture
def smoke(tmp_path):
    return collect(ROOT / "configs/gemma3-4b-local.json", ROOT / "data/neutral.jsonl", tmp_path / "smoke", True)


def test_collection_provenance_and_history(smoke):
    manifest, rows, arrays = load_dataset(smoke)
    assert manifest["synthetic"] and len(rows) == 12
    assert arrays["layer_17"].shape == (12, 32)
    assert rows[1]["messages"][1]["content"] == rows[0]["response"]
    assert rows[2]["messages"] == [{"role": "user", "content": rows[2]["prompt"]}]
    assert all(r["outcome"] == "unlabeled" for r in rows)
    with pytest.raises(ValueError, match="Synthetic"):
        train_probe(smoke, smoke / "annotations.csv", smoke / "bad")
    with pytest.raises(ValueError, match="real Gemma"):
        features(smoke, smoke / "bad-features")
    with (smoke / "steps.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(smoke)


def test_umap_and_trajectory(smoke, tmp_path):
    a = reduce(smoke, tmp_path / "map-a", 17)
    b = reduce(smoke, tmp_path / "map-b", 17)
    np.testing.assert_array_equal(np.load(a / "coordinates.npy"), np.load(b / "coordinates.npy"))
    points = read_jsonl(a / "points.jsonl")
    assert points[0]["previous_point_id"] is None
    assert points[1]["previous_point_id"] == points[0]["point_id"]
    assert points[2]["previous_point_id"] is None
    assert read_json(a / "reduction.json")["synthetic"]


def test_jump_relu_has_no_bias_subtraction():
    weights = {"w_enc": torch.eye(3), "b_enc": torch.tensor([.5, 0., 0.]),
               "b_dec": torch.ones(3) * 100, "threshold": torch.tensor([1., 1., 0.])}
    torch.testing.assert_close(encode(torch.tensor([[1., 1., -1.]]), weights), torch.tensor([[1.5, 0., 0.]]))


@pytest.fixture
def tiny_adapter():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Gemma3Config, Gemma3TextConfig, Gemma3ForConditionalGeneration, PreTrainedTokenizerFast, SiglipVisionConfig
    torch.manual_seed(42)
    text = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=8, max_position_embeddings=256, sliding_window=32,
                           layer_types=["sliding_attention", "full_attention"],
                           bos_token_id=1, eos_token_id=2, pad_token_id=0)
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=8, patch_size=4)
    cfg = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                       image_token_index=31, mm_tokens_per_image=4, boi_token_index=29, eoi_token_index=30)
    model = Gemma3ForConditionalGeneration(cfg).eval()
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                               "user": 4, "assistant": 5, "hello": 6, "world": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.chat_template = "[BOS] {% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    adapter = HFAdapter.__new__(HFAdapter)
    adapter.model, adapter.tokenizer = model, tokenizer
    adapter.backbone, adapter.blocks = model.model.language_model, model.model.language_model.layers
    adapter.module_path = "model.language_model.layers"
    adapter.config = {"layers": [0, 1], "token_selection": "last_prompt", "max_input_tokens": 128, "max_new_tokens": 2}
    return adapter


def test_real_gemma_architecture_hook_before_final_norm(tiny_adapter):
    a = tiny_adapter
    ids = torch.tensor([[1, 4, 6, 7, 5]])
    with torch.inference_mode(), residual_hooks(a.blocks, [0, 1], -1) as captured:
        out = a.backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
    np.testing.assert_allclose(captured[0], out.hidden_states[1][0, -1].numpy(), rtol=1e-6)
    normalized = a.backbone.norm(torch.from_numpy(captured[1]))
    torch.testing.assert_close(normalized, out.last_hidden_state[0, -1])
    assert not np.allclose(captured[1], out.last_hidden_state[0, -1].numpy())
    assert all(not b._forward_hooks for b in a.blocks)
    with pytest.raises(RuntimeError):
        with residual_hooks(a.blocks, [0], -1):
            raise RuntimeError("simulated failure")
    assert not a.blocks[0]._forward_hooks


def test_adapter_token_positions_and_determinism(tiny_adapter):
    a = tiny_adapter
    messages = [{"role": "user", "content": "hello world"}]
    first, vec = a.step(messages)
    second, vec2 = a.step(messages)
    assert first == second
    assert first["token_index"] == len(first["prompt_token_ids"]) - 1
    assert first["token_id"] == first["prompt_token_ids"][-1]
    np.testing.assert_array_equal(vec[1], vec2[1])
    assert all(not b._forward_hooks for b in a.blocks)
    a.config["token_selection"] = "last_response"
    result, _ = a.step(messages)
    assert result["phase"] == "post_response" and result["token_index"] >= len(result["prompt_token_ids"])
    a.config["max_input_tokens"] = 1
    with pytest.raises(ValueError, match="no silent truncation"):
        a.step(messages)


def test_collection_through_tiny_gemma(tiny_adapter, tmp_path, monkeypatch):
    import avoidance.model as module
    config = read_json(ROOT / "configs/gemma3-4b-local.json")
    config.update(layers=[0, 1], max_new_tokens=2, max_input_tokens=128, dtype="float32")
    tiny_adapter.config = config
    monkeypatch.setattr(module, "HFAdapter", lambda _: tiny_adapter)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    inputs = tmp_path / "input.jsonl"
    inputs.write_text(json.dumps({"conversation_id": "test", "group_id": "test", "turns": ["hello", "world"]}))
    out = collect(cfg, inputs, tmp_path / "tiny")
    manifest, rows, arrays = load_dataset(out)
    assert arrays["layer_1"].shape == (2, 16)
    assert rows[1]["messages"][1]["content"] == rows[0]["response"]
    assert rows[1]["input_token_count"] > rows[0]["input_token_count"]
    assert manifest["generation_overrides"]["max_new_tokens"] == 2


def test_duplicate_conversations():
    row = {"conversation_id": "x", "group_id": "x", "turns": ["hello"]}
    with pytest.raises(ValueError, match="unique"):
        validate_conversations([row, row])


def labeled_fixture(tmp_path):
    rows, annotations = [], []
    rng = np.random.default_rng(4)
    x = []
    for split, count in [("train", 8), ("calibration", 6), ("test", 6)]:
        for y in [0, 1]:
            for j in range(count):
                key = f"{split}-{y}-{j}"
                rows.append({"point_id": key, "group_id": key, "conversation_id": key})
                annotations.append({"point_id": key, "outcome": "refusal" if y else "allow",
                                    "annotator": "test fixture", "split": split, "notes": "synthetic unit test only"})
                x.append(rng.normal(size=8) + 1.5 * y)
    path = tmp_path / "labels.csv"
    def save():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(annotations[0])); w.writeheader(); w.writerows(annotations)
    save()
    return rows, annotations, np.array(x, dtype=np.float32), path, save


def test_probe_split_leakage_and_insufficient_labels(tmp_path):
    rows, annotations, x, path, save = labeled_fixture(tmp_path)
    assert len(read_annotations(path, rows)) == len(rows)
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        read_annotations(path, rows)
    rows[-1]["group_id"] = rows[-1]["point_id"]
    for a in annotations:
        a["outcome"] = "allow"
    save()
    with pytest.raises(ValueError, match="independent groups"):
        read_annotations(path, rows)


def test_probe_training_and_signature_checks(tmp_path, monkeypatch):
    # Fabricated separable arrays test statistical plumbing, never saved as real research data.
    import avoidance.probe as p
    rows, annotations, x, path, _ = labeled_fixture(tmp_path)
    manifest = {"synthetic": False, "config": {"model_id": "TEST-ONLY", "revision": "x", "adapter": "test",
                "dtype": "float32", "token_selection": "last_prompt"}, "chat_template_sha256": "x",
                "environment": {"packages": {"transformers": "test"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(p, "load_dataset", lambda _: (manifest, rows, {"layer_17": x}))
    out = train_probe(tmp_path, path, tmp_path / "probe")
    evaluation = read_json(out / "evaluation.json")
    assert evaluation["roc_auc"] > .8
    assert len(read_jsonl(out / "heldout_scores.jsonl")) == 12
    score(tmp_path, out / "probe.joblib", tmp_path / "scores")
    manifest["config"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="signature mismatch"):
        score(tmp_path, out / "probe.joblib", tmp_path / "wrong")
