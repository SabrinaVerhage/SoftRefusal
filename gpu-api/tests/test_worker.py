"""Activation recording through the real worker code path, on a tiny random
Gemma 3 architecture on CPU. Exercises no downloaded checkpoint or GPU."""
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soft_refusal.progress import request_cancel
from soft_refusal.worker import Runner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig,
                              PreTrainedTokenizerFast, SiglipVisionConfig)
    monkeypatch.setenv("SOFT_REFUSAL_DATA", str(tmp_path))
    torch.manual_seed(42)
    text = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=256, sliding_window=32,
                            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
                            bos_token_id=1, eos_token_id=2, pad_token_id=0)
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=8, patch_size=4)
    config = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                          image_token_index=31, mm_tokens_per_image=4, boi_token_index=29, eoi_token_index=30)
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                               "user": 4, "model": 5, "hello": 6, "world": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]",
                                        eos_token="[EOS]", pad_token="[PAD]")
    # Gemma's processor passes message content as a list of typed parts.
    tokenizer.chat_template = ("[BOS] {% for m in messages %}{{ m['role'] }} {{ m['content'][0]['text'] }} "
                               "{% endfor %}{% if add_generation_prompt %}model{% endif %}")
    runner = Runner("gemma")
    runner.model = Gemma3ForConditionalGeneration(config).eval()
    runner.processor, runner.torch, runner.device = tokenizer, torch, "cpu"
    return runner


def chat(**extra):
    return {"model": "gemma", "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 4, "temperature": 0, "top_p": 0.95, "seed": 0, **extra}


def test_records_resid_post_at_final_prompt_token(runner, tmp_path):
    result = runner.run("a" * 32, chat(record_layers=[2, 0], trace=True))
    assert "error" not in result
    meta = result["activations"]
    path = tmp_path / "activations" / f"{'a' * 32}.npz"
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["layers"] == [2, 0] and meta["dimensions"] == 16 and meta["num_layers"] == 3
    assert meta["token_index"] == len(result["input_token_ids"]) - 1
    assert meta["module_path"] == "model.language_model.layers"
    ids = torch.tensor([result["input_token_ids"]])
    with torch.inference_mode():
        out = runner.model.model.language_model(input_ids=ids, output_hidden_states=True, use_cache=False)
    with np.load(path, allow_pickle=False) as data:
        assert sorted(data.files) == ["layer_0", "layer_2"]
        # hidden_states[0] is the embedding output; block N writes index N + 1.
        np.testing.assert_allclose(data["layer_0"], out.hidden_states[1][0, -1].numpy(), rtol=1e-5, atol=1e-6)
        # HF reports the last block after final RMSNorm; the recording is resid_post, before it.
        last = torch.from_numpy(data["layer_2"])
        torch.testing.assert_close(runner.model.model.language_model.norm(last), out.last_hidden_state[0, -1])
        assert not np.allclose(data["layer_2"], out.last_hidden_state[0, -1].numpy())
    trace = result["education_trace"]
    assert len(trace["candidates"]) == result["usage"]["completion_tokens"]
    assert [x["layer"] for x in trace["layers"]] == [0, 1, 2]
    assert trace["tokens"]["total"] == len(result["input_token_ids"])
    assert {s["layer"] for s in result["snapshot_preview"]} == {0, 2}
    assert result["recording_metadata"]["model_config"]["text_config"]["num_hidden_layers"] == 3
    progress = json.loads((tmp_path / "progress" / f"{'a' * 32}.json").read_text())
    assert progress["trace"]["snapshots"] and progress["input_tokens"] == len(result["input_token_ids"])
    assert all(not block._forward_hooks for block in runner.model.model.language_model.layers)


def test_plain_chat_records_nothing(runner, tmp_path):
    result = runner.run("b" * 32, chat())
    assert "error" not in result and "activations" not in result and "education_trace" not in result
    assert not list((tmp_path / "activations").glob("*")) if (tmp_path / "activations").exists() else True


def test_invalid_layer_and_cancellation_keep_explicit_outcomes(runner, tmp_path):
    result = runner.run("c" * 32, chat(record_layers=[3]))
    assert result["error"]["code"] == "invalid_record_layers" and result["outcome"] == "technical_error"
    request_cancel(tmp_path, "d" * 32)
    result = runner.run("d" * 32, chat(record_layers=[0]))
    assert result["error"]["code"] == "cancelled_by_user"
    assert result["outcome"] == "cancelled" and result["label_source"] == "user"
    assert not (tmp_path / "activations" / f"{'d' * 32}.npz").exists()
