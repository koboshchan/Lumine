"""CPU native tokenizer and tiny real transformers model, no full weights required."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_tokens import ACTION_TOKENS, initialize_action_tokens, save_action_tokens


def test_mean_initialization_and_checkpoint_reload(tmp_path):
    import json
    from transformers import AutoProcessor, GPT2Config, GPT2LMHeadModel, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'config.json').read_text())
    processor = AutoProcessor.from_pretrained(config['model'], revision=config['model_revision'], trust_remote_code=True)
    tokenizer = processor.tokenizer
    assert len(ACTION_TOKENS) == 8 and ACTION_TOKENS[-1] == 'LB'
    old_vocab = len(tokenizer)
    specials = set(tokenizer.all_special_tokens)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=old_vocab, n_embd=8, n_layer=1, n_head=1))
    original = model.get_input_embeddings().weight.detach().clone()
    means = {t: original[tokenizer.encode(t, add_special_tokens=False)].float().mean(0) for t in ACTION_TOKENS}
    audit = initialize_action_tokens(processor, model)
    assert specials <= set(tokenizer.all_special_tokens)
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    for token in ACTION_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert ids == [audit[token]['id']]
        if ids[0] >= old_vocab:
            assert torch.equal(model.get_input_embeddings().weight[ids[0]], means[token])
        assert tokenizer.decode(ids, skip_special_tokens=False) == token
    processor.save_pretrained(tmp_path)
    save_action_tokens(tmp_path, audit)
    restored = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=True).tokenizer
    for token in ACTION_TOKENS:
        assert restored.encode(token, add_special_tokens=False) == [audit[token]['id']]
