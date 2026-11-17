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
    assert len(ACTION_TOKENS) == 15 and ACTION_TOKENS[-7:] == (
        'Caps', 'six', 'seven', 'nine', 'Seven', 'Eight', 'Nine')
    from train import ActionCollator, SYSTEM
    from test_native import sample
    processor.chat_template = (root / 'chat_template.jinja').read_text()
    collator = ActionCollator(processor, config)
    plain = 'Capsule sixteen seventeen ninety Seventeen Eighteen Ninety Elevenfold Twelvefold LBrace'
    embedded = ' '.join(f'x{t} {t}x _{t} {t}_ é{t} {t}é' for t in ACTION_TOKENS if t.isalnum())
    untouched = {text: tokenizer.encode(text, add_special_tokens=False)
                 for text in (SYSTEM, plain, embedded)}
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
    for text, ids in untouched.items():
        assert tokenizer.encode(text, add_special_tokens=False) == ids
    # Whole words legitimately change IDs even in prose, never decoded content.
    context = ' \t' + ' ; '.join(ACTION_TOKENS) + '\n  '
    assert tokenizer.decode(tokenizer.encode(context, add_special_tokens=False),
                            skip_special_tokens=False) == context
    current = sample()
    current['current']['action'] = '<|action_start|>0 0 0 ; Caps six seven nine Seven Eight Nine ; <|action_end|>'
    rendered = processor.apply_chat_template(collator.messages(current), tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=False) == rendered
    for token in ACTION_TOKENS[-7:]:
        assert ids.count(audit[token]['id']) == 1
    processor.save_pretrained(tmp_path)
    save_action_tokens(tmp_path, audit)
    restored = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=True).tokenizer
    for token in ACTION_TOKENS:
        assert restored.encode(token, add_special_tokens=False) == [audit[token]['id']]

    for text, ids in untouched.items():
        assert restored.encode(text, add_special_tokens=False) == ids
    for token in ACTION_TOKENS:
        entry = restored.added_tokens_decoder[audit[token]['id']]
        assert entry.single_word == token.isalnum()
        assert not entry.lstrip and not entry.rstrip and not entry.normalized
