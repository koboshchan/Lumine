"""Fifteen action protocol tokens for NEXT FRESH runs, not legacy checkpoint upgrades."""
import json
from pathlib import Path

ACTION_TOKENS = ('<|action_start|>', '<|action_end|>', '<use_absolute>',
                 '<|thought_start|>', '<|thought_end|>', 'Eleven', 'Twelve', 'LB',
                 'Caps', 'six', 'seven', 'nine', 'Seven', 'Eight', 'Nine')


def register_action_tokens(tokenizer):
    """Register atomic markers and boundary-safe lexical actions without weights."""
    from tokenizers import AddedToken
    # Plain strings default to substring matching. Lexical actions must not split
    # words like Capsule/seventeen/Ninety, or consume their surrounding whitespace.
    # Markers remain matchable directly adjacent to action payloads.
    added = [AddedToken(token, single_word=token.isalnum(), lstrip=False,
                        rstrip=False, normalized=False, special=True)
             for token in ACTION_TOKENS]
    tokenizer.add_special_tokens({'extra_special_tokens': added}, replace_extra_special_tokens=False)


def initialize_action_tokens(processor, model):
    """Initialize new rows to their own old subword mean, never random semantics."""
    import torch
    tokenizer = processor.tokenizer
    old_size = len(tokenizer)
    pieces = {token: tokenizer.encode(token, add_special_tokens=False) for token in ACTION_TOKENS}
    inputs = model.get_input_embeddings().weight
    outputs = model.get_output_embeddings()
    with torch.no_grad():
        input_means = {token: inputs[ids].float().mean(0).clone() for token, ids in pieces.items()}
        output_means = ({token: outputs.weight[ids].float().mean(0).clone()
                         for token, ids in pieces.items()} if outputs is not None else {})
    register_action_tokens(tokenizer)
    if len(tokenizer) != old_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        for token in ACTION_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id >= old_size:
                model.get_input_embeddings().weight[token_id].copy_(input_means[token])
                output = model.get_output_embeddings()
                if output is not None:
                    output.weight[token_id].copy_(output_means[token])
    audit = {token: {'id': tokenizer.convert_tokens_to_ids(token), 'previous_ids': pieces[token],
                     'single_word': token.isalnum(), 'lstrip': False, 'rstrip': False,
                     'normalized': False, 'special': True}
             for token in ACTION_TOKENS}
    for token, entry in audit.items():
        if tokenizer.encode(token, add_special_tokens=False) != [entry['id']]:
            raise ValueError('Action token is not atomic: ' + token)
    return audit


def save_action_tokens(output, audit):
    Path(output, 'action-token-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
