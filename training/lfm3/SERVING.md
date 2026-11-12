# LFM3 checkpoint serving

Install `pip install -r training/lfm3/requirements-serve.txt`. Serve a trusted,
extracted full checkpoint with its saved processor and training-policy metadata.
Remote model code executes Python; do not load an untrusted checkpoint.

```sh
LUMINE_SERVE_KEY="$(cat /content/serve-key.txt)" \
python -u training/lfm3/serve.py --checkpoint /content/lumine-live/checkpoint-1000 \
  --host 127.0.0.1 --port 8000
```

Forward localhost using an authenticated encrypted private tunnel, not a public
port. Every endpoint requires `Authorization: Bearer $LUMINE_SERVE_KEY`.
`/v1/chat/completions` is stateless debugging, not an equivalent substitute for
training history. `/v1/session/step` and `/v1/session/reset` use the existing runner
session protocol. Only acknowledged executed actions enter normal history.
The original canonical `<use_absolute>` action is preserved for the training
renderer; only the runner response aliases this to `<|use_abs|>`.

For an explicitly authorized **no-input dry-run**, first POST
`/v1/session/bootstrap` with a fresh `session_id`, `frames` containing 1–8 real
chronological PNG data URLs, matching actual `timestamps_s`, and
`acknowledge_no_inputs: true`. Frames must span at most 60 seconds. This seeds
canonical no-input observations, never fabricated images or model predictions.
Continuing unacknowledged dry-run captures use the same explicit no-input
assumption. Such a session rejects executed-action acknowledgements; reset or use
a fresh ordinary session before any real input. Preserve the runner's global F8
emergency stop. Bootstrap is not permission to inject controls.

The server imports the training renderer, uses native processor patch resizing,
retains recent8 observations and sparse5-second middle thumbnails within60seconds,
and trims history using actual processor token counts in the training tier order.
It does not fabricate annotation notes. Malformed actions fail closed, not coerced
into no-ops. A cold session is a known distribution mismatch for checkpoint1000;
valid seeded predictions are not evidence of successful autonomous gameplay.

# Next fresh run action tokens

Pass `--action-special-tokens` to the training command to add exactly
`<|action_start|>`, `<|action_end|>`, `<use_absolute>`, `<|thought_start|>`,
`<|thought_end|>`, `Eleven`, `Twelve`, and `LB`. New input/output embedding rows
start at the mean of their own former subword embeddings. Existing vision specials
are preserved. Every checkpoint saves the processor and action-token-ID audit.
Legacy checkpoints are not upgraded. This initial fresh-only training switch
rejects `--resume`; resuming a tokenized run needs a separate tokenizer-aware
resume implementation before relying on it. A tokenized config without the switch
also fails rather than silently loading inconsistent embeddings.

Native tokenizer inventory found additional split strings including `Caps`, `six`,
`seven`, `nine`, `Seven`, `Eight`, and `Nine`. These are intentionally not added
because the requested change is exactly the eight tokens above.

```sh
pip install pytest httpx
python -m pytest -q training/lfm3/tests/test_serve.py training/lfm3/tests/test_action_tokens.py
```

Native tests fetch only the pinned base processor/tokenizer. The tiny real CPU
model tests new-row means and save/reload; serving GPU smoke tests require the
actual checkpoint. Synthetic black frames test transport, not gameplay quality.
