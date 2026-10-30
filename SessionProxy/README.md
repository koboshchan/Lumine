Run from the Lumine repository root with python -m SessionProxy.service after installing SessionProxy/requirements.txt. The service binds only to 127.0.0.1, default port 8766. Expose it only behind an authenticated, encrypted transport. It never launches an upstream server or allocates a GPU.

Set LUMINE_PROXY_SECRET to a random ASCII secret with at least 32 bytes and 12 distinct characters, and LUMINE_MODEL to the configured upstream model. LUMINE_UPSTREAM defaults to http://127.0.0.1:8000/v1/chat/completions and must remain a literal loopback HTTP address with that exact path. LUMINE_UPSTREAM_KEY optionally authenticates the local inference endpoint. Redirects, proxies inherited from the environment, and response decompression are disabled.

The default protocol is image-only. LUMINE_MODEL_PROTOCOL=tiered-v1 is an explicit compatible-model opt-in, not a claim that an old checkpoint supports history. All modes require the shared exact-processor adapter; no approximate token counter or dummy notes are provided. LUMINE_PROMPT_ADAPTER defaults to SessionProxy.adapter:build_step. This is a callable, not a factory. The parent owns that adapter and its LUMINE_PROCESSOR_DIR local tokenizer/processor loading. Missing shared code must fail closed.

The callable receives keyword arguments state, frame, executed, now, protocol, max_input_tokens and allow_compaction. State is a private deep copy. Frame is a validated canonical RGB PNG data URL padded to 1280x736 with black bottom rows. Executed is None unless the client exactly acknowledges the previous action and action ID; acknowledged values include action, action_id, frame, note and note_provenance. Notes remain model-generated, not assertions of gameplay truth. The builder must retain observations separately from predictions and never insert an unexecuted action into its prompt.

The adapter returns state, messages, input_tokens and compacted. Input tokens must be an exact positive integer from the real native processor/template expansion. Tiered messages must preserve the previous messages as an exact list prefix except at completed-snapshot adoption or a dry-run generation boundary. The bundled adapter declares snapshot_mode=True and additionally accepts memory_snapshot immutable bytes; allow_compaction is always false and never enables inline work. Its async compact hook is owned exclusively by the bounded worker. The old injected-builder interface remains available for isolated contract tests, not a substitute production compactor.

Every step reserves a per-session busy slot before upstream inference. A concurrent first request for the same new ID cannot create a second session. Inference errors, invalid completions, cancellation and detected disconnects leave history and pending predictions unchanged. Request IDs are intentionally burned before attempted inference, including failed attempts. The bounded replay window lasts until session TTL expiry, defaults to 900 seconds, and retains at most 8192 IDs per session. Reset clears gameplay state but retains replay tombstones until expiry. Clients should choose a fresh session ID on each Start and never retry an ambiguous request.

Defaults permit eight sessions, two concurrent inferences, 24 MiB per session, 96 MiB aggregate committed and reserved state, 30000 exact input tokens and 512 output tokens. Body size is capped at 6 MiB, PNG compressed bytes at 4 MiB, and upstream completion bodies at 32 KiB. Session exhaustion fails closed rather than silently evicting active state. No raw model completion is returned on failure. Successful response delivery cannot be proven over HTTP; clients must stop and start a new session after an ambiguous network failure.

CPU tests run with python -m pytest -q --asyncio-mode=auto SessionProxy. Fixture tests use aiohttp TestServer, synthetic pixels and local fake inference, never model weights or a GPU. Optional LUMINE_TEST_PROCESSOR=/path/to/checkpoint enables real saved native-tokenizer and AutoProcessor grid tests; fixture token counts alone are not evidence of processor integration.

For the bundled adapter, set LUMINE_PROCESSOR_DIR to the same saved checkpoint directory used by your inference server. Loading is local-only, remote Python code is forbidden, and intact action/image tokenizer markers and native saved chat template are required. No model weights are loaded by the adapter. Pixel validation, 920/60 fixed image-grid token expansion and the saved tokenizer provide exact input counts. Install pytest and pytest-asyncio separately to run the fixture suite.

The bundled SessionProxy.memory builder reuses native rendering from lumine_memory.prompt. Its online snapshot state machine is intentionally separate from the legacy synchronous lumine_memory.live training experiment. Snapshot adoption changes the prefix and increments a generation counter; it does not pretend to retain a KV prefix after replacement. A stateless local server sees the changed prefix automatically; no global cache-flush endpoint is called because that could disturb other sessions.

Completed snapshots retain up to16 recent acknowledged full frames. Older rows keep 320x192 padded thumbnails, sampled every2 seconds for at most5 minutes. Rolling text summaries are explicitly model-generated-unreviewed and hard bounded by2048 native tokens, with at most512 generated tokens per summary. Old notes and eligible acknowledged evidence feed the next genuine background summary; invalid output leaves the prior snapshot untouched. This is no claim of one-hour learned retention. Explicit tiered-v1 mode is a protocol opt-in, not evidence that any existing checkpoint was trained for it. Dataset notes still require independent teacher provenance and review; serving notes are not silently converted into reviewed training labels.

A proxy and its secret together are a single trust boundary. Session IDs are unguessable client-generated identifiers, not independent multi-tenant accounts; everyone holding the same secret is trusted to reset a known session. Bind remains loopback; forward the proxy port through your authenticated encrypted tunnel. For the existing image-only checkpoint, retain the upstream image count1 and max-model-len2048 flags. A future tiered-compatible checkpoint requires upstream max-model-len at least30512, enough image capacity for its selected context, and do_resize=false, with separate actual GPU dry-run validation before live input.

Example configuration, with secrets supplied privately through environment variables, is LUMINE_MODEL=lumine-mini-2b, LUMINE_PROCESSOR_DIR=/content/lumine-test and LUMINE_MODEL_PROTOCOL=image-only. Run python -m SessionProxy.service from the repository root, tunnel the chosen LUMINE_PROXY_PORT to Windows, select Session proxy mode in the runner, and use http://127.0.0.1:PORT/v1 as endpoint. No proxy is started by installing the files.

Tier counts describe the compacted history snapshot, not a sliding prefix edited every request. Between completed-snapshot adoptions, full observations append. Reaching the exact cap fails closed if the background worker has not produced an adoptable snapshot; it never forces inline image or model work. Thus the active segment can temporarily contain more than16 full images; it remains hard bounded by30000 input tokens and session bytes. Preserving an immutable cached prefix and simultaneously deleting a sliding old full frame on every step are incompatible. The offline training work must reproduce this state machine before claiming training/serving cache-policy parity. Shared native rendering and tier selection alone are not that proof. This proxy is NOT an annotation/training allocation gate and must not be used to lift a compaction-related GPU hold.

Dry-run observations have no corresponding executed assistant action. Continuing a native generation prefix would require inventing one, so this case is marked as a generation-prefix discontinuity instead. It preserves observations without creating an assistant action, and does no image/model compaction. Cache-prefix tests apply to successfully acknowledged action continuations. Reset, errors and ambiguous delivery start a fresh runner session; no fake noop action is inserted to preserve cache.


Asynchronous completed memory snapshots

The bundled adapter never downsamples history images or asks the model to write memory on an action request. Its action path validates the newest frame, appends only actual execution acknowledgments, adopts the last completed immutable snapshot at a request boundary, and performs exact native-token budget checks. If memory is late and the hard budget is reached, it fails closed without doing compaction inline or waiting for the worker.

A lifecycle-owned background coroutine owns one active job globally and at most one coalesced pending job per existing session. Admission accounts for serialized active/queued copies alongside session state and immutable completed snapshots under the existing per-session/global byte limits. There is no queue of unbounded per-frame tasks. Default admission cadence is two seconds per session; an already queued job is replaced with the newest eligible committed state. Prefixes stay append-only between completed-snapshot adoptions, rather than being rebuilt on every action.

Jobs serialize committed state before admission. The dedicated single-thread CPU executor prepares thumbnails and native-budgeted evidence, then a separate asynchronous local model completion writes a genuine factual summary, then the CPU executor validates/render-counts the completed snapshot. No session/action lock is held across image work or a model request. Running native CPU work cannot be canceled by asyncio; cancellation therefore drains that one owned CPU call before another job can be submitted, preventing canceled tasks from accumulating in the executor queue. Shutdown joins owned jobs before closing its resources.

Only acknowledged observations strictly before the job boundary are eligible. The latest unexecuted observation is excluded. The retained snapshot has recent full frames, sparse 320x192 thumbnails from the last five minutes, and an unreviewed model-generated rolling text note. Its prefix ends with an actually executed assistant action. Adoption replaces only the acknowledged prefix through that sequence; every newer observation and execution acknowledgment survives. Old or repeated snapshots cannot roll the prompt backward. Raw action completion notes are ignored, and action requests stop at the memory marker, so memory writing belongs to the background path.

Completed snapshots are immutable serialized bytes. Publication runs without an await between checking the live session epoch/TTL and swapping that byte reference. Reset changes the epoch and removes queued jobs; TTL removal does the same invalidation. A running old job never creates a session or resurrects old memory. A failed, canceled, invalid or expired job cannot replace the last good snapshot. Publishing during an in-flight action does not alter that action's prompt; the next request adopts the completed snapshot.

Action inference and note generation use separate lifecycle-bound HTTP connection pools; a blocked note request does not consume the action connector's capacity. Both remain authenticated local-upstream requests with bounded response bodies and deadlines. This isolates application scheduling, not physical GPU resources: sharing one model server/GPU can still cause scheduling contention. No GPU latency, five-actions-per-second throughput, automatic model competence, hour-long retention quality, training/proxy parity, or training-hold release is established by CPU concurrency tests.

Optional settings are `LUMINE_MEMORY_TIMEOUT` (default 30 seconds) and `LUMINE_MEMORY_INTERVAL` (default 2 seconds). The service remains loopback-bound, all session step/reset access requires the same secret, and all existing runner safety behavior is unchanged.

CPU tests include deliberately blocked real thumbnail preparation and deliberately blocked real summary awaits while a separate action HTTP request completes before either blocker is released. They also exercise separate real HTTP upstream connections, immutable-prefix/suffix adoption, coalescing and byte admission, reset/reuse and TTL invalidation, failure preservation, strict acknowledgment cutoffs, native processor grids, and cancellation draining. These are deterministic dependency tests, not GPU timing benchmarks.

Training hold remains in force. This online worker's causal evidence selection and rolling note generation are not yet accepted as identical to the annotation/training builder.


Action tokenizer identity

The adapter checks only the seven genuinely new custom tokens against the exact
IDs frozen from the hash-verified `lumine/test-2026-10-29/` checkpoint. It fails
closed on different IDs, even if every entry is still individually atomic.
History contains only runner-acknowledged executed actions, serialized verbatim
with the training action delimiters, six semicolon-separated key chunks and
optional absolute-position marker. Pending predictions are never history.

The separate 37-entry action-vocabulary test fixture is NOT 37 newly added context-invariant
special tokens. Seven were newly added; thirty key-name entries already existed
in the base vocabulary and use leading-space BPE IDs in canonical action strings
(e.g. bare `Nine` is 83329, but ` Nine` is 37066). This is the saved test model's
actual tokenizer behavior, preserved rather than silently changing training
serialization. Output is decoded by the local model server, then validated by
the proxy's strict text action parser. Thought markers are not executable actions.
Exact saved-checkpoint tests retain all 37 standalone IDs as compatibility evidence, not custom-token requirements, and verify full serialized action
ID sequences inside native prompts, output decode/parser roundtrips, and HTTP
ACK-only history. Set `LUMINE_TEST_PROCESSOR` to the saved test checkpoint to run
the exact-ID regressions; a missing checkpoint skips them rather than asserting
identity from a vocabulary count. No GPU is used.

Custom IDs remain action_start 151669, action_end 151670, use_absolute 151671,
thought_start 151672, thought_end 151673, Eleven 151674 and Twelve 151675.
The remaining thirty words are not custom-token requirements. Their parser
allowlist and existing vocabulary stay intact.
