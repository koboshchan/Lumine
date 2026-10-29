Lumine Windows runner

A Windows x64 native Go GUI, with no Python runtime, browser, installer, cgo or external DLLs beyond Windows system libraries. Build from Linux with Go 1.23 or newer.

```sh
cd Runner
go test -race ./...
GOOS=windows GOARCH=amd64 go vet ./...
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags='-s -w -H=windowsgui' -o LumineRunner.exe .
```

Launch LumineRunner.exe, enter the /v1 endpoint, served model name and API key, then pick the Genshin window. Dry run is enabled by default. Press Start and focus the game during the five-second countdown. F8 works globally and cancels inference, stops execution and releases held inputs. Stop, closing the app, target focus loss, malformed output and endpoint errors also stop execution. If another app owns F8, the runner refuses to start.

The GUI shows the latest frame, raw predicted action, capture/inference latency and actual decision rate. Capture samples at 5Hz with one latest-only slot; inference and playback are serialized, so slow endpoints reduce decision rate rather than queueing actions. Dry run parses and displays output but never calls SendInput.

Use a visible, unobstructed 16:9 windowed or borderless game. Native 1280x720 is the exact training geometry; larger 16:9 client areas are downscaled. The explicit GDI fallback uses screen capture, not Windows Graphics Capture. Exclusive fullscreen, protected/flip-model surfaces, overlays and occlusion can produce incorrect or black frames. Check the preview in dry run before enabling input. No anti-cheat bypass is included. SendInput can be rejected by elevated games or game protections; this has not been validated inside Genshin.

The GUI uses physical-pixel native controls and is not a responsive/per-monitor-scaled layout. It is an initial runner, not a polished installer.

OBS game capture mode

Select OBS in the Capture selector to read the OBS game-capture source rather than GDI's visible desktop pixels. OBS must already be open, rendering the game and running its v5 websocket server. Enable the server yourself in Tools > WebSocket Server Settings, normally on port 4455. This runner never launches, reconfigures or closes OBS and does not start a recording. Enter an address such as ws://127.0.0.1, its separate port and the masked password. Use wss:// for TLS where available; otherwise keep websocket traffic on localhost or a trusted encrypted tunnel. Authentication does not encrypt screenshots.

Set Source to the exact input/source name (for example your Game Capture source). Leave it blank to resolve GetCurrentProgramScene before each screenshot; this follows program scene changes, not studio-mode preview. The program scene is preferable when training used scene transforms, cropping or overlays. A named input screenshots that input directly and may not include those scene transforms. Confirm the preview in dry run before enabling input.

Both password-protected and authentication-disabled OBS servers are supported. The server's Hello determines the v5 handshake. With an authentication challenge, the runner computes the specified two-stage SHA256/base64 response and sends Identify with that response. A blank password on an authentication-required server stops with a clear password-required error, never an unauthenticated retry. With no authentication object in Hello, Identify omits authentication entirely, whether the password box was blank or still contained a previously typed value. Passwords are memory-only, never logged or saved, and the masked field is cleared on Start. Re-enter the password for another protected connection. Authentication-disabled OBS servers should only be used on a trusted local interface.

Each GetSourceScreenshot requests JPEG at 1280x720 and quality 90. OBS preserves aspect ratio by scaling to fit, so a non-16:9 result is rejected rather than stretched or silently cropped. The JPEG header is checked for exact dimensions before decoding. Captures use one request sequence at a time, bounded websocket messages and operation deadlines; there is one latest-frame slot, no queued ticks and no overlapping screenshot calls. Sampling is paced to at most 5Hz, not a promise that OBS can supply 5fps. Slow screenshot or inference work reduces throughput instead of accumulating a backlog.

For training alignment, keep the same OBS game capture, scene composition and 16:9 geometry used for the recordings, with training's 1280x720 output and Area scale setting. GetSourceScreenshot is a separate OBS source-rendering/scaling API, not a read of the encoder output. Merely requesting 1280x720 cannot guarantee byte-identical output or the same resampling filter as the recording pipeline; JPEG is also lossy. Match the scene/source and inspect frames rather than treating the websocket screenshot as a perfect reconstruction of the training recording.

GDI remains a manually selected fallback. OBS authentication, capture, disconnect and geometry errors stop the run; they never silently switch to a different source. Select GDI explicitly and Start again to use the visible target window instead. Status identifies OBS or GDI and reports screenshot/capture latency separately from inference. F8, cancellation and foreground-window checks still apply in OBS mode. The target must remain focused and its geometry valid even though OBS supplies the pixels.

Protocol and dependency references checked for this implementation are the official [obs-websocket v5.7.4 protocol](https://github.com/obsproject/obs-websocket/blob/5.7.4/docs/generated/protocol.md) and [coder/websocket Go API](https://pkg.go.dev/github.com/coder/websocket). Tests use real local websocket v5 mocks, not native OBS. A 30-sample Linux loopback mock run on October 29, 2026 measured mean 14.75ms and p95 17.33ms with a pre-encoded 1280x720 JPEG, without the race detector. This includes local protocol exchange, JPEG decode and RGBA conversion but excludes OBS rendering, GPU readback and JPEG encoding; it does not measure native capture speed or prove native 5fps. Native Windows GUI, OBS rendering/filter behavior and in-game input still require a smoke test on Windows.

Inference matches this test checkpoint's stage-1 dataset. Each request contains exactly one image-only user message, no system prompt, goals, action history or invented instructions. The server must use the checkpoint's official Qwen3-VL ChatML template. The runner captures RGB1280x720 and adds sixteen opaque black rows at the bottom, producing1280x736. Do not let the server resize this image again. The expected image grid is1x46x80 and920 image tokens. Training images and endpoint transport are lossless PNG. OBS capture is JPEG-decoded first, so the OBS path adds lossy compression before the same PNG endpoint transport.

The parser requires intact <|action_start|> and <|action_end|> delimiters, optional <use_absolute>, three integers and exactly six semicolon-separated key chunks. It validates quantization, frame bounds, the training key vocabulary and a four-key limit. OS task-switch/termination shortcuts are rejected. Relative movement and wheel totals are distributed over six chunks across200ms; absolute menu coordinates map from1280x720 into the target client and Windows virtual desktop. Hardware scancodes support held keys and releases. Predicted input is always released before network inference, deliberately sacrificing continuous movement for safety. Windows pointer acceleration and game sensitivity still affect relative camera motion.

Only use a trusted endpoint. Screenshots are sent to it, and plain HTTP transmits API keys unencrypted outside an encrypted tunnel. Prefer HTTPS or an SSH localhost forward. Credentials stay in memory, are not saved, and redirects are refused. Optional environment defaults are LUMINE_ENDPOINT and LUMINE_API_KEY.

Serving the saved test checkpoint

The test export lives in the private HF bucket, not a model repository. No GPU is provisioned by this project. On an already-authorized Linux CUDA host, use an isolated environment and a current vLLM release supporting Qwen3-VL and your GPU. Installing vLLM can replace torch, so do not install into a training environment.

```sh
python -m venv .venv-serve
. .venv-serve/bin/activate
pip install -U vllm huggingface_hub
hf auth login
hf buckets sync hf://buckets/koboshchan/checkpoints/lumine/test-2026-10-29/ ./lumine-test
python /path/to/Lumine/Runner/verify_processor.py ./lumine-test
export LUMINE_SERVE_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
vllm serve ./lumine-test --served-model-name lumine-mini-2b \
  --host 127.0.0.1 --port 8000 --api-key "$LUMINE_SERVE_KEY" \
  --dtype bfloat16 --max-model-len 2048 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"do_resize":false}'
```

Keep the saved tokenizer, chat_template and all37 custom action tokens together with the model. Do not substitute the base model tokenizer, a generic ChatML override, a reasoning parser or token-stripping middleware. The request sends vLLM's skip_special_tokens=false extension; other OpenAI-compatible servers must implement equivalent token-preserving decoding. Missing delimiters fail closed instead of guessing actions. The preflight checks the local processor, not vLLM's live engine; validate an actual dry-run response before enabling input. These serving commands have not been GPU-tested as part of the runner build.

For Colab, choose an already-approved CUDA GPU runtime manually, use /content/lumine-test as the download destination, clone/open this private repo with your own credentials and run the same commands there. Run vLLM in a terminal or subprocess so the notebook remains usable. Forward8000 to Windows through an authenticated SSH tunnel on your existing private network, for example ssh -N -L8000:127.0.0.1:8000 USER@COLAB_PRIVATE_ADDRESS. Enter http://127.0.0.1:8000/v1 in the runner. Do not expose an unauthenticated public port or print HF/API tokens into notebook outputs. Shut down the runtime manually when finished.

Tests cover padding, request shape/token preservation, redirect refusal, strict action parsing, real dataset examples, chunk deadlines, cancellation and release. Windows amd64 cross-compilation and vet pass; native Windows GUI, capture, hotkey and actual game input require a Windows smoke test. No desktop, encoder, WSL or GPU runtime was changed by this build.

Session proxy mode

Enable Session proxy mode to use the authenticated GPU-local proxy described in SessionProxy/README.md. Point the endpoint at the proxy's /v1 address and enter LUMINE_PROXY_SECRET in the API key box. The model field is ignored in this mode; the proxy owner configures the served model. Each step uploads only the newest padded frame, a random session/request id and, only after successful real input playback, the last executed action with its matching server action id. Dry-run predictions are never reported as executed. Start, stop and failures reset session continuity, and ambiguous requests are not retried. Server memory does not change the current checkpoint's training protocol: its default image-only mode stays single-frame. Tiered memory requires an explicitly compatible future checkpoint and the shared exact-processor adapter. This rebuild has CPU/network-fixture and cross-build evidence only, not live gameplay or GPU acceptance.
