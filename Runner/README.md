Lumine Windows runner

A Windows x64 native Go GUI, with no Python runtime, browser, installer, cgo or external DLLs beyond Windows system libraries. Build from Linux with Go 1.23 or newer.

```sh
cd Runner
go test -race ./...
GOOS=windows GOARCH=amd64 go vet ./...
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags='-s -w -H=windowsgui' -o LumineRunner.exe .
```

Launch LumineRunner.exe, enter the /v1 endpoint, served model name and API key, then pick the Genshin window. Dry run is enabled by default. Press Start and focus the game during the five-second countdown. F8 works globally and cancels inference, stops execution and releases held inputs. Stop, closing the app, target focus loss, malformed output and endpoint errors also stop execution. If another app owns F8, the runner refuses to start.

The GUI shows the latest frame, raw predicted action, inference latency and actual decision rate. Capture samples at 5Hz with one latest-only slot; inference and playback are serialized, so slow endpoints reduce decision rate rather than queueing actions. Dry run parses and displays output but never calls SendInput.

Use a visible, unobstructed 16:9 windowed or borderless game. Native 1280x720 is the exact training geometry; larger 16:9 client areas are downscaled. This version uses GDI screen capture, not Windows Graphics Capture. Exclusive fullscreen, protected/flip-model surfaces, overlays and occlusion can produce incorrect or black frames. Check the preview in dry run before enabling input. No anti-cheat bypass is included. SendInput can be rejected by elevated games or game protections; this has not been validated inside Genshin.

The GUI uses physical-pixel native controls and is not a responsive/per-monitor-scaled layout. It is an initial runner, not a polished installer.

Inference matches this test checkpoint's stage-1 dataset. Each request contains exactly one image-only user message, no system prompt, goals, action history or invented instructions. The server must use the checkpoint's official Qwen3-VL ChatML template. The runner captures RGB1280x720 and adds sixteen opaque black rows at the bottom, producing1280x736. Do not let the server resize this image again. The expected image grid is1x46x80 and920 image tokens. Training images and transport are lossless PNG.

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
