Each 200ms action still has one movement/scroll prefix and six 33ms key chunks, with at most four keys per chunk.

Camera mode uses the existing form `<|action_start|>dX dY dZ ; K1 ; K2 ; K3 ; K4 ; K5 ; K6<|action_end|>`. Relative deltas retain the existing quantization and clamps. Scroll remains Windows wheel delta divided by 120, rounded and clamped to [-5, 5].

Menu mode uses `<|action_start|><use_absolute> X Y dZ ; K1 ; K2 ; K3 ; K4 ; K5 ; K6<|action_end|>`. X and Y are integer target pixels in the output frame, normally 1280x720, clamped to [0, 1279] and [0, 719]. They are not relative displacements. A known LOCK or HIDE forces camera mode. Otherwise an explicit UNLOCK or SHOW enables menu mode when a position and capture bounds are available.

The last known cursor mode and absolute target before the end of the half-open action window select its movement prefix. State and position carry forward through motionless frames. An action has only one movement prefix, so a mode change within a window uses the final mode. Older logs without cursor-mode events stay relative.

The logger keeps the existing FILETIME CSV event names. Every raw mouse event, including button-only and wheel events, emits both `MOUSE_REL,dx,dy` and `MOUSE_ABS,x,y`, even while clipped/hidden. REL is raw device displacement for relative-input mice; absolute-input devices use successive desktop cursor pixel differences. ABS comes from GetCursorPos. If Windows cannot query the cursor position, the logger preserves REL/buttons/wheel without fabricating ABS.

The logger samples `MOUSE,LOCK|UNLOCK` and `MOUSE,SHOW|HIDE` on raw input and on the existing 30Hz timer. Startup and resume include state and position snapshots, including when the pointer is stationary. Polling detects sampled state changes, not a hook into ClipCursor/ShowCursor, so a transient between samples with no raw input can be missed. Pause still suppresses recording.

The new `SCREEN,left,top,width,height` event describes the primary desktop capture bounds in physical pixels. Record the full primary screen for that default mapping. For a cropped video, another monitor, or an older log without SCREEN, pass `--source-left`, `--source-top`, `--source-width` and `--source-height` to describe the actual recording bounds in desktop pixels. These override SCREEN. Output `--width` and `--height` control normalization and frame extraction together. Without capture geometry, processing falls back to relative actions rather than guessing.

The token is registered alongside the existing action tokens in VeOmni's training tokenizer list, enabled by `model.add_custom_tokens` in the stage 1 config. Initialize or update the pinned submodule with `git submodule update --init` before training or testing. Later stages inherit the saved tokenizer from stage 1.

Run the synthetic-log regression tests from the repository root with `python3 -m unittest discover -s tests -v`. They require the processor's existing tqdm dependency but no model, video, GPU or Windows input injection. The logger requires a separate Windows build and live recording check; Python tests do not establish Windows runtime behavior.
