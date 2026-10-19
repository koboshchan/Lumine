Lumine OBS Input Logger

A Windows OBS Studio 32+ plugin for recording your own gameplay inputs beside your video. Logging is off each time OBS opens. Open Tools > Lumine input logging... and explicitly enable it before recording. The confirmation also shows the current setting. It never records outside an OBS recording session, uploads anything, injects inputs, or installs a keyboard hook.

Starting a recording creates a same-basename .log next to the video. Existing logs are never overwritten. Stopping recording flushes the log and publishes a .log.complete success artifact containing its byte count. Keep those two files together. A missing or mismatched completion artifact means the capture is incomplete. Pausing OBS pauses inputs too. The logger stops collecting during recording shutdown, before muxer drain. Change the toggle between recordings.

Input capture is system-wide while enabled and recording, including other apps. Close private chats and password fields first. Use this only on your own machine or with explicit consent. Raw Input runs on a dedicated message-only Windows window thread. It does not use RIDEV_NOLEGACY or block normal input. Because Raw Input registration is process-wide, capture refuses to start if OBS or another plugin already owns keyboard/mouse registrations. Run it without conflicting plugins. It restores/removes only its own registrations on shutdown.

Build

Use Visual Studio 2022 or newer with Desktop development with C++, a Windows 10/11 SDK, CMake 3.28+, and the x64 OBS 32 development SDK/build exports. Both libobs and obs-frontend-api CMake packages must be available. An installed OBS runtime alone does not include these packages. This uses the same OBS::libobs and OBS::obs-frontend-api imported targets as obs-plugintemplate, without Qt or a separate UI toolkit.

From the repository root in a Visual Studio x64 developer terminal, substitute your SDK/build export prefix below.

~~~powershell
cmake -S OBSInputLogger -B OBSInputLogger/build -A x64 -DCMAKE_PREFIX_PATH="C:/path/to/obs-sdk"
cmake --build OBSInputLogger/build --config Release
cmake --install OBSInputLogger/build --config Release --prefix OBSInputLogger/package
~~~

With OBS closed, copy package/obs-plugins/64bit/lumine-input-logger.dll into the matching obs-plugins/64bit directory of a portable OBS installation, or package it in the standard OBS plugin layout. Do not mix x86 and x64 builds. The plugin has no data assets. Reopen OBS and look for Lumine input logging... in Tools. To remove it, close OBS and remove that DLL.

Timing and training data

Every event uses KeyRecorder's timestamp,event,data CSV schema and token names. Timestamps are FILETIME-compatible 100 ns units anchored once to os_gettime_ns(), so changing the wall clock cannot create jumps. The header records the monotonic and FILETIME anchors, the current OBS video-frame timestamp, exact rational fps, output/base resolution, and primary-screen physical-pixel bounds.

The recording-start frontend notification is not assumed to be the first encoded video frame. For each encoded video packet on track zero, the log includes a # FRAME comment with output-adjusted PTS, timebase numerator/denominator, composition time in OBS monotonic ns, and system DTS in microseconds. This comes from OBS's packet callback. Its encoder thread only enqueues timing metadata; the input thread writes the file. The recording output reference stays alive until the callback has been disconnected and the worker joined.

Input timestamps are message receipt times, not hardware acquisition times. Frame timing maps them to actual encoded frames, but cannot eliminate Windows scheduling latency or capture/render latency. No claim of zero-latency physical synchronization is made. Encoders without composition timestamps fail closed during alignment instead of silently guessing from filesystem time. The fps/resolution header describes OBS output settings; custom encoder rescaling/frame-rate divisors may differ and are not supported for training.

Regular held-key chunks target 30 Hz, with additional chunks at key/button transitions so short presses are retained. Cursor state is polled roughly every 5 ms; Windows timers are not realtime. Raw events also poll cursor state, and periodic samples are serviced even during high-rate mouse traffic. No fake catch-up samples are generated after a stall.

MOUSE_REL and MOUSE_ABS are both emitted for every raw mouse packet and periodic sample, even when locked or hidden. Relative-only HID motion is not synthesized from absolute coordinates. Absolute-only devices emit zero relative motion and a physical desktop cursor sample. MOUSE,WHEEL uses the signed Windows raw wheel units, normally 120 per notch; DataProcessor normalizes these. The original KeyRecorder HIWORD wheel bug is not copied. KEY_CHUNK uses the existing LB/RB/MB/XB1/XB2, letters, number/function-key words and modifier tokens. F5/F11/F12 are normal logged keys; only OBS controls recording. SCREEN, MOUSE,LOCK/UNLOCK, MOUSE,SHOW/HIDE, PAUSE and RESUME are included. LOCK means ClipCursor confinement, not an infallible game relative-mode detector; games may hide/recenter without clipping.

Using existing DataProcessor

The current DataProcessor ignores comment metadata and estimates the video origin from Windows file creation time. Passing the raw sidecar directly would therefore discard the precise synchronization. First run the bundled alignment tool on Windows using the final, unmodified recording file, then use the .aligned.log as the input log for DataProcessor.

~~~powershell
python OBSInputLogger/tools/align_log.py "D:/recordings/game.mkv"
~~~

It creates game.aligned.log without overwriting anything. It retains the raw log and completion artifact. The aligner uses composition timestamps and encoded PTS, removes paused wall time, rejects incomplete captures or missing timing, and discards inputs outside recorded frame intervals. Initial screen/cursor/key state is carried to the first available frame. Input queued across a resume boundary is discarded conservatively and current state reseeded. The raw frontend receipt boundaries remain authoritative even if the worker is delayed.

The aligned log uses a creation-time FILETIME origin only as an adapter for DataProcessor's existing convention. If the video is moved to a filesystem that changes creation time, remuxed, trimmed, or edited, align again against that file only if its PTS still matches the original. For a metadata-aware consumer, --video-start-filetime sets a deliberate origin. Raw packet timing remains the authoritative record.

Use one local video file per recording. Automatic/manual file splitting, replay buffers, live streams, multi-track video, custom FFmpeg/raw outputs, and auto-remux workflows are not supported. Disable file splitting and auto-remux for datasets. Logging begins after RECORDING_STARTED, so frames before that callback have no inputs; the aligner does not invent them.

Verification

~~~powershell
python -m unittest discover -s OBSInputLogger/tests -v
~~~

Sixteen alignment tests cover emitted header parsing, both mouse modes, B-frame packet reordering, rational frame timing, pause gaps, authoritative pause boundaries, final-frame cutoff, initial state, missing composition time, truncated/error logs, completion artifacts, backwards timestamps and overwrite protection.

The C++ source has been compiled and linked as an x64 Windows DLL with MSVC 19.50 against OBS 32 public headers and actual installed OBS DLL exports. The normal CMake build is also checked against those imported SDK targets. No OBS install or user recording was modified during development. A real OBS recording session is still required to verify runtime loading, start/stop, pause/resume, cursor states and video alignment with your chosen encoder. Build success is not a runtime acceptance result.
