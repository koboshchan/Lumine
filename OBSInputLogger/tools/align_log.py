"""Map the logger's monotonic input receipt times onto actual encoded video PTS.

No dependencies. FRAME timing comments come from OBS's packet callback, not the
recording-start notification or the video's filesystem creation timestamp.
"""
from __future__ import annotations

import argparse
import bisect
import os
from pathlib import Path

FILETIME_EPOCH = 116444736000000000


def align(source: Path, destination: Path, video_start_filetime: int) -> int:
    if source.resolve() == destination.resolve():
        raise ValueError("The aligned destination must not replace the raw sidecar")
    # First pass retains only per-frame timing; input events are streamed on pass 2.
    marker = Path(str(source) + ".complete")
    expected_marker = f"LUMINE_COMPLETE_V1 {source.stat().st_size}\n"
    if not marker.exists() or marker.read_text(encoding="ascii") != expected_marker:
        raise ValueError("Missing/mismatched completion artifact; capture may be truncated")
    origin = {}
    boundaries = []
    frames = []
    complete = False
    previous_raw = -1
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "# COMPLETE":
                complete = True
            elif complete and line.strip():
                raise ValueError("Data after completion footer")
            if not line.startswith("#") and line.strip():
                raw_ts = int(line.split(",", 1)[0])
                if raw_ts < previous_raw:
                    raise ValueError("Input event time went backwards")
                previous_raw = raw_ts
            if line.startswith("# ERROR,"):
                raise ValueError("Capture reported an error; refusing incomplete data")
            if line.startswith("# start_filetime,"):
                origin["wall"] = int(line.split(",")[1])
            elif line.startswith("# start_monotonic_ns,"):
                origin["mono"] = int(line.split(",")[1])
            elif line.startswith("# FRONTEND_EVENT,"):
                _, ns, action = line.strip().split(",")
                boundaries.append((int(ns), action))
            elif line.startswith("# FRAME,"):
                _, pts, num, den, cts, _ = line.strip().split(",")
                pts, num, den, cts = map(int, (pts, num, den, cts))
                if den <= 0 or num <= 0:
                    raise ValueError("Invalid video timebase")
                if cts == 0:
                    raise ValueError("Encoder supplied no composition time; exact mapping unavailable")
                if pts >= 0:
                    frames.append((cts, pts * num * 10_000_000 // den))
    if not complete:
        raise ValueError("Missing clean completion footer; capture may be truncated")
    if not frames or len(origin) != 2:
        raise ValueError("No usable OBS frame timing/origin in this log")
    frames.sort()
    if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(frames, frames[1:])):
        raise ValueError("Non-monotonic/duplicate frame timing; unsupported recording mode")
    composition_times = [frame[0] for frame in frames]
    last_duration = frames[-1][1] - frames[-2][1] if len(frames) > 1 else 0
    # Use fps for the final frame, where no next packet bounds its duration.
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# fps,"):
                _, num, den = line.strip().split(",")
                if int(num) > 0 and int(den) > 0:
                    last_duration = 10_000_000 * int(den) // int(num)
                break
    if last_duration <= 0:
        raise ValueError("No final frame duration")
    pending = {}  # Initial SCREEN/cursor/key state before first known frame.
    written = 0
    previous_ts = -1
    paused = False
    # Exclusive destination: neither raw data nor a previous alignment is overwritten.
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write("# Lumine OBS aligned log v1; paused video time already removed\n")
        output.write(f"# video_start_filetime,{video_start_filetime}\n")
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("#") or not line.strip():
                    continue
                raw_ts, event = line.rstrip("\n").split(",", 1)
                ns = origin["mono"] + (int(raw_ts) - origin["wall"]) * 100
                if event == "PAUSE":
                    paused = True
                    continue
                if event == "RESUME":
                    paused = False
                    continue
                if event == "STOP" or paused:
                    continue
                # Authoritative frontend receipt boundaries, independent of worker lag.
                state = "RESUME"
                for boundary_ns, action in boundaries:
                    if boundary_ns > ns:
                        break
                    state = action
                if state in ("PAUSE", "STOP"):
                    continue
                index = bisect.bisect_right(composition_times, ns) - 1
                kind = event.split(",", 1)[0]
                state_key = event if event in (
                    "MOUSE,SHOW", "MOUSE,HIDE", "MOUSE,LOCK", "MOUSE,UNLOCK"
                ) else kind
                if state_key in ("MOUSE,SHOW", "MOUSE,HIDE"):
                    state_key = "visibility"
                if state_key in ("MOUSE,LOCK", "MOUSE,UNLOCK"):
                    state_key = "lock"
                if index < 0:
                    if kind in ("SCREEN", "KEY_CHUNK", "MOUSE_ABS") or state_key in ("visibility", "lock"):
                        pending[state_key] = event
                    continue
                cts, pts = frames[index]
                if ns >= cts + last_duration * 100:
                    continue  # Pause/stop gap: never attach input to a nonexistent frame.
                duration = (frames[index + 1][1] - pts) if index + 1 < len(frames) else last_duration
                # Clamp to the selected frame. Never spill a stopped/paused wall-clock
                # gap into the next frame or invent a nonexistent output frame.
                offset = min(max(0, (ns - cts) // 100), max(0, duration - 1))
                mapped = video_start_filetime + pts + offset
                if mapped < previous_ts:
                    raise ValueError("Input event time went backwards")
                for initial in pending.values():
                    output.write(f"{video_start_filetime + frames[0][1]},{initial}\n")
                    written += 1
                pending.clear()
                output.write(f"{mapped},{event}\n")
                previous_ts = mapped
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--log", type=Path, help="Raw sidecar (defaults to video basename.log)")
    parser.add_argument("--output", type=Path, help="New aligned log path")
    parser.add_argument("--video-start-filetime", type=int,
                        help="Override FILETIME origin for a metadata-aware consumer/test")
    args = parser.parse_args()
    if args.video_start_filetime is None and os.name != "nt":
        parser.error("Run on Windows for DataProcessor's creation-time origin, or explicitly set --video-start-filetime")
    origin = args.video_start_filetime
    if origin is None:
        origin = FILETIME_EPOCH + args.video.stat().st_ctime_ns // 100
    source = args.log or args.video.with_suffix(".log")
    destination = args.output or args.video.with_suffix(".aligned.log")
    count = align(source, destination, origin)
    print(f"Wrote {count} events to {destination}")


if __name__ == "__main__":
    main()
