#!/usr/bin/env python3
"""
DataProcessor - Convert KeyRecorder logs + video to Lumine training format

Usage:
    python DataProcessor.py --log input_log.txt --video input.mkv --output dataset/
"""

import argparse
import os
import re
import subprocess
import json
import bisect
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import shutil
from tqdm import tqdm


@dataclass
class KeyChunkEvent:
    timestamp: int  # 100-nanosecond intervals (FILETIME)
    keys: List[str]


@dataclass
class MouseEvent:
    timestamp: int
    event_type: str  # ABS, REL, WHEEL
    x: int = 0
    y: int = 0
    dx: int = 0
    dy: int = 0
    delta: int = 0


@dataclass
class ActionChunk:
    """One 33ms chunk of actions"""

    dx: int = 0  # mouse X movement
    dy: int = 0  # mouse Y movement
    scroll: int = 0  # scroll amount
    keys: List[str] = field(default_factory=list)


@dataclass
class ActionFrame:
    """200ms (6 chunks) of actions at 5fps"""

    chunks: List[ActionChunk]

    def to_lumine_format(self) -> str:
        # Format: <|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>

        # Mouse Movement: First, specify the relative displacement X, Y and scroll amount Z
        # The paper says: "discretize mouse movement values using units of 5 pixels along the X-axis and 4 pixels along the Y-axis."
        total_dx = sum(c.dx for c in self.chunks)
        total_dy = sum(c.dy for c in self.chunks)
        total_scroll = sum(c.scroll for c in self.chunks)

        # Paper: dX/dY integers in (-1000, 1000), scroll steps in [-5, 5].
        # Windows WHEEL_DELTA is 120 per notch, so convert raw delta to steps.
        dx = max(-995, min(995, int(round(total_dx / 5) * 5)))
        dy = max(-996, min(996, int(round(total_dy / 4) * 4)))
        z = max(-5, min(5, int(round(total_scroll / 120))))

        key_parts = []
        for chunk in self.chunks:
            # Each chunk can contain up to 4 keys.
            keys = chunk.keys[:4]
            key_parts.append(" ".join(keys))

        return f"<|action_start|>{dx} {dy} {z} ; {' ; '.join(key_parts)}<|action_end|>"


class KeyRecorderParser:
    """Parse KeyRecorder log file"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.key_chunks: List[KeyChunkEvent] = []
        self.mouse_events: List[MouseEvent] = []
        self.start_timestamp: Optional[int] = None
        self.pause_ranges: List[
            Tuple[int, int]
        ] = []  # List of (pause_start, resume_time) tuples

    def parse(self):
        """Parse the log file"""
        if not os.path.exists(self.log_path):
            print(f"Error: Log file {self.log_path} not found")
            return self

        pause_start = None
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])
                except ValueError:
                    continue

                # Set start timestamp
                if self.start_timestamp is None:
                    self.start_timestamp = timestamp

                event_type = parts[1]

                # Handle PAUSE/RESUME events
                if event_type == "PAUSE":
                    pause_start = timestamp
                    continue
                elif event_type == "RESUME":
                    if pause_start is not None:
                        self.pause_ranges.append((pause_start, timestamp))
                        pause_start = None
                    continue

                if event_type == "KEY_CHUNK":
                    keys = parts[2].split() if len(parts) > 2 else []
                    self.key_chunks.append(KeyChunkEvent(timestamp, keys))

                elif event_type == "MOUSE_ABS":
                    x = int(parts[2]) if len(parts) > 2 else 0
                    y = int(parts[3]) if len(parts) > 3 else 0
                    self.mouse_events.append(MouseEvent(timestamp, "ABS", x=x, y=y))

                elif event_type == "MOUSE_REL":
                    dx = int(parts[2]) if len(parts) > 2 else 0
                    dy = int(parts[3]) if len(parts) > 3 else 0
                    evt = MouseEvent(timestamp, "REL", dx=dx, dy=dy)
                    self.mouse_events.append(evt)
                    # Debug output for first few REL events
                    if len(self.mouse_events) <= 5 and (dx != 0 or dy != 0):
                        print(
                            f"DEBUG: Parsed MOUSE_REL - timestamp={timestamp}, dx={dx}, dy={dy}, stored event: dx={evt.dx}, dy={evt.dy}"
                        )

                elif event_type == "MOUSE":
                    if len(parts) > 2 and parts[2] == "WHEEL":
                        delta = int(parts[3]) if len(parts) > 3 else 0
                        self.mouse_events.append(
                            MouseEvent(timestamp, "WHEEL", delta=delta)
                        )

        # Handle unclosed pause (pause without resume at end of log)
        if pause_start is not None:
            # Set end to a very large timestamp
            self.pause_ranges.append((pause_start, 2**63 - 1))

        # Sort by timestamp
        self.key_chunks.sort(key=lambda x: x.timestamp)
        self.mouse_events.sort(key=lambda x: x.timestamp)

        # Precompute timestamps for binary search
        self.key_timestamps = [e.timestamp for e in self.key_chunks]
        self.mouse_timestamps = [e.timestamp for e in self.mouse_events]

        print(
            f"Parsed {len(self.key_chunks)} key chunks, {len(self.mouse_events)} mouse events, {len(self.pause_ranges)} pause ranges"
        )
        return self

    def is_paused(self, timestamp: int) -> bool:
        """Check if a timestamp falls within a paused range"""
        for pause_start, resume_time in self.pause_ranges:
            if pause_start <= timestamp < resume_time:
                return True
        return False

    def get_actions_at_time(
        self, start_time: int, duration_ms: int = 200
    ) -> ActionFrame:
        """Get actions for a time window (default 200ms for 1 frame at 5fps)"""
        end_time = start_time + (duration_ms * 10000)

        chunk_duration_100ns = int((duration_ms / 6) * 10000)

        chunks = []
        for i in range(6):
            c_start = start_time + (i * chunk_duration_100ns)
            c_end = c_start + chunk_duration_100ns

            # Binary search for mouse events in range
            mouse_start_idx = bisect.bisect_left(self.mouse_timestamps, c_start)
            mouse_end_idx = bisect.bisect_right(self.mouse_timestamps, c_end - 1)

            dx = dy = scroll = 0
            rel_count = 0
            for j in range(mouse_start_idx, mouse_end_idx):
                e = self.mouse_events[j]
                if e.event_type == "REL":
                    dx += e.dx
                    dy += e.dy
                    if e.dx != 0 or e.dy != 0:
                        rel_count += 1
                elif e.event_type == "WHEEL":
                    scroll += e.delta

            # Debug: show if we found REL events but got 0 movement
            if rel_count > 0 and dx == 0 and dy == 0 and i == 0:
                print(
                    f"WARNING: Found {rel_count} REL events in chunk {i} but total movement is 0"
                )

            # Binary search for key events - find last key before c_end
            key_idx = bisect.bisect_right(self.key_timestamps, c_end - 1)
            if key_idx > 0:
                keys = self.key_chunks[key_idx - 1].keys
            else:
                keys = []

            chunks.append(ActionChunk(dx=dx, dy=dy, scroll=scroll, keys=keys))

        return ActionFrame(chunks=chunks)


class VideoProcessor:
    """Extract frames from video using FFmpeg"""

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fps: int = 5,
        width: int = 1280,
        height: int = 720,
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.width = width
        self.height = height

    def extract_frames(self) -> List[Path]:
        """Extract frames at specified fps and resolution"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if frames already exist
        existing_frames = sorted(self.output_dir.glob("frame_*.png"))
        if existing_frames:
            print(f"Found {len(existing_frames)} existing frames, skipping extraction")
            return existing_frames

        cmd = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            self.video_path,
            "-vf",
            f"fps=5,scale_cuda={self.width}:{self.height},hwdownload,format=nv12",
            "-c:v",
            "png",
            "-start_number",
            "0",
            "-y",
            str(self.output_dir / "frame_%05d.png"),
        ]

        print(f"Extracting frames: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return []
        except FileNotFoundError:
            print("Error: ffmpeg not found.")
            return []

        frames = sorted(self.output_dir.glob("frame_*.png"))
        print(f"Extracted {len(frames)} frames")

        return frames

    def get_video_info(self) -> Tuple[int, int]:
        """Get video start and end timestamps in 100ns units"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "csv=p=0",
            self.video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            start_sec = float(result.stdout.strip())
            start_100ns = int(start_sec * 10000000)
        except (FileNotFoundError, ValueError):
            start_100ns = 0

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration_sec = float(result.stdout.strip())
            duration_100ns = int(duration_sec * 10000000)
            end_100ns = start_100ns + duration_100ns
        except (FileNotFoundError, ValueError):
            end_100ns = 0

        return start_100ns, end_100ns


class LumineDataset:
    """Create Lumine-compatible dataset"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict] = []

    def add_sample(
        self,
        frame_idx: int,
        frame_path: Path,
        action: str,
        instruction: Optional[str] = None,
        thought: Optional[str] = None,
    ):
        """Add a training sample"""
        sample = {"frame_idx": frame_idx, "image": frame_path.name, "action": action}

        if instruction:
            sample["instruction"] = instruction
        if thought:
            sample["thought"] = thought

        self.samples.append(sample)

    def save(self):
        """Save dataset metadata"""
        with open(self.output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for sample in self.samples:
                metadata_entry = {
                    "file_name": sample["image"],
                    "text": sample["action"],
                }
                f.write(json.dumps(metadata_entry, ensure_ascii=False) + "\n")

        # Stage 1: Pretrain (image -> action)
        with open(
            self.output_dir / "stage1_pretrain.jsonl", "w", encoding="utf-8"
        ) as f:
            for sample in self.samples:
                out = {
                    "images": [f"frames/{sample['image']}"],
                    "text": sample["action"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KeyRecorder logs to Lumine training format"
    )
    parser.add_argument("--log", required=True, help="Path to log file or directory")
    parser.add_argument(
        "--video", required=True, help="Path to video file or directory"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="FPS (default: 5)")
    parser.add_argument("--width", type=int, default=1280, help="Width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Height (default: 720)")
    parser.add_argument(
        "--offset", type=int, default=0, help="Timestamp offset (100ns)"
    )
    parser.add_argument("--skip-video", action="store_true", help="Skip extraction")

    args = parser.parse_args()

    log_path = Path(args.log)
    video_path = Path(args.video)
    output_base = Path(args.output)

    # Simple single-pair logic for brevity in this tool call
    if not log_path.is_dir():
        log_files = [log_path]
        video_files = [video_path]
    else:
        log_files = sorted(log_path.glob("*.txt"))
        video_files = [video_path / f"{f.stem}.mkv" for f in log_files]

    for log_f, video_f in tqdm(
        zip(log_files, video_files), total=len(log_files), desc="Processing pairs"
    ):
        if not video_f.exists():
            continue

        print(f"\nProcessing: {log_f.name}")
        output_dir = output_base / log_f.stem

        parser_obj = KeyRecorderParser(str(log_f)).parse()

        video_proc = VideoProcessor(
            str(video_f),
            str(output_dir / "frames"),
            fps=args.fps,
            width=args.width,
            height=args.height,
        )

        frames = []
        if not args.skip_video:
            frames = video_proc.extract_frames()
        else:
            frames = sorted((output_dir / "frames").glob("frame_*.png"))

        if not frames:
            continue

        video_start, video_end = video_proc.get_video_info()
        video_start = video_start if video_start else 0

        # Convert video PTS to absolute FILETIME using file creation time
        video_ctime = os.path.getctime(str(video_f))
        video_ctime_filetime = int(video_ctime * 10000000) + 116444736000000000
        # Video absolute start = file creation time + video PTS offset
        video_abs_start = video_ctime_filetime + video_start

        # Key recorder time range
        key_start = parser_obj.start_timestamp if parser_obj.start_timestamp else 0
        key_end = (
            parser_obj.key_chunks[-1].timestamp if parser_obj.key_chunks else key_start
        )

        # Calculate video absolute end time
        frame_interval_100ns = 10000000 // args.fps
        video_frames_count = len(frames)
        video_abs_end = video_abs_start + (video_frames_count * frame_interval_100ns)

        # Find overlap between video and key timestamps
        overlap_start = max(video_abs_start, key_start)
        overlap_end = min(video_abs_end, key_end)

        print(
            f"Video: [{video_abs_start}, {video_abs_end}] ({video_abs_start / 10000000:.2f}s - {video_abs_end / 10000000:.2f}s Unix)"
        )
        print(
            f"Keys: [{key_start}, {key_end}] ({key_start / 10000000:.2f}s - {key_end / 10000000:.2f}s Unix)"
        )
        print(f"Overlap: [{overlap_start}, {overlap_end}]")

        if overlap_start >= overlap_end:
            print("Warning: No overlapping time range")
            continue

        dataset = LumineDataset(str(output_dir))

        valid_count = 0
        skipped_paused = 0
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Frame absolute time = video absolute start + frame offset
            frame_time = video_abs_start + (i * frame_interval_100ns)

            # Skip frames outside overlap range
            if frame_time < overlap_start or frame_time >= overlap_end:
                continue

            # Skip frames that fall within paused time ranges
            if parser_obj.is_paused(frame_time):
                skipped_paused += 1
                continue

            action_frame = parser_obj.get_actions_at_time(frame_time, duration_ms=200)
            dataset.add_sample(
                frame_idx=valid_count,
                frame_path=frame,
                action=action_frame.to_lumine_format(),
            )
            valid_count += 1

        print(
            f"Valid frames: {valid_count}/{len(frames)} (skipped {skipped_paused} paused frames)"
        )

        dataset.save()
        print(f"Done: {log_f.stem}")


if __name__ == "__main__":
    main()
