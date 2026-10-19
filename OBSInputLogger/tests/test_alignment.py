import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "align_log", Path(__file__).resolve().parents[1] / "tools" / "align_log.py"
)
align_log = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(align_log)

WALL = 130000000000000000
MONO = 10_000_000_000
VIDEO = 140000000000000000


class AlignmentTest(unittest.TestCase):
    def fixture(self, frames=None, events=None, extra=""):
        frames = frames or [(0, MONO), (1, MONO + 33_333_333), (2, MONO + 66_666_666)]
        events = events or [(0, "SCREEN,0,0,1920,1080"), (10000, "MOUSE_REL,8,-3"),
                            (10000, "MOUSE_ABS,640,360"), (333334, "KEY_CHUNK,W Shift")]
        text = (f"# start_filetime,{WALL}\n# start_monotonic_ns,{MONO}\n# fps,30,1\n# FRAME_SCHEMA,pts,timebase_num,timebase_den,composition_ns,system_dts_us\n"
                + "".join(f"# FRAME,{pts},1,30,{cts},0\n" for pts, cts in frames)
                + extra
                + "".join(f"{WALL + dt},{event}\n" for dt, event in events) + "# COMPLETE\n")
        return text

    def run_alignment(self, text):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "raw.log", Path(temp) / "aligned.log"
            source.write_text(text)
            Path(str(source) + ".complete").write_text(f"LUMINE_COMPLETE_V1 {source.stat().st_size}\n")
            count = align_log.align(source, output, VIDEO)
            return count, output.read_text()

    def test_pts_and_both_mouse_modes(self):
        count, text = self.run_alignment(self.fixture())
        self.assertEqual(count, 4)
        self.assertIn(f"{VIDEO + 10000},MOUSE_REL,8,-3", text)
        self.assertIn(f"{VIDEO + 10000},MOUSE_ABS,640,360", text)
        self.assertIn(f"{VIDEO + 333333},KEY_CHUNK,W Shift", text)

    def test_pause_removed_using_actual_pts_not_wall_gap(self):
        frames = [(0, MONO), (1, MONO + 33_333_333),
                  (2, MONO + 5_066_666_666), (3, MONO + 5_099_999_999)]
        events = [(0, "KEY_CHUNK,W"), (400000, "PAUSE"),
                  (1000000, "KEY_CHUNK,B"), (50666667, "RESUME"),
                  (50666668, "KEY_CHUNK,A")]
        count, text = self.run_alignment(self.fixture(frames, events))
        self.assertEqual(count, 2)
        self.assertNotIn("PAUSE", text)
        self.assertNotIn("KEY_CHUNK,B", text)
        self.assertIn(f"{VIDEO + 666667},KEY_CHUNK,A", text)

    def test_frame_map_sorted_for_b_frames(self):
        ordered = self.fixture()
        reversed_frames = [(2, MONO + 66_666_666), (0, MONO), (1, MONO + 33_333_333)]
        self.assertEqual(self.run_alignment(ordered), self.run_alignment(self.fixture(reversed_frames)))

    def test_initial_state_carried_not_initial_motion(self):
        events = [(-100, "SCREEN,0,0,1920,1080"), (-100, "MOUSE,SHOW"),
                  (-100, "MOUSE,UNLOCK"), (-100, "MOUSE_REL,99,99"), (1, "KEY_CHUNK,W")]
        _, text = self.run_alignment(self.fixture(events=events))
        self.assertIn(f"{VIDEO},MOUSE,UNLOCK", text)
        self.assertNotIn("99,99", text)

    def test_failure_missing_frame_timing(self):
        with self.assertRaisesRegex(ValueError, "No usable"):
            self.run_alignment(f"# start_filetime,{WALL}\n# start_monotonic_ns,{MONO}\n# COMPLETE\n")

    def test_failure_missing_composition_time(self):
        with self.assertRaisesRegex(ValueError, "no composition"):
            self.run_alignment(self.fixture(frames=[(0, 0)]))

    def test_failure_capture_error(self):
        with self.assertRaisesRegex(ValueError, "reported an error"):
            self.run_alignment(self.fixture(extra="# ERROR,frame_timing_queue_overflow\n"))

    def test_interior_pause_gap_dropped(self):
        frames = [(0, MONO), (1, MONO + 33_333_333), (2, MONO + 5_066_666_666)]
        count, text = self.run_alignment(self.fixture(frames=frames, events=[(2000000, "MOUSE_REL,44,55")]))
        self.assertEqual(count, 0)

    def test_frontend_pause_boundaries_override_worker_delay(self):
        extra = f"# FRONTEND_EVENT,{MONO + 1000000},PAUSE\n# FRONTEND_EVENT,{MONO + 30000000},RESUME\n"
        count, text = self.run_alignment(self.fixture(events=[(20000, "MOUSE_REL,44,55")], extra=extra))
        self.assertEqual(count, 0)

    def test_completion_artifact_required(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "raw.log", Path(temp) / "aligned.log"
            source.write_text(self.fixture())
            with self.assertRaisesRegex(ValueError, "completion artifact"):
                align_log.align(source, output, VIDEO)
            Path(str(source) + ".complete").write_text("LUMINE_COMPLETE_V1 1\n")
            with self.assertRaisesRegex(ValueError, "completion artifact"):
                align_log.align(source, output, VIDEO)

    def test_no_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "raw.log", Path(temp) / "aligned.log"
            source.write_text(self.fixture())
            Path(str(source) + ".complete").write_text(f"LUMINE_COMPLETE_V1 {source.stat().st_size}\n")
            output.write_text("precious")
            with self.assertRaises(FileExistsError):
                align_log.align(source, output, VIDEO)
            self.assertEqual(output.read_text(), "precious")
            with self.assertRaises(ValueError):
                align_log.align(source, source, VIDEO)

    def test_reject_duplicate_and_backwards_pts(self):
        for frames in [[(0, MONO), (1, MONO)], [(2, MONO), (1, MONO + 33_333_333)]]:
            with self.assertRaisesRegex(ValueError, "Non-monotonic"):
                self.run_alignment(self.fixture(frames=frames))

    def test_late_input_is_dropped(self):
        count, text = self.run_alignment(self.fixture(events=[(100000000, "KEY_CHUNK,W")]))
        self.assertEqual(count, 0)
        self.assertNotIn("KEY_CHUNK,W", text)

    def test_incomplete_log_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion"):
            self.run_alignment(self.fixture().replace("# COMPLETE\n", ""))

    def test_events_after_complete_rejected(self):
        with self.assertRaisesRegex(ValueError, "after completion"):
            self.run_alignment(self.fixture() + f"{WALL + 500000},KEY_CHUNK,W\n")

    def test_backwards_events_rejected(self):
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.run_alignment(self.fixture(events=[(100, "KEY_CHUNK,W"), (10, "KEY_CHUNK,A")]))


if __name__ == "__main__":
    unittest.main()
