import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DataProcessor"))
from DataProcessor import ActionChunk, ActionFrame, KeyRecorderParser


class CursorActionsTests(unittest.TestCase):
    def parse_text(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "input.txt"
            log.write_text(text)
            return KeyRecorderParser(str(log), **kwargs).parse()

    def fixture(self, **kwargs):
        return KeyRecorderParser(
            str(ROOT / "tests/fixtures/cursor_modes.txt"), **kwargs
        ).parse()

    def test_menu_scaled_and_scroll_keys_unchanged(self):
        frame = self.fixture().get_actions_at_time(0)
        self.assertEqual(frame.absolute_position, (640, 360))
        self.assertEqual(
            frame.to_lumine_format(),
            "<|action_start|><use_absolute> 640 360 1 ; W A S D ; W A S D ; W A S D ; W A S D ; W A S D ; W A S D<|action_end|>",
        )
        self.assertEqual(len(frame.chunks), 6)

    def test_locked_camera_keeps_relative_even_with_abs(self):
        frame = self.fixture().get_actions_at_time(2000000)
        self.assertIsNone(frame.absolute_position)
        self.assertTrue(frame.to_lumine_format().startswith("<|action_start|>50 -8 -2 ; "))

    def test_unlock_and_clamp_to_output_pixels(self):
        frame = self.fixture().get_actions_at_time(4000000)
        self.assertEqual(frame.absolute_position, (1279, 719))

    def test_stationary_menu_carries_target(self):
        self.assertEqual(
            self.fixture().get_actions_at_time(5500000, 1).absolute_position,
            (1279, 719),
        )

    def test_resume_snapshots_replace_paused_state(self):
        parser = self.fixture()
        self.assertTrue(parser.is_paused(7000000))
        self.assertFalse(parser.is_paused(8000000))
        self.assertEqual(
            parser.get_actions_at_time(8000000).absolute_position, (320, 180)
        )

    def test_hide_without_lock_uses_relative(self):
        parser = self.parse_text(
            "0,SCREEN,0,0,1280,720\n0,MOUSE,UNLOCK\n0,MOUSE,HIDE\n0,MOUSE_ABS,42,60\n"
        )
        self.assertIsNone(parser.get_actions_at_time(0).absolute_position)

    def test_lock_with_visible_cursor_uses_relative(self):
        parser = self.parse_text(
            "0,SCREEN,0,0,1280,720\n0,MOUSE,LOCK\n0,MOUSE,SHOW\n0,MOUSE_ABS,42,60\n"
        )
        self.assertIsNone(parser.get_actions_at_time(0).absolute_position)

    def test_legacy_logs_remain_relative(self):
        for state in ("", "0,MOUSE_ABS,640,360\n"):
            with self.subTest(state=state):
                parser = self.parse_text(state + "0,MOUSE_REL,30,12\n0,KEY_CHUNK,W\n")
                self.assertTrue(
                    parser.get_actions_at_time(0).to_lumine_format().startswith(
                        "<|action_start|>30 12 0 ; W ; W ; W ; W ; W ; W"
                    )
                )

    def test_legacy_abs_with_explicit_source_bounds(self):
        parser = self.parse_text(
            "0,MOUSE,SHOW\n0,MOUSE_ABS,960,540\n",
            source_bounds=(0, 0, 1920, 1080),
        )
        self.assertEqual(parser.get_actions_at_time(0).absolute_position, (640, 360))

    def test_no_geometry_does_not_guess(self):
        parser = self.parse_text("0,MOUSE,UNLOCK\n0,MOUSE_ABS,960,540\n")
        self.assertIsNone(parser.get_actions_at_time(0).absolute_position)

    def test_future_events_not_used_and_half_open_window(self):
        parser = self.parse_text(
            "0,SCREEN,0,0,1280,720\n0,MOUSE,UNLOCK\n0,MOUSE,SHOW\n"
            "2000000,MOUSE_ABS,100,200\n"
        )
        self.assertIsNone(parser.get_actions_at_time(0).absolute_position)
        self.assertEqual(parser.get_actions_at_time(2000000).absolute_position, (100, 200))
        self.assertIsNone(parser.get_absolute_position(-1))

    def test_crop_origin_override_and_negative_clamp(self):
        parser = self.parse_text(
            "0,SCREEN,0,0,1920,1080\n0,MOUSE,UNLOCK\n0,MOUSE_ABS,-960,540\n"
            "2000000,MOUSE_ABS,-5000,-5000\n",
            source_bounds=(-1920, 0, 1920, 1080),
        )
        self.assertEqual(parser.get_actions_at_time(0).absolute_position, (640, 360))
        self.assertEqual(parser.get_actions_at_time(2000000).absolute_position, (0, 0))

    def test_configurable_output_size(self):
        self.assertEqual(
            self.fixture(output_width=640, output_height=360)
            .get_actions_at_time(0).absolute_position,
            (320, 180),
        )

    def test_empty_log_and_missing_position(self):
        for text in ("", "0,MOUSE,UNLOCK\n0,MOUSE,SHOW\n0,SCREEN,0,0,1280,720\n"):
            self.assertIsNone(self.parse_text(text).get_actions_at_time(0).absolute_position)

    def test_relative_and_scroll_clamps_unchanged(self):
        frame = ActionFrame([ActionChunk(dx=5000, dy=-5000, scroll=1200)] * 6)
        self.assertTrue(frame.to_lumine_format().startswith("<|action_start|>995 -996 5 ; "))
        frame = ActionFrame([ActionChunk(dx=-5000, dy=5000, scroll=-1200)] * 6)
        self.assertTrue(frame.to_lumine_format().startswith("<|action_start|>-995 996 -5 ; "))

    def test_mode_transition_final_mode_wins(self):
        parser = self.parse_text(
            "0,MOUSE,LOCK\n0,MOUSE,HIDE\n0,MOUSE_ABS,100,200\n"
            "0,SCREEN,0,0,1280,720\n1000000,MOUSE,UNLOCK\n1000000,MOUSE,SHOW\n"
        )
        self.assertEqual(parser.get_actions_at_time(0).absolute_position, (100, 200))

    def test_last_fraction_of_window_not_dropped(self):
        parser = self.parse_text("1999999,MOUSE_REL,10,4\n1999999,MOUSE,WHEEL,120\n")
        frame = parser.get_actions_at_time(0)
        self.assertEqual((frame.chunks[-1].dx, frame.chunks[-1].dy, frame.chunks[-1].scroll),
                         (10, 4, 120))

    def test_invalid_dimensions(self):
        for kwargs in ({"output_width": 0}, {"output_height": -1},
                       {"source_bounds": (0, 0, 0, 100)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KeyRecorderParser("unused", **kwargs)

    def test_cli_requires_source_dimension_pair(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "DataProcessor/DataProcessor.py"),
             "--log", "unused", "--video", "unused", "--output", "unused",
             "--source-width", "1920"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be used together", result.stderr)

    def test_custom_token_registered_in_training_submodule(self):
        path = ROOT / "VeOmni/tasks/omni/train_qwen_vl.py"
        self.assertTrue(path.exists(), "Initialize VeOmni with git submodule update --init")
        tree = ast.parse(path.read_text())
        custom_lists = [
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "custom_tokens" for t in node.targets)
        ]
        self.assertEqual(len(custom_lists), 1)
        self.assertEqual(custom_lists[0].count("<use_absolute>"), 1)


if __name__ == "__main__":
    unittest.main()
