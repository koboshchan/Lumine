import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "align", Path(__file__).resolve().parents[1] / "tools" / "align_log.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class MuxedTimingTest(unittest.TestCase):
    def test_b_frames_shift_rounding_and_trailing_stop(self):
        frames = [(100, 0), (400, 1000000), (200, 333333),
                  (300, 666666), (700, 2000000), (500, 1333333)]
        actual = [440000, 1440000, 770000, 1110000, 2440000]
        self.assertEqual(a.reconcile_frames(frames, actual),
                         [(f[0], p) for f, p in zip(frames, actual)])

    def test_reject_changed_order(self):
        with self.assertRaises(ValueError):
            a.reconcile_frames([(1, 0), (2, 333333), (3, 666666)],
                               [440000, 1110000, 770000])

    def test_reject_missing_start(self):
        with self.assertRaises(ValueError):
            a.reconcile_frames([(1, 0), (4, 1000000), (2, 333333)],
                               [1440000, 770000])

    def test_reject_extra_video_or_excessive_missing(self):
        with self.assertRaises(ValueError):
            a.reconcile_frames([(1, 0)], [0, 333333])
        with self.assertRaises(ValueError):
            a.reconcile_frames([(i, i * 333333) for i in range(20)], [0])

    def test_zero_shift(self):
        self.assertEqual(a.reconcile_frames([(1, 0), (2, 333333)], [0, 333333]),
                         [(1, 0), (2, 333333)])
