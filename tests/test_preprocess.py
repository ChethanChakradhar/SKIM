"""
Tests for skim.preprocess.

Two kinds of test here, and the split is deliberate.

The geometry and contrast functions are pure math, so they get ordinary
synthetic tests -- a warp that maps four corners to a rectangle is either
correct or it isn't, and a hand-drawn shape proves that as well as a
photograph would.

Corner *detection* is the opposite: it is defined entirely by real-world
mess (angles, table grain, folds, fingers, faded thermal ink). A
synthetic white rectangle on a black background would pass every time and
tell us nothing. So detection is tested only against the real photos in
data/raw/, and those tests skip themselves when the folder is empty --
it's gitignored, so a fresh clone has none.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.capture import SUPPORTED_EXTENSIONS, capture
from skim.preprocess import (
    _order_corners,
    boost_contrast,
    four_point_warp,
    preprocess,
)

RAW_DIR = ROOT / "data" / "raw"
REAL_PHOTOS = sorted(
    p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
) if RAW_DIR.exists() else []


class OrderCornersTests(unittest.TestCase):
    def test_scrambled_corners_are_sorted_clockwise_from_top_left(self):
        # Deliberately out of order: bottom-right, bottom-left, top-left,
        # top-right. Detection returns corners in whatever order the
        # contour tracer happened to walk them.
        scrambled = np.array([[90, 80], [10, 70], [20, 5], [100, 10]], dtype="float32")
        top_left, top_right, bottom_right, bottom_left = _order_corners(scrambled)

        self.assertEqual(tuple(top_left), (20, 5))
        self.assertEqual(tuple(top_right), (100, 10))
        self.assertEqual(tuple(bottom_right), (90, 80))
        self.assertEqual(tuple(bottom_left), (10, 70))


class FourPointWarpTests(unittest.TestCase):
    def test_tilted_quadrilateral_is_squared_up(self):
        # A white quadrilateral on black, tilted the way a receipt
        # photographed at an angle would be. If the warp math is right,
        # the output is filled with white edge to edge.
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        corners = np.array([[100, 50], [320, 90], [300, 340], [80, 300]], dtype="float32")
        cv2.fillConvexPoly(canvas, corners.astype(np.int32), (255, 255, 255))

        warped = four_point_warp(canvas, corners)

        # Sample well inside the border so we aren't testing the
        # interpolated one-pixel edge.
        interior = warped[5:-5, 5:-5]
        self.assertGreater(float(interior.mean()), 250)

    def test_output_size_follows_the_longer_edge_of_each_pair(self):
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        # Top edge 200px, bottom edge 100px -- a strong perspective
        # squeeze. The slanted sides are each hypot(50, 150) = 158px
        # long, which is the distance the warp cares about, not the
        # 150px vertical drop.
        corners = np.array([[0, 0], [200, 0], [150, 150], [50, 150]], dtype="float32")

        warped = four_point_warp(canvas, corners)

        # Width should follow the 200px top edge, not the squeezed
        # bottom -- downscaling to the compressed edge would throw away
        # readable detail from the near half of the receipt.
        self.assertEqual(warped.shape[1], 200)
        self.assertEqual(warped.shape[0], 158)


class BoostContrastTests(unittest.TestCase):
    def test_faded_image_gains_contrast(self):
        # A low-contrast gray ramp: values 100-155 only, the way faded
        # thermal ink photographs.
        faded = np.tile(
            np.linspace(100, 155, 256, dtype=np.uint8), (256, 1)
        )
        faded = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)

        boosted = boost_contrast(faded)

        before = cv2.cvtColor(faded, cv2.COLOR_BGR2GRAY).std()
        after = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY).std()
        self.assertGreater(after, before)
        self.assertEqual(boosted.shape, faded.shape)
        self.assertEqual(boosted.dtype, np.uint8)


@unittest.skipUnless(REAL_PHOTOS, "no real receipt photos in data/raw/")
class RealPhotoTests(unittest.TestCase):
    """The only tests that say anything real about corner detection."""

    def test_every_real_photo_survives_preprocessing(self):
        # Preprocess must never raise on a hard photo -- a photo we can't
        # crop is still a photo the VLM can read.
        for path in REAL_PHOTOS:
            with self.subTest(photo=path.name):
                result = preprocess(capture(path))
                self.assertIsInstance(result.image, Image.Image)
                self.assertEqual(result.image.mode, "RGB")
                self.assertGreater(min(result.image.size), 100)

    def test_detection_still_works_on_at_least_one_photo(self):
        # A regression guard on the detector itself: if a tuning change
        # makes it stop finding receipts entirely, this fails loudly
        # instead of quietly degrading every downstream extraction.
        results = [preprocess(capture(p)) for p in REAL_PHOTOS]
        self.assertTrue(any(r.deskewed for r in results))

    def test_result_fields_agree_with_each_other(self):
        for path in REAL_PHOTOS:
            with self.subTest(photo=path.name):
                captured = capture(path)
                result = preprocess(captured)
                if result.deskewed:
                    self.assertEqual(result.corners.shape, (4, 2))
                    self.assertIsNone(result.detection_note)
                    # Cropping should have removed background, so the
                    # output must cover less area than the input.
                    in_area = captured.width * captured.height
                    out_area = result.image.width * result.image.height
                    self.assertLess(out_area, in_area)
                else:
                    self.assertIsNone(result.corners)
                    self.assertTrue(result.detection_note)


if __name__ == "__main__":
    unittest.main()
