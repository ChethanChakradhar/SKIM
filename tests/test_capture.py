"""
Tests for skim.capture.

These build tiny synthetic images with PIL rather than depending on a
real receipt photo -- we don't have one yet (that's a separate task).
Once we do, we'll add a second test module that runs the whole pipeline
against real, messy photos.
"""

import struct
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skim import capture as cap


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_capture_test"
        self.tmp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.tmp_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

    def _make_jpeg(self, name="receipt.jpg", size=(1200, 1600)):
        path = self.tmp_dir / name
        Image.new("RGB", size, color=(255, 255, 255)).save(path, "JPEG")
        return path

    def test_valid_jpeg_is_captured(self):
        path = self._make_jpeg()
        result = cap.capture(path)
        self.assertEqual(result.width, 1200)
        self.assertEqual(result.height, 1600)
        self.assertEqual(result.original_format, "JPEG")
        self.assertEqual(result.image.mode, "RGB")

    def test_missing_file_raises(self):
        with self.assertRaises(cap.FileNotFoundCaptureError):
            cap.capture(self.tmp_dir / "does_not_exist.jpg")

    def test_unsupported_extension_raises(self):
        path = self.tmp_dir / "receipt.txt"
        path.write_text("not an image")
        with self.assertRaises(cap.UnsupportedFormatError):
            cap.capture(path)

    def test_corrupt_image_raises(self):
        # Right extension, garbage bytes -- simulates a truncated upload.
        path = self.tmp_dir / "corrupt.jpg"
        # Needs to clear MIN_FILE_SIZE_BYTES so this actually exercises
        # the decode-failure path, not the too-small-to-be-real path.
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 2000)
        with self.assertRaises(cap.CorruptImageError):
            cap.capture(path)

    def test_too_small_image_raises(self):
        path = self._make_jpeg(name="tiny.jpg", size=(200, 300))
        with self.assertRaises(cap.ImageTooSmallError):
            cap.capture(path)

    def test_empty_file_raises(self):
        path = self.tmp_dir / "empty.jpg"
        path.write_bytes(b"")
        with self.assertRaises(cap.ImageSizeOutOfBoundsError):
            cap.capture(path)

    def test_exif_rotation_is_applied(self):
        # Build a 1200x1600 (portrait) image, tag it as "rotated 90 CW
        # from a 1600x1200 landscape source" via EXIF orientation=6, and
        # confirm capture() gives us back the correctly-oriented portrait
        # image rather than the raw sideways pixel grid.
        path = self.tmp_dir / "rotated.jpg"
        img = Image.new("RGB", (1600, 1200), color=(0, 128, 255))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation tag, value 6 = rotate 90 CW
        img.save(path, "JPEG", exif=exif)

        result = cap.capture(path)
        # After exif_transpose corrects a 6-orientation 1600x1200 source,
        # the pixel grid itself becomes 1200x1600.
        self.assertEqual((result.width, result.height), (1200, 1600))


if __name__ == "__main__":
    unittest.main()
