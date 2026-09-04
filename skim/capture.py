"""
Step 1 of the Skim pipeline: Capture.

Responsibility of this module: take whatever file path comes in (a phone
photo, an upload) and turn it into a *validated, normalized* in-memory
image, or fail loudly with a specific reason.

Why this matters more than it looks like it should:
Every image that gets past this module goes on to cost money (a paid VLM
call in Step 3) and consumes a human's attention if it needs manual review
(Step 4). A blurry, corrupt, sideways, or absurdly low-res photo will
either waste the API call or produce silently wrong extracted prices. It's
far cheaper to reject bad input here, in milliseconds, for free, than to
discover it three stages downstream.

Concretely we check, in order:
  1. The file exists and is readable.
  2. The extension is one we support (jpg/jpeg/png/heic/heif).
  3. Pillow can actually decode it (catches truncated/corrupt files that
     have a valid-looking extension but garbage bytes).
  4. EXIF orientation is applied. Phone cameras write an "this image is
     rotated 90 degrees, please display it upright" tag instead of
     rotating the pixels themselves. If you skip this, roughly half of
     phone photos come out sideways and every downstream step (deskew,
     crop, VLM) gets a rotated receipt.
  5. The resulting image meets a minimum resolution. Below a certain pixel
     count, text is physically not there to recover -- no amount of
     preprocessing or a smarter model fixes that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    # pillow-heif isn't installed in this environment yet (see
    # requirements.txt). We degrade gracefully: .heic/.heif files will hit
    # UnsupportedFormatError with a message that tells you exactly why,
    # instead of a confusing Pillow traceback.
    HEIF_SUPPORTED = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

# Below this on the shorter side, receipt text is generally too small for
# either OCR or a VLM to recover reliably. This is a starting estimate,
# not a measured threshold -- once we have real failed extractions, we
# should revisit it using actual data rather than a guess.
MIN_SHORT_SIDE_PX = 600

# A receipt photo shouldn't be a 40MB RAW-quality image or a 200-byte
# thumbnail. These are sanity bounds, not hard science.
MIN_FILE_SIZE_BYTES = 1_000
MAX_FILE_SIZE_BYTES = 25_000_000


class CaptureError(Exception):
    """Base class for every reason capture can reject an input."""


class FileNotFoundCaptureError(CaptureError):
    pass


class UnsupportedFormatError(CaptureError):
    pass


class CorruptImageError(CaptureError):
    pass


class ImageTooSmallError(CaptureError):
    pass


class ImageSizeOutOfBoundsError(CaptureError):
    pass


@dataclass
class CaptureResult:
    """What Step 2 (preprocess) receives."""

    image: Image.Image  # RGB, EXIF-orientation already applied
    source_path: Path
    original_format: str  # "JPEG", "PNG", "HEIF", ...
    width: int
    height: int
    file_size_bytes: int


def capture(path: str | Path) -> CaptureResult:
    """Validate and load a receipt photo. Raises a CaptureError subclass
    on any problem, with a message specific enough to act on."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundCaptureError(f"No file at {path}")
    if not path.is_file():
        raise FileNotFoundCaptureError(f"{path} is not a file")

    # Check format before size: if the extension is wrong, that's the
    # most specific and useful thing to tell the caller, regardless of
    # how many bytes the file happens to be.
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"{path.name}: unsupported extension '{ext}'. Supported: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext in {".heic", ".heif"} and not HEIF_SUPPORTED:
        raise UnsupportedFormatError(
            f"{path.name} is HEIC/HEIF but pillow-heif isn't installed in "
            "this environment. Install it (see requirements.txt), or on "
            "the iPhone: Settings > Camera > Formats > Most Compatible, "
            "which saves new photos as JPEG instead."
        )

    file_size = path.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        raise ImageSizeOutOfBoundsError(
            f"{path.name} is only {file_size} bytes -- almost certainly "
            "empty or truncated, not a real photo."
        )
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ImageSizeOutOfBoundsError(
            f"{path.name} is {file_size / 1_000_000:.1f}MB, over the "
            f"{MAX_FILE_SIZE_BYTES / 1_000_000:.0f}MB cap. Probably not a "
            "phone photo of a receipt -- check what was actually uploaded."
        )

    try:
        with Image.open(path) as img:
            img.load()  # force full decode now, not lazily later
            original_format = img.format or "UNKNOWN"
            # Apply EXIF orientation, then drop to RGB so every image
            # downstream has a uniform 3-channel representation
            # regardless of source (PNG can be RGBA/palette, HEIC can be
            # other color modes).
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise CorruptImageError(
            f"{path.name} has extension '{ext}' but Pillow can't decode "
            "it as an image -- likely corrupt or truncated."
        ) from e

    width, height = img.size
    if min(width, height) < MIN_SHORT_SIDE_PX:
        raise ImageTooSmallError(
            f"{path.name} is {width}x{height}, below the "
            f"{MIN_SHORT_SIDE_PX}px minimum on the short side. Retake the "
            "photo closer to the receipt."
        )

    return CaptureResult(
        image=img,
        source_path=path,
        original_format=original_format,
        width=width,
        height=height,
        file_size_bytes=file_size,
    )
