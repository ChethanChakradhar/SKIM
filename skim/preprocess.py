"""
Step 2 of the Skim pipeline: Preprocess.

Responsibility of this module: take the validated photo from Step 1 and
hand Step 3 (the VLM) the cleanest possible picture of the receipt --
straightened, background removed, faded ink darkened.

Why bother, when the VLM is already good at reading messy images?
Because every pixel that isn't receipt is a pixel the model has to
consider and might hallucinate from, and every degree of rotation makes
column alignment (QTY | PRICE | TOTAL) harder to read. Preprocessing is
free and deterministic; model errors are neither.

The central idea: "crop to the receipt" and "deskew the receipt" are the
same operation.

A photo of a receipt lying on a table is never taken with the camera
perfectly parallel to the paper. So the receipt doesn't arrive rotated --
it arrives as a *trapezoid*, with the edge nearer the lens wider than the
edge further away. Rotating the image cannot undo that, because the
distortion isn't a rotation.

What does undo it is a perspective transform (a homography): find the
receipt's four corners in the photo, then compute the mapping that drags
those four corners onto the four corners of a perfect rectangle, and push
every pixel through that mapping. The output is the receipt as if shot
from directly overhead. Straightened and cropped, in one operation. This
is what phone scanner apps do.

So the pipeline here is:

  1. Find the four corners        <- the only genuinely hard part
  2. Warp them to a rectangle     <- crop and deskew together
  3. Boost local contrast         <- rescue faded thermal ink

Step 1 fails on some real photos and that is expected. Measured on the
first three real receipts: it succeeds on both photos shot flat on a
table, and fails on the one shot hand-held, because fingers overlapping
the paper's edge break the boundary into fragments that no longer form a
closed shape.

When it fails we return the photo *uncropped* rather than guessing. A
crop that slices off the TOTAL line silently corrupts the extracted data;
an uncropped photo merely costs the model a little more work. Guessing is
the more expensive mistake, so we don't.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from skim.capture import CaptureResult

# Corner-finding runs on a downscaled copy: it is looking for one big
# shape, and the paper's outline is just as findable at 1000px as at
# 4032px while the edge detector runs ~16x faster. The corners we find
# are scaled back up and the actual warp is done on the full-resolution
# pixels, so nothing is lost from the image we hand downstream.
DETECT_MAX_DIM = 1000

# Gates a candidate shape must pass to be believed as "the receipt".
# Calibrated against the first three real photos, where genuine
# detections covered 25-29% of the frame and filled 94-98% of their own
# bounding box, and the one false candidate covered 64% with a fill of
# 0.67. Revisit these with more real photos, not with intuition.
MIN_RECEIPT_AREA_FRAC = 0.10
MIN_RECT_FILL = 0.80

# CLAHE = Contrast Limited Adaptive Histogram Equalization. Plain
# histogram equalization stretches contrast using one histogram for the
# whole image, so a photo that is bright at the top and shadowed at the
# bottom gets one compromise correction that suits neither. CLAHE instead
# equalizes each tile of an 8x8 grid separately, which is what a receipt
# lying under uneven light actually needs. "Contrast limited" caps how
# far any one tile can stretch, which is what stops it from turning paper
# grain and JPEG noise into fake speckled text.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


@dataclass
class PreprocessResult:
    """What Step 3 (VLM extraction) receives."""

    image: Image.Image  # RGB, deskewed and cropped if that was possible
    deskewed: bool  # False means we passed the photo through uncropped
    corners: Optional[np.ndarray]  # 4x2 float32, original-photo coordinates
    detection_note: Optional[str]  # why detection failed, when it did


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """PIL stores pixels as RGB; OpenCV expects BGR. Same bytes, reversed
    channel order -- skip this and every color operation is subtly wrong."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _cv_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Sort four unordered corners into top-left, top-right, bottom-right,
    bottom-left.

    The warp needs to know which corner is which -- otherwise it maps the
    top-left of the receipt onto the bottom-right of the output and hands
    back a mirrored or upside-down image.

    The trick: for a quadrilateral, x+y is smallest at the top-left corner
    (both coordinates small) and largest at the bottom-right. And y-x is
    smallest at the top-right (large x, small y) and largest at the
    bottom-left. Two sums, two subtractions, no trigonometry.
    """
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    coord_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(coord_sum)]  # top-left
    ordered[2] = points[np.argmax(coord_sum)]  # bottom-right

    coord_diff = np.diff(points, axis=1).ravel()  # y - x
    ordered[1] = points[np.argmin(coord_diff)]  # top-right
    ordered[3] = points[np.argmax(coord_diff)]  # bottom-left

    return ordered


def find_receipt_corners(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Locate the receipt's four corners, or explain why we couldn't.

    Returns (corners, None) on success and (None, reason) on failure. The
    reason is worth keeping: once a few hundred receipts have run through,
    the tally of failure reasons tells us what to fix next, which beats
    guessing at improvements.
    """
    height, width = bgr.shape[:2]
    scale = DETECT_MAX_DIM / max(height, width)
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)))
    small_area = small.shape[0] * small.shape[1]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Blur before edge detection or the detector fires on wood grain,
    # tabletop scratches and sensor noise as enthusiastically as on the
    # paper's edge. We want one big boundary, not thousands of small ones.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    # Thicken the edges so a boundary broken by a shadow or a fold still
    # closes into a loop. findContours only reports enclosed area for
    # closed loops; a boundary with a one-pixel gap has an area of zero
    # and gets discarded as noise.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no edges found in photo"

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area_frac = cv2.contourArea(contour) / small_area
        if area_frac < MIN_RECEIPT_AREA_FRAC:
            # Sorted by area, so everything after this is smaller too.
            return None, f"largest candidate covers only {area_frac:.1%} of the frame"

        # Canny traces the paper's edge as hundreds of slightly wobbly
        # points. approxPolyDP simplifies that to the fewest points that
        # still describe the shape within a tolerance -- here 2% of the
        # outline's length. A rectangle of paper collapses to 4 points; a
        # hand holding a receipt does not, which is exactly the signal we
        # want.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # Final sanity check: a real rectangle of paper nearly fills its
        # own minimum-area bounding box. A blob that merged the receipt
        # with a hand or a shadow does not.
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        fill = cv2.contourArea(contour) / max(cv2.contourArea(box.astype("float32")), 1.0)
        if fill < MIN_RECT_FILL:
            continue

        return _order_corners(approx / scale), None

    return None, "no four-cornered shape found (edges likely broken by fingers or clutter)"


def four_point_warp(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral at `corners` into a head-on rectangle."""
    top_left, top_right, bottom_right, bottom_left = corners

    # The trapezoid's two horizontal edges differ in length (that's the
    # perspective distortion). Take the longer of each pair so the output
    # preserves the most detailed part of the receipt rather than
    # squeezing it down to match the compressed far edge.
    out_width = int(max(np.linalg.norm(top_right - top_left),
                        np.linalg.norm(bottom_right - bottom_left)))
    out_height = int(max(np.linalg.norm(bottom_left - top_left),
                         np.linalg.norm(bottom_right - top_right)))

    destination = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(bgr, matrix, (out_width, out_height))


def boost_contrast(bgr: np.ndarray) -> np.ndarray:
    """Darken faded thermal ink without destroying color.

    We convert to LAB, which splits an image into one lightness channel
    (L) and two color channels (A, B), and apply CLAHE to L alone. Doing
    this in RGB would mean equalizing red, green and blue separately,
    which shifts the colors as a side effect. Here the paper gets its
    contrast back and the color is left exactly as it was -- which
    matters because a hand-drawn pen mark on a receipt is a real signal
    we may want later.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    merged = cv2.merge([clahe.apply(lightness), a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess(captured: CaptureResult) -> PreprocessResult:
    """Deskew, crop and enhance a captured receipt photo.

    Unlike capture(), this never raises on a difficult photo. Capture's
    job was to reject input that isn't usable at all; by this point the
    photo is known-good and our job is to improve it as much as we can.
    A photo we can't find corners in is still a photo the VLM can read.
    """
    bgr = _pil_to_cv(captured.image)

    corners, note = find_receipt_corners(bgr)
    if corners is not None:
        bgr = four_point_warp(bgr, corners)

    return PreprocessResult(
        image=_cv_to_pil(boost_contrast(bgr)),
        deskewed=corners is not None,
        corners=corners,
        detection_note=note,
    )
