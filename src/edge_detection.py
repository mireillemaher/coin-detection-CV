"""
Edge Detection from Scratch
===========================

Implements the Canny-style edge detection pipeline manually using NumPy + SciPy:

    1. Sobel gradients (Kx, Ky kernels convolved via scipy.signal.convolve2d)
    2. Gradient magnitude and direction
    3. Non-maximum suppression (thins edges to 1-pixel using gradient direction)
    4. Double threshold (low_ratio=0.05, high_ratio=0.15) -> strong / weak / suppressed
    5. Hysteresis (keep weak pixels only if connected to strong ones)
    6. Validation against cv2.Canny on the same image

The only OpenCV calls used are:
    - cv2.imread          (loading)
    - cv2.cvtColor        (BGR -> grayscale)
    - cv2.GaussianBlur    (blurring) — done by the preprocessing stage
    - cv2.Canny           (validation only)

Input  : blurred grayscale image (NumPy uint8 2-D array) from preprocessing
Output : binary edge map saved as edge_map.png  (input to circle detection)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class EdgeDetector:
    """From-scratch Canny edge detector."""

    # Sobel kernels (defined once at the class level)
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    def __init__(self,
                 low_ratio=0.05,
                 high_ratio=0.15,
                 output_dir="data/edges",
                 debug_dir="outputs/edge_debug"):
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Sobel filtering                                            #
    # ------------------------------------------------------------------ #
    def sobel_filters(self, img):
        """Convolve image with Kx, Ky; return (magnitude, direction) arrays."""
        img_f = img.astype(np.float32)

        Ix = convolve2d(img_f, self.Kx, mode='same', boundary='symm')
        Iy = convolve2d(img_f, self.Ky, mode='same', boundary='symm')

        magnitude = np.hypot(Ix, Iy)              # sqrt(Ix^2 + Iy^2)
        # Normalise magnitude to 0-255 for visualisation / thresholding
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255.0

        direction = np.arctan2(Iy, Ix)            # radians, range [-pi, pi]
        return magnitude.astype(np.float32), direction.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 2 — Non-maximum suppression                                    #
    # ------------------------------------------------------------------ #
    def non_max_suppression(self, magnitude, direction):
        """Thin edges to 1px using the gradient direction."""
        H, W = magnitude.shape
        out = np.zeros((H, W), dtype=np.float32)

        # Convert radians -> degrees in [0, 180)
        angle = np.rad2deg(direction) % 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                a = angle[i, j]

                # Pick the two neighbours along the gradient direction
                if (0 <= a < 22.5) or (157.5 <= a < 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= a < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= a < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                else:  # 112.5 <= a < 157.5
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Keep only local maxima along the gradient
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    out[i, j] = magnitude[i, j]

        return out

    # ------------------------------------------------------------------ #
    # Step 3 — Double threshold                                           #
    # ------------------------------------------------------------------ #
    def double_threshold(self, img):
        """Classify pixels as strong (255), weak (75) or suppressed (0).

        low_ratio  and high_ratio are both fractions of img.max() — the
        canonical Canny formulation.
        """
        high_thresh = img.max() * self.high_ratio
        low_thresh  = img.max() * self.low_ratio

        STRONG = np.uint8(255)
        WEAK   = np.uint8(75)

        result = np.zeros_like(img, dtype=np.uint8)
        result[img >= high_thresh] = STRONG
        result[(img >= low_thresh) & (img < high_thresh)] = WEAK
        return result, WEAK, STRONG

    # ------------------------------------------------------------------ #
    # Step 4 — Hysteresis                                                 #
    # ------------------------------------------------------------------ #
    def hysteresis(self, img, weak, strong=255):
        """Promote weak pixels that are 8-connected to a strong pixel."""
        H, W = img.shape
        out = img.copy()

        # Iterative sweep: keep promoting weak->strong until no change
        changed = True
        while changed:
            changed = False
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if out[i, j] == weak:
                        if np.any(out[i - 1:i + 2, j - 1:j + 2] == strong):
                            out[i, j] = strong
                            changed = True

        # Anything still marked "weak" was not connected to a strong edge
        out[out != strong] = 0
        return out

    # ------------------------------------------------------------------ #
    # Full pipeline                                                       #
    # ------------------------------------------------------------------ #
    def detect(self, blurred_gray):
        """
        Run the full pipeline on a blurred grayscale image.

        Returns
        -------
        dict with intermediate stages:
            - magnitude
            - direction
            - nms
            - thresholded   (uint8 with strong/weak/zero values)
            - edge_map      (final binary uint8 — 0 / 255)
        """
        if blurred_gray.ndim != 2:
            raise ValueError("Input must be a single-channel grayscale image")

        magnitude, direction = self.sobel_filters(blurred_gray)
        nms                  = self.non_max_suppression(magnitude, direction)
        thresh, weak, strong = self.double_threshold(nms)
        edges                = self.hysteresis(thresh, weak, strong)

        return {
            "input":       blurred_gray,
            "magnitude":   magnitude,
            "direction":   direction,
            "nms":         nms,
            "thresholded": thresh,
            "edge_map":    edges,
        }

    # ------------------------------------------------------------------ #
    # Visualisation + validation                                          #
    # ------------------------------------------------------------------ #
    def visualize(self, stages, save_path=None, show=True):
        """6-panel comparison plot — every intermediate stage + cv2.Canny."""
        # Validation: cv2.Canny on the same blurred image (uint8 only)
        canny = cv2.Canny(stages["input"].astype(np.uint8), 50, 150)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            ("1. Blurred Grayscale (input)",        stages["input"],       "gray"),
            ("2. Sobel Gradient Magnitude",         stages["magnitude"],   "gray"),
            ("3. Non-Maximum Suppression",          stages["nms"],         "gray"),
            ("4. Double Threshold (strong/weak)",   stages["thresholded"], "gray"),
            ("5. Hysteresis — Final Edge Map",      stages["edge_map"],    "gray"),
            ("6. cv2.Canny (validation)",           canny,                 "gray"),
        ]

        for ax, (title, im, cmap) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.suptitle("Edge Detection — Pipeline Stages",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved 6-panel plot -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return canny

    # ------------------------------------------------------------------ #
    # End-to-end helper used by main.py                                   #
    # ------------------------------------------------------------------ #
    def run(self, blurred_gray,
            edge_map_name="edge_map.png",
            panel_name="panel.png",
            show=True):
        """Detect edges, save edge_map.png + the 6-panel plot, return paths."""
        stages = self.detect(blurred_gray)

        edge_path  = os.path.join(self.output_dir, edge_map_name)
        panel_path = os.path.join(self.debug_dir,  panel_name)

        # Save the binary edge map for the circle-detection stage
        cv2.imwrite(edge_path, stages["edge_map"])
        print(f"Saved edge map  -> {edge_path}")

        # Save the 6-panel comparison
        self.visualize(stages, save_path=panel_path, show=show)
        return edge_path, panel_path, stages
