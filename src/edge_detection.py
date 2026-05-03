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
from scipy.ndimage import binary_dilation

class EdgeDetector:
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)

    def __init__(self, low_ratio=0.05, high_ratio=0.15, 
                 output_dir="data/edges", debug_dir="outputs/edge_debug"):
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def sobel_filters(self, img):
        img_f = img.astype(np.float32)
        Ix = convolve2d(img_f, self.Kx, mode='same', boundary='symm')
        Iy = convolve2d(img_f, self.Ky, mode='same', boundary='symm')

        magnitude = np.hypot(Ix, Iy)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255.0

        direction = np.arctan2(Iy, Ix)
        return magnitude.astype(np.float32), direction.astype(np.float32)

    def non_max_suppression(self, magnitude, direction):
        """Vectorized thinning."""
        H, W = magnitude.shape
        out = np.zeros((H, W), dtype=np.float32)
        angle = np.rad2deg(direction) % 180

        # Shifted arrays for fast neighbor comparisons
        mag_up = np.pad(magnitude, ((1, 0), (0, 0)))[:-1, :]
        mag_down = np.pad(magnitude, ((0, 1), (0, 0)))[1:, :]
        mag_left = np.pad(magnitude, ((0, 0), (1, 0)))[:, :-1]
        mag_right = np.pad(magnitude, ((0, 0), (0, 1)))[:, 1:]
        mag_up_left = np.pad(magnitude, ((1, 0), (1, 0)))[:-1, :-1]
        mag_up_right = np.pad(magnitude, ((1, 0), (0, 1)))[:-1, 1:]
        mag_down_left = np.pad(magnitude, ((0, 1), (1, 0)))[1:, :-1]
        mag_down_right = np.pad(magnitude, ((0, 1), (0, 1)))[1:, 1:]

        mask_0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 180))
        keep_0 = (magnitude >= mag_left) & (magnitude >= mag_right)

        mask_45 = (angle >= 22.5) & (angle < 67.5)
        keep_45 = (magnitude >= mag_down_left) & (magnitude >= mag_up_right)

        mask_90 = (angle >= 67.5) & (angle < 112.5)
        keep_90 = (magnitude >= mag_up) & (magnitude >= mag_down)

        mask_135 = (angle >= 112.5) & (angle < 157.5)
        keep_135 = (magnitude >= mag_up_left) & (magnitude >= mag_down_right)

        keep = (mask_0 & keep_0) | (mask_45 & keep_45) | (mask_90 & keep_90) | (mask_135 & keep_135)
        out[keep] = magnitude[keep]
        return out

    def double_threshold(self, img):
        high_thresh = img.max() * self.high_ratio
        low_thresh  = img.max() * self.low_ratio
        STRONG, WEAK = np.uint8(255), np.uint8(75)

        result = np.zeros_like(img, dtype=np.uint8)
        result[img >= high_thresh] = STRONG
        result[(img >= low_thresh) & (img < high_thresh)] = WEAK
        return result, WEAK, STRONG

    def hysteresis(self, img, weak, strong=255):
        """Vectorized hysteresis using morphological dilation."""
        strong_mask = (img == strong)
        weak_mask = (img == weak)
        structure = np.ones((3, 3), dtype=bool)

        while True:
            dilated = binary_dilation(strong_mask, structure=structure)
            new_strong = dilated & weak_mask

            if not new_strong.any():
                break

            strong_mask |= new_strong
            weak_mask &= ~new_strong

        out = np.zeros_like(img)
        out[strong_mask] = strong
        return out

    def detect(self, blurred_gray):
        magnitude, direction = self.sobel_filters(blurred_gray)
        nms = self.non_max_suppression(magnitude, direction)
        thresh, weak, strong = self.double_threshold(nms)
        edges = self.hysteresis(thresh, weak, strong) # type: ignore

        return {
            "input": blurred_gray, "magnitude": magnitude,
            "direction": direction, "nms": nms,
            "thresholded": thresh, "edge_map": edges,
        }

    def visualize(self, stages, save_path=None, show=True):
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

        plt.suptitle("Edge Detection — Pipeline Stages", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show: plt.show()
        else: plt.close(fig)
        return canny

    def run(self, blurred_gray, edge_map_name="edge_map.png", panel_name="panel.png", show=True):
        stages = self.detect(blurred_gray)
        edge_path  = os.path.join(self.output_dir, edge_map_name)
        panel_path = os.path.join(self.debug_dir,  panel_name)
        cv2.imwrite(edge_path, stages["edge_map"])
        self.visualize(stages, save_path=panel_path, show=show)
        return edge_path, panel_path, stages