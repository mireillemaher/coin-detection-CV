"""
Run the Edge Detection pipeline (from scratch).
================================================

Usage
-----
    python run_edge_detection.py                      # uses the first image in data/raw
    python run_edge_detection.py path/to/image.jpg    # uses a specific image
    python run_edge_detection.py --no-show ...        # save plot to disk, don't pop a window

Pipeline:
    Raw image -> Preprocessor   -> blurred grayscale
              -> EdgeDetector   -> binary edge_map.png

Outputs:
    data/edges/edge_map.png                       <- input to circle detection
    outputs/edge_debug/edge_detection_stages.png  <- 6-panel matplotlib plot
"""

import os
import sys
import argparse

from src.preprocessing import Preprocessor
from src.edge_detection import EdgeDetector


def first_image_in(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No images found in {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to an input image (defaults to first image in data/raw)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the 6-panel plot but don't open a window")
    parser.add_argument("--low-ratio",  type=float, default=0.05)
    parser.add_argument("--high-ratio", type=float, default=0.15)
    args = parser.parse_args()

    image_path = args.image or first_image_in("data/raw")
    print(f"Input image: {image_path}")

    # ------------------------------------------------------------------ #
    # Stage 1 - Preprocessing                                             #
    # ------------------------------------------------------------------ #
    preprocessor = Preprocessor()
    blurred_gray, prep_path = preprocessor.preprocess_image(image_path,
                                                            visualize=False)
    print(f"Preprocessing saved blurred grayscale -> {prep_path}")

    # ------------------------------------------------------------------ #
    # Stage 2 - Edge detection                                            #
    # ------------------------------------------------------------------ #
    detector = EdgeDetector(low_ratio=args.low_ratio,
                            high_ratio=args.high_ratio)
    edge_path, panel_path, _ = detector.run(
        blurred_gray,
        edge_map_name="edge_map.png",
        panel_name="edge_detection_stages.png",
        show=not args.no_show,
    )

    print("\nDone.")
    print(f"  edge_map  : {edge_path}")
    print(f"  6-panel   : {panel_path}")
    print("  -> ready for the circle-detection stage")


if __name__ == "__main__":
    main()
