"""
Project pipeline orchestrator.

Stages:
    1. Preprocessing  (raw image  -> blurred grayscale, data/processed/)
    2. Edge detection (blurred    -> binary edge map,   data/edges/)
    3.

BATCH mode iterates the entire data/raw/ folder and saves an edge map for each
image to data/edges/edge_<filename>.jpg. Slower (a few minutes for 215 images).
"""

import os
import argparse
import cv2

from src.preprocessing import Preprocessor
from src.edge_detection import EdgeDetector


def first_image_in(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No images found in {folder}")


def run_demo(image_path, low_ratio, high_ratio, show):
    """Single-image run: preprocess -> edge detect -> show 6-panel plot."""
    print(f"DEMO mode — input: {image_path}")

    preprocessor = Preprocessor()
    blurred, prep_path = preprocessor.preprocess_image(image_path,
                                                       visualize=False)
    print(f"  preprocessing -> {prep_path}")

    detector = EdgeDetector(low_ratio=low_ratio, high_ratio=high_ratio)
    edge_path, panel_path, _ = detector.run(
        blurred,
        edge_map_name="edge_map.png",
        panel_name="edge_detection_stages.png",
        show=show,
    )
    print("\nDone.")
    print(f"  edge_map  : {edge_path}")
    print(f"  6-panel   : {panel_path}")


def run_batch(input_folder):
    """Batch run: preprocess every image, then edge-detect every image."""
    print(f"BATCH mode — folder: {input_folder}")

    preprocessor = Preprocessor()
    prep_results = preprocessor.process_folder(input_folder)
    print(f"  preprocessing complete: {len(prep_results)} image(s)")

    detector = EdgeDetector()
    saved = 0
    for name, processed_path in prep_results:
        blurred = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
        if blurred is None:
            print(f"  skip (could not read): {processed_path}")
            continue
        stages = detector.detect(blurred)
        edge_path = os.path.join(detector.output_dir, f"edge_{name}")
        cv2.imwrite(edge_path, stages["edge_map"])
        saved += 1

    print(f"  edge detection complete: {saved} edge map(s) saved to "
          f"{detector.output_dir}/")
    print("\nReady for the circle-detection stage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to an image (DEMO mode). "
                             "Defaults to the first image in data/raw.")
    parser.add_argument("--batch", action="store_true",
                        help="Run BATCH mode (process all images in data/raw)")
    parser.add_argument("--no-show", action="store_true",
                        help="DEMO mode: save the plot but don't open a window")
    parser.add_argument("--low-ratio",  type=float, default=0.05,
                        help="DEMO mode: low threshold ratio (default 0.05)")
    parser.add_argument("--high-ratio", type=float, default=0.15,
                        help="DEMO mode: high threshold ratio (default 0.15)")
    args = parser.parse_args()

    if args.batch:
        run_batch("data/raw")
    else:
        image_path = args.image or first_image_in("data/raw")
        run_demo(image_path,
                 low_ratio=args.low_ratio,
                 high_ratio=args.high_ratio,
                 show=not args.no_show)


if __name__ == "__main__":
    main()
