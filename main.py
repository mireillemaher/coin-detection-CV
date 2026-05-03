"""
Project pipeline orchestrator.

Stages:
    1. Preprocessing  (raw image  -> blurred grayscale, data/processed/)
    2. Edge detection (blurred    -> binary edge map,   data/edges/)
    3. Circle detection (edge map -> list of (x, y, r) + annotated image, data/circles/)

BATCH mode iterates the entire data/raw/ folder and saves an edge map for each
image to data/edges/edge_<filename>.jpg, and circles to data/circles/detected_<filename>.jpg. 
"""

import os
import argparse
import cv2
import math

from src.preprocessing import Preprocessor
from src.edge_detection import EdgeDetector
from src.circle_detection import CircleDetector
from src.watershed import separate_overlapping
from src.visualization import create_circle_panel, annotate_image


def first_image_in(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No images found in {folder}")


def scale_circles(circles, from_shape, to_shape):
    """Map circles from one image size to another (x, y, r)."""
    if not circles:
        return []

    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]
    if from_h == to_h and from_w == to_w:
        return circles

    sx = to_w / float(from_w)
    sy = to_h / float(from_h)
    sr = (sx + sy) / 2.0

    scaled = []
    for x, y, r in circles:
        scaled.append((int(round(x * sx)), int(round(y * sy)), int(round(r * sr))))
    return scaled


def cleanup_circles(circles, image_shape):
    """Remove border-touching artifacts and near-duplicate circles."""
    if not circles:
        return []

    h, w = image_shape[:2]
    in_frame = []
    for x, y, r in circles:
        if x - r < 2 or y - r < 2 or x + r >= w - 2 or y + r >= h - 2:
            continue
        in_frame.append((int(x), int(y), int(r)))

    in_frame.sort(key=lambda c: c[2], reverse=True)
    deduped = []
    for x1, y1, r1 in in_frame:
        keep = True
        for x2, y2, r2 in deduped:
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < 0.5 * min(r1, r2) and abs(r1 - r2) / max(r1, r2) < 0.4:
                keep = False
                break
        if keep:
            deduped.append((x1, y1, r1))

    return deduped


def run_demo(image_path, low_ratio, high_ratio, show, min_r, max_r, vote_thresh):
    """Single-image run: preprocess -> edge detect -> circle detect -> show panels."""
    print(f"DEMO mode — input: {image_path}")

    # Load original image for watershed and visualization
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image at {image_path}")

    preprocessor = Preprocessor()
    blurred, prep_path = preprocessor.preprocess_image(image_path,
                                                       visualize=False)
    print(f"  preprocessing -> {prep_path}")

    detector = EdgeDetector(low_ratio=low_ratio, high_ratio=high_ratio)
    edge_path, panel_path, stages = detector.run(
        blurred,
        edge_map_name="edge_map.png",
        panel_name="edge_detection_stages.png",
        show=False, # Don't show edge panel yet
    )
    print(f"  edge_map  : {edge_path}")
    print(f"  edge debug: {panel_path}")

    # Stage 3: Circle Detection
    print("  running circle detection...")
    circle_detector = CircleDetector(min_r=min_r, max_r=max_r, vote_thresh=vote_thresh)
    results = circle_detector.detect(blurred)

    circles_for_original = scale_circles(
        results["circles"],
        blurred.shape,
        original_img.shape,
    )

    # Watershed (if overlapping)
    final_circles = separate_overlapping(original_img, circles_for_original)
    final_circles = cleanup_circles(final_circles, original_img.shape)

    # Visualization
    panel_path_circle = "outputs/circle_debug/circle_panel.png"
    create_circle_panel(original_img, stages["edge_map"], results,
                        save_path=panel_path_circle,
                        show=show)
    print(f"  circle debug: {panel_path_circle}")

    # Save annotated image
    annotated = annotate_image(original_img, final_circles, color=(0, 255, 0))
    detected_path = "data/circles/detected.png"
    cv2.imwrite(detected_path, annotated)
    print(f"  annotated : {detected_path}")

    print("\nDone.")


def run_batch(input_folder, min_r, max_r, vote_thresh):
    """Batch run: preprocess, edge-detect, and circle-detect every image."""
    print(f"BATCH mode — folder: {input_folder}")

    preprocessor = Preprocessor()
    prep_results = preprocessor.process_folder(input_folder)
    print(f"  preprocessing complete: {len(prep_results)} image(s)")

    detector = EdgeDetector()
    circle_detector = CircleDetector(min_r=min_r, max_r=max_r, vote_thresh=vote_thresh)
    
    saved = 0
    for name, processed_path in prep_results:
        blurred = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
        if blurred is None:
            print(f"  skip (could not read): {processed_path}")
            continue
            
        stages = detector.detect(blurred)
        edge_path = os.path.join(detector.output_dir, f"edge_{name}")
        cv2.imwrite(edge_path, stages["edge_map"])
        
        # Circle Detection
        results = circle_detector.detect(blurred)
        
        # Load original image for annotation
        original_img_path = os.path.join(input_folder, name)
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            print(f"  skip (could not read): {original_img_path}")
            continue
        
        circles_for_original = scale_circles(
            results["circles"],
            blurred.shape,
            original_img.shape,
        )

        # Watershed
        final_circles = separate_overlapping(original_img, circles_for_original)
        final_circles = cleanup_circles(final_circles, original_img.shape)
        
        # Annotation
        annotated = annotate_image(original_img, final_circles, color=(0, 255, 0))
        detected_path = os.path.join(circle_detector.output_dir, f"detected_{name}")
        cv2.imwrite(detected_path, annotated)
        
        saved += 1

    print(f"  pipeline complete: {saved} image(s) processed.")
    print(f"  Results saved to {circle_detector.output_dir}/")


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
    
    # New CLI flags for Stage 3
    parser.add_argument("--min-r", type=int, default=8, help="Minimum search radius")
    parser.add_argument("--max-r", type=int, default=200, help="Maximum search radius")
    parser.add_argument("--vote-thresh", type=float, default=0.45, help="Vote threshold")

    args = parser.parse_args()

    if args.batch:
        run_batch("data/raw", min_r=args.min_r, max_r=args.max_r, vote_thresh=args.vote_thresh)
    else:
        image_path = args.image or first_image_in("data/raw")
        run_demo(image_path,
                 low_ratio=args.low_ratio,
                 high_ratio=args.high_ratio,
                 show=not args.no_show,
                 min_r=args.min_r,
                 max_r=args.max_r,
                 vote_thresh=args.vote_thresh)


if __name__ == "__main__":
    main()
