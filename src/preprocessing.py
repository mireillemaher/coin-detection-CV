import cv2
import numpy as np
import os

class Preprocessor:
    def __init__(self, output_dir="data/processed", debug_dir="outputs/debug"):
        self.output_dir = output_dir
        self.debug_dir = debug_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

    def preprocess_image(self, image_path, visualize=True):
        # Step 1: Load image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        # Resize (optional but recommended for consistency)
        img = cv2.resize(img, (512, 512))

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Gaussian Blur
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Step 4: Median Blur (salt & pepper noise)
        median = cv2.medianBlur(gaussian, 5)

        # Step 5 (EXTRA): Histogram Equalization (improves lighting)
        equalized = cv2.equalizeHist(median)

        if visualize:
            self._save_debug(image_path, img, gray, gaussian, median, equalized)

        # Save final output
        filename = os.path.basename(image_path)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, equalized)

        return equalized, output_path

    def _save_debug(self, path, img, gray, gaussian, median, equalized):
        filename = os.path.basename(path)

        debug_image = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(median, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        ])

        debug_path = os.path.join(self.debug_dir, "debug_" + filename)
        cv2.imwrite(debug_path, debug_image)

    def process_folder(self, folder_path):
        results = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, file)
                processed_img, out_path = self.preprocess_image(full_path)
                results.append((file, out_path))

        return results