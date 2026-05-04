import cv2
import numpy as np
import os
import scipy.ndimage as ndi

class Preprocessor:
    def __init__(self, output_dir="data/processed", debug_dir="outputs/debug", resize_to=None):
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        self.resize_to = resize_to

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

    def preprocess_image(self, image_path, visualize=True):
        # Step 1: Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # FIX: Removed the MORPH_OPEN and MORPH_CLOSE operations that were 
        # creating artificial sharp noise inside the coins.
        median = cv2.medianBlur(gray, 7) # Increased to 7 to wipe out coin textures
        blurred = cv2.GaussianBlur(median, (7, 7), 1.5)

        if visualize:
            # Just pass 'gray' to satisfy the existing debug variables
            self._save_debug(image_path, img, gray, gray, gray, median, blurred)

        filename = os.path.basename(image_path)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, blurred)

        return blurred, output_path

    def _save_debug(self, path, img, gray, opened, closed, median, blurred):
        filename = os.path.basename(path)

        debug_image = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(median, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
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