import cv2
import numpy as np

def extract_features(image, circles):
    features_list = []
    h, w = image.shape[:2]

    for i, (x, y, r) in enumerate(circles):
        x, y, r = int(x), int(y), int(r)

        # --- Step 1: Safe ROI cropping ---
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, w)
        y2 = min(y + r, h)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # --- Step 2: Create circular mask ---
        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)

        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        radius = min(center[0], center[1], r)

        cv2.circle(mask, center, radius, 255, -1)

        # --- Step 3: Convert ROI to HSV ---
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- Step 4: Mean HSV values ---
        mean = cv2.mean(hsv_roi, mask=mask)
        mean_hue = mean[0]
        mean_sat = mean[1]
        mean_val = mean[2]

        # --- Step 5: Normalize radius ---
        r_norm = r / w

        # --- Step 6: Histograms (H, S, V) ---
        hist_h = cv2.calcHist([hsv_roi], [0], mask, [16], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], mask, [16], [0, 256])
        hist_v = cv2.calcHist([hsv_roi], [2], mask, [16], [0, 256])

        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        # --- Step 7: Store features ---
        coin_features = {
            "id": i,
            "x": x,
            "y": y,
            "r_norm": r_norm,
            "mean_hue": mean_hue,
            "mean_sat": mean_sat,
            "mean_val": mean_val,
            "hist_h": hist_h,
            "hist_s": hist_s,
            "hist_v": hist_v
        }

        features_list.append(coin_features)

    return features_list