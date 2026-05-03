"""
Watershed Post-Processing
=========================

Handles overlapping circles detected by the CHT.
"""

import cv2
import numpy as np

def separate_overlapping(original_img, circles, overlap_margin=5):
    """
    Detects overlapping circles and uses cv2.watershed to separate them.
    If no overlaps are found, returns the original circles.
    """
    if len(circles) < 2:
        return circles

    # 1. Detect overlaps
    has_overlap = False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = circles[i], circles[j]
            dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
            if dist < (c1[2] + c2[2] - overlap_margin):
                has_overlap = True
                break
        if has_overlap:
            break

    # Fast path: no overlaps
    if not has_overlap:
        return circles

    # 2. Build markers
    H, W = original_img.shape[:2]
    markers = np.zeros((H, W), dtype=np.int32)
    
    # We also need a background marker (e.g., 1). So circles start from 2.
    # To determine background, we can just mark pixels far from any circle as background.
    # A simple way: create a mask of all circles. The inverse is background.
    mask_all_circles = np.zeros((H, W), dtype=np.uint8)
    for idx, (x, y, r) in enumerate(circles):
        cv2.circle(mask_all_circles, (int(x), int(y)), int(r), 255, -1)
        # Give each center a unique marker
        # Just drawing a small circle at the center as the marker for the coin
        cv2.circle(markers, (int(x), int(y)), max(1, int(r * 0.1)), idx + 2, -1)
    
    # Background marker (1) where it's not inside any circle
    markers[mask_all_circles == 0] = 1

    # 3. Apply cv2.watershed
    # Watershed expects a BGR image
    if original_img.ndim == 2:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_img.copy()

    cv2.watershed(original_bgr, markers)

    # 4. Re-fit circles to each watershed segment
    refined_circles = []
    for idx in range(len(circles)):
        # Extract region for this specific label (idx + 2)
        region = np.uint8(markers == (idx + 2)) * 255
        
        # Find contours
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Take the largest contour for this label
            c = max(contours, key=cv2.contourArea)
            if len(c) >= 5: # Need at least 5 points to fit well, or just minEnclosingCircle
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                refined_circles.append((int(cx), int(cy), int(radius)))
            else:
                # Fallback to original
                refined_circles.append(circles[idx])
        else:
            # Fallback to original
            refined_circles.append(circles[idx])

    return refined_circles
