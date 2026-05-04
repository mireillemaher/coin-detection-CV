import numpy as np
import cv2
import math
import os

class CircleDetector:
    """Uses OpenCV's HoughCircles for Circle Detection."""
    def __init__(self,
                 min_r=8,
                 max_r=200,
                 r_step=1, 
                 theta_step=5,
                 vote_thresh=0.45,
                 min_center_dist=None,
                 output_dir="data/circles",
                 debug_dir="outputs/circle_debug"):
        self.minimmRadii = min_r
        self.maximmRadii = max_r
        
        # Float threshold values in [0,1] are mapped to OpenCV param2 range.
        if isinstance(vote_thresh, float) and vote_thresh < 1.0:
            self.intenThres = int(20 + 80 * max(0.0, min(1.0, vote_thresh)))
        else:
            self.intenThres = int(vote_thresh)
            
        self.regRadii = int(max(10, min_r * 1.2)) if min_center_dist is None else int(min_center_dist)
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def _circle_edge_support(self, edge_map, x, y, r, samples=96):
        """Estimate how much of the circumference is supported by edges."""
        h, w = edge_map.shape[:2]
        hits = 0
        total = 0

        for theta in np.linspace(0, 2 * np.pi, num=samples, endpoint=False):
            px = int(round(x + r * math.cos(theta)))
            py = int(round(y + r * math.sin(theta)))

            if px < 0 or py < 0 or px >= w or py >= h:
                continue

            total += 1
            y0, y1 = max(0, py - 1), min(h, py + 2)
            x0, x1 = max(0, px - 1), min(w, px + 2)
            if np.any(edge_map[y0:y1, x0:x1] > 0):
                hits += 1

        if total == 0:
            return 0.0

        return hits / total

    def _nms(self, circles):
        """Suppress near-duplicate circles AND internal texture noise."""
        if not circles:
            return []

        # Sort by edge support (confidence) descending
        circles_sorted = sorted(circles, key=lambda c: c[3], reverse=True)
        kept = []

        for c in circles_sorted:
            x1, y1, r1, _ = c
            keep = True
            for k in kept:
                x2, y2, r2, _ = k
                center_dist = math.hypot(x1 - x2, y1 - y2)
                
                # RULE 1: Near-duplicate circles on the exact same coin
                radius_ratio = abs(r1 - r2) / max(r1, r2)
                if center_dist < 0.45 * min(r1, r2) and radius_ratio < 0.35:
                    keep = False
                    break
                    
                # RULE 2: "Circle inside a circle" (Internal texture noise)
                # If the center of this circle is inside the radius of the stronger circle
                # (with a 0.8 margin to safely allow genuine, overlapping adjacent coins)
                if center_dist < max(r1, r2) * 0.8:
                    keep = False
                    break

            if keep:
                kept.append(c)

        return [(int(x), int(y), int(r)) for x, y, r, _ in kept]

    def detect(self, image, custom_edge_map=None): # <-- Added custom_edge_map parameter
        # image should be an 8-bit single-channel image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        h, w = image.shape[:2]
        max_allowed_r = min(self.maximmRadii, int(0.48 * min(h, w)))
        if max_allowed_r <= self.minimmRadii:
            max_allowed_r = self.minimmRadii + 1

        # FIX 1: Do not double-blur the image. It is already preprocessed perfectly.
        hough_input = image 

        # FIX 2: Use the flawless custom edge map from Stage 2 to validate circles!
        if custom_edge_map is not None:
            edge_map = custom_edge_map
        else:
            # Fallback (with safer thresholds) if no custom edge map is provided
            edge_map = cv2.Canny(hough_input, 30, 100)

        # FIX 3: Lower param1 (Canny threshold) from 150 to 50. 
        # This allows OpenCV to propose circles on softer gradients. 
        # We then rely on our perfect `edge_map` to filter out the false ones.
        circles_cv = cv2.HoughCircles(
            hough_input, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(self.regRadii, 1),
            param1=100, param2=self.intenThres, 
            minRadius=self.minimmRadii, maxRadius=max_allowed_r
        )
        
        filtered = []
        if circles_cv is not None and len(circles_cv) > 0:
            circles_cv = np.round(circles_cv[0, :]).astype("int")
            for (x, y, r) in circles_cv:
                # Reject circles that sit against frame boundaries.
                if x - r < 2 or y - r < 2 or x + r >= w - 2 or y + r >= h - 2:
                    continue

                # This now checks your perfectly unbroken edges!
                support = self._circle_edge_support(edge_map, x, y, r)
                
                # Slightly relaxed support ratio to ensure we don't miss coins
                if support < 0.30:
                    continue

                filtered.append((int(x), int(y), int(r), float(support)))

        circles = self._nms(filtered)
                
        return {
            "circles": circles,
            "candidates_raw": circles,
            "accumulator": edge_map[..., np.newaxis].astype(np.float32),
            "radii": np.array([0])
        }
    def run(self, image, original_img=None, name=None, show=True):
        return self.detect(image)
