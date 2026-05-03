"""
Visualization Module
====================

Centralised plotting for the circle detection stage.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def annotate_image(original_img, circles, color, thickness=2):
    """
    Draws circles and center markers onto a copy of the image.
    """
    if original_img.ndim == 2:
        annotated = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        annotated = original_img.copy()

    for x, y, r in circles:
        # Draw outer circle
        cv2.circle(annotated, (int(x), int(y)), int(r), color, thickness)
        # Draw center
        cv2.circle(annotated, (int(x), int(y)), 2, (0, 0, 255) if color != (0, 0, 255) else (0, 255, 0), 3)
        
    return annotated

def create_circle_panel(original_img, edge_map, results, save_path, show=True):
    """
    Generates a 2x2 multi-panel figure for debugging.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    # 1. Input Image
    if original_img.ndim == 3:
        ax_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        ax_img = original_img
        
    # 3. Accumulator Heatmap
    # A.max(axis=2) - max-projection across all radii
    A = results["accumulator"]
    heatmap = A.max(axis=2)
    
    # 4. Custom CHT Detections
    custom_annotated = annotate_image(original_img, results["circles"], (0, 255, 0)) # Green
    if custom_annotated.ndim == 3:
        custom_annotated = cv2.cvtColor(custom_annotated, cv2.COLOR_BGR2RGB)

    # Convert to standard RGB for plotting
    panels = [
        ("1. Input Image", ax_img, "gray" if original_img.ndim == 2 else None),
        ("2. Edge Map", edge_map, "gray"),
        ("3. Accumulator Heatmap", heatmap, "hot"),
        ("4. Custom CHT Detections", custom_annotated, None),
    ]

    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("Circle Detection — Pipeline Stages",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 4-panel plot -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
