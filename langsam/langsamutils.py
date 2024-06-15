import requests
import torch
from PIL import Image

from io import BytesIO
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, colors=None, thickness=2):
    """
    Draw bounding boxes on the image.

    Args:
        image (numpy.ndarray): Input image.
        boxes (list of tuples): List of bounding box coordinates in format (x_min, y_min, x_max, y_max).
        colors (list of tuples, optional): List of BGR colors for each bounding box. If None, default to red color.
        thickness (int, optional): Thickness of the bounding box lines.

    Returns:
        numpy.ndarray: Image with bounding boxes drawn on it.
    """
    if colors is None:
        colors = [(0, 0, 255)] * len(boxes)

    for box, color in zip(boxes, colors):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    return image

def draw_segmentation_masks(image, masks, colors=None, alpha=0.5):
    """
    Draw segmentation masks on the image.

    Args:
        image (numpy.ndarray): Input image.
        masks (list of numpy.ndarray): List of binary segmentation masks.
        colors (list of tuples, optional): List of BGR colors for each mask. If None, default to random colors.
        alpha (float, optional): Transparency of the drawn masks.

    Returns:
        numpy.ndarray: Image with segmentation masks drawn on it.
    """
    if colors is None:
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(masks))]

    for mask, color in zip(masks, colors):
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image = cv2.addWeighted(image, 1, mask_rgb, alpha, 0)

    return image

# Example usage:
# Assuming `image` is your input image, `boxes` is a list of bounding box coordinates,
# `masks` is a list of segmentation masks.
# You can call these functions as follows:
# image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
# image_with_masks = draw_segmentation_masks(image.copy(), masks)

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_imagefile(image):

    image_pil = Image.open(image).convert("RGB")

    return image_pil
def load_image(image):
    if image.startswith("http"):
        image_pil = download_image(image)
    else:
        image_pil = Image.open(image).convert("RGB")

    return image_pil
def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def save_image(image_np, filename):
    image_np.save(filename)
def display_image(image):

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image ")
    ax.axis('off')
    plt.show()

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits,phrases):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit, phrases in zip(boxes, logits,phrases):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"{phrases}: {confidence_score}", fontsize=8, color='white', verticalalignment='top')

    plt.show()

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")


