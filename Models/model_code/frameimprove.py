import os
import glob
import cv2
import numpy as np
from pathlib import Path

# Dynamically find the root directory of this script
BASE_DIR = Path(__file__).resolve().parent

# Configuration
input_folder = BASE_DIR / "dataset_1280x1280"
output_folder = BASE_DIR / "dataset_1280x1280_clean"

blur_threshold = 100.0
similarity_threshold = 0.90
apply_sharpening = True

output_folder.mkdir(parents=True, exist_ok=True)

def is_blurry(image, threshold=100.0):
    """Calculates variance of Laplacian to measure focus/blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def calculate_similarity(img1, img2):
    """Calculates structural similarity ratio between two images."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    non_zero = np.count_nonzero(diff > 25)
    total_pixels = gray1.size
    
    return 1.0 - (non_zero / total_pixels)

def sharpen_image(image):
    """Applies a subtle sharpening kernel to boost edge contrast."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

image_paths = sorted(glob.glob(str(input_folder / "*.jpg")))

if not image_paths:
    print(f"No images found in '{input_folder}'!")
    exit()

saved_count = 0
skipped_blur = 0
skipped_duplicate = 0
prev_image = None

print(f"Processing {len(image_paths)} images from '{input_folder}'...")

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    if is_blurry(img, blur_threshold):
        skipped_blur += 1
        continue

    if prev_image is not None:
        similarity = calculate_similarity(img, prev_image)
        if similarity > similarity_threshold:
            skipped_duplicate += 1
            continue

    final_img = sharpen_image(img) if apply_sharpening else img

    filename = f"clean_frame_{saved_count:05d}.jpg"
    cv2.imwrite(str(output_folder / filename), final_img)

    prev_image = img
    saved_count += 1

print("\n--- Cleaning Complete ---")
print(f"Total Original Images : {len(image_paths)}")
print(f"Skipped Blurry Images : {skipped_blur}")
print(f"Skipped Duplicates    : {skipped_duplicate}")
print(f"Final Clean Images    : {saved_count} saved to '{output_folder}'")