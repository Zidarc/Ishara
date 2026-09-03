import os
import glob
import cv2
import numpy as np

# Configuration
input_folder = "dataset_1280x1280"       # Folder containing your extracted frames
output_folder = "dataset_1280x1280_clean" # Where clean frames will be saved

blur_threshold = 100.0      # Lower = allows blurrier, Higher = strictly sharp
similarity_threshold = 0.90 # Skip if image is >90% identical to previous frame
apply_sharpening = True     # Set to True to sharpen clean frames

os.makedirs(output_folder, exist_ok=True)

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

# Get sorted list of images from input folder
image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

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

    # 1. Filter Blurry Frames
    if is_blurry(img, blur_threshold):
        skipped_blur += 1
        continue

    # 2. Filter Duplicate Frames
    if prev_image is not None:
        similarity = calculate_similarity(img, prev_image)
        if similarity > similarity_threshold:
            skipped_duplicate += 1
            continue

    # 3. Apply Optional Sharpening
    final_img = sharpen_image(img) if apply_sharpening else img

    # 4. Save to Clean Folder
    filename = f"clean_frame_{saved_count:05d}.jpg"
    cv2.imwrite(os.path.join(output_folder, filename), final_img)
    
    prev_image = img
    saved_count += 1

print("\n--- Cleaning Complete ---")
print(f"Total Original Images : {len(image_paths)}")
print(f"Skipped Blurry Images : {skipped_blur}")
print(f"Skipped Duplicates    : {skipped_duplicate}")
print(f"Final Clean Images    : {saved_count} saved to '{output_folder}'")