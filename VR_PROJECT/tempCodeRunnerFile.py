import cv2
import numpy as np
import os

# ========== STEP 1: DEFINE PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of VR_Project
IMG_DIR = os.path.join(BASE_DIR, "MSFD", "1", "img")  # Path to original images
OUTPUT_DIR = os.path.join(BASE_DIR, "MSFD", "1", "segmented_masks")  # Output folder

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get first 100 images
image_files = sorted(os.listdir(IMG_DIR))[:100]

for img_index, img_file in enumerate(image_files):
    img_path = os.path.join(IMG_DIR, img_file)

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Error loading {img_path}")
        continue

    img_height, img_width = image.shape[:2]  # Get dimensions

    # ========== STEP 2: REMOVE SKIN TONES & BLACK OUT BELOW skin_lowest_y ==========
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    # Define HSV range for skin tones
    lower_skin = np.array([0, 30, 80], dtype=np.uint8)  # Light peach tones
    upper_skin = np.array([30, 170, 255], dtype=np.uint8)  # Darker peach tones

    # Create a mask for skin-colored pixels
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Find y-coordinate of the lowest skin-colored pixel
    skin_pixels = np.column_stack(np.where(skin_mask > 0))
    if len(skin_pixels) > 0:
        skin_lowest_y = np.max(skin_pixels[:, 0])  # Get the max y-value of skin-colored pixels
    else:
        skin_lowest_y = img_height  # If no skin detected, set it to bottom of image

    # Remove skin-colored areas
    image[skin_mask > 0] = [0, 0, 0]

    # BLACK OUT EVERYTHING BELOW skin_lowest_y
    image[skin_lowest_y:, :] = [0, 0, 0]

    # ========== STEP 3: APPLY K-MEANS CLUSTERING ==========
    reshaped = image.reshape((-1, 3))  # Reshape for K-Means
    reshaped = np.float32(reshaped)  # Convert to float32

    # K-Means Parameters
    K = 6  # Number of clusters (adjustable)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert cluster centers to uint8
    centers = np.uint8(centers)

    # Identify the most common cluster (background)
    most_common_cluster = np.bincount(labels.flatten()).argmax()

    # Reshape labels for proper indexing
    label_map = labels.reshape((img_height, img_width))

    # Create the clustered image
    clustered_image = centers[labels.flatten()].reshape((img_height, img_width, 3))

    # Set the most common cluster (background) to black
    clustered_image[label_map == most_common_cluster] = [0, 0, 0]

    # ========== STEP 4: SAVE THE FINAL IMAGE ==========
    output_filename = f"{img_index}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, clustered_image)

    print(f"✅ Processed {img_file}, Skin Removed & Area Below y={skin_lowest_y} Blacked Out: {output_filename}")
