# Importing necessary libraries
import cv2
import numpy as np
import os

# ========== SETTING UP FILE PATHS ==========
# Getting the base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Defining paths for input images and ground truth images
IMG_DIR = os.path.join(BASE_DIR, "MSFD", "1", "face_crop")  # Where we're getting our input images from
GT_DIR = os.path.join(BASE_DIR, "MSFD", "1", "face_crop_segmentation")  # Where the correct answers (ground truth) are stored

# ========== LOADING FACE DETECTION MODELS ==========
# Loading pre-trained models for detecting faces and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # For detecting faces
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # For detecting regular eyes
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  # For detecting eyes with glasses

# ========== BACKGROUND REMOVAL FUNCTION ==========
def remove_background(img):
    """
    Removing the background from an image using GrabCut algorithm
    Args:
        img: The input image we're processing
    Returns:
        Image with background removed
    """
    # Creating initial empty mask and models that GrabCut will use
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Getting image dimensions
    height, width = img.shape[:2]
    
    # Defining rectangle where our main object (face) likely is
    rect = (width // 10, height // 10, width - width // 10, height - height // 10)
    
    # Running GrabCut algorithm to separate foreground from background
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Creating final mask where background is 0 and foreground is 1
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    
    # Applying the mask to the original image
    return img * mask_binary[:, :, np.newaxis]

# ========== SKIN DETECTION FUNCTION ==========
def detect_skin(image):
    """
    Identifying skin regions in an image using multiple color spaces
    Args:
        image: The image where we're looking for skin
    Returns:
        A mask showing where skin is detected
    """
    # Converting image to different color spaces for better skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # HSV color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  # YCrCb color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)  # Lab color space

    # Defining color ranges that typically represent skin in each color space
    lower_hsv, upper_hsv = np.array([0, 15, 40]), np.array([35, 255, 255])
    lower_ycrcb, upper_ycrcb = np.array([0, 133, 77]), np.array([255, 173, 127])
    lower_lab, upper_lab = np.array([20, 135, 135]), np.array([230, 170, 160])

    # Creating masks for skin regions in each color space
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

    # Combining all the masks to get comprehensive skin detection
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    skin_mask = cv2.bitwise_or(skin_mask, mask_lab)

    # Cleaning up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Closing small holes
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Removing small noise

    return skin_mask

# ========== IoU CALCULATION FUNCTION ==========
def calculate_iou(pred_mask, gt_mask):
    """
    Calculating how well our prediction matches the ground truth
    Args:
        pred_mask: The mask our algorithm created
        gt_mask: The correct mask (ground truth)
    Returns:
        IoU score between 0 and 1 (higher is better)
    """
    # Finding where both masks agree (intersection)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # Finding all areas covered by either mask (union)
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculating the ratio (avoiding division by zero)
    return intersection / union if union > 0 else 0

# ========== PROCESSING EACH IMAGE ==========
iou_scores = []  # Storing how well we do on each image

# Going through each file in our input image directory
for filename in os.listdir(IMG_DIR):
    # Building full path to the current image
    img_path = os.path.join(IMG_DIR, filename)
    
    # Loading the image
    img = cv2.imread(img_path)

    # Skipping if we can't load the image
    if img is None:
        continue

    # Step 1: Removing background using GrabCut
    img = remove_background(img)

    # Converting to grayscale for face/eye detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape

    # Step 2: Blacking out top third of image (often contains hair/forehead)
    strip_height = height // 3
    img[:strip_height, :] = (0, 0, 0)

    # Step 3: Detecting faces and eyes in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    highest_eye_y = float("inf")  # Tracking highest eye position

    for (x, y, w, h) in faces:
        # Looking at just the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detecting regular eyes and eyes with glasses
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))
        glasses = glasses_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))

        # Blacking out detected eye regions
        for (ex, ey, ew, eh) in eyes:
            img[y + ey:y + ey + eh, x + ex:x + ex + ew] = (0, 0, 0)
            highest_eye_y = min(highest_eye_y, y + ey)  # Updating highest eye position

        # Blacking out glasses regions
        for (gx, gy, gw, gh) in glasses:
            img[y + gy:y + gy + gh, x + gx:x + gx + gw] = (0, 0, 0)

    # Step 4: Blacking out everything above the highest detected eye
    if highest_eye_y != float("inf"):
        img[:highest_eye_y, :] = (0, 0, 0)

    # Step 5: Detecting and removing skin regions
    skin_mask = detect_skin(img)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Step 6: Converting to final binary mask
    gray_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized_img = cv2.adaptiveThreshold(gray_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    # Step 7: Comparing with ground truth if available
    gt_path = os.path.join(GT_DIR, filename)
    if os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)

        # Converting both masks to binary (0 or 1)
        pred_mask = binarized_img // 255
        gt_mask = gt_mask // 255

        # Calculating and storing IoU score
        iou = calculate_iou(pred_mask, gt_mask)
        iou_scores.append(iou)
        print(f"{filename} - IoU score: {iou:.4f}")

# ========== FINAL PERFORMANCE REPORT ==========
if iou_scores:
    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"\nAverage IoU across all images: {avg_iou:.4f}")

print("Finished processing all images.")