import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path, mode=cv2.IMREAD_GRAYSCALE):
    """Load an image from the given path."""
    image = cv2.imread(image_path, mode)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")
    return image

def preprocess_image(image):
    """Normalize and blur the image."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    return smoothed

def select_roi(image):
    """Select the region of interest (ROI) for focusing on the prostate area."""
    h, w = image.shape
    x_start = int(w * 0.3)
    x_end = int(w * 0.7)
    y_start = int(h * 0.4)
    y_end = int(h * 0.6)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)

def fill_holes(mask):
    """Fill holes inside the mask."""
    h, w = mask.shape[:2]
    flood_filled = mask.copy()
    mask_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask_fill, (0, 0), 255)
    inverted_flood = cv2.bitwise_not(flood_filled)
    filled_mask = cv2.bitwise_or(mask, inverted_flood)
    return filled_mask

def refine_mask(mask, kernel_size=(7, 7), blur_size=(5, 5)):
    """Clean the mask by morphological closing, filling holes, and smoothing edges."""
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    filled = fill_holes(closed)
    smoothed = cv2.GaussianBlur(filled, blur_size, 0)
    return smoothed

def get_intensity_bounds(image, label, lower_percentile=30, upper_percentile=70):
    """Get intensity bounds for thresholding based on labeled region."""
    prostate_pixels = image[label > 0]
    lower_bound = int(np.percentile(prostate_pixels, lower_percentile))
    upper_bound = int(np.percentile(prostate_pixels, upper_percentile))
    return lower_bound, upper_bound

def segment_prostate(image, label, lower_bound, upper_bound):
    """Segment the prostate region based on intensity thresholding and reference label."""
    # Thresholding based on intensity range
    mask = cv2.inRange(image, lower_bound, upper_bound)
    refined_mask = refine_mask(mask)

    # Extract the largest contour from the reference label
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(image, dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Combine the refined mask with the reference label contour
    combined_mask = cv2.bitwise_and(refined_mask, final_mask)
    return combined_mask

def visualize_results(original_image, segmented_image, reference_label):
    """Visualize the original image, segmented output, and reference label."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmented Output")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Referans Etiket")
    plt.imshow(reference_label, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def process_image(image_path, label_path):
    """Run the entire segmentation process."""
    # Load and preprocess images
    image = load_image(image_path)
    label = load_image(label_path)
    preprocessed_image = preprocess_image(image)

    # Select ROI
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Determine intensity bounds from the reference label within ROI
    lower_bound, upper_bound = get_intensity_bounds(roi_image, label[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]])
    print(f"Optimize Edilmiş Yoğunluk Aralığı: {lower_bound}-{upper_bound}")

    # Perform segmentation within ROI
    segmented_roi = segment_prostate(roi_image, label[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]], lower_bound, upper_bound)

    # Create full mask with segmented ROI
    full_mask = np.zeros_like(preprocessed_image, dtype=np.uint8)
    full_mask[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]] = segmented_roi

    # Visualize results
    visualize_results(preprocessed_image, full_mask, label)

    # Save the segmented output
    output_path = "segmented_output_filled.png"
    cv2.imwrite(output_path, full_mask)
    print(f"Segmentasyon sonucu '{output_path}' olarak kaydedildi.")

# Input files
image_path = 'data/img9.png'
label_path = 'data/label9.png'

process_image(image_path, label_path)
