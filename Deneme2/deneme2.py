import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from skimage.filters.rank import entropy
from skimage.morphology import disk, remove_small_objects, binary_closing, binary_opening, binary_dilation
from skimage.draw import line
from skimage.feature import hog, local_binary_pattern
from skimage.measure import label, regionprops

####################################################
###          GÜRÜLTÜ ELİMİNASYONU VE NORMALİZASYON ###
####################################################
# Load the MRI image
image_path = "data/img7.png"  # Using the uploaded file
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at path: {image_path}. Please check the file path and try again.")

# Normalize the image (scale pixel values to range [0, 1])
normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Apply noise reduction techniques
# Gaussian Blur
gaussian_blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

# Convert the image back to 8-bit for further processing
gaussian_blurred_8bit = (gaussian_blurred * 255).astype(np.uint8)

# Display the original and Gaussian blurred images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Gaussian Blurred Image")
plt.imshow(gaussian_blurred_8bit, cmap='gray')
plt.axis('off')

plt.show()

####################################################
###        ADAPTİF EŞİKLEME VE ROI TANIMLAMA     ###
####################################################

# Define a region of interest (ROI) to focus on the prostate area
roi = gaussian_blurred_8bit[100:300, 100:300]  # Manually set; should be adjusted according to the image

# Apply Adaptive Thresholding to find the prostate region
adaptive_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Display Adaptive Thresholded Image
plt.figure(figsize=(8, 6))
plt.title("Adaptive Thresholded Image (ROI)")
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')
plt.show()

# Expand the mask back to the original image size
prostate_mask_full = np.zeros_like(gaussian_blurred_8bit)
prostate_mask_full[100:300, 100:300] = adaptive_thresh

# Post-processing with morphological operations to remove small objects and fill holes
prostate_binary = (prostate_mask_full > 0).astype(np.uint8)
prostate_cleaned = remove_small_objects(prostate_binary, min_size=1000, connectivity=1)  # Adjusted min_size for better prostate isolation
prostate_cleaned = binary_closing(prostate_cleaned, disk(5)).astype(np.uint8)  # Decreased disk size for closing to avoid overfilling
prostate_cleaned = binary_opening(prostate_cleaned, disk(3)).astype(np.uint8)  # Decreased disk size for opening to avoid removing relevant regions
prostate_cleaned = binary_dilation(prostate_cleaned, disk(3)).astype(np.uint8)  # Decreased disk size for dilation for more precise region highlighting

# Label connected components and select the largest one
labeled_prostate = label(prostate_cleaned)
regions = regionprops(labeled_prostate)
if regions:
    largest_region = max(regions, key=lambda r: r.area)
    prostate_mask = labeled_prostate == largest_region.label
else:
    prostate_mask = np.zeros_like(labeled_prostate)

# Convert mask to uint8 format
prostate_mask_uint8 = (prostate_mask * 255).astype(np.uint8)

# Display the final prostate mask
plt.figure(figsize=(8, 6))
plt.title("Final Prostate Mask")
plt.imshow(prostate_mask_uint8, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###        ÖZNİTELİK ÇIKARMA VE GÖRSELLEŞTİRME   ###
####################################################

# Define multiple kernels for different feature extraction
kernels = {
    "Sobel X (Horizontal Edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y (Vertical Edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Laplacian": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
}

# Apply convolution for each kernel on the original image using the prostate mask
masked_image = cv2.bitwise_and(gaussian_blurred_8bit, gaussian_blurred_8bit, mask=prostate_mask_uint8)
results = {"Masked Prostate Region": masked_image}

# Apply convolution for each kernel
for kernel_name, kernel in kernels.items():
    # Convolution
    conv_result = cv2.filter2D(masked_image, -1, kernel)
    results[f"{kernel_name} - Convolution"] = conv_result

# Display results of all kernels with convolution
plt.figure(figsize=(20, 10))
num_images = len(results)
columns = 3  # Number of columns in the display grid

for i, (title, image) in enumerate(results.items(), 1):
    plt.subplot((num_images // columns) + 1, columns, i)
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

####################################################
#             SOBEL Y CONVULASYON SEÇİLDİ        #
####################################################

# Sobel Y Convolution Final görüntüsünü al
sobel_y_final = results["Sobel Y (Vertical Edges) - Convolution"]

# Display Sobel Y Convolution separately
plt.figure(figsize=(8, 6))
plt.title("Sobel Y Convolution (Final)")
plt.imshow(sobel_y_final, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###    ENTROPI, RANGE VE STD FILTER AŞAMALARI    ###
####################################################

# 1. Entropi Hesaplama
# Entropi hesaplaması için bir disk yapılandırması kullanılıyor
entropy_image = entropy(sobel_y_final, disk(3))  # Reduced disk size for entropy calculation for better detail preservation

# 2. Range Filter
# Range filtresi: her komşuluk bölgesindeki maksimum - minimum piksel değeri
def range_filter(image):
    return generic_filter(image, lambda x: np.ptp(x), size=3)  # 3x3 bölge kullanılıyor

range_image = range_filter(sobel_y_final)

# 3. Standart Sapma (Std Filter)
# Standart sapma filtresi: her komşuluk bölgesindeki standart sapma
def std_filter(image):
    return generic_filter(image, np.std, size=3)

std_image = std_filter(sobel_y_final)

# Görselleri görselleştirme
plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
plt.title("Sobel Y Convolution (Final)")
plt.imshow(sobel_y_final, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Entropy Filter")
plt.imshow(entropy_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Range Filter")
plt.imshow(range_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Std Filter")
plt.imshow(std_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

####################################################
#       SOBEL Y + RANGE FILTER BİRLEŞİMİ         #
####################################################

# Combine Sobel Y Convolution and Range Filter
combined_result = cv2.addWeighted(sobel_y_final, 0.5, range_image, 0.5, 0)

# Normalize the combined result for better visualization
combined_result_normalized = cv2.normalize(combined_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display final combined image
plt.figure(figsize=(10, 8))
plt.title("Final Image: Sobel Y + Range Filter")
plt.imshow(combined_result_normalized, cmap='gray')
plt.axis('off')
plt.show()

####################################################
#         HoG UYGULAMASI                          #
####################################################
# Apply Histogram of Oriented Gradients (HoG)
hog_features, hog_image = hog(combined_result_normalized, visualize=True, block_norm='L2-Hys')

####################################################
#         CoHog UYGULAMASI                         #
####################################################
# Co-occurrence of Histograms of Oriented Gradients (CoHOG)
# We'll simplify this to using HoG over different orientations as a form of CoHOG analysis.
# Just taking another parameterized version of HoG with slightly altered block sizes
_, cohog_image = hog(combined_result_normalized, visualize=True, orientations=18, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

####################################################
#         LBP UYGULAMASI                           #
####################################################
# Apply Local Binary Pattern (LBP)
radius = 3
n_points = 8 * radius
lbp_image = local_binary_pattern(combined_result_normalized, n_points, radius, method='uniform').astype(np.uint8)

####################################################
#         HOUGH TRANSFORM UYGULAMASI               #
####################################################
# Apply Hough Transform to detect lines
def hough_transform(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    hough_image = np.zeros_like(image)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            rr, cc = line(y1, x1, y2, x2)
            rr = np.clip(rr, 0, hough_image.shape[0] - 1)
            cc = np.clip(cc, 0, hough_image.shape[1] - 1)
            hough_image[rr, cc] = 255
    return hough_image

hough_image = hough_transform(combined_result_normalized)

####################################################
#            SONUÇLARI GÖRSELLEŞTİRME              #
####################################################
# Display the results of all the feature extraction techniques
plt.figure(figsize=(20, 20))

plt.subplot(3, 2, 1)
plt.title("HoG Image")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title("CoHoG Image")
plt.imshow(cohog_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title("LBP Image")
plt.imshow(lbp_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title("Hough Transform Image")
plt.imshow(hough_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
