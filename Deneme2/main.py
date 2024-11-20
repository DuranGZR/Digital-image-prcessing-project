import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from skimage.filters.rank import entropy
from skimage.morphology import disk

########################################################
###        NORMALİZASYON İŞLEMİ                      ###
########################################################
# Load the MRI image
image_path = "data/img7.png"  # Using the uploaded file
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at path: {image_path}. Please check the file path and try again.")

# Normalize the image (scale pixel values to range [0, 1])
normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Convert the image back to 8-bit for saving
normalized_image_8bit = (normalized_image * 255).astype(np.uint8)

# Display the original and normalized images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Normalized Image")
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')

plt.show()

#######################################################
###          GÜRÜLTÜ ELİMİNASYONU İŞLEMLERİ         ###
#######################################################

# Apply noise reduction techniques
# Gaussian Blur
gaussian_blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

# Median Blur
median_blurred = cv2.medianBlur(normalized_image_8bit, 5)

# Bilateral Filter
bilateral_filtered = cv2.bilateralFilter(normalized_image_8bit, 9, 75, 75)

# Non-Local Means Denoising
nl_means_denoised = cv2.fastNlMeansDenoising(normalized_image_8bit, None, 30, 7, 21)

# Display the original, Gaussian Blurred, Median Blurred, Bilateral Filtered, and NL Means Denoised images
plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.title("Normalized Image")
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Gaussian Blurred")
plt.imshow(gaussian_blurred, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Median Blurred")
plt.imshow(median_blurred, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Bilateral Filtered")
plt.imshow(bilateral_filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("NL Means Denoised")
plt.imshow(nl_means_denoised, cmap='gray')
plt.axis('off')

plt.show()

##################################################
#    GAUSSİAN BLUR YÖNTEMİ SEÇİLDİ               #
##################################################

# Display Gaussian Blurred image separately
plt.figure(figsize=(8, 6))
plt.title("Gaussian Blurred Image (Final)")
plt.imshow(gaussian_blurred, cmap='gray')
plt.axis('off')
plt.show()

##################################################
###       KESKİNLEŞTİRME İŞLEMLERİ             ###
##################################################

# Convert Gaussian Blurred image to 8-bit for sharpening
gaussian_blurred_8bit = (gaussian_blurred * 255).astype(np.uint8)

# 1. Simple Kernel Sharpening
kernel_sharpen_1 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_1 = cv2.filter2D(gaussian_blurred_8bit, -1, kernel_sharpen_1)

# 2. Stronger Kernel Sharpening
kernel_sharpen_2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
sharpened_2 = cv2.filter2D(gaussian_blurred_8bit, -1, kernel_sharpen_2)

# 3. Unsharp Masking
blurred = cv2.GaussianBlur(gaussian_blurred_8bit, (9, 9), 10.0)
unsharp_mask = cv2.addWeighted(gaussian_blurred_8bit, 1.5, blurred, -0.5, 0)

# 4. Laplacian Sharpening
laplacian = cv2.Laplacian(gaussian_blurred_8bit, cv2.CV_64F)
sharpened_laplacian = cv2.convertScaleAbs(laplacian)

# Display all sharpening results
plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
plt.title("Gaussian Blurred Image")
plt.imshow(gaussian_blurred_8bit, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Simple Kernel Sharpening")
plt.imshow(sharpened_1, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Stronger Kernel Sharpening")
plt.imshow(sharpened_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Unsharp Masking")
plt.imshow(unsharp_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Laplacian Sharpening")
plt.imshow(sharpened_laplacian, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
######################################################
#         STRONGER KERNEL SEÇİLDİ                    #
######################################################

# Display Stronger Kernel Sharpened Image separately
plt.figure(figsize=(8, 6))
plt.title("Stronger Kernel Sharpened Image (Final)")
plt.imshow(sharpened_2, cmap='gray')
plt.axis('off')
plt.show()

######################################################
###       DÖNÜŞÜMLER YÖNTEMLERİ EKLENDİ             ###
######################################################

# Apply transformations to the Stronger Kernel Sharpened image
# 1. Rotate 90 degrees clockwise
rotated_90_cw = cv2.rotate(sharpened_2, cv2.ROTATE_90_CLOCKWISE)

# 2. Flip horizontally
flipped_horizontal = cv2.flip(sharpened_2, 1)

# 3. Resize to half the original size
resized_half = cv2.resize(sharpened_2, (sharpened_2.shape[1] // 2, sharpened_2.shape[0] // 2))

# 4. Rotate 90 degrees counter-clockwise
rotated_90_ccw = cv2.rotate(sharpened_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 5. Flip vertically
flipped_vertical = cv2.flip(sharpened_2, 0)

# 6. Rotate to custom angle (45 degrees)
angle = 45
(h, w) = sharpened_2.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_custom_angle = cv2.warpAffine(sharpened_2, rotation_matrix, (w, h))

# 7. Thresholding
_, thresholded = cv2.threshold(sharpened_2, 127, 255, cv2.THRESH_BINARY)

# 8. Erosion
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(sharpened_2, kernel, iterations=1)

# 9. Dilation
dilated = cv2.dilate(sharpened_2, kernel, iterations=1)

# 10. Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
affine_matrix = cv2.getAffineTransform(pts1, pts2)
affine_transformed = cv2.warpAffine(sharpened_2, affine_matrix, (w, h))

# Display transformations including Stronger Kernel Sharpened Image
plt.figure(figsize=(25, 20))

plt.subplot(3, 4, 1)
plt.title("Stronger Kernel Sharpened Image")
plt.imshow(sharpened_2, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title("Rotated 90 Degrees CW")
plt.imshow(rotated_90_cw, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.title("Flipped Horizontally")
plt.imshow(flipped_horizontal, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.title("Resized to Half")
plt.imshow(resized_half, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.title("Rotated 90 Degrees CCW")
plt.imshow(rotated_90_ccw, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title("Flipped Vertically")
plt.imshow(flipped_vertical, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.title("Rotated 45 Degrees")
plt.imshow(rotated_custom_angle, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.title("Thresholded")
plt.imshow(thresholded, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 9)
plt.title("Eroded")
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title("Dilated")
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.title("Affine Transformed")
plt.imshow(affine_transformed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

####################################################
#              DİLATED SEÇİLDİ                     #
####################################################

# Display Dilated Image separately
plt.figure(figsize=(8, 6))
plt.title("Dilated Image (Final)")
plt.imshow(dilated, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###      HİSTOGRAM EŞİTLEME YÖNTEMLERİ UYGULANDI  ###
####################################################

# 1. Apply Histogram Equalization to the Dilated image
hist_eq_image = cv2.equalizeHist(dilated)

# 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(dilated)

# 3. Apply Adaptive Histogram Equalization (AHE)
ahe_image = cv2.equalizeHist(dilated)

# 4. Apply Global Histogram Equalization (GHE)
ghe_image = cv2.equalizeHist(dilated)

# Display all Histogram Equalization results including Dilated Image
plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 5)
plt.title("Histogram Equalized Image")
plt.imshow(hist_eq_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("CLAHE Image")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("AHE Image")
plt.imshow(ahe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("GHE Image")
plt.imshow(ghe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 1)
plt.title("Dilated Image")
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

####################################################
#              CLAHE SEÇİLDİ                     #
####################################################

# Display CLAHE Image separately
plt.figure(figsize=(8, 6))
plt.title("CLAHE Image (Final)")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###    KONVOLÜSYON VE KORELASYON TÜM AŞAMALAR   ###
####################################################

# Define multiple kernels for different feature extraction
kernels = {
    "Sobel X (Horizontal Edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y (Vertical Edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Laplacian": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Average Blur": np.ones((3, 3), np.float32) / 9,
}

# Dictionary to store results for both convolution and correlation
results = {"CLAHE Image (Final)": clahe_image}

# Apply convolution and correlation for each kernel
for kernel_name, kernel in kernels.items():
    # Convolution
    conv_result = cv2.filter2D(clahe_image, -1, kernel)
    results[f"{kernel_name} - Convolution"] = conv_result

    # Correlation (identical to convolution in OpenCV)
    corr_result = cv2.filter2D(clahe_image, -1, kernel)
    results[f"{kernel_name} - Correlation"] = corr_result

# Display results of all kernels with convolution and correlation side by side
plt.figure(figsize=(20, 20))
num_images = len(results)
columns = 4  # Number of columns in the display grid (CLAHE + Conv + Corr)

for i, (title, image) in enumerate(results.items(), 1):
    plt.subplot((num_images // columns) + 1, columns, i)
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

####################################################
#             SOBEL Y CONVULASYON SEÇİLDİ          #
####################################################

# Display Sobel Y Convolution separately
plt.figure(figsize=(8, 6))
plt.title("Sobel Y Convolution (Final)")
plt.imshow(results["Sobel Y (Vertical Edges) - Convolution"], cmap='gray')
plt.axis('off')
plt.show()

"""
####################################################
###    ENTROPI, RANGE VE STD FILTER AŞAMALARI    ###
####################################################

# Define larger regions for extended analysis
entropy_large = entropy(sobel_y_final, disk(10))  # Larger disk for Entropy
range_large = generic_filter(sobel_y_final, lambda x: np.ptp(x), size=7)  # 7x7 Range Filter
std_large = generic_filter(sobel_y_final, np.std, size=7)  # 7x7 Std Filter

# Normalize results for better visualization
entropy_large_normalized = cv2.normalize(entropy_large, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
range_large_normalized = cv2.normalize(range_large, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
std_large_normalized = cv2.normalize(std_large, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display extended results
plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
plt.title("Sobel Y Convolution (Final)")
plt.imshow(sobel_y_final, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Entropy Filter (Larger Region)")
plt.imshow(entropy_large_normalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Range Filter (7x7)")
plt.imshow(range_large_normalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Std Filter (7x7)")
plt.imshow(std_large_normalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
"""


