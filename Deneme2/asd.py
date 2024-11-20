import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from skimage.filters.rank import entropy
from skimage.morphology import disk

####################################################
###        NORMALİZASYON İŞLEMİ                 ###
####################################################
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

####################################################
###          GÜRÜLTÜ ELİMİNASYONU               ###
####################################################

# Apply noise reduction techniques
# Gaussian Blur
gaussian_blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

# Display Gaussian Blurred image separately
plt.figure(figsize=(8, 6))
plt.title("Gaussian Blurred Image (Final)")
plt.imshow(gaussian_blurred, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###          KESKİNLEŞTİRME                     ###
####################################################

# Convert Gaussian Blurred image to 8-bit for sharpening
gaussian_blurred_8bit = (gaussian_blurred * 255).astype(np.uint8)

# 2. Stronger Kernel Sharpening
kernel_sharpen_2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
sharpened_2 = cv2.filter2D(gaussian_blurred_8bit, -1, kernel_sharpen_2)

# Display Stronger Kernel Sharpened Image separately
plt.figure(figsize=(8, 6))
plt.title("Stronger Kernel Sharpened Image (Final)")
plt.imshow(sharpened_2, cmap='gray')
plt.axis('off')
plt.show()

####################################################
###        KONVOLÜSYON VE KORELASYON            ###
####################################################

# Define multiple kernels for different feature extraction
kernels = {
    "Sobel X (Horizontal Edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y (Vertical Edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Laplacian": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
}

# Dictionary to store results for both convolution and correlation
results = {"CLAHE Image (Final)": sharpened_2}

# Apply convolution for each kernel
for kernel_name, kernel in kernels.items():
    # Convolution
    conv_result = cv2.filter2D(sharpened_2, -1, kernel)
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
entropy_image = entropy(sobel_y_final, disk(5))

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

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import disk
from skimage.draw import line

####################################################
#         HoG UYGULAMASI                          #
####################################################
# Apply Histogram of Oriented Gradients (HoG)
hog_features, hog_image = hog(combined_result, visualize=True, block_norm='L2-Hys')

####################################################
#         CoHog UYGULAMASI                         #
####################################################
# Co-occurrence of Histograms of Oriented Gradients (CoHOG)
# We'll simplify this to using HoG over different orientations as a form of CoHOG analysis.
# Just taking another parameterized version of HoG with slightly altered block sizes
_, cohog_image = hog(combined_result, visualize=True, orientations=18, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

####################################################
#         LBP UYGULAMASI                           #
####################################################
# Apply Local Binary Pattern (LBP)
radius = 3
n_points = 8 * radius
lbp_image = local_binary_pattern(combined_result, n_points, radius, method='uniform').astype(np.uint8)

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

hough_image = hough_transform(combined_result)

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
