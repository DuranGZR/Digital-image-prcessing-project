import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Normalization
def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 2. Noise Elimination (Gaussian Smoothing)
def smooth_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 3. Sharpening (Laplacian Filter)
def sharpen_image(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.addWeighted(image, 1.5, laplacian.astype(np.uint8), -0.5, 0)

# 4. Transformations (Rotation Example)
def transform_image(image, angle=45, scale=1):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# 5. Histogram Equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# 6. Convolution (Sobel Edge Detection)
def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely).astype(np.uint8)
    return sobel

# 7. Entropy, Range, and Std Filters
def calculate_entropy(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
    hist_norm = hist / hist.sum()
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return entropy

# Apply the steps to an example image
image = cv2.imread('data/img10.png', cv2.IMREAD_GRAYSCALE)

# Sequential processing
normalized_image = normalize_image(image)
smoothed_image = smooth_image(normalized_image)
sharpened_image = sharpen_image(smoothed_image)
transformed_image = transform_image(sharpened_image)
hist_eq_image = histogram_equalization(normalized_image)
sobel_image = sobel_edge_detection(smoothed_image)

# Calculate entropy, range, and standard deviation
entropy = calculate_entropy(normalized_image)
pixel_range = np.ptp(image)  # Max - Min
std_deviation = np.std(image)

# Display results
titles = ['Normalized', 'Smoothed', 'Sharpened', 'Transformed', 'Histogram Equalized', 'Sobel Edge Detection']
images = [normalized_image, smoothed_image, sharpened_image, transformed_image, hist_eq_image, sobel_image]

plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Print statistics
print(f"Entropy: {entropy}")
print(f"Pixel Range: {pixel_range}")
print(f"Standard Deviation: {std_deviation}")
