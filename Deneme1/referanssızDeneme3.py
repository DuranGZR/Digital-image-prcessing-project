import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.feature import hog, local_binary_pattern
from skimage import color


def preprocess_image(image_path):
    """Görüntüyü normalize eder ve bulanıklaştırır."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)  # Gaussian bulanıklaştırma
    return smoothed

def select_roi(image):
    """Prostat bölgesine odaklanmak için ROI seçimi."""
    h, w = image.shape
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)
    y_start = int(h * 0.35)
    y_end = int(h * 0.65)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)

def apply_gmm(roi):
    """GMM kullanarak prostat bölgesini segment eder."""
    roi_flat = roi.reshape((-1, 1))
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(roi_flat)
    segmented = gmm.predict(roi_flat).reshape(roi.shape)
    return (segmented * 255).astype(np.uint8)

def apply_hog(roi):
    """HOG ile prostat bölgesinin özelliklerini çıkarır."""
    features, hog_image = hog(roi, visualize=True, block_norm='L2-Hys')
    return (hog_image * 255).astype(np.uint8)

def apply_cohog(roi):
    """CoHOG yöntemini uygular."""
    edges = cv2.Canny(roi, 50, 150)
    return edges

def apply_lbp(roi):
    """LPB ile prostat bölgesini belirginleştirir."""
    lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    return (lbp * (255 / lbp.max())).astype(np.uint8)

def apply_hough(roi):
    """Hough dönüşümü ile prostat bölgesindeki dairesel yapıları tespit eder."""
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=5, maxRadius=50)
    hough_image = np.zeros_like(roi)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(hough_image, (x, y), r, 255, 2)
    return hough_image

def adaptive_threshold(roi):
    """Adaptif eşikleme ile prostat bölgesini belirginleştirir."""
    binary_mask = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    return binary_mask

def refine_and_shape_mask(binary_mask, roi_coords, original_shape):
    """Maskeyi temizler ve şekillendir."""
    x_start, y_start, x_end, y_end = roi_coords
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros(original_shape, dtype=np.uint8)

    if contours:
        # En büyük konturu seç
        largest_contour = max(contours, key=cv2.contourArea)

        # Morfolojik işlemlerle iyileştir
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(full_mask[y_start:y_end, x_start:x_end], [hull], -1, 255, thickness=cv2.FILLED)

        # Şekli düzleştir ve boşlukları doldur
        kernel = np.ones((7, 7), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    return full_mask

def process_image(image_path):
    """Segmentasyon sürecini çalıştırır ve çeşitli yöntemlerle görüntüleri işler."""
    preprocessed_image = preprocess_image(image_path)

    # ROI seçimi
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Farklı algoritmalar ile işlem
    gmm_segmented = apply_gmm(roi_image)
    hog_image = apply_hog(roi_image)
    cohog_image = apply_cohog(roi_image)
    lbp_image = apply_lbp(roi_image)
    hough_image = apply_hough(roi_image)
    adaptive_thresh_image = adaptive_threshold(roi_image)

    # Maskeyi temizle ve şekillendir
    gmm_mask = refine_and_shape_mask(gmm_segmented, roi_coords, preprocessed_image.shape)
    adaptive_thresh_mask = refine_and_shape_mask(adaptive_thresh_image, roi_coords, preprocessed_image.shape)

    # Görselleştir
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 4, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("GMM Segmentasyonu")
    plt.imshow(gmm_segmented, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("HOG Görüntüsü")
    plt.imshow(hog_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title("CoHOG (Canny Edges)")
    plt.imshow(cohog_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.title("LPB Görüntüsü")
    plt.imshow(lbp_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title("Hough Dönüşümü")
    plt.imshow(hough_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title("Adaptif Eşikleme")
    plt.imshow(adaptive_thresh_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("GMM Mask")
    plt.imshow(gmm_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Sonuçları kaydet
    cv2.imwrite("gmm_segmented.png", gmm_segmented)
    cv2.imwrite("hog_image.png", hog_image)
    cv2.imwrite("cohog_image.png", cohog_image)
    cv2.imwrite("lbp_image.png", lbp_image)
    cv2.imwrite("hough_image.png", hough_image)
    cv2.imwrite("adaptive_thresh_image.png", adaptive_thresh_image)
    cv2.imwrite("gmm_mask.png", gmm_mask)
    cv2.imwrite("adaptive_thresh_mask.png", adaptive_thresh_mask)
    print("Sonuçlar kaydedildi.")

# Girdi dosyası
image_path = '../Deneme2/data/img5.png'

process_image(image_path)
