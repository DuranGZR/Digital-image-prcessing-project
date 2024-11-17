import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path):
    """Görüntüyü normalize eder ve bulanıklaştırır."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    return smoothed

def select_roi(image):
    """Prostat bölgesine odaklanmak için ROI seçimi."""
    h, w = image.shape
    x_start = int(w * 0.3)
    x_end = int(w * 0.7)
    y_start = int(h * 0.4)
    y_end = int(h * 0.65)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)

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
    """Segmentasyon sürecini çalıştırır."""
    preprocessed_image = preprocess_image(image_path)

    # ROI seçimi
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Adaptif eşikleme
    thresholded = adaptive_threshold(roi_image)

    # Maskeyi temizle ve şekillendir
    full_mask = refine_and_shape_mask(thresholded, roi_coords, preprocessed_image.shape)

    # Görselleştir
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Adaptif Eşikleme")
    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Output")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Segmentasyonu kaydet
    cv2.imwrite("segmented_output_adaptive_shape.png", full_mask)
    print("Segmentasyon sonucu 'segmented_output_adaptive_shape.png' olarak kaydedildi.")

# Girdi dosyası
image_path = 'data/img8.png'

process_image(image_path)
