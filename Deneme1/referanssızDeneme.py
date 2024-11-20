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


def dynamic_threshold(roi):
    """Dinamik yoğunluk eşikleme."""
    mean_intensity = np.mean(roi)
    lower_bound = mean_intensity - 25
    upper_bound = mean_intensity + 25
    binary_mask = cv2.inRange(roi, lower_bound, upper_bound)
    return binary_mask


def refine_and_shape_mask(binary_mask, roi_coords, original_shape):
    """Maskeyi temizler ve şekillendir."""
    x_start, y_start, x_end, y_end = roi_coords
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros(original_shape, dtype=np.uint8)

    if contours:
        # En büyük konturu seç
        largest_contour = max(contours, key=cv2.contourArea)

        # Konturları düzgünleştir
        hull = cv2.convexHull(largest_contour)
        center, radius = cv2.minEnclosingCircle(hull)
        center = tuple(map(int, center))
        radius = int(radius * 0.9)  # Yarıçapı optimize et
        cv2.circle(full_mask, (center[0] + x_start, center[1] + y_start), radius, 255, thickness=cv2.FILLED)

    return full_mask


def process_image(image_path):
    """Segmentasyon sürecini çalıştırır."""
    preprocessed_image = preprocess_image(image_path)

    # ROI seçimi
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Dinamik eşikleme
    thresholded = dynamic_threshold(roi_image)

    # Maskeyi temizle ve şekillendir
    full_mask = refine_and_shape_mask(thresholded, roi_coords, preprocessed_image.shape)

    # Görselleştir
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Dinamik Eşikleme")
    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Output")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Segmentasyonu kaydet
    cv2.imwrite("segmented_output_final.png", full_mask)
    print("Segmentasyon sonucu 'segmented_output_final.png' olarak kaydedildi.")


# Girdi dosyası
image_path = '../Deneme2/data/img10.png'

process_image(image_path)
