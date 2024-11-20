import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(image_path):
    """
    Görüntü ön işleme aşamaları:
    1) Normalizasyon (Kontrast artırma)
    2) Gürültü eliminasyonu (Yumuşatma)
    3) Histogram eşitleme ve CLAHE
    """
    # Görüntü yükleme ve grayscale'e çevirme
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")

    # Normalizasyon (Görüntünün kontrastını artırır)
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Histogram eşitleme (Kontrast artırma)
    equalized = cv2.equalizeHist(normalized)

    # CLAHE (Kontrastı sınırlı histogram eşitleme)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_equalized = clahe.apply(equalized)

    # Median bulanıklaştırma (Gürültü eliminasyonu)
    median_blurred = cv2.medianBlur(clahe_equalized, 5)

    # Gaussian bulanıklaştırma (Gürültü eliminasyonu)
    smoothed = cv2.GaussianBlur(median_blurred, (5, 5), 0)

    return image, normalized, equalized, clahe_equalized, median_blurred, smoothed


def select_roi(image):
    """
    Prostat bölgesine odaklanmak için ROI (Region of Interest) seçimi.
    """
    # ROI koordinatları belirleme (merkeze daha fazla odaklanma)
    h, w = image.shape
    x_start = int(w * 0.4)
    x_end = int(w * 0.6)
    y_start = int(h * 0.4)
    y_end = int(h * 0.6)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)


def adaptive_threshold(roi):
    """
    3) Adaptif eşikleme ile prostat bölgesini belirginleştirme.
    """
    # Parametrelerin optimize edilmesi
    binary_mask = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 4)
    return binary_mask


def refine_and_shape_mask(binary_mask, roi_coords, original_shape):
    """
    Maskeyi temizleme ve şekillendirme için morfolojik işlemler:
    - Morfolojik kapama ve açma işlemleriyle maskeyi iyileştirme.
    """
    x_start, y_start, x_end, y_end = roi_coords
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros(original_shape, dtype=np.uint8)

    if contours:
        # En büyük konturu seç (Organ dokusunu modellemek)
        largest_contour = max(contours, key=cv2.contourArea)

        # Morfolojik işlem: konveks zarf oluşturma ve konturu çizme
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(full_mask[y_start:y_end, x_start:x_end], [hull], -1, 255, thickness=cv2.FILLED)

        # Şekli düzleştirme ve boşlukları doldurma (Morfolojik iyileştirme)
        kernel = np.ones((9, 9), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # Daha fazla iyileştirme için erozyon ve genişleme işlemleri ekle
        full_mask = cv2.erode(full_mask, kernel, iterations=1)
        full_mask = cv2.dilate(full_mask, kernel, iterations=3)

    return full_mask


def process_image(image_path):
    """
    Segmentasyon sürecini çalıştırır:
    1) Görüntü ön işleme (Normalizasyon, histogram eşitleme, CLAHE ve yumuşatma)
    2) ROI seçimi (Prostat bölgesine odaklanma)
    3) Adaptif eşikleme
    4) Maskeyi temizleme ve şekillendirme (Morfolojik işlemler)
    """
    # Görüntü ön işleme
    original_image, normalized, equalized, clahe_equalized, median_blurred, preprocessed_image = preprocess_image(
        image_path)

    # ROI seçimi (Prostat bölgesine odaklanma)
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Adaptif eşikleme
    thresholded = adaptive_threshold(roi_image)

    # Maskeyi temizle ve şekillendir (Morfolojik işlemler)
    full_mask = refine_and_shape_mask(thresholded, roi_coords, preprocessed_image.shape)

    # Sonuçları görselleştir
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("Normalizasyon")
    plt.imshow(normalized, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("Histogram Eşitleme")
    plt.imshow(equalized, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title("CLAHE")
    plt.imshow(clahe_equalized, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.title("Median Blur")
    plt.imshow(median_blurred, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title("Gaussian Blur")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title("Adaptif Eşikleme")
    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("Segmented Output")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Segmentasyonu kaydet
    cv2.imwrite("outputs/img-output.png", full_mask)
    print("Segmentasyon sonucu 'outputs/img-output.png' olarak kaydedildi.")


# Girdi dosyası
image_path = ('data/img5.png')

process_image(image_path)
