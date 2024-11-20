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

def manual_histogram_analysis(image, label_path):
    """Manuel olarak histogram analizi yaparak yoğunluk aralığını kontrol et."""
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    mask = (label > 0).astype(np.uint8)
    prostate_pixels = image[mask > 0]
    plt.hist(prostate_pixels, bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title("Prostatın Yoğunluk Histogramı")
    plt.xlabel("Yoğunluk Değerleri")
    plt.ylabel("Frekans")
    plt.show()
    return prostate_pixels

def segment_prostate(image, roi_coords, lower_bound, upper_bound):
    """Yoğunluk eşikleme ve morfolojik işlemlerle prostat segmentasyonu."""
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]  # ROI seçimi

    # Yoğunluk eşikleme
    mask = cv2.inRange(roi, lower_bound, upper_bound)

    # Morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # En büyük konturu seç ve ROI'ye genişlet
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_mask = np.zeros_like(image, dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(full_mask[y:y+h, x:x+w], [largest_contour], -1, 255, thickness=cv2.FILLED)

    return full_mask

def process_image(image_path, label_path):
    """Tüm segmentasyon sürecini çalıştırır."""
    preprocessed_image = preprocess_image(image_path)

    # Referans etiketiyle yoğunluk aralığını manuel olarak kontrol et
    prostate_pixels = manual_histogram_analysis(preprocessed_image, label_path)

    # Yoğunluk aralığını manuel belirle
    lower_bound = np.percentile(prostate_pixels, 10)
    upper_bound = np.percentile(prostate_pixels, 90)
    print(f"Manuel Yoğunluk Aralığı: {lower_bound}-{upper_bound}")

    # ROI (prostatın bulunduğu alanı tahmin et)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi_coords = (x, y, w, h)

    # Segmentasyon
    segmented = segment_prostate(preprocessed_image, roi_coords, lower_bound, upper_bound)

    # Görselleştir
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmented Output")
    plt.imshow(segmented, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Referans Etiket")
    plt.imshow(label, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Segmentasyonu kaydet
    cv2.imwrite("segmented_output.png", segmented)
    print("Segmentasyon sonucu 'segmented_output.png' olarak kaydedildi.")

# Girdi dosyaları
image_path = '../Deneme2/data/img10.png'
label_path = '../Deneme2/data/label10.png'

process_image(image_path, label_path)
