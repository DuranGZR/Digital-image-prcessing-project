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

def fill_holes(mask):
    """Maskenin içindeki boşlukları doldurur."""
    # Ters çevirerek flood fill uygula
    h, w = mask.shape[:2]
    flood_filled = mask.copy()
    mask_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask_fill, (0, 0), 255)

    # Ters çevir ve doldurulmuş alanlarla birleştir
    inverted_flood = cv2.bitwise_not(flood_filled)
    filled_mask = cv2.bitwise_or(mask, inverted_flood)
    return filled_mask

def refine_mask(mask):
    """Maskeyi temizler, boşlukları doldurur ve şekli düzeltir."""
    # Morfolojik kapama işlemi
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Küçük boşlukları doldurma
    filled = fill_holes(closed)

    # Kenarları pürüzsüzleştirme
    smoothed = cv2.GaussianBlur(filled, (5, 5), 0)
    return smoothed

def segment_prostate(image, label_path, lower_bound, upper_bound):
    """Referans etikete göre prostat segmentasyonu."""
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Yoğunluk eşikleme
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Morfolojik işlemlerle temizleme ve dolgu
    refined = refine_mask(mask)

    # Referans etiketten kontur al ve maskeye uygula
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(image, dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Maskeyi referans etikete göre birleştirme
    combined_mask = cv2.bitwise_and(refined, final_mask)
    return combined_mask

def process_image(image_path, label_path):
    """Tüm segmentasyon sürecini çalıştırır."""
    preprocessed_image = preprocess_image(image_path)

    # Referans etiketten yoğunluk aralığını belirle
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    mask = (label > 0).astype(np.uint8)
    prostate_pixels = preprocessed_image[mask > 0]
    lower_bound = int(np.percentile(prostate_pixels, 30))  # Alt sınır
    upper_bound = int(np.percentile(prostate_pixels, 70))  # Üst sınır
    print(f"Optimize Edilmiş Yoğunluk Aralığı: {lower_bound}-{upper_bound}")

    # Segmentasyon
    segmented = segment_prostate(preprocessed_image, label_path, lower_bound, upper_bound)

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
    cv2.imwrite("segmented_output_filled.png", segmented)
    print("Segmentasyon sonucu 'segmented_output_filled.png' olarak kaydedildi.")

# Girdi dosyaları
image_path = 'data/img10.png'
label_path = 'data/label10.png'

process_image(image_path, label_path)
