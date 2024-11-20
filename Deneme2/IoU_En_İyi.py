import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_equalized = clahe.apply(equalized)

    # Bilateral filtreleme (Hem kenar koruma hem de gürültü azaltma)
    bilateral_filtered = cv2.bilateralFilter(clahe_equalized, d=9, sigmaColor=75, sigmaSpace=75)

    return image, normalized, equalized, clahe_equalized, bilateral_filtered

def select_roi(image):
    """
    Prostat bölgesine odaklanmak için ROI (Region of Interest) seçimi.
    """
    # ROI koordinatları belirleme (merkeze daha fazla odaklanma)
    h, w = image.shape
    x_start = int(w * 0.35)
    x_end = int(w * 0.65)
    y_start = int(h * 0.35)
    y_end = int(h * 0.65)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)

def adaptive_threshold(roi):
    """
    3) Adaptif eşikleme ile prostat bölgesini belirginleştirme.
    """
    # Parametrelerin optimize edilmesi
    binary_mask = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
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
        kernel = np.ones((7, 7), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # Daha fazla iyileştirme için erozyon ve genişleme işlemleri ekle
        full_mask = cv2.erode(full_mask, kernel, iterations=1)
        full_mask = cv2.dilate(full_mask, kernel, iterations=2)

    return full_mask

def calculate_iou(segmented_mask, ground_truth_mask):
    """
    IoU (Intersection over Union) hesaplama.
    """
    intersection = np.logical_and(segmented_mask, ground_truth_mask).sum()
    union = np.logical_or(segmented_mask, ground_truth_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def process_images(image_paths, label_paths):
    """
    Segmentasyon sürecini birden fazla görüntü için çalıştırır ve her biri için IoU hesaplar:
    1) Görüntü ön işleme (Normalizasyon, histogram eşitleme, CLAHE ve yumuşatma)
    2) ROI seçimi (Prostat bölgesine odaklanma)
    3) Adaptif eşikleme
    4) Maskeyi temizleme ve şekillendirme (Morfolojik işlemler)
    5) IoU hesaplama
    """
    total_iou = 0.0
    num_images = len(image_paths)

    for image_path, label_path in zip(image_paths, label_paths):
        print(f"Processing: {image_path} and {label_path}")
        try:
            # Görüntü ön işleme
            original_image, normalized, equalized, clahe_equalized, preprocessed_image = preprocess_image(
                image_path)

            # ROI seçimi (Prostat bölgesine odaklanma)
            roi_image, roi_coords = select_roi(preprocessed_image)

            # Adaptif eşikleme
            thresholded = adaptive_threshold(roi_image)

            # Maskeyi temizle ve şekillendir (Morfolojik işlemler)
            full_mask = refine_and_shape_mask(thresholded, roi_coords, preprocessed_image.shape)

            # Ground truth mask (Etiketli veri) yükle
            ground_truth_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth_mask is None:
                raise FileNotFoundError(f"{label_path} dosyası bulunamadı.")

            # Ground truth mask ve segmentasyon sonucunu ikili maske olarak kullan
            _, segmented_binary = cv2.threshold(full_mask, 127, 1, cv2.THRESH_BINARY)
            _, ground_truth_binary = cv2.threshold(ground_truth_mask, 127, 1, cv2.THRESH_BINARY)

            # IoU hesapla
            iou_score = calculate_iou(segmented_binary, ground_truth_binary)
            total_iou += iou_score

            print(f"IoU Skoru: {iou_score:.4f}")

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
            plt.title("Bilateral Filtreleme")
            plt.imshow(preprocessed_image, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 6)
            plt.title("Adaptif Eşikleme")
            plt.imshow(thresholded, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 7)
            plt.title(f"Segmented Output\nIoU: {iou_score:.4f}")
            plt.imshow(full_mask, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            # Segmentasyonu kaydet
            output_path = f"outputs/{image_path.split('/')[-1].replace('.png', '-output.png')}"
            cv2.imwrite(output_path, full_mask)
            print(f"Segmentasyon sonucu '{output_path}' olarak kaydedildi.")
        except FileNotFoundError as e:
            print(e)

    # IoU ortalamasını hesapla
    if num_images > 0:
        average_iou = total_iou / num_images
        print(f"Ortalama IoU Skoru: {average_iou:.4f}")

# Girdi dosyaları
image_paths = sorted(glob.glob('data/img*.png'))
label_paths = sorted(glob.glob('data/label*.png'))

# Segmentasyon ve IoU hesaplama
process_images(image_paths, label_paths)
