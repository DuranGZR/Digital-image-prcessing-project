import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern, hog
from sklearn.mixture import GaussianMixture


def preprocess_image(image_path):
    """Görüntü ön işleme: Normalizasyon, gürültü giderme, histogram eşitleme ve keskinleştirme."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")

    # Normalizasyon
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Histogram Eşitleme
    equalized = cv2.equalizeHist(normalized)

    # Gürültü Eliminasyonu (Gaussian Blur ile yumuşatma)
    smoothed = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Keskinleştirme (Laplacian ile)
    sharpened = cv2.Laplacian(smoothed, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(sharpened)

    return sharpened


def select_roi(image):
    """Prostat bölgesine odaklanmak için ROI seçimi."""
    h, w = image.shape
    x_start = int(w * 0.3)
    x_end = int(w * 0.7)
    y_start = int(h * 0.4)
    y_end = int(h * 0.65)
    return image[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)


def otsu_threshold(image):
    """Otsu eşikleme ile prostat bölgesini belirginleştirir."""
    _, otsu_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_mask


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

        # Şekli düzleştir ve boşlukları doldur (Morfolojik operasyonlar: Kapatma ve Açma)
        kernel = np.ones((7, 7), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    return full_mask


def extract_features(image):
    """Organ dokusunu modellemek için özellik çıkarımı: GMM, HoG, LBP."""
    # Local Binary Pattern (LBP)
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")

    # Histogram of Oriented Gradients (HoG)
    hog_features, _ = hog(image, visualize=True)

    # Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=2, random_state=0)
    image_flat = image.flatten().reshape(-1, 1)
    gmm.fit(image_flat)
    gmm_labels = gmm.predict(image_flat).reshape(image.shape)

    return lbp, hog_features, gmm_labels


def calculate_iou(predicted_mask, ground_truth_mask):
    """IoU hesaplama ve karşılaştırma."""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def process_image(image_path, ground_truth_path=None):
    """Segmentasyon sürecini çalıştırır."""
    preprocessed_image = preprocess_image(image_path)

    # ROI seçimi
    roi_image, roi_coords = select_roi(preprocessed_image)

    # Otsu eşikleme
    otsu_mask = otsu_threshold(roi_image)

    # Adaptif eşikleme
    thresholded = adaptive_threshold(roi_image)

    # Maskeyi temizle ve şekillendir
    full_mask = refine_and_shape_mask(thresholded, roi_coords, preprocessed_image.shape)

    # Özellik çıkarımı
    lbp, hog_features, gmm_labels = extract_features(preprocessed_image)

    # Görselleştir
    plt.subplot(1, 4, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Otsu Eşikleme")
    plt.imshow(otsu_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Adaptif Eşikleme")
    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Segmented Output")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Segmentasyonu kaydet
    cv2.imwrite("segmented_output_adaptive_shape.png", full_mask)
    print("Segmentasyon sonucu 'segmented_output_adaptive_shape.png' olarak kaydedildi.")

    # IoU hesaplama
    if ground_truth_path:
        ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            raise FileNotFoundError(f"{ground_truth_path} dosyası bulunamadı.")
        iou = calculate_iou(full_mask, ground_truth_mask)
        print(f"IoU değeri: {iou}")


# Girdi dosyası
image_path = '../Deneme2/data/img8.png'
ground_truth_path = 'data/ground_truth.png'

process_image(image_path, ground_truth_path)

# Ar-Ge aşamalarına göre eksiklikler ve düzenlemeler:
# 1. Görüntü ön işleme:
#    - Normalizasyon, gürültü giderme, histogram eşitleme ve keskinleştirme eklendi.
#    - Konvolüyon/korelasyon, entropi hesaplama gibi eklemeler gerekebilir.
# 2. Organ dokusu modelleme:
#    - GMM, HoG, LBP gibi teknikler eklendi.
# 3. Otsu eşikleme:
#    - Otsu eşikleme eklendi.
# 4. Morfolojik operasyonlar:
#    - Kapatma ve açma işlemleri doğru seçildi.
# 5. Farklı kişilerin görüntüleri üzerinde denenme gerekliliği var.
# 6. IoU hesaplama ve LABEL ile karşılaştırma eklendi.
