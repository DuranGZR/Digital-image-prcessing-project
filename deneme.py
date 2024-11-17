import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(image_path):
    """Görüntüyü yükleyip normalize eder ve bulanıklaştırır."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{image_path} dosyası bulunamadı.")
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    return smoothed


def segment_prostate_region_growing(image, seed_point, threshold=20):
    """Region Growing algoritması ile prostat segmentasyonu."""
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    stack = [seed_point]
    original_intensity = image[seed_point]  # Başlangıç noktasının yoğunluk değeri

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True

        # Komşuları kontrol et
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and not visited[nx, ny]:
                if abs(int(image[nx, ny]) - int(original_intensity)) <= threshold:
                    segmented[nx, ny] = 255
                    stack.append((nx, ny))
    return segmented


def process_image(image_path, seed_point, threshold=20):
    """Bir görüntüyü işler ve prostat bölgesini segmentler."""
    preprocessed_image = preprocess_image(image_path)
    segmented = segment_prostate_region_growing(preprocessed_image, seed_point, threshold)

    # Görselleştirme
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(preprocessed_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Output")
    plt.imshow(segmented, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return segmented


# Örnek Görüntü ve Segmentasyon
image_path = 'data/img10.png'  # Görüntü dosya yolu
seed_point = (240, 320)  # Prostatın tahmini bir noktası
threshold = 15  # Segmentasyon için yoğunluk eşik değeri

segmented_output = process_image(image_path, seed_point, threshold)
