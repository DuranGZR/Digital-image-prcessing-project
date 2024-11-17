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

def analyze_label(label_path):
    """Referans etiketin özelliklerini analiz ederek prostatın yoğunluk aralığını ve ROI'yi belirler."""
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"{label_path} dosyası bulunamadı.")

    # Etiketin bounding box'ını (ROI) bul
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Etikette prostat konturu bulunamadı.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Etiketin yoğunluk değerlerini analiz et
    mask = (label > 0).astype(np.uint8)
    label_pixels = label[mask > 0]
    lower_bound = int(np.percentile(label_pixels, 5))  # Alt yoğunluk sınırı
    upper_bound = int(np.percentile(label_pixels, 95))  # Üst yoğunluk sınırı

    return (x, y, w, h), lower_bound, upper_bound

def visualize_roi(image, roi_coords):
    """ROI'yi görüntü üzerinde görselleştirir."""
    x, y, w, h = roi_coords
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(vis_image)
    plt.title("ROI Görselleştirme")
    plt.axis("off")
    plt.show()

def segment_prostate(image, roi_coords, lower_bound, upper_bound, min_area=500):
    """Referans etikete dayalı prostat segmentasyonu."""
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]  # ROI seçimi

    # Yoğunluk eşikleme
    mask = cv2.inRange(roi, lower_bound, upper_bound)

    # Morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Kontur görselleştirme
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = np.zeros_like(roi, dtype=np.uint8)
    cv2.drawContours(contour_vis, contours, -1, 255, 1)
    plt.imshow(contour_vis, cmap="gray")
    plt.title("Kontur Görselleştirme")
    plt.axis("off")
    plt.show()

    # En büyük bölgeyi seç
    final_mask = np.zeros_like(roi, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Minimum alan filtresi
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Maskeyi ROI dışına genişlet
    full_mask = np.zeros_like(image, dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = final_mask

    return full_mask

def process_image(image_path, label_path, min_area=500):
    """Bir görüntüyü işler, prostatı segmentler ve doğruluğu ölçer."""
    preprocessed_image = preprocess_image(image_path)

    # Referans etiketi analiz et
    roi_coords, lower_bound, upper_bound = analyze_label(label_path)
    print(f"ROI Koordinatları: {roi_coords}")
    print(f"Yoğunluk Aralığı: {lower_bound}-{upper_bound}")

    # ROI'yi görselleştir
    visualize_roi(preprocessed_image, roi_coords)

    # Segmentasyon
    segmented = segment_prostate(preprocessed_image, roi_coords, lower_bound, upper_bound, min_area)

    # Segmentasyon sonucunu görselleştir
    plt.imshow(segmented, cmap="gray")
    plt.title("Segmented Output")
    plt.axis("off")
    plt.show()

    return segmented

# Örnek Görüntü ve Segmentasyon
image_path = 'data/img10.png'
label_path = 'data/label10.png'

min_area = 1000  # Minimum alan filtresi (piksel cinsinden)

segmented_output = process_image(image_path, label_path, min_area)

# Segmentasyon sonucunu kaydetme
cv2.imwrite("segmented_output.png", segmented_output)
print("Segmentasyon sonucu 'segmented_output.png' olarak kaydedildi.")
