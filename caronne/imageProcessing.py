import numpy as np
import cv2


def webots_image_to_bgr(image, width, height):
    """
    Convertit l'image Webots (BGRA) en image OpenCV BGR.
    """
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img_bgr


def detect_color_hsv(roi_bgr):
    """
    Détecte la couleur dominante dans une ROI.
    Retourne : 'rouge', 'vert' ou 'inconnu'
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Rouge en 2 plages HSV
    lower_red1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 80, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Vert
    lower_green = np.array([40, 50, 50], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    nb_red = cv2.countNonZero(mask_red)
    nb_green = cv2.countNonZero(mask_green)

    seuil_min = int(0.05 * roi_bgr.shape[0] * roi_bgr.shape[1])

    if nb_red > nb_green and nb_red > seuil_min:
        return "rouge", 0
    elif nb_green > nb_red and nb_green > seuil_min:
        return "vert", 1
    else:
        return "inconnu", None


def analyze_walls(image_bgr, band_ratio=0.30):
    """
    Garde une bande horizontale centrale de l'image,
    la divise en 3 parties : gauche / centre / droite,
    puis retourne les couleurs détectées.

    Retourne :
    {
        "original_annotated": image complète annotée avec la bande centrale,
        "band": bande horizontale seule, divisée en 3,
        "colors": [couleur_gauche, couleur_centre, couleur_droite]
    }
    """
    h, w, _ = image_bgr.shape

    original_annotated = image_bgr.copy()

    # =========================
    # Bande horizontale centrale
    # =========================
    band_h = int(h * band_ratio)
    y1 = (h - band_h) // 2
    y2 = y1 + band_h

    band = image_bgr[y1:y2, :].copy()

    # Dessin de la bande sur l'image originale
    cv2.rectangle(original_annotated, (0, y1), (w - 1, y2 - 1), (255, 255, 255), 2)
    cv2.putText(
        original_annotated,
        "",
        (10, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =========================
    # Division en 3 zones
    # =========================
    third = w // 3

    zone_left = band[:, 0:third]
    zone_center = band[:, third:2 * third]
    zone_right = band[:, 2 * third:w]

    color_left, left_value = detect_color_hsv(zone_left)
    color_center, center_value = detect_color_hsv(zone_center)
    color_right, right_value = detect_color_hsv(zone_right)

    # lignes de séparation sur la bande
    cv2.line(band, (third, 0), (third, band.shape[0] - 1), (255, 255, 255), 2)
    cv2.line(band, (2 * third, 0), (2 * third, band.shape[0] - 1), (255, 255, 255), 2)

    # textes sur la bande
    cv2.putText(band, color_left, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(band, color_center, (third + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(band, color_right, (2 * third + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return {
        "original_annotated": original_annotated,
        "band": band,
        "colors": [color_left, color_center, color_right],
        "value" : [left_value,center_value, right_value]
    }