#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# Réglages
# =========================
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30
BAND_RATIO = 0.30          # hauteur de la bande centrale (30% de l'image)
UNKNOWN_VALUE = -1         # valeur renvoyée si la zone n'est ni rouge ni verte
SHOW_WINDOW = True         # mettre False si tu es en SSH sans affichage

# HSV à ajuster si besoin selon ton éclairage réel
LOWER_RED1 = np.array([0, 90, 60], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)

LOWER_RED2 = np.array([170, 90, 60], dtype=np.uint8)
UPPER_RED2 = np.array([180, 255, 255], dtype=np.uint8)

LOWER_GREEN = np.array([40, 70, 50], dtype=np.uint8)
UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8)

MIN_RATIO = 0.03           # au moins 3% de la zone
DOMINANCE = 1.15           # la couleur gagnante doit battre l'autre de 15%

FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_color_hsv(roi_bgr):
    """
    Retourne:
        color_name: 'rouge' | 'vert' | 'inconnu'
        bit_value : 0 | 1 | UNKNOWN_VALUE
        red_ratio : proportion de rouge dans la ROI
        green_ratio : proportion de vert dans la ROI
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Petit nettoyage du bruit
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    total = roi_bgr.shape[0] * roi_bgr.shape[1]
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)

    red_ratio = red_count / total
    green_ratio = green_count / total

    if red_ratio >= MIN_RATIO and red_ratio > green_ratio * DOMINANCE:
        return "rouge", 0, red_ratio, green_ratio

    if green_ratio >= MIN_RATIO and green_ratio > red_ratio * DOMINANCE:
        return "vert", 1, red_ratio, green_ratio

    return "inconnu", UNKNOWN_VALUE, red_ratio, green_ratio


def analyze_frame(frame_bgr):
    """
    Analyse une bande horizontale centrale.
    Découpe en 3 zones: gauche / centre / droite.
    Retourne:
        annotated_frame
        colors  -> liste de 3 chaînes
        bits    -> liste de 3 valeurs
    """
    h, w, _ = frame_bgr.shape
    out = frame_bgr.copy()

    # Bande centrale
    band_h = int(h * BAND_RATIO)
    y1 = (h - band_h) // 2
    y2 = y1 + band_h

    band = frame_bgr[y1:y2, :]
    third = w // 3

    rois = [
        band[:, 0:third],         # gauche
        band[:, third:2*third],   # centre
        band[:, 2*third:w]        # droite
    ]

    zone_names = ["G", "C", "D"]
    colors = []
    bits = []

    # Dessin de la bande et des séparations
    cv2.rectangle(out, (0, y1), (w - 1, y2 - 1), (255, 255, 255), 2)
    cv2.line(out, (third, y1), (third, y2), (255, 255, 255), 2)
    cv2.line(out, (2 * third, y1), (2 * third, y2), (255, 255, 255), 2)

    for i, roi in enumerate(rois):
        color_name, bit_value, red_ratio, green_ratio = detect_color_hsv(roi)
        colors.append(color_name)
        bits.append(bit_value)

        x_text = 10 + i * third
        label = f"{zone_names[i]}: {color_name} -> {bit_value}"
        stats = f"R={red_ratio:.2f} V={green_ratio:.2f}"

        cv2.putText(out, label, (x_text, max(25, y1 - 30)),
                    FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(out, stats, (x_text, max(50, y1 - 5)),
                    FONT, 0.5, (255, 255, 255), 1)

    # Résultat global
    cv2.putText(out, f"bits = {bits}", (10, h - 20),
                FONT, 0.8, (0, 255, 255), 2)

    return out, colors, bits


def main():
    # Pour SSH/headless, désactive la fenêtre
    show_window = SHOW_WINDOW and ("DISPLAY" in os.environ or os.name == "nt")

    picam2 = Picamera2()

    # RGB888 est le bon choix avec OpenCV sur Picamera2
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
        queue=False
    )
    picam2.configure(config)
    picam2.start()

    # Petite pause pour laisser l'auto-exposition se stabiliser
    time.sleep(1.0)

    prev_bits = None
    last_print = 0.0
    last_t = time.time()

    try:
        while True:
            frame = picam2.capture_array("main")   # image NumPy
            # On traite directement frame avec OpenCV
            annotated, colors, bits = analyze_frame(frame)

            # FPS affiché
            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            cv2.putText(annotated, f"{fps:.1f} fps", (10, 30),
                        FONT, 0.8, (0, 255, 0), 2)

            # Affichage terminal seulement si changement ou toutes les 0.5 s
            if bits != prev_bits or (now - last_print) > 0.5:
                print(f"colors={colors}  bits={bits}")
                prev_bits = bits
                last_print = now

            if show_window:
                cv2.imshow("Test camera murs rouge/vert", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Mode sans fenêtre (SSH/headless)
                time.sleep(0.01)

    finally:
        picam2.stop()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()