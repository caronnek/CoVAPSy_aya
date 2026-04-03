#!/usr/bin/env python3
"""
Script caméra pour Raspberry Pi 4 - Camera Module 2
Utilise l'API picamera2 (basée sur libcamera)
"""

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import FileOutput
import time
import os

def afficher_infos_camera(cam):
    """Affiche les infos et propriétés de la caméra"""
    print("=== Infos Caméra ===")
    print(f"Modèle    : {cam.camera_properties.get('Model', 'Inconnu')}")
    print(f"Résolution: {cam.camera_properties.get('PixelArraySize', 'N/A')}")
    print(f"Config    : {cam.camera_config}")
    print()

def prendre_photo(cam, chemin="photo.jpg"):
    """Capture une photo et la sauvegarde"""
    print(f"📸 Capture photo -> {chemin}")
    cam.capture_file(chemin)
    taille = os.path.getsize(chemin) / 1024
    print(f"   Sauvegardée ({taille:.1f} Ko)")

def apercu_live(cam, duree=5):
    """Affiche un aperçu live pendant N secondes (nécessite écran)"""
    print(f"👁️  Aperçu live pendant {duree}s...")
    cam.start_preview(Preview.QTGL)
    time.sleep(duree)
    cam.stop_preview()

def enregistrer_video(cam, chemin="video.h264", duree=5):
    """Enregistre une vidéo H264"""
    print(f"🎥 Enregistrement vidéo {duree}s -> {chemin}")
    encoder = H264Encoder(bitrate=10000000)
    cam.start_recording(encoder, chemin)
    time.sleep(duree)
    cam.stop_recording()
    taille = os.path.getsize(chemin) / 1024
    print(f"   Sauvegardée ({taille:.1f} Ko)")

def capturer_raw(cam, chemin="capture.dng"):
    """Capture une image RAW DNG (qualité maximale)"""
    print(f"📷 Capture RAW -> {chemin}")
    config = cam.create_still_configuration(raw={})
    cam.configure(config)
    cam.start()
    cam.capture_file(chemin, name="raw")
    cam.stop()
    print("   Fichier DNG sauvegardé")

# ─── Programme principal ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialisation Camera Module 2...")

    with Picamera2() as cam:
        # Configuration haute résolution (3280x2464 pour le Camera Module 2)
        config = cam.create_still_configuration(
            main={"size": (3280, 2464), "format": "RGB888"},
            lores={"size": (640, 480)},   # flux basse résolution pour aperçu
            display="lores"
        )
        cam.set_controls({"HflipAuto": False, "VflipAuto": False})
        cam.set_controls({"HflipAuto": False, "VflipAuto": False})
        cam.configure(config)
        cam.start()
        time.sleep(2)  # laisser l'AE/AWB se stabiliser

        afficher_infos_camera(cam)

        # --- Actions ---
        prendre_photo(cam, "photo_hd.jpg")

        # Décommenter selon besoin :
        # apercu_live(cam, duree=5)          # nécessite un écran connecté
        # enregistrer_video(cam, duree=5)    # vidéo 5 secondes
        # capturer_raw(cam)                  # RAW DNG

    print("\nTerminé.")