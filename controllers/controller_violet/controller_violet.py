# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Adaptation : contrôleur neuronal virtuel différentiel
# reprojeté en commandes Ackermann
# + filtrage moyenneur des données LiDAR
# + utilisation de 5 points exacts :
#   gauche: 60° et 70°
#   front : 0°
#   droite: -60° et -70°

from vehicle import Driver
from controller import Lidar
import numpy as np
import cv2

from commun import filtre_moyenneur, lire_point_lidar, normaliser_distance, differentiel_vers_ackermann, calculer_commande_auto
import config
# =========================
# Paramètres véhicule
# =========================
maxSpeed = config.VITESSE_AUTO_MAX_M_S * 3.6       # km/h
maxangle_degre = config.ANGLE_DEGRE_MAX                      # degrés

# --- Paramètres géométriques du TT-02 (à ajuster selon le modèle Webots) ---
L_entraxe = config.L_ENTRAXE_M  # m  — distance entre roues gauche/droite (voie)
W_empattement = config.W_EMPATTEMENT_M  # m  — distance entre essieu avant et arrière

# =========================
# Fonctions véhicule propre à Weebots
# =========================
driver = Driver()

def set_vitesse_m_s(vitesse_m_s):
    speed = vitesse_m_s * 3.6
    if speed > maxSpeed:
        speed = maxSpeed
    if speed < -maxSpeed:
        speed = -maxSpeed
    driver.setCruisingSpeed(speed)

def set_direction_degre(angle_degre):
    if angle_degre > maxangle_degre:
        angle_degre = maxangle_degre
    elif angle_degre < -maxangle_degre:
        angle_degre = -maxangle_degre

    angle_rad = -angle_degre * np.pi / 180.0
    driver.setSteeringAngle(angle_rad)
    
# =========================
# Fonctions d'execution pour Weebots
# =========================
def main():
    # =========================
    # Mode de fonctionnement
    # =========================
    modeAuto = False
    print("CoVAPSy — Conduite Autonome pour Weebots")
    print("Cliquer sur la vue 3D pour commencer")
    print("a : mode auto")
    print("n : stop")
    
    # =========================
    # Initialisation du Driver Weebots
    # =========================
    basicTimeStep = int(driver.getBasicTimeStep())
    sensorTimeStep = 4 * basicTimeStep
    driver.setSteeringAngle(0)
    driver.setCruisingSpeed(0)
    
    # =========================
    # Initialisation du LiDAR
    # =========================
    lidar = Lidar("RpLidarA2")
    lidar.enable(sensorTimeStep)
    lidar.enablePointCloud()
    tableau_lidar_mm = [0] * 360
    
    # =========================
    # Initialisation clavier
    # =========================
    keyboard = driver.getKeyboard()
    keyboard.enable(sensorTimeStep)
    
    # =========================
    # Initialisation caméra
    # =========================
    camera = driver.getDevice("pi_camera")
    camera_ok = False
    if camera is None:
        print("Camera non trouvée : pi_camera")
    else:
        camera.enable(sensorTimeStep)
        camera_ok = True
        print("Camera trouvée :", camera.getName())
        print("Resolution camera :", camera.getWidth(), "x", camera.getHeight())  


    while driver.step() != -1:
        # =========================
        # Lecture caméra
        # =========================
        if camera_ok:
            image = camera.getImage()

            # protection contre NULL pointer
            if image is not None:
                width = camera.getWidth()
                height = camera.getHeight()

                # conversion Webots -> numpy
                image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))

                # RGBA -> BGR (OpenCV)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

                cv2.imshow("Camera TT02", image_bgr)
                cv2.waitKey(1)
        # =========================
        # Lecture clavier
        # =========================
        while True:
            currentKey = keyboard.getKey()

            if currentKey == -1:
                break

            elif currentKey == ord('n') or currentKey == ord('N'):
                if modeAuto:
                    modeAuto = False
                    print("-------- Mode Auto Désactivé -------")

            elif currentKey == ord('a') or currentKey == ord('A'):
                if not modeAuto:
                    modeAuto = True
                    print("-------- Mode Auto Activé -------")

        # =========================
        # Acquisition LiDAR brut
        # =========================
        donnees_lidar_brutes = lidar.getRangeImage()

        for i in range(360):
            if (donnees_lidar_brutes[-i] > 0) and (donnees_lidar_brutes[-i] < 20):
                tableau_lidar_mm[i - 180] = 1000 * donnees_lidar_brutes[-i]
            else:
                tableau_lidar_mm[i - 180] = 0

        # =========================
        # Filtre moyenneur AVANT normalisation
        # =========================
        tableau_lidar_filtre = filtre_moyenneur(tableau_lidar_mm, fenetre=2)

        # =========================
        # Mode manuel / arrêt
        # =========================
        if not modeAuto:
            set_direction_degre(0)
            set_vitesse_m_s(0)
            continue
        
        # ========================= 
        # Programme auto : appel de la fonction autonome
        # =========================
        v_cmd, angle_cmd = calculer_commande_auto(
            tableau_lidar_filtre,
            L_entraxe=L_entraxe,
            W_empattement=W_empattement,
            maxangle_degre=maxangle_degre,
            dmax=3000.0,
            v_min=0.4,
            v_max=1.2,
            debug=True,
        )

        # Si le sens de rotation est inversé, passer -angle_cmd ici :
        # angle_cmd = -angle_cmd

        # 7) Commande véhicule
        set_direction_degre(angle_cmd)
        set_vitesse_m_s(v_cmd)


if __name__ == "__main__":
    main()