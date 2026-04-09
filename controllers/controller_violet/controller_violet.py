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

from commun import (
    filtre_moyenneur,
    analyze_walls,
    calculer_commande_automate,
    reset_automate,
    get_automate_state,
)

import config
# =========================
# Paramètres véhicule
# =========================
maxSpeed = config.VITESSE_AUTO_MAX_M_S * 3.6       # km/h
maxangle_degre = config.ANGLE_DEGRE_MAX                      # degrés

# --- Paramètres géométriques du TT-02 (à ajuster selon le modèle Webots) ---
L_entraxe = config.L_ENTRAXE_M  # m  — distance entre roues gauche/droite (voie)
W_empattement = config.W_EMPATTEMENT_M  # m  — distance entre essieu avant et arrière

ETAT_NAMES = {0: "NAVIGATION", 1: "BLOCAGE"}
SOUS_ETAT_NAMES = {0: "BACKWARD", 1: "CAMERA_CHECKING", 2: "TURN_LEFT", 3: "TURN_RIGHT"}



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

    # Repart d'un automate propre au lancement.
    reset_automate()


    while driver.step() != -1:
        # =========================
        # Lecture caméra
        # =========================
        wall_info = None
        if camera_ok:
            image = camera.getImage()

            # protection contre NULL pointer
            if image is not None:
                width = camera.getWidth()
                height = camera.getHeight()

                # conversion Webots -> numpy
                image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))

                # RGBA -> BGR (OpenCV)
                frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
                wall_info = analyze_walls(
                        frame_bgr,
                        band_ratio=float(config.CAMERA_BAND_RATIO),
                        min_ratio=float(config.CAMERA_MIN_RATIO),
                        dominance=float(config.CAMERA_DOMINANCE),
                        unknown_value=int(config.CAMERA_UNKNOWN_VALUE),
                    )

                cv2.imshow("Camera TT02", frame_bgr)
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
                    reset_automate()
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
            reset_automate()
            continue
        
        # ========================= 
        # Programme auto : appel de la fonction autonome
        # =========================
        wall_values = wall_info["value"] if wall_info else None

        v_cmd, angle_cmd = calculer_commande_automate(
            tableau_lidar_filtre,
            L_entraxe=config.L_ENTRAXE_M,
            W_empattement=config.W_EMPATTEMENT_M,
            maxangle_degre=config.ANGLE_DEGRE_MAX,
            dmax=config.LIDAR_DMAX_MM,
            v_min=config.VITESSE_AUTO_MIN_M_S,
            v_max=config.VITESSE_AUTO_MAX_M_S,
            wall_values=wall_values,
            sec_front_fenetre_deg=config.SECURITE_FRONT_FENETRE_DEG,
            sec_front_min_points=config.SECURITE_FRONT_MIN_POINTS,
            sec_vitesse_incertaine=config.SECURITE_VITESSE_INCERTAINE,
            sec_front_stop_mm=config.SECURITE_FRONT_STOP_MM,
            sec_front_ralenti_mm=config.SECURITE_FRONT_RALENTI_MM,
            seuil_front_blocage_mm=config.SEUIL_FRONT_BLOCAGE_MM,
            seuil_front_degagement_mm=config.SEUIL_FRONT_DEGAGEMENT_MM,
            seuil_arriere_degagement_mm=config.SEUIL_ARRIERE_DEGAGEMENT_MM,
            lidar_rear_window_deg=config.LIDAR_REAR_WINDOW_DEG,
            lidar_rear_min_points=config.LIDAR_REAR_MIN_POINTS,
            blocage_action_duration_s=config.BLOCAGE_ACTION_DURATION_S,
            boucle_periode_s=config.BOUCLE_PERIODE_S,
            vitesse_blocage_m_s=config.VITESSE_BLOCAGE_M_S,
            angle_recul_fixe_deg=config.ANGLE_RECUL_FIXE_DEG,
            seuil_blocage_persist_steps=config.SEUIL_BLOCAGE_PERSIST_STEPS,
            camera_direction_expected=config.CAMERA_DIRECTION_EXPECTED,
            camera_unknown_value=config.CAMERA_UNKNOWN_VALUE,
            camera_confirm_steps=config.CAMERA_CONFIRM_STEPS,
            avoid_front_diag_deg=config.AVOID_FRONT_DIAG_DEG,
            avoid_side_deg=config.AVOID_SIDE_DEG,
            avoid_sector_half_deg=config.AVOID_SECTOR_HALF_DEG,
            avoid_narrow_mm=config.AVOID_NARROW_MM,
            obstacle_window_deg=config.OBSTACLE_WINDOW_DEG,
            obstacle_cluster_gap_mm=config.OBSTACLE_CLUSTER_GAP_MM,
            obstacle_dynamic_speed_m_s=config.OBSTACLE_DYNAMIC_SPEED_M_S,
            debug=bool(config.AUTO_DEBUG),
        )

        # Si le sens de rotation est inversé, passer -angle_cmd ici :
        # angle_cmd = -angle_cmd

        # 7) Commande véhicule
        set_direction_degre(angle_cmd)
        set_vitesse_m_s(v_cmd)
        
        if bool(config.DEBUG_ACTIONNEURS):
            fsm = get_automate_state()
            etat = int(fsm.get("etat", 0))
            sous_etat = int(fsm.get("sous_etat", 0))
            d_front = fsm.get("last_front_mm", None)
            d_rear = fsm.get("last_rear_mm", None)
            obj_kind = fsm.get("last_obj_kind", "unknown")
            obj_speed = float(fsm.get("last_obj_speed_m_s", 0.0))
            s_left = float(fsm.get("score_left", 0.0))
            s_right = float(fsm.get("score_right", 0.0))

            ss = SOUS_ETAT_NAMES.get(sous_etat, "?") if etat == 1 else "-"
            front_txt = "NA" if d_front is None else f"{float(d_front):.0f}"
            rear_txt = "NA" if d_rear is None else f"{float(d_rear):.0f}"
            wall_txt = "None" if wall_values is None else str(wall_values)
            print(
                f"FSM [{ETAT_NAMES.get(etat, '?')}|{ss}] "
                f"v={float(v_cmd):.2f} ang={float(angle_cmd):.1f} "
                f"dF={front_txt} dR={rear_txt} back={int(fsm.get('counter_etat_backward', 0))} "
                f"obj={obj_kind} vrel={obj_speed:.2f} sL={s_left:.2f} sR={s_right:.2f} walls={wall_txt}"
            )


if __name__ == "__main__":
    main()