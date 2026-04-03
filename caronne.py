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
# + machine à états NAVIGATION / BLOCAGE

from vehicle import Driver
from controller import Lidar
import numpy as np
import time
import cv2
import random

driver = Driver()


basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep


# Initialisation du sonar
sonar = driver.getDevice("us_rear")
sonar.enable(sensorTimeStep)
print(sonar)  



# Initialisation caméra
camera = driver.getDevice("pi_camera")

camera_ok = False

if camera is None:
    print("Camera non trouvée : pi_camera")
else:
    camera.enable(sensorTimeStep)
    camera_ok = True
    print("Camera trouvée :", camera.getName())
    print("Resolution camera :", camera.getWidth(), "x", camera.getHeight())  
# =========================
# Initialisation du LiDAR
# =========================
lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

# =========================
# Initialisation clavier
# =========================
keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

# =========================
# Paramètres véhicule
# =========================
maxSpeed = 50       # km/h
maxangle_degre = 19

# --- Paramètres géométriques du TT-02 (à ajuster selon le modèle Webots) ---
L_entraxe = 0.180  # m  — distance entre roues gauche/droite (voie)
W_empattement = 0.250  # m  — distance entre essieu avant et arrière

driver.setSteeringAngle(0)
driver.setCruisingSpeed(0)

tableau_lidar_mm = [0] * 360

# =========================
# Machine à états
# =========================
NAVIGATION        = "NAVIGATION"
BLOCAGE           = "BLOCAGE"

# Sous-états BLOCAGE
BACKWARD      = "BACKWARD"
CAMERA_CHECKING = "CAMERA_CHECKING"
TURN_LEFT      = "TURN_LEFT"
TURN_RIGHT     = "TURN_RIGHT"

etat      = NAVIGATION
sous_etat = BACKWARD  # actif uniquement quand etat == BLOCAGE

SEUIL_VITESSE_BLOCAGE    = 0.5    # m/s — en dessous = "vitesse faible"
SEUIL_FRONT_BLOCAGE      = 500.0  # mm  — en dessous = "obstacle front"
SEUIL_FRONT_DEGAGEMENT   = 1500.0  # mm  — au dessus  = dégagement frontal suffisant
SEUIL_ARRIERE_DEGAGEMENT = 300.0  # mm  — en dessous  = arrêt du recul

ANGLE_RECUL_FIXE = 15.0   # degrés — braquage fixe au recul
VITESSE_RECUL    = 0.5    # m/s    — vitesse de recul


Flag_turn_right = False
action_counter = 0
ACTION_DURATION = 1.0  # secondes
TIME_STEP = 32  # ms (ton timestep habituel)
ACTION_STEPS = int((ACTION_DURATION * 1000) / TIME_STEP)  # = 125 steps

# =========================
# Fonctions véhicule
# =========================
def set_vitesse_m_s(vitesse_m_s):
    speed = vitesse_m_s * 3.6
    if speed > maxSpeed:
        speed = maxSpeed
    if speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)

def set_direction_degre(angle_degre):
    if angle_degre > maxangle_degre:
        angle_degre = maxangle_degre
    elif angle_degre < -maxangle_degre:
        angle_degre = -maxangle_degre

    angle_rad = -angle_degre * np.pi / 180.0
    driver.setSteeringAngle(angle_rad)

def recule(vitesse_m_s):
    driver.setCruisingSpeed(-vitesse_m_s * 3.6)

def recule_avec_angle(angle_degre, vitesse_m_s):
    """Recule à vitesse donnée avec un angle de braquage fixe."""
    set_direction_degre(angle_degre)
    driver.setCruisingSpeed(-vitesse_m_s * 3.6)

def camera_valide_direction():
    """
    Stub : retourne True si la caméra confirme que la voiture
    est dans le bon sens pour repartir en NAVIGATION.
    À remplacer par l'algorithme caméra réel.
    """
    return True

# =========================
# Fonctions traitement LiDAR
# =========================
def filtre_moyenneur(tab, fenetre=2):
    """
    Filtre moyenneur circulaire.
    fenetre=2 => moyenne sur 5 points.
    Ignore les valeurs nulles.
    """
    n = len(tab)
    tab_filtre = [0] * n

    for i in range(n):
        somme = 0.0
        count = 0

        for k in range(-fenetre, fenetre + 1):
            idx = (i + k) % n
            val = tab[idx]

            if val > 0:
                somme += val
                count += 1

        if count > 0:
            tab_filtre[i] = somme / count
        else:
            tab_filtre[i] = 0

    return tab_filtre

def lire_point_lidar(tab, angle_deg, valeur_defaut=3000.0):
    """
    Lit un point exact du lidar pour un angle donné.
    Les indices sont accessibles dans [-180, 179].
    """
    idx = int(angle_deg)

    if idx < -180:
        idx += 360
    elif idx > 179:
        idx -= 360

    valeur = tab[idx]
    if valeur <= 0:
        return valeur_defaut
    return valeur

def normaliser_distance(d, dmax):
    d = max(0.0, min(d, dmax))
    return d / dmax

# =========================
# Conversion différentiel → Ackermann
# =========================
def differentiel_vers_ackermann(u_g, u_d, L, W, v_min, v_max, angle_max_deg):
    """
    Convertit les sorties du réseau neuronal différentiel (u_g, u_d)
    en angle de braquage Ackermann (angle moyen de la roue intérieure).

    Étape 1 : v et ω depuis le modèle différentiel
        v = (u_g + u_d) / 2
        ω = (u_d - u_g) / L

    Étape 2 : rayon de courbure
        R = v / ω   (géré si ω ≈ 0)

    Étape 3 : angle Ackermann (roue idéale de référence au centre essieu avant)
        δ = arctan(W / R)

    Paramètres
    ----------
    u_g, u_d    : sorties tanh réseau [-1, 1]
    L           : entraxe (voie) en m
    W           : empattement en m
    v_min/max   : plage de vitesse linéaire en m/s
    angle_max_deg : saturation angle en degrés

    Retourne
    --------
    v_cmd       : vitesse linéaire en m/s
    angle_deg   : angle de braquage en degrés (+ = gauche, - = droite)
    """

    # --- Étape 1 : modèle différentiel → v, ω ---
    v_norm = (u_g + u_d) / 2.0   # vitesse normalisée [-1, 1]
    omega  = (u_d - u_g) / L     # vitesse angulaire  [rad/s normalisé]

    # --- Étape 2 : vitesse de commande ---
    v_cmd = v_min + (v_max - v_min) * max(0.0, v_norm)

    # --- Étape 3 : rayon de courbure ---
    omega_seuil = 1e-4            # évite la division par zéro (ligne droite)
    if abs(omega) < omega_seuil:
        # Tout droit : angle nul
        angle_deg = 0.0
    else:
        R = v_norm / omega        # rayon signé (négatif = droite)

        # --- Étape 4 : angle Ackermann (roue centrale de référence) ---
        # δ = arctan(W / R)
        # Négatif car convention Webots : angle+ = droite mécanique
        angle_rad = np.arctan(W / R)
        angle_deg = np.degrees(angle_rad)

    # Saturation
    angle_deg = float(np.clip(angle_deg, -angle_max_deg, angle_max_deg))

    return v_cmd, angle_deg

# =========================
# Mode de fonctionnement
# =========================
modeAuto = False

print("Cliquer sur la vue 3D pour commencer")
print("a : mode auto")
print("n : stop")

while driver.step() != -1:


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
        etat      = NAVIGATION   # reset état
        sous_etat = BACKWARD  # reset sous-état
        continue

    # =========================================================
    # Programme auto : 5 points exacts + réseau + Ackermann
    # =========================================================

    # Angles des points pertinents
    angle_l1    =  60
    angle_l2    =  70
    angle_lf1   =  5
    angle_front =   0
    angle_rf1   =  -5
    angle_r1    = -60
    angle_r2    = -70

    # 1) Lecture des 5 points exacts
    d_l1    = lire_point_lidar(tableau_lidar_filtre, angle_l1)
    d_l2    = lire_point_lidar(tableau_lidar_filtre, angle_l2)
    d_lf1 = lire_point_lidar(tableau_lidar_filtre, angle_lf1)
    d_front = lire_point_lidar(tableau_lidar_filtre, angle_front)
    d_rf1 = lire_point_lidar(tableau_lidar_filtre, angle_rf1)
    d_r1    = lire_point_lidar(tableau_lidar_filtre, angle_r1)
    d_r2    = lire_point_lidar(tableau_lidar_filtre, angle_r2)

    # 2) Normalisation
    dmax = 3000.0  # mm
    l1 = normaliser_distance(d_l1,    dmax)
    l2 = normaliser_distance(d_l2,    dmax)
    lf1 = normaliser_distance(d_lf1,  dmax)
    f  = normaliser_distance(d_front, dmax)
    rf1 = normaliser_distance(d_rf1,  dmax)
    r1 = normaliser_distance(d_r1,    dmax)
    r2 = normaliser_distance(d_r2,    dmax)

    # 3) Conversion en proximité
    p_l1 = 1.0 - l1
    p_l2 = 1.0 - l2
    p_lf1 = 1.0 - lf1
    p_f  = 1.0 - f
    p_rf1 = 1.0 - rf1
    p_r1 = 1.0 - r1
    p_r2 = 1.0 - r2

    # 4) Vecteur d'entrée du réseau  [biais, p_l1, p_l2, p_lf1, p_f, p_rf1, p_r1, p_r2]
    x = np.array([1.0, p_l1, p_l2, p_lf1, p_f, p_rf1, p_r1, p_r2])

    # 5) Réseau virtuel différentiel
    w_g = np.array([ 1.2,  0.8,  0.8, -0.2, -1.2,  0.2, -0.6, -0.6])
    w_d = np.array([ 1.2, -0.6, -0.6,  0.2, -1.2, -0.2,  0.8,  0.8])

    u_g = np.tanh(np.dot(x, w_g))
    u_d = np.tanh(np.dot(x, w_d))

    # 6) Conversion différentiel → Ackermann
    v_cmd, angle_cmd = differentiel_vers_ackermann(
        u_g, u_d,
        L=L_entraxe,
        W=W_empattement,
        v_min=0.4,
        v_max=1.2,
        angle_max_deg=maxangle_degre
    )

    # Si le sens de rotation est inversé, passer -angle_cmd ici :
    # angle_cmd = -angle_cmd

    # =========================================================
    # 7) Machine à états NAVIGATION / BLOCAGE
    # =========================================================

    # Lecture sonar arrière (mètres → mm)
    d_rear = sonar.getValue()
    d_moy = np.mean([d_l1, d_l2, d_front, d_r1, d_r2])
    

    match etat:

        case "NAVIGATION":
            if  d_front < SEUIL_FRONT_BLOCAGE:
                etat      = BLOCAGE
                sous_etat = BACKWARD
                action_counter = 0
                set_direction_degre(0)
                set_vitesse_m_s(0)
                print(">>> Transition : NAVIGATION → BLOCAGE")
            else:
                set_direction_degre(angle_cmd)
                set_vitesse_m_s(v_cmd)

        case "BLOCAGE":
            match sous_etat:

                case "BACKWARD":
                    if action_counter >= ACTION_STEPS or d_rear <= SEUIL_ARRIERE_DEGAGEMENT:
                        # Obstacle derrière → stop et passe à TURN_LEFT
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        if Flag_turn_right:
                            sous_etat = TURN_RIGHT
                            print(">>> BACKWARD → TURN_RIGHT (sonar)")
                        else :
                            sous_etat = TURN_LEFT
                        print(">>> BACKWARD → TURN_LEFT (sonar)")
                    else:
                        recule_avec_angle(0, VITESSE_RECUL)
                        action_counter += 1

                case "TURN_LEFT":
                    if action_counter >= (ACTION_STEPS) and d_front > SEUIL_FRONT_BLOCAGE :
                        # Timeout → validation caméra quand même
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        sous_etat = CAMERA_CHECKING
                    elif action_counter >= (ACTION_STEPS) and d_front < SEUIL_FRONT_BLOCAGE : 
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        sous_etat = BACKWARD
                    else:
                        set_direction_degre(ANGLE_RECUL_FIXE)
                        set_vitesse_m_s(VITESSE_RECUL)
                        action_counter += 1

                case "CAMERA_CHECKING":
                    if camera_valide_direction():
                        etat      = NAVIGATION
                        sous_etat = BACKWARD    
                        action_counter = 0
                        Flag_turn_right = False
                    else:
                        etat      = BLOCAGE
                        sous_etat = BACKWARD
                        action_counter = 0
                        Flag_turn_right = True

                case "TURN_RIGHT":
                    if action_counter >= (ACTION_STEPS) and d_front > SEUIL_FRONT_BLOCAGE  :
                        # Timeout → validation caméra quand même
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        sous_etat = CAMERA_CHECKING
                    elif action_counter >= (ACTION_STEPS) and d_front < SEUIL_FRONT_BLOCAGE  : 
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        sous_etat = BACKWARD
                    else:
                        set_direction_degre(-ANGLE_RECUL_FIXE)
                        set_vitesse_m_s(VITESSE_RECUL)
                        action_counter += 1

    # =========================
    # Debug
    # =========================
    print("--------------------------------------------------")
    print(f"[ETAT]     {etat}  |  [SOUS-ETAT] {sous_etat if etat == BLOCAGE else '-'}")
    print(f"d_l1       = {d_l1:.1f} mm  |  d_l2    = {d_l2:.1f} mm")
    print(f"d_front    = {d_front:.1f} mm")
    print(f"d_r1       = {d_r1:.1f} mm  |  d_r2    = {d_r2:.1f} mm")
    print(f"d_lf1      = {d_lf1:.1f} mm  | d_rf1   = {d_rf1:.1f} mm")
    print(f"d_rear     = {d_rear:.1f} mm")
    print(f"p_l1={p_l1:.3f}  p_l2={p_l2:.3f}  p_f={p_f:.3f}  p_r1={p_r1:.3f}  p_r2={p_r2:.3f}")
    print(f"p_lf1={p_lf1:.3f}  p_rf1={p_rf1:.3f}")
    print(f"u_g        = {u_g:.3f}  |  u_d     = {u_d:.3f}")
    print(f"v_cmd      = {v_cmd:.3f} m/s")
    print(f"angle      = {angle_cmd:.3f} deg")