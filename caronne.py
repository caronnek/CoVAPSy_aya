# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Adaptation : contrôleur neuronal virtuel différentiel
# reprojeté en commandes Ackermann
# + filtrage moyenneur des données LiDAR
# + logique de suivi murs conservée
# + PISTE A v14 : déclencheur + mémoire récurrente sur proximités ET diff_lr
# + ajout d'un neurone p_stop pour stopper la voiture face à un mur large et proche

from vehicle import Driver
from controller import Lidar
import numpy as np
import cv2

from imageProcessing import webots_image_to_bgr, analyze_walls

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep



TIME_STEP = 32  # ms (ton timestep habituel)
# =========================
# Constantes FSM
# =========================
SEUIL_BLOCAGE            = 0.30
SEUIL_ARRIERE_DEGAGEMENT = 300.0  # mm — en dessous = arrêt du recul
DIRECTION                = 0      # 0 = rouge à gauche, vert à droite

ANGLE_RECUL_FIXE         = 15.0   # degrés
VITESSE_RECUL            = 0.5    # m/s

ACTION_DURATION          = 1.0    # secondes
ACTION_STEPS             = int((ACTION_DURATION * 1000) / TIME_STEP)  # = 125 steps

time_stop                = 0.0
time_stop_step           = int((time_stop * 1000) / TIME_STEP)

CAMERA_CONFIRM_STEPS     = 5      # nb de frames consécutives à valider

# =========================
# Compteurs FSM
# =========================
counter_time             = 0
counter_camera_confirm   = 0
action_counter           = 0
Flag_turn_right          = False


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
    
    
    
# =========================
# Initialisation sonar
# =========================

sonar = driver.getDevice("us_rear")

if sonar is None : 
    print("Sonar non trouvée : us_rear")
else : 
    sonar.enable(sensorTimeStep)
    print("Sonar trouvée :", sonar.getName())   

# =========================
# Initialisation LiDAR
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
maxSpeed = 50
maxangle_degre = 18

L_entraxe = 0.180
W_empattement = 0.250

driver.setSteeringAngle(0)
driver.setCruisingSpeed(0)

tableau_lidar_mm = [0] * 360

# =========================
# Mémoire récurrente
# =========================
p_fL_brut_memo = 0.0
p_fR_brut_memo = 0.0
diff_lr_memo   = 0.0
alpha          = 0.95

# =========================
# Modes
# =========================
modeAuto = False
cameraTestMode = False


# =========================
# Machine à états — entiers pour comparaison rapide
# =========================
NAVIGATION      = 0
BLOCAGE         = 1

BACKWARD        = 0
CAMERA_CHECKING = 1
TURN_LEFT       = 2
TURN_RIGHT      = 3

etat      = NAVIGATION
sous_etat = BACKWARD
counter_etat_backward = 0
seuil_loop = 2

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
    
def recule_avec_angle(angle_degre, vitesse_m_s):
    """Recule à vitesse donnée avec un angle de braquage fixe."""
    set_direction_degre(angle_degre)
    driver.setCruisingSpeed(-vitesse_m_s * 3.6)

# =========================
# Fonctions traitement LiDAR
# =========================
def filtre_moyenneur(tab, fenetre=2):
    n = len(tab)
    tab_filtre = [0.0] * n
    for i in range(n):
        somme = 0.0
        count = 0
        for k in range(-fenetre, fenetre + 1):
            idx = (i + k) % n
            val = tab[idx]
            if val > 0:
                somme += val
                count += 1
        tab_filtre[i] = somme / count if count > 0 else 0.0
    return tab_filtre

def lire_point_lidar(tab, angle_deg, valeur_defaut=3000.0):
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
# Fonction traitement camera
# =========================

def check_camera(values):
    """
    Retourne True si on est dans le bon sens :
    vert (1) doit être à droite → values[2] == 1
    """
    if values is None:
        return False
    return values[0] == DIRECTION and values[2]== (1- DIRECTION)

# =========================
# Conversion différentiel -> Ackermann
# =========================
def differentiel_vers_ackermann(u_g, u_d, L, W, v_min, v_max, angle_max_deg):
    v_norm = (u_g + u_d) / 2.0
    omega  = (u_d - u_g) / L

    # Permet un vrai arrêt si v_norm -> 0
    v_cmd = v_max * max(0.0, v_norm)

    omega_seuil = 1e-4
    if abs(omega) < omega_seuil:
        angle_deg = 0.0
    else:
        R = v_norm / omega if abs(v_norm) > 1e-4 else 1e6
        angle_rad = np.arctan(W / R)
        angle_deg = np.degrees(angle_rad)

    angle_deg = float(np.clip(angle_deg, -angle_max_deg, angle_max_deg))
    return v_cmd, angle_deg

print("Cliquer sur la vue 3D pour commencer")
print("a : mode auto")
print("n : stop")
print("t : test traitement camera")

while driver.step() != -1:

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

        elif currentKey == ord('t') or currentKey == ord('T'):
            cameraTestMode = not cameraTestMode
            if cameraTestMode:
                print("-------- Test Camera Activé -------")
            else:
                print("-------- Test Camera Désactivé -------")
                try:
                    cv2.destroyWindow("Camera TT02")
                    cv2.destroyWindow("ROI gauche")
                    cv2.destroyWindow("ROI droite")
                except:
                    pass

    # =========================
    # Lecture caméra + traitement image (uniquement en mode test)
    # =========================
    wall_info = None

    if camera_ok and cameraTestMode:
        image = camera.getImage()
        if image is not None:
            width = camera.getWidth()
            height = camera.getHeight()

            image_bgr = webots_image_to_bgr(image, width, height)
            wall_info = analyze_walls(image_bgr)
            

            cv2.imshow("Camera TT02", image_bgr)
            #cv2.imshow("ROI gauche", wall_info["roi_left"])
            #cv2.imshow("ROI droite", wall_info["roi_right"])
            cv2.waitKey(1)

    # =========================
    # Acquisition LiDAR
    # =========================
    donnees_lidar_brutes = lidar.getRangeImage()

    for i in range(360):
        if (donnees_lidar_brutes[-i] > 0) and (donnees_lidar_brutes[-i] < 20):
            tableau_lidar_mm[i - 180] = 1000 * donnees_lidar_brutes[-i]
        else:
            tableau_lidar_mm[i - 180] = 0

    tableau_lidar_filtre = filtre_moyenneur(tableau_lidar_mm, fenetre=2)

    # =========================
    # Mode manuel
    # =========================
    if not modeAuto:
        set_direction_degre(0)
        set_vitesse_m_s(0)

        if wall_info is not None:
            print(
                #f"CAMERA | gauche={wall_info['left_color']} "
                #f"(R={wall_info['left_red']}, V={wall_info['left_green']}) | "
                #f"droite={wall_info['right_color']} "
                #f"(R={wall_info['right_red']}, V={wall_info['right_green']})"
            )
        continue

    # =========================
    # Angles LiDAR — murs
    # =========================
    angle_l1 =  63
    angle_l2 =  73
    angle_f0 =   0
    angle_r1 = -63
    angle_r2 = -73

    # =========================
    # Angles LiDAR — obstacle proche
    # =========================
    angle_fL =  6
    angle_fR = -6

    # =========================
    # Angles LiDAR — détection mur large frontal
    # =========================
    angle_fwL =  30
    angle_fwR = -30

    # =========================
    # Lecture distances
    # =========================
    d_l1 = lire_point_lidar(tableau_lidar_filtre, angle_l1)
    d_l2 = lire_point_lidar(tableau_lidar_filtre, angle_l2)
    d_f0 = lire_point_lidar(tableau_lidar_filtre, angle_f0)
    d_r1 = lire_point_lidar(tableau_lidar_filtre, angle_r1)
    d_r2 = lire_point_lidar(tableau_lidar_filtre, angle_r2)

    d_fL = lire_point_lidar(tableau_lidar_filtre, angle_fL)
    d_fR = lire_point_lidar(tableau_lidar_filtre, angle_fR)

    d_fwL = lire_point_lidar(tableau_lidar_filtre, angle_fwL)
    d_fwR = lire_point_lidar(tableau_lidar_filtre, angle_fwR)

    # =========================
    # Proximité murs (dmax large)
    # =========================
    dmax = 3000.0
    p_l1 = 1.0 - normaliser_distance(d_l1, dmax)
    p_l2 = 1.0 - normaliser_distance(d_l2, dmax)
    p_f0 = 1.0 - normaliser_distance(d_f0, dmax)
    p_r1 = 1.0 - normaliser_distance(d_r1, dmax)
    p_r2 = 1.0 - normaliser_distance(d_r2, dmax)

    # =========================
    # PISTE A v14 — signal évitement + double mémoire récurrente
    # =========================
    dmax_obs = 1000.0

    p_f0_obs  = 1.0 - normaliser_distance(d_f0, dmax_obs)
    p_fL_brut = 1.0 - normaliser_distance(d_fL, dmax_obs)
    p_fR_brut = 1.0 - normaliser_distance(d_fR, dmax_obs)

    # nouveaux points pour détecter un mur large
    p_fwL = 1.0 - normaliser_distance(d_fwL, dmax_obs)
    p_fwR = 1.0 - normaliser_distance(d_fwR, dmax_obs)

    p_fL_brut_memo = max(p_fL_brut, alpha * p_fL_brut_memo)
    p_fR_brut_memo = max(p_fR_brut, alpha * p_fR_brut_memo)

    declencheur = max(p_f0_obs, p_fL_brut_memo, p_fR_brut_memo)

    diff_lr_brut = (d_fL - d_fR) / dmax_obs
    diff_lr_brut = float(np.clip(diff_lr_brut, -1.0, 1.0))
    diff_lr_memo = alpha * diff_lr_memo + (1.0 - alpha) * diff_lr_brut
    diff_lr      = diff_lr_memo

    espace_gauche = normaliser_distance(d_l1, dmax)
    espace_droite = normaliser_distance(d_r1, dmax)

    biais_gauche = max(0.0,  diff_lr) + 0.3 * espace_droite
    biais_droite = max(0.0, -diff_lr) + 0.3 * espace_gauche

    p_fL_combine = declencheur * biais_gauche
    p_fR_combine = declencheur * biais_droite

    # =========================
    # Neurone stop p_stop
    # actif si objet large et proche devant
    # =========================
    p_stop = (p_f0_obs ** 2) * min(p_fwL, p_fwR)
    p_stop = float(np.clip(p_stop, 0.0, 1.0))

    # =========================
    # Entrée réseau
    # [biais, l1, l2, f0, fLc, fRc, r1, r2, stop]
    # =========================
    x = np.array([
        1.0,
        p_l1,
        p_l2,
        p_f0,
        p_fL_combine,
        p_fR_combine,
        p_r1,
        p_r2,
        p_stop
    ])

    # =========================
    # Réseau virtuel différentiel
    # =========================
    w_g = np.array([ 1.2,  1.00,  0.70, -1.2, -1.20,  1.20, -0.65, -0.90, -0.5])
    w_d = np.array([ 1.2, -0.90, -0.65, -1.2,  1.20, -1.20,  0.70,  1.00, -0.5])

    u_g = np.tanh(np.dot(x, w_g))
    u_d = np.tanh(np.dot(x, w_d))

    # =========================
    # Conversion vers Ackermann
    # =========================
    v_cmd, angle_cmd = differentiel_vers_ackermann(
        u_g, u_d,
        L=L_entraxe,
        W=W_empattement,
        v_min=0.4,
        v_max=1.2,
        angle_max_deg=maxangle_degre
    )

    # =========================
    # Machine à états — if/elif sur entiers
    # =========================
    if etat == NAVIGATION:
        if p_stop >= SEUIL_BLOCAGE:
            etat = BLOCAGE
            sous_etat = BACKWARD
            counter_time = 0
        else:
            counter_time = 0
            set_direction_degre(angle_cmd)
            set_vitesse_m_s(v_cmd)

    elif etat == BLOCAGE:
        d_rear = sonar.getValue()

        if sous_etat == BACKWARD:
            if action_counter >= ACTION_STEPS or d_rear <= SEUIL_ARRIERE_DEGAGEMENT:
                set_direction_degre(0)
                set_vitesse_m_s(0)
                action_counter = 0
                if Flag_turn_right or counter_etat_backward > seuil_loop:
                    sous_etat = TURN_RIGHT
                else:
                    sous_etat = TURN_LEFT
            else:
                recule_avec_angle(0, VITESSE_RECUL)
                action_counter += 1

        elif sous_etat == TURN_LEFT:
            if action_counter >= ACTION_STEPS and p_stop < SEUIL_BLOCAGE:
                set_direction_degre(0)
                set_vitesse_m_s(0)
                action_counter = 0
                sous_etat = CAMERA_CHECKING
            elif action_counter >= ACTION_STEPS and p_stop >= SEUIL_BLOCAGE:
                set_direction_degre(0)
                set_vitesse_m_s(0)
                action_counter = 0
                sous_etat = BACKWARD
                counter_etat_backward += 1
            else:
                set_direction_degre(ANGLE_RECUL_FIXE)
                set_vitesse_m_s(VITESSE_RECUL)
                action_counter += 1

        elif sous_etat == CAMERA_CHECKING:
            if check_camera(wall_info["value"] if wall_info else None):
                etat = NAVIGATION
                sous_etat = BACKWARD
                action_counter = 0
                Flag_turn_right = False
            else:
                etat = BLOCAGE
                sous_etat = BACKWARD
                action_counter = 0
                Flag_turn_right = True

        elif sous_etat == TURN_RIGHT:
            if action_counter >= ACTION_STEPS and p_stop < SEUIL_BLOCAGE:
                set_direction_degre(0)
                set_vitesse_m_s(0)
                action_counter = 0
                sous_etat = CAMERA_CHECKING
            elif action_counter >= ACTION_STEPS and p_stop >= SEUIL_BLOCAGE:
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
    ETAT_NAMES     = {NAVIGATION: "NAVIGATION", BLOCAGE: "BLOCAGE"}
    SOUS_ETAT_NAMES = {BACKWARD: "BACKWARD", CAMERA_CHECKING: "CAMERA_CHECKING",
                       TURN_LEFT: "TURN_LEFT", TURN_RIGHT: "TURN_RIGHT"}
    print("--------------------------------------------------")
    print(f"[ETAT]     {ETAT_NAMES[etat]}  |  [SOUS-ETAT] {SOUS_ETAT_NAMES[sous_etat] if etat == BLOCAGE else '-'}")
    print(f"u_g={u_g:.3f}  u_d={u_d:.3f}")
    print(f"v_cmd={v_cmd:.3f} m/s  |  angle={angle_cmd:.3f} deg")
    print(f"p_fL={p_fL_combine:.2f}  p_fR={p_fR_combine:.2f}  declencheur={declencheur:.2f}")
    print(f"diff_lr_brut={diff_lr_brut:.2f}  diff_lr_memo={diff_lr_memo:.2f}")
    print(f"p_fL_memo={p_fL_brut_memo:.2f}  p_fR_memo={p_fR_brut_memo:.2f}")
    print(f"p_fwL={p_fwL:.2f}  p_fwR={p_fwR:.2f}  p_stop={p_stop:.2f}")
    print(f"d_fL={d_fL:.0f}mm  d_fR={d_fR:.0f}mm  d_f0={d_f0:.0f}mm")
    print(f"d_fwL={d_fwL:.0f}mm  d_fwR={d_fwR:.0f}mm")
    print(f"CAMERA  L={wall_info['value'][0] if wall_info else '-'}  C={wall_info['value'][1] if wall_info else '-'}  R={wall_info['value'][2] if wall_info else '-'}")