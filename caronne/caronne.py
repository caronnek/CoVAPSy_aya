# tt02_fsm.py
# Contrôleur principal TT-02 — CoVAPSy / Webots 2023b
# FSM uniquement — le traitement LiDAR/réseau est dans traitement.py

from vehicle import Driver
from controller import Lidar
import numpy as np
import cv2

from imageProcessing import webots_image_to_bgr, analyze_walls
from traitement import filtre_moyenneur, calculer

# =========================
# Driver
# =========================
driver = Driver()
basicTimeStep  = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

# =========================
# Constantes FSM
# =========================
TIME_STEP                = 32     # ms

SEUIL_BLOCAGE            = 0.30
SEUIL_ARRIERE_DEGAGEMENT = 300.0  # mm
DIRECTION                = 0

ANGLE_RECUL_FIXE         = 15.0   # degrés
VITESSE_RECUL            = 0.5    # m/s

ACTION_DURATION          = 1.0    # secondes
ACTION_STEPS             = int((ACTION_DURATION * 1000) / TIME_STEP)

CAMERA_CONFIRM_STEPS     = 5

# =========================
# Compteurs FSM
# =========================
counter_time           = 0
counter_camera_confirm = 0
action_counter         = 0
Flag_turn_right        = False

# =========================
# États (entiers)
# =========================
NAVIGATION      = 0
BLOCAGE         = 1
BACKWARD        = 0
CAMERA_CHECKING = 1
TURN_LEFT       = 2
TURN_RIGHT      = 3

ETAT_NAMES      = {NAVIGATION: "NAVIGATION", BLOCAGE: "BLOCAGE"}
SOUS_ETAT_NAMES = {BACKWARD: "BACKWARD", CAMERA_CHECKING: "CAMERA_CHECKING",
                   TURN_LEFT: "TURN_LEFT", TURN_RIGHT: "TURN_RIGHT"}

etat                  = NAVIGATION
sous_etat             = BACKWARD
counter_etat_backward = 0
seuil_loop            = 5

# =========================
# Initialisation capteurs
# =========================
camera = driver.getDevice("pi_camera")
camera_ok = False
if camera is None:
    print("Camera non trouvée : pi_camera")
else:
    camera.enable(sensorTimeStep)
    camera_ok = True
    print("Camera trouvée :", camera.getName(),
          camera.getWidth(), "x", camera.getHeight())

sonar = driver.getDevice("us_rear")
if sonar is None:
    print("Sonar non trouvé : us_rear")
else:
    sonar.enable(sensorTimeStep)
    print("Sonar trouvé :", sonar.getName())

lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

# =========================
# Paramètres véhicule
# =========================
maxSpeed       = 50
maxangle_degre = 18

driver.setSteeringAngle(0)
driver.setCruisingSpeed(0)

tableau_lidar_mm = [0] * 360

# =========================
# Modes
# =========================
modeAuto       = False
cameraTestMode = False
DEBUG          = True

# =========================
# Fonctions véhicule
# =========================
def set_vitesse_m_s(v):
    speed = max(0.0, min(v * 3.6, maxSpeed))
    driver.setCruisingSpeed(speed)

def set_direction_degre(angle):
    angle = max(-maxangle_degre, min(angle, maxangle_degre))
    driver.setSteeringAngle(-angle * np.pi / 180.0)

def recule_avec_angle(angle, v):
    set_direction_degre(angle)
    driver.setCruisingSpeed(-v * 3.6)

def check_camera(values):
    if values is None:
        return False
    return values[0] == DIRECTION and values[2] == (1 - DIRECTION)

# =========================
# Boucle principale
# =========================
print("Cliquer sur la vue 3D pour commencer")
print("a : mode auto  |  n : stop  |  t : test caméra")

while driver.step() != -1:

    # ── Clavier ──────────────────────────────────────────────
    while True:
        key = keyboard.getKey()
        if key == -1:
            break
        elif key in (ord('n'), ord('N')):
            modeAuto = False
            print("-------- Mode Auto Désactivé -------")
        elif key in (ord('a'), ord('A')):
            modeAuto = True
            print("-------- Mode Auto Activé -------")
        elif key in (ord('t'), ord('T')):
            cameraTestMode = not cameraTestMode
            if not cameraTestMode:
                try: cv2.destroyWindow("Camera TT02")
                except: pass

    # ── Caméra ───────────────────────────────────────────────
    wall_info = None
    if camera_ok and cameraTestMode:
        image = camera.getImage()
        if image is not None:
            bgr = webots_image_to_bgr(image, camera.getWidth(), camera.getHeight())
            wall_info = analyze_walls(bgr)
            cv2.imshow("Camera TT02", bgr)
            cv2.waitKey(1)

    # ── LiDAR → tableau mm ───────────────────────────────────
    raw = lidar.getRangeImage()
    for i in range(360):
        v = raw[-i]
        tableau_lidar_mm[i - 180] = int(v * 1000) if 0 < v < 20 else 0
    lidar_filtre = filtre_moyenneur(tableau_lidar_mm, fenetre=2)

    # ── Mode manuel ──────────────────────────────────────────
    if not modeAuto:
        set_direction_degre(0)
        set_vitesse_m_s(0)
        continue

    # ── Traitement (LiDAR + réseau) ──────────────────────────
    v_cmd, angle_cmd, p_stop, dbg = calculer(lidar_filtre)

    # ── FSM ──────────────────────────────────────────────────
    match etat:
        case NAVIGATION:
            if p_stop >= SEUIL_BLOCAGE:
                etat = BLOCAGE
                sous_etat = BACKWARD
                counter_time = 0
            else:
                counter_time = 0
                set_direction_degre(angle_cmd)
                set_vitesse_m_s(v_cmd)

        case BLOCAGE:
            d_rear = sonar.getValue() if sonar else 9999.0

            match sous_etat:
                case BACKWARD:
                    if action_counter >= ACTION_STEPS or d_rear <= SEUIL_ARRIERE_DEGAGEMENT:
                        set_direction_degre(0)
                        set_vitesse_m_s(0)
                        action_counter = 0
                        if Flag_turn_right and counter_etat_backward > seuil_loop:
                            sous_etat = TURN_RIGHT
                        else:
                            sous_etat = TURN_LEFT
                    else:
                        recule_avec_angle(0, VITESSE_RECUL)
                        action_counter += 1

                case TURN_LEFT:
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

                case CAMERA_CHECKING:
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

                case TURN_RIGHT:
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

                case _:
                    sous_etat = BACKWARD
        
            if etat not in (NAVIGATION, BLOCAGE):
                etat = NAVIGATION

    # ── Debug ────────────────────────────────────────────────
    if DEBUG:
        ss = SOUS_ETAT_NAMES[sous_etat] if etat == BLOCAGE else "-"
        print(f"[{ETAT_NAMES[etat]}|{ss}] "
              f"v={dbg['v_cmd']:.2f} angle={dbg['angle_cmd']:.1f} "
              f"p_stop={dbg['p_stop']:.2f} back={counter_etat_backward}")
