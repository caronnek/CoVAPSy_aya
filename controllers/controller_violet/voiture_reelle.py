# Programme principal de conduite autonome CoVAPSy
#
# Démarrage : python main_autonomous.py
# Commandes disponibles dans le terminal :
#   a    → démarre la conduite autonome
#   n  → arrête la voiture
#   q  → arrête la voiture et quitte le programme
#
# Conformité règlement CoVAPSy 2026 :
#   - La voiture n'avance pas avant réception de la commande GO
#   - La voiture s'arrête immédiatement sur commande STOP
#   - Marche arrière automatique en cas de blocage
#   - La voiture va dans le bon sens par rapport aux 2 couleurs des murs
#
# contrôleur neuronal virtuel différentiel
# reprojeté en commandes Ackermann
# + filtrage moyenneur des données LiDAR
# + utilisation de 7 points exacts :
#   gauche: 60° et 70°
#   front lateral: +5° et -5°
#   front : 0°
#   droite: -60° et -70°
# + machine a etats NAVIGATION / BLOCAGE

import logging
import sys
import threading
import time
import cv2
import numpy as np

from commun import filtre_moyenneur, lire_point_lidar, calculer_commande_auto

import config
from robot_base import Actionneurs, CapteurLidar


# ============================================================
# Configuration du logging (un logger c'est juste un print amélioré, avec timestamp, niveau de gravité, etc.)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("covapsy.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


NAVIGATION = "NAVIGATION"
BLOCAGE = "BLOCAGE"
BACKWARD = "BACKWARD"
CAMERA_CHECKING = "CAMERA_CHECKING"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"


def distance_front_securite(tableau_lidar_mm):
    """Estime une distance frontale robuste dans une fenetre angulaire.

    Retourne None si le front n'est pas assez observe.
    """
    fen = int(getattr(config, "SECURITE_FRONT_FENETRE_DEG", 15))
    min_points = int(getattr(config, "SECURITE_FRONT_MIN_POINTS", 5))
    dmax = float(config.LIDAR_DMAX_MM)

    valeurs = []
    for a in range(-fen, fen + 1):
        idx = a % 360
        d = tableau_lidar_mm[idx]
        if 0 < d <= dmax:
            valeurs.append(float(d))

    if len(valeurs) < min_points:
        return None

    # Quantile bas pour rester prudent sans etre trop sensible au bruit ponctuel.
    return float(np.percentile(valeurs, 20))


def distance_arriere_lidar(tableau_lidar_mm):
    """Estime la distance arriere avec une fenetre autour de 180 degres."""
    return lire_point_lidar(tableau_lidar_mm, 180, fenetre_deg=12, min_points=4)


def camera_valide_direction():
    """Stub camera : a remplacer plus tard par la vraie logique vision."""
    return True


def details_depuis_scan(tableau_lidar_filtre, dmax_mm):
    """Construit un detail minimal si calculer_commande_auto ne renvoie pas de details."""
    d_l1 = float(lire_point_lidar(tableau_lidar_filtre, 60, fenetre_deg=3, min_points=2))
    d_l2 = float(lire_point_lidar(tableau_lidar_filtre, 70, fenetre_deg=3, min_points=2))
    d_lf1 = float(lire_point_lidar(tableau_lidar_filtre, 5, fenetre_deg=3, min_points=2))
    d_front = float(lire_point_lidar(tableau_lidar_filtre, 0, fenetre_deg=10, min_points=6))
    d_rf1 = float(lire_point_lidar(tableau_lidar_filtre, -5, fenetre_deg=3, min_points=2))
    d_r1 = float(lire_point_lidar(tableau_lidar_filtre, -60, fenetre_deg=3, min_points=2))
    d_r2 = float(lire_point_lidar(tableau_lidar_filtre, -70, fenetre_deg=3, min_points=2))

    def prox(d):
        return 1.0 - max(0.0, min(float(d), dmax_mm)) / dmax_mm

    return {
        "d_l1": d_l1,
        "d_l2": d_l2,
        "d_lf1": d_lf1,
        "d_front": d_front,
        "d_rf1": d_rf1,
        "d_r1": d_r1,
        "d_r2": d_r2,
        "p_l1": prox(d_l1),
        "p_l2": prox(d_l2),
        "p_lf1": prox(d_lf1),
        "p_f": prox(d_front),
        "p_rf1": prox(d_rf1),
        "p_r1": prox(d_r1),
        "p_r2": prox(d_r2),
        "u_g": float("nan"),
        "u_d": float("nan"),
    }


def gestion_commandes_clavier(mode_auto_event: threading.Event, stop_event: threading.Event, actionneurs: Actionneurs):
    """Thread de lecture commandes utilisateur (A/N/Q) sans bloquer la boucle de conduite."""
    while not stop_event.is_set():
        try:
            cmd = input("\nCommande > ").strip().upper() #input est bloquant, mais c'est pas grave car c'est dans un thread séparé
        except EOFError:
            cmd = "Q"
        except KeyboardInterrupt:
            logger.info("Interruption clavier recue (Ctrl+C) — arret du programme")
            stop_event.set()
            break

        if cmd == "A":
            if mode_auto_event.is_set():
                print("  Deja en mode auto.")
            else:
                logger.info("Mode auto active par l'utilisateur")
                print("  Mode auto active.")
                mode_auto_event.set()
                actionneurs.demarrer()

        elif cmd == "N":
            if not mode_auto_event.is_set():
                print("  Deja en mode manuel.")
            else:
                logger.info("Mode auto desactive par l'utilisateur")
                print("  Mode auto desactive.")
                mode_auto_event.clear()
                actionneurs.arreter()

        elif cmd == "Q":
            logger.info("Quitter le programme.")
            stop_event.set()

        elif cmd:
            print("  Commande inconnue. Utiliser A, N ou Q.")


def main():
    # =========================
    # Mode de fonctionnement
    # =========================
    mode_auto_event = threading.Event() # Precedamment appelé modeAuto, mais un Event est plus adapté pour la synchronisation entre threads
    stop_event = threading.Event()
    print("CoVAPSy — Conduite Autonome pour Webots")
    print("Cliquer sur la vue 3D pour commencer")
    print("a : mode auto")
    print("n : stop")
    print("q : quitter")
    
    # =========================
    # Initialisation des actionneurs
    # =========================
    act   = Actionneurs()
    # act.demarrer()
    logger.info("Actionneurs initialises et PWM actives")
    logger.info(
        "Bornes vitesse auto: min=%.3f m/s, max=%.3f m/s",
        float(config.VITESSE_AUTO_MIN_M_S),
        float(config.VITESSE_AUTO_MAX_M_S),
    )
    
    # =========================
    # Initialisation du LiDAR
    # =========================
    lidar = CapteurLidar()
    try:
        # Initialisation du matériel
        logger.info("Connexion au lidar...")
        lidar.connecter()
        lidar.demarrer()
        
        print("\n  Lidar connecté")
    except Exception as e:
        logger.error("Erreur critique : %s", e, exc_info=True)

    # =========================
    # Initialisation clavier
    # =========================
    clavier_thread = threading.Thread(
        target=gestion_commandes_clavier,
        args=(mode_auto_event, stop_event, act),
        daemon=True,
        name="thread_commandes",
    )
    clavier_thread.start()
    
    # =========================
    # Initialisation caméra
    # =========================
    # (non implémentée dans cette version, mais on peut imaginer une classe CapteurCamera similaire à CapteurLidar)
    camera = None #driver.getDevice("pi_camera")
    camera_ok = False
    sensor_time_step_ms = 50
    if camera is None:
        print("Camera non trouvée : pi_camera")
    else:
        camera.enable(sensor_time_step_ms)
        camera_ok = True
        print("Camera trouvée :", camera.getName())
        print("Resolution camera :", camera.getWidth(), "x", camera.getHeight())  

    # Etats filtres pour lisser les commandes et eviter les bascules brutales.
    v_cmd_filtre = 0.0
    angle_cmd_filtre = 0.0
    last_print_debug = 0.0

    # Machine a etats de deblocage
    etat = NAVIGATION
    sous_etat = BACKWARD
    flag_turn_right = False
    action_counter = 0
    action_steps = max(
        1,
        int(float(getattr(config, "BLOCAGE_ACTION_DURATION_S", 1.0)) / max(1e-3, float(config.BOUCLE_PERIODE_S))),
    )
    compat_signature_warned = False
    
    
    try:
        while not stop_event.is_set():
        # =========================
        # Lecture caméra
        # =========================
                # (non implémentée dans cette version)
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
        # Acquisition LiDAR et Filtre moyenneur AVANT normalisation
        # =========================
            if not lidar.lire():
                time.sleep(config.BOUCLE_PERIODE_S)
                continue
            tableau_lidar_filtre = filtre_moyenneur(lidar.tableau_mm)
        
        # =========================
        # Mode manuel / arrêt
        # =========================
            if not mode_auto_event.is_set():
                act.set_direction_degre(0)
                act.set_vitesse_m_s(0)
                etat = NAVIGATION
                sous_etat = BACKWARD
                flag_turn_right = False
                action_counter = 0
                v_cmd_filtre = 0.0
                angle_cmd_filtre = 0.0
                continue

        # ========================= 
        # Programme auto : appel de la fonction autonome
        # =========================
            try:
                v_cmd, angle_cmd, details = calculer_commande_auto(
                    tableau_lidar_filtre,
                    L_entraxe=config.L_ENTRAXE_M,
                    W_empattement=config.W_EMPATTEMENT_M,
                    maxangle_degre=config.ANGLE_DEGRE_MAX,
                    dmax=config.LIDAR_DMAX_MM,
                    v_min=config.VITESSE_AUTO_MIN_M_S,
                    v_max=config.VITESSE_AUTO_MAX_M_S,
                    debug=False,
                    retour_detail=True,
                )
            except TypeError:
                v_cmd, angle_cmd = calculer_commande_auto(
                    tableau_lidar_filtre,
                    L_entraxe=config.L_ENTRAXE_M,
                    W_empattement=config.W_EMPATTEMENT_M,
                    maxangle_degre=config.ANGLE_DEGRE_MAX,
                    dmax=config.LIDAR_DMAX_MM,
                    v_min=config.VITESSE_AUTO_MIN_M_S,
                    v_max=config.VITESSE_AUTO_MAX_M_S,
                    debug=False,
                )
                details = details_depuis_scan(tableau_lidar_filtre, float(config.LIDAR_DMAX_MM))
                if not compat_signature_warned:
                    logger.warning(
                        "Signature ancienne detectee pour calculer_commande_auto (sans retour_detail). "
                        "Mode compatibilite actif."
                    )
                    compat_signature_warned = True

            d_front = float(details["d_front"])
            d_rear = float(distance_arriere_lidar(tableau_lidar_filtre))

            # Variables sorties de boucle (commande finale appliquee aux actionneurs).
            cmd_v_out = 0.0
            cmd_angle_out = 0.0
            raison_secu = "n/a"
            d_front_sec = None
            v_cible = 0.0

            seuil_front_blocage = float(getattr(config, "SEUIL_FRONT_BLOCAGE_MM", 500.0))
            seuil_front_degagement = float(getattr(config, "SEUIL_FRONT_DEGAGEMENT_MM", 1500.0))
            seuil_arriere_degagement = float(getattr(config, "SEUIL_ARRIERE_DEGAGEMENT_MM", 300.0))
            vitesse_blocage = abs(float(getattr(config, "VITESSE_BLOCAGE_M_S", 0.5)))
            angle_recul_fixe = float(getattr(config, "ANGLE_RECUL_FIXE_DEG", 15.0))

            if etat == NAVIGATION:
                # Garde-fou: impose explicitement les bornes de vitesse autonome.
                v_cmd = max(config.VITESSE_AUTO_MIN_M_S, min(config.VITESSE_AUTO_MAX_M_S, float(v_cmd)))

                # Securite frontale robuste (override vitesse)
                d_front_sec = distance_front_securite(tableau_lidar_filtre)
                v_cible = float(v_cmd)
                raison_secu = "normal"

                if d_front_sec is None:
                    v_cible = min(v_cible, float(getattr(config, "SECURITE_VITESSE_INCERTAINE", 0.05)))
                    raison_secu = "front_incertain"
                else:
                    stop_mm = float(getattr(config, "SECURITE_FRONT_STOP_MM", 700.0))
                    slow_mm = float(getattr(config, "SECURITE_FRONT_RALENTI_MM", 1500.0))

                    if d_front_sec <= stop_mm:
                        v_cible = 0.0
                        raison_secu = "stop_front"
                    elif d_front_sec < slow_mm:
                        ratio = (d_front_sec - stop_mm) / max(1.0, (slow_mm - stop_mm))
                        v_lim = max(0.0, min(1.0, ratio)) * float(config.VITESSE_AUTO_MAX_M_S)
                        v_cible = min(v_cible, v_lim)
                        raison_secu = "ralenti_front"

                v_cible = max(float(config.VITESSE_AUTO_MIN_M_S), min(float(config.VITESSE_AUTO_MAX_M_S), v_cible))

                # Lissage commandes
                alpha_v = float(getattr(config, "FILTRE_ALPHA_VITESSE", 0.35))
                alpha_a = float(getattr(config, "FILTRE_ALPHA_ANGLE", 0.20))
                v_cmd_filtre = (1.0 - alpha_v) * v_cmd_filtre + alpha_v * v_cible
                angle_cmd_filtre = (1.0 - alpha_a) * angle_cmd_filtre + alpha_a * float(angle_cmd)

                if d_front_sec is not None and d_front_sec <= float(getattr(config, "SECURITE_FRONT_STOP_MM", 700.0)):
                    # En stop frontal, on annule immediatement la vitesse (pas de trainage du filtre).
                    v_cmd_filtre = 0.0
                    # En stop frontal, on recentre progressivement les roues.
                    angle_cmd_filtre = (1.0 - alpha_a) * angle_cmd_filtre

                v_cmd_filtre = max(float(config.VITESSE_AUTO_MIN_M_S), min(float(config.VITESSE_AUTO_MAX_M_S), v_cmd_filtre))

                if d_front < seuil_front_blocage:
                    etat = BLOCAGE
                    sous_etat = BACKWARD
                    flag_turn_right = False
                    action_counter = 0
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    logger.info("Transition NAVIGATION -> BLOCAGE")
                else:
                    cmd_v_out = float(v_cmd_filtre)
                    cmd_angle_out = float(angle_cmd_filtre)

            else:
                raison_secu = "etat_blocage"
                d_front_sec = d_front

                if sous_etat == BACKWARD:
                    if action_counter >= action_steps or d_rear <= seuil_arriere_degagement:
                        cmd_v_out = 0.0
                        cmd_angle_out = 0.0
                        action_counter = 0
                        if flag_turn_right:
                            sous_etat = TURN_RIGHT
                            logger.info("BACKWARD -> TURN_RIGHT")
                        else:
                            sous_etat = TURN_LEFT
                            logger.info("BACKWARD -> TURN_LEFT")
                    else:
                        cmd_angle_out = 0.0
                        cmd_v_out = -vitesse_blocage
                        action_counter += 1

                elif sous_etat == TURN_LEFT:
                    if action_counter >= action_steps:
                        cmd_v_out = 0.0
                        cmd_angle_out = 0.0
                        action_counter = 0
                        if d_front > seuil_front_degagement:
                            sous_etat = CAMERA_CHECKING
                            logger.info("TURN_LEFT -> CAMERA_CHECKING")
                        else:
                            sous_etat = BACKWARD
                            logger.info("TURN_LEFT -> BACKWARD")
                    else:
                        cmd_angle_out = angle_recul_fixe
                        cmd_v_out = vitesse_blocage
                        action_counter += 1

                elif sous_etat == TURN_RIGHT:
                    if action_counter >= action_steps:
                        cmd_v_out = 0.0
                        cmd_angle_out = 0.0
                        action_counter = 0
                        if d_front > seuil_front_degagement:
                            sous_etat = CAMERA_CHECKING
                            logger.info("TURN_RIGHT -> CAMERA_CHECKING")
                        else:
                            sous_etat = BACKWARD
                            logger.info("TURN_RIGHT -> BACKWARD")
                    else:
                        cmd_angle_out = -angle_recul_fixe
                        cmd_v_out = vitesse_blocage
                        action_counter += 1

                elif sous_etat == CAMERA_CHECKING:
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    action_counter = 0
                    if camera_valide_direction():
                        etat = NAVIGATION
                        sous_etat = BACKWARD
                        flag_turn_right = False
                        v_cmd_filtre = 0.0
                        angle_cmd_filtre = 0.0
                        logger.info("CAMERA_CHECKING -> NAVIGATION")
                    else:
                        etat = BLOCAGE
                        sous_etat = BACKWARD
                        flag_turn_right = True
                        logger.info("CAMERA_CHECKING -> BLOCAGE")

                else:
                    # Fallback robuste en cas de sous-etat inattendu.
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    sous_etat = BACKWARD
                    action_counter = 0

            now = time.time()
            if now - last_print_debug >= float(getattr(config, "DEBUG_PRINT_PERIOD_S", 0.5)):
                last_print_debug = now
                print("--------------------------------------------------")
                print(f"[ETAT] {etat}  |  [SOUS-ETAT] {sous_etat if etat == BLOCAGE else '-'}")
                print(f"d_l1={details['d_l1']:.1f} d_l2={details['d_l2']:.1f} d_lf1={details['d_lf1']:.1f} d_front={d_front:.1f} d_rf1={details['d_rf1']:.1f} d_r1={details['d_r1']:.1f} d_r2={details['d_r2']:.1f} d_rear={d_rear:.1f}")
                print(f"u_g={details['u_g']:.3f} u_d={details['u_d']:.3f} v_raw={v_cmd:.3f} angle_raw={angle_cmd:.3f}")
                if d_front_sec is None:
                    print(f"[SECU] front=INCERTAIN v_safe={v_cible:.3f} reason={raison_secu}")
                else:
                    print(f"[SECU] front={d_front_sec:.1f}mm v_safe={v_cible:.3f} reason={raison_secu}")
                print(f"[CMD] v_out={cmd_v_out:.3f} m/s angle_out={cmd_angle_out:.3f} deg")
        
            # Si le sens de rotation est inversé, passer -angle_cmd ici :
            # angle_cmd = -angle_cmd
        
            # 7) Commande véhicule
            act.set_direction_degre(cmd_angle_out)
            act.set_vitesse_m_s(cmd_v_out)
    except KeyboardInterrupt:
        logger.info("Interruption clavier recue (Ctrl+C) — arret propre")
        stop_event.set()
    
    finally:
        # =========================
        # Arrêt propre
        # =========================
        logger.info("Arrêt propre en cours...")
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        try:
            act.arreter()
        except Exception:
            pass
        lidar.arreter()
        logger.info("Programme terminé.")
        print("Au revoir.")

if __name__ == "__main__":
    main()