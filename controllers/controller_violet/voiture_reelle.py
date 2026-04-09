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
# + utilisation de 5 points exacts :
#   gauche: 60° et 70°
#   front : 0°
#   droite: -60° et -70°

import logging
import os
import sys
import threading
import time
import importlib

from commun import (
    filtre_moyenneur,
    analyze_walls,
    calculer_commande_automate,
    reset_automate,
    get_automate_state,
)

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

ETAT_NAMES = {0: "NAVIGATION", 1: "BLOCAGE"}
SOUS_ETAT_NAMES = {0: "BACKWARD", 1: "CAMERA_CHECKING", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def initialiser_camera():
    """Initialise la camera Raspberry Pi si disponible.

    Retourne:
        camera_obj | None
        show_window (bool)
    """
    if not bool(config.CAMERA_ACTIVE):
        logger.info("Camera desactivee par config")
        return None, False

    try:
        picamera2_mod = importlib.import_module("picamera2")
        picamera2_cls = getattr(picamera2_mod, "Picamera2", None)
    except Exception:
        picamera2_cls = None

    if picamera2_cls is None:
        logger.warning("Picamera2 indisponible: verification camera desactivee")
        return None, False

    show_window = bool(config.CAMERA_SHOW_WINDOW) and (
        ("DISPLAY" in os.environ) or (os.name == "nt")
    )

    try:
        camera = picamera2_cls()
        width = int(config.CAMERA_WIDTH)
        height = int(config.CAMERA_HEIGHT)

        cfg = camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            queue=False,
        )
        camera.configure(cfg)
        camera.start()
        time.sleep(1.0)
        logger.info("Camera initialisee (%dx%d)", width, height)
        return camera, show_window
    except Exception as exc:
        logger.warning("Echec initialisation camera: %s", exc)
        return None, False


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
    # Initialisation camera
    # =========================
    camera, show_camera_window = initialiser_camera()

    # =========================
    # Initialisation automate
    # =========================
    reset_automate()

    try:
        while not stop_event.is_set():
            # =========================
            # Acquisition LiDAR et filtre
            # =========================
            if not lidar.lire():
                time.sleep(config.BOUCLE_PERIODE_S)
                continue
            tableau_lidar_filtre = filtre_moyenneur(lidar.tableau_mm)

            # =========================
            # Camera (optionnelle)
            # =========================
            wall_info = None
            if camera is not None:
                try:
                    frame_rgb = camera.capture_array("main")
                    frame_bgr = frame_rgb[:, :, ::-1].copy()
                    wall_info = analyze_walls(
                        frame_bgr,
                        band_ratio=float(config.CAMERA_BAND_RATIO),
                        min_ratio=float(config.CAMERA_MIN_RATIO),
                        dominance=float(config.CAMERA_DOMINANCE),
                        unknown_value=int(config.CAMERA_UNKNOWN_VALUE),
                    )

                    if show_camera_window and wall_info is not None:
                        try:
                            cv2_mod = importlib.import_module("cv2")
                            cv2_mod.imshow("Camera murs rouge/vert", wall_info["annotated"])
                            cv2_mod.waitKey(1)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("Erreur lecture camera: %s", exc)

            # =========================
            # Mode manuel / arret
            # =========================
            if not mode_auto_event.is_set():
                reset_automate()
                time.sleep(config.BOUCLE_PERIODE_S)
                continue

            # =========================
            # Commande auto finale (reseau + automate)
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

            act.set_direction_degre(float(angle_cmd))
            act.set_vitesse_m_s(float(v_cmd))

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
                logger.info(
                    "FSM [%s|%s] v=%.2f ang=%.1f dF=%s dR=%s back=%d obj=%s vrel=%.2f sL=%.2f sR=%.2f walls=%s",
                    ETAT_NAMES.get(etat, "?"),
                    ss,
                    float(v_cmd),
                    float(angle_cmd),
                    front_txt,
                    rear_txt,
                    int(fsm.get("counter_etat_backward", 0)),
                    obj_kind,
                    obj_speed,
                    s_left,
                    s_right,
                    wall_txt,
                )

            time.sleep(config.BOUCLE_PERIODE_S)
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
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        if show_camera_window:
            try:
                cv2_mod = importlib.import_module("cv2")
                cv2_mod.destroyAllWindows()
            except Exception:
                pass
        lidar.arreter()
        logger.info("Programme terminé.")
        print("Au revoir.")

if __name__ == "__main__":
    main()