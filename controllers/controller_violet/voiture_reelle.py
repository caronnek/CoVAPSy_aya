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
    calculer_commande_auto,
    distance_secteur_lidar,
    analyze_walls,
    check_camera_direction,
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

# ============================================================
# Etats FSM
# ============================================================
NAVIGATION = 0
BLOCAGE = 1

BACKWARD = 0
CAMERA_CHECKING = 1
TURN_LEFT = 2
TURN_RIGHT = 3

ETAT_NAMES = {NAVIGATION: "NAVIGATION", BLOCAGE: "BLOCAGE"}
SOUS_ETAT_NAMES = {
    BACKWARD: "BACKWARD",
    CAMERA_CHECKING: "CAMERA_CHECKING",
    TURN_LEFT: "TURN_LEFT",
    TURN_RIGHT: "TURN_RIGHT",
}


def initialiser_camera():
    """Initialise la camera Raspberry Pi si disponible.

    Retourne:
        camera_obj | None
        show_window (bool)
    """
    if not bool(getattr(config, "CAMERA_ACTIVE", True)):
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

    show_window = bool(getattr(config, "CAMERA_SHOW_WINDOW", False)) and (
        ("DISPLAY" in os.environ) or (os.name == "nt")
    )

    try:
        camera = picamera2_cls()
        width = int(getattr(config, "CAMERA_WIDTH", 640))
        height = int(getattr(config, "CAMERA_HEIGHT", 480))

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
    # Initialisation FSM
    # =========================
    etat = NAVIGATION
    sous_etat = BACKWARD
    action_counter = 0
    counter_camera_confirm = 0
    counter_etat_backward = 0
    flag_turn_right = False

    action_steps = max(
        1,
        int(
            float(getattr(config, "BLOCAGE_ACTION_DURATION_S", 1.0))
            / max(1e-4, float(getattr(config, "BOUCLE_PERIODE_S", 0.01)))
        ),
    )

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
                        band_ratio=float(getattr(config, "CAMERA_BAND_RATIO", 0.30)),
                        min_ratio=float(getattr(config, "CAMERA_MIN_RATIO", 0.03)),
                        dominance=float(getattr(config, "CAMERA_DOMINANCE", 1.15)),
                        unknown_value=int(getattr(config, "CAMERA_UNKNOWN_VALUE", -1)),
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
                etat = NAVIGATION
                sous_etat = BACKWARD
                action_counter = 0
                counter_camera_confirm = 0
                counter_etat_backward = 0
                flag_turn_right = False
                time.sleep(config.BOUCLE_PERIODE_S)
                continue

            # =========================
            # Commande auto de base (reseau)
            # =========================
            v_cmd, angle_cmd = calculer_commande_auto(
                tableau_lidar_filtre,
                L_entraxe=config.L_ENTRAXE_M,
                W_empattement=config.W_EMPATTEMENT_M,
                maxangle_degre=config.ANGLE_DEGRE_MAX,
                dmax=config.LIDAR_DMAX_MM,
                v_min=config.VITESSE_AUTO_MIN_M_S,
                v_max=config.VITESSE_AUTO_MAX_M_S,
                debug=bool(getattr(config, "AUTO_DEBUG", True)),
            )

            # =========================
            # Distances robustes front/arriere
            # =========================
            d_front = distance_secteur_lidar(
                tableau_lidar_filtre,
                centre_deg=0,
                demi_fenetre_deg=int(getattr(config, "SECURITE_FRONT_FENETRE_DEG", 15)),
                dmax=float(config.LIDAR_DMAX_MM),
                min_points=int(getattr(config, "SECURITE_FRONT_MIN_POINTS", 5)),
                quantile=20,
            )
            d_rear = distance_secteur_lidar(
                tableau_lidar_filtre,
                centre_deg=180,
                demi_fenetre_deg=int(getattr(config, "LIDAR_REAR_WINDOW_DEG", 15)),
                dmax=float(config.LIDAR_DMAX_MM),
                min_points=int(getattr(config, "LIDAR_REAR_MIN_POINTS", 4)),
                quantile=20,
            )

            front_blocked = d_front is not None and d_front <= float(config.SEUIL_FRONT_BLOCAGE_MM)
            front_clear = d_front is not None and d_front >= float(config.SEUIL_FRONT_DEGAGEMENT_MM)

            # =========================
            # Anti-collision frontale (limite vitesse)
            # =========================
            v_safe = float(v_cmd)
            if d_front is None:
                v_safe = min(v_safe, float(getattr(config, "SECURITE_VITESSE_INCERTAINE", 0.05)))
            else:
                stop_mm = float(getattr(config, "SECURITE_FRONT_STOP_MM", 700.0))
                slow_mm = float(getattr(config, "SECURITE_FRONT_RALENTI_MM", 1500.0))

                if d_front <= stop_mm:
                    v_safe = 0.0
                elif d_front < slow_mm:
                    ratio = (d_front - stop_mm) / max(1.0, slow_mm - stop_mm)
                    v_lim = max(0.0, min(1.0, ratio)) * float(config.VITESSE_AUTO_MAX_M_S)
                    v_safe = min(v_safe, v_lim)

            v_safe = max(0.0, min(float(config.VITESSE_AUTO_MAX_M_S), v_safe))

            # =========================
            # FSM (navigation / deblocage)
            # =========================
            match etat:
                case 0:
                    if front_blocked:
                        etat = BLOCAGE
                        sous_etat = BACKWARD
                        action_counter = 0
                        counter_camera_confirm = 0
                    else:
                        act.set_direction_degre(float(angle_cmd))
                        act.set_vitesse_m_s(v_safe)

                case 1:
                    match sous_etat:
                        case 0:
                            rear_too_close = (
                                d_rear is not None
                                and d_rear <= float(config.SEUIL_ARRIERE_DEGAGEMENT_MM)
                            )
                            if action_counter >= action_steps or rear_too_close:
                                act.set_direction_degre(0.0)
                                act.set_vitesse_m_s(0.0)
                                action_counter = 0
                                if flag_turn_right and counter_etat_backward > int(getattr(config, "SEUIL_BLOCAGE_PERSIST_STEPS", 5)):
                                    sous_etat = TURN_RIGHT
                                else:
                                    sous_etat = TURN_LEFT
                            else:
                                act.set_direction_degre(0.0)
                                act.set_vitesse_m_s(-abs(float(config.VITESSE_BLOCAGE_M_S)))
                                action_counter += 1

                        case 2:
                            if action_counter >= action_steps:
                                act.set_direction_degre(0.0)
                                act.set_vitesse_m_s(0.0)
                                action_counter = 0
                                if front_clear:
                                    counter_camera_confirm = 0
                                    sous_etat = CAMERA_CHECKING
                                else:
                                    sous_etat = BACKWARD
                                    counter_etat_backward += 1
                            else:
                                act.set_direction_degre(float(config.ANGLE_RECUL_FIXE_DEG))
                                act.set_vitesse_m_s(abs(float(config.VITESSE_BLOCAGE_M_S)))
                                action_counter += 1

                        case 1:
                            act.set_direction_degre(0.0)
                            act.set_vitesse_m_s(0.0)

                            values = wall_info["value"] if wall_info else None
                            camera_ok = check_camera_direction(
                                values,
                                direction=int(getattr(config, "CAMERA_DIRECTION_EXPECTED", 0)),
                                unknown_value=int(getattr(config, "CAMERA_UNKNOWN_VALUE", -1)),
                            )

                            if camera_ok:
                                counter_camera_confirm += 1
                            else:
                                counter_camera_confirm = 0

                            action_counter += 1
                            if counter_camera_confirm >= int(getattr(config, "CAMERA_CONFIRM_STEPS", 3)):
                                etat = NAVIGATION
                                sous_etat = BACKWARD
                                action_counter = 0
                                flag_turn_right = False
                                counter_etat_backward = 0
                                counter_camera_confirm = 0
                            elif action_counter >= int(getattr(config, "CAMERA_CONFIRM_STEPS", 3)):
                                etat = BLOCAGE
                                sous_etat = BACKWARD
                                action_counter = 0
                                flag_turn_right = True

                        case 3:
                            if action_counter >= action_steps:
                                act.set_direction_degre(0.0)
                                act.set_vitesse_m_s(0.0)
                                action_counter = 0
                                if front_clear:
                                    counter_camera_confirm = 0
                                    sous_etat = CAMERA_CHECKING
                                else:
                                    sous_etat = BACKWARD
                            else:
                                act.set_direction_degre(-float(config.ANGLE_RECUL_FIXE_DEG))
                                act.set_vitesse_m_s(abs(float(config.VITESSE_BLOCAGE_M_S)))
                                action_counter += 1

                        case _:
                            sous_etat = BACKWARD

                case _:
                    etat = NAVIGATION
                    sous_etat = BACKWARD

            if bool(getattr(config, "DEBUG_ACTIONNEURS", False)):
                ss = SOUS_ETAT_NAMES[sous_etat] if etat == BLOCAGE else "-"
                front_txt = "NA" if d_front is None else f"{d_front:.0f}"
                rear_txt = "NA" if d_rear is None else f"{d_rear:.0f}"
                logger.info(
                    "FSM [%s|%s] v=%.2f ang=%.1f dF=%s dR=%s back=%d",
                    ETAT_NAMES.get(etat, "?"),
                    ss,
                    v_safe,
                    float(angle_cmd),
                    front_txt,
                    rear_txt,
                    counter_etat_backward,
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