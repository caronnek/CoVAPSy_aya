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

from commun import filtre_moyenneur, AutomateConduite

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


def camera_valide_direction():
    """Stub camera : a remplacer plus tard par la vraie logique vision."""
    return True


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

    # Debug d'affichage periodique des etats de l'automate.
    last_print_debug = 0.0

    automate = AutomateConduite(
        L_entraxe=config.L_ENTRAXE_M,
        W_empattement=config.W_EMPATTEMENT_M,
        maxangle_degre=config.ANGLE_DEGRE_MAX,
        dmax=config.LIDAR_DMAX_MM,
        v_min=config.VITESSE_AUTO_MIN_M_S,
        v_max=config.VITESSE_AUTO_MAX_M_S,
        securite_front_fenetre_deg=config.SECURITE_FRONT_FENETRE_DEG,
        securite_front_min_points=config.SECURITE_FRONT_MIN_POINTS,
        securite_vitesse_incertaine=config.SECURITE_VITESSE_INCERTAINE,
        securite_front_stop_mm=config.SECURITE_FRONT_STOP_MM,
        securite_front_ralenti_mm=config.SECURITE_FRONT_RALENTI_MM,
        filtre_alpha_vitesse=config.FILTRE_ALPHA_VITESSE,
        filtre_alpha_angle=config.FILTRE_ALPHA_ANGLE,
        seuil_front_blocage_mm=config.SEUIL_FRONT_BLOCAGE_MM,
        seuil_front_degagement_mm=config.SEUIL_FRONT_DEGAGEMENT_MM,
        seuil_arriere_degagement_mm=config.SEUIL_ARRIERE_DEGAGEMENT_MM,
        angle_recul_fixe_deg=config.ANGLE_RECUL_FIXE_DEG,
        vitesse_blocage_m_s=config.VITESSE_BLOCAGE_M_S,
        blocage_action_duration_s=config.BLOCAGE_ACTION_DURATION_S,
        boucle_periode_s=config.BOUCLE_PERIODE_S,
        camera_valide_fn=camera_valide_direction,
    )
    
    
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
                automate.reset()
                continue

        # ========================= 
        # Programme auto : appel de la fonction autonome
        # =========================
            v_cmd, angle_cmd = automate.calculer_commande(tableau_lidar_filtre)
            debug_auto = automate.last_debug
            details = debug_auto.get("details", {})
            d_front = float(debug_auto.get("d_front", 0.0))
            d_rear = float(debug_auto.get("d_rear", 0.0))
            d_front_sec = debug_auto.get("d_front_sec", None)
            v_cible = float(debug_auto.get("v_cible", 0.0))
            raison_secu = str(debug_auto.get("raison_secu", "n/a"))
            etat = str(debug_auto.get("etat", "?"))
            sous_etat = str(debug_auto.get("sous_etat", "-"))

            now = time.time()
            if now - last_print_debug >= float(config.DEBUG_PRINT_PERIOD_S):
                last_print_debug = now
                print("--------------------------------------------------")
                print(f"[ETAT] {etat}  |  [SOUS-ETAT] {sous_etat}")
                print(
                    f"d_l1={float(details.get('d_l1', 0.0)):.1f} "
                    f"d_l2={float(details.get('d_l2', 0.0)):.1f} "
                    f"d_lf1={float(details.get('d_lf1', 0.0)):.1f} "
                    f"d_front={d_front:.1f} "
                    f"d_rf1={float(details.get('d_rf1', 0.0)):.1f} "
                    f"d_r1={float(details.get('d_r1', 0.0)):.1f} "
                    f"d_r2={float(details.get('d_r2', 0.0)):.1f} "
                    f"d_rear={d_rear:.1f}"
                )
                print(
                    f"u_g={float(details.get('u_g', float('nan'))):.3f} "
                    f"u_d={float(details.get('u_d', float('nan'))):.3f} "
                    f"v_raw={float(debug_auto.get('v_raw', 0.0)):.3f} "
                    f"angle_raw={float(debug_auto.get('angle_raw', 0.0)):.3f}"
                )
                if d_front_sec is None:
                    print(f"[SECU] front=INCERTAIN v_safe={v_cible:.3f} reason={raison_secu}")
                else:
                    print(f"[SECU] front={d_front_sec:.1f}mm v_safe={v_cible:.3f} reason={raison_secu}")
                print(f"[CMD] v_out={v_cmd:.3f} m/s angle_out={angle_cmd:.3f} deg")
        
            # Si le sens de rotation est inversé, passer -angle_cmd ici :
            # angle_cmd = -angle_cmd
        
            # 7) Commande véhicule
            act.set_direction_degre(angle_cmd)
            act.set_vitesse_m_s(v_cmd)
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