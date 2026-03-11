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
import sys
import threading
import time
import cv2
import numpy as np

from commun import filtre_moyenneur, lire_point_lidar, normaliser_distance, differentiel_vers_ackermann, calculer_commande_auto

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
    last_print_secu = 0.0
    
    
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
                continue

        # ========================= 
        # Programme auto : appel de la fonction autonome
        # =========================
            v_cmd, angle_cmd = calculer_commande_auto(
                tableau_lidar_filtre,
                L_entraxe=config.L_ENTRAXE_M,
                W_empattement=config.W_EMPATTEMENT_M,
                maxangle_degre=config.ANGLE_DEGRE_MAX,
                dmax=config.LIDAR_DMAX_MM,
                v_min=config.VITESSE_AUTO_MIN_M_S,
                v_max=config.VITESSE_AUTO_MAX_M_S,
                debug=True,
            )

            # 6bis) Securite frontale robuste (override vitesse)
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

            # 6ter) Lissage commandes
            alpha_v = float(getattr(config, "FILTRE_ALPHA_VITESSE", 0.35))
            alpha_a = float(getattr(config, "FILTRE_ALPHA_ANGLE", 0.20))
            v_cmd_filtre = (1.0 - alpha_v) * v_cmd_filtre + alpha_v * v_cible
            angle_cmd_filtre = (1.0 - alpha_a) * angle_cmd_filtre + alpha_a * float(angle_cmd)

            if d_front_sec is not None and d_front_sec <= float(getattr(config, "SECURITE_FRONT_STOP_MM", 700.0)):
                # En stop frontal, on recentre progressivement les roues.
                angle_cmd_filtre = (1.0 - alpha_a) * angle_cmd_filtre

            now = time.time()
            if now - last_print_secu >= float(getattr(config, "DEBUG_PRINT_PERIOD_S", 0.5)):
                last_print_secu = now
                if d_front_sec is None:
                    print(f"[SECU] front=INCERTAIN  v_raw={v_cmd:.3f}  v_safe={v_cible:.3f}  v_out={v_cmd_filtre:.3f}  reason={raison_secu}")
                else:
                    print(f"[SECU] front={d_front_sec:.1f}mm  v_raw={v_cmd:.3f}  v_safe={v_cible:.3f}  v_out={v_cmd_filtre:.3f}  reason={raison_secu}")
        
        # Si le sens de rotation est inversé, passer -angle_cmd ici :
        # angle_cmd = -angle_cmd
        
            # 7) Commande véhicule
            act.set_direction_degre(angle_cmd_filtre)
            act.set_vitesse_m_s(v_cmd_filtre)
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