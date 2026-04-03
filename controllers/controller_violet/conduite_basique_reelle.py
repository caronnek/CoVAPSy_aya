"""Conduite basique reelle: LiDAR -> Actionneurs.

Comportement:
- Si le front est libre: avance.
- Si un obstacle apparait devant: ralentit progressivement.
- Si obstacle tres proche: arret.

Usage:
    /usr/bin/python controllers/controller_violet/conduite_basique_reelle.py
"""

import time

import config
from robot_base import Actionneurs, CapteurLidar


# -----------------------------
# Reglages comportement basique
# -----------------------------
FRONT_WINDOW_DEG = 15      # fenetre [-15, +15] autour du front
FRONT_MIN_POINTS = 6       # nb mini de points valides pour juger le front fiable
STOP_DIST_MM = 450.0       # arret immediat sous ce seuil
SLOW_DIST_MM = 1700.0      # vitesse reduite sous ce seuil
VITESSE_MAX_M_S = 0.22     # vitesse de test prudente
ALPHA_V = 0.35             # lissage vitesse (0 stable, 1 reactif)
PRINT_PERIOD_S = 0.25
# Auto-detect du centre frontal selon montage lidar.
# Si besoin, fixer ensuite manuellement a la meilleure valeur observee.
FRONT_CANDIDATE_CENTERS = (0, 90, 180, 270)


def _window_values(tableau_mm, center_deg, window_deg):
    """Extrait les valeurs valides dans une fenetre angulaire autour d'un centre."""
    valeurs = []
    dmax = float(config.LIDAR_DMAX_MM)
    for dtheta in range(-window_deg, window_deg + 1):
        idx = (center_deg + dtheta) % 360
        d = tableau_mm[idx]
        if 0 < d <= dmax:
            valeurs.append(float(d))
    return valeurs


def front_distance_mm(tableau_mm):
    """Retourne une distance frontale robuste, nb points et centre frontal retenu.

    La distance renvoyee est un quantile bas (25%) pour etre prudente.
    Retourne (None, n, center) si pas assez de points valides.
    """
    best_center = None
    best_values = []

    for center in FRONT_CANDIDATE_CENTERS:
        vals = _window_values(tableau_mm, center, FRONT_WINDOW_DEG)
        if len(vals) > len(best_values):
            best_values = vals
            best_center = center

    n = len(best_values)
    if n < FRONT_MIN_POINTS:
        return None, n, best_center

    best_values.sort()
    idx = int(0.25 * (n - 1))
    return best_values[idx], n, best_center


def vitesse_cible_m_s(dist_front_mm):
    """Calcule la vitesse cible selon la distance frontale."""
    if dist_front_mm is None:
        return 0.0, "front_incertain"

    if dist_front_mm <= STOP_DIST_MM:
        return 0.0, "stop"

    if dist_front_mm < SLOW_DIST_MM:
        ratio = (dist_front_mm - STOP_DIST_MM) / max(1.0, (SLOW_DIST_MM - STOP_DIST_MM))
        return max(0.0, min(1.0, ratio)) * VITESSE_MAX_M_S, "ralenti"

    return VITESSE_MAX_M_S, "avance"


def main():
    act = Actionneurs()
    lidar = CapteurLidar()

    print("Conduite basique reelle (LiDAR -> Actionneurs)")
    print("Ctrl+C pour arreter")

    v_cmd = 0.0
    last_print = 0.0

    try:
        lidar.connecter()
        lidar.demarrer()
        act.demarrer()

        while True:
            if not lidar.lire():
                time.sleep(config.BOUCLE_PERIODE_S)
                continue

            dist_front_mm, nb_pts, front_center = front_distance_mm(lidar.tableau_mm)
            v_target, etat = vitesse_cible_m_s(dist_front_mm)

            # Lissage simple de vitesse pour eviter les a-coups.
            v_cmd = (1.0 - ALPHA_V) * v_cmd + ALPHA_V * v_target
            if v_target == 0.0 and v_cmd < 0.01:
                v_cmd = 0.0

            act.set_direction_degre(0.0)
            act.set_vitesse_m_s(v_cmd)

            now = time.time()
            if now - last_print >= PRINT_PERIOD_S:
                last_print = now
                if dist_front_mm is None:
                    print(
                        f"[BASIQUE] etat={etat:13s} front=INCERTAIN "
                        f"pts={nb_pts:2d} center={front_center} "
                        f"v_target={v_target:.3f} v_cmd={v_cmd:.3f}"
                    )
                else:
                    print(
                        f"[BASIQUE] etat={etat:13s} front={dist_front_mm:7.1f} mm "
                        f"pts={nb_pts:2d} center={front_center} "
                        f"v_target={v_target:.3f} v_cmd={v_cmd:.3f}"
                    )

    except KeyboardInterrupt:
        print("\nInterruption clavier (Ctrl+C)")

    finally:
        print("Arret propre...")
        try:
            act.arreter()
        except Exception:
            pass
        try:
            lidar.arreter()
        except Exception:
            pass
        print("Termine")


if __name__ == "__main__":
    main()
