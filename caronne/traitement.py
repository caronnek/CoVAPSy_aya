# traitement.py
# Traitement LiDAR + réseau neuronal différentiel → v_cmd, angle_cmd, p_stop

import numpy as np

# =========================
# Paramètres véhicule
# =========================
maxSpeed       = 50
maxangle_degre = 18
L_entraxe      = 0.180
W_empattement  = 0.250

# =========================
# Poids réseau (sortis de la boucle)
# =========================
W_G = np.array([ 1.2,  1.00,  0.70, -1.2, -1.20,  1.20, -0.65, -0.90, -0.5])
W_D = np.array([ 1.2, -0.90, -0.65, -1.2,  1.20, -1.20,  0.70,  1.00, -0.5])

# =========================
# Mémoire récurrente
# =========================
_p_fL_memo = 0.0
_p_fR_memo = 0.0
_diff_lr_memo = 0.0
ALPHA = 0.95

# =========================
# Fonctions LiDAR
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
    if idx < -180: idx += 360
    elif idx > 179: idx -= 360
    valeur = tab[idx]
    return valeur if valeur > 0 else valeur_defaut

def _norm(d, dmax):
    return max(0.0, min(d, dmax)) / dmax

# =========================
# Conversion différentiel -> Ackermann
# =========================
def differentiel_vers_ackermann(u_g, u_d):
    v_norm = (u_g + u_d) / 2.0
    omega  = (u_d - u_g) / L_entraxe
    v_cmd  = 1.2 * max(0.0, v_norm)
    if abs(omega) < 1e-4:
        angle_deg = 0.0
    else:
        R = v_norm / omega if abs(v_norm) > 1e-4 else 1e6
        angle_deg = float(np.degrees(np.arctan(W_empattement / R)))
    angle_deg = float(np.clip(angle_deg, -maxangle_degre, maxangle_degre))
    return v_cmd, angle_deg

# =========================
# Fonction principale appelée à chaque step
# =========================
def calculer(tableau_lidar_filtre):
    """
    Prend le tableau LiDAR filtré (360 valeurs en mm).
    Retourne : v_cmd, angle_cmd, p_stop, debug_dict
    """
    global _p_fL_memo, _p_fR_memo, _diff_lr_memo

    dmax     = 3000.0
    dmax_obs = 1000.0

    # Lecture distances
    d_l1  = lire_point_lidar(tableau_lidar_filtre,  63)
    d_l2  = lire_point_lidar(tableau_lidar_filtre,  73)
    d_f0  = lire_point_lidar(tableau_lidar_filtre,   0)
    d_r1  = lire_point_lidar(tableau_lidar_filtre, -63)
    d_r2  = lire_point_lidar(tableau_lidar_filtre, -73)
    d_fL  = lire_point_lidar(tableau_lidar_filtre,   6)
    d_fR  = lire_point_lidar(tableau_lidar_filtre,  -6)
    d_fwL = lire_point_lidar(tableau_lidar_filtre,  30)
    d_fwR = lire_point_lidar(tableau_lidar_filtre, -30)

    # Proximités murs
    p_l1 = 1.0 - _norm(d_l1, dmax)
    p_l2 = 1.0 - _norm(d_l2, dmax)
    p_f0 = 1.0 - _norm(d_f0, dmax)
    p_r1 = 1.0 - _norm(d_r1, dmax)
    p_r2 = 1.0 - _norm(d_r2, dmax)

    # Évitement frontal + mémoire récurrente
    p_f0_obs  = 1.0 - _norm(d_f0,  dmax_obs)
    p_fL_brut = 1.0 - _norm(d_fL,  dmax_obs)
    p_fR_brut = 1.0 - _norm(d_fR,  dmax_obs)
    p_fwL     = 1.0 - _norm(d_fwL, dmax_obs)
    p_fwR     = 1.0 - _norm(d_fwR, dmax_obs)

    _p_fL_memo = max(p_fL_brut, ALPHA * _p_fL_memo)
    _p_fR_memo = max(p_fR_brut, ALPHA * _p_fR_memo)

    declencheur  = max(p_f0_obs, _p_fL_memo, _p_fR_memo)
    diff_lr_brut = float(np.clip((d_fL - d_fR) / dmax_obs, -1.0, 1.0))
    _diff_lr_memo = ALPHA * _diff_lr_memo + (1.0 - ALPHA) * diff_lr_brut

    biais_g = max(0.0,  _diff_lr_memo) + 0.3 * _norm(d_r1, dmax)
    biais_d = max(0.0, -_diff_lr_memo) + 0.3 * _norm(d_l1, dmax)

    p_fL_combine = declencheur * biais_g
    p_fR_combine = declencheur * biais_d

    # Neurone stop
    p_stop = float(np.clip((p_f0_obs ** 2) * min(p_fwL, p_fwR), 0.0, 1.0))

    # Réseau neuronal
    x = np.array([1.0, p_l1, p_l2, p_f0,
                  p_fL_combine, p_fR_combine, p_r1, p_r2, p_stop])

    u_g = float(np.tanh(np.dot(x, W_G)))
    u_d = float(np.tanh(np.dot(x, W_D)))

    v_cmd, angle_cmd = differentiel_vers_ackermann(u_g, u_d)

    debug = {
        "u_g": u_g, "u_d": u_d,
        "v_cmd": v_cmd, "angle_cmd": angle_cmd,
        "p_stop": p_stop, "declencheur": declencheur,
        "p_fL": p_fL_combine, "p_fR": p_fR_combine,
        "diff_lr_brut": diff_lr_brut, "diff_lr_memo": _diff_lr_memo,
        "p_fL_memo": _p_fL_memo, "p_fR_memo": _p_fR_memo,
        "p_fwL": p_fwL, "p_fwR": p_fwR,
        "d_fL": d_fL, "d_fR": d_fR, "d_f0": d_f0,
        "d_fwL": d_fwL, "d_fwR": d_fwR,
    }

    return v_cmd, angle_cmd, p_stop, debug
