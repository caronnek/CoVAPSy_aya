import numpy as np
import cv2

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

def lire_point_lidar(tab, angle_deg, valeur_defaut=3000.0, fenetre_deg=4, min_points=1):
    """
    Lit un point lidar pour un angle donné.
    Si le point exact est invalide, cherche dans une petite fenetre angulaire
    et renvoie la mediane des valeurs valides.

    Les indices sont accessibles dans [-180, 179].
    """
    idx = int(angle_deg)

    if idx < -180:
        idx += 360
    elif idx > 179:
        idx -= 360

    valeurs = []
    for delta in range(-fenetre_deg, fenetre_deg + 1):
        k = (idx + delta) % 360
        valeur = tab[k]
        if 0 < valeur <= valeur_defaut:
            valeurs.append(valeur)

    if len(valeurs) < int(min_points):
        return valeur_defaut

    return float(np.median(valeurs))

def normaliser_distance(d, dmax):
    d = max(0.0, min(d, dmax))
    return d / dmax


def distance_secteur_lidar(tab_mm, centre_deg, demi_fenetre_deg=15,
                           dmax=3000.0, min_points=5, quantile=20):
    """
    Estime une distance robuste dans un secteur angulaire du LiDAR.

    Exemple: centre_deg=0 pour le front, centre_deg=180 pour l'arriere.
    Retourne None si trop peu de points valides.
    """
    centre = int(centre_deg) % 360
    fen = int(max(0, demi_fenetre_deg))

    valeurs = []
    for delta in range(-fen, fen + 1):
        idx = (centre + delta) % 360
        d = float(tab_mm[idx])
        if 0.0 < d <= float(dmax):
            valeurs.append(d)

    if len(valeurs) < int(min_points):
        return None

    q = float(np.clip(quantile, 0.0, 100.0))
    return float(np.percentile(valeurs, q))


def detect_color_hsv(roi_bgr,
                     min_ratio=0.03,
                     dominance=1.15,
                     unknown_value=-1):
    """
    Detecte la couleur dominante dans une ROI.
    Retourne: (nom_couleur, valeur_bit, ratio_rouge, ratio_vert)
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 90, 60], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 90, 60], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    lower_green = np.array([40, 70, 50], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Petit nettoyage du bruit sur chaque masque binaire.
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    total = max(1, roi_bgr.shape[0] * roi_bgr.shape[1])
    red_ratio = cv2.countNonZero(mask_red) / total
    green_ratio = cv2.countNonZero(mask_green) / total

    if red_ratio >= float(min_ratio) and red_ratio > green_ratio * float(dominance):
        return "rouge", 0, red_ratio, green_ratio

    if green_ratio >= float(min_ratio) and green_ratio > red_ratio * float(dominance):
        return "vert", 1, red_ratio, green_ratio

    return "inconnu", int(unknown_value), red_ratio, green_ratio


def analyze_walls(image_bgr, band_ratio=0.30, min_ratio=0.03,
                  dominance=1.15, unknown_value=-1):
    """
    Analyse une bande horizontale centrale et decoupe en 3 zones.
    Retourne un dictionnaire avec les couleurs et bits [gauche, centre, droite].
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w, _ = image_bgr.shape
    out = image_bgr.copy()

    band_h = max(1, int(h * float(band_ratio)))
    y1 = max(0, (h - band_h) // 2)
    y2 = min(h, y1 + band_h)
    band = image_bgr[y1:y2, :].copy()

    third = max(1, w // 3)
    rois = [
        band[:, 0:third],
        band[:, third:2 * third],
        band[:, 2 * third:w],
    ]

    colors = []
    values = []
    stats = []

    for roi in rois:
        color_name, bit_value, red_ratio, green_ratio = detect_color_hsv(
            roi,
            min_ratio=min_ratio,
            dominance=dominance,
            unknown_value=unknown_value,
        )
        colors.append(color_name)
        values.append(bit_value)
        stats.append((red_ratio, green_ratio))

    # Annotation visuelle utile pour debug local.
    cv2.rectangle(out, (0, y1), (w - 1, y2 - 1), (255, 255, 255), 2)
    cv2.line(out, (third, y1), (third, y2), (255, 255, 255), 2)
    cv2.line(out, (2 * third, y1), (2 * third, y2), (255, 255, 255), 2)

    labels = ["G", "C", "D"]
    for i, name in enumerate(colors):
        x_text = 10 + i * third
        r_ratio, g_ratio = stats[i]
        cv2.putText(out, f"{labels[i]}: {name} -> {values[i]}", (x_text, max(25, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"R={r_ratio:.2f} V={g_ratio:.2f}", (x_text, max(50, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(out, f"bits={values}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    return {
        "annotated": out,
        "colors": colors,
        "value": values,
    }


def check_camera_direction(values, direction=0, unknown_value=-1):
    """
    Verifie le sens de circulation via les couleurs mur gauche/droite.

    Convention:
    - rouge -> 0
    - vert -> 1
    - inconnu -> unknown_value

    direction=0  => gauche rouge, droite verte
    direction=1  => gauche verte, droite rouge
    """
    if values is None or len(values) < 3:
        return False

    left = values[0]
    right = values[2]

    if left == unknown_value or right == unknown_value:
        return False

    expected_left = int(direction)
    expected_right = 1 - expected_left
    return left == expected_left and right == expected_right

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

    # --- Étape 3 : angle Ackermann (forme continue, sans branchement) ---
    v_eps = 1e-3
    denom = np.sign(v_norm + 1e-9) * max(abs(v_norm), v_eps)
    angle_rad = np.arctan(W * omega / denom)
    angle_deg = np.degrees(angle_rad)

    # Saturation mecanique de l'angle
    angle_deg = float(np.clip(angle_deg, -angle_max_deg, angle_max_deg))
    print(f"angle_deg={angle_deg:.1f}  v_cmd_base={v_cmd:.2f}")
    

    return v_cmd, angle_deg


def calculer_commande_auto(tableau_lidar_filtre, L_entraxe, W_empattement, maxangle_degre,
                           dmax=3000.0, v_min=0.4, v_max=1.2, debug=False):
    """
    Calcule la commande autonome a partir du lidar filtre.

    Retourne
    --------
    v_cmd, angle_cmd
    """

    # Angles des points pertinents
    angle_l1    =  60
    angle_l2    =  70
    angle_front =   0
    angle_r1    = -60
    angle_r2    = -70

    # 1) Lecture des 5 points exacts
    d_l1 = lire_point_lidar(tableau_lidar_filtre, angle_l1, fenetre_deg=4, min_points=2)
    d_l2 = lire_point_lidar(tableau_lidar_filtre, angle_l2, fenetre_deg=4, min_points=2)
    # Le front est tres sensible aux retours parasites :
    # fenetre plus large et seuil de validation plus strict pour eviter les bascules 3000 <-> 330 mm.
    d_front = lire_point_lidar(tableau_lidar_filtre, angle_front, fenetre_deg=4, min_points=2)
    d_r1 = lire_point_lidar(tableau_lidar_filtre, angle_r1, fenetre_deg=4, min_points=2)
    d_r2 = lire_point_lidar(tableau_lidar_filtre, angle_r2, fenetre_deg=4, min_points=2)

    # 2) Normalisation
    l1 = normaliser_distance(d_l1,    dmax)
    l2 = normaliser_distance(d_l2,    dmax)
    f  = normaliser_distance(d_front, dmax)
    r1 = normaliser_distance(d_r1,    dmax)
    r2 = normaliser_distance(d_r2,    dmax)

    # 3) Conversion en proximite
    p_l1 = 1.0 - l1
    p_l2 = 1.0 - l2
    p_f  = 1.0 - f
    p_r1 = 1.0 - r1
    p_r2 = 1.0 - r2

    # 4) Vecteur d'entree du reseau  [biais, p_l1, p_l2, p_f, p_r1, p_r2]
    x = np.array([1.0, p_l1, p_l2, p_f, p_r1, p_r2])

    # 5) Reseau virtuel differentiel
    w_g = np.array([ 1.2,  0.8,  0.8, -1.6, -0.6, -0.6])
    w_d = np.array([ 1.2, -0.6, -0.6, -1.6,  0.8,  0.8])

    u_g = np.tanh(np.dot(x, w_g))
    u_d = np.tanh(np.dot(x, w_d))

    # 6) Conversion differentiel -> Ackermann
    v_cmd, angle_cmd = differentiel_vers_ackermann(
        u_g, u_d,
        L=L_entraxe,
        W=W_empattement,
        v_min=v_min,
        v_max=v_max,
        angle_max_deg=maxangle_degre
    )

    # 6bis) Vitesse continue (sans if):
    # - avance issue du reseau
    # - reduction si desequilibre lateral important
    # - reduction si front proche

    v_norm = max(0.0, (u_g + u_d) / 2.0)
    lat_g = 0.5 * (p_l1 + p_l2)
    lat_d = 0.5 * (p_r1 + p_r2)
    gain_equilibre = np.exp(-2.0 * (lat_g - lat_d) ** 2)
    gain_front = max(0.0, f) ** 1.5
    gain_avance = v_norm ** 1.4

    v_cmd = v_min + (v_max - v_min) * gain_avance * gain_equilibre * gain_front
    v_cmd = float(np.clip(v_cmd, v_min, v_max))

    if debug:
        # =========================
        # Debug
        # =========================
        print("--------------------------------------------------")
        print(f"d_l1     = {d_l1:.1f} mm  |  d_l2    = {d_l2:.1f} mm")
        print(f"d_front  = {d_front:.1f} mm")
        print(f"d_r1     = {d_r1:.1f} mm  |  d_r2    = {d_r2:.1f} mm")
        print(f"p_l1={p_l1:.3f}  p_l2={p_l2:.3f}  p_f={p_f:.3f}  p_r1={p_r1:.3f}  p_r2={p_r2:.3f}")
        print(f"u_g      = {u_g:.3f}  |  u_d     = {u_d:.3f}")
        print(f"g_eq={gain_equilibre:.3f}  g_front={gain_front:.3f}  g_v={gain_avance:.3f}")
        print(f"v_cmd    = {v_cmd:.3f} m/s")
        print(f"angle    = {angle_cmd:.3f} deg")

    return v_cmd, angle_cmd


# =========================
# Automate navigation/blocage (etat interne)
# =========================
_NAVIGATION = 0
_BLOCAGE = 1

_BACKWARD = 0
_CAMERA_CHECKING = 1
_TURN_LEFT = 2
_TURN_RIGHT = 3

_fsm = {
    "etat": _NAVIGATION,
    "sous_etat": _BACKWARD,
    "action_counter": 0,
    "counter_camera_confirm": 0,
    "counter_etat_backward": 0,
    "flag_turn_right": False,
    "last_front_mm": None,
    "last_rear_mm": None,
    "front_missing_steps": 0,
    "last_obj_dist_mm": None,
    "last_obj_speed_m_s": 0.0,
    "last_obj_kind": "unknown",
    "last_obj_side": 0,
    "score_left": 0.0,
    "score_right": 0.0,
}


def reset_automate():
    """Reinitialise l'automate (utiliser lors du passage en mode manuel)."""
    _fsm["etat"] = _NAVIGATION
    _fsm["sous_etat"] = _BACKWARD
    _fsm["action_counter"] = 0
    _fsm["counter_camera_confirm"] = 0
    _fsm["counter_etat_backward"] = 0
    _fsm["flag_turn_right"] = False
    _fsm["last_front_mm"] = None
    _fsm["last_rear_mm"] = None
    _fsm["front_missing_steps"] = 0
    _fsm["last_obj_dist_mm"] = None
    _fsm["last_obj_speed_m_s"] = 0.0
    _fsm["last_obj_kind"] = "unknown"
    _fsm["last_obj_side"] = 0
    _fsm["score_left"] = 0.0
    _fsm["score_right"] = 0.0


def get_automate_state():
    """Retourne une copie de l'etat interne pour debug/affichage."""
    return dict(_fsm)


def _distance_front_robuste(tab, centre_deg, demi_fenetre_deg, dmax, min_points):
    """Distance robuste avec degradation progressive en cas de nuage pauvre."""
    d = distance_secteur_lidar(
        tab,
        centre_deg=centre_deg,
        demi_fenetre_deg=int(demi_fenetre_deg),
        dmax=float(dmax),
        min_points=int(min_points),
        quantile=20,
    )
    if d is not None:
        return float(d)

    d = distance_secteur_lidar(
        tab,
        centre_deg=centre_deg,
        demi_fenetre_deg=max(3, int(demi_fenetre_deg) // 2),
        dmax=float(dmax),
        min_points=max(1, int(min_points) // 3),
        quantile=35,
    )
    if d is not None:
        return float(d)

    d_point = lire_point_lidar(
        tab,
        centre_deg,
        valeur_defaut=-1.0,
        fenetre_deg=max(2, int(demi_fenetre_deg) // 2),
        min_points=1,
    )
    if d_point > 0:
        return float(d_point)
    return None


def _extract_front_clusters(tableau_lidar_filtre, dmax, angle_min=-70, angle_max=70, max_gap_mm=220.0):
    """Extrait des clusters frontaux de points LiDAR valides."""
    clusters = []
    current = []

    for a in range(int(angle_min), int(angle_max) + 1):
        d = lire_point_lidar(
            tableau_lidar_filtre,
            a,
            valeur_defaut=0.0,
            fenetre_deg=0,
            min_points=1,
        )

        if not (0.0 < float(d) <= float(dmax)):
            if current:
                clusters.append(current)
                current = []
            continue

        ar = np.deg2rad(a)
        x = float(d) * float(np.cos(ar))
        y = float(d) * float(np.sin(ar))
        point = (a, float(d), x, y)

        if not current:
            current = [point]
        else:
            _, _, x_prev, y_prev = current[-1]
            gap = float(np.hypot(x - x_prev, y - y_prev))
            if gap <= float(max_gap_mm):
                current.append(point)
            else:
                clusters.append(current)
                current = [point]

    if current:
        clusters.append(current)

    return clusters


def _describe_cluster(cluster):
    """Calcule des features simples pour classer mur / voiture / inconnu."""
    angles = [p[0] for p in cluster]
    dists = [p[1] for p in cluster]
    xs = [p[2] for p in cluster]
    ys = [p[3] for p in cluster]

    n = len(cluster)
    min_dist = float(min(dists))
    width = float(max(ys) - min(ys))
    depth = float(max(xs) - min(xs))
    angle_span = float(max(angles) - min(angles))
    centroid_y = float(np.mean(ys))

    # Heuristique simple et interpretable.
    kind = "unknown"
    if n >= 10 and angle_span >= 28.0 and width >= 700.0:
        kind = "wall"
    elif n >= 4 and 180.0 <= width <= 850.0 and angle_span <= 30.0:
        kind = "vehicle"

    side = 0
    if centroid_y > 40.0:
        side = 1   # obstacle majoritairement a gauche
    elif centroid_y < -40.0:
        side = -1  # obstacle majoritairement a droite

    return {
        "n": n,
        "min_dist": min_dist,
        "width": width,
        "depth": depth,
        "angle_span": angle_span,
        "centroid_y": centroid_y,
        "kind": kind,
        "side": side,
    }


def _normalise_dist_score(d_mm, dmax):
    if d_mm is None:
        return 0.0
    return float(np.clip(float(d_mm), 0.0, float(dmax)) / max(1.0, float(dmax)))


def _compute_side_scores(tableau_lidar_filtre, dmax, avoid_front_diag_deg, avoid_side_deg,
                         avoid_sector_half_deg, avoid_narrow_mm):
    """Calcule des scores gauche/droite pour choisir la manoeuvre."""
    d_fl = _distance_front_robuste(
        tableau_lidar_filtre,
        centre_deg=avoid_front_diag_deg,
        demi_fenetre_deg=avoid_sector_half_deg,
        dmax=dmax,
        min_points=2,
    )
    d_fr = _distance_front_robuste(
        tableau_lidar_filtre,
        centre_deg=-avoid_front_diag_deg,
        demi_fenetre_deg=avoid_sector_half_deg,
        dmax=dmax,
        min_points=2,
    )
    d_l = _distance_front_robuste(
        tableau_lidar_filtre,
        centre_deg=avoid_side_deg,
        demi_fenetre_deg=avoid_sector_half_deg,
        dmax=dmax,
        min_points=2,
    )
    d_r = _distance_front_robuste(
        tableau_lidar_filtre,
        centre_deg=-avoid_side_deg,
        demi_fenetre_deg=avoid_sector_half_deg,
        dmax=dmax,
        min_points=2,
    )

    left_open = 0.60 * _normalise_dist_score(d_fl, dmax) + 0.40 * _normalise_dist_score(d_l, dmax)
    right_open = 0.60 * _normalise_dist_score(d_fr, dmax) + 0.40 * _normalise_dist_score(d_r, dmax)

    dmin_left = min(x for x in [d_fl, d_l] if x is not None) if (d_fl is not None or d_l is not None) else 0.0
    dmin_right = min(x for x in [d_fr, d_r] if x is not None) if (d_fr is not None or d_r is not None) else 0.0

    penalty_left = max(0.0, (float(avoid_narrow_mm) - float(dmin_left)) / max(1.0, float(avoid_narrow_mm)))
    penalty_right = max(0.0, (float(avoid_narrow_mm) - float(dmin_right)) / max(1.0, float(avoid_narrow_mm)))

    score_left = left_open - 0.45 * penalty_left
    score_right = right_open - 0.45 * penalty_right

    return float(score_left), float(score_right)


def calculer_commande_automate(
    tableau_lidar_filtre,
    L_entraxe,
    W_empattement,
    maxangle_degre,
    d_rear,
    dmax=3000.0,
    v_min=0.4,
    v_max=1.2,
    wall_values=None,
    sec_front_fenetre_deg=15,
    sec_front_min_points=5,
    sec_front_stop_mm=700.0,
    sec_front_ralenti_mm=1500.0,
    seuil_front_blocage_mm=500.0,
    seuil_front_degagement_mm=1500.0,
    seuil_arriere_degagement_mm=300.0,
    lidar_rear_window_deg=15,
    lidar_rear_min_points=4,
    blocage_action_duration_s=1.0,
    boucle_periode_s=0.01,
    vitesse_blocage_m_s=0.5,
    angle_recul_fixe_deg=15.0,
    seuil_blocage_persist_steps=5,
    camera_direction_expected=0,
    camera_unknown_value=-1,
    camera_confirm_steps=3,
    avoid_front_diag_deg=30,
    avoid_side_deg=65,
    avoid_sector_half_deg=12,
    avoid_narrow_mm=450.0,
    obstacle_window_deg=70,
    obstacle_cluster_gap_mm=220.0,
    obstacle_dynamic_speed_m_s=0.12,
    debug=False,
):
    """
    Automate complet: navigation + anti-collision + deblocage + verification camera.

    Retourne toujours:
        v_cmd, angle_cmd
    """
    v_base, angle_base = calculer_commande_auto(
        tableau_lidar_filtre,
        L_entraxe=L_entraxe,
        W_empattement=W_empattement,
        maxangle_degre=maxangle_degre,
        dmax=dmax,
        v_min=v_min,
        v_max=v_max,
        debug=debug,
    )

    d_front = _distance_front_robuste(
        tableau_lidar_filtre,
        centre_deg=0,
        demi_fenetre_deg=sec_front_fenetre_deg,
        dmax=dmax,
        min_points=sec_front_min_points,
    )

    # Si aucune mesure frontale exploitable pendant un court instant,
    # conserve la derniere mesure valide pour eviter les oscillations.
    if d_front is None:
        _fsm["front_missing_steps"] += 1
        hold_steps = max(1, int(0.30 / max(1e-4, float(boucle_periode_s))))
        if _fsm["front_missing_steps"] <= hold_steps and _fsm["last_front_mm"] is not None:
            d_front = float(_fsm["last_front_mm"])
    else:
        _fsm["front_missing_steps"] = 0

    _fsm["last_front_mm"] = d_front
    _fsm["last_rear_mm"] = d_rear

    seuil_blocage_effectif = max(float(seuil_front_blocage_mm), float(sec_front_stop_mm))
    front_blocked = d_front is not None and d_front <= seuil_blocage_effectif
    front_clear = d_front is not None and d_front >= float(seuil_front_degagement_mm)

    # Securite frontale appliquee seulement en navigation.
    v_safe = float(v_base)
    stop_mm = float(sec_front_stop_mm)
    slow_mm = float(sec_front_ralenti_mm)
    if d_front is None:
        pass
    elif d_front <= stop_mm:
        v_safe = 0.0
    elif d_front < slow_mm:
        ratio = (d_front - stop_mm) / max(1.0, slow_mm - stop_mm)
        v_lim = max(0.0, min(1.0, ratio)) * float(v_max)
        v_safe = min(v_safe, v_lim)
    v_safe = max(0.0, min(float(v_max), v_safe))

    # --- Decision intelligente de cote d'evitement (gauche/droite) ---
    score_left, score_right = _compute_side_scores(
        tableau_lidar_filtre,
        dmax=float(dmax),
        avoid_front_diag_deg=float(avoid_front_diag_deg),
        avoid_side_deg=float(avoid_side_deg),
        avoid_sector_half_deg=int(avoid_sector_half_deg),
        avoid_narrow_mm=float(avoid_narrow_mm),
    )
    _fsm["score_left"] = score_left
    _fsm["score_right"] = score_right

    clusters = _extract_front_clusters(
        tableau_lidar_filtre,
        dmax=float(dmax),
        angle_min=-int(obstacle_window_deg),
        angle_max=int(obstacle_window_deg),
        max_gap_mm=float(obstacle_cluster_gap_mm),
    )

    nearest = None
    if clusters:
        described = [_describe_cluster(c) for c in clusters if len(c) >= 3]
        if described:
            nearest = min(described, key=lambda c: c["min_dist"])

    obj_kind = "unknown"
    obj_side = 0
    obj_speed = 0.0
    if nearest is not None:
        obj_kind = nearest["kind"]
        obj_side = int(nearest["side"])
        prev_dist = _fsm.get("last_obj_dist_mm")
        if prev_dist is not None and float(boucle_periode_s) > 1e-4:
            obj_speed = float(prev_dist - nearest["min_dist"]) / (1000.0 * float(boucle_periode_s))
        _fsm["last_obj_dist_mm"] = float(nearest["min_dist"])
    else:
        _fsm["last_obj_dist_mm"] = None

    _fsm["last_obj_kind"] = obj_kind
    _fsm["last_obj_side"] = obj_side
    _fsm["last_obj_speed_m_s"] = obj_speed

    # Aide camera pour le choix de cote:
    # - majorite rouge  -> tourner a droite
    # - majorite verte  -> tourner a gauche
    # - sinon           -> pas de biais camera
    camera_bias = 0
    if isinstance(wall_values, (list, tuple)) and len(wall_values) >= 3:
        red_count = sum(1 for v in wall_values if v == 0)
        green_count = sum(1 for v in wall_values if v == 1)
        known_count = red_count + green_count
        if known_count >= 2:
            if red_count > green_count:
                camera_bias = 1
            elif green_count > red_count:
                camera_bias = -1

    # Choix intelligent du cote de manoeuvre.
    prefer_right = score_right > score_left

    # Si obstacle dynamique et decale lateralement, on tourne a l'oppose.
    is_dynamic = abs(obj_speed) >= float(obstacle_dynamic_speed_m_s) and obj_kind != "wall"
    if is_dynamic and obj_side != 0:
        prefer_right = obj_side > 0

    # En cas d'ambiguite, alterner progressivement apres plusieurs echecs.
    if abs(score_left - score_right) < 0.05 and _fsm["counter_etat_backward"] > int(seuil_blocage_persist_steps):
        prefer_right = (_fsm["counter_etat_backward"] % 2 == 0)

    # Priorite camera quand la perception couleur est suffisamment nette.
    if camera_bias == 1:
        prefer_right = True
    elif camera_bias == -1:
        prefer_right = False

    _fsm["flag_turn_right"] = bool(prefer_right)

    action_steps = max(
        1,
        int(float(blocage_action_duration_s) / max(1e-4, float(boucle_periode_s))),
    )
    # Le recul est pilote par la config via blocage_action_duration_s.
    backward_steps = action_steps

    # Les vitesses de manoeuvre sont pilotees par la config via vitesse_blocage_m_s.
    v_backward = abs(float(vitesse_blocage_m_s))
    v_turn = abs(float(vitesse_blocage_m_s))

    v_out = 0.0
    angle_out = 0.0

    match _fsm["etat"]:
        case 0:
            if front_blocked:
                _fsm["etat"] = _BLOCAGE
                _fsm["sous_etat"] = _BACKWARD
                _fsm["action_counter"] = 0
                _fsm["counter_camera_confirm"] = 0
                v_out = 0.0
                angle_out = 0.0
            else:
                v_out = v_safe
                angle_out = float(angle_base)

        case 1:
            match _fsm["sous_etat"]:
                case 0:
                    rear_too_close = (
                        d_rear is not None
                        and d_rear <= float(seuil_arriere_degagement_mm)
                    )
                    if _fsm["action_counter"] >= backward_steps or rear_too_close:
                        _fsm["action_counter"] = 0
                        if _fsm["flag_turn_right"]:
                            _fsm["sous_etat"] = _TURN_RIGHT
                        else:
                            _fsm["sous_etat"] = _TURN_LEFT
                        v_out = 0.0
                        angle_out = 0.0
                    else:
                        _fsm["action_counter"] += 1
                        v_out = -v_backward
                        angle_out = 0.0

                case 2:
                    if _fsm["action_counter"] >= action_steps:
                        _fsm["action_counter"] = 0
                        if front_clear:
                            _fsm["counter_camera_confirm"] = 0
                            _fsm["sous_etat"] = _CAMERA_CHECKING
                        else:
                            _fsm["sous_etat"] = _BACKWARD
                            _fsm["counter_etat_backward"] += 1
                            # Si la strategie gauche n'a pas debloque, on forcera droite au cycle suivant.
                            _fsm["flag_turn_right"] = True
                        v_out = 0.0
                        angle_out = 0.0
                    else:
                        _fsm["action_counter"] += 1
                        v_out = v_turn
                        angle_out = float(angle_recul_fixe_deg)

                case 1:
                    camera_ok = check_camera_direction(
                        wall_values,
                        direction=int(camera_direction_expected),
                        unknown_value=int(camera_unknown_value),
                    )

                    if camera_ok:
                        _fsm["counter_camera_confirm"] += 1
                    else:
                        _fsm["counter_camera_confirm"] = 0

                    _fsm["action_counter"] += 1
                    if _fsm["counter_camera_confirm"] >= int(camera_confirm_steps):
                        _fsm["etat"] = _NAVIGATION
                        _fsm["sous_etat"] = _BACKWARD
                        _fsm["action_counter"] = 0
                        _fsm["flag_turn_right"] = False
                        _fsm["counter_etat_backward"] = 0
                        _fsm["counter_camera_confirm"] = 0
                    elif _fsm["action_counter"] >= int(camera_confirm_steps):
                        _fsm["etat"] = _BLOCAGE
                        _fsm["sous_etat"] = _BACKWARD
                        _fsm["action_counter"] = 0
                        _fsm["flag_turn_right"] = not _fsm["flag_turn_right"]

                    v_out = 0.0
                    angle_out = 0.0

                case 3:
                    if _fsm["action_counter"] >= action_steps:
                        _fsm["action_counter"] = 0
                        if front_clear:
                            _fsm["counter_camera_confirm"] = 0
                            _fsm["sous_etat"] = _CAMERA_CHECKING
                        else:
                            _fsm["sous_etat"] = _BACKWARD
                            _fsm["counter_etat_backward"] += 1
                            # Alterne droite/gauche en cas d'echec pour sortir des cycles repetitifs.
                            _fsm["flag_turn_right"] = False
                        v_out = 0.0
                        angle_out = 0.0
                    else:
                        _fsm["action_counter"] += 1
                        v_out = v_turn
                        angle_out = -float(angle_recul_fixe_deg)

                case _:
                    _fsm["sous_etat"] = _BACKWARD
                    v_out = 0.0
                    angle_out = 0.0

        case _:
            reset_automate()
            v_out = 0.0
            angle_out = 0.0

    return float(v_out), float(angle_out)

