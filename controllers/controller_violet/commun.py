import numpy as np

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

    # --- Étape 3 : rayon de courbure ---
    omega_seuil = 1e-4            # évite la division par zéro (ligne droite)
    if abs(omega) < omega_seuil:
        # Tout droit : angle nul
        angle_deg = 0.0
    else:
        R = v_norm / omega        # rayon signé (négatif = droite)

        # --- Étape 4 : angle Ackermann (roue centrale de référence) ---
        # δ = arctan(W / R)
        # Négatif car convention Webots : angle+ = droite mécanique
        angle_rad = np.arctan(W / R)
        angle_deg = np.degrees(angle_rad)

    # Saturation
    angle_deg = float(np.clip(angle_deg, -angle_max_deg, angle_max_deg))
    
    ratio= abs(angle_deg/angle_max_deg)
    v_cmd = v_max-(v_max-v_min)*ratio
    print(f"angle_deg={angle_deg:.1f}  ratio={ratio:.2f}  v_cmd={v_cmd:.2f}")
    

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
    d_l1    = lire_point_lidar(tableau_lidar_filtre, angle_l1, fenetre_deg=3, min_points=2)
    d_l2    = lire_point_lidar(tableau_lidar_filtre, angle_l2, fenetre_deg=3, min_points=2)
    # Le front est tres sensible aux retours parasites :
    # fenetre plus large et seuil de validation plus strict pour eviter les bascules 3000 <-> 330 mm.
    d_front = lire_point_lidar(tableau_lidar_filtre, angle_front, fenetre_deg=10, min_points=6)
    d_r1    = lire_point_lidar(tableau_lidar_filtre, angle_r1, fenetre_deg=3, min_points=2)
    d_r2    = lire_point_lidar(tableau_lidar_filtre, angle_r2, fenetre_deg=3, min_points=2)

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
        print(f"v_cmd    = {v_cmd:.3f} m/s")
        print(f"angle    = {angle_cmd:.3f} deg")

    return v_cmd, angle_cmd

