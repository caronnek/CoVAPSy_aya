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

    return v_cmd, angle_deg


def calculer_commande_auto(tableau_lidar_filtre, L_entraxe, W_empattement, maxangle_degre,
                           dmax=3000.0, v_min=0.4, v_max=1.2, debug=False, retour_detail=False):
    """
    Calcule la commande autonome a partir du lidar filtre.

    Retourne
    --------
    v_cmd, angle_cmd
    """

    # Angles des points pertinents
    angle_l1    =  60
    angle_l2    =  70
    angle_lf1   =   5
    angle_front =   0
    angle_rf1   =  -5
    angle_r1    = -60
    angle_r2    = -70

    # 1) Lecture des points lidar
    d_l1    = lire_point_lidar(tableau_lidar_filtre, angle_l1, fenetre_deg=3, min_points=2)
    d_l2    = lire_point_lidar(tableau_lidar_filtre, angle_l2, fenetre_deg=3, min_points=2)
    d_lf1   = lire_point_lidar(tableau_lidar_filtre, angle_lf1, fenetre_deg=3, min_points=2)
    # Le front est tres sensible aux retours parasites :
    # fenetre plus large et seuil de validation plus strict pour eviter les bascules 3000 <-> 330 mm.
    d_front = lire_point_lidar(tableau_lidar_filtre, angle_front, fenetre_deg=10, min_points=6)
    d_rf1   = lire_point_lidar(tableau_lidar_filtre, angle_rf1, fenetre_deg=3, min_points=2)
    d_r1    = lire_point_lidar(tableau_lidar_filtre, angle_r1, fenetre_deg=3, min_points=2)
    d_r2    = lire_point_lidar(tableau_lidar_filtre, angle_r2, fenetre_deg=3, min_points=2)

    # 2) Normalisation
    l1 = normaliser_distance(d_l1,    dmax)
    l2 = normaliser_distance(d_l2,    dmax)
    lf1 = normaliser_distance(d_lf1,  dmax)
    f  = normaliser_distance(d_front, dmax)
    rf1 = normaliser_distance(d_rf1,  dmax)
    r1 = normaliser_distance(d_r1,    dmax)
    r2 = normaliser_distance(d_r2,    dmax)

    # 3) Conversion en proximite
    p_l1 = 1.0 - l1
    p_l2 = 1.0 - l2
    p_lf1 = 1.0 - lf1
    p_f  = 1.0 - f
    p_rf1 = 1.0 - rf1
    p_r1 = 1.0 - r1
    p_r2 = 1.0 - r2

    # 4) Vecteur d'entree du reseau  [biais, p_l1, p_l2, p_lf1, p_f, p_rf1, p_r1, p_r2]
    x = np.array([1.0, p_l1, p_l2, p_lf1, p_f, p_rf1, p_r1, p_r2])

    # 5) Reseau virtuel differentiel
    w_g = np.array([ 1.2,  0.8,  0.8, -0.2, -1.2,  0.2, -0.6, -0.6])
    w_d = np.array([ 1.2, -0.6, -0.6,  0.2, -1.2, -0.2,  0.8,  0.8])

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
        print(f"d_lf1    = {d_lf1:.1f} mm")
        print(f"d_front  = {d_front:.1f} mm")
        print(f"d_rf1    = {d_rf1:.1f} mm")
        print(f"d_r1     = {d_r1:.1f} mm  |  d_r2    = {d_r2:.1f} mm")
        print(f"p_l1={p_l1:.3f}  p_l2={p_l2:.3f}  p_lf1={p_lf1:.3f}  p_f={p_f:.3f}  p_rf1={p_rf1:.3f}  p_r1={p_r1:.3f}  p_r2={p_r2:.3f}")
        print(f"u_g      = {u_g:.3f}  |  u_d     = {u_d:.3f}")
        print(f"v_cmd    = {v_cmd:.3f} m/s")
        print(f"angle    = {angle_cmd:.3f} deg")

    if not retour_detail:
        return v_cmd, angle_cmd

    details = {
        "d_l1": float(d_l1),
        "d_l2": float(d_l2),
        "d_lf1": float(d_lf1),
        "d_front": float(d_front),
        "d_rf1": float(d_rf1),
        "d_r1": float(d_r1),
        "d_r2": float(d_r2),
        "p_l1": float(p_l1),
        "p_l2": float(p_l2),
        "p_lf1": float(p_lf1),
        "p_f": float(p_f),
        "p_rf1": float(p_rf1),
        "p_r1": float(p_r1),
        "p_r2": float(p_r2),
        "u_g": float(u_g),
        "u_d": float(u_d),
    }
    return v_cmd, angle_cmd, details


class AutomateConduite:
    """Automate de conduite partage entre simulation et voiture reelle.

    Utilisation :
        automate = AutomateConduite(...)
        v_cmd, angle_cmd = automate.calculer_commande(scan_filtre)
    """

    NAVIGATION = "NAVIGATION"
    BLOCAGE = "BLOCAGE"
    BACKWARD = "BACKWARD"
    CAMERA_CHECKING = "CAMERA_CHECKING"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"

    def __init__(
        self,
        L_entraxe,
        W_empattement,
        maxangle_degre,
        dmax=3000.0,
        v_min=0.0,
        v_max=0.6,
        securite_front_fenetre_deg=15,
        securite_front_min_points=5,
        securite_vitesse_incertaine=0.05,
        securite_front_stop_mm=700.0,
        securite_front_ralenti_mm=1500.0,
        filtre_alpha_vitesse=0.35,
        filtre_alpha_angle=0.20,
        seuil_front_blocage_mm=None,
        seuil_front_degagement_mm=1500.0,
        seuil_arriere_degagement_mm=300.0,
        angle_recul_fixe_deg=15.0,
        vitesse_blocage_m_s=0.5,
        blocage_action_duration_s=1.0,
        boucle_periode_s=0.01,
        camera_valide_fn=None,
    ):
        self.L_entraxe = float(L_entraxe)
        self.W_empattement = float(W_empattement)
        self.maxangle_degre = float(maxangle_degre)
        self.dmax = float(dmax)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.securite_front_fenetre_deg = int(securite_front_fenetre_deg)
        self.securite_front_min_points = int(securite_front_min_points)
        self.securite_vitesse_incertaine = float(securite_vitesse_incertaine)
        self.securite_front_stop_mm = float(securite_front_stop_mm)
        self.securite_front_ralenti_mm = float(securite_front_ralenti_mm)

        self.filtre_alpha_vitesse = float(filtre_alpha_vitesse)
        self.filtre_alpha_angle = float(filtre_alpha_angle)

        if seuil_front_blocage_mm is None:
            self.seuil_front_blocage_mm = float(self.securite_front_stop_mm)
        else:
            self.seuil_front_blocage_mm = float(seuil_front_blocage_mm)

        self.seuil_front_degagement_mm = float(seuil_front_degagement_mm)
        self.seuil_arriere_degagement_mm = float(seuil_arriere_degagement_mm)
        self.angle_recul_fixe_deg = float(angle_recul_fixe_deg)
        self.vitesse_blocage_m_s = abs(float(vitesse_blocage_m_s))

        self.action_steps = max(
            1,
            int(float(blocage_action_duration_s) / max(1e-3, float(boucle_periode_s))),
        )

        if camera_valide_fn is None:
            self.camera_valide_fn = lambda: True
        else:
            self.camera_valide_fn = camera_valide_fn

        self._compat_signature_warned = False
        self.last_debug = {}
        self.reset()

    def reset(self):
        """Remet l'automate a son etat initial."""
        self.etat = self.NAVIGATION
        self.sous_etat = self.BACKWARD
        self.flag_turn_right = False
        self.action_counter = 0
        self.v_cmd_filtre = 0.0
        self.angle_cmd_filtre = 0.0
        self.last_debug = {}

    def _distance_front_securite(self, tableau_lidar_mm):
        valeurs = []
        for a in range(-self.securite_front_fenetre_deg, self.securite_front_fenetre_deg + 1):
            idx = a % 360
            d = tableau_lidar_mm[idx]
            if 0 < d <= self.dmax:
                valeurs.append(float(d))

        if len(valeurs) < self.securite_front_min_points:
            return None

        return float(np.percentile(valeurs, 20))

    def _distance_arriere_lidar(self, tableau_lidar_mm):
        return float(lire_point_lidar(tableau_lidar_mm, 180, fenetre_deg=12, min_points=4))

    def _details_depuis_scan(self, tableau_lidar_filtre):
        d_l1 = float(lire_point_lidar(tableau_lidar_filtre, 60, fenetre_deg=3, min_points=2))
        d_l2 = float(lire_point_lidar(tableau_lidar_filtre, 70, fenetre_deg=3, min_points=2))
        d_lf1 = float(lire_point_lidar(tableau_lidar_filtre, 5, fenetre_deg=3, min_points=2))
        d_front = float(lire_point_lidar(tableau_lidar_filtre, 0, fenetre_deg=10, min_points=6))
        d_rf1 = float(lire_point_lidar(tableau_lidar_filtre, -5, fenetre_deg=3, min_points=2))
        d_r1 = float(lire_point_lidar(tableau_lidar_filtre, -60, fenetre_deg=3, min_points=2))
        d_r2 = float(lire_point_lidar(tableau_lidar_filtre, -70, fenetre_deg=3, min_points=2))

        def prox(d):
            return 1.0 - max(0.0, min(float(d), self.dmax)) / self.dmax

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

    def calculer_commande(self, tableau_lidar_filtre):
        """Calcule la commande finale [v_cmd, angle_cmd] a partir d'un scan filtre."""
        try:
            v_raw, angle_raw, details = calculer_commande_auto(
                tableau_lidar_filtre,
                L_entraxe=self.L_entraxe,
                W_empattement=self.W_empattement,
                maxangle_degre=self.maxangle_degre,
                dmax=self.dmax,
                v_min=self.v_min,
                v_max=self.v_max,
                debug=False,
                retour_detail=True,
            )
        except TypeError:
            v_raw, angle_raw = calculer_commande_auto(
                tableau_lidar_filtre,
                L_entraxe=self.L_entraxe,
                W_empattement=self.W_empattement,
                maxangle_degre=self.maxangle_degre,
                dmax=self.dmax,
                v_min=self.v_min,
                v_max=self.v_max,
                debug=False,
            )
            details = self._details_depuis_scan(tableau_lidar_filtre)
            self._compat_signature_warned = True

        d_front = float(details["d_front"])
        d_rear = self._distance_arriere_lidar(tableau_lidar_filtre)

        cmd_v_out = 0.0
        cmd_angle_out = 0.0
        raison_secu = "n/a"
        d_front_sec = None
        v_cible = 0.0

        if self.etat == self.NAVIGATION:
            v_raw = max(self.v_min, min(self.v_max, float(v_raw)))

            d_front_sec = self._distance_front_securite(tableau_lidar_filtre)
            v_cible = float(v_raw)
            raison_secu = "normal"

            if d_front_sec is None:
                v_cible = min(v_cible, self.securite_vitesse_incertaine)
                raison_secu = "front_incertain"
            else:
                if d_front_sec <= self.securite_front_stop_mm:
                    v_cible = 0.0
                    raison_secu = "stop_front"
                elif d_front_sec < self.securite_front_ralenti_mm:
                    ratio = (d_front_sec - self.securite_front_stop_mm) / max(
                        1.0,
                        self.securite_front_ralenti_mm - self.securite_front_stop_mm,
                    )
                    v_lim = max(0.0, min(1.0, ratio)) * self.v_max
                    v_cible = min(v_cible, v_lim)
                    raison_secu = "ralenti_front"

            v_cible = max(self.v_min, min(self.v_max, v_cible))

            self.v_cmd_filtre = (1.0 - self.filtre_alpha_vitesse) * self.v_cmd_filtre + self.filtre_alpha_vitesse * v_cible
            self.angle_cmd_filtre = (1.0 - self.filtre_alpha_angle) * self.angle_cmd_filtre + self.filtre_alpha_angle * float(angle_raw)

            if d_front_sec is not None and d_front_sec <= self.securite_front_stop_mm:
                self.v_cmd_filtre = 0.0
                self.angle_cmd_filtre = (1.0 - self.filtre_alpha_angle) * self.angle_cmd_filtre

            self.v_cmd_filtre = max(self.v_min, min(self.v_max, self.v_cmd_filtre))

            front_blocage = d_front_sec if d_front_sec is not None else d_front
            if front_blocage <= self.seuil_front_blocage_mm:
                self.etat = self.BLOCAGE
                self.sous_etat = self.BACKWARD
                self.flag_turn_right = False
                self.action_counter = 0
                cmd_v_out = 0.0
                cmd_angle_out = 0.0
            else:
                cmd_v_out = float(self.v_cmd_filtre)
                cmd_angle_out = float(self.angle_cmd_filtre)

        else:
            raison_secu = "etat_blocage"
            d_front_sec = d_front

            if self.sous_etat == self.BACKWARD:
                if self.action_counter >= self.action_steps or d_rear <= self.seuil_arriere_degagement_mm:
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    self.action_counter = 0
                    if self.flag_turn_right:
                        self.sous_etat = self.TURN_RIGHT
                    else:
                        self.sous_etat = self.TURN_LEFT
                else:
                    cmd_angle_out = 0.0
                    cmd_v_out = -self.vitesse_blocage_m_s
                    self.action_counter += 1

            elif self.sous_etat == self.TURN_LEFT:
                if self.action_counter >= self.action_steps:
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    self.action_counter = 0
                    if d_front > self.seuil_front_degagement_mm:
                        self.sous_etat = self.CAMERA_CHECKING
                    else:
                        self.sous_etat = self.BACKWARD
                else:
                    cmd_angle_out = self.angle_recul_fixe_deg
                    cmd_v_out = -self.vitesse_blocage_m_s
                    self.action_counter += 1

            elif self.sous_etat == self.TURN_RIGHT:
                if self.action_counter >= self.action_steps:
                    cmd_v_out = 0.0
                    cmd_angle_out = 0.0
                    self.action_counter = 0
                    if d_front > self.seuil_front_degagement_mm:
                        self.sous_etat = self.CAMERA_CHECKING
                    else:
                        self.sous_etat = self.BACKWARD
                else:
                    cmd_angle_out = -self.angle_recul_fixe_deg
                    cmd_v_out = -self.vitesse_blocage_m_s
                    self.action_counter += 1

            elif self.sous_etat == self.CAMERA_CHECKING:
                cmd_v_out = 0.0
                cmd_angle_out = 0.0
                self.action_counter = 0
                if self.camera_valide_fn():
                    self.etat = self.NAVIGATION
                    self.sous_etat = self.BACKWARD
                    self.flag_turn_right = False
                    self.v_cmd_filtre = 0.0
                    self.angle_cmd_filtre = 0.0
                else:
                    self.etat = self.BLOCAGE
                    self.sous_etat = self.BACKWARD
                    self.flag_turn_right = True

            else:
                cmd_v_out = 0.0
                cmd_angle_out = 0.0
                self.sous_etat = self.BACKWARD
                self.action_counter = 0

        self.last_debug = {
            "etat": self.etat,
            "sous_etat": self.sous_etat if self.etat == self.BLOCAGE else "-",
            "details": details,
            "d_front": float(d_front),
            "d_rear": float(d_rear),
            "v_raw": float(v_raw),
            "angle_raw": float(angle_raw),
            "d_front_sec": None if d_front_sec is None else float(d_front_sec),
            "v_cible": float(v_cible),
            "raison_secu": raison_secu,
            "v_out": float(cmd_v_out),
            "angle_out": float(cmd_angle_out),
            "compat_signature_warned": bool(self._compat_signature_warned),
        }

        return float(cmd_v_out), float(cmd_angle_out)
