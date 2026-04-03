# config.py — Paramètres de configuration centralisés CoVAPSy
# Valeurs calibrées depuis conduite_autonome_basique.py (validé sur la voiture réelle)
# Pour recalibrer : utiliser test_pwm_propulsion.py et test_pwm_direction.py

# ============================================================
# LIDAR
# ============================================================
LIDAR_PORT     = "/dev/ttyUSB0"
LIDAR_BAUDRATE = 256000
LIDAR_SCAN_TYPE = "express"   # "express" ou "normal"
# Si True, ignore le secteur susceptible de voir l'interieur de la voiture.
# En cas de nuage tres pauvre, laisser False.
LIDAR_IGNORE_INTERIOR_SECTOR = False
LIDAR_INTERIOR_MIN_DEG = 90
LIDAR_INTERIOR_MAX_DEG = 270

# ============================================================
# PROPULSION — HardwarePWM channel 0, 50 Hz
# ============================================================
DIRECTION_PROP      = 1     # 1 = variateur normal, -1 = variateur inversé
PWM_STOP_PROP       = 7.36  # duty cycle correspondant à l'arrêt (1.5 ms)
POINT_MORT_PROP     = 0.37  # seuil minimal en dessous duquel la voiture ne bouge pas
DELTA_PWM_MAX_PROP  = 1.0   # plage PWM entre l'arrêt et la vitesse maximale
VITESSE_MAX_M_S_HARD = 8.0  # vitesse physique maximale de la voiture (m/s)
VITESSE_MAX_M_S_SOFT = 2.0  # vitesse maximale souhaitee pour la fonction vitesse_m_s

# ============================================================
# DIRECTION — HardwarePWM channel 1, 50 Hz
# ============================================================
DIRECTION_DIR    = 1   # 1 = angle_pwm_min à droite, +1 = angle_pwm_min à gauche
ANGLE_PWM_MIN    = 4.5  # butée physique droite (duty cycle)
ANGLE_PWM_MAX    = 8  # butée physique gauche (duty cycle)
ANGLE_PWM_CENTRE = 6.25  # centre (roues droites)
ANGLE_DEGRE_MAX  = 18   # angle max en degrés (vers la gauche)

# ============================================================
# CONDUITE AUTONOME
# ============================================================
L_ENTRAXE_M            = 0.180  # voie du modele utilise par la conversion Ackermann
W_EMPATTEMENT_M        = 0.250  # empattement du modele utilise par la conversion Ackermann
LIDAR_DMAX_MM          = 3000.0 # distance max de normalisation lidar
VITESSE_AUTO_MIN_M_S   = 0.1    # borne basse de la vitesse issue du reseau
VITESSE_AUTO_MAX_M_S   = 0.6   # borne haute en conduite autonome reelle (augmenter progressivement)

BOUCLE_PERIODE_S       = 0.01   # période de la boucle de contrôle (10 ms)

# ============================================================
# SECURITE ANTI-COLLISION (front)
# ============================================================
SECURITE_FRONT_STOP_MM       = 700.0   # stop immediat si obstacle frontal proche
SECURITE_FRONT_RALENTI_MM    = 1500.0  # reduction progressive de vitesse sous ce seuil
SECURITE_FRONT_FENETRE_DEG   = 15      # fenetre angulaire frontale pour la decision de securite
SECURITE_FRONT_MIN_POINTS    = 5       # nombre mini de points valides pour juger le front fiable
SECURITE_VITESSE_INCERTAINE  = 0.05    # vitesse max si front non fiable (peu de points)

# ============================================================
# MACHINE A ETATS NAVIGATION / BLOCAGE
# ============================================================
SEUIL_FRONT_BLOCAGE_MM       = 500.0   # passage en mode blocage si obstacle proche
SEUIL_FRONT_DEGAGEMENT_MM    = 1500.0  # seuil de front libre pour sortir des manoeuvres
SEUIL_ARRIERE_DEGAGEMENT_MM  = 300.0   # stop recul si obstacle arriere proche
ANGLE_RECUL_FIXE_DEG         = 15.0    # angle fixe pour manoeuvres gauche/droite
VITESSE_BLOCAGE_M_S          = 0.5     # vitesse de manoeuvre en mode blocage
BLOCAGE_ACTION_DURATION_S    = 1.0     # duree max d'une action BACKWARD/TURN

# ============================================================
# FILTRAGE DES COMMANDES (stabilisation)
# ============================================================
FILTRE_ALPHA_VITESSE         = 0.35    # 0=stable mais lent, 1=brut
FILTRE_ALPHA_ANGLE           = 0.20    # 0=stable mais lent, 1=brut

# ============================================================
# DEBUG (prints console de diagnostic)
# ============================================================
DEBUG_ACTIONNEURS      = True   # affiche v_cmd/angle_cmd et PWM appliques
DEBUG_LIDAR_RAW        = True   # affiche un resume brut du scan lidar
DEBUG_PRINT_PERIOD_S   = 0.5    # periode mini entre 2 prints debug

# ============================================================
# SÉQUENCE DE RECUL
# ============================================================
VITESSE_RECUL_M_S  = -0.5   # vitesse lors du recul — MODE TEST
DUREE_RECUL_S      = 0.4    # durée du recul
