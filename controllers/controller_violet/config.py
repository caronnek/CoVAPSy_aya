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
VITESSE_AUTO_MIN_M_S   = 0.4    # borne basse de la vitesse issue du reseau
VITESSE_AUTO_MAX_M_S   = 1.2   # borne haute en conduite autonome reelle (augmenter progressivement)
AUTO_DEBUG             = True   # active les prints debug de calculer_commande_auto

BOUCLE_PERIODE_S       = 0.01   # période de la boucle de contrôle (10 ms)

# ============================================================
# CAMERA (verification du sens via couleur des murs)
# ============================================================
CAMERA_ACTIVE                = True    # False pour desactiver completement la camera
CAMERA_WIDTH                 = 640
CAMERA_HEIGHT                = 480
CAMERA_FRAMERATE             = 30
CAMERA_SHOW_WINDOW           = False   # True si ecran local, False en SSH/headless
CAMERA_BAND_RATIO            = 0.30    # portion centrale analysee
CAMERA_MIN_RATIO             = 0.03    # proportion minimale de pixels de couleur
CAMERA_DOMINANCE             = 1.15    # ratio min entre couleur gagnante et perdante
CAMERA_UNKNOWN_VALUE         = -1
CAMERA_DIRECTION_EXPECTED    = 0       # 0: gauche rouge/droite verte ; 1: inverse
CAMERA_CONFIRM_STEPS         = 3       # nb de validations consecutives avant retour NAVIGATION

# ============================================================
# SECURITE ANTI-COLLISION (front)
# ============================================================
SECURITE_FRONT_STOP_MM       = 400.0   # stop immediat si obstacle frontal proche
SECURITE_FRONT_RALENTI_MM    = 410.0  # reduction progressive de vitesse sous ce seuil
SECURITE_FRONT_FENETRE_DEG   = 4      # fenetre angulaire frontale pour la decision de securite
SECURITE_FRONT_MIN_POINTS    = 5       # nombre mini de points valides pour juger le front fiable

# ============================================================
# MACHINE A ETATS NAVIGATION / BLOCAGE
# ============================================================
SEUIL_FRONT_BLOCAGE_MM       = SECURITE_FRONT_STOP_MM   # passage en mode blocage si obstacle proche
SEUIL_FRONT_DEGAGEMENT_MM    = SECURITE_FRONT_RALENTI_MM   # seuil de front libre pour sortir des manoeuvres
SEUIL_ARRIERE_DEGAGEMENT_MM  = 100.0   # stop recul si obstacle arriere proche
SEUIL_BLOCAGE_PERSIST_STEPS  = 5       # nb d'echecs avant autoriser TURN_RIGHT
ANGLE_RECUL_FIXE_DEG         = 15.0    # angle fixe pour manoeuvres gauche/droite
VITESSE_BLOCAGE_M_S          = 0.7     # vitesse de manoeuvre en mode blocage
BLOCAGE_ACTION_DURATION_S    = 0.3     # duree max d'une action BACKWARD/TURN
LIDAR_REAR_WINDOW_DEG        = 10      # fenetre de mesure arriere autour de 180 deg
LIDAR_REAR_MIN_POINTS        = 4       # nb mini de points valides pour juger l'arriere fiable

# ============================================================
# STRATEGIE EVITEMENT INTELLIGENT (LiDAR)
# ============================================================
AVOID_FRONT_DIAG_DEG             = 30    # secteur avant-gauche / avant-droite pour score de passage
AVOID_SIDE_DEG                   = 65    # secteur lateral gauche / droite pour score de passage
AVOID_SECTOR_HALF_DEG            = 12    # demi largeur de chaque secteur de score
AVOID_NARROW_MM                  = 300.0 # penalite si passage estime trop etroit

OBSTACLE_WINDOW_DEG              = 70    # fenetre frontale pour clustering obstacle
OBSTACLE_CLUSTER_GAP_MM          = 220.0 # distance max entre 2 points consecutifs d'un meme cluster
OBSTACLE_DYNAMIC_SPEED_M_S       = 0.12  # seuil de vitesse relative pour considerer obstacle dynamique

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

