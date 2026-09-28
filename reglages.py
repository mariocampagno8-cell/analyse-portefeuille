"""
Seuils partages par l'application et la veille Telegram.

Ces deux programmes regardent le meme portefeuille et doivent dire la meme
chose. Tant que chacun portait ses propres constantes, ils ont diverge sans
que rien ne le signale : la veille alertait sur une concentration a 15 %
quand l'application la tolerait jusqu'a 25, et une meme ligne pouvait etre
« a surveiller » d'un cote et « normale » de l'autre.

Un seul endroit, donc. Modifier une valeur ici la change partout.
"""

# --- Mouvements de cours
# Seuil d'alerte calibre en ecarts-types du titre plutot qu'en pourcentage
# fixe : 5 % sur Coca-Cola est un evenement, sur une valeur quantique c'est
# une seance ordinaire.
SIGMA_MOUVEMENT = 2.5
PLANCHER_MOUVEMENT = 3.0         # % — jamais d'alerte en deca
PLAFOND_MOUVEMENT = 12.0         # % — toujours une alerte au-dela

# --- Proximite des seuils
PROXIMITE_SEUIL = 3.0            # % — approche d'un stop ou d'un prix d'entree
PROXIMITE_CIBLE = 10.0           # % — approche d'un prix cible de surveillance

# --- Concentration
CONCENTRATION = 25.0             # % du portefeuille sur une seule ligne

# --- Stops suiveurs du tableau de bord
HORIZON_STOP = 20                # seances — un mois de bourse
STOP_SIGMA = 2.0                 # ecarts-types
STOP_MIN = 8.0                   # % — plancher
STOP_MAX = 30.0                  # % — plafond
FENETRE_HAUT = 120               # seances — plus haut de reference
FENETRE_MOYENNE = 50             # seances — moyenne de reference

# --- Dimensionnement
RISQUE_PAR_IDEE = 1.0            # % du portefeuille perdu si le stop tombe
PLAFOND_LIGNE = 20.0             # % du portefeuille sur une ligne

# --- Onglets de la feuille Google
ONGLET_PORTEFEUILLE = "PORTEFEUILLE"
ONGLET_MOUVEMENTS = "MOUVEMENTS"
ONGLET_SURVEILLANCE = "VALEURS EN SURVEILLANCE"
