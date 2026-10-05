"""
Allocations optimales, et ce qu'elles valent.

Le mot « optimal » demande une mise en garde avant la premiere ligne de code.
Une allocation optimale l'est sur la periode qui a servi a la calculer, donc
en connaissant deja son resultat. Transposee a la suite, elle perd une part
variable de son avantage, et parfois la totalite.

Cette fragilite n'est pas la meme partout, et c'est le point utile :

— Les methodes qui minimisent le risque ne dependent que de la covariance.
  La covariance est instable, mais beaucoup moins que les rendements : une
  matrice estimee sur trois ans garde une partie de sa valeur l'annee
  suivante. Ces allocations tiennent raisonnablement hors echantillon.

— Les methodes qui maximisent le rendement, Sharpe compris, dependent des
  rendements attendus. Personne ne sait les estimer. En pratique on utilise
  les rendements passes comme approximation, et l'optimiseur se precipite
  alors sur ce qui a le mieux marche, c'est-a-dire sur ce qui est deja cher.
  C'est la raison pour laquelle le portefeuille a Sharpe maximal deçoit si
  regulierement a l'usage.

— L'equipondere ne depend de rien. Il sert de temoin, et il est plus
  difficile a battre que son apparente naivete ne le laisse croire.

Trois dispositions limitent les degats sans pretendre les supprimer : un
retrecissement de la matrice de covariance vers une cible plus stable, un
plafond par ligne, et l'affichage systematique de l'equipondere a cote des
autres. Le reste est affaire de jugement, et il reste au lecteur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

JOURS_BOURSE = 252.0

# Retrecissement de la covariance vers une cible a correlation constante.
# Une matrice estimee sur peu d'observations comporte des correlations
# extremes qui sont du bruit, et l'optimiseur les exploite avec enthousiasme.
# Le melange avec une cible lisse les attenue. L'intensite est une convention
# raisonnable, pas un optimum calcule.
RETRECISSEMENT = 0.3

PLAFOND_LIGNE = 15.0        # % — poids maximal d'une ligne
# Poids minimal d'une ligne retenue. Sans plancher, l'optimiseur attribue
# regulierement zero a la moitie des valeurs selectionnees : le portefeuille
# annonce vingt lignes et en detient treize. Un plancher rend le compte exact
# et evite les positions trop petites pour justifier leurs frais.
PLANCHER_LIGNE = 1.0        # % — poids minimal d'une ligne retenue
PLANCHER_OBSERVATIONS = 120  # seances minimales pour estimer quoi que ce soit


# ==========================================================================
# Estimation
# ==========================================================================

def covariance_retrecie(rendements: pd.DataFrame,
                        intensite: float = RETRECISSEMENT,
                        freq: float = JOURS_BOURSE) -> pd.DataFrame:
    """
    Covariance annualisee, melangee a une cible a correlation constante.

    La cible conserve les volatilites individuelles mais remplace toutes les
    correlations par leur moyenne. Elle est grossiere et c'est voulu : son
    role est d'etre stable, pas juste.
    """
    echantillon = rendements.dropna().cov(ddof=1) * freq
    if echantillon.empty or intensite <= 0:
        return echantillon

    ecarts = np.sqrt(np.diag(echantillon.to_numpy()))
    correlations = echantillon.to_numpy() / np.outer(ecarts, ecarts)
    hors_diagonale = correlations[~np.eye(len(ecarts), dtype=bool)]
    moyenne = float(np.mean(hors_diagonale)) if len(hors_diagonale) else 0.0

    cible = np.full_like(correlations, moyenne)
    np.fill_diagonal(cible, 1.0)
    cible = cible * np.outer(ecarts, ecarts)

    melange = intensite * cible + (1 - intensite) * echantillon.to_numpy()
    return pd.DataFrame(melange, index=echantillon.index,
                        columns=echantillon.columns)


def rendements_annualises(rendements: pd.DataFrame,
                          freq: float = JOURS_BOURSE) -> pd.Series:
    """
    Rendement annualise de chaque ligne, par composition.

    Sert d'approximation des rendements attendus, faute de mieux. C'est la
    faiblesse centrale de toute optimisation par le rendement, et elle est
    rappelee a l'ecran plutot que dissimulee ici.
    """
    r = rendements.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    total = (1 + r).prod()
    return total ** (freq / len(r)) - 1


# ==========================================================================
# Allocations
# ==========================================================================

def _contraintes(n: int, plafond: float, plancher: float = 0.0):
    bornes = [(plancher / 100, plafond / 100) for _ in range(n)]
    somme = {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}
    return bornes, [somme]


def _resoudre(objectif, n: int, plafond: float, contraintes_sup=None,
              plancher: float = 0.0):
    bornes, contraintes = _contraintes(n, plafond, plancher)
    if contraintes_sup:
        contraintes = contraintes + list(contraintes_sup)
    depart = np.full(n, 1.0 / n)
    sortie = minimize(objectif, depart, method="SLSQP",
                      bounds=bornes, constraints=contraintes,
                      options={"maxiter": 400, "ftol": 1e-10})
    poids = np.clip(sortie.x, 0.0, None)
    total = poids.sum()
    return poids / total if total > 0 else depart


def variance_minimale(cov: pd.DataFrame,
                      plafond: float = PLAFOND_LIGNE,
                      plancher: float = 0.0) -> pd.Series:
    """Le portefeuille le moins volatil. Ne depend que de la covariance."""
    matrice = cov.to_numpy()
    poids = _resoudre(lambda w: float(w @ matrice @ w), len(cov), plafond, plancher=plancher)
    return pd.Series(poids, index=cov.index)


def parite_risque(cov: pd.DataFrame,
                  plafond: float = PLAFOND_LIGNE,
                  plancher: float = 0.0) -> pd.Series:
    """
    Chaque ligne contribue autant au risque total.

    Intermediaire entre l'equipondere et la variance minimale : ne depend pas
    non plus des rendements attendus, mais tient compte des correlations.
    """
    matrice = cov.to_numpy()
    n = len(cov)

    def ecart_aux_contributions(w):
        variance = float(w @ matrice @ w)
        if variance <= 0:
            return 1e6
        contributions = w * (matrice @ w) / np.sqrt(variance)
        return float(np.sum((contributions - contributions.mean()) ** 2))

    return pd.Series(_resoudre(ecart_aux_contributions, n, plafond, plancher=plancher),
                     index=cov.index)


def sharpe_maximal(cov: pd.DataFrame, esperances: pd.Series,
                   plafond: float = PLAFOND_LIGNE,
                   sans_risque: float = 0.0,
                   plancher: float = 0.0) -> pd.Series:
    """
    Le meilleur rapport rendement sur volatilite — sur le passe.

    Depend des rendements attendus, donc des rendements passes, donc de ce
    qui a deja monte. A lire comme un constat retrospectif, pas comme une
    recommandation.
    """
    matrice = cov.to_numpy()
    mu = esperances.reindex(cov.index).fillna(0.0).to_numpy()

    def negatif_du_sharpe(w):
        volatilite = np.sqrt(float(w @ matrice @ w))
        if volatilite <= 0:
            return 1e6
        return -(float(w @ mu) - sans_risque) / volatilite

    return pd.Series(_resoudre(negatif_du_sharpe, len(cov), plafond, plancher=plancher),
                     index=cov.index)


def equipondere(actifs) -> pd.Series:
    """Le temoin. Ne depend d'aucune estimation."""
    actifs = list(actifs)
    return pd.Series(1.0 / len(actifs), index=actifs) if actifs else pd.Series(dtype=float)


METHODES = {
    "Variance minimale": "variance_minimale",
    "Parité de risque": "parite_risque",
    "Sharpe maximal": "sharpe_maximal",
    "Équipondéré": "equipondere",
}

# Methodes qui n'utilisent aucune estimation de rendement attendu, donc les
# plus robustes hors echantillon.
SANS_ESPERANCES = {"Variance minimale", "Parité de risque", "Équipondéré"}


def allouer(methode: str, cov: pd.DataFrame, esperances: pd.Series,
            plafond: float = PLAFOND_LIGNE,
            plancher: float = 0.0) -> pd.Series:
    if methode == "Variance minimale":
        return variance_minimale(cov, plafond, plancher)
    if methode == "Parité de risque":
        return parite_risque(cov, plafond, plancher)
    if methode == "Sharpe maximal":
        return sharpe_maximal(cov, esperances, plafond, 0.0, plancher)
    return equipondere(cov.index)


# ==========================================================================
# Selection d'un nombre fixe de lignes
# ==========================================================================

def selectionner(methode: str, cov: pd.DataFrame, esperances: pd.Series,
                 nombre: int = 20,
                 plafond: float = PLAFOND_LIGNE,
                 plancher: float = PLANCHER_LIGNE) -> pd.Series:
    """
    Reduit l'univers au nombre de lignes voulu, puis alloue.

    L'elimination est progressive : on resout, on retire la ligne la moins
    pondere, on recommence. Retirer d'un coup les lignes faibles donnerait un
    resultat different, car les poids se redistribuent a chaque retrait.

    Ce n'est pas une resolution exacte du probleme sous contrainte de
    cardinalite, qui est combinatoire et hors de portee ici. C'est une
    heuristique courante, et elle doit etre lue comme telle.
    """
    retenus = list(cov.index)
    if nombre >= len(retenus):
        return allouer(methode, cov, esperances, plafond)

    # Le plafond doit laisser la somme atteindre 1 sur le nombre final, et le
    # plancher ne doit pas la depasser.
    plafond = max(plafond, 100.0 / nombre + 1e-9)
    plancher = min(plancher, 100.0 / nombre)

    # L'elimination se fait sans plancher : il fausserait le classement en
    # donnant le meme poids minimal a toutes les lignes faibles.
    while len(retenus) > nombre:
        poids = allouer(methode, cov.loc[retenus, retenus],
                        esperances.reindex(retenus), plafond)
        retenus.remove(poids.idxmin())

    return allouer(methode, cov.loc[retenus, retenus],
                   esperances.reindex(retenus), plafond, plancher)


# ==========================================================================
# Mesures
# ==========================================================================

def mesures(poids: pd.Series, cov: pd.DataFrame,
            esperances: pd.Series) -> dict:
    """Volatilite, rendement, Sharpe, concentration."""
    if poids is None or poids.empty:
        return {}
    w = poids.reindex(cov.index).fillna(0.0)
    variance = float(w @ cov.to_numpy() @ w)
    volatilite = float(np.sqrt(variance)) if variance > 0 else np.nan
    rendement = float(w @ esperances.reindex(cov.index).fillna(0.0))
    carres = float((w ** 2).sum())
    return {
        "volatilite": volatilite * 100,
        "rendement": rendement * 100,
        "sharpe": rendement / volatilite if volatilite and volatilite > 0 else np.nan,
        "lignes": int((w > 1e-6).sum()),
        "lignes_effectives": 1 / carres if carres > 0 else np.nan,
        "poids_max": float(w.max()) * 100,
    }


def frontiere(cov: pd.DataFrame, esperances: pd.Series, points: int = 25,
              plafond: float = PLAFOND_LIGNE) -> pd.DataFrame:
    """
    La courbe des compromis : volatilite minimale pour chaque rendement visé.

    C'est elle qui montre qu'un « portefeuille optimal » unique n'existe pas.
    Chaque point est optimal pour un niveau de risque accepté, et le choix de
    ce niveau n'est pas un calcul.
    """
    mu = esperances.reindex(cov.index).fillna(0.0).to_numpy()
    matrice = cov.to_numpy()
    n = len(cov)
    if n == 0:
        return pd.DataFrame(columns=["Rendement (%)", "Volatilité (%)"])

    bas = float(mu @ variance_minimale(cov, plafond).to_numpy())
    haut = float(np.sort(mu)[-max(1, int(np.ceil(100 / plafond))):].mean())
    if not np.isfinite(bas) or not np.isfinite(haut) or haut <= bas:
        return pd.DataFrame(columns=["Rendement (%)", "Volatilité (%)"])

    lignes = []
    for cible in np.linspace(bas, haut, points):
        contrainte = {"type": "eq",
                      "fun": (lambda w, c=cible: float(w @ mu) - c)}
        poids = _resoudre(lambda w: float(w @ matrice @ w), n, plafond,
                          [contrainte])
        variance = float(poids @ matrice @ poids)
        lignes.append({"Rendement (%)": float(poids @ mu) * 100,
                       "Volatilité (%)": float(np.sqrt(variance)) * 100})
    return pd.DataFrame(lignes)
