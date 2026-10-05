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

import secteurs as sec

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

# Contraintes sectorielles. Le plancher sur la technologie americaine est une
# conviction de l'investisseur, pas un resultat de calcul : l'optimiseur la
# respecte sans la discuter. Le plafond sur les autres secteurs est ce qui
# empeche le solde de se concentrer ailleurs — sans lui, « diversifie » ne
# veut rien dire.
PLANCHER_TECH_US = 40.0     # % — exposition minimale a la technologie US
PLAFOND_SECTEUR = 25.0      # % — par secteur autre que la technologie US
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


def contraintes_sectorielles(actifs, plancher_tech: float = PLANCHER_TECH_US,
                             plafond_secteur: float = PLAFOND_SECTEUR) -> list:
    """
    Traduit les regles de secteur en contraintes d'inegalite.

    SLSQP attend des fonctions positives ou nulles a l'optimum. Le plancher
    technologique s'ecrit « somme des poids tech moins le seuil >= 0 », le
    plafond d'un secteur « seuil moins somme des poids du secteur >= 0 ».

    Un jeu de contraintes infaisable ne produit pas d'erreur : SLSQP rend un
    vecteur quelconque, silencieusement faux. Les plafonds sont donc ecartes
    quand ils ne laissent pas de quoi placer 100 % du portefeuille — par
    exemple si l'univers entier tient dans un seul secteur non technologique.
    """
    actifs = list(actifs)
    groupes = sec.par_secteur(actifs)
    index = {t: i for i, t in enumerate(actifs)}
    regles = []

    tech = [index[t] for t in groupes.get(sec.TECH_US, [])]
    if tech and plancher_tech > 0:
        regles.append({"type": "ineq",
                       "fun": (lambda w, i=tech, s=plancher_tech / 100:
                               float(np.sum(w[i])) - s)})

    autres = [nom for nom in groupes if nom != sec.TECH_US]
    # La technologie n'a pas de plafond ici : elle peut absorber le reste.
    capacite = plafond_secteur * len(autres) + (100.0 if tech else 0.0)
    if plafond_secteur >= 100 or capacite < 100.0:
        return regles

    for nom in autres:
        positions = [index[t] for t in groupes[nom]]
        regles.append({"type": "ineq",
                       "fun": (lambda w, i=positions, s=plafond_secteur / 100:
                               s - float(np.sum(w[i])))})
    return regles


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
                      plancher: float = 0.0, regles=None) -> pd.Series:
    """Le portefeuille le moins volatil. Ne depend que de la covariance."""
    matrice = cov.to_numpy()
    poids = _resoudre(lambda w: float(w @ matrice @ w), len(cov), plafond,
                      regles, plancher=plancher)
    return pd.Series(poids, index=cov.index)


def parite_risque(cov: pd.DataFrame,
                  plafond: float = PLAFOND_LIGNE,
                  plancher: float = 0.0, regles=None) -> pd.Series:
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

    return pd.Series(_resoudre(ecart_aux_contributions, n, plafond, regles,
                               plancher=plancher), index=cov.index)


def sharpe_maximal(cov: pd.DataFrame, esperances: pd.Series,
                   plafond: float = PLAFOND_LIGNE,
                   sans_risque: float = 0.0,
                   plancher: float = 0.0, regles=None) -> pd.Series:
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

    return pd.Series(_resoudre(negatif_du_sharpe, len(cov), plafond, regles,
                               plancher=plancher), index=cov.index)


def rendement_maximal(cov: pd.DataFrame, esperances: pd.Series,
                      plafond: float = PLAFOND_LIGNE,
                      plancher: float = 0.0,
                      regles=None) -> pd.Series:
    """
    Le rendement passe le plus eleve que les contraintes autorisent.

    C'est l'allocation la plus fragile de toutes : elle ne regarde que ce qui
    a deja monte et n'oppose aucune resistance a la concentration, sinon le
    plafond par ligne et les regles de secteur. Elle a sa place ici comme
    borne superieure — le maximum atteignable sous ces contraintes — et non
    comme une allocation a suivre.
    """
    mu = esperances.reindex(cov.index).fillna(0.0).to_numpy()
    return pd.Series(
        _resoudre(lambda w: -float(w @ mu), len(cov), plafond, regles,
                  plancher=plancher),
        index=cov.index)


def equipondere(actifs) -> pd.Series:
    """Le temoin. Ne depend d'aucune estimation."""
    actifs = list(actifs)
    return pd.Series(1.0 / len(actifs), index=actifs) if actifs else pd.Series(dtype=float)


METHODES = {
    "Rendement maximal": "rendement_maximal",
    "Sharpe maximal": "sharpe_maximal",
    "Parité de risque": "parite_risque",
    "Variance minimale": "variance_minimale",
    "Équipondéré": "equipondere",
}

# Methodes qui n'utilisent aucune estimation de rendement attendu, donc les
# plus robustes hors echantillon.
SANS_ESPERANCES = {"Variance minimale", "Parité de risque", "Équipondéré"}


def allouer(methode: str, cov: pd.DataFrame, esperances: pd.Series,
            plafond: float = PLAFOND_LIGNE,
            plancher: float = 0.0, regles=None) -> pd.Series:
    """
    L'equipondere ne prend pas de regles : il n'optimise rien, donc il ne
    peut rien respecter. Son exposition sectorielle est celle de l'univers,
    et c'est precisement ce qui en fait un temoin utile.
    """
    if methode == "Rendement maximal":
        return rendement_maximal(cov, esperances, plafond, plancher, regles)
    if methode == "Sharpe maximal":
        return sharpe_maximal(cov, esperances, plafond, 0.0, plancher, regles)
    if methode == "Parité de risque":
        return parite_risque(cov, plafond, plancher, regles)
    if methode == "Variance minimale":
        return variance_minimale(cov, plafond, plancher, regles)
    return equipondere(cov.index)


# ==========================================================================
# Selection d'un nombre fixe de lignes
# ==========================================================================

def selectionner(methode: str, cov: pd.DataFrame, esperances: pd.Series,
                 nombre: int = 20,
                 plafond: float = PLAFOND_LIGNE,
                 plancher: float = PLANCHER_LIGNE,
                 plancher_tech: float = PLANCHER_TECH_US,
                 plafond_secteur: float = PLAFOND_SECTEUR) -> pd.Series:
    """
    Reduit l'univers au nombre de lignes voulu, puis alloue.

    L'elimination est progressive : on resout, on retire la ligne la moins
    pondere, on recommence. Retirer d'un coup les lignes faibles donnerait un
    resultat different, car les poids se redistribuent a chaque retrait.

    Ce n'est pas une resolution exacte du probleme sous contrainte de
    cardinalite, qui est combinatoire et hors de portee ici. C'est une
    heuristique courante, et elle doit etre lue comme telle.
    """
    # L'equipondere n'a aucun critere pour choisir vingt valeurs sur
    # trente-et-une : tous ses poids sont egaux, et le retrait de la « plus
    # faible » revient a tirer au sort. Comme temoin, il porte donc l'univers
    # entier — c'est aussi ce qui en fait une reference honnete, puisqu'il ne
    # beneficie d'aucune selection.
    if methode == "Équipondéré":
        return equipondere(cov.index)

    retenus = list(cov.index)

    def regles(actifs):
        return contraintes_sectorielles(actifs, plancher_tech, plafond_secteur)

    if nombre >= len(retenus):
        return allouer(methode, cov, esperances, plafond, 0.0,
                       regles(retenus))

    # Le plafond doit laisser la somme atteindre 1 sur le nombre final, et le
    # plancher ne doit pas la depasser.
    plafond = max(plafond, 100.0 / nombre + 1e-9)
    plancher = min(plancher, 100.0 / nombre)

    # L'elimination se fait sans plancher : il fausserait le classement en
    # donnant le meme poids minimal a toutes les lignes faibles.
    # L'elimination doit preserver de quoi satisfaire le plancher sectoriel :
    # retirer toutes les valeurs technologiques rendrait le probleme
    # insoluble, et l'optimiseur renverrait silencieusement n'importe quoi.
    while len(retenus) > nombre:
        poids = allouer(methode, cov.loc[retenus, retenus],
                        esperances.reindex(retenus), plafond, 0.0,
                        regles(retenus))
        candidate = poids.idxmin()
        tech_restantes = [t for t in retenus if sec.secteur(t) == sec.TECH_US]
        besoin_tech = int(np.ceil(plancher_tech / plafond))
        if (sec.secteur(candidate) == sec.TECH_US
                and len(tech_restantes) <= besoin_tech):
            # Protegee : on retire la plus faible parmi les autres.
            autres = poids.drop([t for t in tech_restantes if t in poids.index])
            if autres.empty:
                break
            candidate = autres.idxmin()
        retenus.remove(candidate)

    return allouer(methode, cov.loc[retenus, retenus],
                   esperances.reindex(retenus), plafond, plancher,
                   regles(retenus))


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
              plafond: float = PLAFOND_LIGNE, regles=None) -> pd.DataFrame:
    """
    La courbe des compromis : volatilite minimale pour chaque rendement visé.

    C'est elle qui montre qu'un « portefeuille optimal » unique n'existe pas.
    Chaque point est optimal pour un niveau de risque accepté, et le choix de
    ce niveau n'est pas un calcul.

    `regles` recoit les memes contraintes sectorielles que les allocations :
    sans elles, la courbe serait celle d'un univers plus libre et les
    methodes se placeraient au-dessus d'une frontiere qu'elles ne peuvent
    pas atteindre — une comparaison fausse.
    """
    mu = esperances.reindex(cov.index).fillna(0.0).to_numpy()
    matrice = cov.to_numpy()
    n = len(cov)
    if n == 0:
        return pd.DataFrame(columns=["Rendement (%)", "Volatilité (%)"])

    regles = list(regles or [])
    bas = float(mu @ variance_minimale(cov, plafond,
                                       regles=regles).to_numpy())
    # Borne haute : le rendement du portefeuille le plus agressif admissible,
    # contraintes comprises, et non la moyenne des meilleures esperances.
    haut = float(mu @ rendement_maximal(cov, esperances, plafond,
                                        regles=regles).to_numpy())
    if not np.isfinite(bas) or not np.isfinite(haut) or haut <= bas:
        return pd.DataFrame(columns=["Rendement (%)", "Volatilité (%)"])

    lignes = []
    for cible in np.linspace(bas, haut, points):
        contrainte = {"type": "eq",
                      "fun": (lambda w, c=cible: float(w @ mu) - c)}
        poids = _resoudre(lambda w: float(w @ matrice @ w), n, plafond,
                          regles + [contrainte])
        variance = float(poids @ matrice @ poids)
        lignes.append({"Rendement (%)": float(poids @ mu) * 100,
                       "Volatilité (%)": float(np.sqrt(variance)) * 100})
    return pd.DataFrame(lignes)


def exposition_sectorielle(poids: pd.Series) -> pd.Series:
    """Poids cumulé par secteur, du plus lourd au plus leger, en %."""
    if poids is None or poids.empty:
        return pd.Series(dtype=float)
    cumul: dict[str, float] = {}
    for ticker, w in poids.items():
        nom = sec.secteur(ticker)
        cumul[nom] = cumul.get(nom, 0.0) + float(w) * 100
    serie = pd.Series(cumul).sort_values(ascending=False)
    serie.index.name = "Secteur"
    return serie
