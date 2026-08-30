"""
Construction de portefeuille, simulation et backtest.

Trois blocs :
  1. Estimation robuste (covariance retrecie, EWMA, rendements attendus lisses)
  2. Allocation (variance minimale, Sharpe max, parite de risque, HRP, frontiere)
  3. Validation (bootstrap, stress tests, backtest walk-forward avec couts)

La covariance empirique brute est un mauvais point de depart : avec N actifs il
faut estimer N(N+1)/2 parametres sur un historique court, et l'optimiseur
concentre ses poids precisement sur les erreurs d'estimation. D'ou les
estimateurs retrecis ci-dessous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

JOURS_BOURSE = 252


# ==========================================================================
# 1. Estimation
# ==========================================================================

def covariance_empirique(r: pd.DataFrame, freq: float = JOURS_BOURSE) -> pd.DataFrame:
    return r.dropna().cov(ddof=1) * freq


def covariance_ewma(r: pd.DataFrame, lambda_: float = 0.94,
                    freq: float = JOURS_BOURSE) -> pd.DataFrame:
    """
    Covariance a memoire exponentielle (RiskMetrics).

    Pondere les observations recentes plus fortement : la volatilite est
    persistante, donc l'estimateur reagit plus vite aux changements de regime.
    """
    x = r.dropna()
    if len(x) < 2:
        return pd.DataFrame(index=r.columns, columns=r.columns, dtype=float)
    ecarts = (x - x.mean()).to_numpy()
    poids = lambda_ ** np.arange(len(ecarts) - 1, -1, -1)
    poids = poids / poids.sum()
    cov = (ecarts * poids[:, None]).T @ ecarts
    return pd.DataFrame(cov * freq, index=x.columns, columns=x.columns)


def covariance_retrecie(r: pd.DataFrame, freq: float = JOURS_BOURSE,
                        intensite: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    Retrecissement de Ledoit-Wolf vers une cible spherique.

    Melange la covariance empirique avec une matrice diagonale de variance
    moyenne. L'intensite optimale est calculee analytiquement : plus
    l'historique est court par rapport au nombre d'actifs, plus on retrecit.
    Renvoie la matrice et l'intensite retenue.
    """
    x = r.dropna()
    T, N = x.shape
    if T < 3 or N == 0:
        return covariance_empirique(r, freq), 1.0

    ecarts = (x - x.mean()).to_numpy()
    S = ecarts.T @ ecarts / T
    mu = np.trace(S) / N
    cible = mu * np.eye(N)

    d2 = ((S - cible) ** 2).sum() / N
    b2 = sum(((np.outer(e, e) - S) ** 2).sum() for e in ecarts) / (T ** 2 * N)
    b2 = min(b2, d2)

    delta = float(np.clip(b2 / d2, 0.0, 1.0)) if d2 > 0 else 1.0
    if intensite is not None:
        delta = float(np.clip(intensite, 0.0, 1.0))

    sigma = delta * cible + (1 - delta) * S * T / max(T - 1, 1)
    return pd.DataFrame(sigma * freq, index=x.columns, columns=x.columns), delta


def rendements_attendus(r: pd.DataFrame, methode: str = "retreci",
                        freq: float = JOURS_BOURSE,
                        intensite: float = 0.5) -> pd.Series:
    """
    Rendements esperes.

    La moyenne historique est un estimateur notoirement instable — l'erreur
    type sur une moyenne annuelle est de l'ordre de la volatilite elle-meme.
    Le mode `retreci` la ramene vers la moyenne transversale (James-Stein),
    `egal` renonce completement a prevoir, `capm` deduit les rendements des
    betas et d'une prime de marche.
    """
    x = r.dropna()
    historique = x.mean() * freq

    if methode == "historique":
        return historique
    if methode == "egal":
        return pd.Series(historique.mean(), index=x.columns)
    if methode == "ewma":
        poids = 0.97 ** np.arange(len(x) - 1, -1, -1)
        poids = poids / poids.sum()
        return pd.Series((x.to_numpy() * poids[:, None]).sum(axis=0) * freq,
                         index=x.columns)
    return (1 - intensite) * historique + intensite * historique.mean()


# ==========================================================================
# 2. Allocation
# ==========================================================================

def _contraintes(n: int, bornes: tuple[float, float]):
    return [{"type": "eq", "fun": lambda w: w.sum() - 1.0}], [bornes] * n


def _depart(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def variance_minimale(cov: pd.DataFrame,
                      bornes: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Le seul portefeuille de la frontiere qui ne depend pas des rendements attendus."""
    n = len(cov)
    C = cov.to_numpy()
    cont, bnds = _contraintes(n, bornes)
    res = minimize(lambda w: w @ C @ w, _depart(n), method="SLSQP",
                   bounds=bnds, constraints=cont,
                   options={"maxiter": 500, "ftol": 1e-12})
    return pd.Series(res.x, index=cov.index)


def sharpe_maximal(mu: pd.Series, cov: pd.DataFrame,
                   taux_sans_risque: float = 0.0,
                   bornes: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Portefeuille tangent. Tres sensible aux erreurs sur `mu` — a manier avec prudence."""
    n = len(cov)
    C, m = cov.to_numpy(), mu.reindex(cov.index).to_numpy()
    cont, bnds = _contraintes(n, bornes)

    def negatif_sharpe(w):
        vol = np.sqrt(max(w @ C @ w, 1e-16))
        return -(w @ m - taux_sans_risque) / vol

    res = minimize(negatif_sharpe, _depart(n), method="SLSQP",
                   bounds=bnds, constraints=cont,
                   options={"maxiter": 800, "ftol": 1e-12})
    return pd.Series(res.x, index=cov.index)


def parite_de_risque(cov: pd.DataFrame, budgets: pd.Series | None = None,
                     iterations: int = 2000) -> pd.Series:
    """
    Chaque ligne contribue autant que les autres a la volatilite totale.

    N'utilise aucune prevision de rendement, seulement les covariances — d'ou
    sa stabilite dans le temps. Resolu par point fixe multiplicatif.
    """
    C = cov.to_numpy()
    n = len(C)
    b = (np.repeat(1.0 / n, n) if budgets is None
         else budgets.reindex(cov.index).to_numpy())
    b = b / b.sum()

    w = _depart(n)
    for _ in range(iterations):
        marginal = C @ w
        marginal = np.where(np.abs(marginal) < 1e-14, 1e-14, marginal)
        w_neuf = b / marginal
        w_neuf = np.clip(w_neuf, 1e-12, None)
        w_neuf = w_neuf / w_neuf.sum()
        if np.max(np.abs(w_neuf - w)) < 1e-12:
            w = w_neuf
            break
        w = w_neuf
    return pd.Series(w, index=cov.index)


def diversification_maximale(cov: pd.DataFrame,
                             bornes: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Maximise le rapport entre volatilite moyenne ponderee et volatilite realisee."""
    n = len(cov)
    C = cov.to_numpy()
    sigma = np.sqrt(np.diag(C))
    cont, bnds = _contraintes(n, bornes)

    def objectif(w):
        return -(w @ sigma) / np.sqrt(max(w @ C @ w, 1e-16))

    res = minimize(objectif, _depart(n), method="SLSQP", bounds=bnds,
                   constraints=cont, options={"maxiter": 800, "ftol": 1e-12})
    return pd.Series(res.x, index=cov.index)


def hrp(cov: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado).

    Regroupe les actifs par similarite, puis repartit le risque de haut en bas.
    N'inverse jamais la matrice de covariance, donc reste stable quand les
    actifs sont nombreux ou fortement correles — la ou les optimiseurs
    classiques deviennent erratiques.
    """
    actifs = list(cov.index)
    n = len(actifs)
    if n == 1:
        return pd.Series([1.0], index=actifs)

    C = cov.to_numpy()
    ecarts = np.sqrt(np.outer(np.diag(C), np.diag(C)))
    correl = np.clip(C / np.where(ecarts == 0, 1e-16, ecarts), -1, 1)
    distance = np.sqrt(np.clip((1 - correl) / 2, 0, 1))
    np.fill_diagonal(distance, 0.0)

    arbre = linkage(squareform(distance, checks=False), method="single")
    ordre = [n_.id for n_ in to_tree(arbre).pre_order(lambda x: x)]

    w = pd.Series(1.0, index=[actifs[i] for i in ordre])
    groupes = [list(w.index)]
    while groupes:
        groupes = [g[d:f] for g in groupes
                   for d, f in ((0, len(g) // 2), (len(g) // 2, len(g)))
                   if len(g) > 1]
        for i in range(0, len(groupes), 2):
            gauche, droite = groupes[i], groupes[i + 1]
            v_g = _variance_groupe(cov, gauche)
            v_d = _variance_groupe(cov, droite)
            alpha = 1 - v_g / (v_g + v_d) if (v_g + v_d) > 0 else 0.5
            w[gauche] *= alpha
            w[droite] *= 1 - alpha
    return w.reindex(cov.index).fillna(0.0)


def _variance_groupe(cov: pd.DataFrame, membres: list) -> float:
    """Variance du sous-portefeuille a variance inverse — brique interne de HRP."""
    sous = cov.loc[membres, membres]
    inv = 1 / np.diag(sous.to_numpy())
    w = inv / inv.sum()
    return float(w @ sous.to_numpy() @ w)


def frontiere_efficiente(mu: pd.Series, cov: pd.DataFrame, points: int = 40,
                         bornes: tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
    """Volatilite minimale pour chaque niveau de rendement cible atteignable."""
    n = len(cov)
    C, m = cov.to_numpy(), mu.reindex(cov.index).to_numpy()
    w_min = variance_minimale(cov, bornes).to_numpy()
    cibles = np.linspace(float(w_min @ m), float(m.max()), points)

    lignes = []
    for cible in cibles:
        cont = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, c=cible: w @ m - c}]
        res = minimize(lambda w: w @ C @ w, _depart(n), method="SLSQP",
                       bounds=[bornes] * n, constraints=cont,
                       options={"maxiter": 500, "ftol": 1e-11})
        if res.success:
            lignes.append({"Rendement": float(res.x @ m),
                           "Volatilité": float(np.sqrt(res.x @ C @ res.x)),
                           "Poids": pd.Series(res.x, index=cov.index)})
    return pd.DataFrame(lignes)


# ==========================================================================
# 3. Validation
# ==========================================================================

def bootstrap_horizon(r_ptf: pd.Series, horizon: int = 252,
                      tirages: int = 4000, taille_bloc: int = 10,
                      graine: int = 0) -> np.ndarray:
    """
    Distribution du capital final par bootstrap par blocs.

    Le tirage par blocs conserve l'autocorrelation et le regroupement de la
    volatilite, que le tirage independant detruirait — ce qui sous-estimerait
    fortement les drawdowns.
    """
    x = r_ptf.dropna().to_numpy()
    if len(x) < taille_bloc + 1:
        return np.array([])
    rng = np.random.default_rng(graine)
    n_blocs = int(np.ceil(horizon / taille_bloc))
    depart = rng.integers(0, len(x) - taille_bloc, size=(tirages, n_blocs))
    idx = depart[:, :, None] + np.arange(taille_bloc)
    chemins = x[idx].reshape(tirages, -1)[:, :horizon]
    return np.prod(1 + chemins, axis=1)


def stress_historique(r_ptf: pd.Series, fenetre: int = 21,
                      nombre: int = 5) -> pd.DataFrame:
    """Les pires fenetres glissantes reellement traversees par le portefeuille."""
    x = r_ptf.dropna()
    if len(x) < fenetre:
        return pd.DataFrame()
    cumul = (1 + x).rolling(fenetre).apply(np.prod, raw=True) - 1
    pires = cumul.nsmallest(nombre)
    return pd.DataFrame({
        "Fin de fenêtre": pires.index.date,
        f"Perte sur {fenetre} périodes (%)": pires.to_numpy() * 100,
    })


def stress_facteur(beta: float, chocs: dict[str, float],
                   valeur: float) -> pd.DataFrame:
    """Impact estime d'un choc de marche, propage par le beta du portefeuille."""
    return pd.DataFrame([
        {"Scénario": nom,
         "Choc indice (%)": choc * 100,
         "Impact estimé (%)": beta * choc * 100,
         "Perte estimée": beta * choc * valeur}
        for nom, choc in chocs.items()
    ])


CHOCS_TYPES = {
    "Correction modérée": -0.10,
    "Correction sévère": -0.20,
    "Krach type 2008": -0.40,
    "Choc type mars 2020": -0.34,
    "Rebond marqué": 0.15,
}


def backtest_walk_forward(rendements: pd.DataFrame, methode: str,
                          fenetre: int = 252, pas_rebalancement: int = 21,
                          cout_bps: float = 10.0,
                          taux_sans_risque: float = 0.0,
                          freq: float = JOURS_BOURSE,
                          estimateur: str = "retreci",
                          methode_mu: str = "retreci") -> dict:
    """
    Rejoue l'allocation dans le temps sans jamais utiliser d'information future.

    A chaque rebalancement, les poids sont estimes uniquement sur les `fenetre`
    periodes precedentes, puis appliques a la periode suivante. Les couts de
    transaction sont preleves sur le turnover reel. C'est le seul test qui
    distingue une methode robuste d'une methode qui a bien colle au passe.
    """
    r = rendements.dropna()
    if len(r) <= fenetre + pas_rebalancement:
        return {"rendements": pd.Series(dtype=float), "poids": pd.DataFrame(),
                "turnover": pd.Series(dtype=float)}

    dates, serie, historique_poids, turnovers = r.index, [], {}, {}
    w_actuel = pd.Series(0.0, index=r.columns)

    for i in range(fenetre, len(r)):
        if (i - fenetre) % pas_rebalancement == 0:
            passe = r.iloc[i - fenetre:i]
            w_cible = _poids_methode(passe, methode, taux_sans_risque, freq,
                                     estimateur, methode_mu)
            turnovers[dates[i]] = float((w_cible - w_actuel).abs().sum())
            cout = turnovers[dates[i]] * cout_bps / 10_000
            w_actuel = w_cible
            historique_poids[dates[i]] = w_cible.copy()
        else:
            cout = 0.0

        rdt = float((w_actuel * r.iloc[i]).sum()) - cout
        serie.append(rdt)
        # Derive naturelle des poids entre deux rebalancements
        croissance = w_actuel * (1 + r.iloc[i])
        somme = croissance.sum()
        if somme > 0:
            w_actuel = croissance / somme

    return {
        "rendements": pd.Series(serie, index=dates[fenetre:]),
        "poids": pd.DataFrame(historique_poids).T,
        "turnover": pd.Series(turnovers),
    }


def _poids_methode(passe: pd.DataFrame, methode: str, taux_sans_risque: float,
                   freq: float, estimateur: str, methode_mu: str) -> pd.Series:
    n = passe.shape[1]
    if methode == "equipondere":
        return pd.Series(1.0 / n, index=passe.columns)

    if estimateur == "ewma":
        cov = covariance_ewma(passe, freq=freq)
    elif estimateur == "empirique":
        cov = covariance_empirique(passe, freq=freq)
    else:
        cov, _ = covariance_retrecie(passe, freq=freq)

    if methode == "variance_min":
        return variance_minimale(cov)
    if methode == "parite_risque":
        return parite_de_risque(cov)
    if methode == "hrp":
        return hrp(cov)
    if methode == "diversification_max":
        return diversification_maximale(cov)
    if methode == "sharpe_max":
        mu = rendements_attendus(passe, methode_mu, freq)
        return sharpe_maximal(mu, cov, taux_sans_risque)
    return pd.Series(1.0 / n, index=passe.columns)


METHODES = {
    "equipondere": "Équipondéré",
    "variance_min": "Variance minimale",
    "parite_risque": "Parité de risque",
    "hrp": "HRP (hiérarchique)",
    "diversification_max": "Diversification maximale",
    "sharpe_max": "Sharpe maximal",
}
