"""
Module de calcul quantitatif — aucune dependance a Streamlit.

Toutes les fonctions prennent des rendements SIMPLES (pct_change), pas des prix,
sauf mention contraire. Les resultats annualises supposent 252 seances par an.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JOURS_BOURSE = 252


# --------------------------------------------------------------------------
# Bases
# --------------------------------------------------------------------------

def rendements(prix: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Rendements simples periode par periode."""
    return prix.pct_change().dropna(how="all")


def frequence_annuelle(index: pd.Index) -> float:
    """Deduit le nombre de periodes par an a partir de l'espacement des dates."""
    if len(index) < 3 or not isinstance(index, pd.DatetimeIndex):
        return JOURS_BOURSE
    jours = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    if jours <= 0:
        return JOURS_BOURSE
    if jours <= 1.5:
        return JOURS_BOURSE
    if jours <= 9:
        return 52.0
    if jours <= 45:
        return 12.0
    return 4.0


def rendement_annualise(r: pd.Series, freq: float = JOURS_BOURSE) -> float:
    """Taux de croissance annuel compose (CAGR) reconstruit depuis les rendements."""
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    if total <= 0:
        return np.nan
    return total ** (freq / len(r)) - 1


def volatilite(r: pd.Series, freq: float = JOURS_BOURSE) -> float:
    """Ecart-type annualise des rendements."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(freq))


def variance_annualisee(r: pd.Series, freq: float = JOURS_BOURSE) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.var(ddof=1) * freq)


def semi_deviation(r: pd.Series, seuil: float = 0.0,
                   freq: float = JOURS_BOURSE) -> float:
    """Volatilite calculee uniquement sur les rendements sous le seuil."""
    r = r.dropna()
    baisses = r[r < seuil] - seuil
    if len(baisses) < 2:
        return np.nan
    return float(np.sqrt((baisses ** 2).mean() * freq))


# --------------------------------------------------------------------------
# Ratios ajustes du risque
# --------------------------------------------------------------------------

def sharpe(r: pd.Series, taux_sans_risque: float = 0.0,
           freq: float = JOURS_BOURSE) -> float:
    """Exces de rendement par unite de volatilite totale."""
    vol = volatilite(r, freq)
    if not vol or np.isnan(vol) or vol == 0:
        return np.nan
    return (rendement_annualise(r, freq) - taux_sans_risque) / vol


def sortino(r: pd.Series, taux_sans_risque: float = 0.0,
            freq: float = JOURS_BOURSE) -> float:
    """Comme le Sharpe, mais ne penalise que la volatilite baissiere."""
    cible = (1 + taux_sans_risque) ** (1 / freq) - 1
    sd = semi_deviation(r, cible, freq)
    if not sd or np.isnan(sd) or sd == 0:
        return np.nan
    return (rendement_annualise(r, freq) - taux_sans_risque) / sd


def calmar(r: pd.Series, freq: float = JOURS_BOURSE) -> float:
    """Rendement annualise rapporte a la pire perte subie."""
    dd = abs(drawdown_max(r))
    if not dd or np.isnan(dd) or dd == 0:
        return np.nan
    return rendement_annualise(r, freq) / dd


# --------------------------------------------------------------------------
# Pertes
# --------------------------------------------------------------------------

def courbe_drawdown(r: pd.Series) -> pd.Series:
    """Ecart en % par rapport au plus haut historique atteint, a chaque date."""
    cumul = (1 + r.fillna(0)).cumprod()
    return cumul / cumul.cummax() - 1


def drawdown_max(r: pd.Series) -> float:
    dd = courbe_drawdown(r)
    return float(dd.min()) if len(dd) else np.nan


def duree_drawdown_max(r: pd.Series) -> int:
    """Nombre de periodes entre le sommet precedent et le point bas."""
    dd = courbe_drawdown(r)
    if dd.empty:
        return 0
    creux = dd.idxmin()
    avant = dd.loc[:creux]
    sommets = avant[avant >= -1e-12]
    if sommets.empty:
        return len(avant)
    return int(len(avant.loc[sommets.index[-1]:]))


def var_historique(r: pd.Series, seuil: float = 0.05) -> float:
    """Perte de la periode qui n'est depassee que dans `seuil` % des cas."""
    r = r.dropna()
    return float(np.quantile(r, seuil)) if len(r) else np.nan


def cvar_historique(r: pd.Series, seuil: float = 0.05) -> float:
    """Perte moyenne dans la queue au-dela de la VaR (expected shortfall)."""
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    var = np.quantile(r, seuil)
    queue = r[r <= var]
    return float(queue.mean()) if len(queue) else float(var)


def var_parametrique(r: pd.Series, seuil: float = 0.05) -> float:
    """VaR gaussienne — utile a comparer a l'historique pour juger la normalite."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    z = {0.01: -2.326, 0.05: -1.645, 0.10: -1.282}.get(round(seuil, 2), -1.645)
    return float(r.mean() + z * r.std(ddof=1))


# --------------------------------------------------------------------------
# Relation au marche
# --------------------------------------------------------------------------

def regression_marche(r_actif: pd.Series, r_marche: pd.Series,
                      taux_sans_risque: float = 0.0,
                      freq: float = JOURS_BOURSE) -> dict:
    """
    Regresse les exces de rendement de l'actif sur ceux du marche.

    Renvoie beta (sensibilite), alpha annualise (surperformance non expliquee
    par le marche), R2 (part du risque explique par le marche), et les mesures
    d'ecart au benchmark.
    """
    paire = pd.concat([r_actif, r_marche], axis=1, join="inner").dropna()
    paire.columns = ["actif", "marche"]
    vide = {k: np.nan for k in
            ("beta", "alpha", "r2", "correlation", "tracking_error",
             "information_ratio", "risque_specifique", "n")}

    if len(paire) < 20:
        return vide

    rf_periode = (1 + taux_sans_risque) ** (1 / freq) - 1
    y = (paire["actif"] - rf_periode).to_numpy()
    x = (paire["marche"] - rf_periode).to_numpy()

    var_x = x.var(ddof=1)
    if var_x == 0:
        return vide

    beta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    alpha_periode = float(y.mean() - beta * x.mean())
    residus = y - (alpha_periode + beta * x)
    var_y = y.var(ddof=1)
    r2 = float(1 - residus.var(ddof=1) / var_y) if var_y > 0 else np.nan

    ecart = paire["actif"] - paire["marche"]
    te = float(ecart.std(ddof=1) * np.sqrt(freq))

    return {
        "beta": beta,
        "alpha": (1 + alpha_periode) ** freq - 1,
        "r2": r2,
        "correlation": float(np.corrcoef(y, x)[0, 1]),
        "tracking_error": te,
        "information_ratio": float(ecart.mean() * freq / te) if te > 0 else np.nan,
        "risque_specifique": float(residus.std(ddof=1) * np.sqrt(freq)),
        "n": int(len(paire)),
    }


def beta_conditionnel(r_actif: pd.Series, r_marche: pd.Series) -> dict:
    """Beta estime separement sur les seances de hausse et de baisse du marche."""
    paire = pd.concat([r_actif, r_marche], axis=1, join="inner").dropna()
    paire.columns = ["actif", "marche"]
    out = {}
    for nom, masque in (("hausse", paire["marche"] > 0),
                        ("baisse", paire["marche"] < 0)):
        sous = paire[masque]
        if len(sous) < 15 or sous["marche"].var(ddof=1) == 0:
            out[f"beta_{nom}"] = np.nan
            out[f"capture_{nom}"] = np.nan
            continue
        out[f"beta_{nom}"] = float(
            np.cov(sous["actif"], sous["marche"], ddof=1)[0, 1]
            / sous["marche"].var(ddof=1)
        )
        denom = float((1 + sous["marche"]).prod() - 1)
        num = float((1 + sous["actif"]).prod() - 1)
        out[f"capture_{nom}"] = num / denom if denom != 0 else np.nan
    return out


# --------------------------------------------------------------------------
# Portefeuille
# --------------------------------------------------------------------------

def matrice_covariance(rendements_df: pd.DataFrame,
                       freq: float = JOURS_BOURSE) -> pd.DataFrame:
    """Covariance annualisee entre les lignes du portefeuille."""
    return rendements_df.dropna().cov(ddof=1) * freq


def rendements_portefeuille(rendements_df: pd.DataFrame,
                            poids: pd.Series) -> pd.Series:
    """Serie de rendements du portefeuille a poids constants (rebalance)."""
    aligne = rendements_df[poids.index].dropna()
    return (aligne * poids).sum(axis=1)


def decomposition_risque(poids: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Repartit la volatilite du portefeuille entre ses lignes.

    La contribution marginale est la derivee de la volatilite totale par rapport
    au poids : elle indique ou une reduction de position fait vraiment baisser
    le risque. La somme des contributions absolues egale la volatilite totale.
    """
    w = poids.reindex(cov.index).fillna(0.0)
    variance = float(w @ cov.to_numpy() @ w)
    vol = np.sqrt(variance) if variance > 0 else np.nan
    if not vol or np.isnan(vol) or vol == 0:
        return pd.DataFrame(index=cov.index)

    marginale = (cov.to_numpy() @ w.to_numpy()) / vol
    contribution = w.to_numpy() * marginale
    return pd.DataFrame({
        "Poids (%)": w.to_numpy() * 100,
        "Volatilité seule (%)": np.sqrt(np.diag(cov.to_numpy())) * 100,
        "Contribution marginale (%)": marginale * 100,
        "Contribution au risque (%)": contribution * 100,
        "Part du risque (%)": contribution / vol * 100,
    }, index=cov.index)


def volatilite_portefeuille(poids: pd.Series, cov: pd.DataFrame) -> float:
    w = poids.reindex(cov.index).fillna(0.0).to_numpy()
    variance = float(w @ cov.to_numpy() @ w)
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def ratio_diversification(poids: pd.Series, cov: pd.DataFrame) -> float:
    """
    Volatilite moyenne ponderee des lignes divisee par la volatilite reelle.

    Vaut 1 si tout est parfaitement correle, monte quand la diversification joue.
    """
    w = poids.reindex(cov.index).fillna(0.0)
    moyenne = float((w * np.sqrt(np.diag(cov.to_numpy()))).sum())
    vol = volatilite_portefeuille(poids, cov)
    return moyenne / vol if vol and not np.isnan(vol) and vol > 0 else np.nan


def nombre_effectif_lignes(poids: pd.Series) -> float:
    """
    Inverse de l'indice de Herfindahl.

    Dix lignes dont une pese 80 % ne valent qu'environ 1,5 ligne effective.
    """
    w = poids.to_numpy(dtype=float)
    somme = (w ** 2).sum()
    return float(1 / somme) if somme > 0 else np.nan


# --------------------------------------------------------------------------
# Synthese
# --------------------------------------------------------------------------

def treynor(r: pd.Series, r_marche: pd.Series, taux_sans_risque: float = 0.0,
            freq: float = JOURS_BOURSE) -> float:
    """
    Exces de rendement par unite de risque de MARCHE (beta), pas de risque total.

    A preferer au Sharpe quand la position s'ajoute a un portefeuille deja
    diversifie : le risque specifique y sera dilue, seul le beta compte.
    """
    beta = regression_marche(r, r_marche, taux_sans_risque, freq)["beta"]
    if not beta or np.isnan(beta) or beta == 0:
        return np.nan
    return (rendement_annualise(r, freq) - taux_sans_risque) / beta


def omega(r: pd.Series, seuil: float = 0.0) -> float:
    """
    Rapport entre gains et pertes au-dela d'un seuil.

    Contrairement au Sharpe, n'impose aucune hypothese de normalite : utilise
    toute la distribution, asymetrie et queues epaisses comprises.
    """
    x = r.dropna() - seuil
    gains = x[x > 0].sum()
    pertes = -x[x < 0].sum()
    return float(gains / pertes) if pertes > 0 else np.nan


def beta_baissier(r: pd.Series, r_marche: pd.Series) -> float:
    """
    Beta estime uniquement sur les seances de baisse du marche.

    C'est le beta qui compte vraiment : celui qui mesure ce que tu perds
    quand tout baisse. Un ecart marque avec le beta global est un signal.
    """
    return beta_conditionnel(r, r_marche).get("beta_baisse", np.nan)


def var_cornish_fisher(r: pd.Series, seuil: float = 0.05) -> float:
    """
    VaR corrigee de l'asymetrie et des queues epaisses.

    La VaR gaussienne sous-estime le risque des distributions financieres.
    Cette correction ajuste le quantile par les moments d'ordre 3 et 4.
    """
    x = r.dropna()
    if len(x) < 30:
        return np.nan
    z = {0.01: -2.326, 0.05: -1.645, 0.10: -1.282}.get(round(seuil, 2), -1.645)
    s, k = float(x.skew()), float(x.kurt())
    zc = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
          - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(x.mean() + zc * x.std(ddof=1))


def ratio_queue(r: pd.Series) -> float:
    """
    Meilleur 5 % rapporte au pire 5 %.

    Au-dessus de 1, les gains extremes depassent les pertes extremes.
    En dessous, la distribution penche du mauvais cote.
    """
    x = r.dropna()
    if len(x) < 30:
        return np.nan
    bas = abs(np.percentile(x, 5))
    return float(np.percentile(x, 95) / bas) if bas > 0 else np.nan


def ratio_gain_perte(r: pd.Series) -> float:
    """Gain moyen des periodes positives sur perte moyenne des negatives."""
    x = r.dropna()
    gains, pertes = x[x > 0], x[x < 0]
    if len(gains) == 0 or len(pertes) == 0:
        return np.nan
    return float(gains.mean() / abs(pertes.mean()))


def k_ratio(r: pd.Series) -> float:
    """
    Regularite de la progression : pente de la courbe de capital rapportee
    a son erreur type.

    Deux portefeuilles au meme rendement final n'ont pas la meme qualite si
    l'un progresse regulierement et l'autre par a-coups. Le K-ratio les
    distingue la ou le Sharpe les confond.
    """
    x = r.dropna()
    if len(x) < 30:
        return np.nan
    y = np.log((1 + x).cumprod()).to_numpy()
    t = np.arange(len(y))
    pente, ordonnee = np.polyfit(t, y, 1)
    residus = y - (pente * t + ordonnee)
    erreur = np.sqrt((residus ** 2).sum() / (len(y) - 2)) / np.sqrt(
        ((t - t.mean()) ** 2).sum())
    return float(pente / erreur) if erreur > 0 else np.nan


def m2_modigliani(r: pd.Series, r_marche: pd.Series,
                  taux_sans_risque: float = 0.0,
                  freq: float = JOURS_BOURSE) -> float:
    """
    Rendement qu'aurait le portefeuille ramene a la volatilite du marche.

    Traduit le Sharpe en points de performance, directement comparables a
    ceux de l'indice — bien plus parlant qu'un ratio sans unite.
    """
    s = sharpe(r, taux_sans_risque, freq)
    vol_marche = volatilite(r_marche, freq)
    if np.isnan(s) or np.isnan(vol_marche):
        return np.nan
    return s * vol_marche + taux_sans_risque


def tableau_metriques(r: pd.Series, r_marche: pd.Series | None = None,
                      taux_sans_risque: float = 0.0,
                      freq: float = JOURS_BOURSE) -> dict:
    """Toutes les mesures principales pour une serie de rendements."""
    metriques = {
        "Rendement annualisé (%)": rendement_annualise(r, freq) * 100,
        "Volatilité annualisée (%)": volatilite(r, freq) * 100,
        "Variance annualisée": variance_annualisee(r, freq),
        "Semi-déviation (%)": semi_deviation(r, 0.0, freq) * 100,
        "Sharpe": sharpe(r, taux_sans_risque, freq),
        "Sortino": sortino(r, taux_sans_risque, freq),
        "Calmar": calmar(r, freq),
        "Omega": omega(r),
        "K-ratio": k_ratio(r),
        "Ratio gain/perte": ratio_gain_perte(r),
        "Ratio de queue": ratio_queue(r),
        "Drawdown max (%)": drawdown_max(r) * 100,
        "Durée du drawdown max (périodes)": duree_drawdown_max(r),
        "VaR 95 % (%)": var_historique(r, 0.05) * 100,
        "CVaR 95 % (%)": cvar_historique(r, 0.05) * 100,
        "VaR 95 % gaussienne (%)": var_parametrique(r, 0.05) * 100,
        "Asymétrie": float(r.skew()),
        "Kurtosis excédentaire": float(r.kurt()),
        "% de périodes positives": float((r > 0).mean() * 100),
    }
    if r_marche is not None:
        metriques.update({
            "Bêta": regression_marche(r, r_marche, taux_sans_risque, freq)["beta"],
        })
        reg = regression_marche(r, r_marche, taux_sans_risque, freq)
        metriques.update({
            "Alpha annualisé (%)": reg["alpha"] * 100,
            "R²": reg["r2"],
            "Corrélation au marché": reg["correlation"],
            "Tracking error (%)": reg["tracking_error"] * 100,
            "Information ratio": reg["information_ratio"],
            "Risque spécifique (%)": reg["risque_specifique"] * 100,
            "Treynor": treynor(r, r_marche, taux_sans_risque, freq),
            "M² de Modigliani (%)": m2_modigliani(
                r, r_marche, taux_sans_risque, freq) * 100,
            "VaR 95 % Cornish-Fisher (%)": var_cornish_fisher(r, 0.05) * 100,
        })
        metriques.update(beta_conditionnel(r, r_marche))
    return metriques
