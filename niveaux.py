"""
Seuils de vente et objectifs, a trois horizons.

Le principe est le meme aux trois echelles, seule la fenetre change : un stop
suiveur accroche au plus haut recent et ecarte d'un multiple de la volatilite
reelle du titre, un objectif exprime en multiple du risque effectivement pris.

Sur la valeur de ces deux familles de niveaux, il faut etre clair, parce que
les traiter comme equivalentes serait trompeur.

Le stop est defendable. Il repose sur l'ATR, mesure de l'amplitude moyenne des
seances, et sur la methode dite du chandelier (LeBeau) : plus haut de la
periode moins k fois l'ATR. Son merite n'est pas de predire quoi que ce soit
mais de dimensionner la tolerance au bruit propre a chaque titre — un seuil a
8 % sur une valeur qui bouge de 1 % par jour et a 30 % sur une valeur qui
bouge de 6 % ont la meme signification statistique. C'est un outil de
controle du risque, et c'est a ce titre qu'il est utile.

L'objectif est beaucoup plus faible. Aucune methode ne sait ou un titre va
s'arreter de monter, et les cibles tirees des extensions de Fibonacci ou des
projections de figures n'ont pas de fondement empirique serieux. Ce qui est
retenu ici est donc volontairement modeste : un multiple du risque pris, qui
ne dit pas ou ira le cours mais a partir de quel gain la position a paye son
propre risque. C'est une regle de gestion, pas une prevision. La resistance
structurelle est affichee a cote, a titre d'information sur le niveau ou des
vendeurs se sont deja manifestes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import reglages as rg

# Fenetre d'observation, multiple d'ATR pour le stop, multiple de risque pour
# l'objectif. Le multiple d'ATR croit avec l'horizon : un stop de long terme
# doit survivre a des corrections qui n'ont pas de sens a court terme.
HORIZONS = {
    "Court terme": {"fenetre": 22, "atr": 2.5, "gain": 1.5,
                    "duree": "quelques semaines"},
    "Moyen terme": {"fenetre": 55, "atr": 3.0, "gain": 2.5,
                    "duree": "quelques mois"},
    "Long terme": {"fenetre": 120, "atr": 3.5, "gain": 4.0,
                   "duree": "un an et plus"},
}

PERIODE_ATR = 14

# Perte maximale acceptee sur une position, en % du prix de revient.
PERTE_CAPITAL = rg.PERTE_CAPITAL


def atr(haut: pd.Series, bas: pd.Series, cloture: pd.Series,
        n: int = PERIODE_ATR) -> pd.Series:
    """
    Amplitude vraie moyenne, lissage de Wilder.

    L'amplitude vraie d'une seance est le plus grand des trois ecarts : haut
    moins bas, haut moins cloture veille, bas moins cloture veille. Les deux
    derniers capturent les ouvertures en decalage, qu'un simple haut moins bas
    ignorerait — c'est precisement ce qui fait sauter les stops.
    """
    veille = cloture.shift(1)
    amplitude = pd.concat([haut - bas,
                           (haut - veille).abs(),
                           (bas - veille).abs()], axis=1).max(axis=1)

    # Lissage de Wilder, amorce comprise : la premiere valeur est la moyenne
    # simple des n premieres amplitudes, les suivantes suivent la recurrence
    # ATR = ATR + (amplitude - ATR) / n. Une moyenne exponentielle ordinaire
    # s'amorce sur la premiere amplitude seule et s'ecarte de quelques pour
    # cent pendant une centaine de seances — assez pour deplacer un stop.
    return _wilder(amplitude, n)


def _wilder(valeurs: pd.Series, n: int) -> pd.Series:
    """
    Lissage de Wilder : moyenne des n premieres valeurs, puis recurrence.

    Extrait du calcul de l'ATR, ou il etait deja ecrit. Le RSI emploie le meme
    lissage — c'est le meme auteur — et une moyenne exponentielle ordinaire
    donnerait des valeurs proches mais fausses, ce qui est pire que
    franchement differentes.
    """
    a = valeurs.to_numpy(dtype=float)
    sortie = np.full(len(a), np.nan)
    if len(a) < n:
        return pd.Series(sortie, index=valeurs.index)
    courant = float(np.nanmean(a[:n]))
    sortie[n - 1] = courant
    for i in range(n, len(a)):
        if np.isfinite(a[i]):
            courant += (a[i] - courant) / n
        sortie[i] = courant
    return pd.Series(sortie, index=valeurs.index)


def moyenne_mobile(cloture: pd.Series, n: int) -> pd.Series:
    """Moyenne arithmetique simple sur n seances."""
    return cloture.rolling(n, min_periods=n).mean()


def rsi(cloture: pd.Series, n: int = 14) -> pd.Series:
    """
    Indice de force relative de Wilder.

    Rapport entre la hausse moyenne et la baisse moyenne des n dernieres
    seances, ramene sur une echelle de 0 a 100. Au-dela de 70 le titre a
    beaucoup monte recemment, en deca de 30 beaucoup baisse — ce qui est un
    constat sur le passe, pas une prevision : un titre en forte tendance
    reste « surachete » pendant des mois sans se retourner.
    """
    variation = cloture.diff()
    hausses = variation.clip(lower=0.0)
    baisses = (-variation).clip(lower=0.0)
    moyenne_hausse = _wilder(hausses, n)
    moyenne_baisse = _wilder(baisses, n)
    # Baisse moyenne nulle : que des hausses, le RSI sature a 100.
    rapport = moyenne_hausse / moyenne_baisse.replace(0.0, np.nan)
    sortie = 100 - 100 / (1 + rapport)
    return sortie.where(moyenne_baisse != 0, 100.0)


def momentum(cloture: pd.Series, n: int = 20) -> pd.Series:
    """Variation en pourcentage sur n seances."""
    return (cloture / cloture.shift(n) - 1) * 100


def _dernier(serie: pd.Series) -> float:
    valeurs = serie.dropna()
    return float(valeurs.iloc[-1]) if len(valeurs) else np.nan


def niveaux(ohlc: pd.DataFrame, pru: float = np.nan,
            perte_capital: float = PERTE_CAPITAL) -> pd.DataFrame:
    """
    Trois horizons, deux familles de seuils, un tableau.

    Le stop de marche vient du titre : plus haut de la periode moins un
    multiple de l'ATR. Il ignore ce que vous avez paye, et c'est voulu — le
    cours l'ignore aussi. C'est la seule facon de dimensionner la tolerance
    au bruit d'un titre.

    Le stop de capital vient de vous : le prix de revient diminue de la perte
    maximale que vous acceptez sur une position. Ce n'est pas une lecture du
    marche mais une regle de gestion, et elle a sa place a cote de l'autre.

    Le seuil retenu est le PLUS HAUT des deux, parce qu'a la baisse c'est
    celui-la qui se declenche en premier. La colonne Origine dit lequel mord,
    ce qui evite de croire a une lecture technique la ou c'est une regle
    personnelle qui a tranche, et inversement.

    Les deux dernieres colonnes rapportent tout au prix de revient : ce que
    le stop ferait perdre, ce que l'objectif ferait gagner. Les pourcentages
    de marge et de potentiel, eux, restent rapportes au cours — deux
    references differentes pour deux questions differentes.
    """
    # Convention de signe, valable dans tout le module : une marge positive
    # est une baisse encore encaissable avant de toucher le stop, une marge
    # negative signifie que le cours est deja passe dessous. Le signe se lit
    # donc comme une sante : positif bon, negatif mauvais.
    colonnes = ["Horizon", "Stop", "Origine", "Marge (%)",
                "Résultat au stop (%)", "Objectif", "Potentiel (%)",
                "Gain à l'objectif (%)", "Stop marché", "Stop capital",
                "Risque (%)", "Gain visé (%)", "Résistance", "Support",
                "Franchi", "Sous le PRU"]
    if ohlc is None or ohlc.empty or len(ohlc) < 40:
        return pd.DataFrame(columns=colonnes)

    haut, bas = ohlc["High"].astype(float), ohlc["Low"].astype(float)
    cloture = ohlc["Close"].astype(float)
    actuel = float(cloture.dropna().iloc[-1])
    a = _dernier(atr(haut, bas, cloture))
    if not np.isfinite(a) or a <= 0:
        return pd.DataFrame(columns=colonnes)

    lignes = []
    for nom, p in HORIZONS.items():
        f = min(p["fenetre"], len(cloture))
        plus_haut = float(haut.tail(f).max())
        plus_bas = float(bas.tail(f).min())

        # Stop du chandelier, sans autre correction que la garantie d'etre
        # sous le cours. Un premier essai le bornait par le creux de la
        # periode, pour ne pas accepter une perte superieure au dernier repli
        # connu ; en tendance haussiere ce creux se situe juste sous le cours
        # et ecrasait les trois horizons a la meme valeur. Le creux reste
        # affiche a titre indicatif, il ne contraint plus le calcul.
        stop_marche = plus_haut - p["atr"] * a

        # Stop de capital : le prix de revient ampute de la perte acceptee.
        # Sans prix de revient connu — une valeur simplement surveillee — il
        # n'existe pas, et seul le stop de marche s'applique.
        pru_connu = np.isfinite(pru) and pru > 0
        stop_capital = pru * (1 - perte_capital / 100) if pru_connu else np.nan

        # A la baisse, le premier seuil touche est le plus haut des deux.
        if pru_connu:
            stop = max(stop_marche, stop_capital)
            origine = "marché" if stop_marche >= stop_capital else "capital"
        else:
            stop, origine = stop_marche, "marché"

        # Un stop au-dessus du cours signifie que le titre a deja perdu plus
        # que sa tolerance au bruit depuis son plus haut : le seuil est
        # franchi. Le ramener sous le cours donnerait l'illusion d'une
        # position encore dans son enveloppe de risque. On le laisse ou il est
        # et on le signale ; l'objectif, lui, n'a plus de sens, puisqu'il se
        # calcule a partir d'un risque que la position a deja depasse.
        franchi = stop >= actuel
        risque = np.nan if franchi else (actuel - stop) / actuel * 100
        objectif = (np.nan if franchi
                    else actuel + p["gain"] * (actuel - stop))

        lignes.append({
            "Horizon": nom,
            "Stop": stop,
            "Origine": origine,
            "Marge (%)": (1 - stop / actuel) * 100,
            # Rapporte au prix de revient : ce que la sortie laisserait
            # reellement, positif comme negatif.
            "Résultat au stop (%)": ((stop / pru - 1) * 100
                                     if pru_connu else np.nan),
            "Objectif": objectif,
            "Potentiel (%)": (np.nan if franchi
                              else (objectif / actuel - 1) * 100),
            "Gain à l'objectif (%)": ((objectif / pru - 1) * 100
                                      if pru_connu and not franchi else np.nan),
            "Stop marché": stop_marche,
            "Stop capital": stop_capital,
            "Risque (%)": risque,
            "Gain visé (%)": np.nan if franchi else p["gain"] * risque,
            "Résistance": plus_haut,
            "Support": plus_bas,
            "Franchi": bool(franchi),
            "Sous le PRU": (bool(stop < pru) if np.isfinite(pru) and pru > 0
                            else None)})

    return pd.DataFrame(lignes)[colonnes]


def synthese(ohlc: pd.DataFrame, horizon: str = "Moyen terme",
             pru: float = np.nan,
             perte_capital: float = PERTE_CAPITAL) -> dict:
    """Les chiffres d'un seul horizon, pour un tableau recapitulatif."""
    table = niveaux(ohlc, pru, perte_capital)
    if table.empty:
        return {}
    ligne = table[table["Horizon"] == horizon]
    return ligne.iloc[0].to_dict() if not ligne.empty else {}


def volatilite_relative(ohlc: pd.DataFrame) -> float:
    """ATR rapporte au cours, en pourcentage : l'amplitude d'une seance type."""
    if ohlc is None or ohlc.empty or len(ohlc) < 40:
        return np.nan
    a = _dernier(atr(ohlc["High"].astype(float), ohlc["Low"].astype(float),
                     ohlc["Close"].astype(float)))
    actuel = float(ohlc["Close"].dropna().iloc[-1])
    return a / actuel * 100 if actuel > 0 else np.nan
