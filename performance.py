"""
Mesure de performance : la question que l'outil evitait.

Savoir qu'un portefeuille a gagne 12 % ne dit rien tant qu'on ignore ce
qu'aurait donne le meme argent place passivement. C'est le seul chiffre qui
tranche entre competence et marche porteur, et il est desagreable assez
souvent pour meriter d'etre calcule.

Trois mesures, qui ne repondent pas a la meme question.

Le TWR, rendement pondere par le temps, neutralise les versements et les
retraits. Il mesure la qualite des choix de titres, pas le moment ou l'argent
est arrive. C'est la mesure normalisee de la profession, celle qui permet de
comparer deux gerants.

Le TRI, ou rendement pondere par les capitaux, tient compte des montants et
des dates. Il mesure ce que l'investisseur a reellement gagne. Quand il est
tres inferieur au TWR, c'est que les renforcements sont tombes aux mauvais
moments — un defaut de comportement, pas de selection.

Le contrefactuel est le plus brutal et le plus utile : les memes versements,
aux memes dates, places sur un indice. Il repond a la seule question qui
compte vraiment, et la recherche sur la performance des particuliers laisse
peu de doute sur le resultat le plus frequent.

Aucune de ces mesures ne prouve quoi que ce soit sur quelques mois : sur une
periode courte, l'ecart avec un indice est du bruit. Elles ne deviennent
interpretables qu'apres plusieurs annees, et le module le rappelle.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

import mouvements as mo

JOURS_AN = 365.25
DUREE_INTERPRETABLE = 3.0        # annees en deca desquelles l'ecart est du bruit


# ==========================================================================
# Reconstitution de l'historique des positions
# ==========================================================================

def quantites_quotidiennes(mvts: pd.DataFrame,
                           dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Quantite detenue de chaque ligne, jour par jour.

    Les operations sont rejouees dans l'ordre et reportees sur le calendrier
    de bourse. Une division multiplie la quantite, comme dans le journal :
    les deux moteurs doivent donner le meme etat final, sans quoi la
    performance serait calculee sur des positions differentes de celles
    affichees.
    """
    tickers = sorted(mvts["Ticker"].unique()) if not mvts.empty else []
    table = pd.DataFrame(0.0, index=dates, columns=tickers)
    if mvts.empty:
        return table

    courant = {t: 0.0 for t in tickers}
    operations = mvts.sort_values(["Date", "Ligne"])
    i = 0
    for jour in dates:
        while i < len(operations) and \
                pd.Timestamp(operations.iloc[i]["Date"]).normalize() <= jour:
            m = operations.iloc[i]
            t, q = m["Ticker"], float(m["Quantité"])
            if m["Sens"] == mo.DIVISION:
                courant[t] *= q
            elif m["Sens"] == mo.ACHAT:
                courant[t] += q
            else:
                courant[t] = max(0.0, courant[t] - min(q, courant[t]))
            i += 1
        table.loc[jour] = [courant[t] for t in tickers]
    return table


def valeur_quotidienne(quantites: pd.DataFrame, cours: pd.DataFrame,
                       change: dict | None = None) -> pd.Series:
    """Valeur du portefeuille a chaque date, dans la devise de reference."""
    if quantites.empty or cours.empty:
        return pd.Series(dtype=float)
    communs = [t for t in quantites.columns if t in cours.columns]
    if not communs:
        return pd.Series(dtype=float)
    prix = cours[communs].reindex(quantites.index).ffill()
    taux = pd.Series({t: (change or {}).get(t, 1.0) for t in communs})
    return (quantites[communs] * prix * taux).sum(axis=1)


def flux_externes(mvts: pd.DataFrame, change: dict | None = None) -> pd.Series:
    """
    Argent entre et sorti du portefeuille, par date.

    Positif a l'achat, negatif a la vente. Une division ne fait entrer ni
    sortir d'argent et n'apparait pas. Les frais sont comptes a l'entree
    comme a la sortie : ils sont sortis de la poche.
    """
    if mvts is None or mvts.empty:
        return pd.Series(dtype=float)
    lignes = {}
    for _, m in mvts.iterrows():
        if m["Sens"] == mo.DIVISION:
            continue
        taux = (change or {}).get(m["Ticker"], 1.0)
        montant = float(m["Quantité"]) * float(m["Prix unitaire"]) * taux
        frais = float(m["Frais"]) * taux
        jour = pd.Timestamp(m["Date"]).normalize()
        signe = 1.0 if m["Sens"] == mo.ACHAT else -1.0
        lignes[jour] = lignes.get(jour, 0.0) + signe * montant + frais
    return pd.Series(lignes).sort_index()


# ==========================================================================
# Rendement pondere par le temps
# ==========================================================================

def twr(valeur: pd.Series, flux: pd.Series) -> dict:
    """
    Rendement pondere par le temps, methode des periodes chainees.

    Sur chaque intervalle, le rendement retranche le flux de la valeur finale
    avant de la rapporter a la valeur initiale : un versement ne doit pas
    passer pour une performance. Les rendements sont ensuite chaines.
    """
    valeur = valeur.dropna()
    if len(valeur) < 2:
        return {}

    flux = flux.reindex(valeur.index).fillna(0.0) if len(flux) else \
        pd.Series(0.0, index=valeur.index)

    facteurs = []
    for i in range(1, len(valeur)):
        debut = float(valeur.iloc[i - 1])
        fin = float(valeur.iloc[i])
        entree = float(flux.iloc[i])
        if debut <= 0:
            # Portefeuille vide en debut de periode : le versement cree la
            # position, il n'y a pas de rendement a mesurer.
            continue
        facteurs.append((fin - entree) / debut)

    if not facteurs:
        return {}
    cumul = float(np.prod(facteurs))
    annees = (valeur.index[-1] - valeur.index[0]).days / JOURS_AN
    annualise = cumul ** (1 / annees) - 1 if annees > 0 and cumul > 0 else np.nan
    return {"total": cumul - 1, "annualise": annualise, "annees": annees,
            "periodes": len(facteurs)}


# ==========================================================================
# Rendement pondere par les capitaux
# ==========================================================================

def tri(flux: pd.Series, valeur_finale: float, date_finale) -> float:
    """
    Taux de rendement interne, resolu par bissection.

    La bissection est plus lente que Newton mais ne diverge pas : sur des flux
    irreguliers, Newton s'echappe regulierement vers des valeurs absurdes, et
    un taux de rendement faux est pire que pas de taux du tout.
    """
    if flux is None or flux.empty or not np.isfinite(valeur_finale):
        return np.nan

    mouvements = [(pd.Timestamp(d), float(m)) for d, m in flux.items()]
    mouvements.append((pd.Timestamp(date_finale), -float(valeur_finale)))
    origine = min(d for d, _ in mouvements)

    def valeur_actuelle(taux: float) -> float:
        total = 0.0
        for jour, montant in mouvements:
            annees = (jour - origine).days / JOURS_AN
            total += montant / (1 + taux) ** annees
        return total

    bas, haut = -0.9999, 10.0
    v_bas, v_haut = valeur_actuelle(bas), valeur_actuelle(haut)
    if not np.isfinite(v_bas) or not np.isfinite(v_haut) or v_bas * v_haut > 0:
        return np.nan
    for _ in range(200):
        milieu = (bas + haut) / 2
        v = valeur_actuelle(milieu)
        if abs(v) < 1e-9:
            return milieu
        if v_bas * v < 0:
            haut, v_haut = milieu, v
        else:
            bas, v_bas = milieu, v
    return (bas + haut) / 2


# ==========================================================================
# Contrefactuel
# ==========================================================================

def contrefactuel(flux: pd.Series, cours_indice: pd.Series) -> dict:
    """
    Les memes versements, aux memes dates, places sur un indice.

    Chaque entree achete des parts au cours du jour, chaque sortie en vend.
    C'est la comparaison la plus severe et la plus honnete : elle neutralise
    a la fois le talent de selection et le moment des versements, puisque les
    dates sont imposees par le portefeuille reel.
    """
    serie = cours_indice.dropna() if cours_indice is not None else pd.Series(dtype=float)
    if flux is None or flux.empty or serie.empty:
        return {}

    parts, investi = 0.0, 0.0
    for jour, montant in flux.items():
        jour = pd.Timestamp(jour).normalize()
        anterieurs = serie[serie.index <= jour]
        if anterieurs.empty:
            continue
        prix = float(anterieurs.iloc[-1])
        if prix <= 0:
            continue
        parts += montant / prix
        investi += montant
        parts = max(parts, 0.0)

    if parts <= 0:
        return {}
    finale = float(serie.iloc[-1]) * parts
    return {"parts": parts, "valeur": finale, "investi": investi,
            "gain": finale - investi,
            "rendement": finale / investi - 1 if investi > 0 else np.nan}


def comparaison(valeur_reelle: float, contre: dict) -> dict:
    """Ecart entre le portefeuille et son contrefactuel, en euros et en %."""
    if not contre or not np.isfinite(valeur_reelle):
        return {}
    ecart = valeur_reelle - contre["valeur"]
    return {"ecart": ecart,
            "ecart_pct": (ecart / contre["valeur"] * 100
                          if contre["valeur"] else np.nan),
            "gagne": ecart > 0}


def fiable(annees: float) -> bool:
    """En deca de trois ans, l'ecart avec un indice ne se distingue pas du bruit."""
    return bool(np.isfinite(annees) and annees >= DUREE_INTERPRETABLE)
