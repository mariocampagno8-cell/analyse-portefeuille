"""
Strategies d'options.

Quatre structures classiques, calculees sur les chaines d'options reelles.

  COVERED CALL       detenir le titre, vendre un call. Encaisse une prime,
                     plafonne le gain.
  CASH-SECURED PUT   vendre un put en immobilisant le montant d'achat.
                     Miroir synthetique du covered call.
  COLLAR             covered call plus put protecteur. Verrouille une
                     fourchette, souvent a cout quasi nul.
  IRON CONDOR        vendre un call et un put hors de la monnaie, en acheter
                     de plus eloignes. Parie sur l'immobilite.

Trois avertissements que le module rappelle a l'ecran, parce qu'ils sont
systematiquement sous-estimes.

Le gain d'un covered call est PLAFONNE et la perte ne l'est pas. Encaisser 2 %
de prime contre l'abandon d'une hausse de 30 % est un mauvais echange, et il
se produit precisement quand on avait raison sur le titre.

Le rendement annualise d'une prime est trompeur. Multiplier une prime
mensuelle par douze suppose de pouvoir la reproduire chaque mois aux memes
conditions, ce que la volatilite ne garantit jamais.

La probabilite d'assignation approchee par le delta n'est pas une prevision :
c'est ce que le marche price, et le marche se trompe regulierement.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf

STRATEGIES = {
    "covered_call": {
        "nom": "Covered call",
        "tendance": "Neutre à légèrement haussier",
        "resume": "Détenir le titre et vendre un call hors de la monnaie.",
        "gain_max": "Prime + (strike − cours d'entrée)",
        "risque_max": "Cours d'entrée − prime, si le titre tombe à zéro",
        "volatilite": "Une baisse de la volatilité implicite est favorable",
        "quand": "Le titre stagne ou monte peu, et l'on accepte de le céder "
                 "au strike.",
        "piege": "Le gain est plafonné, la perte ne l'est pas. On abandonne "
                 "les fortes hausses exactement quand on avait raison.",
    },
    "cash_secured_put": {
        "nom": "Cash-secured put",
        "tendance": "Neutre à haussier",
        "resume": "Vendre un put en immobilisant le montant nécessaire à "
                  "l'achat.",
        "gain_max": "La prime encaissée, rien de plus",
        "risque_max": "(Strike × quantité) − prime, si le titre tombe à zéro",
        "volatilite": "Une baisse de la volatilité implicite est favorable",
        "quand": "On souhaite acquérir le titre plus bas et l'on est payé "
                 "pour attendre.",
        "piege": "On est assigné précisément quand le titre baisse — donc "
                 "quand on le voulait le moins.",
    },
    "collar": {
        "nom": "Collar",
        "tendance": "Protecteur, marché incertain",
        "resume": "Covered call dont la prime finance un put protecteur.",
        "gain_max": "Strike du call − cours + prime nette",
        "risque_max": "Cours − strike du put − prime nette. Perte bornée.",
        "volatilite": "Effet limité : les deux jambes se compensent",
        "quand": "Position importante à protéger, sans vouloir vendre.",
        "piege": "La fourchette est étroite : on renonce au potentiel autant "
                 "qu'on se protège.",
    },
    "iron_condor": {
        "nom": "Iron condor",
        "tendance": "Strictement neutre",
        "resume": "Vendre un call et un put hors de la monnaie, en acheter de "
                  "plus éloignés pour borner le risque.",
        "gain_max": "Le crédit net encaissé",
        "risque_max": "Largeur des ailes − crédit net. Strictement borné.",
        "volatilite": "Une chute de la volatilité est très favorable",
        "quand": "L'actif reste dans une fourchette jusqu'à l'échéance.",
        "piege": "Beaucoup de petits gains, quelques pertes lourdes. "
                 "L'espérance est souvent voisine de zéro après frais.",
    },
}


# ==========================================================================
# Chaîne d'options
# ==========================================================================

def echeances(ticker: str) -> list[str]:
    """Echeances disponibles. Vide sur la plupart des valeurs europeennes."""
    try:
        return list(yf.Ticker(ticker).options or [])
    except Exception:
        return []


def chaine(ticker: str, echeance: str) -> dict:
    """
    Chaine d'options pour une echeance, nettoyee.

    On ecarte les lignes sans cotation : une option affichee sans acheteur ni
    vendeur n'est pas negociable au prix indique, et l'inclure fausserait
    tous les calculs de rendement.
    """
    try:
        valeur = yf.Ticker(ticker)
        donnees = valeur.option_chain(echeance)
        try:
            spot = float(valeur.fast_info["lastPrice"])
        except Exception:
            spot = float(valeur.history(period="5d")["Close"].iloc[-1])

        def _nettoyer(table, genre):
            if table is None or table.empty:
                return pd.DataFrame()
            t = table.copy()
            t["prix"] = np.where(
                (t["bid"] > 0) & (t["ask"] > 0),
                (t["bid"] + t["ask"]) / 2,
                t["lastPrice"])
            t["ecart_pct"] = np.where(
                t["bid"] > 0, (t["ask"] - t["bid"]) / t["bid"] * 100, np.nan)
            t["moneyness_pct"] = (t["strike"] / spot - 1) * 100
            t["genre"] = genre
            t = t[t["prix"] > 0]
            colonnes = ["strike", "prix", "bid", "ask", "ecart_pct", "volume",
                        "openInterest", "impliedVolatility", "moneyness_pct",
                        "genre"]
            return t[[c for c in colonnes if c in t.columns]]

        jours = max((pd.Timestamp(echeance)
                     - pd.Timestamp.now().normalize()).days, 1)
        return {"spot": spot, "jours": jours, "echeance": echeance,
                "calls": _nettoyer(donnees.calls, "call"),
                "puts": _nettoyer(donnees.puts, "put")}
    except Exception as erreur:
        print(f"Chaîne indisponible pour {ticker} : {type(erreur).__name__}",
              file=sys.stderr)
        return {}


def proche(table: pd.DataFrame, cible: float) -> pd.Series | None:
    """Ligne dont le strike est le plus proche d'une valeur donnee."""
    if table is None or table.empty:
        return None
    return table.iloc[(table["strike"] - cible).abs().argsort().iloc[0]]


# ==========================================================================
# Profils de gain
# ==========================================================================

def profil(strategie: str, spot: float, jambes: dict,
           quantite: int = 100) -> dict:
    """
    Gain a l'echeance en fonction du cours, et mesures associees.

    Le profil est calcule sur une grille de cours allant de la moitie au
    double du cours actuel : assez large pour montrer les deux bornes sans
    ecraser la zone qui compte.
    """
    cours = np.linspace(spot * 0.5, spot * 1.5, 401)

    if strategie == "covered_call":
        strike = jambes["call_strike"]
        prime = jambes["call_prime"]
        gain = (np.minimum(cours, strike) - spot + prime) * quantite
        seuil = spot - prime
        maximum = (strike - spot + prime) * quantite
        minimum = -(spot - prime) * quantite

    elif strategie == "cash_secured_put":
        strike = jambes["put_strike"]
        prime = jambes["put_prime"]
        gain = (prime - np.maximum(strike - cours, 0)) * quantite
        seuil = strike - prime
        maximum = prime * quantite
        minimum = -(strike - prime) * quantite

    elif strategie == "collar":
        strike_call = jambes["call_strike"]
        strike_put = jambes["put_strike"]
        nette = jambes["call_prime"] - jambes["put_prime"]
        gain = (np.clip(cours, strike_put, strike_call) - spot + nette) * quantite
        seuil = spot - nette
        maximum = (strike_call - spot + nette) * quantite
        minimum = (strike_put - spot + nette) * quantite

    elif strategie == "iron_condor":
        pv, pa = jambes["put_vendu"], jambes["put_achete"]
        cv, ca = jambes["call_vendu"], jambes["call_achete"]
        credit = (jambes["put_vendu_prime"] - jambes["put_achete_prime"]
                  + jambes["call_vendu_prime"] - jambes["call_achete_prime"])
        gain = (credit
                - np.maximum(pv - cours, 0) + np.maximum(pa - cours, 0)
                - np.maximum(cours - cv, 0) + np.maximum(cours - ca, 0)) * quantite
        maximum = credit * quantite
        aile = max(pv - pa, ca - cv)
        minimum = -(aile - credit) * quantite
        seuil = (pv - credit, cv + credit)
    else:
        return {}

    return {"cours": cours, "gain": gain, "seuil": seuil,
            "gain_max": float(maximum), "perte_max": float(minimum),
            "quantite": quantite}


def rendements(strategie: str, spot: float, jambes: dict, jours: int,
               quantite: int = 100) -> dict:
    """
    Rendement de la prime, brut puis annualise.

    L'annualisation est fournie parce qu'elle sert a comparer des echeances,
    mais elle suppose de reproduire l'operation a l'identique toute l'annee —
    hypothese que la volatilite ne garantit jamais. Elle est donc a lire
    comme une base de comparaison, pas comme un rendement attendu.
    """
    jours = max(jours, 1)

    if strategie == "covered_call":
        prime = jambes["call_prime"]
        capital = spot
        rendement_statique = prime / capital * 100
        rendement_assigne = ((jambes["call_strike"] - spot + prime)
                             / capital * 100)
        sortie = {"rendement_statique_pct": rendement_statique,
                  "rendement_si_assigne_pct": rendement_assigne}
    elif strategie == "cash_secured_put":
        prime = jambes["put_prime"]
        capital = jambes["put_strike"]
        sortie = {"rendement_statique_pct": prime / capital * 100}
    elif strategie == "collar":
        prime = jambes["call_prime"] - jambes["put_prime"]
        capital = spot
        sortie = {"cout_net_pct": -prime / capital * 100}
    else:
        credit = (jambes["put_vendu_prime"] - jambes["put_achete_prime"]
                  + jambes["call_vendu_prime"] - jambes["call_achete_prime"])
        aile = max(jambes["put_vendu"] - jambes["put_achete"],
                   jambes["call_achete"] - jambes["call_vendu"])
        capital = aile - credit
        prime = credit
        sortie = {"rendement_sur_risque_pct": (credit / capital * 100
                                               if capital > 0 else np.nan)}

    base = prime / capital * 100 if capital > 0 else np.nan
    sortie["prime_encaissee"] = prime * quantite
    sortie["capital_immobilise"] = capital * quantite
    sortie["rendement_periode_pct"] = base
    sortie["rendement_annualise_pct"] = base * 365 / jours
    sortie["jours"] = jours
    return sortie


def probabilite_assignation(ligne: pd.Series) -> float:
    """
    Probabilite approchee que l'option finisse dans la monnaie.

    Le delta d'une option en est une approximation courante et suffisante.
    Ce n'est pas une prevision : c'est ce que le marche price, et le marche
    se trompe regulierement.
    """
    if ligne is None:
        return np.nan
    for champ in ("delta", "Delta"):
        if champ in ligne.index and pd.notna(ligne[champ]):
            return abs(float(ligne[champ])) * 100
    return np.nan


def couverture(spot: float, prime: float) -> float:
    """Baisse absorbee par la prime avant de perdre de l'argent."""
    return prime / spot * 100 if spot > 0 else np.nan
