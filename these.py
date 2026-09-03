"""
Theses d'investissement et journal de decision.

Deux idees reprises d'un cahier des charges plus ambitieux, et ce sont les
deux qui changent reellement quelque chose.

SURVEILLANCE D'INVALIDATION. Une these s'ecrit avant l'achat, avec des
conditions chiffrees de ce qui la rendrait fausse. Le systeme les verifie a
chaque publication et signale l'ecart AU MOMENT OU IL APPARAIT, en rappelant
le texte exact ecrit a l'achat. Sans ce rappel, la these se reformule
inconsciemment pour coller aux faits — c'est le mecanisme par lequel on
conserve indefiniment une position perdante en croyant rester coherent.

JOURNAL DE DECISION. Toute decision est consignee, y compris les non-actions.
Une occasion etudiee puis rejetee compte autant qu'une position prise : sans
elle, l'echantillon d'apprentissage est ampute de moitie et biaise, puisqu'on
ne se souvient que de ce qu'on a fait.

Le post-mortem separe la qualite du processus de celle du resultat. Un gain
obtenu pour une raison absente de la these est un bon resultat et un mauvais
processus : le confondre avec une reussite garantit de repeter l'erreur.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

STATUTS = ("intacte", "sous surveillance", "invalidée", "clôturée")

COLONNES = {
    "ticker": ["ticker", "symbole", "valeur", "code"],
    "raison": ["raison", "these", "pourquoi", "argument"],
    "hypotheses": ["hypotheses", "conditions", "criteres", "indicateurs"],
    "invalidation": ["invalidation", "sortie", "vendre si", "stop"],
    "horizon": ["horizon", "duree", "echeance"],
    "conviction": ["conviction", "confiance", "note"],
    "cible_bear": ["bear", "pessimiste", "bas"],
    "cible_base": ["base", "central", "median"],
    "cible_bull": ["bull", "optimiste", "haut"],
    "proba_bear": ["proba bear", "probabilite bear", "p bear"],
    "proba_base": ["proba base", "probabilite base", "p base"],
    "proba_bull": ["proba bull", "probabilite bull", "p bull"],
    "date": ["date", "maj", "mise a jour", "ecrite le"],
}

# Indicateurs reconnus dans les hypothèses écrites en langage courant
INDICATEURS = {
    "marge nette": "marge_nette_pct",
    "marge brute": "marge_brute_pct",
    "marge op": "marge_op_pct",
    "marge operationnelle": "marge_op_pct",
    "marge ebitda": "marge_ebitda_pct",
    "croissance": "croissance_ca_pct",
    "chiffre d'affaires": "croissance_ca_pct",
    "levier": "levier",
    "dette": "levier",
    "roe": "roe_pct",
    "roic": "roic_pct",
    "per": "per",
    "cours": "cours",
    "conversion": "conversion_cash",
    "fcf": "fcf",
}


def _propre(texte) -> str:
    return str(texte).lower().strip().translate(
        str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))


def _nombre(valeur) -> float:
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return np.nan
    texte = str(valeur).strip().replace(",", ".").replace("%", "").replace(" ", "")
    try:
        return float(texte)
    except ValueError:
        return np.nan


# ==========================================================================
# Lecture
# ==========================================================================

def lire(url: str) -> pd.DataFrame:
    """Charge les theses depuis une feuille Google publiee."""
    import veille as v

    brut = None
    for adresse in v.url_csv(url):
        try:
            essai = pd.read_csv(adresse)
            if not essai.empty:
                brut = essai
                break
        except Exception:
            continue
    if brut is None:
        raise ValueError("Feuille de thèses inaccessible.")

    correspondance = {}
    for colonne in brut.columns:
        nom = _propre(colonne)
        for cible, variantes in COLONNES.items():
            if cible in correspondance.values():
                continue
            if nom in variantes or any(x in nom for x in variantes):
                correspondance[colonne] = cible
                break
    brut = brut.rename(columns=correspondance)

    if "ticker" not in brut.columns:
        raise ValueError("Colonne Ticker introuvable dans la feuille de thèses.")

    lignes = []
    for _, ligne in brut.iterrows():
        ticker = str(ligne.get("ticker", "")).strip().upper()
        if not ticker or ticker in ("NAN", "TICKER"):
            continue
        entree = {"ticker": ticker}
        for champ in ("raison", "hypotheses", "invalidation", "horizon"):
            entree[champ] = str(ligne.get(champ, "") or "").strip()
        entree["conviction"] = _nombre(ligne.get("conviction"))
        for champ in ("cible_bear", "cible_base", "cible_bull",
                      "proba_bear", "proba_base", "proba_bull"):
            entree[champ] = _nombre(ligne.get(champ))
        date = pd.to_datetime(ligne.get("date"), dayfirst=True, errors="coerce")
        entree["date"] = None if pd.isna(date) else date
        lignes.append(entree)

    return pd.DataFrame(lignes).set_index("ticker") if lignes else pd.DataFrame()


# ==========================================================================
# Espérance de rentabilité
# ==========================================================================

def esperance(these: dict, cours: float) -> dict:
    """
    Rendement attendu, pondere par les probabilites des scenarios.

    Une these dont l'esperance est inferieure a celle d'un indice ne justifie
    pas le travail de selection ni le risque specifique qu'elle fait porter.
    """
    if not cours or cours <= 0:
        return {}

    scenarios = []
    for nom, cle_cible, cle_proba in [
            ("bear", "cible_bear", "proba_bear"),
            ("base", "cible_base", "proba_base"),
            ("bull", "cible_bull", "proba_bull")]:
        cible = these.get(cle_cible)
        proba = these.get(cle_proba)
        if cible is None or np.isnan(cible) or cible <= 0:
            continue
        if proba is None or np.isnan(proba):
            continue
        scenarios.append({"nom": nom, "cible": float(cible),
                          "proba": float(proba) / (100 if proba > 1 else 1),
                          "rendement_pct": (float(cible) / cours - 1) * 100})

    if not scenarios:
        return {}

    somme = sum(s["proba"] for s in scenarios)
    if somme <= 0:
        return {}
    if abs(somme - 1) > 0.02:
        for s in scenarios:
            s["proba"] /= somme          # normalisation si la somme n'est pas 1

    attendu = sum(s["proba"] * s["rendement_pct"] for s in scenarios)
    baisse = sum(s["proba"] * min(s["rendement_pct"], 0) for s in scenarios)
    hausse = sum(s["proba"] * max(s["rendement_pct"], 0) for s in scenarios)

    return {"scenarios": scenarios,
            "esperance_pct": attendu,
            "gain_espere_pct": hausse,
            "perte_esperee_pct": baisse,
            "asymetrie": abs(hausse / baisse) if baisse < 0 else np.inf,
            "somme_probas": somme}


# ==========================================================================
# Analyse et vérification des hypothèses
# ==========================================================================

def analyser_conditions(texte: str) -> list[dict]:
    """
    Extrait les seuils verifiables d'un texte ecrit en langage courant.

    Reconnait « marge nette > 42 % », « levier < 2,5x », « croissance > 8 % ».
    Une condition non chiffree n'est pas verifiable automatiquement, et c'est
    signale plutot que devine.
    """
    if not texte or not str(texte).strip():
        return []

    conditions = []
    motif = re.compile(r"([a-zéèêàûôç'\s]+?)\s*([<>≤≥]=?)\s*(-?\d+(?:[.,]\d+)?)\s*(%|x)?")
    for correspondance in motif.finditer(_propre(texte).replace(",", ".")):
        libelle = correspondance.group(1).strip()
        champ = next((v for k, v in INDICATEURS.items() if k in libelle), None)
        if champ:
            conditions.append({
                "libelle": libelle,
                "champ": champ,
                "operateur": correspondance.group(2)[0],
                "seuil": float(correspondance.group(3)),
                "unite": correspondance.group(4) or ""})
    return conditions


def verifier(these: dict, mesures: dict) -> dict:
    """
    Confronte chaque condition aux chiffres publies.

    Le statut resultant : intacte si tout tient, sous surveillance si une
    hypothese est demente, invalidee si une condition de sortie explicite est
    remplie. La distinction compte : une hypothese ratee appelle une revision,
    une condition d'invalidation appelle une decision.
    """
    hypotheses = analyser_conditions(these.get("hypotheses", ""))
    invalidations = analyser_conditions(these.get("invalidation", ""))

    def _evaluer(conditions):
        resultats = []
        for condition in conditions:
            valeur = mesures.get(condition["champ"])
            if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
                resultats.append({**condition, "valeur": None,
                                  "statut": "non vérifiable"})
                continue
            valeur = float(valeur)
            respectee = (valeur >= condition["seuil"]
                         if condition["operateur"] == ">"
                         else valeur <= condition["seuil"])
            resultats.append({**condition, "valeur": valeur,
                              "statut": "vérifiée" if respectee else "démentie"})
        return resultats

    # Les deux registres sont évalués séparément : le sens de « vérifiée »
    # diffère selon qu'il s'agit d'une hypothèse ou d'une invalidation.

    verdict_hypotheses = _evaluer(hypotheses)
    verdict_invalidations = _evaluer(invalidations)

    # Attention à la symétrie des deux registres, qui est inverse.
    #
    # Une HYPOTHÈSE décrit ce qui doit rester vrai : « marge > 24 % ». Elle est
    # démentie quand elle cesse d'être vérifiée.
    #
    # Une condition d'INVALIDATION décrit ce qui ne doit pas arriver :
    # « marge < 20 % ». Elle se déclenche quand elle devient vraie.
    #
    # Confondre les deux inverse le verdict, ce qui est le pire défaut
    # possible pour cette fonction.
    declenchees = [v for v in verdict_invalidations if v["statut"] == "vérifiée"]
    dementies = [v for v in verdict_hypotheses if v["statut"] == "démentie"]

    verifiables = [v for v in verdict_hypotheses + verdict_invalidations
                   if v["statut"] != "non vérifiable"]

    if not verifiables:
        statut = "non vérifiable"
    elif declenchees:
        statut = "invalidée"
    elif dementies:
        statut = "sous surveillance"
    else:
        statut = "intacte"

    anciennete = None
    if these.get("date") is not None:
        try:
            anciennete = (datetime.now() - pd.Timestamp(these["date"]).to_pydatetime()).days
        except Exception:
            anciennete = None

    return {"statut": statut,
            "hypotheses": verdict_hypotheses,
            "invalidations": verdict_invalidations,
            "nombre_dementies": len(dementies),
            "nombre_declenchees": len(declenchees),
            "non_verifiables": sum(1 for v in verdict_hypotheses + verdict_invalidations
                                   if v["statut"] == "non vérifiable"),
            "anciennete_jours": anciennete,
            "a_revoir": anciennete is not None and anciennete > 180}


# ==========================================================================
# Journal de décision
# ==========================================================================

ACTIONS = ("ACHAT", "VENTE", "RENFORCEMENT", "ALLÈGEMENT", "CONSERVATION",
           "REJET")


def statistiques(decisions: pd.DataFrame) -> dict:
    """
    Statistiques de decision, dont la calibration de la conviction.

    La colonne la plus instructive est la performance par niveau de conviction
    declare : si les convictions a 5 ne font pas mieux que celles a 2, le
    niveau de conviction ne mesure rien d'exploitable.
    """
    if decisions.empty or "resultat_pct" not in decisions.columns:
        return {}

    closes = decisions[decisions["resultat_pct"].notna()]
    if closes.empty:
        return {}

    gains = closes[closes["resultat_pct"] > 0]["resultat_pct"]
    pertes = closes[closes["resultat_pct"] <= 0]["resultat_pct"]

    resultat = {
        "decisions": int(len(closes)),
        "taux_reussite_pct": float(len(gains) / len(closes) * 100),
        "gain_moyen_pct": float(gains.mean()) if len(gains) else 0.0,
        "perte_moyenne_pct": float(pertes.mean()) if len(pertes) else 0.0,
    }
    resultat["esperance_pct"] = float(closes["resultat_pct"].mean())

    if "conviction" in closes.columns:
        par_conviction = (closes.groupby("conviction")["resultat_pct"]
                          .agg(["mean", "count"]).round(2))
        resultat["par_conviction"] = par_conviction.to_dict("index")
        if len(par_conviction) >= 3:
            correlation = closes["conviction"].corr(closes["resultat_pct"])
            resultat["calibration"] = float(correlation)
            resultat["lecture_calibration"] = (
                "conviction prédictive : les fortes convictions font mieux"
                if correlation > 0.3 else
                "conviction inversement liée au résultat — signal d'excès de confiance"
                if correlation < -0.3 else
                "la conviction déclarée ne prédit pas le résultat")

    if "action" in closes.columns:
        rejets = closes[closes["action"] == "REJET"]
        if not rejets.empty:
            resultat["rejets_suivis"] = int(len(rejets))
            resultat["performance_rejets_pct"] = float(rejets["resultat_pct"].mean())

    return resultat


def matrice_processus(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice 2×2 : qualite du processus contre qualite du resultat.

    Un gain obtenu pour une raison absente de la these appartient a la case
    « bon resultat, mauvais processus ». C'est la case la plus dangereuse :
    elle valide une methode qui n'a pas fonctionne.
    """
    if decisions.empty or "processus" not in decisions.columns:
        return pd.DataFrame()

    closes = decisions[decisions["resultat_pct"].notna()].copy()
    if closes.empty:
        return pd.DataFrame()

    closes["resultat"] = np.where(closes["resultat_pct"] > 0, "favorable",
                                  "défavorable")
    table = pd.crosstab(closes["processus"], closes["resultat"])
    return table


MODELE_THESE = pd.DataFrame({
    "Ticker": ["AAPL", "AIR.PA"],
    "Raison": ["Écosystème verrouillé, marge de service en hausse",
               "Carnet de commandes couvrant 8 ans de production"],
    "Hypothèses": ["marge nette > 24 % et croissance > 5 %",
                   "marge op > 8 % et levier < 2x"],
    "Invalidation": ["marge nette < 20 % ou croissance < 0 %",
                     "marge op < 5 %"],
    "Horizon": ["3 ans", "5 ans"],
    "Conviction": [4, 3],
    "Bear": [180, 130],
    "Base": [280, 210],
    "Bull": [340, 260],
    "Proba bear": [25, 30],
    "Proba base": [50, 50],
    "Proba bull": [25, 20],
    "Date": ["01/06/2026", "15/07/2026"],
})

MODELE_DECISION = pd.DataFrame({
    "Date": ["20/03/2024", "12/06/2024", "05/09/2024"],
    "Ticker": ["AAPL", "MU", "NVDA"],
    "Action": ["ACHAT", "REJET", "ALLÈGEMENT"],
    "Conviction": [4, 2, 3],
    "Raison": ["Marge de service en accélération",
               "Cyclique en haut de cycle, pas de marge de sécurité",
               "Poids devenu excessif après hausse"],
    "Alternatives écartées": ["MSFT, renforcement AIR.PA", "Attendre le creux",
                              "Tout conserver"],
    "Contexte marché": ["Indice au-dessus de sa MM200", "Volatilité élevée",
                        "Plus haut historique"],
    "Processus": ["", "", ""],
    "Résultat %": ["", "", ""],
})
