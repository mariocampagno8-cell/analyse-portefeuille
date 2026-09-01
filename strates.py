"""
Structure de l'univers en trois strates.

  A — Portefeuille : lignes detenues. Tout ce qui touche la these.
  B — Candidats    : fiche de these etablie et prix d'entree fixe.
  C — Veille       : le solde. Prix cible et evenement exceptionnel seulement.

La regle d'entree en B est volontairement dure : sans prix d'entree, la valeur
reste en C. C'est ce filtre qui protege le canal — il empeche la strate B de
devenir un fourre-tout, ce qui arrive systematiquement sans contrainte
mecanique.

Les seuils inscrits dans la feuille deviennent les declencheurs d'alerte.
C'est ce qui transforme un flux d'information en outil de decision.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Colonnes reconnues, avec leurs synonymes tolérés
COLONNES = {
    "ticker": ["ticker", "symbole", "symbol", "code", "valeur"],
    "strate": ["strate", "strata", "categorie", "type", "niveau"],
    "quantite": ["quantite", "quantity", "qte", "nombre", "titres"],
    "pru": ["pru", "prix d'achat", "prix dachat", "prix de revient", "cout"],
    "prix_entree": ["prix entree", "prix d'entree", "prix cible", "cible",
                    "target", "prix achat vise"],
    "prix_sortie": ["prix sortie", "prix de sortie", "seuil vente", "stop"],
    "these_maj": ["these", "date these", "maj these", "mise a jour"],
    "concurrents": ["concurrents", "peers", "comparables"],
    "note": ["note", "commentaire", "remarque", "kpi"],
}

STRATES_VALIDES = {"A", "B", "C"}
DELAI_OBSOLESCENCE_B = 180        # jours avant rétrogradation automatique
DELAI_OBSOLESCENCE_C = 365


def _sans_accents(texte: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return str(texte).lower().strip().translate(table)


def _nombre(valeur) -> float:
    """Convertit une saisie en nombre, en tolerant les formats francais."""
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return float(valeur) if np.isfinite(valeur) else np.nan
    if valeur is None:
        return np.nan
    texte = (str(valeur).strip().replace("\u202f", "").replace("\xa0", "")
             .replace(" ", "").replace("€", "").replace("$", ""))
    if not texte:
        return np.nan
    if "," in texte and "." in texte:
        texte = (texte.replace(".", "").replace(",", ".")
                 if texte.rindex(",") > texte.rindex(".")
                 else texte.replace(",", ""))
    elif "," in texte:
        texte = texte.replace(",", ".")
    try:
        return float(texte)
    except ValueError:
        return np.nan


def _date(valeur):
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return None
    try:
        resultat = pd.to_datetime(valeur, dayfirst=True, errors="coerce")
        return None if pd.isna(resultat) else resultat
    except Exception:
        return None


def normaliser(df: pd.DataFrame) -> pd.DataFrame:
    """Fait correspondre les intitules de la feuille aux champs attendus."""
    correspondance = {}
    for colonne in df.columns:
        propre = _sans_accents(colonne)
        for cible, variantes in COLONNES.items():
            if cible in correspondance.values():
                continue
            if propre in variantes or any(v in propre for v in variantes):
                correspondance[colonne] = cible
                break
    return df.rename(columns=correspondance)


def lire(url: str) -> pd.DataFrame:
    """
    Charge l'univers depuis une feuille Google publiee.

    Une seule feuille pour les trois strates : c'est ce qui rend la promotion
    d'une valeur de C vers B triviale — on change une lettre.
    """
    import feuille as fe

    df = None
    for adresse in fe.urls_candidates(url):
        try:
            essai = pd.read_csv(adresse)
            if not essai.empty:
                df = essai
                break
        except Exception:
            continue
    if df is None or df.empty:
        raise ValueError(
            "Feuille d'univers inaccessible. Vérifie qu'elle est publiée au "
            "format CSV (Fichier → Partager → Publier sur le web)."
        )

    df = normaliser(df)
    if "ticker" not in df.columns:
        raise ValueError(
            f"Colonne Ticker introuvable. La feuille contient : "
            f"{', '.join(map(str, df.columns))}."
        )

    lignes = []
    for _, brut in df.iterrows():
        ticker = str(brut.get("ticker", "")).strip().upper()
        if not ticker or ticker in ("NAN", "TICKER"):
            continue

        strate = str(brut.get("strate", "C")).strip().upper()[:1]
        if strate not in STRATES_VALIDES:
            strate = "C"

        lignes.append({
            "ticker": ticker,
            "strate": strate,
            "quantite": _nombre(brut.get("quantite")),
            "pru": _nombre(brut.get("pru")),
            "prix_entree": _nombre(brut.get("prix_entree")),
            "prix_sortie": _nombre(brut.get("prix_sortie")),
            "these_maj": _date(brut.get("these_maj")),
            "concurrents": str(brut.get("concurrents", "") or "").strip(),
            "note": str(brut.get("note", "") or "").strip(),
        })

    univers = pd.DataFrame(lignes).drop_duplicates(subset="ticker")
    return appliquer_regles(univers)


def appliquer_regles(univers: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les regles structurelles du cahier des charges.

    Deux corrections automatiques. Une valeur avec une quantite detenue est
    en strate A, quoi qu'indique la feuille — on ne peut pas etre candidat a
    ce qu'on detient deja. Et une valeur declaree B sans prix d'entree
    redescend en C : c'est le filtre dur qui protege le canal.
    """
    if univers.empty:
        return univers

    univers = univers.copy()
    univers["strate_declaree"] = univers["strate"]
    univers["motif_reclassement"] = ""

    detenue = univers["quantite"].fillna(0) > 0
    a_corriger = detenue & (univers["strate"] != "A")
    univers.loc[a_corriger, "motif_reclassement"] = "position détenue"
    univers.loc[detenue, "strate"] = "A"

    sans_prix = (univers["strate"] == "B") & univers["prix_entree"].isna()
    univers.loc[sans_prix, "motif_reclassement"] = "prix d'entrée absent"
    univers.loc[sans_prix, "strate"] = "C"

    return univers


def revue_trimestrielle(univers: pd.DataFrame) -> dict:
    """
    Propose les mouvements de strate, sans les appliquer.

    Le systeme propose, l'utilisateur tranche : une retrogradation automatique
    ferait disparaitre une valeur du radar sans decision consciente.
    """
    maintenant = datetime.now()
    propositions = {"retrograder_B_vers_C": [], "sortir_de_C": [],
                    "these_a_mettre_a_jour": []}
    if univers.empty:
        return propositions

    for _, ligne in univers.iterrows():
        # Une date absente peut arriver comme None, NaT ou NaN selon la
        # facon dont pandas a construit la colonne.
        maj = ligne.get("these_maj")
        anciennete = None
        if maj is not None and not (isinstance(maj, float) and np.isnan(maj)):
            try:
                horodatage = pd.Timestamp(maj)
                if not pd.isna(horodatage):
                    anciennete = (maintenant - horodatage.to_pydatetime()).days
            except Exception:
                anciennete = None

        if ligne["strate"] == "B":
            if anciennete is None or anciennete > DELAI_OBSOLESCENCE_B:
                propositions["retrograder_B_vers_C"].append({
                    "ticker": ligne["ticker"],
                    "motif": ("thèse jamais datée" if anciennete is None
                              else f"thèse vieille de {anciennete} jours")})
        elif ligne["strate"] == "C":
            if anciennete is not None and anciennete > DELAI_OBSOLESCENCE_C:
                propositions["sortir_de_C"].append({
                    "ticker": ligne["ticker"],
                    "motif": f"non consultée depuis {anciennete} jours"})

        if ligne["strate"] == "A" and (anciennete is None
                                       or anciennete > DELAI_OBSOLESCENCE_B):
            propositions["these_a_mettre_a_jour"].append(ligne["ticker"])

    return propositions


# ==========================================================================
# Accès
# ==========================================================================

def tickers(univers: pd.DataFrame, strate: str | None = None) -> list[str]:
    if univers.empty:
        return []
    if strate:
        return univers[univers["strate"] == strate]["ticker"].tolist()
    return univers["ticker"].tolist()


def concurrents(univers: pd.DataFrame) -> dict[str, list[str]]:
    """
    Concurrents declares pour chaque ligne detenue.

    Ils ne forment pas une strate : ce sont un champ de la fiche. Un
    avertissement chez un pair arrive avant le votre et laisse le temps d'agir.
    """
    resultat = {}
    if univers.empty:
        return resultat
    for _, ligne in univers[univers["strate"] == "A"].iterrows():
        brut = ligne.get("concurrents", "")
        if not brut:
            continue
        liste = [t.strip().upper() for t in
                 str(brut).replace(";", ",").split(",") if t.strip()]
        if liste:
            resultat[ligne["ticker"]] = liste
    return resultat


def seuils(univers: pd.DataFrame) -> dict[str, dict]:
    """Seuils de prix par valeur, pour les alertes."""
    resultat = {}
    if univers.empty:
        return resultat
    for _, ligne in univers.iterrows():
        entrees = {}
        for cle in ("prix_entree", "prix_sortie", "pru"):
            valeur = ligne.get(cle)
            if valeur is not None and np.isfinite(valeur):
                entrees[cle] = float(valeur)
        if entrees:
            entrees["strate"] = ligne["strate"]
            resultat[ligne["ticker"]] = entrees
    return resultat


def resume(univers: pd.DataFrame) -> dict:
    """Photographie de l'univers, pour contrôle."""
    if univers.empty:
        return {"total": 0}
    compte = univers["strate"].value_counts().to_dict()
    reclassees = univers[univers["motif_reclassement"] != ""]
    return {
        "total": int(len(univers)),
        "A": int(compte.get("A", 0)),
        "B": int(compte.get("B", 0)),
        "C": int(compte.get("C", 0)),
        "reclassées": [
            f"{r['ticker']} : {r['strate_declaree']} → {r['strate']} "
            f"({r['motif_reclassement']})"
            for _, r in reclassees.iterrows()],
        "avec_prix_cible": int(univers["prix_entree"].notna().sum()),
        "avec_concurrents": int((univers["concurrents"] != "").sum()),
    }


MODELE = pd.DataFrame({
    "Ticker": ["AAPL", "AIR.PA", "ASML.AS", "NVDA", "TSM"],
    "Strate": ["A", "A", "B", "B", "C"],
    "Quantité": [12, 25, "", "", ""],
    "PRU": [165.20, 128.40, "", "", ""],
    "Prix entrée": ["", "", 620, 95, 180],
    "Prix sortie": [320, 180, "", "", ""],
    "Concurrents": ["MSFT, GOOGL", "BA, SAF.PA", "", "", ""],
    "Thèse MAJ": ["01/06/2026", "15/07/2026", "20/08/2026", "", ""],
    "Note": ["Marge brute > 45 %", "Carnet > 8 000 avions", "", "", ""],
})
