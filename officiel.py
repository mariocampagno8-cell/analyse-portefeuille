"""
Sources officielles : EDGAR et FRED.

EDGAR est la base publique de la SEC. Elle diffuse les documents deposes par
les societes cotees aux Etats-Unis, gratuitement, avec un code de formulaire
qui rend la classification par materialite MECANIQUE plutot qu'interpretative.
Un 8-K item 5.02 est un changement de dirigeant, sans ambiguite possible.

C'est la difference decisive avec un flux de presse : ici, on lit la source,
pas le commentaire sur la source.

Limite : couverture americaine uniquement. Il n'existe pas d'equivalent
structure en Europe, et scraper les pages « communiques » de chaque societe
casse a chaque refonte de site. Les valeurs francaises et allemandes ne
generent donc aucun communique classe.

Sur la macro : la surprise en ecarts-types suppose un consensus, qui est
payant. On mesure a la place l'ecart a la distribution historique de la
statistique elle-meme, ce qui repond a une question voisine — cette
publication est-elle inhabituelle — avec une donnee verifiable.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# La SEC exige un en-tête d'identification et limite à 10 requêtes/seconde
EN_TETES = {"User-Agent": os.environ.get(
    "EDGAR_CONTACT", "Veille Portefeuille veille@example.com")}
PAUSE_SEC = 0.15


# ==========================================================================
# Classification des formulaires 8-K
# ==========================================================================

# Codes d'item du formulaire 8-K, avec leur niveau de matérialité.
# Cette table est la valeur du module : elle rend la classification
# reproductible au lieu de dépendre d'une lecture de titre.
ITEMS_8K = {
    # --- P1 : appelle une décision
    "1.01": ("P1", "Contrat structurant conclu"),
    "1.02": ("P1", "Contrat structurant résilié"),
    "1.03": ("P1", "Procédure collective"),
    "2.01": ("P1", "Acquisition ou cession d'actifs"),
    "2.04": ("P1", "Exigibilité anticipée de dette"),
    "2.06": ("P1", "Dépréciation d'actifs"),
    "4.01": ("P1", "Changement de commissaire aux comptes"),
    "4.02": ("P1", "Comptes antérieurs non fiables"),
    "5.02": ("P1", "Changement de dirigeant ou d'administrateur"),
    "7.01": ("P2", "Communication réglementaire"),
    "8.01": ("P2", "Autre événement significatif"),
    # --- Résultats
    "2.02": ("P1", "Résultats publiés"),
    # --- P2 : à savoir, sans urgence
    "1.05": ("P2", "Incident de cybersécurité"),
    "2.03": ("P2", "Engagement financier significatif"),
    "2.05": ("P2", "Plan de restructuration"),
    "3.01": ("P2", "Non-conformité aux règles de cotation"),
    "3.02": ("P2", "Émission de titres non enregistrée"),
    "3.03": ("P2", "Modification des droits des actionnaires"),
    "5.01": ("P2", "Changement de contrôle"),
    "5.03": ("P2", "Modification des statuts"),
    "5.07": ("P3", "Résultats du vote en assemblée"),
    # --- P3 : digest
    "5.05": ("P3", "Modification du code de conduite"),
    "5.08": ("P3", "Nominations d'administrateurs"),
    "9.01": ("P3", "Pièces jointes"),
}

FORMULAIRES = {
    "8-K": "Événement significatif",
    "10-Q": "Rapport trimestriel",
    "10-K": "Rapport annuel",
    "4": "Transaction de dirigeant",
    "SC 13D": "Franchissement de seuil (intention active)",
    "SC 13G": "Franchissement de seuil (passif)",
    "DEF 14A": "Convocation d'assemblée",
}


def _appel(url: str, params: dict | None = None):
    """Appel EDGAR, avec la pause imposee par la SEC."""
    try:
        time.sleep(PAUSE_SEC)
        reponse = requests.get(url, headers=EN_TETES, params=params, timeout=20)
        if not reponse.ok:
            return None
        return reponse.json()
    except Exception:
        return None


_CACHE_CIK: dict[str, str] = {}


def cik(ticker: str) -> str | None:
    """
    Identifiant SEC d'une societe, a partir de son ticker.

    Les valeurs non americaines n'en ont pas : la fonction renvoie None, et
    l'appelant sait que la source officielle n'est pas disponible.
    """
    if not _CACHE_CIK:
        donnees = _appel("https://www.sec.gov/files/company_tickers.json")
        if not donnees:
            return None
        for entree in donnees.values():
            _CACHE_CIK[str(entree["ticker"]).upper()] = str(
                entree["cik_str"]).zfill(10)
    return _CACHE_CIK.get(ticker.upper())


def depots(ticker: str, jours: int = 3) -> list[dict]:
    """
    Documents deposes recemment, classes par materialite.

    Le code d'item du 8-K determine la priorite : c'est ce qui rend la
    classification reproductible plutot qu'interpretative.
    """
    identifiant = cik(ticker)
    if not identifiant:
        return []

    donnees = _appel(
        f"https://data.sec.gov/submissions/CIK{identifiant}.json")
    if not donnees:
        return []

    recents = donnees.get("filings", {}).get("recent", {})
    if not recents:
        return []

    limite = datetime.now().date() - timedelta(days=jours)
    sortie = []

    for i, formulaire in enumerate(recents.get("form", [])):
        try:
            date = datetime.strptime(
                recents["filingDate"][i], "%Y-%m-%d").date()
        except (ValueError, KeyError, IndexError):
            continue
        if date < limite:
            break

        items = recents.get("items", [""] * len(recents["form"]))[i] or ""
        accession = recents["accessionNumber"][i].replace("-", "")
        document = recents.get("primaryDocument", [""] * len(recents["form"]))[i]
        lien = (f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(identifiant)}/{accession}/{document}")

        if formulaire == "8-K" and items:
            for item in [x.strip() for x in items.split(",") if x.strip()]:
                priorite, libelle = ITEMS_8K.get(item, ("P3", f"Item {item}"))
                sortie.append({
                    "ticker": ticker, "date": date, "formulaire": "8-K",
                    "item": item, "priorite": priorite, "libelle": libelle,
                    "lien": lien})
        elif formulaire in FORMULAIRES:
            priorite = {"10-Q": "P2", "10-K": "P2", "4": "P2",
                        "SC 13D": "P1", "SC 13G": "P2"}.get(formulaire, "P3")
            sortie.append({
                "ticker": ticker, "date": date, "formulaire": formulaire,
                "item": "", "priorite": priorite,
                "libelle": FORMULAIRES[formulaire], "lien": lien})

    return sortie


# ==========================================================================
# Transactions de dirigeants
# ==========================================================================

def transactions_dirigeants(ticker: str, jours: int = 7,
                            seuil_euros: float = 100_000) -> list[dict]:
    """
    Formulaires 4 recents : achats et ventes des dirigeants.

    Signal reel, gratuit, et systematiquement sous-exploite. Un achat de
    dirigeant est plus informatif qu'une vente : les ventes obeissent souvent
    a des plans programmes ou a des besoins personnels, les achats a une
    seule motivation.

    EDGAR ne structure pas les montants dans l'index des depots : on signale
    l'existence de la transaction et on renvoie au document, sans pretendre
    en connaitre le montant.
    """
    depots_recents = depots(ticker, jours)
    return [
        {**d, "priorite": "P2",
         "libelle": "Transaction de dirigeant (formulaire 4)",
         "note": "Montant non structuré par EDGAR — voir le document."}
        for d in depots_recents if d["formulaire"] == "4"]


# ==========================================================================
# Macroéconomie
# ==========================================================================

SERIES_MACRO = {
    "CPIAUCSL": ("Inflation américaine (CPI)", "P1", True),
    "CPILFESL": ("Inflation sous-jacente US", "P1", True),
    "PAYEMS": ("Emploi américain (NFP)", "P1", True),
    "UNRATE": ("Chômage américain", "P1", False),
    "FEDFUNDS": ("Taux directeur Fed", "P1", False),
    "PCEPILFE": ("PCE sous-jacent", "P1", True),
    "GDPC1": ("PIB américain", "P3", True),
    "INDPRO": ("Production industrielle US", "P3", True),
    "UMCSENT": ("Confiance des ménages US", "P3", False),
    "T10Y2Y": ("Pente 10 ans − 2 ans", "P3", False),
    "BAMLH0A0HYM2": ("Spread haut rendement", "P3", False),
}


def serie_fred(code: str) -> pd.Series:
    """Serie FRED, en acces direct sans cle."""
    try:
        df = pd.read_csv(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}",
            parse_dates=[0], index_col=0)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


def publications_macro(jours: int = 2) -> list[dict]:
    """
    Statistiques publiees recemment, avec leur ecart a l'ordinaire.

    Faute de consensus macro gratuit, la surprise est mesuree contre la
    DISTRIBUTION HISTORIQUE de la statistique : combien d'ecarts-types
    separe cette publication de ses variations habituelles. Ce n'est pas la
    surprise au sens des marches, et c'est signale comme tel.
    """
    sortie = []
    limite = datetime.now().date() - timedelta(days=jours)

    for code, (libelle, priorite, en_variation) in SERIES_MACRO.items():
        serie = serie_fred(code)
        if len(serie) < 30:
            continue
        derniere_date = serie.index[-1].date()
        if derniere_date < limite:
            continue

        valeur = float(serie.iloc[-1])
        precedent = float(serie.iloc[-2])

        if en_variation:
            variations = serie.pct_change().dropna() * 100
            mesure = float(variations.iloc[-1])
            reference = variations.tail(60)
            unite = "%"
        else:
            variations = serie.diff().dropna()
            mesure = float(variations.iloc[-1])
            reference = variations.tail(60)
            unite = "pt"

        ecart_type = float(reference.std(ddof=1))
        moyenne = float(reference.mean())
        z = (mesure - moyenne) / ecart_type if ecart_type > 0 else 0.0

        sortie.append({
            "code": code, "libelle": libelle, "priorite": priorite,
            "date": derniere_date, "valeur": valeur, "precedent": precedent,
            "variation": mesure, "unite": unite,
            "ecarts_types": z,
            "inhabituel": abs(z) >= 2.0,
            "note": ("Écart mesuré contre la distribution historique de la "
                     "série, non contre un consensus d'analystes — "
                     "indisponible gratuitement."),
        })
    return sortie


def macro_a_signaler(publications: list[dict]) -> list[dict]:
    """Ne retient que ce qui merite une notification."""
    return [p for p in publications
            if p["priorite"] == "P1" and p["inhabituel"]]
