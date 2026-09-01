"""
Veille evenementielle sur une liste de valeurs surveillees.

Rupture avec la logique de signaux : ici, pas de calcul, pas d'indicateur.
Uniquement des faits dates — une societe publie ses comptes dans trois jours,
une depeche est parue aujourd'hui. S'il ne se passe rien sur une valeur, elle
ne genere aucun message. Le silence est le comportement par defaut.

Deux contraintes pratiques sur une centaine de valeurs :

  - Yahoo limite le debit. Les appels sont donc paralleles mais brides, avec
    une reprise sur echec. Compter deux a quatre minutes par passage.
  - La couverture des actualites est excellente aux Etats-Unis, inegale en
    Europe et faible ailleurs. Une valeur sans actualite n'est pas forcement
    une valeur sans nouvelle.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

FICHIER_ETAT = Path(__file__).parent / "etat_evenements.json"
MEMOIRE_JOURS = 10
FILS_PARALLELES = 5          # au-delà, Yahoo commence à refuser
PAUSE = 0.15                 # seconde entre deux lancements

# Strict : uniquement les sources primaires. Mettre SOURCES_STRICTES=0 dans
# l'environnement pour elargir aux sources acceptables.
STRICT = os.environ.get("SOURCES_STRICTES", "1") != "0"


# ==========================================================================
# Pertinence des dépêches
# ==========================================================================

# Une valeur suivie génère beaucoup de bruit : notes d'analystes recyclées,
# listes automatiques, contenus promotionnels. On ne retient que ce qui
# émane de la société ou modifie ce qu'on sait d'elle.
MOTS_FORTS = [
    "results", "earnings", "revenue", "guidance", "outlook", "forecast",
    "dividend", "buyback", "acquisition", "merger", "takeover", "stake",
    "ceo", "cfo", "resign", "appoint", "lawsuit", "investigation", "probe",
    "recall", "approval", "fda", "contract", "order", "partnership",
    "restructuring", "layoff", "profit warning", "downgrade", "upgrade",
    "résultats", "bénéfice", "chiffre d'affaires", "dividende", "rachat",
    "acquisition", "fusion", "contrat", "commande", "avertissement",
    "nomination", "démission", "enquête", "amende", "plan social",
]

# Sources primaires : communiques d'entreprise et agences de presse. Ce qu'elles
# publient est la source officielle, pas un commentaire sur celle-ci.
SOURCES_PRIMAIRES = [
    "business wire", "businesswire", "pr newswire", "prnewswire",
    "globenewswire", "globe newswire", "accesswire", "newsfile",
    "reuters", "bloomberg", "associated press", "ap news", "dow jones",
    "financial times", "wall street journal", "cnbc", "marketwatch",
    "les echos", "les échos", "afp", "agence france-presse", "boursorama",
    "euronext", "sec filing", "edgar", "company release", "investor relations",
]

# Sources secondaires : commentaire, agregation, contenu automatise.
SOURCES_ECARTEES = [
    "zacks", "motley fool", "simply wall st", "insider monkey", "benzinga",
    "investorplace", "gurufocus", "tipranks", "24/7 wall st", "seeking alpha",
    "stocktwits", "invezz", "barchart",
]

BRUIT = [
    "here's why", "should you buy", "3 stocks", "5 stocks", "best stocks",
    "zacks rank", "motley fool", "jim cramer", "stocks to watch",
    "options activity", "unusual options", "penny stock", "price target hit",
    "moving average", "rsi", "technical analysis", "chart of the day",
    "what analysts are saying", "trading halt resumed",
]


def qualite_source(source: str) -> int:
    """
    Note la fiabilite d'une source : 2 primaire, 1 acceptable, 0 ecartee.

    Une information n'a de valeur que si l'on sait d'ou elle vient. Un
    communique d'entreprise relaye par Business Wire est un fait ; le meme
    fait commente par un agregateur est une opinion sur un fait.
    """
    minuscule = (source or "").lower()
    if any(e in minuscule for e in SOURCES_ECARTEES):
        return 0
    if any(p in minuscule for p in SOURCES_PRIMAIRES):
        return 2
    return 1


def pertinent(titre: str, source: str = "", strict: bool = True) -> bool:
    """
    Retient une depeche.

    En mode strict, seules les sources primaires passent : c'est ce qui
    garantit que l'information transmise est l'annonce officielle, et non
    un commentaire sur celle-ci.
    """
    if qualite_source(source) == 0:
        return False
    if strict and qualite_source(source) < 2:
        return False

    minuscule = titre.lower()
    if any(b in minuscule for b in BRUIT):
        return False
    return any(m in minuscule for m in MOTS_FORTS) or len(minuscule) > 40


def _horodatage(article: dict) -> datetime | None:
    contenu = article.get("content", article)
    brut = contenu.get("pubDate") or article.get("providerPublishTime")
    if isinstance(brut, (int, float)):
        return datetime.fromtimestamp(brut, tz=timezone.utc)
    if isinstance(brut, str):
        for format_date in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                valeur = datetime.strptime(brut, format_date)
                return valeur if valeur.tzinfo else valeur.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _normaliser(article: dict, ticker: str) -> dict | None:
    contenu = article.get("content", article) if isinstance(article, dict) else {}
    titre = contenu.get("title") or article.get("title")
    if not titre:
        return None

    editeur = "—"
    fournisseur = contenu.get("provider")
    if isinstance(fournisseur, dict):
        editeur = fournisseur.get("displayName", "—")
    elif article.get("publisher"):
        editeur = article["publisher"]

    lien = ""
    url = contenu.get("canonicalUrl")
    if isinstance(url, dict):
        lien = url.get("url", "")
    elif article.get("link"):
        lien = article["link"]

    return {"ticker": ticker, "titre": titre.strip(), "source": editeur,
            "lien": lien, "date": _horodatage(article)}


# ==========================================================================
# Collecte
# ==========================================================================

def _pour_un_ticker(ticker: str, jours_avant: int,
                    jours_apres: int = 1) -> dict:
    """
    Tout ce qui concerne une valeur : depeches du jour, publication a venir,
    resultats fraichement publies, revisions d'estimations.
    """
    resultat = {"ticker": ticker, "actualites": [], "publication": None,
                "resultat_publie": None, "revision": None}
    debut_jour = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)

    try:
        valeur = yf.Ticker(ticker)
    except Exception:
        return resultat

    # --- Dépêches du jour
    try:
        for article in (valeur.news or [])[:20]:
            normalise = _normaliser(article, ticker)
            if not normalise or not normalise["date"]:
                continue
            if normalise["date"] < debut_jour:
                continue          # paru avant aujourd'hui : on ignore
            if not pertinent(normalise["titre"], normalise["source"],
                             strict=STRICT):
                continue
            normalise["fiabilite"] = qualite_source(normalise["source"])
            resultat["actualites"].append(normalise)
    except Exception:
        pass

    # --- Publication de résultats approchant
    try:
        dates = valeur.earnings_dates
        if dates is not None and not dates.empty:
            index = pd.to_datetime(dates.index)
            index = index.tz_localize(None) if index.tz is not None else index
            table = dates.copy()
            table.index = index
            aujourdhui = pd.Timestamp.now().normalize()
            futures = table[table.index >= aujourdhui]
            if "Reported EPS" in table.columns:
                futures = futures[futures["Reported EPS"].isna()]
            if not futures.empty:
                prochaine = futures.sort_index()
                date = prochaine.index[0]
                delai = int((date - aujourdhui).days)
                if 0 <= delai <= jours_avant:
                    resultat["publication"] = {
                        "ticker": ticker,
                        "date": date,
                        "jours": delai,
                        "bpa_attendu": float(prochaine.iloc[0].get(
                            "EPS Estimate", float("nan"))),
                    }
    except Exception:
        pass

    # --- Résultats fraîchement publiés, avec la surprise
    try:
        dates = valeur.earnings_dates
        if dates is not None and not dates.empty:
            index = pd.to_datetime(dates.index)
            index = index.tz_localize(None) if index.tz is not None else index
            table = dates.copy()
            table.index = index
            aujourdhui = pd.Timestamp.now().normalize()

            colonne = ("Reported EPS" if "Reported EPS" in table.columns
                       else None)
            if colonne:
                publiees = table[table[colonne].notna()].sort_index()
                if not publiees.empty:
                    derniere = publiees.index[-1]
                    ecoules = int((aujourdhui - derniere).days)
                    if 0 <= ecoules <= jours_apres:
                        ligne = publiees.iloc[-1]
                        attendu = ligne.get("EPS Estimate")
                        publie = ligne.get(colonne)
                        surprise = ligne.get("Surprise(%)")
                        if surprise is None or surprise != surprise:
                            if attendu and attendu == attendu and attendu != 0:
                                surprise = (publie - attendu) / abs(attendu) * 100
                        resultat["resultat_publie"] = {
                            "ticker": ticker, "date": derniere,
                            "jours": ecoules,
                            "bpa_publie": float(publie),
                            "bpa_attendu": (float(attendu)
                                            if attendu == attendu else None),
                            "surprise_pct": (float(surprise)
                                             if surprise == surprise else None),
                        }
    except Exception:
        pass

    # --- Révision des prévisions de bénéfice
    try:
        tendance = valeur.eps_trend
        if tendance is not None and not tendance.empty:
            for periode in ("0y", "+1y", "0q"):
                if periode not in tendance.index:
                    continue
                ligne = tendance.loc[periode]
                actuelle = ligne.get("current")
                ancienne = ligne.get("30daysAgo")
                if (actuelle is None or ancienne is None
                        or actuelle != actuelle or ancienne != ancienne
                        or ancienne == 0):
                    continue
                variation = (actuelle - ancienne) / abs(ancienne) * 100
                # Sous 4 %, la révision relève du bruit d'arrondi
                if abs(variation) >= 4:
                    resultat["revision"] = {
                        "ticker": ticker, "periode": periode,
                        "variation_pct": float(variation),
                        "estimation": float(actuelle),
                    }
                    break
    except Exception:
        pass

    return resultat


def collecter(tickers: list[str], jours_avant: int = 3) -> dict:
    """
    Parcourt la liste en parallèle.

    Le parallélisme est bridé à cinq fils : au-delà, Yahoo renvoie des erreurs
    de débit et la collecte devient plus lente qu'en séquentiel.
    """
    actualites, publications, echecs = [], [], 0
    resultats, revisions = [], []

    with ThreadPoolExecutor(max_workers=FILS_PARALLELES) as pool:
        taches = {}
        for ticker in tickers:
            taches[pool.submit(_pour_un_ticker, ticker, jours_avant)] = ticker
            time.sleep(PAUSE)

        for tache in as_completed(taches):
            try:
                resultat = tache.result()
            except Exception:
                echecs += 1
                continue
            actualites.extend(resultat["actualites"])
            if resultat["publication"]:
                publications.append(resultat["publication"])
            if resultat["resultat_publie"]:
                resultats.append(resultat["resultat_publie"])
            if resultat["revision"]:
                revisions.append(resultat["revision"])

    # Une même dépêche est souvent reprise sur plusieurs valeurs
    vus, uniques = set(), []
    for article in actualites:
        empreinte = " ".join(re.sub(r"[^\w\s]", "", article["titre"].lower())
                             .split()[:8])
        if empreinte in vus:
            continue
        vus.add(empreinte)
        uniques.append(article)

    uniques.sort(key=lambda a: (-a.get("fiabilite", 1),
                               a["date"] or datetime.min.replace(
                                   tzinfo=timezone.utc)), reverse=False)
    publications.sort(key=lambda p: p["jours"])
    revisions.sort(key=lambda r: -abs(r["variation_pct"]))
    return {"actualites": uniques, "publications": publications,
            "resultats": resultats, "revisions": revisions,
            "echecs": echecs, "analysees": len(tickers)}


# ==========================================================================
# Mémoire
# ==========================================================================

def _cle(element: dict) -> str:
    if "titre" in element:
        return "n|" + " ".join(
            re.sub(r"[^\w\s]", "", element["titre"].lower()).split()[:8])
    if "bpa_publie" in element:
        return f"r|{element['ticker']}|{element['date'].date()}"
    if "variation_pct" in element:
        # Arrondi a 5 points : une revision qui s'accentue est un fait nouveau,
        # une revision stable ne l'est pas.
        palier = int(element["variation_pct"] // 5) * 5
        return f"v|{element['ticker']}|{element['periode']}|{palier}"
    return f"p|{element['ticker']}|{element['date'].date()}|{element['jours']}"


def filtrer_nouveaux(elements: list[dict]) -> list[dict]:
    """Écarte ce qui a déjà été transmis."""
    etat = {}
    if FICHIER_ETAT.exists():
        try:
            etat = json.loads(FICHIER_ETAT.read_text())
        except Exception:
            pass

    limite = datetime.now(timezone.utc) - timedelta(days=MEMOIRE_JOURS)
    etat = {k: v for k, v in etat.items()
            if datetime.fromisoformat(v) > limite}

    nouveaux = [e for e in elements if _cle(e) not in etat]
    maintenant = datetime.now(timezone.utc).isoformat()
    for element in nouveaux:
        etat[_cle(element)] = maintenant

    try:
        FICHIER_ETAT.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    except Exception:
        pass
    return nouveaux


# ==========================================================================
# Liste surveillée
# ==========================================================================

def charger_liste(inclure_portefeuille: bool = True) -> list[str]:
    """
    Liste des valeurs suivies.

    Par ordre de priorite : la variable TICKERS_SURVEILLANCE, puis une feuille
    Google publiee, puis les positions detenues.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    tickers, detenues = [], []

    # --- Positions détenues : elles priment toujours
    url_ptf = os.environ.get("URL_FEUILLE", "").strip()
    if inclure_portefeuille and url_ptf:
        try:
            import feuille as fe
            detenues = list(dict.fromkeys(fe.lire(url_ptf)["Ticker"]))
            print(f"  {len(detenues)} valeur(s) détenue(s)")
        except Exception as erreur:
            print(f"Portefeuille illisible : {erreur}", file=sys.stderr)

    # --- Liste de surveillance, par variable puis par feuille
    brut = os.environ.get("TICKERS_SURVEILLANCE", "").strip()
    if brut:
        tickers = [t.strip().upper() for t in
                   brut.replace(";", ",").replace("\n", ",").split(",")
                   if t.strip()]
        print(f"  {len(tickers)} valeur(s) surveillée(s) (variable)")
    else:
        url = os.environ.get("URL_SURVEILLANCE", "").strip()
        if url:
            try:
                import feuille as fe
                tickers = fe.lire_liste(url)
                print(f"  {len(tickers)} valeur(s) surveillée(s) (feuille)")
            except Exception as erreur:
                print(f"Liste de surveillance illisible : {erreur}",
                      file=sys.stderr)

    return list(dict.fromkeys(detenues + tickers))


def charger_detenues() -> set[str]:
    """Positions reellement detenues, pour distinguer suivi et detention."""
    url = os.environ.get("URL_FEUILLE", "").strip()
    if not url:
        return set()
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import feuille as fe
        return set(fe.lire(url)["Ticker"])
    except Exception:
        return set()
