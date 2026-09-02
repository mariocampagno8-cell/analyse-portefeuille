"""
Veille de portefeuille — fichier unique.

Tout le systeme tient ici, en six sections lisibles de haut en bas :

  1. REGLAGES   les seuils, rassembles en un seul endroit
  2. UNIVERS    lecture de la feuille Google, strates A / B / C
  3. ALERTES    franchissements de prix, mouvements, publications
  4. BUDGET     plafonds, silence nocturne, anti-repetition
  5. MESSAGES   mise en forme et envoi Telegram
  6. EXECUTION  enchainement

Principe directeur : une notification n'est envoyee que si elle peut declencher
une action — acheter, vendre, alleger, renforcer. Le reste va au digest ou
nulle part. Les jours ou il ne se passe rien, rien n'est envoye : c'est ce
silence qui rend credibles les messages qui partent.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as heure, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==========================================================================
# 1. RÉGLAGES
# ==========================================================================

# Le mouvement de séance est calibré en écarts-types, jamais en pourcentage
# fixe : 5 % est un événement sur Coca-Cola et une séance ordinaire sur IONQ.
SIGMA = 2.5
PLANCHER = 3.0                  # jamais d'alerte en deçà
PLAFOND = 12.0                  # toujours une alerte au-delà

PROXIMITE_ENTREE = 3.0          # % — approche du prix d'entrée (strate B)
PROXIMITE_CIBLE = 10.0          # % — approche du prix cible (strate C)
CONCENTRATION = 15.0            # % du portefeuille sur une seule ligne
CONCURRENT = 5.0                # % — mouvement chez un pair

MAX_SONORES = 4                 # push sonores par jour
MAX_MESSAGES = 12               # messages par jour
SILENCE_DEBUT, SILENCE_FIN = heure(21, 0), heure(7, 0)
GROUPEMENT = 3                  # événements de même nature avant regroupement
MEMOIRE_JOURS = 5               # anti-répétition
FILS = 5                        # appels Yahoo en parallèle

ETAT = Path(__file__).parent / "etat_veille.json"
ETIQUETTES = {"A": "Portefeuille", "B": "Candidat", "C": "Veille"}


# ==========================================================================
# 2. UNIVERS
# ==========================================================================

SYNONYMES = {
    "ticker": ["ticker", "symbole", "symbol", "code", "valeur"],
    "strate": ["strate", "categorie", "type", "niveau"],
    "quantite": ["quantite", "qte", "nombre", "titres", "quantity"],
    "pru": ["pru", "prix d'achat", "prix dachat", "prix de revient"],
    "prix_entree": ["prix entree", "prix d'entree", "prix cible", "cible"],
    "prix_sortie": ["prix sortie", "prix de sortie", "seuil vente", "stop"],
    "concurrents": ["concurrents", "peers", "comparables"],
}


def _propre(texte) -> str:
    return str(texte).lower().strip().translate(
        str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))


def _nombre(valeur) -> float:
    """Convertit une saisie en nombre, en tolérant les formats français."""
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return float(valeur) if np.isfinite(valeur) else np.nan
    if valeur is None:
        return np.nan
    t = (str(valeur).strip().replace("\u202f", "").replace("\xa0", "")
         .replace(" ", "").replace("€", "").replace("$", ""))
    if not t:
        return np.nan
    if "," in t and "." in t:
        t = (t.replace(".", "").replace(",", ".")
             if t.rindex(",") > t.rindex(".") else t.replace(",", ""))
    elif "," in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def url_csv(url: str) -> list[str]:
    """Adresses de téléchargement possibles pour une feuille Google."""
    url = url.strip()
    if not url:
        return []
    if "output=csv" in url or "format=csv" in url:
        return [url]
    trouve = re.search(r"/spreadsheets/d/(?:e/)?([a-zA-Z0-9-_]+)", url)
    if not trouve:
        return [url]
    cle = trouve.group(1)
    gid = re.search(r"[#&?]gid=([0-9]+)", url)
    suffixe = f"&gid={gid.group(1)}" if gid else ""
    if cle.startswith("2PACX") or "/d/e/" in url:
        return [f"https://docs.google.com/spreadsheets/d/e/{cle}"
                f"/pub?output=csv{suffixe}"]
    racine = f"https://docs.google.com/spreadsheets/d/{cle}"
    return [f"{racine}/export?format=csv{suffixe}",
            f"{racine}/gviz/tq?tqx=out:csv"]


def lire_univers(url: str) -> pd.DataFrame:
    """
    Charge l'univers depuis la feuille et applique les règles de strate.

    Deux corrections automatiques. Une valeur détenue est en A quoi qu'indique
    la feuille — on ne peut pas être candidat à ce qu'on possède. Une valeur
    déclarée B sans prix d'entrée redescend en C : c'est le filtre dur qui
    empêche la strate B de devenir un fourre-tout.
    """
    brut = None
    for adresse in url_csv(url):
        try:
            essai = pd.read_csv(adresse)
            if not essai.empty:
                brut = essai
                break
        except Exception:
            continue
    if brut is None:
        raise ValueError("Feuille inaccessible. Vérifie qu'elle est publiée "
                         "au format CSV.")

    correspondance = {}
    for colonne in brut.columns:
        nom = _propre(colonne)
        for cible, variantes in SYNONYMES.items():
            if cible in correspondance.values():
                continue
            if nom in variantes or any(v in nom for v in variantes):
                correspondance[colonne] = cible
                break
    brut = brut.rename(columns=correspondance)

    if "ticker" not in brut.columns:
        raise ValueError(f"Colonne Ticker introuvable. Colonnes trouvées : "
                         f"{', '.join(map(str, brut.columns))}.")

    lignes = []
    for _, ligne in brut.iterrows():
        ticker = str(ligne.get("ticker", "")).strip().upper()
        if not ticker or ticker in ("NAN", "TICKER"):
            continue
        quantite = _nombre(ligne.get("quantite"))
        entree = _nombre(ligne.get("prix_entree"))
        strate = str(ligne.get("strate", "C")).strip().upper()[:1]
        if strate not in ("A", "B", "C"):
            strate = "C"
        if quantite > 0:
            strate = "A"
        elif strate == "B" and not np.isfinite(entree):
            strate = "C"

        lignes.append({
            "ticker": ticker, "strate": strate, "quantite": quantite,
            "pru": _nombre(ligne.get("pru")), "prix_entree": entree,
            "prix_sortie": _nombre(ligne.get("prix_sortie")),
            "concurrents": str(ligne.get("concurrents", "") or "").strip()})
    return pd.DataFrame(lignes).drop_duplicates(subset="ticker")


def concurrents(univers: pd.DataFrame) -> dict[str, list[str]]:
    """Pairs déclarés pour chaque ligne détenue."""
    sortie = {}
    for _, ligne in univers[univers["strate"] == "A"].iterrows():
        liste = [t.strip().upper() for t in
                 str(ligne.get("concurrents", "") or "").replace(";", ",").split(",")
                 if t.strip()]
        if liste:
            sortie[ligne["ticker"]] = liste
    return sortie


# ==========================================================================
# 3. ALERTES
# ==========================================================================

def charger_cours(tickers: list[str], periode: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    brut = yf.download(tickers, period=periode, interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    if brut.empty:
        return pd.DataFrame()
    cours = (brut["Close"] if isinstance(brut.columns, pd.MultiIndex)
             else brut[["Close"]].rename(columns={"Close": tickers[0]}))
    return cours.dropna(how="all").ffill()


def seuil(prix: pd.Series) -> float:
    """Seuil de mouvement propre au titre, en pourcentage."""
    r = prix.pct_change().dropna().tail(120)
    if len(r) < 40:
        return PLANCHER * 2
    ecart = float(r.std(ddof=1)) * 100
    return PLANCHER if ecart <= 0 else float(
        np.clip(ecart * SIGMA, PLANCHER, PLAFOND))


def _publication(ticker: str) -> dict | None:
    try:
        dates = yf.Ticker(ticker).earnings_dates
        if dates is None or dates.empty:
            return None
        index = pd.to_datetime(dates.index)
        table = dates.copy()
        table.index = index.tz_localize(None) if index.tz is not None else index
        futures = table[table.index >= pd.Timestamp.now().normalize()]
        if "Reported EPS" in table.columns:
            futures = futures[futures["Reported EPS"].isna()]
        if futures.empty:
            return None
        prochaine = futures.sort_index()
        return {"ticker": ticker, "date": prochaine.index[0],
                "jours": int((prochaine.index[0]
                              - pd.Timestamp.now().normalize()).days),
                "bpa": float(prochaine.iloc[0].get("EPS Estimate", np.nan))}
    except Exception:
        return None


def calendrier(tickers: list[str]) -> list[dict]:
    """Prochaines publications, en parallèle bridé pour ménager Yahoo."""
    sortie = []
    with ThreadPoolExecutor(max_workers=FILS) as pool:
        taches = []
        for ticker in tickers:
            taches.append(pool.submit(_publication, ticker))
            time.sleep(0.15)
        for tache in as_completed(taches):
            try:
                if (resultat := tache.result()):
                    sortie.append(resultat)
            except Exception:
                continue
    return sorted(sortie, key=lambda p: p["jours"])


def detecter(univers: pd.DataFrame, cours: pd.DataFrame,
             publications: list[dict], mode: str) -> list[dict]:
    """
    Toutes les alertes du passage.

    Chaque alerte porte sa priorité, sa strate, et un indicateur `sonore`
    explicite : le franchissement d'un prix d'entrée en strate B doit sonner
    malgré son rang P2, parce que c'est l'alerte que l'on attend vraiment.
    """
    alertes = []
    if cours.empty:
        return alertes
    strates = dict(zip(univers["ticker"], univers["strate"]))

    for _, ligne in univers.iterrows():
        t = ligne["ticker"]
        if t not in cours.columns:
            continue
        prix = cours[t].dropna()
        if len(prix) < 6:
            continue

        actuel, veille = float(prix.iloc[-1]), float(prix.iloc[-2])
        var = (actuel / veille - 1) * 100
        cinq = (actuel / float(prix.iloc[-6]) - 1) * 100
        s, entree, sortie = ligne["strate"], ligne["prix_entree"], ligne["prix_sortie"]
        base = {"Cours": f"{actuel:.2f}", "Séance": f"{var:+.1f} %"}

        if s == "B" and np.isfinite(entree):
            if actuel <= entree < veille:
                alertes.append({"ticker": t, "strate": "B", "priorite": "P2",
                    "sonore": True, "nature": "prix_entree", "emoji": "⚡️",
                    "titre": "Prix d'entrée franchi",
                    "faits": {**base, "Prix visé": f"{entree:.2f}"}})
            elif 0 < (actuel / entree - 1) * 100 <= PROXIMITE_ENTREE:
                alertes.append({"ticker": t, "strate": "B", "priorite": "P2",
                    "nature": "approche", "emoji": "📉",
                    "titre": "Approche du prix d'entrée",
                    "faits": {**base, "Prix visé": f"{entree:.2f}",
                              "Écart": f"{(actuel / entree - 1) * 100:+.1f} %"}})

        if s == "A":
            if np.isfinite(sortie) and actuel <= sortie < veille:
                alertes.append({"ticker": t, "strate": "A", "priorite": "P1",
                    "nature": "seuil_vente", "emoji": "🔴",
                    "titre": "Seuil de vente franchi",
                    "faits": {**base, "Seuil": f"{sortie:.2f}"}})
            elif abs(var) >= (limite := seuil(prix)):
                alertes.append({"ticker": t, "strate": "A", "priorite": "P1",
                    "nature": "mouvement", "emoji": "⚠️",
                    "titre": f"Mouvement de séance {var:+.1f} %",
                    "faits": {**base, "Seuil du titre": f"±{limite:.1f} %",
                              "5 séances": f"{cinq:+.1f} %"}})

        if s == "C":
            if np.isfinite(entree):
                ecart = (actuel / entree - 1) * 100
                if actuel <= entree < veille:
                    alertes.append({"ticker": t, "strate": "C", "priorite": "P2",
                        "nature": "prix_cible", "emoji": "🎯",
                        "titre": "Prix cible atteint",
                        "faits": {**base, "Cible": f"{entree:.2f}"}})
                elif 0 < ecart <= PROXIMITE_CIBLE:
                    alertes.append({"ticker": t, "strate": "C", "priorite": "P3",
                        "nature": "proximite", "emoji": "🎯",
                        "titre": f"À {ecart:.0f} % du prix cible",
                        "faits": {**base, "Cible": f"{entree:.2f}"}})
            limite_5j = -min(seuil(prix) * 2.2, 35.0)
            if cinq <= limite_5j:
                alertes.append({"ticker": t, "strate": "C", "priorite": "P2",
                    "nature": "decrochage", "emoji": "⚠️",
                    "titre": f"Repli de {abs(cinq):.0f} % en 5 séances",
                    "faits": {**base, "5 séances": f"{cinq:+.1f} %",
                              "Seuil du titre": f"{limite_5j:.0f} %"}})

    # Concentration : une seule alerte, sur la ligne la plus lourde
    detenues = univers[(univers["strate"] == "A")
                       & (univers["quantite"].fillna(0) > 0)]
    valeurs = {r["ticker"]: float(cours[r["ticker"]].dropna().iloc[-1])
                            * float(r["quantite"])
               for _, r in detenues.iterrows()
               if r["ticker"] in cours.columns
               and not cours[r["ticker"]].dropna().empty}
    total = sum(valeurs.values())
    if total > 0:
        poids = sorted(((t, v / total * 100) for t, v in valeurs.items()),
                       key=lambda x: -x[1])
        if poids[0][1] > CONCENTRATION:
            alertes.append({"ticker": poids[0][0], "strate": "A",
                "priorite": "P2", "nature": "concentration", "emoji": "⚠️",
                "titre": f"Concentration : {poids[0][1]:.0f} % du portefeuille",
                "faits": {"Poids": f"{poids[0][1]:.1f} %",
                          "Seuil": f"{CONCENTRATION:.0f} %"}})

    # Concurrents : un avertissement chez un pair précède souvent le vôtre
    for detenue, pairs in concurrents(univers).items():
        for pair in pairs:
            if pair not in cours.columns:
                continue
            prix = cours[pair].dropna()
            if len(prix) < 6:
                continue
            var = (float(prix.iloc[-1]) / float(prix.iloc[-2]) - 1) * 100
            if abs(var) >= max(seuil(prix), CONCURRENT):
                alertes.append({"ticker": pair, "strate": "A", "priorite": "P2",
                    "nature": "concurrent", "emoji": "👥",
                    "titre": f"{var:+.1f} % — concurrent de {detenue}",
                    "faits": {"Cours": f"{float(prix.iloc[-1]):.2f}",
                              "Séance": f"{var:+.1f} %",
                              "Ligne détenue": detenue}})

    # Rappel J-1, le matin uniquement
    if mode == "matin":
        for p in publications:
            s = strates.get(p["ticker"], "C")
            if s == "C" or p["jours"] != 1:
                continue
            faits = {"Date": p["date"].strftime("%d/%m/%Y")}
            if p["bpa"] == p["bpa"]:
                faits["BPA attendu"] = f"{p['bpa']:.2f}"
            alertes.append({"ticker": p["ticker"], "strate": s,
                "priorite": "P1" if s == "A" else "P2",
                "nature": "publication", "emoji": "📅",
                "titre": "Résultats demain", "faits": faits})

    # Un franchissement de seuil rend l'alerte de mouvement redondante
    par_ticker: dict[str, list[dict]] = {}
    for a in alertes:
        par_ticker.setdefault(a["ticker"], []).append(a)
    retenues = []
    for groupe in par_ticker.values():
        natures = {a["nature"] for a in groupe}
        if natures & {"seuil_vente", "prix_entree", "prix_cible"}:
            groupe = [a for a in groupe if a["nature"] not in
                      ("mouvement", "approche", "proximite")]
        retenues.extend(groupe)
    return retenues


# ==========================================================================
# 4. BUDGET
# ==========================================================================

def lire_etat() -> dict:
    aujourdhui = datetime.now().strftime("%Y-%m-%d")
    vide = {"date": aujourdhui, "sonores": 0, "messages": 0, "envoyees": {}}
    if not ETAT.exists():
        return vide
    try:
        etat = json.loads(ETAT.read_text())
    except Exception:
        return vide
    if etat.get("date") != aujourdhui:
        etat.update({"date": aujourdhui, "sonores": 0, "messages": 0})
    limite = (datetime.now() - timedelta(days=MEMOIRE_JOURS)).isoformat()
    etat["envoyees"] = {k: v for k, v in etat.get("envoyees", {}).items()
                        if v > limite}
    return etat


def en_silence() -> bool:
    maintenant = datetime.now().time()
    return maintenant >= SILENCE_DEBUT or maintenant < SILENCE_FIN


def arbitrer(alertes: list[dict], etat: dict) -> dict:
    """
    Applique le budget : anti-répétition, groupement, plafonds, silence.

    Rien n'est perdu : ce qui ne part pas est reporté au digest avec son motif.
    """
    nouvelles = [a for a in alertes
                 if f"{a['ticker']}|{a['nature']}" not in etat["envoyees"]]

    familles: dict[tuple, list[dict]] = {}
    for a in nouvelles:
        familles.setdefault((a["ticker"], a["nature"]), []).append(a)
    nouvelles = []
    for (ticker, nature), groupe in familles.items():
        if len(groupe) >= GROUPEMENT:
            principale = dict(groupe[0])
            principale["titre"] = f"{len(groupe)} alertes ({nature})"
            nouvelles.append(principale)
        else:
            nouvelles.extend(groupe)

    rang_p = {"P1": 0, "P2": 1, "P3": 2}
    rang_s = {"A": 0, "B": 1, "C": 2}
    nouvelles.sort(key=lambda a: (
        0 if a.get("sonore") else rang_p.get(a["priorite"], 9),
        rang_s.get(a["strate"], 9)))

    sonores, silencieuses, reportees = [], [], []
    compte_s, compte_m = etat.get("sonores", 0), etat.get("messages", 0)

    for a in nouvelles:
        if a["priorite"] == "P3":
            reportees.append({**a, "motif": "P3 — digest"})
        elif compte_m >= MAX_MESSAGES:
            reportees.append({**a, "motif": "plafond quotidien"})
        elif en_silence() and not (a["priorite"] == "P1" and a["strate"] == "A"):
            reportees.append({**a, "motif": "plage de silence"})
        else:
            # bool() explicite : `a.get("sonore")` vaut None quand la clé
            # est absente, ce qui propagerait None dans toute l'expression.
            sonne = bool(a["strate"] != "C" and not en_silence()
                         and (a["priorite"] == "P1" or a.get("sonore"))
                         and compte_s < MAX_SONORES)
            (sonores if sonne else silencieuses).append(a)
            compte_s += int(sonne)
            compte_m += 1

    return {"sonores": sonores, "silencieuses": silencieuses,
            "reportees": reportees, "compte_s": compte_s, "compte_m": compte_m}


def enregistrer(arbitrage: dict, etat: dict) -> None:
    etat["sonores"], etat["messages"] = arbitrage["compte_s"], arbitrage["compte_m"]
    maintenant = datetime.now().isoformat()
    for a in arbitrage["sonores"] + arbitrage["silencieuses"]:
        etat["envoyees"][f"{a['ticker']}|{a['nature']}"] = maintenant
    try:
        ETAT.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    except Exception:
        pass


# ==========================================================================
# 5. MESSAGES
# ==========================================================================

def echapper(texte) -> str:
    """Un titre contenant « AT&T » ferait rejeter le message par Telegram."""
    return html.escape(str(texte), quote=False)


def formater(a: dict) -> str:
    """Un message par alerte : titre, chiffres alignés, source."""
    marque = " 💼" if a["strate"] == "A" else ""
    entete = (f"{a.get('emoji', '•')} <b>{echapper(a['ticker'])}{marque} — "
              f"{echapper(a['titre'])}</b>")
    lignes = [f"{echapper(k)[:16].ljust(16, '.')} {echapper(v)}"
              for k, v in a.get("faits", {}).items()]
    tableau = "<pre>" + "\n".join(lignes) + "</pre>" if lignes else ""
    pied = (f"<i>Yahoo Finance · {ETIQUETTES[a['strate']]} · "
            f"{datetime.now().strftime('%d/%m %H:%M')}</i>")
    return "\n\n".join(b for b in (entete, tableau, pied) if b)


def digest(univers: pd.DataFrame, cours: pd.DataFrame,
           publications: list[dict], reportees: list[dict]) -> str:
    """Synthèse du vendredi : le seul message qui prend du recul."""
    if cours.empty:
        return ""
    blocs = [f"🗂 <b>SEMAINE — {datetime.now().strftime('%d/%m/%Y')}</b>"]

    mouvements = []
    for _, ligne in univers.iterrows():
        t = ligne["ticker"]
        if t not in cours.columns:
            continue
        prix = cours[t].dropna()
        if len(prix) >= 6:
            var = (float(prix.iloc[-1]) / float(prix.iloc[-6]) - 1) * 100
            if abs(var) >= 5:
                mouvements.append((abs(var), t, ligne["strate"], var))
    mouvements.sort(reverse=True)

    if mouvements:
        blocs.append("<b>Mouvements > 5 %</b>\n<pre>" + "\n".join(
            f"{t[:8].ljust(9)}{s}  {v:+6.1f} %"
            for _, t, s, v in mouvements[:10]) + "</pre>")
    else:
        blocs.append("<i>Aucun mouvement supérieur à 5 % cette semaine.</i>")

    prochaines = [p for p in publications if p["jours"] <= 10]
    if prochaines:
        blocs.append("<b>Publications à venir</b>\n<pre>" + "\n".join(
            f"{p['ticker'][:8].ljust(9)}{p['date'].strftime('%d/%m')}  "
            f"J-{p['jours']}" for p in prochaines[:8]) + "</pre>")

    if reportees:
        compte: dict[str, int] = {}
        for e in reportees:
            compte[e["nature"]] = compte.get(e["nature"], 0) + 1
        blocs.append("<b>Écarté des notifications</b>\n<pre>" + "\n".join(
            f"{echapper(k)[:16].ljust(16, '.')} {v}"
            for k, v in compte.items()) + "</pre>")

    c = univers["strate"].value_counts().to_dict()
    blocs.append(f"<i>Univers : {c.get('A', 0)} A · {c.get('B', 0)} B · "
                 f"{c.get('C', 0)} C</i>")
    return "\n\n".join(blocs)


def envoyer(message: str, silencieux: bool = False) -> bool:
    jeton = os.environ.get("TELEGRAM_JETON", "").strip()
    destinataire = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    if not jeton or not destinataire:
        print("Telegram non configuré :\n" + message, file=sys.stderr)
        return False
    if len(message) > 4096:
        message = message[:4000] + "…"
    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": destinataire, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True,
                  "disable_notification": silencieux}, timeout=20)
        if not reponse.ok:
            print(f"Telegram a refusé ({reponse.status_code}) : "
                  f"{reponse.text[:200]}", file=sys.stderr)
        return reponse.ok
    except Exception as erreur:
        print(f"Échec Telegram : {erreur}", file=sys.stderr)
        return False


# ==========================================================================
# 6. EXÉCUTION
# ==========================================================================

def principal() -> int:
    mode = os.environ.get("MODE", "seance")
    url = os.environ.get("URL_UNIVERS", "").strip()
    if not url:
        print("Variable URL_UNIVERS absente.", file=sys.stderr)
        return 1

    print(f"Mode : {mode}")
    try:
        univers = lire_univers(url)
    except ValueError as erreur:
        print(f"Univers illisible : {erreur}", file=sys.stderr)
        return 1

    c = univers["strate"].value_counts().to_dict()
    print(f"Univers : {c.get('A', 0)} A, {c.get('B', 0)} B, {c.get('C', 0)} C.")

    pairs = concurrents(univers)
    tickers = list(dict.fromkeys(univers["ticker"].tolist()
                   + [p for liste in pairs.values() for p in liste]))
    cours = charger_cours(tickers)
    if cours.empty:
        print("Aucun cours disponible.", file=sys.stderr)
        return 1

    publications = []
    if mode in ("matin", "digest"):
        publications = calendrier(
            univers[univers["strate"].isin(["A", "B"])]["ticker"].tolist())
        print(f"{len(publications)} publication(s) au calendrier.")

    alertes = detecter(univers, cours, publications, mode)
    print(f"{len(alertes)} alerte(s) détectée(s).")

    etat = lire_etat()
    arbitrage = arbitrer(alertes, etat)
    print(f"Arbitrage : {len(arbitrage['sonores'])} sonore(s), "
          f"{len(arbitrage['silencieuses'])} silencieuse(s), "
          f"{len(arbitrage['reportees'])} reportée(s).")

    envois = 0
    for a in arbitrage["sonores"]:
        if envoyer(formater(a)):
            envois += 1
            time.sleep(1)
    for a in arbitrage["silencieuses"]:
        if envoyer(formater(a), silencieux=True):
            envois += 1
            time.sleep(1)

    if mode == "digest":
        message = digest(univers, cours, publications, arbitrage["reportees"])
        if message and envoyer(message, silencieux=True):
            envois += 1

    if envois:
        enregistrer(arbitrage, etat)
    print(f"{envois} message(s) envoyé(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
