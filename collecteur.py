"""
Collecteur de la phase 1.

Perimetre volontairement restreint au cahier des charges : calendrier des
publications sur les strates A et B, alerte J-1, tableau de chiffres le jour J,
alertes de prix sur les trois strates. Donnees gratuites uniquement.

Ce qui n'est pas ici est deliberement absent : cycle J-15 et J-5, revisions de
consensus, lecture redigee. Ces briques appartiennent aux phases 2 et 3, et le
plan prevoit un mois de fonctionnement avant d'y toucher.

Regle appliquee partout : une alerte n'est produite que si elle peut declencher
une action. Le reste va au digest ou nulle part.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

FILS = 5
PAUSE = 0.15

# Seuils de risque, par strate
#
# Le mouvement de seance est calibre en ECARTS-TYPES, non en pourcentage fixe.
# Un seuil unique de 5 % est ininterpretable : c'est un evenement majeur sur
# Coca-Cola et une seance ordinaire sur IONQ. Normaliser par la volatilite
# propre a chaque titre rend le critere comparable, et evite d'etre noye par
# les valeurs volatiles tout en restant sensible sur les valeurs calmes.
SEUIL_SIGMA = 2.5               # ecarts-types pour un mouvement notable
PLANCHER_MOUVEMENT = 3.0        # en deca, jamais d'alerte quelle que soit la vol
PLAFOND_MOUVEMENT = 12.0        # au-dela, alerte meme si la vol est extreme
SEUIL_INTRADAY_A = 5.0          # conserve pour reference, non utilise
SEUIL_CHUTE_C = -15.0           # baisse sur cinq seances en strate C
SEUIL_PROXIMITE_C = 10.0        # rayon autour du prix cible en strate C
SEUIL_CONCENTRATION = 15.0      # poids maximal d'une ligne
SEUIL_CONCURRENT = 7.0          # mouvement chez un pair


# ==========================================================================
# Collecte des cours
# ==========================================================================

def charger_cours(tickers: list[str], periode: str = "6mo") -> pd.DataFrame:
    """Cloture ajustee de toutes les valeurs suivies, en un appel groupe."""
    if not tickers:
        return pd.DataFrame()
    brut = yf.download(tickers, period=periode, interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    if brut.empty:
        return pd.DataFrame()
    cours = (brut["Close"] if isinstance(brut.columns, pd.MultiIndex)
             else brut[["Close"]].rename(columns={"Close": tickers[0]}))
    return cours.dropna(how="all").ffill()


def _publication(ticker: str) -> dict | None:
    """Prochaine date de publication annoncee pour une valeur."""
    try:
        dates = yf.Ticker(ticker).earnings_dates
        if dates is None or dates.empty:
            return None
        index = pd.to_datetime(dates.index)
        index = index.tz_localize(None) if index.tz is not None else index
        table = dates.copy()
        table.index = index

        aujourdhui = pd.Timestamp.now().normalize()
        futures = table[table.index >= aujourdhui]
        if "Reported EPS" in table.columns:
            futures = futures[futures["Reported EPS"].isna()]
        if futures.empty:
            return None

        prochaine = futures.sort_index()
        date = prochaine.index[0]
        return {"ticker": ticker, "date": date,
                "jours": int((date - aujourdhui).days),
                "bpa_attendu": float(prochaine.iloc[0].get("EPS Estimate", np.nan))}
    except Exception:
        return None


def calendrier(tickers: list[str]) -> list[dict]:
    """Prochaines publications, en parallele bride."""
    import time as _t
    resultats = []
    with ThreadPoolExecutor(max_workers=FILS) as pool:
        taches = []
        for ticker in tickers:
            taches.append(pool.submit(_publication, ticker))
            _t.sleep(PAUSE)
        for tache in as_completed(taches):
            try:
                valeur = tache.result()
                if valeur:
                    resultats.append(valeur)
            except Exception:
                continue
    return sorted(resultats, key=lambda p: p["jours"])


# ==========================================================================
# Alertes de prix — le cœur du dispositif
# ==========================================================================

def alertes_prix(univers: pd.DataFrame, cours: pd.DataFrame) -> list[dict]:
    """
    Franchissements de seuils, par strate.

    C'est le mecanisme le plus rentable du systeme : il ne depend d'aucune
    donnee payante, ne se declenche presque jamais, et quand il se declenche
    il appelle une decision immediate.
    """
    alertes = []
    if univers.empty or cours.empty:
        return alertes

    for _, ligne in univers.iterrows():
        ticker = ligne["ticker"]
        if ticker not in cours.columns:
            continue
        prix = cours[ticker].dropna()
        if len(prix) < 6:
            continue

        actuel = float(prix.iloc[-1])
        veille = float(prix.iloc[-2])
        variation_jour = (actuel / veille - 1) * 100
        cinq_seances = (actuel / float(prix.iloc[-6]) - 1) * 100
        strate = ligne["strate"]

        entree = ligne.get("prix_entree")
        sortie = ligne.get("prix_sortie")

        # --- Strate B : franchissement du prix d'entrée. L'alerte attendue.
        if strate == "B" and entree is not None and np.isfinite(entree):
            if actuel <= entree < veille:
                alertes.append({
                    "ticker": ticker, "strate": "B", "priorite": "P2",
                    "sonore": True,
                    "nature": "prix_entree",
                    "titre": "Prix d'entrée franchi",
                    "faits": {"Cours": f"{actuel:.2f}",
                              "Prix visé": f"{entree:.2f}",
                              "Séance": f"{variation_jour:+.1f} %"},
                })
            elif abs(actuel / entree - 1) * 100 <= 3 and actuel > entree:
                alertes.append({
                    "ticker": ticker, "strate": "B", "priorite": "P2",
                    "nature": "approche_entree",
                    "titre": "Approche du prix d'entrée",
                    "faits": {"Cours": f"{actuel:.2f}",
                              "Prix visé": f"{entree:.2f}",
                              "Écart": f"{(actuel / entree - 1) * 100:+.1f} %"},
                })

        # --- Strate A : seuil de vente, mouvement intraday
        if strate == "A":
            if sortie is not None and np.isfinite(sortie):
                if actuel <= sortie < veille:
                    alertes.append({
                        "ticker": ticker, "strate": "A", "priorite": "P1",
                        "nature": "seuil_vente",
                        "titre": "Seuil de vente franchi",
                        "faits": {"Cours": f"{actuel:.2f}",
                                  "Seuil": f"{sortie:.2f}",
                                  "Séance": f"{variation_jour:+.1f} %"},
                    })
            z, seuil_titre = _mouvement_notable(prix, variation_jour)
            if z is not None:
                alertes.append({
                    "ticker": ticker, "strate": "A", "priorite": "P1",
                    "nature": "mouvement",
                    "titre": f"Mouvement de séance {variation_jour:+.1f} %",
                    "faits": {"Cours": f"{actuel:.2f}",
                              "Veille": f"{veille:.2f}",
                              "Ampleur": f"{abs(z):.1f} écarts-types",
                              "Seuil du titre": f"±{seuil_titre:.1f} %",
                              "5 séances": f"{cinq_seances:+.1f} %"},
                })

        # --- Strate C : uniquement le prix cible et le décrochage
        if strate == "C":
            if entree is not None and np.isfinite(entree):
                ecart = (actuel / entree - 1) * 100
                if actuel <= entree < veille:
                    alertes.append({
                        "ticker": ticker, "strate": "C", "priorite": "P2",
                        "nature": "prix_cible",
                        "titre": "Prix cible atteint",
                        "faits": {"Cours": f"{actuel:.2f}",
                                  "Cible": f"{entree:.2f}"},
                    })
                elif 0 < ecart <= SEUIL_PROXIMITE_C:
                    alertes.append({
                        "ticker": ticker, "strate": "C", "priorite": "P3",
                        "nature": "proximite_cible",
                        "titre": f"À {ecart:.0f} % du prix cible",
                        "faits": {"Cours": f"{actuel:.2f}",
                                  "Cible": f"{entree:.2f}"},
                    })
            rendements_5j = prix.pct_change().dropna().tail(120)
            seuil_5j = SEUIL_CHUTE_C
            if len(rendements_5j) >= 40:
                vol_5j = float(rendements_5j.std(ddof=1)) * 100 * np.sqrt(5)
                seuil_5j = float(np.clip(-vol_5j * 2.0, -35.0, -8.0))
            if cinq_seances <= seuil_5j:
                alertes.append({
                    "ticker": ticker, "strate": "C", "priorite": "P2",
                    "nature": "decrochage",
                    "titre": f"Repli de {abs(cinq_seances):.0f} % en 5 séances",
                    "faits": {"Cours": f"{actuel:.2f}",
                              "5 séances": f"{cinq_seances:+.1f} %",
                              "Seuil du titre": f"{seuil_5j:.0f} %"},
                })

    return dedoublonner(alertes)


def dedoublonner(alertes: list[dict]) -> list[dict]:
    """
    Fusionne les alertes qui decrivent le meme evenement.

    Une chute qui franchit un seuil de vente declenche a la fois l'alerte de
    seuil et l'alerte de mouvement : c'est le meme fait, et le recevoir deux
    fois use l'attention. On conserve la plus specifique.
    """
    # Du plus specifique au plus generique : le premier trouve l'emporte
    hierarchie = ["seuil_vente", "prix_entree", "prix_cible", "decrochage",
                  "mouvement", "approche_entree", "proximite_cible"]

    par_ticker: dict[str, list[dict]] = {}
    for alerte in alertes:
        par_ticker.setdefault(alerte["ticker"], []).append(alerte)

    retenues = []
    for ticker, groupe in par_ticker.items():
        natures = {a["nature"] for a in groupe}
        # Un franchissement de seuil rend l'alerte de mouvement redondante
        if natures & {"seuil_vente", "prix_entree", "prix_cible"}:
            groupe = [a for a in groupe if a["nature"] != "mouvement"]
        # L'approche devient inutile une fois le prix franchi
        if "prix_entree" in natures:
            groupe = [a for a in groupe if a["nature"] != "approche_entree"]
        if "prix_cible" in natures:
            groupe = [a for a in groupe if a["nature"] != "proximite_cible"]

        groupe.sort(key=lambda a: hierarchie.index(a["nature"])
                    if a["nature"] in hierarchie else 99)
        retenues.extend(groupe)
    return retenues


def _mouvement_notable(prix: pd.Series,
                       variation: float) -> tuple[float | None, float]:
    """
    Determine si une variation sort de l'ordinaire POUR CE TITRE.

    Renvoie l'ecart en nombre d'ecarts-types et le seuil correspondant en
    pourcentage, ou None si le mouvement est banal. Le plancher evite de
    signaler 1,5 % sur une obligation, le plafond garantit qu'un mouvement
    de 12 % remonte toujours, meme sur un titre habituellement chaotique.
    """
    rendements = prix.pct_change().dropna().tail(120)
    if len(rendements) < 40:
        seuil = PLANCHER_MOUVEMENT * 2
        return (variation / seuil * SEUIL_SIGMA if abs(variation) >= seuil
                else None), seuil

    ecart_type = float(rendements.std(ddof=1)) * 100
    if ecart_type <= 0:
        return None, PLANCHER_MOUVEMENT

    seuil = float(np.clip(ecart_type * SEUIL_SIGMA,
                          PLANCHER_MOUVEMENT, PLAFOND_MOUVEMENT))
    if abs(variation) < seuil:
        return None, seuil
    return variation / ecart_type, seuil


def alerte_concentration(univers: pd.DataFrame,
                         cours: pd.DataFrame) -> list[dict]:
    """Une ligne qui depasse le seuil de poids fixe."""
    detenues = univers[(univers["strate"] == "A")
                       & (univers["quantite"].fillna(0) > 0)]
    if detenues.empty or cours.empty:
        return []

    valeurs = {}
    for _, ligne in detenues.iterrows():
        ticker = ligne["ticker"]
        if ticker in cours.columns and not cours[ticker].dropna().empty:
            valeurs[ticker] = (float(cours[ticker].dropna().iloc[-1])
                               * float(ligne["quantite"]))
    total = sum(valeurs.values())
    if total <= 0:
        return []

    # Trois alertes de concentration le meme jour disent une seule chose : le
    # portefeuille est concentre. On ne signale que la ligne la plus lourde.
    depassements = [(t, v / total * 100) for t, v in valeurs.items()
                    if v / total * 100 > SEUIL_CONCENTRATION]
    if not depassements:
        return []
    depassements.sort(key=lambda x: -x[1])

    alertes = []
    for ticker, poids in depassements[:1]:
        if True:
            alertes.append({
                "ticker": ticker, "strate": "A", "priorite": "P2",
                "nature": "concentration",
                "titre": f"Concentration : {poids:.0f} % sur {ticker}",
                "faits": {"Ligne la plus lourde": f"{ticker} — {poids:.1f} %",
                          "Seuil": f"{SEUIL_CONCENTRATION:.0f} %",
                          "Lignes au-dessus du seuil": str(len(depassements))},
            })
    return alertes


def alertes_concurrents(pairs: dict[str, list[str]],
                        cours: pd.DataFrame) -> list[dict]:
    """
    Mouvement marque chez un concurrent d'une ligne detenue.

    Un avertissement chez un pair arrive avant le votre et laisse le temps
    d'agir. C'est l'un des rares signaux reellement avances et gratuits.
    """
    alertes = []
    if not pairs or cours.empty:
        return alertes

    for detenue, liste in pairs.items():
        for pair in liste:
            if pair not in cours.columns:
                continue
            prix = cours[pair].dropna()
            if len(prix) < 2:
                continue
            variation = (float(prix.iloc[-1]) / float(prix.iloc[-2]) - 1) * 100
            z_pair, seuil_pair = _mouvement_notable(prix, variation)
            if z_pair is not None and abs(variation) >= SEUIL_CONCURRENT / 2:
                alertes.append({
                    "ticker": pair, "strate": "A", "priorite": "P2",
                    "nature": "concurrent",
                    "titre": f"{pair} {variation:+.1f} % — concurrent de {detenue}",
                    "faits": {"Concurrent": pair, "Ligne détenue": detenue,
                              "Séance": f"{variation:+.1f} %",
                              "Ampleur": f"{abs(z_pair):.1f} écarts-types"},
                })
    return alertes


# ==========================================================================
# Calendrier et rappel J-1
# ==========================================================================

def alertes_calendrier(publications: list[dict],
                       univers: pd.DataFrame) -> list[dict]:
    """
    Rappel J-1 sur les strates A et B.

    La phase 1 se limite au rappel. Le cycle J-15 et J-5 suppose des donnees
    de consensus fiables, qui relevent de la phase 2.
    """
    strates = dict(zip(univers["ticker"], univers["strate"])) \
        if not univers.empty else {}
    alertes = []

    for publication in publications:
        strate = strates.get(publication["ticker"], "C")
        if strate == "C" or publication["jours"] != 1:
            continue
        bpa = publication.get("bpa_attendu")
        faits = {"Date": publication["date"].strftime("%d/%m/%Y")}
        if bpa is not None and bpa == bpa:
            faits["BPA attendu"] = f"{bpa:.2f}"
        alertes.append({
            "ticker": publication["ticker"], "strate": strate,
            "priorite": "P1" if strate == "A" else "P2",
            "nature": "prepublication",
            "titre": "Résultats demain",
            "faits": faits,
        })
    return alertes


# ==========================================================================
# Assemblage
# ==========================================================================

def collecter(univers: pd.DataFrame) -> dict:
    """Passe complete de la phase 1."""
    import strates as st

    suivies = st.tickers(univers)
    pairs = st.concurrents(univers)
    tous = list(dict.fromkeys(
        suivies + [p for liste in pairs.values() for p in liste]))

    print(f"{len(suivies)} valeur(s) suivie(s), "
          f"{len(tous) - len(suivies)} concurrent(s).")

    cours = charger_cours(tous)
    if cours.empty:
        print("Aucun cours disponible.", file=sys.stderr)
        return {"alertes": [], "publications": []}

    a_et_b = [t for t in suivies
              if univers.loc[univers["ticker"] == t, "strate"].iloc[0] in ("A", "B")]
    publications = calendrier(a_et_b)

    alertes = (alertes_prix(univers, cours)
               + alerte_concentration(univers, cours)
               + alertes_concurrents(pairs, cours)
               + alertes_calendrier(publications, univers))

    print(f"{len(alertes)} alerte(s) brute(s), "
          f"{len(publications)} publication(s) au calendrier.")
    return {"alertes": alertes, "publications": publications, "cours": cours}
