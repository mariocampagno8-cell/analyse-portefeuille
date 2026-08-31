"""
Veille macroeconomique.

Trois rythmes complementaires, pour informer sans saturer :

  - LUNDI, apercu de la semaine : les echeances a venir, avec ce que chacune
    determine ;
  - CHAQUE MATIN, brief court : ce qui tombe aujourd'hui et ce qui a bouge sur
    les marches depuis la veille ;
  - A TOUT MOMENT, alerte : un mouvement anormal qui merite d'etre su tout de
    suite.

Le troisieme point est le plus delicat. Une alerte quotidienne sur un
mouvement banal detruit l'attention : les seuils sont donc volontairement
hauts, calibres sur des ecarts-types plutot que sur des pourcentages fixes,
pour s'adapter au regime de volatilite du moment.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
import macro as mc
import mise_en_forme as mf
import presse as pr

# Indicateurs suivis quotidiennement, avec leur seuil d'alerte en écarts-types
SUIVI = {
    "^VIX": ("VIX", 2.5),
    "^GSPC": ("S&P 500", 2.5),
    "^STOXX50E": ("Euro Stoxx 50", 2.5),
    "^TNX": ("Taux 10 ans US", 2.5),
    "DX-Y.NYB": ("Dollar", 2.5),
    "GC=F": ("Or", 2.5),
    "CL=F": ("Pétrole WTI", 2.5),
    "BTC-USD": ("Bitcoin", 3.0),
}

# Niveaux absolus qui changent la lecture du marché, indépendamment des écarts
SEUILS_ABSOLUS = {
    "^VIX": [(30, "Le VIX dépasse 30 : régime de stress."),
             (20, "Le VIX repasse au-dessus de 20 : fin du régime calme.")],
}


def envoyer(message: str) -> bool:
    """Envoi Telegram en texte brut."""
    jeton = os.environ.get("TELEGRAM_JETON", "").strip()
    destinataire = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    if not jeton or not destinataire:
        print("Telegram non configuré. Message :\n" + message, file=sys.stderr)
        return False
    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": destinataire, "text": message[:4000],
                  "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20,
        )
        if not reponse.ok:
            print(f"Telegram a refusé (code {reponse.status_code}) : "
                  f"{reponse.text[:250]}", file=sys.stderr)
        return reponse.ok
    except Exception as erreur:
        print(f"Échec Telegram : {erreur}", file=sys.stderr)
        return False


def charger_marches(periode: str = "1y") -> pd.DataFrame:
    """Cours de clôture des indicateurs suivis."""
    brut = yf.download(list(SUIVI), period=periode, interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    if brut.empty:
        return pd.DataFrame()
    cours = (brut["Close"] if isinstance(brut.columns, pd.MultiIndex)
             else brut[["Close"]])
    return cours.dropna(how="all").ffill()


# ==========================================================================
# Lundi : aperçu de la semaine
# ==========================================================================

def apercu_semaine() -> str:
    """Échéances des sept prochains jours, regroupées par jour."""
    calendrier = mc.calendrier_macro(mois_a_venir=1)
    limite = pd.Timestamp.now().normalize() + pd.Timedelta(days=7)
    semaine = calendrier[calendrier["Date"] <= limite]
    if semaine.empty:
        return ""

    jours = {}
    for date, groupe in semaine.groupby("Date"):
        libelle = f"{mf.JOURS[date.dayofweek].capitalize()} {date.strftime('%d/%m')}"
        jours[libelle] = [f"{e['Heure']} — {e['Événement']}"
                          for _, e in groupe.iterrows()]

    banques = semaine[semaine["Type"] == "Banque centrale"]
    temps_fort = ""
    if not banques.empty:
        premiere = banques.iloc[0]
        temps_fort = (f"{premiere['Événement']}, "
                      f"{mf.JOURS[premiere['Date'].dayofweek]} "
                      f"{premiere['Date'].strftime('%d/%m')} à {premiere['Heure']}.")
    return mf.message_semaine(jours, temps_fort)


# ==========================================================================
# Chaque matin : brief court
# ==========================================================================

def brief_quotidien(cours: pd.DataFrame) -> str:
    """Publications du jour et variations marquantes de la veille."""
    aujourdhui = pd.Timestamp.now().normalize()
    calendrier = mc.calendrier_macro(mois_a_venir=1)
    du_jour = calendrier[calendrier["Date"] == aujourdhui]

    agenda = [f"{e['Heure']} — {e['Événement']} ({e['Zone']})"
              for _, e in du_jour.iterrows()]

    marches = []
    if not cours.empty:
        variations = []
        for code, (libelle, _) in SUIVI.items():
            if code not in cours.columns:
                continue
            serie = cours[code].dropna()
            if len(serie) < 2:
                continue
            variation = float(serie.iloc[-1] / serie.iloc[-2] - 1) * 100
            variations.append((abs(variation), libelle,
                               float(serie.iloc[-1]), variation))
        variations.sort(reverse=True)
        for _, libelle, niveau, variation in variations[:5]:
            fleche = "▲" if variation > 0 else "▼"
            niveau_lisible = f"{niveau:,.0f}".replace(",", " ")
            marches.append(f"{fleche} {libelle} {niveau_lisible} "
                           f"({variation:+.1f} %)")

    contexte = ""
    if "^GSPC" in cours.columns:
        serie = cours["^GSPC"].dropna()
        if len(serie) > 200:
            mm = float(serie.rolling(200).mean().iloc[-1])
            au_dessus = float(serie.iloc[-1]) > mm
            contexte = ("Indice au-dessus de sa moyenne 200 séances : "
                        "régime favorable." if au_dessus else
                        "Indice sous sa moyenne 200 séances : régime prudent.")

    return mf.message_brief(agenda, marches, contexte)


# ==========================================================================
# Alertes de mouvement anormal
# ==========================================================================

def mouvements_anormaux(cours: pd.DataFrame) -> list[str]:
    """
    Variations dépassant le seuil, exprimé en écarts-types.

    Un seuil en pourcentage fixe est inadapté : 2 % sur le VIX n'a rien
    d'exceptionnel, 2 % sur le taux 10 ans est un événement. Normaliser par la
    volatilité récente rend le critère comparable entre actifs et robuste aux
    changements de régime.
    """
    alertes = []
    for code, (libelle, seuil) in SUIVI.items():
        if code not in cours.columns:
            continue
        serie = cours[code].dropna()
        if len(serie) < 60:
            continue

        rendements = serie.pct_change().dropna()
        dernier = float(rendements.iloc[-1])
        ecart_type = float(rendements.tail(120).std(ddof=1))
        if ecart_type <= 0:
            continue

        z = dernier / ecart_type
        if abs(z) >= seuil:
            sens = "bondit" if dernier > 0 else "chute"
            # Le séparateur de milliers est remplacé sur le NOMBRE seul :
            # l'appliquer à la phrase entière effacerait sa ponctuation.
            niveau_lisible = f"{serie.iloc[-1]:,.0f}".replace(",", " ")
            alertes.append(mf.message_urgence(
                f"{libelle} {dernier * 100:+.1f} %",
                f"{libelle} {sens} à {niveau_lisible}, soit "
                f"{abs(z):.1f} écarts-types.",
                "Seuil calibré sur la volatilité des 120 dernières séances."))

        for niveau, texte in SEUILS_ABSOLUS.get(code, []):
            avant, apres = float(serie.iloc[-2]), float(serie.iloc[-1])
            if avant < niveau <= apres:
                alertes.append(mf.message_urgence(
                    f"{libelle} franchit {niveau}", texte))
    return alertes


def alerte_courbe() -> str:
    """Signale un changement de pente de la courbe des taux."""
    ecart = mc.charger_serie("T10Y2Y")
    if ecart.empty or len(ecart) < 30:
        return ""
    actuel = float(ecart.iloc[-1])
    passe = float(ecart.iloc[-22]) if len(ecart) > 22 else actuel

    if passe < 0 <= actuel:
        return ("La courbe des taux se repentifie : l'écart 10 ans − 2 ans "
                f"repasse en positif ({actuel:+.2f}). Historiquement, la sortie "
                "d'inversion précède la récession de peu — l'inversion elle-même "
                "était le signal avancé.")
    if passe >= 0 > actuel:
        return ("La courbe des taux s'inverse : l'écart 10 ans − 2 ans passe "
                f"en négatif ({actuel:+.2f}). Le signal a précédé toutes les "
                "récessions américaines depuis 1955, avec six à dix-huit mois "
                "d'avance.")
    return ""


# ==========================================================================
# Exécution
# ==========================================================================

def principal() -> int:
    mode = os.environ.get("MODE_MACRO", "auto")
    aujourdhui = datetime.now()

    cours = charger_marches()
    print(f"{len(cours.columns) if not cours.empty else 0} indicateur(s) chargé(s).")

    messages = []

    if mode in ("auto", "semaine") and (aujourdhui.weekday() == 0
                                        or mode == "semaine"):
        apercu = apercu_semaine()
        if apercu:
            messages.append(apercu)

    # En mode urgence, pas de brief ni de presse : uniquement l'anormal.
    if mode in ("auto", "brief"):
        messages.append(brief_quotidien(cours))

    anomalies = mouvements_anormaux(cours)
    if anomalies:
        messages.extend(anomalies)      # un mouvement, un message

    courbe = alerte_courbe()
    if courbe:
        messages.append(mf.message_urgence("Courbe des taux", courbe))

    # Revue de presse : un message par article, jamais de bloc
    if mode != "urgence" and os.environ.get("PRESSE", "1") != "0":
        print("Collecte de la presse :")
        try:
            articles = pr.filtrer_nouveaux(pr.collecter())
            print(f"{len(articles)} article(s) nouveau(x) et pertinent(s).")
            for article in pr.selectionner(articles):
                messages.append(mf.message_article(
                    article["titre"], article["source"],
                    article.get("lien", ""), article.get("pourquoi", ""),
                    article.get("libre", True)))
        except Exception as erreur:
            print(f"Revue de presse indisponible : {type(erreur).__name__} — "
                  f"{erreur}", file=sys.stderr)

    print(f"{len(messages)} message(s) à envoyer.")
    envois = sum(1 for m in messages if m.strip() and envoyer(m))
    print(f"{envois} message(s) envoyé(s).")
    return 0 if envois or not messages else 1


if __name__ == "__main__":
    raise SystemExit(principal())
