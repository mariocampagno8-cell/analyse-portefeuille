"""
Execution de la phase 1.

Enchaine la collecte, l'arbitrage par le budget, la mise en forme et l'envoi.
Un mode par moment de la journee :

  seance  — alertes de prix et de risque, plusieurs fois par jour
  matin   — rappel J-1 des publications du lendemain
  digest  — synthese du vendredi soir

Rien n'est envoye s'il n'y a rien a dire. Le silence est le comportement par
defaut, et c'est lui qui rend credibles les messages qui partent.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import budget as bu
import collecteur as co
import format_pro as fp
import strates as st

CATEGORIE_PAR_NATURE = {
    "prix_entree": "operation",
    "approche_entree": "prepublication",
    "seuil_vente": "avertissement",
    "mouvement": "risque",
    "prix_cible": "operation",
    "proximite_cible": "prepublication",
    "decrochage": "risque",
    "concentration": "risque",
    "concurrent": "risque",
    "prepublication": "prepublication",
}


# ==========================================================================
# Envoi
# ==========================================================================

def envoyer(message: str, silencieux: bool = False) -> bool:
    """Envoi Telegram, avec ou sans notification sonore."""
    jeton = os.environ.get("TELEGRAM_JETON", "").strip()
    destinataire = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    if not jeton or not destinataire:
        print("Telegram non configuré. Message :\n" + message, file=sys.stderr)
        return False

    valide, motif = fp.valider(message)
    if not valide:
        print(f"Message écarté : {motif}", file=sys.stderr)
        return False

    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": destinataire, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True,
                  "disable_notification": silencieux},
            timeout=20)
        if not reponse.ok:
            print(f"Telegram a refusé (code {reponse.status_code}) : "
                  f"{reponse.text[:250]}", file=sys.stderr)
        return reponse.ok
    except Exception as erreur:
        print(f"Échec Telegram : {type(erreur).__name__} — {erreur}",
              file=sys.stderr)
        return False


# ==========================================================================
# Mise en forme des alertes
# ==========================================================================

def formater(alerte: dict) -> str:
    """Un message par alerte, au format impose par le cahier des charges."""
    if alerte.get("nombre_groupe"):
        return formater_groupe(alerte)

    categorie = CATEGORIE_PAR_NATURE.get(alerte.get("nature"), "risque")
    detenue = alerte.get("strate") == "A"

    blocs = [fp.titre(categorie, alerte["ticker"], alerte["titre"], detenue)]
    if alerte.get("faits"):
        blocs.append(fp.liste_valeurs(alerte["faits"], largeur_cle=18))

    etiquette = {"A": "Portefeuille", "B": "Candidat", "C": "Veille"}.get(
        alerte.get("strate"), "")
    blocs.append(fp.source("Cours Yahoo Finance", complement=etiquette))
    return fp.assembler(*blocs)


def formater_groupe(alerte: dict) -> str:
    """Message unique pour des evenements de meme nature regroupes."""
    groupe = alerte.get("groupe", [])
    blocs = [fp.titre("digest", alerte["ticker"],
                      f"{len(groupe)} alertes groupées",
                      alerte.get("strate") == "A")]
    lignes = {}
    for element in groupe[:6]:
        lignes[element["titre"][:24]] = element.get("priorite", "")
    blocs.append(fp.liste_valeurs(lignes, largeur_cle=26))
    blocs.append(fp.italique("Regroupées : trois événements de même nature "
                             "dans l'heure."))
    return fp.assembler(*blocs)


# ==========================================================================
# Digest hebdomadaire
# ==========================================================================

def digest_hebdomadaire(univers: pd.DataFrame, cours: pd.DataFrame,
                        publications: list[dict], reportes: list[dict]) -> str:
    """
    Synthese du vendredi.

    Le seul message de la semaine qui prend du recul : ce qui a bouge, ce qui
    arrive, et ce qui a ete ecarte des notifications au fil des jours.
    """
    if cours.empty:
        return ""

    blocs = [fp.titre("digest", "SEMAINE", fp.date_fr())]

    # --- Mouvements notables
    mouvements = []
    for _, ligne in univers.iterrows():
        ticker = ligne["ticker"]
        if ticker not in cours.columns:
            continue
        prix = cours[ticker].dropna()
        if len(prix) < 6:
            continue
        variation = (float(prix.iloc[-1]) / float(prix.iloc[-6]) - 1) * 100
        if abs(variation) >= 5:
            mouvements.append([ticker[:8], ligne["strate"],
                               fp.nombre(variation, 1, "%", signe=True)])

    if mouvements:
        mouvements.sort(key=lambda l: -abs(float(
            l[2].replace("%", "").replace(",", ".").replace(" ", ""))))
        blocs.append(fp.sous_titre("Mouvements > 5 %") + "\n" + fp.tableau(
            mouvements[:10], entetes=["Valeur", "St.", "Semaine"],
            largeurs=[8, 3, 9]))
    else:
        blocs.append(fp.italique("Aucun mouvement supérieur à 5 % cette semaine."))

    # --- Publications à venir
    prochaines = [p for p in publications if p["jours"] <= 10]
    if prochaines:
        lignes = [[p["ticker"][:8], p["date"].strftime("%d/%m"),
                   f"J-{p['jours']}"] for p in prochaines[:8]]
        blocs.append(fp.sous_titre("Publications à venir") + "\n" + fp.tableau(
            lignes, entetes=["Valeur", "Date", "Délai"], largeurs=[8, 7, 6]))

    # --- Ce qui a été écarté des notifications
    if reportes:
        familles = {}
        for element in reportes:
            familles[element.get("nature", "autre")] = \
                familles.get(element.get("nature", "autre"), 0) + 1
        blocs.append(fp.sous_titre("Écarté des notifications") + "\n"
                     + fp.liste_valeurs(
                         {k: str(v) for k, v in familles.items()},
                         largeur_cle=20))

    # --- Revue de structure
    propositions = st.revue_trimestrielle(univers)
    a_signaler = {k: v for k, v in propositions.items() if v}
    if a_signaler:
        lignes = {}
        for cle, valeurs in a_signaler.items():
            libelle = {"retrograder_B_vers_C": "À rétrograder en C",
                       "sortir_de_C": "À sortir de la liste",
                       "these_a_mettre_a_jour": "Thèse à mettre à jour"}.get(cle, cle)
            noms = [v["ticker"] if isinstance(v, dict) else v for v in valeurs]
            lignes[libelle] = ", ".join(noms[:5])
        blocs.append(fp.sous_titre("Revue de structure") + "\n"
                     + fp.liste_valeurs(lignes, largeur_cle=22))

    compte = st.resume(univers)
    blocs.append(fp.liste_valeurs({
        "Strate A": str(compte.get("A", 0)),
        "Strate B": str(compte.get("B", 0)),
        "Strate C": str(compte.get("C", 0)),
    }, largeur_cle=10))

    blocs.append(fp.source("Cours Yahoo Finance"))
    return fp.assembler(*blocs)


# ==========================================================================
# Exécution
# ==========================================================================

def principal() -> int:
    mode = os.environ.get("MODE", "seance")
    url = os.environ.get("URL_UNIVERS", "").strip() \
        or os.environ.get("URL_SURVEILLANCE", "").strip()
    if not url:
        print("Variable URL_UNIVERS absente.", file=sys.stderr)
        return 1

    print(f"Mode : {mode}")
    try:
        univers = st.lire(url)
    except ValueError as erreur:
        print(f"Univers illisible : {erreur}", file=sys.stderr)
        return 1

    compte = st.resume(univers)
    print(f"Univers : {compte['A']} A, {compte['B']} B, {compte['C']} C.")
    for reclassement in compte.get("reclassées", []):
        print(f"  reclassée — {reclassement}")

    resultat = co.collecter(univers)
    alertes = resultat["alertes"]

    # Le rappel J-1 n'a de sens que le matin
    if mode != "matin":
        alertes = [a for a in alertes if a.get("nature") != "prepublication"]

    etat = bu.lire_etat()
    arbitrage = bu.arbitrer(alertes, etat)
    print(f"Arbitrage : {len(arbitrage['sonores'])} sonore(s), "
          f"{len(arbitrage['silencieux'])} silencieuse(s), "
          f"{len(arbitrage['reportes'])} reportée(s).")

    envois = 0
    for alerte in arbitrage["sonores"]:
        if envoyer(formater(alerte), silencieux=False):
            envois += 1
            time.sleep(1)
    for alerte in arbitrage["silencieux"]:
        if envoyer(formater(alerte), silencieux=True):
            envois += 1
            time.sleep(1)

    if mode == "digest":
        message = digest_hebdomadaire(univers, resultat.get("cours", pd.DataFrame()),
                                      resultat.get("publications", []),
                                      arbitrage["reportes"])
        if message and envoyer(message, silencieux=True):
            envois += 1

    if envois:
        bu.enregistrer(arbitrage, etat)

    consommation = bu.etat_du_jour()
    print(f"{envois} message(s) envoyé(s). "
          f"Budget : {consommation['sonores']} sonores, "
          f"{consommation['messages']} messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
