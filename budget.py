"""
Budget de notifications.

Contrainte appliquee par le systeme, non par la discipline. C'est le point le
plus important du dispositif : un canal qui deborde n'est plus lu, et un canal
qui n'est plus lu ne vaut rien, quelle que soit la qualite de son contenu.

Quatre regles, dans l'ordre d'application :

  1. Silence nocturne 21h-7h, sauf P1 sur une ligne detenue.
  2. Trois evenements de meme nature dans l'heure sont regroupes en un message.
  3. Quatre push sonores par jour au maximum. Au-dela, les P1 excedentaires
     partent en silencieux.
  4. Douze messages par jour au maximum. Au-dela, tout est reporte au digest.

L'arbitrage se fait par priorite puis par strate : un P1 sur une ligne detenue
passe toujours avant un P2 sur un candidat.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

FICHIER_BUDGET = Path(__file__).parent / "budget_notifications.json"

MAX_SONORES = 4
MAX_MESSAGES = 12
DEBUT_SILENCE = time(21, 0)
FIN_SILENCE = time(7, 0)
FENETRE_GROUPEMENT = timedelta(hours=1)
SEUIL_GROUPEMENT = 3

# Priorité puis strate : plus le rang est bas, plus l'envoi est prioritaire
RANG_PRIORITE = {"P1": 0, "P2": 1, "P3": 2}
RANG_STRATE = {"A": 0, "B": 1, "C": 2}


def _maintenant() -> datetime:
    return datetime.now()


def _aujourdhui() -> str:
    return _maintenant().strftime("%Y-%m-%d")


# ==========================================================================
# État
# ==========================================================================

def lire_etat() -> dict:
    """Compteurs du jour et historique recent, pour le groupement."""
    vide = {"date": _aujourdhui(), "sonores": 0, "messages": 0, "envoyes": []}
    if not FICHIER_BUDGET.exists():
        return vide
    try:
        etat = json.loads(FICHIER_BUDGET.read_text())
    except Exception:
        return vide

    # Les compteurs se remettent à zéro chaque jour
    if etat.get("date") != _aujourdhui():
        return vide

    limite = (_maintenant() - timedelta(hours=6)).isoformat()
    etat["envoyes"] = [e for e in etat.get("envoyes", [])
                       if e.get("horodatage", "") > limite]
    return etat


def ecrire_etat(etat: dict) -> None:
    try:
        FICHIER_BUDGET.write_text(json.dumps(etat, indent=1, ensure_ascii=False))
    except Exception:
        pass


# ==========================================================================
# Règles
# ==========================================================================

def en_silence(quand: datetime | None = None) -> bool:
    """Vrai pendant la plage nocturne."""
    heure = (quand or _maintenant()).time()
    return heure >= DEBUT_SILENCE or heure < FIN_SILENCE


def sonore_autorise(alerte: dict, etat: dict) -> bool:
    """
    Determine si une alerte declenche le son.

    Le son ne decoule pas mecaniquement de la priorite. Certaines alertes de
    rang P2 doivent sonner — le franchissement d'un prix d'entree en strate B
    est celle que l'on attend vraiment, et la manquer vide le systeme de son
    interet. Le champ `sonore` permet de le declarer explicitement.

    Pendant la plage nocturne, seul un P1 sur une ligne detenue passe : rien
    d'autre ne justifie de reveiller quelqu'un.
    """
    if alerte.get("strate") == "C":
        return False
    if not (alerte.get("priorite") == "P1" or alerte.get("sonore")):
        return False
    if en_silence():
        return alerte.get("strate") == "A" and alerte.get("priorite") == "P1"
    return etat.get("sonores", 0) < MAX_SONORES


def grouper(alertes: list[dict], etat: dict) -> list[dict]:
    """
    Regroupe les evenements de meme nature survenus dans la meme heure.

    Trois dépêches sur la même société dans l'heure forment un seul message :
    au-delà, on ne lit plus, on subit.
    """
    familles = {}
    for alerte in alertes:
        cle = (alerte.get("nature", "autre"), alerte.get("ticker", ""))
        familles.setdefault(cle, []).append(alerte)

    sortie = []
    for (nature, ticker), groupe in familles.items():
        if len(groupe) < SEUIL_GROUPEMENT:
            sortie.extend(groupe)
            continue

        groupe.sort(key=lambda a: RANG_PRIORITE.get(a.get("priorite"), 9))
        principale = dict(groupe[0])
        principale["groupe"] = groupe
        principale["nombre_groupe"] = len(groupe)
        principale["titre_groupe"] = (
            f"{ticker} — {len(groupe)} événements ({nature})")
        sortie.append(principale)
    return sortie


def arbitrer(alertes: list[dict], etat: dict | None = None) -> dict:
    """
    Applique le budget et repartit les alertes.

    Renvoie ce qui part en push sonore, en push silencieux, et ce qui est
    reporte au digest. Rien n'est jamais perdu : le report est explicite.
    """
    etat = etat if etat is not None else lire_etat()

    alertes = grouper(alertes, etat)
    # Une alerte declaree sonore est traitee au rang de son importance reelle,
    # pas de son etiquette de priorite.
    alertes.sort(key=lambda a: (
        0 if a.get("sonore") else RANG_PRIORITE.get(a.get("priorite"), 9),
        RANG_STRATE.get(a.get("strate"), 9),
        RANG_PRIORITE.get(a.get("priorite"), 9)))

    sonores, silencieux, reportes = [], [], []
    compteur_sonores = etat.get("sonores", 0)
    compteur_messages = etat.get("messages", 0)

    for alerte in alertes:
        # P3 ne fait jamais l'objet d'un push
        if alerte.get("priorite") == "P3":
            reportes.append({**alerte, "motif_report": "P3 — digest uniquement"})
            continue

        if compteur_messages >= MAX_MESSAGES:
            reportes.append({**alerte,
                             "motif_report": f"plafond de {MAX_MESSAGES} messages"})
            continue

        if en_silence() and not (alerte.get("priorite") == "P1"
                                 and alerte.get("strate") == "A"):
            reportes.append({**alerte, "motif_report": "plage de silence"})
            continue

        if sonore_autorise(alerte, {"sonores": compteur_sonores}):
            sonores.append(alerte)
            compteur_sonores += 1
        else:
            silencieux.append(alerte)
        compteur_messages += 1

    return {
        "sonores": sonores,
        "silencieux": silencieux,
        "reportes": reportes,
        "compteurs": {"sonores": compteur_sonores, "messages": compteur_messages},
    }


def enregistrer(resultat: dict, etat: dict | None = None) -> None:
    """Met a jour les compteurs apres envoi effectif."""
    etat = etat if etat is not None else lire_etat()
    etat["date"] = _aujourdhui()
    etat["sonores"] = resultat["compteurs"]["sonores"]
    etat["messages"] = resultat["compteurs"]["messages"]

    horodatage = _maintenant().isoformat()
    for alerte in resultat["sonores"] + resultat["silencieux"]:
        etat.setdefault("envoyes", []).append({
            "horodatage": horodatage,
            "ticker": alerte.get("ticker", ""),
            "nature": alerte.get("nature", ""),
            "priorite": alerte.get("priorite", ""),
        })
    ecrire_etat(etat)


def etat_du_jour() -> dict:
    """Consommation du budget, pour affichage."""
    etat = lire_etat()
    return {
        "date": etat.get("date"),
        "sonores": f"{etat.get('sonores', 0)} / {MAX_SONORES}",
        "messages": f"{etat.get('messages', 0)} / {MAX_MESSAGES}",
        "silence": "oui" if en_silence() else "non",
        "reste_sonores": max(0, MAX_SONORES - etat.get("sonores", 0)),
        "reste_messages": max(0, MAX_MESSAGES - etat.get("messages", 0)),
    }
