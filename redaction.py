"""
Redaction des commentaires de signaux par Claude.

Deux ameliorations sur les gabarits :

  1. Le ton varie et s'ajuste a la situation, au lieu de repeter les memes
     tournures a chaque alerte.
  2. Le modele ARBITRE entre signaux contradictoires. Une valeur peut etre a
     la fois en queue de momentum et tres eloignee de son plus haut : le
     gabarit produit deux commentaires redondants, le modele en fait une
     lecture unique et hierarchisee.

Regle inchangee : le modele ne calcule rien. Il recoit des chiffres deja
etablis par Python et se contente de les interpreter. Si un chiffre manque, il
doit le dire plutot que de l'estimer.

Economie : un appel par societe et par passage, mis en cache sur la signature
des chiffres. Une situation inchangee ne redeclenche pas d'appel.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

FICHIER_CACHE = Path(__file__).parent / "cache_commentaires.json"
MODELE_DEFAUT = os.environ.get("MODELE_CLAUDE", "claude-sonnet-5")

CONSIGNES = """Tu rédiges des alertes pour un investisseur particulier français \
qui suit son portefeuille avec un outil quantitatif. Il connaît les bases : \
inutile de définir la volatilité ou le PER.

TA TÂCHE

On te donne une société, sa place dans le portefeuille, ses chiffres, et un ou \
plusieurs signaux qui viennent de se déclencher sur elle. Tu écris un \
commentaire par signal.

RÈGLES ABSOLUES

Ne calcule jamais. Tous les chiffres utiles sont fournis. Si une donnée manque, \
n'en parle pas — ne l'estime pas, ne la déduis pas. Un chiffre inventé ruine la \
confiance dans tout l'outil.

Ne recommande jamais d'acheter, de vendre ou de conserver. Tu expliques ce que \
le signal signifie pour cette société précisément, et ce qu'il faut en penser. \
La décision appartient au lecteur.

MANIÈRE

Deux à quatre phrases par signal. Dense, pas de remplissage.

Ancre chaque commentaire dans les chiffres de CETTE société : son poids dans le \
portefeuille, sa contribution au risque, sa valorisation, son historique de \
publications. Un commentaire qui vaudrait pour n'importe quelle valeur est un \
commentaire raté.

Quand plusieurs signaux se déclenchent sur la même société, relie-les. S'ils se \
contredisent, dis-le et explique lequel pèse le plus. C'est l'essentiel de ton \
utilité.

Signale les limites qui changent la lecture : peu d'analystes, historique court, \
signal affaibli par une divergence. Mais sans mettre une réserve partout.

Varie les formulations. Pas de « il convient de noter », pas de « il est \
important de souligner ». Écris comme un analyste qui parle à un collègue.

FORMAT

Réponds uniquement par un objet JSON, sans texte autour et sans balises de code. \
Une clé par type de signal reçu, la valeur étant le commentaire.

Exemple : {"momentum": "…", "plus_haut": "…"}"""


# ==========================================================================
# Cache
# ==========================================================================

def _signature(societe: dict, signaux: list[dict]) -> str:
    """Empreinte de la situation : memes chiffres, meme commentaire."""
    contenu = json.dumps(
        {"s": {k: v for k, v in sorted(societe.items()) if k != "nom"},
         "g": sorted([json.dumps(s, sort_keys=True, default=str)
                      for s in signaux])},
        sort_keys=True, default=str)
    return hashlib.sha256(contenu.encode()).hexdigest()[:20]


def _lire_cache() -> dict:
    if FICHIER_CACHE.exists():
        try:
            return json.loads(FICHIER_CACHE.read_text())
        except Exception:
            pass
    return {}


def _ecrire_cache(cache: dict) -> None:
    if len(cache) > 400:                       # on ne garde que le plus récent
        cache = dict(list(cache.items())[-400:])
    try:
        FICHIER_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    except Exception:
        pass


# ==========================================================================
# Appel au modèle
# ==========================================================================

def _client(cle: str):
    """
    Client Anthropic.

    Une cle « liee a une identite » exige un en-tete precisant l'espace de
    travail dans lequel elle agit. On le transmet quand la variable
    ANTHROPIC_WORKSPACE_ID est definie ; sinon l'appel echoue avec un message
    explicite que l'appelant remonte.
    """
    import anthropic
    espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    entetes = {"anthropic-workspace-id": espace} if espace else None
    return anthropic.Anthropic(api_key=cle, default_headers=entetes)


def rediger(societe: dict, signaux: list[dict], cle: str,
            modele: str = MODELE_DEFAUT, cache: dict | None = None) -> dict:
    """
    Commentaires pour tous les signaux d'une societe, en un seul appel.

    Renvoie un dictionnaire {type_de_signal: commentaire}. En cas d'echec,
    renvoie un dictionnaire vide — l'appelant retombera sur les gabarits.
    """
    if not signaux:
        return {}

    cache = _lire_cache() if cache is None else cache
    empreinte = _signature(societe, signaux)
    if empreinte in cache:
        return cache[empreinte]

    donnees = {
        "société": {k: v for k, v in societe.items() if v is not None},
        "signaux_déclenchés": signaux,
        "source": "yfinance, scraper non officiel de Yahoo Finance",
    }

    message = (
        "Rédige un commentaire pour chacun de ces signaux.\n\n"
        f"```json\n{json.dumps(donnees, ensure_ascii=False, indent=1, default=str)}\n```\n\n"
        "Types de signaux à commenter : "
        + ", ".join(s["type"] for s in signaux)
        + ".\n\nRéponds uniquement par le JSON demandé."
    )

    # Les erreurs réseau sont fréquentes et souvent passagères : on réessaie
    # deux fois, avec une attente croissante, avant de renoncer.
    passagères = ("APIConnectionError", "APITimeoutError", "RateLimitError",
                  "InternalServerError", "OverloadedError")
    reponse = None
    for tentative in range(3):
        try:
            reponse = _client(cle).messages.create(
                model=modele, max_tokens=1500, system=CONSIGNES,
                messages=[{"role": "user", "content": message}],
            )
            break
        except Exception as erreur:
            nom = type(erreur).__name__
            if nom in passagères and tentative < 2:
                print(f"   {nom}, nouvelle tentative dans "
                      f"{2 ** tentative} s…")
                time.sleep(2 ** tentative)
                continue
            _expliquer(erreur)
            return {}

    if reponse is None:
        return {}

    try:
        texte = "".join(bloc.text for bloc in reponse.content
                        if getattr(bloc, "type", "") == "text").strip()
        texte = texte.removeprefix("```json").removeprefix("```").removesuffix("```")
        commentaires = json.loads(texte.strip())
        if not isinstance(commentaires, dict):
            print("   Réponse inattendue du modèle, commentaires standard retenus.")
            return {}
        cache[empreinte] = commentaires
        _ecrire_cache(cache)
        return commentaires
    except json.JSONDecodeError:
        print("   Le modèle n'a pas renvoyé de JSON exploitable.")
        return {}


def _expliquer(erreur: Exception) -> None:
    """Affiche la cause d'un échec d'appel, et la piste de correction."""
    nom = type(erreur).__name__
    detail = str(erreur)[:400]
    print(f"Rédaction IA indisponible — {nom} : {detail}")

    cause = getattr(erreur, "__cause__", None)
    if cause:
        print(f"   cause sous-jacente : {type(cause).__name__} — {str(cause)[:250]}")

    if "workspace-id" in detail:
        print("   → Clé liée à une identité. Crée une clé dont la Portée est "
              "un espace de travail (Default), pas « identique au compte lié ».")
    elif nom == "AuthenticationError" or "401" in detail:
        print("   → Clé invalide, révoquée ou mal recopiée.")
    elif nom == "APIConnectionError":
        print("   → Échec réseau vers api.anthropic.com. Vérifie que le paquet "
              "`anthropic` est bien installé et que la clé n'est pas vide.")
    elif "credit" in detail.lower() or "balance" in detail.lower():
        print("   → Crédit épuisé sur le compte Anthropic.")
    print("   Les commentaires standard prennent le relais.")


def enrichir(alertes: list[dict], contextes: dict[str, dict],
             cle: str | None = None, modele: str = MODELE_DEFAUT) -> list[dict]:
    """
    Remplace les commentaires par leur version redigee, quand c'est possible.

    Sans cle, ou en cas d'echec de l'appel, les commentaires d'origine sont
    conserves : l'outil continue de fonctionner, simplement en moins fin.
    """
    cle = cle or os.environ.get("CLE_ANTHROPIC", "")
    if not cle or not alertes:
        return alertes

    cache = _lire_cache()
    par_societe: dict[str, list[dict]] = {}
    for alerte in alertes:
        par_societe.setdefault(alerte["ticker"], []).append(alerte)

    for ticker, groupe in par_societe.items():
        contexte = contextes.get(ticker)
        if not contexte:
            continue
        signaux = [a["donnees"] for a in groupe]
        commentaires = rediger(contexte, signaux, cle, modele, cache)
        for alerte in groupe:
            redige = commentaires.get(alerte["type"])
            if redige and isinstance(redige, str) and len(redige) > 40:
                alerte["commentaire_standard"] = alerte["commentaire"]
                alerte["commentaire"] = redige.strip()
                alerte["rédigé_par_ia"] = True

    return alertes


def cout_estime(nb_societes: int, modele: str = MODELE_DEFAUT) -> dict:
    """
    Ordre de grandeur du cout, en dollars.

    Environ 900 jetons d'entree et 400 de sortie par societe.
    Tarifs publics par million de jetons ; ils evoluent, voir claude.com/pricing.
    """
    tarifs = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0),
              "claude-haiku-4-5-20251001": (1.0, 5.0)}
    entree, sortie = tarifs.get(modele, (2.0, 10.0))
    cout = nb_societes * (900 * entree + 400 * sortie) / 1e6
    return {"sociétés": nb_societes,
            "coût_par_passage_dollars": round(cout, 4),
            "coût_mensuel_estimé_dollars": round(cout * 21, 2)}
