"""
Lecture authentifiee des feuilles Google.

Remplace la publication sur le web par un acces nominatif. La difference est
importante : une feuille publiee est lisible par quiconque connait son
adresse, et cette adresse circule en clair dans les journaux d'execution et
l'historique de navigation. Un compte de service, lui, est un acces revocable
d'un clic, limite aux feuilles explicitement partagees avec lui.

Le module fonctionne avec ou sans compte de service. En son absence, il
retombe sur la lecture publique et le dit clairement — mieux vaut un
avertissement visible qu'une degradation silencieuse.
"""

from __future__ import annotations

import re
import sys

import pandas as pd


def _identifiant(url: str) -> str | None:
    """Extrait l'identifiant du classeur depuis n'importe quelle forme d'adresse."""
    trouve = re.search(r"/spreadsheets/d/(?:e/)?([a-zA-Z0-9-_]+)", url or "")
    return trouve.group(1) if trouve else None


def _gid(url: str) -> str | None:
    trouve = re.search(r"[#&?]gid=([0-9]+)", url or "")
    return trouve.group(1) if trouve else None


def disponible(st) -> bool:
    """Vrai si un compte de service est configure."""
    try:
        return bool(st.secrets.get("gcp_service_account"))
    except Exception:
        return False


def _client(st):
    """
    Client Google Sheets authentifie.

    Les dependances sont importees ici plutot qu'en tete de fichier : sans
    compte de service configure, elles ne sont pas necessaires et leur absence
    ne doit pas empecher l'application de demarrer.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    portee = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    identifiants = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=portee)
    return gspread.authorize(identifiants)


def lire(st, url: str, onglet: str | int | None = None) -> pd.DataFrame:
    """
    Charge une feuille, par compte de service si disponible.

    `onglet` accepte un nom, un index, ou rien — auquel cas le gid present
    dans l'adresse determine l'onglet, et a defaut le premier est retenu.
    """
    if not disponible(st):
        raise PermissionError(
            "Aucun compte de service configuré. Ajoute la section "
            "`[gcp_service_account]` dans les secrets Streamlit, ou continue "
            "avec une feuille publiée.")

    cle = _identifiant(url)
    if not cle:
        raise ValueError("Adresse de classeur non reconnue.")

    try:
        classeur = _client(st).open_by_key(cle)
    except Exception as erreur:
        nom = type(erreur).__name__
        if "SpreadsheetNotFound" in nom or "PermissionError" in nom or "403" in str(erreur):
            raise PermissionError(
                "Le compte de service n'a pas accès à ce classeur. Partage-le "
                "avec son adresse en `...iam.gserviceaccount.com`, en lecture."
            ) from erreur
        raise

    if isinstance(onglet, int):
        feuille = classeur.get_worksheet(onglet)
    elif isinstance(onglet, str) and onglet:
        feuille = classeur.worksheet(onglet)
    else:
        gid = _gid(url)
        feuille = None
        if gid:
            for candidate in classeur.worksheets():
                if str(candidate.id) == gid:
                    feuille = candidate
                    break
        feuille = feuille or classeur.sheet1

    valeurs = feuille.get_all_values()
    if not valeurs:
        return pd.DataFrame()

    entetes = valeurs[0]
    # Les colonnes sans intitulé reçoivent un nom stable plutôt qu'un vide,
    # qui casserait la correspondance ultérieure.
    entetes = [e.strip() if e.strip() else f"colonne_{i}"
               for i, e in enumerate(entetes)]
    return pd.DataFrame(valeurs[1:], columns=entetes).replace("", pd.NA)


def onglets(st, url: str) -> list[str]:
    """Noms des onglets d'un classeur, pour laisser l'utilisateur choisir."""
    if not disponible(st):
        return []
    cle = _identifiant(url)
    if not cle:
        return []
    try:
        return [f.title for f in _client(st).open_by_key(cle).worksheets()]
    except Exception:
        return []


def lire_ou_public(st, url: str, lecture_publique, onglet=None) -> tuple:
    """
    Tente la lecture authentifiee, retombe sur la lecture publique.

    Renvoie le tableau et la methode employee, pour que l'appelant puisse
    avertir l'utilisateur quand ses donnees transitent encore par une feuille
    publiee.
    """
    if disponible(st):
        try:
            return lire(st, url, onglet), "compte de service"
        except Exception as erreur:
            print(f"Lecture authentifiée impossible : {erreur}", file=sys.stderr)
    return lecture_publique(url), "feuille publiée"
