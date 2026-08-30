"""
Lecture d'un portefeuille depuis une feuille Google Sheets.

Principe : une feuille publiee sur le web est accessible en CSV sans
authentification. Pas de compte de service, pas de cle d'API, pas de fichier
JSON — une simple URL.

Contrepartie : l'acces est en lecture seule et la feuille est accessible a
qui connait son adresse. N'y mets que des tickers et des quantites, jamais de
numero de compte ni de donnee personnelle.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

COLONNES_ATTENDUES = ["Ticker", "Quantité", "Prix d'achat"]

# Intitules acceptes pour chaque colonne, insensibles a la casse et aux accents
SYNONYMES = {
    "Ticker": ["ticker", "symbole", "symbol", "code", "valeur", "action",
               "isin", "titre"],
    "Quantité": ["quantite", "quantity", "qte", "qty", "nombre", "nb",
                 "parts", "actions", "shares"],
    "Prix d'achat": ["prix d'achat", "prix dachat", "prix achat", "pru",
                     "prix de revient", "cout", "cost", "price", "prix",
                     "achat", "prix unitaire"],
}


def _sans_accents(texte: str) -> str:
    remplacements = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return texte.lower().strip().translate(remplacements)


def url_csv(url: str) -> str:
    """
    Convertit n'importe quelle URL Google Sheets en URL d'export CSV.

    Accepte l'adresse d'edition, l'adresse de partage ou une URL publiee.
    L'identifiant de l'onglet (gid) est conserve quand il est present, ce qui
    permet de pointer un onglet precis d'un classeur.
    """
    url = url.strip()
    if not url:
        return ""

    # Deja au format CSV
    if "format=csv" in url or url.endswith(".csv") or "output=csv" in url:
        return url

    identifiant = re.search(r"/spreadsheets/d/(?:e/)?([a-zA-Z0-9-_]+)", url)
    if not identifiant:
        return url

    cle = identifiant.group(1)
    gid = re.search(r"[#&?]gid=([0-9]+)", url)
    suffixe = f"&gid={gid.group(1)}" if gid else ""

    # Les feuilles publiees ont un identifiant commencant par 2PACX
    if "/e/" in url or cle.startswith("2PACX"):
        return f"https://docs.google.com/spreadsheets/d/e/{cle}/pub?output=csv{suffixe}"
    return f"https://docs.google.com/spreadsheets/d/{cle}/export?format=csv{suffixe}"


def _normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Fait correspondre les intitules de la feuille aux colonnes attendues."""
    correspondance = {}
    for colonne in df.columns:
        propre = _sans_accents(str(colonne))
        for cible, variantes in SYNONYMES.items():
            if propre in variantes or any(v in propre for v in variantes):
                if cible not in correspondance.values():
                    correspondance[colonne] = cible
                break
    return df.rename(columns=correspondance)


def _nombre(valeur) -> float:
    """
    Convertit une saisie en nombre.

    Gere la virgule decimale francaise, les espaces de milliers, les symboles
    monetaires et les espaces insecables que Google Sheets insere parfois.
    """
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return float(valeur)
    if valeur is None:
        return np.nan
    texte = str(valeur).strip()
    if not texte:
        return np.nan
    texte = (texte.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
             .replace("€", "").replace("$", "").replace("£", ""))
    # 1.234,56 -> 1234.56 ; 1,234.56 -> 1234.56
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


def lire(url: str) -> pd.DataFrame:
    """
    Telecharge et nettoie le portefeuille depuis la feuille.

    Leve une ValueError explicite si la feuille est inaccessible ou mal
    structuree : mieux vaut un message clair qu'un portefeuille vide et
    silencieux.
    """
    adresse = url_csv(url)
    if not adresse:
        raise ValueError("Adresse de feuille vide.")

    try:
        df = pd.read_csv(adresse)
    except Exception as erreur:
        raise ValueError(
            "Feuille inaccessible. Vérifie qu'elle est bien publiée sur le web "
            f"(Fichier → Partager → Publier sur le web). Détail : {erreur}"
        ) from erreur

    if df.empty:
        raise ValueError("La feuille est vide.")

    df = _normaliser_colonnes(df)
    manquantes = [c for c in COLONNES_ATTENDUES if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes introuvables : {', '.join(manquantes)}. "
            f"La feuille contient : {', '.join(map(str, df.columns))}. "
            "La première ligne doit porter les intitulés Ticker, Quantité et "
            "Prix d'achat."
        )

    df = df[COLONNES_ATTENDUES].copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Quantité"] = df["Quantité"].map(_nombre)
    df["Prix d'achat"] = df["Prix d'achat"].map(_nombre)

    df = df[(df["Ticker"] != "") & (df["Ticker"] != "NAN")]
    df = df[df["Quantité"].fillna(0) > 0]

    if df.empty:
        raise ValueError(
            "Aucune ligne exploitable. Vérifie que les quantités sont des "
            "nombres positifs."
        )
    return df.reset_index(drop=True)


def diagnostic(df: pd.DataFrame) -> list[str]:
    """Anomalies de saisie qui n'empechent pas la lecture mais meritent un signalement."""
    alertes = []

    doublons = df[df["Ticker"].duplicated(keep=False)]["Ticker"].unique()
    if len(doublons):
        alertes.append(
            f"Tickers en double ({', '.join(doublons)}) — les lignes seront "
            "regroupées, avec un prix d'achat moyenné.")

    sans_prix = df[df["Prix d'achat"].isna() | (df["Prix d'achat"] <= 0)]
    if not sans_prix.empty:
        alertes.append(
            f"{len(sans_prix)} ligne(s) sans prix d'achat — la plus-value ne "
            "sera pas calculée pour celles-ci.")

    suspects = df[df["Ticker"].str.contains(r"\s", regex=True, na=False)]
    if not suspects.empty:
        alertes.append(
            f"Tickers contenant un espace ({', '.join(suspects['Ticker'])}) — "
            "probablement une erreur de saisie.")

    return alertes


MODELE = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "AIR.PA", "TTE.PA"],
    "Quantité": [12, 8, 25, 40],
    "Prix d'achat": [165.20, 310.00, 128.40, 55.10],
})
