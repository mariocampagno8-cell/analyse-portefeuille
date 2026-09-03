"""
Journal de transactions — source unique de verite.

Idee reprise d'un cahier des charges plus ambitieux, et c'est la meilleure
qu'il contenait : le journal des mouvements est saisi, tout le reste en est
DERIVE. Positions, prix de revient, plus-values realisees, performance : rien
n'est stocke, tout se recalcule.

Ce que cela debloque, et qui etait impossible avec une simple photo des
positions :

  PRU EXACT      cout moyen pondere conforme a la methode fiscale francaise,
                 correct apres achats successifs, splits et ventes partielles
  PLUS-VALUES    montant realise par cession, base de la declaration
  TWR            performance neutralisee des apports, seule comparable a un
                 indice — un portefeuille qui recoit 10 000 EUR ne « gagne »
                 pas 10 000 EUR
  TRI            performance reellement obtenue compte tenu du calendrier des
                 versements, qui est une tout autre question

Le calcul monetaire utilise `Decimal`. Un `float` accumule des erreurs
d'arrondi qui se voient sur un PRU apres vingt operations : 0,1 + 0,2 ne fait
pas 0,3 en binaire, et une comptabilite ne pardonne pas ce genre d'ecart.
"""

from __future__ import annotations

import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

CENTIME = Decimal("0.01")
PRECISION = Decimal("0.000001")

TYPES = {
    "achat": "ACHAT", "buy": "ACHAT", "a": "ACHAT",
    "vente": "VENTE", "sell": "VENTE", "v": "ACHAT",
    "dividende": "DIVIDENDE", "div": "DIVIDENDE",
    "frais": "FRAIS", "fee": "FRAIS",
    "versement": "VERSEMENT", "depot": "VERSEMENT", "apport": "VERSEMENT",
    "retrait": "RETRAIT", "withdrawal": "RETRAIT",
    "split": "SPLIT", "division": "SPLIT",
}

COLONNES = {
    "date": ["date", "date operation", "jour"],
    "type": ["type", "operation", "sens", "nature"],
    "ticker": ["ticker", "symbole", "valeur", "code"],
    "quantite": ["quantite", "qte", "nombre", "titres"],
    "prix": ["prix", "cours", "prix unitaire", "pu"],
    "frais": ["frais", "commission", "courtage"],
    "devise": ["devise", "currency", "monnaie"],
    "note": ["note", "commentaire", "motif"],
}


def _propre(texte) -> str:
    return str(texte).lower().strip().translate(
        str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))


def _dec(valeur, defaut: str = "0") -> Decimal:
    """Conversion en decimal, tolerante aux formats francais."""
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return Decimal(defaut)
    if isinstance(valeur, Decimal):
        return valeur
    texte = (str(valeur).strip().replace("\u202f", "").replace("\xa0", "")
             .replace(" ", "").replace("€", "").replace("$", ""))
    if not texte:
        return Decimal(defaut)
    if "," in texte and "." in texte:
        texte = (texte.replace(".", "").replace(",", ".")
                 if texte.rindex(",") > texte.rindex(".") else texte.replace(",", ""))
    elif "," in texte:
        texte = texte.replace(",", ".")
    try:
        return Decimal(texte)
    except InvalidOperation:
        return Decimal(defaut)


# ==========================================================================
# Lecture du journal
# ==========================================================================

def lire(url: str) -> pd.DataFrame:
    """Charge le journal depuis une feuille Google publiee en CSV."""
    import veille as v            # réutilise la résolution d'adresse

    brut = None
    for adresse in v.url_csv(url):
        try:
            essai = pd.read_csv(adresse)
            if not essai.empty:
                brut = essai
                break
        except Exception:
            continue
    if brut is None:
        raise ValueError("Journal inaccessible. Vérifie que la feuille est "
                         "publiée au format CSV.")

    correspondance = {}
    for colonne in brut.columns:
        nom = _propre(colonne)
        for cible, variantes in COLONNES.items():
            if cible in correspondance.values():
                continue
            if nom in variantes or any(x in nom for x in variantes):
                correspondance[colonne] = cible
                break
    brut = brut.rename(columns=correspondance)

    manquantes = [c for c in ("date", "type", "ticker") if c not in brut.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes : {', '.join(manquantes)}. "
            f"La feuille contient : {', '.join(map(str, brut.columns))}.")

    lignes = []
    for numero, ligne in brut.iterrows():
        date = pd.to_datetime(ligne.get("date"), dayfirst=True, errors="coerce")
        if pd.isna(date):
            continue
        operation = TYPES.get(_propre(ligne.get("type", "")), None)
        if operation is None:
            continue

        lignes.append({
            "ligne": int(numero) + 2,          # numéro dans la feuille
            "date": date.normalize(),
            "type": operation,
            "ticker": str(ligne.get("ticker", "")).strip().upper(),
            "quantite": _dec(ligne.get("quantite")),
            "prix": _dec(ligne.get("prix")),
            "frais": _dec(ligne.get("frais")),
            "devise": str(ligne.get("devise", "EUR") or "EUR").strip().upper(),
            "note": str(ligne.get("note", "") or "").strip(),
        })

    journal = pd.DataFrame(lignes)
    return journal.sort_values(["date", "ligne"]).reset_index(drop=True)


def controler(journal: pd.DataFrame) -> list[dict]:
    """
    Verifie la coherence du journal avant tout calcul.

    Une vente superieure a la quantite detenue ou une date future revelent
    une erreur de saisie. Mieux vaut la signaler que produire un PRU faux.
    """
    anomalies = []
    detenu: dict[str, Decimal] = {}
    aujourdhui = pd.Timestamp.now().normalize()

    for _, ligne in journal.iterrows():
        ticker = ligne["ticker"]
        if ligne["date"] > aujourdhui:
            anomalies.append({
                "ligne": ligne["ligne"], "gravite": "erreur",
                "message": f"Date future ({ligne['date'].date()})."})

        if ligne["type"] == "ACHAT":
            detenu[ticker] = detenu.get(ticker, Decimal(0)) + ligne["quantite"]
        elif ligne["type"] == "VENTE":
            disponible = detenu.get(ticker, Decimal(0))
            if ligne["quantite"] > disponible:
                anomalies.append({
                    "ligne": ligne["ligne"], "gravite": "erreur",
                    "message": (f"Vente de {ligne['quantite']} {ticker} alors "
                                f"que {disponible} sont détenus.")})
            detenu[ticker] = disponible - ligne["quantite"]
        elif ligne["type"] == "SPLIT":
            if ligne["quantite"] <= 0:
                anomalies.append({
                    "ligne": ligne["ligne"], "gravite": "erreur",
                    "message": "Ratio de split absent ou nul."})

        if ligne["type"] in ("ACHAT", "VENTE") and ligne["prix"] <= 0:
            anomalies.append({
                "ligne": ligne["ligne"], "gravite": "erreur",
                "message": f"Prix nul sur un {ligne['type'].lower()}."})

    return anomalies


# ==========================================================================
# Positions et prix de revient
# ==========================================================================

def positions(journal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rejoue le journal et produit les positions et les cessions.

    Le prix de revient suit la methode du cout moyen pondere, retenue par
    l'administration fiscale francaise : chaque achat le recalcule, une vente
    le laisse inchange. Un split multiplie les quantites et divise le PRU,
    sans creer de plus-value.
    """
    etat: dict[str, dict] = {}
    cessions = []

    for _, ligne in journal.iterrows():
        ticker = ligne["ticker"]
        if ligne["type"] in ("VERSEMENT", "RETRAIT"):
            continue

        position = etat.setdefault(ticker, {
            "quantite": Decimal(0), "cout_total": Decimal(0),
            "frais_cumules": Decimal(0), "dividendes": Decimal(0),
            "realise": Decimal(0), "devise": ligne["devise"],
            "premiere_operation": ligne["date"]})

        if ligne["type"] == "ACHAT":
            montant = ligne["quantite"] * ligne["prix"] + ligne["frais"]
            position["quantite"] += ligne["quantite"]
            position["cout_total"] += montant
            position["frais_cumules"] += ligne["frais"]

        elif ligne["type"] == "VENTE":
            if position["quantite"] <= 0:
                continue
            # Le PRU du moment sert de base à la plus-value
            pru = position["cout_total"] / position["quantite"]
            quantite = min(ligne["quantite"], position["quantite"])
            produit = quantite * ligne["prix"] - ligne["frais"]
            cout = pru * quantite
            gain = produit - cout

            position["quantite"] -= quantite
            position["cout_total"] -= cout
            position["frais_cumules"] += ligne["frais"]
            position["realise"] += gain

            cessions.append({
                "Date": ligne["date"], "Ticker": ticker,
                "Quantité": float(quantite), "Prix de cession": float(ligne["prix"]),
                "PRU": float(pru), "Produit net": float(produit),
                "Coût d'acquisition": float(cout),
                "Plus-value": float(gain), "Devise": ligne["devise"]})

        elif ligne["type"] == "DIVIDENDE":
            position["dividendes"] += (ligne["quantite"] * ligne["prix"]
                                       if ligne["prix"] > 0 else ligne["quantite"])

        elif ligne["type"] == "SPLIT":
            # Un split ne crée aucune plus-value : le coût total est conservé
            ratio = ligne["quantite"]
            if ratio > 0:
                position["quantite"] *= ratio

        elif ligne["type"] == "FRAIS":
            position["frais_cumules"] += ligne["prix"] or ligne["quantite"]

    lignes = []
    for ticker, position in etat.items():
        if position["quantite"] <= 0 and position["realise"] == 0 \
                and position["dividendes"] == 0:
            continue
        quantite = position["quantite"]
        pru = (position["cout_total"] / quantite if quantite > 0 else Decimal(0))
        lignes.append({
            "Ticker": ticker,
            "Quantité": float(quantite),
            "PRU": float(pru.quantize(PRECISION, ROUND_HALF_UP)),
            "Investi": float(position["cout_total"].quantize(CENTIME, ROUND_HALF_UP)),
            "Frais cumulés": float(position["frais_cumules"]),
            "Dividendes": float(position["dividendes"]),
            "Plus-value réalisée": float(position["realise"]),
            "Devise": position["devise"],
            "Depuis": position["premiere_operation"]})

    return (pd.DataFrame(lignes).set_index("Ticker") if lignes else pd.DataFrame(),
            pd.DataFrame(cessions) if cessions else pd.DataFrame())


def flux(journal: pd.DataFrame) -> pd.Series:
    """Versements et retraits datés, pour le TWR et le TRI."""
    mouvements = journal[journal["type"].isin(["VERSEMENT", "RETRAIT"])]
    if mouvements.empty:
        return pd.Series(dtype=float)
    valeurs = []
    for _, ligne in mouvements.iterrows():
        montant = float(ligne["prix"] or ligne["quantite"])
        valeurs.append((ligne["date"],
                        montant if ligne["type"] == "VERSEMENT" else -montant))
    serie = pd.Series(dict(valeurs))
    return serie.groupby(level=0).sum().sort_index()


# ==========================================================================
# Performance
# ==========================================================================

def twr(valeurs: pd.Series, mouvements: pd.Series) -> pd.Series:
    """
    Rendement pondere par le temps.

    Neutralise l'effet des apports : c'est la seule mesure comparable a un
    indice. Un portefeuille qui recoit 10 000 EUR voit sa valeur grimper sans
    qu'aucune decision d'investissement n'y soit pour quelque chose.

    La serie est chainee sur les sous-periodes delimitees par les flux.
    """
    valeurs = valeurs.dropna().sort_index()
    if len(valeurs) < 2:
        return pd.Series(dtype=float)

    mouvements = (mouvements.reindex(valeurs.index).fillna(0)
                  if not mouvements.empty
                  else pd.Series(0.0, index=valeurs.index))

    rendements = []
    for i in range(1, len(valeurs)):
        debut = float(valeurs.iloc[i - 1])
        fin = float(valeurs.iloc[i])
        flux_jour = float(mouvements.iloc[i])
        # Le flux est réputé intervenir en début de période
        base = debut + flux_jour
        rendements.append(fin / base - 1 if base > 0 else 0.0)

    return pd.Series(rendements, index=valeurs.index[1:])


def tri(mouvements: pd.Series, valeur_finale: float,
        date_finale=None) -> float:
    """
    Taux de rendement interne, par recherche dichotomique.

    Repond a une question differente du TWR : combien ai-je reellement gagne,
    compte tenu du moment ou j'ai investi. Un investisseur qui a renforce au
    plus bas obtient un TRI superieur a son TWR.
    """
    if mouvements.empty:
        return float("nan")

    date_finale = date_finale or pd.Timestamp.now().normalize()
    dates = list(mouvements.index) + [date_finale]
    flux_totaux = [-float(m) for m in mouvements.values] + [float(valeur_finale)]
    origine = dates[0]
    annees = [(d - origine).days / 365.25 for d in dates]

    def valeur_actuelle(taux):
        return sum(f / (1 + taux) ** a for f, a in zip(flux_totaux, annees))

    bas, haut = -0.99, 10.0
    if valeur_actuelle(bas) * valeur_actuelle(haut) > 0:
        return float("nan")
    for _ in range(200):
        milieu = (bas + haut) / 2
        if valeur_actuelle(bas) * valeur_actuelle(milieu) <= 0:
            haut = milieu
        else:
            bas = milieu
    return (bas + haut) / 2


def valeurs_quotidiennes(journal: pd.DataFrame,
                         cours: pd.DataFrame) -> pd.Series:
    """
    Valeur du portefeuille jour par jour, reconstituee depuis le journal.

    Chaque jour de bourse, on rejoue les operations anterieures pour connaitre
    les quantites detenues, puis on valorise aux cours du jour.
    """
    if journal.empty or cours.empty:
        return pd.Series(dtype=float)

    debut = journal["date"].min()
    calendrier = cours.index[cours.index >= debut]
    if len(calendrier) == 0:
        return pd.Series(dtype=float)

    quantites: dict[str, Decimal] = {}
    valeurs = []
    prochaine = 0
    operations = journal.reset_index(drop=True)

    for jour in calendrier:
        while (prochaine < len(operations)
               and operations.loc[prochaine, "date"] <= jour):
            ligne = operations.loc[prochaine]
            ticker = ligne["ticker"]
            if ligne["type"] == "ACHAT":
                quantites[ticker] = quantites.get(ticker, Decimal(0)) + ligne["quantite"]
            elif ligne["type"] == "VENTE":
                quantites[ticker] = quantites.get(ticker, Decimal(0)) - ligne["quantite"]
            elif ligne["type"] == "SPLIT" and ligne["quantite"] > 0:
                quantites[ticker] = quantites.get(ticker, Decimal(0)) * ligne["quantite"]
            prochaine += 1

        total = 0.0
        for ticker, quantite in quantites.items():
            if quantite <= 0 or ticker not in cours.columns:
                continue
            serie = cours[ticker].loc[:jour].dropna()
            if not serie.empty:
                total += float(serie.iloc[-1]) * float(quantite)
        valeurs.append(total)

    return pd.Series(valeurs, index=calendrier)


MODELE = pd.DataFrame({
    "Date": ["15/01/2024", "20/03/2024", "12/06/2024", "05/09/2024", "10/01/2025"],
    "Type": ["Versement", "Achat", "Achat", "Dividende", "Vente"],
    "Ticker": ["", "AAPL", "AAPL", "AAPL", "AAPL"],
    "Quantité": ["", 10, 5, 15, 8],
    "Prix": [10000, 165.20, 190.50, 0.24, 228.00],
    "Frais": ["", 2.50, 2.50, "", 2.50],
    "Devise": ["EUR", "USD", "USD", "USD", "USD"],
    "Note": ["Versement initial", "Initiation", "Renforcement", "", "Allègement"],
})
