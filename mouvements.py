"""
Journal des mouvements : positions et plus-values a partir des operations.

Une ligne par achat ou par vente. Les quantites et les prix de revient ne sont
plus saisis, ils sont deduits. Une cloture partielle est une vente ordinaire,
une cloture totale une vente qui ramene la quantite a zero — aucun cas
particulier a gerer.

Le prix de revient suit la methode du cout moyen pondere, celle qu'impose
l'administration fiscale francaise pour les titres fongibles : une vente ne
choisit pas quels titres partent, elle sort au PRU du moment. La plus-value
realisee est donc definitive des la vente, et les achats posterieurs ne la
modifient pas.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


# En-tetes attendus dans l'onglet, dans cet ordre.
COLONNES = ["Date", "Ticker", "Sens", "Quantité", "Prix unitaire",
            "Devise", "Frais", "Note"]

ACHAT, VENTE, DIVISION = "ACHAT", "VENTE", "DIVISION"

# Variantes tolerees pour chaque colonne, en minuscules sans accents.
ALIAS = {
    "Date": ["date", "jour"],
    "Ticker": ["ticker", "symbole", "code", "valeur"],
    "Sens": ["sens", "operation", "type", "nature"],
    "Quantité": ["quantite", "qte", "nombre", "titres"],
    "Prix unitaire": ["prix unitaire", "prix", "cours", "prix d'achat",
                      "prix de vente"],
    "Devise": ["devise", "monnaie", "currency"],
    "Frais": ["frais", "commission", "courtage"],
    "Note": ["note", "commentaire", "remarque"],
}

SENS_ACHAT = {"achat", "a", "buy", "b", "acquisition", "entree", "+"}
SENS_VENTE = {"vente", "v", "sell", "s", "cession", "sortie", "-"}
# Une division d'actions ne degage aucun resultat : elle multiplie la quantite
# et divise le prix de revient dans le meme rapport, a cout total inchange.
# Sans elle, un titre qui se divise par deux fait apparaitre une perte de 50 %
# qui n'existe pas. Le rapport se saisit dans la colonne Quantite : 2 pour une
# division par deux, 0,5 pour un regroupement de deux titres en un.
SENS_DIVISION = {"division", "split", "regroupement", "d", "div"}


# ==========================================================================
# Lecture
# ==========================================================================

def _sans_accents(texte: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return str(texte).strip().lower().translate(table)


def _nombre(valeur) -> float:
    """Convertit une saisie de tableur en nombre. Virgule, espaces, vide."""
    if valeur is None:
        return np.nan
    texte = str(valeur).strip()
    if not texte or texte.lower() in ("nan", "-", "—"):
        return np.nan
    texte = (texte.replace(" ", "").replace(" ", "")
                  .replace(" ", "").replace("€", "").replace("$", "")
                  .replace("%", ""))
    # Une virgule est un separateur decimal francais ; un point aussi.
    if "," in texte and "." in texte:
        texte = texte.replace(".", "").replace(",", ".")
    else:
        texte = texte.replace(",", ".")
    try:
        return float(texte)
    except ValueError:
        return np.nan


def _sens(valeur) -> str | None:
    mot = _sans_accents(valeur)
    if mot in SENS_ACHAT:
        return ACHAT
    if mot in SENS_VENTE:
        return VENTE
    if mot in SENS_DIVISION:
        return DIVISION
    return None


def lire(brut: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise l'onglet des mouvements.

    Les lignes inexploitables sont ecartees, pas corrigees : une operation
    dont la quantite ou le prix est illisible ne doit pas entrer dans un
    calcul de plus-value.
    """
    if brut is None or brut.empty:
        return pd.DataFrame(columns=COLONNES)

    correspondance = {}
    for colonne in brut.columns:
        nom = _sans_accents(colonne)
        for cible, variantes in ALIAS.items():
            if cible in correspondance.values():
                continue
            if nom in variantes or any(v in nom for v in variantes):
                correspondance[colonne] = cible
                break
    table = brut.rename(columns=correspondance)

    manquantes = [c for c in ("Date", "Ticker", "Sens", "Quantité",
                              "Prix unitaire") if c not in table.columns]
    if manquantes:
        raise ValueError(
            "Onglet des mouvements incomplet, colonnes absentes : "
            f"{', '.join(manquantes)}. Colonnes trouvees : "
            f"{', '.join(map(str, brut.columns))}.")

    lignes, rejets = [], []
    for numero, ligne in table.iterrows():
        ticker = str(ligne.get("Ticker", "")).strip().upper()
        sens = _sens(ligne.get("Sens"))
        quantite = _nombre(ligne.get("Quantité"))
        prix = _nombre(ligne.get("Prix unitaire"))
        date = pd.to_datetime(ligne.get("Date"), errors="coerce", dayfirst=True)

        if not ticker or ticker in ("NAN", "TICKER"):
            continue
        # Une division n'a pas de prix : la colonne Quantite porte le rapport.
        prix_requis = sens != DIVISION
        if sens is None or not np.isfinite(quantite) or quantite <= 0 \
                or pd.isna(date) \
                or (prix_requis and (not np.isfinite(prix) or prix <= 0)):
            rejets.append(int(numero) + 2)      # +2 : en-tete et base 1
            continue
        if not prix_requis and not np.isfinite(prix):
            prix = 0.0

        devise = str(ligne.get("Devise", "") or "").strip().upper() or None
        frais = _nombre(ligne.get("Frais"))
        lignes.append({
            "Date": date, "Ticker": ticker, "Sens": sens,
            "Quantité": quantite, "Prix unitaire": prix,
            "Devise": devise,
            "Frais": frais if np.isfinite(frais) else 0.0,
            "Note": str(ligne.get("Note", "") or "").strip(),
            "Ligne": int(numero) + 2})

    resultat = pd.DataFrame(lignes)
    if not resultat.empty:
        resultat = resultat.sort_values(["Date", "Ligne"]).reset_index(drop=True)
    resultat.attrs["rejets"] = rejets
    return resultat


# ==========================================================================
# Deroulement du journal
# ==========================================================================

def derouler(mvts: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Rejoue les operations dans l'ordre et en tire l'etat de chaque ligne.

    Retourne (positions, anomalies). Les montants restent dans la devise de
    l'operation : la conversion est un probleme distinct, traite ailleurs,
    parce qu'elle depend du cours de change du jour de l'operation.
    """
    colonnes = ["Ticker", "Devise", "Quantité", "PRU", "Investi",
                "PV réalisée", "Frais cumulés", "Dernier mouvement"]
    if mvts is None or mvts.empty:
        return pd.DataFrame(columns=colonnes), []

    etat: dict[str, dict] = {}
    anomalies: list[str] = []

    for _, m in mvts.iterrows():
        t = m["Ticker"]
        e = etat.setdefault(t, {
            "quantite": Decimal(0), "cout": Decimal(0),
            "pv": Decimal(0), "frais": Decimal(0),
            "devise": m["Devise"], "derniere": m["Date"]})

        if m["Devise"] and e["devise"] and m["Devise"] != e["devise"]:
            anomalies.append(
                f"{t} : devise {m['Devise']} en ligne {m['Ligne']} alors que "
                f"les operations precedentes sont en {e['devise']}.")
        if m["Devise"]:
            e["devise"] = m["Devise"]
        e["derniere"] = m["Date"]

        try:
            q = Decimal(str(m["Quantité"]))
            p = Decimal(str(m["Prix unitaire"]))
            f = Decimal(str(m["Frais"]))
        except InvalidOperation:
            anomalies.append(f"{t} : nombre illisible en ligne {m['Ligne']}.")
            continue

        e["frais"] += f

        if m["Sens"] == DIVISION:
            if e["quantite"] <= 0:
                anomalies.append(
                    f"{t} : division en ligne {m['Ligne']} sans position "
                    f"detenue a cette date. Operation ignoree.")
                continue
            e["quantite"] *= q
            continue    # le cout total ne change pas, donc le PRU suit seul

        if m["Sens"] == ACHAT:
            e["quantite"] += q
            e["cout"] += q * p + f
            continue

        # Vente
        if e["quantite"] <= 0:
            anomalies.append(
                f"{t} : vente en ligne {m['Ligne']} sur une position deja "
                f"soldee ou jamais achetee. Operation ignoree.")
            continue
        if q > e["quantite"]:
            anomalies.append(
                f"{t} : vente de {q:g} titres en ligne {m['Ligne']} alors que "
                f"la position n'en compte que {float(e['quantite']):g}. "
                f"Vente ramenee a la quantite detenue.")
            q = e["quantite"]

        pru = e["cout"] / e["quantite"] if e["quantite"] > 0 else Decimal(0)
        e["pv"] += q * (p - pru) - f
        e["cout"] -= q * pru
        e["quantite"] -= q
        if e["quantite"] == 0:
            e["cout"] = Decimal(0)

    lignes = []
    for t, e in etat.items():
        quantite = float(e["quantite"])
        lignes.append({
            "Ticker": t,
            "Devise": e["devise"],
            "Quantité": quantite,
            "PRU": float(e["cout"] / e["quantite"]) if e["quantite"] > 0 else np.nan,
            "Investi": float(e["cout"]),
            "PV réalisée": float(e["pv"]),
            "Frais cumulés": float(e["frais"]),
            "Dernier mouvement": e["derniere"]})

    return (pd.DataFrame(lignes).set_index("Ticker").sort_index(), anomalies)


def piste(mvts: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Deroule les operations d'une ligne en montrant l'etat apres chacune.

    C'est la contrepartie indispensable d'un calcul automatique : sans moyen de
    verifier une plus-value operation par operation, il faut croire le
    programme sur parole. Ici chaque vente affiche le PRU qui lui a ete
    applique et le resultat qu'elle a degage.
    """
    colonnes = ["Date", "Sens", "Quantité", "Prix unitaire", "Frais",
                "PRU appliqué", "Résultat", "Quantité après", "PRU après"]
    if mvts is None or mvts.empty:
        return pd.DataFrame(columns=colonnes)

    operations = mvts[mvts["Ticker"] == ticker.upper()]
    if operations.empty:
        return pd.DataFrame(columns=colonnes)

    q_cum, cout = Decimal(0), Decimal(0)
    lignes = []
    for _, m in operations.iterrows():
        q = Decimal(str(m["Quantité"]))
        p = Decimal(str(m["Prix unitaire"]))
        f = Decimal(str(m["Frais"]))
        pru_applique, resultat = np.nan, np.nan

        if m["Sens"] == DIVISION:
            if q_cum > 0:
                q_cum *= q
        elif m["Sens"] == ACHAT:
            q_cum += q
            cout += q * p + f
        elif q_cum > 0:
            vendue = min(q, q_cum)
            pru = cout / q_cum
            pru_applique = float(pru)
            resultat = float(vendue * (p - pru) - f)
            cout -= vendue * pru
            q_cum -= vendue
            if q_cum == 0:
                cout = Decimal(0)

        lignes.append({
            "Date": m["Date"], "Sens": m["Sens"],
            "Quantité": float(m["Quantité"]),
            "Prix unitaire": float(m["Prix unitaire"]) or np.nan,
            "Frais": float(m["Frais"]),
            "PRU appliqué": pru_applique,
            "Résultat": resultat,
            "Quantité après": float(q_cum),
            "PRU après": float(cout / q_cum) if q_cum > 0 else np.nan})

    return pd.DataFrame(lignes)


def doublons(mvts: pd.DataFrame) -> list[str]:
    """
    Operations rigoureusement identiques : meme date, sens, quantite et prix.

    Ce n'est pas forcement une erreur — deux ordres identiques le meme jour
    arrivent — mais c'est la trace la plus courante d'un copier-coller
    malheureux, et une operation dupliquee faussse durablement le PRU.
    """
    if mvts is None or mvts.empty:
        return []
    cles = ["Date", "Ticker", "Sens", "Quantité", "Prix unitaire"]
    groupes = mvts.groupby(cles, dropna=False)["Ligne"].apply(list)
    return [f"{c[1]} : {len(l)} operations identiques du "
            f"{pd.Timestamp(c[0]).strftime('%d/%m/%Y')} "
            f"({c[2].lower()} de {c[3]:g} a {c[4]:g}), lignes "
            f"{', '.join(map(str, l))}."
            for c, l in groupes.items() if len(l) > 1]


def cloturees(positions: pd.DataFrame) -> pd.DataFrame:
    """Lignes entierement sorties : quantite nulle, plus-value acquise."""
    if positions.empty:
        return positions
    return positions[positions["Quantité"] <= 0]


def ouvertes(positions: pd.DataFrame) -> pd.DataFrame:
    """Lignes encore detenues."""
    if positions.empty:
        return positions
    return positions[positions["Quantité"] > 0]


# ==========================================================================
# Plus-values realisees, converties
# ==========================================================================

def pv_convertie(mvts: pd.DataFrame, positions: pd.DataFrame,
                 change) -> pd.Series:
    """
    Plus-values realisees converties dans la devise de reference.

    `change` est une fonction (devise, date) -> taux. Le taux du jour de la
    vente est utilise, et non celui d'aujourd'hui : une plus-value realisee
    est un fait date, elle ne doit plus bouger ensuite.
    """
    if mvts is None or mvts.empty or positions.empty:
        return pd.Series(dtype=float)

    cumul: dict[str, float] = {}
    etat: dict[str, dict] = {}

    for _, m in mvts.iterrows():
        t = m["Ticker"]
        e = etat.setdefault(t, {"q": Decimal(0), "c": Decimal(0)})
        q = Decimal(str(m["Quantité"]))
        p = Decimal(str(m["Prix unitaire"]))
        f = Decimal(str(m["Frais"]))

        if m["Sens"] == DIVISION:
            if e["q"] > 0:
                e["q"] *= q
            continue

        if m["Sens"] == ACHAT:
            e["q"] += q
            e["c"] += q * p + f
            continue

        if e["q"] <= 0:
            continue
        if q > e["q"]:
            q = e["q"]
        if q <= 0:
            continue
        pru = e["c"] / e["q"]
        brute = float(q * (p - pru) - f)
        taux = change(m["Devise"], m["Date"])
        cumul[t] = cumul.get(t, 0.0) + brute * (taux if taux else 1.0)
        e["c"] -= q * pru
        e["q"] -= q
        if e["q"] == 0:
            e["c"] = Decimal(0)

    return pd.Series(cumul, dtype=float).reindex(positions.index).fillna(0.0)
