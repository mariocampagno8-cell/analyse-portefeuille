"""
Controle des donnees saisies, avant tout calcul.

Le moteur de calcul est desormais teste ; les erreurs qui restent viennent des
donnees. Elles ont une propriete desagreable : elles ne provoquent aucune
exception. Une devise erronee ne fait pas planter l'application, elle deplace
silencieusement une valeur, un poids, une part de risque, et contamine les
seuils qui en decoulent. Un outil qui ne les detecte pas donne des chiffres
faux avec l'assurance des chiffres justes.

Trois familles de controles, par ordre de degat potentiel :

1. La devise. Confrontation entre ce qui est saisi et ce que declare la place
   de cotation. Le cas des pence merite un traitement a part : une valeur
   londonienne cote souvent en GBp, centieme de livre, et yfinance renvoie ses
   prix dans cette unite. Prendre des pence pour des livres multiplie une
   ligne par cent.

2. Le prix de saisie. Un prix unitaire qui n'a jamais ete approche par le
   titre a la date de l'operation trahit presque toujours une confusion entre
   montant total et prix unitaire, ou une virgule mal placee.

3. La coherence d'ensemble : ticker introuvable, quantite nulle, position
   dont la valeur depasse une part deraisonnable du portefeuille.

Aucun controle ne corrige : tous signalent. Corriger a la place de quelqu'un
une donnee qu'on n'a pas vue est le meilleur moyen de masquer un probleme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import reglages as rg

# Devises cotees en centiemes. La conversion n'est pas une question de taux
# mais d'unite : 453 GBp valent 4,53 GBP, pas 453.
CENTIEMES = {"GBP": "GBp", "ZAR": "ZAc", "ILS": "ILA"}

# Seuils des controles
ECARTS_PRIX = 3.0            # ecarts-types toleres sur un prix de saisie
MARGE_EXTREME = 0.5          # prix hors de [0,5 x plus bas ; 2 x plus haut]
CONCENTRATION_ALERTE = rg.CONCENTRATION  # % du portefeuille sur une ligne

GRAVITE = {"bloquant": 0, "serieux": 1, "informatif": 2}


def _anomalie(gravite: str, ticker: str, titre: str, detail: str,
              correction: str = "") -> dict:
    return {"gravite": gravite, "rang": GRAVITE.get(gravite, 2),
            "ticker": ticker, "titre": titre, "detail": detail,
            "correction": correction}


# ==========================================================================
# 1. Devises
# ==========================================================================

def controler_devise(ticker: str, saisie: str | None,
                     reelle: str | None) -> list[dict]:
    """
    Compare la devise saisie a celle que declare la place de cotation.

    `reelle` est la devise renvoyee par la source de cours ; elle peut valoir
    GBp, ZAc ou ILA, qui sont des centiemes et non des devises distinctes.
    """
    anomalies = []
    if not reelle:
        return anomalies

    centieme = reelle in CENTIEMES.values()
    principale = next((k for k, v in CENTIEMES.items() if v == reelle),
                      reelle)

    if centieme:
        anomalies.append(_anomalie(
            "bloquant", ticker, "Cotation en centièmes",
            f"{ticker} cote en {reelle}, c'est-à-dire en centièmes de "
            f"{principale}. Les cours téléchargés sont dans cette unité : "
            f"une valeur affichée 453 vaut 4,53 {principale}.",
            f"Divise par 100 les prix saisis pour cette ligne, ou vérifie "
            f"que l'application applique bien le facteur."))

    if not saisie:
        anomalies.append(_anomalie(
            "informatif", ticker, "Devise non renseignée",
            f"La devise de {ticker} est déduite de la place de cotation "
            f"({principale}). Yahoo se trompe régulièrement sur les petites "
            f"capitalisations européennes.",
            f"Renseigne {principale} dans la colonne Devise pour figer "
            f"la valeur."))
        return anomalies

    if saisie.upper() != principale.upper():
        anomalies.append(_anomalie(
            "bloquant", ticker, "Devise contredite par la place",
            f"{ticker} est saisie en {saisie} alors que sa place de cotation "
            f"déclare {principale}. Toute la ligne — valeur, poids, part de "
            f"risque, seuils — est convertie au mauvais taux.",
            f"Remplace {saisie} par {principale} dans la colonne Devise."))

    return anomalies


# ==========================================================================
# 2. Prix de saisie
# ==========================================================================

def controler_prix(ticker: str, prix: float, date, cours: pd.Series,
                   quantite: float = np.nan) -> list[dict]:
    """
    Verifie qu'un prix d'operation est plausible au regard de l'historique.

    Deux mesures complementaires. L'ecart en nombre d'ecarts-types situe le
    prix dans la distribution des cours autour de la date. Le test des bornes
    attrape les erreurs grossieres — facteur dix, facteur cent, montant total
    saisi a la place du prix unitaire — que l'ecart-type laisserait passer sur
    un titre tres volatil.
    """
    anomalies = []
    serie = cours.dropna() if cours is not None else pd.Series(dtype=float)
    if serie.empty or not np.isfinite(prix) or prix <= 0:
        return anomalies

    # Fenetre de 30 seances autour de la date, ou tout l'historique a defaut.
    fenetre = serie
    try:
        jour = pd.Timestamp(date)
        proches = serie[(serie.index >= jour - pd.Timedelta(days=45)) &
                        (serie.index <= jour + pd.Timedelta(days=45))]
        if len(proches) >= 10:
            fenetre = proches
    except Exception:
        pass

    bas, haut = float(fenetre.min()), float(fenetre.max())
    moyenne = float(fenetre.mean())

    # Deux references distinctes, et c'est tout l'enjeu du controle.
    #
    # L'historique COMPLET dit si le prix est reel : un cours que le titre a
    # effectivement cote un jour n'est pas une erreur de saisie, meme s'il ne
    # correspond pas a la date indiquee. Premier essai de ce module : le test
    # des extremes portait sur la fenetre, et classait donc en bloquant un
    # prix authentique dont seule la date etait fausse.
    #
    # La FENETRE autour de la date dit si la date est coherente. Un prix reel
    # hors de la fenetre signale une date erronee — defaut serieux, puisqu'il
    # deplace le prix de revient, mais d'une autre nature.
    bas_h, haut_h = float(serie.min()), float(serie.max())

    # Le critere est l'intervalle reellement parcouru par le titre, pas un
    # nombre d'ecarts-types. Un titre en forte tendance s'ecarte souvent de
    # trois ecarts-types de sa moyenne sans que rien d'anormal ne se produise :
    # ce test-la produisait des alertes sur des prix parfaitement exacts. Si
    # le cours a traite a ce niveau, le prix est plausible, point.
    # La tolerance de 10 % couvre les extremes de seance, absents des
    # clotures dont on dispose.
    plancher, plafond = bas * 0.90, haut * 1.10

    def piste_unitaire() -> str:
        """Le montant total saisi a la place du prix unitaire est l'erreur
        la plus frequente : on la teste explicitement pour la nommer."""
        if not (np.isfinite(quantite) and quantite > 0):
            return ""
        unitaire = prix / quantite
        if plancher <= unitaire <= plafond:
            return (f" En divisant par la quantité ({quantite:g}), on obtient "
                    f"{unitaire:.2f}, qui est cohérent : le montant total a "
                    f"probablement été saisi à la place du prix unitaire.")
        return ""

    if prix < bas_h * MARGE_EXTREME or prix > haut_h / MARGE_EXTREME:
        rapport = (f"{prix / moyenne:.0f} fois" if prix > moyenne
                   else f"un {moyenne / prix:.0f}e") if moyenne > 0 else "?"
        anomalies.append(_anomalie(
            "bloquant", ticker, "Prix jamais coté par ce titre",
            f"{prix:.2f} saisi alors que {ticker} n'est jamais sorti de "
            f"l'intervalle {bas_h:.2f} – {haut_h:.2f} sur tout l'historique "
            f"disponible, soit {rapport} le cours moyen de la "
            f"période.{piste_unitaire()}",
            "Vérifie s'il s'agit bien d'un prix unitaire, et dans quelle "
            "unité — certaines places cotent en centièmes."))
    elif prix < plancher or prix > plafond:
        # Le prix existe dans l'historique : c'est la date qui ne colle pas.
        # Dire quand ce cours a ete atteint vaut mieux que constater l'ecart.
        ecart = (serie - prix).abs()
        proche_date = ecart.idxmin()
        quand = ""
        try:
            quand = (f" Ce cours correspond plutôt au "
                     f"{pd.Timestamp(proche_date).strftime('%d/%m/%Y')}.")
        except Exception:
            pass
        anomalies.append(_anomalie(
            "serieux", ticker, "Date probablement erronée",
            f"{prix:.2f} est un cours réel de {ticker}, mais le titre n'est "
            f"pas sorti de l'intervalle {bas:.2f} – {haut:.2f} dans les six "
            f"semaines entourant la date saisie.{quand}{piste_unitaire()}",
            "Corrige la date : elle décale le prix de revient et fausse les "
            "plus-values réalisées."))

    return anomalies


# ==========================================================================
# 3. Coherence d'ensemble
# ==========================================================================

def controler_positions(positions: pd.DataFrame, valeurs: pd.Series,
                        cours_connus: set) -> list[dict]:
    """Tickers introuvables, quantites nulles, concentration excessive."""
    anomalies = []
    # `.empty` vaut vrai pour un tableau qui a des lignes mais aucune colonne,
    # et les controles ci-dessous ne portent que sur l'index : tester la
    # longueur de l'index evite de sortir sans rien verifier.
    if positions is None or len(positions.index) == 0:
        return anomalies

    for t in positions.index:
        if t not in cours_connus:
            anomalies.append(_anomalie(
                "bloquant", t, "Aucun cours trouvé",
                f"{t} n'a renvoyé aucun cours. La ligne est absente de tous "
                f"les calculs : valeur, poids, risque, seuils.",
                "Vérifie l'orthographe exacte sur finance.yahoo.com. Les "
                "valeurs européennes portent un suffixe : .PA pour Paris, "
                ".AS pour Amsterdam, .L pour Londres."))

    total = float(valeurs.sum()) if len(valeurs) else 0.0
    if total > 0:
        for t, v in valeurs.items():
            part = float(v) / total * 100
            if part > CONCENTRATION_ALERTE:
                anomalies.append(_anomalie(
                    "serieux", t, f"Concentration : {part:.0f} % du portefeuille",
                    f"Une baisse de 30 % sur {t} coûterait {part * 0.3:.0f} % "
                    f"du portefeuille entier. À ce niveau, la taille de la "
                    f"position décide du résultat avant tout réglage de seuil.",
                    "Examine ce qu'un allègement ferait à la volatilité "
                    "d'ensemble avant d'ajuster quoi que ce soit d'autre."))

    return anomalies


# ==========================================================================
# Assemblage
# ==========================================================================

def _regrouper_dates(anomalies: list[dict], mvts: pd.DataFrame) -> list[dict]:
    """
    Remplace une serie d'alertes de date par une seule, quand elles partagent
    la meme date d'operation.

    Une position reprise en bloc porte souvent une date de convention — le
    premier jour de l'annee, la date de creation de la feuille. Signaler
    separement chaque ligne noie l'information dans la repetition alors qu'il
    n'y a qu'une chose a comprendre et une seule correction a faire.
    """
    dates_fautives = [a for a in anomalies if a["titre"] == "Date probablement erronée"]
    if len(dates_fautives) < 3 or mvts is None or mvts.empty:
        return anomalies

    concernes = {a["ticker"] for a in dates_fautives}
    lignes = mvts[mvts["Ticker"].isin(concernes)]
    dates = pd.to_datetime(lignes["Date"]).dt.normalize().unique()
    if len(dates) != 1:
        return anomalies

    jour = pd.Timestamp(dates[0]).strftime("%d/%m/%Y")
    autres = [a for a in anomalies if a["titre"] != "Date probablement erronée"]
    autres.append(_anomalie(
        "serieux", "—", f"{len(dates_fautives)} opérations datées du {jour}",
        f"Les prix saisis pour {', '.join(sorted(concernes))} ne correspondent "
        f"pas aux cours de cette date, mais à des cours réels d'autres "
        f"périodes. Tout indique une reprise de positions à une date de "
        f"convention plutôt que les dates d'achat effectives.",
        "Sans conséquence sur les quantités ni sur le prix de revient moyen, "
        "qui ne dépendent pas des dates. En revanche les plus-values seront "
        "mal datées, et la durée de détention inexacte. Corrige les dates si "
        "tu veux un suivi fiscal juste."))
    return autres


def controler(mvts: pd.DataFrame, positions: pd.DataFrame,
              valeurs: pd.Series, cours: pd.DataFrame,
              devises_reelles: dict) -> pd.DataFrame:
    """
    Passe tous les controles et renvoie les anomalies triees par gravite.

    Les colonnes sont stables : gravite, rang, ticker, titre, detail,
    correction. L'appelant decide de l'affichage.
    """
    anomalies = []
    connus = set(cours.columns) if cours is not None and not cours.empty else set()

    if positions is not None and not positions.empty:
        for t in positions.index:
            saisie = positions.at[t, "Devise"] if "Devise" in positions else None
            anomalies += controler_devise(t, saisie, devises_reelles.get(t))

    if mvts is not None and not mvts.empty:
        for _, m in mvts.iterrows():
            t = m["Ticker"]
            if t not in connus or m["Sens"] == "DIVISION":
                continue
            anomalies += controler_prix(t, float(m["Prix unitaire"]),
                                        m["Date"], cours[t],
                                        float(m["Quantité"]))

    anomalies += controler_positions(positions, valeurs, connus)
    anomalies = _regrouper_dates(anomalies, mvts)

    if not anomalies:
        return pd.DataFrame(columns=["gravite", "rang", "ticker", "titre",
                                     "detail", "correction"])
    table = pd.DataFrame(anomalies).sort_values(["rang", "ticker"])
    return table.reset_index(drop=True)
