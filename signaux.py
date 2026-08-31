"""
Signaux documentes et commentaires contextualises.

Chaque signal produit une alerte accompagnee d'un commentaire construit a
partir des donnees propres a la societe : sa place dans ton portefeuille, sa
valorisation, son historique de publications, sa contribution au risque. Une
alerte generique du type « RSI a 28 » n'appelle aucune reflexion ; « ta plus
grosse ligne, 44 % sous son plus haut, avec des estimations en hausse » en
appelle une.

Les signaux retenus sont ceux dont la valeur predictive est documentee dans la
litterature academique :

  - momentum 12-1 (Jegadeesh & Titman, 1993) — l'anomalie la plus robuste,
    verifiee sur plus d'un siecle et sur tous les marches ;
  - derive post-annonce (Ball & Brown, 1968) — sous-reaction aux surprises
    de resultats, effet persistant six a neuf semaines ;
  - revision des estimations — la direction, jamais le niveau, qui est
    structurellement optimiste ;
  - proximite du plus haut a 52 semaines (George & Hwang, 2004) ;
  - qualite (Novy-Marx, 2013) — rentabilite brute rapportee aux actifs.

Reserve valable pour tous : McLean & Pontiff (2016) montrent que l'efficacite
des anomalies chute d'environ moitie apres publication academique. Ce qui
etait rentable en 1995 l'est nettement moins aujourd'hui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JOURS_BOURSE = 252


# ==========================================================================
# Contexte propre à chaque société
# ==========================================================================

def contexte_societe(ticker: str, cours: pd.Series, poids_pct: float,
                     part_risque_pct: float | None = None,
                     info: dict | None = None,
                     surprises: pd.DataFrame | None = None,
                     revisions: pd.DataFrame | None = None) -> dict:
    """
    Rassemble ce qui permet de personnaliser un commentaire.

    Tout ce qui manque vaut None : un commentaire incomplet reste utile,
    un commentaire faux ne l'est pas.
    """
    serie = cours.dropna()
    info = info or {}
    contexte = {
        "ticker": ticker,
        "nom": info.get("longName") or info.get("shortName") or ticker,
        "secteur": info.get("sector"),
        "poids_pct": round(float(poids_pct), 1),
        "part_risque_pct": (round(float(part_risque_pct), 1)
                            if part_risque_pct is not None
                            and np.isfinite(part_risque_pct) else None),
        "per": info.get("trailingPE"),
        "objectif_moyen": info.get("targetMeanPrice"),
        "analystes": info.get("numberOfAnalystOpinions"),
    }

    if len(serie) >= 2:
        contexte["cours"] = round(float(serie.iloc[-1]), 2)
    if len(serie) >= 253:
        contexte["perf_12m_pct"] = round(
            float(serie.iloc[-1] / serie.iloc[-252] - 1) * 100, 1)
        contexte["momentum_12_1_pct"] = round(
            float(serie.iloc[-22] / serie.iloc[-252] - 1) * 100, 1)
        contexte["ecart_plus_haut_pct"] = round(
            float(serie.iloc[-1] / serie.tail(252).max() - 1) * 100, 1)
    if len(serie) >= 63:
        contexte["perf_3m_pct"] = round(
            float(serie.iloc[-1] / serie.iloc[-63] - 1) * 100, 1)
        rendements = serie.pct_change().dropna()
        contexte["volatilite_pct"] = round(
            float(rendements.tail(252).std(ddof=1) * np.sqrt(JOURS_BOURSE) * 100), 1)
    if len(serie) >= 200:
        mm = float(serie.rolling(200).mean().iloc[-1])
        contexte["vs_mm200_pct"] = round(float(serie.iloc[-1] / mm - 1) * 100, 1)

    if contexte.get("objectif_moyen") and contexte.get("cours"):
        contexte["potentiel_consensus_pct"] = round(
            (contexte["objectif_moyen"] / contexte["cours"] - 1) * 100, 1)

    if surprises is not None and not surprises.empty \
            and "Surprise (%)" in surprises.columns:
        valides = surprises["Surprise (%)"].dropna().head(8)
        if len(valides):
            contexte["surprise_derniere_pct"] = round(float(valides.iloc[0]), 1)
            contexte["taux_depassement_pct"] = round(
                float((valides > 0).mean() * 100), 0)

    if revisions is not None and not revisions.empty:
        colonne = "Révision sur 90 jours (%)"
        if colonne in revisions.columns:
            valeur = revisions[colonne].dropna()
            if len(valeur):
                contexte["revision_90j_pct"] = round(float(valeur.iloc[0]), 1)

    return contexte


# ==========================================================================
# Signaux
# ==========================================================================

def classement_momentum(cours: pd.DataFrame) -> pd.Series:
    """
    Momentum 12-1 : performance sur douze mois, dernier mois exclu.

    Le mois le plus recent est ecarte parce qu'il presente un effet de
    retournement a court terme qui degrade le signal.
    """
    scores = {}
    for ticker in cours.columns:
        serie = cours[ticker].dropna()
        if len(serie) < 253:
            continue
        scores[ticker] = float(serie.iloc[-22] / serie.iloc[-252] - 1) * 100
    return pd.Series(scores).sort_values(ascending=False)


def signal_momentum(classement: pd.Series, ticker: str,
                    seuil_tiers: float = 1 / 3) -> dict | None:
    """Position du titre dans le classement momentum de l'univers suivi."""
    if ticker not in classement.index or len(classement) < 4:
        return None
    rang = int(classement.index.get_loc(ticker)) + 1
    total = len(classement)
    quantile = rang / total

    if quantile <= seuil_tiers:
        etat = "tête"
    elif quantile >= 1 - seuil_tiers:
        etat = "queue"
    else:
        return None

    return {"type": "momentum", "etat": etat, "rang": rang, "total": total,
            "score_pct": round(float(classement[ticker]), 1)}


def signal_derive_resultats(contexte: dict, seuil_surprise: float = 5.0) -> dict | None:
    """
    Derive post-annonce : sous-reaction du marche aux surprises de resultats.

    L'effet dure six a neuf semaines apres la publication. Le signal se
    renforce quand la surprise s'accompagne de revisions a la hausse.
    """
    surprise = contexte.get("surprise_derniere_pct")
    if surprise is None or abs(surprise) < seuil_surprise:
        return None
    revision = contexte.get("revision_90j_pct")
    return {
        "type": "derive_resultats",
        "sens": "positive" if surprise > 0 else "négative",
        "surprise_pct": surprise,
        "revision_pct": revision,
        "confirme": bool(revision is not None
                         and np.sign(revision) == np.sign(surprise)),
    }


def signal_revisions(contexte: dict, seuil: float = 3.0) -> dict | None:
    """Momentum des revisions d'estimations, dans un sens ou dans l'autre."""
    revision = contexte.get("revision_90j_pct")
    if revision is None or abs(revision) < seuil:
        return None
    return {"type": "revisions",
            "sens": "hausse" if revision > 0 else "baisse",
            "revision_pct": revision}


def signal_plus_haut(contexte: dict, seuil_proche: float = -5.0,
                     seuil_loin: float = -30.0) -> dict | None:
    """
    Proximite du plus haut a 52 semaines.

    Resultat contre-intuitif de George et Hwang : les titres proches de leur
    plus haut surperforment, alors que l'instinct pousse a acheter bas.
    """
    ecart = contexte.get("ecart_plus_haut_pct")
    if ecart is None:
        return None
    if ecart >= seuil_proche:
        return {"type": "plus_haut", "etat": "proche", "ecart_pct": ecart}
    if ecart <= seuil_loin:
        return {"type": "plus_haut", "etat": "éloigné", "ecart_pct": ecart}
    return None


def signal_regime(indice: pd.Series) -> dict | None:
    """
    Regime de marche : position de l'indice face a sa moyenne 200 seances.

    Ce n'est pas un signal d'achat mais un filtre de risque. Rester a l'ecart
    sous la moyenne ne fait pas gagner davantage, mais reduit sensiblement les
    pertes maximales — ce qui compte quand on est concentre.
    """
    serie = indice.dropna()
    if len(serie) < 202:
        return None
    mm = serie.rolling(200).mean()
    avant, apres = serie.iloc[-2] > mm.iloc[-2], serie.iloc[-1] > mm.iloc[-1]
    if avant == apres:
        return None
    return {"type": "regime",
            "etat": "favorable" if apres else "défavorable",
            "ecart_pct": round(float(serie.iloc[-1] / mm.iloc[-1] - 1) * 100, 1)}


def signal_dimensionnement(contexte: dict, facteur: float = 2.0) -> dict | None:
    """
    Ecart entre le poids d'une ligne et sa contribution reelle au risque.

    Une position qui pese 20 % mais porte 45 % du risque n'est pas une ligne
    parmi d'autres : c'est le portefeuille.
    """
    poids = contexte.get("poids_pct")
    risque = contexte.get("part_risque_pct")
    if poids is None or risque is None or poids <= 0:
        return None
    ratio = risque / poids
    if ratio < facteur:
        return None
    return {"type": "dimensionnement", "poids_pct": poids,
            "risque_pct": risque, "ratio": round(ratio, 2)}


def signal_retournement_momentum(rendements_ptf: pd.Series,
                                 exposition_momentum: float,
                                 multiple: float = 1.8) -> dict | None:
    """
    Precurseur des krachs de momentum.

    La strategie momentum subit des retournements brutaux — pres de 40 % de
    perte en trois mois en 2009 — qui surviennent apres un pic de volatilite
    en marche baissier. Le signal combine les deux conditions.
    """
    r = rendements_ptf.dropna()
    if len(r) < 260 or exposition_momentum < 0.4:
        return None
    courte = float(r.tail(21).std(ddof=1) * np.sqrt(JOURS_BOURSE) * 100)
    longue = float(r.tail(252).std(ddof=1) * np.sqrt(JOURS_BOURSE) * 100)
    if longue <= 0 or courte < longue * multiple:
        return None
    return {"type": "retournement_momentum", "vol_courte": round(courte, 1),
            "vol_longue": round(longue, 1),
            "exposition": round(exposition_momentum * 100, 0)}


# ==========================================================================
# Commentaires contextualisés
# ==========================================================================

def _rang(n: int) -> str:
    """Ordinal français : 1er, 2e, 3e."""
    return "1er" if n == 1 else f"{n}e"


def _phrase_position(c: dict) -> str:
    """Situe la ligne dans le portefeuille."""
    poids = c.get("poids_pct")
    risque = c.get("part_risque_pct")
    if poids is None:
        return ""
    qualificatif = ("ta plus grosse ligne" if poids > 35 else
                    "une position importante" if poids > 20 else
                    "une petite ligne" if poids < 8 else "une ligne moyenne")
    base = f"{qualificatif} à {poids:.0f} % du portefeuille"
    if risque is not None and poids > 0 and risque / poids > 1.4:
        base += f", mais {risque:.0f} % du risque total"
    return base


def commenter(signal: dict, c: dict) -> str:
    """
    Commentaire adapte au signal ET a la societe.

    Chaque type de signal a sa propre lecture ; les chiffres injectes sont
    ceux de la valeur concernee, jamais des formules generiques.
    """
    nom = c.get("nom", c["ticker"])
    position = _phrase_position(c)
    secteur = f" ({c['secteur']})" if c.get("secteur") else ""
    type_signal = signal["type"]

    # ---------------------------------------------------------------- momentum
    if type_signal == "momentum":
        if signal["etat"] == "tête":
            texte = (
                f"{nom}{secteur} est {signal['rang']}e sur {signal['total']} "
                f"valeurs suivies au classement momentum, avec {signal['score_pct']:+.0f} % "
                f"sur douze mois hors dernier mois. C'est {position}. "
                "Le momentum à douze mois est l'anomalie la mieux documentée de "
                "la finance empirique, mais elle se retourne brutalement : "
                "sa pire séquence a effacé près de 40 % en trois mois."
            )
            if c.get("ecart_plus_haut_pct") is not None and c["ecart_plus_haut_pct"] > -5:
                texte += (f" Le titre est à {abs(c['ecart_plus_haut_pct']):.0f} % "
                          "de son plus haut annuel, ce qui renforce le signal.")
            if c.get("per"):
                texte += (f" Le PER de {c['per']:.0f} rappelle que le momentum "
                          "ignore complètement la valorisation.")
            return texte

        texte = (
            f"{nom}{secteur} est {_rang(signal['rang'])} sur {signal['total']} au "
            f"classement momentum, avec {signal['score_pct']:+.0f} % sur la période. "
            f"C'est {position}. Les titres en queue de classement tendent à "
            "sous-performer encore quelques mois — l'effet est symétrique."
        )
        if c.get("potentiel_consensus_pct"):
            texte += (f" Le consensus vise pourtant {c['potentiel_consensus_pct']:+.0f} % "
                      "de potentiel, ce qui illustre l'écart habituel entre "
                      "objectifs d'analystes et dynamique de marché.")
        return texte

    # -------------------------------------------------- dérive post-résultats
    if type_signal == "derive_resultats":
        texte = (
            f"{nom} a publié une surprise {signal['sens']} de "
            f"{signal['surprise_pct']:+.1f} % sur son bénéfice. C'est {position}. "
            "La dérive post-annonce est l'un des effets les mieux établis : "
            "le marché sous-réagit et le cours poursuit son mouvement six à "
            "neuf semaines."
        )
        if signal.get("confirme"):
            texte += (f" Les estimations ont suivi dans le même sens "
                      f"({signal['revision_pct']:+.1f} % sur 90 jours), "
                      "ce qui est la configuration où l'effet est le plus net.")
        elif signal.get("revision_pct") is not None:
            texte += (f" En revanche les estimations vont dans l'autre sens "
                      f"({signal['revision_pct']:+.1f} %), ce qui affaiblit "
                      "nettement le signal.")
        taux = c.get("taux_depassement_pct")
        if taux is not None:
            if taux >= 85:
                lecture = ("un taux aussi élevé traduit souvent une direction "
                           "qui guide prudemment pour être sûre de dépasser, "
                           "plutôt qu'une exécution exceptionnelle")
            elif taux >= 60:
                lecture = "ce qui est dans la norme des sociétés suivies"
            elif taux >= 35:
                lecture = ("ce qui est faible : les déceptions sont fréquentes "
                           "et le marché en tient déjà compte")
            else:
                lecture = ("ce qui est très faible et signale des difficultés "
                           "récurrentes à tenir ses propres prévisions")
            texte += (f" La société dépasse le consensus dans {taux:.0f} % "
                      f"des cas, {lecture}.")
        return texte

    # ------------------------------------------------------------- révisions
    if type_signal == "revisions":
        texte = (
            f"Les analystes ont révisé leurs estimations de bénéfice sur {nom} "
            f"de {signal['revision_pct']:+.1f} % en trois mois, à la {signal['sens']}. "
            f"C'est {position}. C'est la direction des révisions qui porte "
            "l'information, jamais le niveau du consensus, structurellement "
            "optimiste."
        )
        if c.get("analystes"):
            texte += (f" {c['analystes']} analystes couvrent le titre"
                      + (", ce qui rend le signal peu fiable."
                         if c["analystes"] < 5 else "."))
        return texte

    # ------------------------------------------------------------- plus haut
    if type_signal == "plus_haut":
        if signal["etat"] == "proche":
            return (
                f"{nom} évolue à {abs(signal['ecart_pct']):.0f} % de son plus haut "
                f"sur 52 semaines. C'est {position}. Contrairement à l'intuition, "
                "George et Hwang ont montré que les titres proches de leur plus "
                "haut surperforment : les investisseurs hésitent à acheter ce qui "
                "semble cher, et l'information met du temps à s'intégrer."
                + (f" Volatilité annualisée de {c['volatilite_pct']:.0f} %, "
                   "à intégrer dans le dimensionnement."
                   if c.get("volatilite_pct") else "")
            )
        texte = (
            f"{nom} est {abs(signal['ecart_pct']):.0f} % sous son plus haut annuel. "
            f"C'est {position}. Un écart de cette ampleur n'est pas une occasion "
            "en soi — statistiquement, les titres éloignés de leur plus haut "
            "continuent plutôt de sous-performer."
        )
        if c.get("revision_90j_pct") is not None:
            texte += (
                f" Les estimations de bénéfice ont bougé de "
                f"{c['revision_90j_pct']:+.1f} % sur trois mois"
                + (", ce qui suggère un problème de fond plutôt qu'un simple excès de pessimisme."
                   if c["revision_90j_pct"] < 0
                   else ", ce qui est plus encourageant : le prix baisse alors que les attentes montent.")
            )
        return texte

    # ---------------------------------------------------------------- régime
    if type_signal == "regime":
        if signal["etat"] == "défavorable":
            return (
                f"L'indice de référence est repassé sous sa moyenne 200 séances "
                f"({signal['ecart_pct']:+.1f} %). Ce n'est pas un signal de vente : "
                "rester à l'écart dans ces phases ne fait pas gagner davantage sur "
                "longue période, mais réduit sensiblement les pertes maximales. "
                "Utile à savoir quand le portefeuille est concentré."
            )
        return (
            f"L'indice de référence est repassé au-dessus de sa moyenne 200 "
            f"séances ({signal['ecart_pct']:+.1f} %). Les phases au-dessus de cette "
            "moyenne concentrent l'essentiel des hausses historiques et présentent "
            "une volatilité plus faible."
        )

    # -------------------------------------------------------- dimensionnement
    if type_signal == "dimensionnement":
        return (
            f"{nom} pèse {signal['poids_pct']:.0f} % de ton portefeuille mais porte "
            f"{signal['risque_pct']:.0f} % de son risque, soit {signal['ratio']} fois "
            "son poids. Sa volatilité"
            + (f" de {c['volatilite_pct']:.0f} %" if c.get("volatilite_pct") else "")
            + " et ses corrélations en font le déterminant principal du "
            "comportement de l'ensemble. Alléger cette ligne réduira le risque "
            "total bien plus que d'alléger n'importe quelle autre — c'est la "
            "définition de la contribution marginale."
        )

    # -------------------------------------------- retournement de momentum
    if type_signal == "retournement_momentum":
        return (
            f"La volatilité du portefeuille est passée à {signal['vol_courte']:.0f} % "
            f"sur un mois contre {signal['vol_longue']:.0f} % sur un an, alors que "
            f"{signal['exposition']:.0f} % de l'exposition est sur des valeurs à fort "
            "momentum. C'est exactement la configuration qui précède les "
            "retournements de cette stratégie : ils surviennent après un pic de "
            "volatilité, quand les titres les plus massacrés rebondissent "
            "violemment et que le momentum se retrouve du mauvais côté."
        )

    return f"{nom} — signal {type_signal}."


def evaluer_tout(cours: pd.DataFrame, contextes: dict[str, dict],
                 indice: pd.Series | None = None,
                 actifs: dict | None = None) -> list[dict]:
    """Passe tous les signaux actifs et renvoie les alertes commentées."""
    actifs = actifs or {}
    alertes = []
    classement = classement_momentum(cours)

    for ticker, contexte in contextes.items():
        candidats = []
        if actifs.get("momentum", True):
            candidats.append(signal_momentum(classement, ticker))
        if actifs.get("derive_resultats", True):
            candidats.append(signal_derive_resultats(contexte))
        if actifs.get("revisions", True):
            candidats.append(signal_revisions(contexte))
        if actifs.get("plus_haut", True):
            candidats.append(signal_plus_haut(contexte))
        if actifs.get("dimensionnement", True):
            candidats.append(signal_dimensionnement(contexte))

        for signal in candidats:
            if signal is None:
                continue
            alertes.append({
                "ticker": ticker,
                "type": signal["type"],
                "titre": f"{contexte.get('nom', ticker)} — {_titre(signal)}",
                "commentaire": commenter(signal, contexte),
                "donnees": signal,
            })

    if indice is not None and actifs.get("regime", True):
        signal = signal_regime(indice)
        if signal:
            alertes.append({
                "ticker": "—", "type": "regime",
                "titre": f"Régime de marché {signal['etat']}",
                "commentaire": commenter(signal, {"ticker": "indice"}),
                "donnees": signal,
            })

    return alertes


def _titre(signal: dict) -> str:
    titres = {
        "momentum": lambda s: ("Momentum en tête de classement"
                               if s["etat"] == "tête"
                               else "Momentum en queue de classement"),
        "derive_resultats": lambda s: f"Surprise de résultats {s['sens']}",
        "revisions": lambda s: f"Estimations révisées à la {s['sens']}",
        "plus_haut": lambda s: ("Proche du plus haut annuel"
                                if s["etat"] == "proche"
                                else "Éloigné du plus haut annuel"),
        "dimensionnement": lambda s: "Concentration du risque",
        "retournement_momentum": lambda s: "Risque de retournement du momentum",
        "regime": lambda s: f"Régime {s['etat']}",
    }
    fonction = titres.get(signal["type"])
    return fonction(signal) if fonction else signal["type"]
