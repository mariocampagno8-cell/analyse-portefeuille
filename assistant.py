"""
Assistant d'analyse fonde sur l'API Claude.

Principe d'architecture, non negociable : le modele NE CALCULE RIEN.

Un modele de langage est mauvais en arithmetique et produit des chiffres
plausibles mais faux avec une assurance totale. Ici, Python calcule tout —
betas, ratios, contributions au risque — et le modele ne recoit que des
resultats deja etablis. Son role est de les lire, de les relier entre eux et
de les expliquer, jamais de les produire.

Consequence pratique : si une donnee n'est pas dans le contexte transmis, le
modele doit dire qu'il ne l'a pas, et non l'estimer.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

MODELES = {
    "claude-sonnet-5": "Sonnet 5 — équilibré, recommandé",
    "claude-opus-5": "Opus 5 — plus fin, plus cher",
    "claude-haiku-4-5-20251001": "Haiku 4.5 — rapide et économique",
}

MODELE_DEFAUT = "claude-sonnet-5"

CONSIGNES = """Tu es un analyste quantitatif qui commente le portefeuille d'un \
investisseur particulier français. Tu t'adresses à lui directement, en français.

RÈGLES ABSOLUES

1. Ne calcule jamais. Tous les chiffres dont tu as besoin sont dans le contexte \
fourni. Si une information manque, dis-le franchement au lieu de l'estimer ou \
de la déduire. Un chiffre inventé est pire que pas de chiffre.

2. Ne donne aucun conseil en investissement. Tu n'es pas conseiller financier. \
Tu expliques ce que les chiffres signifient et ce qu'ils impliquent ; \
l'utilisateur décide. Ne dis jamais d'acheter, de vendre ou de conserver.

3. Nomme les limites quand elles comptent. Les données viennent de yfinance, un \
scraper non officiel. Les bêtas estimés sur moins de trois ans sont bruités. \
Le backtest sur un portefeuille existant souffre d'un biais de sélection. Ne \
répète pas ces réserves à chaque phrase, mais mentionne-les quand elles \
changent la lecture.

MANIÈRE

Va droit au chiffre qui compte plutôt que de tout réciter. Un portefeuille a \
généralement deux ou trois caractéristiques dominantes ; le reste est du bruit.

Privilégie ce qui surprend. Si la ligne qui porte le risque n'est pas la plus \
grosse position, c'est ça qu'il faut dire. Si le bêta baissier dépasse \
nettement le bêta global, c'est ça qu'il faut dire.

Écris en prose claire, sans jargon inutile. Quand tu emploies un terme \
technique, explique-le en une incise. Pas de listes à puces sauf si la \
question en appelle vraiment une. Reste concis : quelques paragraphes suffisent."""


# ==========================================================================
# Construction du contexte
# ==========================================================================

def _propre(valeur) -> float | None:
    """Convertit en flottant affichable, ou None si la valeur est inutilisable."""
    if valeur is None:
        return None
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if np.isfinite(v) else None


def _table(df: pd.DataFrame, lignes_max: int = 20) -> list[dict]:
    """Serialise un tableau en liste de dictionnaires, valeurs nettoyees."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    extrait = df.head(lignes_max).copy()
    sortie = []
    for index, ligne in extrait.iterrows():
        entree = {"_": str(index)}
        for colonne, valeur in ligne.items():
            nombre = _propre(valeur)
            entree[str(colonne)] = nombre if nombre is not None else str(valeur)
        sortie.append(entree)
    return sortie


def construire_contexte(
    valorisation: pd.DataFrame,
    metriques: dict,
    decomposition: pd.DataFrame,
    correlations: pd.DataFrame,
    parametres: dict,
    diversification: dict | None = None,
) -> dict:
    """
    Rassemble les resultats deja calcules en une structure transmissible.

    Tout ce qui arrive ici a ete calcule par Python. Le modele ne verra rien
    d'autre — c'est ce qui garantit qu'il ne pourra pas inventer.
    """
    return {
        "paramètres_analyse": parametres,
        "positions": _table(valorisation),
        "mesures_de_risque": {k: _propre(v) for k, v in metriques.items()},
        "décomposition_du_risque": _table(decomposition),
        "corrélations": _table(correlations, 12),
        "diversification": diversification or {},
        "avertissement": (
            "Données yfinance, scraper non officiel de Yahoo Finance. "
            "Estimations statistiques sur historique limité."
        ),
    }


def resumer(contexte: dict) -> str:
    """Contexte en JSON compact, pret a etre transmis au modele."""
    return json.dumps(contexte, ensure_ascii=False, indent=1, default=str)


# ==========================================================================
# Appels a l'API
# ==========================================================================

def _client(cle: str):
    """Instancie le client Anthropic, avec un message clair si le paquet manque."""
    try:
        import anthropic
    except ImportError as erreur:
        raise RuntimeError(
            "Le paquet `anthropic` n'est pas installé. Ajoute `anthropic` "
            "à requirements.txt et redéploie."
        ) from erreur
    cle = str(cle).strip()
    if not cle:
        raise RuntimeError(
            "Clé API absente. Ajoute `cle_anthropic = \"sk-ant-...\"` dans les "
            "secrets Streamlit. Ne la mets jamais dans le code."
        )
    espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    return anthropic.Anthropic(
        api_key=cle,
        default_headers={"anthropic-workspace-id": espace} if espace else None)


def interroger(question: str, contexte: dict, cle: str,
               historique: list | None = None,
               modele: str = MODELE_DEFAUT,
               max_tokens: int = 1200) -> str:
    """
    Pose une question sur le portefeuille et renvoie la reponse.

    L'historique permet les questions de suivi : le modele n'a aucune memoire
    entre deux appels, il faut lui renvoyer l'echange complet a chaque fois.
    """
    client = _client(cle)

    messages = list(historique or [])
    messages.append({
        "role": "user",
        "content": (
            f"Voici l'état actuel du portefeuille, calculé par l'application :\n\n"
            f"```json\n{resumer(contexte)}\n```\n\n"
            f"Question : {question}"
        ),
    })

    reponse = client.messages.create(
        model=modele,
        max_tokens=max_tokens,
        system=CONSIGNES,
        messages=messages,
    )
    return "".join(bloc.text for bloc in reponse.content
                   if getattr(bloc, "type", "") == "text")


def commentaire(contexte: dict, cle: str, modele: str = MODELE_DEFAUT,
                max_tokens: int = 900) -> str:
    """Analyse libre du portefeuille, sans question prealable."""
    return interroger(
        "Commente ce portefeuille. Concentre-toi sur les deux ou trois "
        "caractéristiques qui le définissent vraiment et sur ce qui pourrait "
        "surprendre son détenteur. Ne récite pas les chiffres un par un.",
        contexte, cle, modele=modele, max_tokens=max_tokens,
    )


def expliquer_mesure(nom: str, valeur, contexte: dict, cle: str,
                     modele: str = MODELE_DEFAUT) -> str:
    """Explique une mesure precise dans le contexte du portefeuille."""
    return interroger(
        f"Explique ce que signifie « {nom} » avec la valeur {valeur} pour ce "
        f"portefeuille précisément. Que faut-il en retenir, et quelles sont "
        f"les limites de cette mesure ?",
        contexte, cle, modele=modele, max_tokens=700,
    )


def cout_estime(reponse_brute) -> dict:
    """
    Cout approximatif d'un appel, a partir des jetons consommes.

    Tarifs publics au moment de l'écriture, en dollars par million de jetons.
    Ils évoluent : vérifier sur claude.com/pricing.
    """
    tarifs = {
        "claude-sonnet-5": (2.0, 10.0),
        "claude-opus-5": (5.0, 25.0),
        "claude-haiku-4-5-20251001": (1.0, 5.0),
    }
    usage = getattr(reponse_brute, "usage", None)
    modele = getattr(reponse_brute, "model", "")
    if not usage:
        return {}
    entree, sortie = tarifs.get(modele, (2.0, 10.0))
    cout = (usage.input_tokens * entree + usage.output_tokens * sortie) / 1e6
    return {
        "jetons_entrée": usage.input_tokens,
        "jetons_sortie": usage.output_tokens,
        "coût_dollars": round(cout, 5),
    }


SUGGESTIONS = [
    "Quelles sont les deux ou trois caractéristiques dominantes de mon portefeuille ?",
    "Où se concentre réellement mon risque, et est-ce là où je l'attends ?",
    "Ma diversification est-elle réelle ou seulement apparente ?",
    "Que se passerait-il pour moi si le marché baissait de 20 % ?",
    "Quelles mesures devrais-je surveiller en priorité, et pourquoi ?",
    "Qu'est-ce qui, dans ces chiffres, mériterait ma méfiance ?",
]
