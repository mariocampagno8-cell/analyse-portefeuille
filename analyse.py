"""
Bloc d'analyse repliable, generique.

S'insere sous n'importe quel tableau ou groupe d'indicateurs. Le principe est
toujours le meme : Python calcule, le modele lit et commente.

Deux precautions techniques qui comptent.

CACHE PAR SIGNATURE. Streamlit relance l'integralite du script a chaque clic.
Sans cache, ouvrir un onglet declencherait un appel par tableau, a chaque
interaction. Les analyses sont donc indexees sur une empreinte des donnees :
memes chiffres, meme analyse, aucun appel supplementaire.

GENERATION A LA DEMANDE PAR DEFAUT. Une analyse ne part que si l'on deplie le
bloc et qu'on la demande. Le mode automatique existe mais se choisit
explicitement dans la barre laterale, en connaissance du cout.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

MODELE_DEFAUT = "claude-sonnet-5"

CONSIGNES = """Tu commentes un tableau ou un groupe d'indicateurs financiers \
pour un investisseur particulier français qui connaît les bases.

RÈGLES ABSOLUES
Ne calcule rien : tous les chiffres sont fournis. Si une donnée manque, dis-le \
plutôt que de l'estimer.
N'affirme rien qui ne soit dans les données. Tu ignores l'actualité, ce qu'a \
dit une direction, pourquoi un cours a bougé. Ne l'invente jamais.

STRUCTURE DE TA RÉPONSE — deux parties, toujours

1. CE QUE DISENT LES CHIFFRES
Retiens les deux ou trois éléments qui comptent vraiment. Un tableau de vingt \
lignes contient rarement plus de trois informations utiles. Privilégie ce qui \
surprend : un chiffre qui contredit l'intuition, un écart inattendu entre deux \
mesures, une valeur qui sort du lot. Explique ce que le chiffre implique \
concrètement, pas ce qu'il est.

2. CE QUE TU PEUX FAIRE
Termine par une ou deux actions concrètes, introduites par « Concrètement : ». \
Elles doivent porter sur l'un de ces registres :

  — un ajustement de structure : alléger une ligne trop lourde, réduire une \
redondance, corriger un déséquilibre de risque, fixer un seuil manquant
  — une vérification : recouper un chiffre douteux, allonger la période \
d'analyse quand elle est trop courte pour être fiable, contrôler une donnée \
qui semble aberrante
  — une question à trancher : ce que l'investisseur doit décider lui-même \
avant d'agir

Chaque action doit indiquer son effet attendu ET son coût ou sa limite. \
« Alléger cette ligne réduirait la volatilité d'environ deux points, mais un \
arbitrage déclenche l'imposition des plus-values » vaut mieux que « allégez \
cette ligne ».

CE QUE TU NE FAIS JAMAIS
Ne recommande pas d'acheter ou de vendre une valeur précise comme un conseil \
d'investissement. Tu ignores la situation patrimoniale, l'horizon, les revenus \
et la tolérance au risque de ton lecteur : sans cela, une recommandation \
d'achat est une devinette habillée en analyse.
Ne prédis pas l'évolution d'un cours.
Ne présente pas une action comme évidente ou urgente quand les données ne le \
justifient pas.

FORME
Quatre à six phrases au total, en prose. Pas de liste à puces, pas de titre, \
pas de formule d'introduction. Commence directement par l'élément le plus \
important. Écris comme un analyste qui parle à un collègue."""


CONSIGNES_PORTEFEUILLE = """Tu commentes la construction d'un portefeuille \
pour un investisseur particulier français qui connaît les bases.

RÈGLES ABSOLUES
Ne calcule rien : tous les chiffres sont fournis.
N'affirme rien qui ne soit dans les données. Tu ignores l'actualité et les \
raisons des mouvements passés.

CE QUE TU DIS, dans cet ordre

1. LE COMPROMIS. Quelle allocation offre quoi, en une phrase par option qui \
mérite d'être mentionnée. Compare rendement ET risque : citer une performance \
sans sa volatilité et sa perte maximale est trompeur.

2. LA MISE EN GARDE SUR L'OPTIMISATION. Ces allocations sont calculées sur la \
période affichée, donc en connaissant son résultat. Rappelle que les méthodes \
maximisant le rendement reposent sur des rendements attendus, impossibles à \
estimer de façon fiable, alors que celles minimisant le risque ne dépendent \
que de la covariance, bien plus stable. Signale-le systématiquement quand une \
allocation à fort rendement apparaît en tête.

3. CE QUI SAUTE AUX YEUX DANS LA STRUCTURE. Concentration excessive sur une \
ligne, nombre de lignes effectives très inférieur au nombre de positions, \
corrélations élevées, perte maximale difficile à supporter.

4. CONCRÈTEMENT. Une ou deux actions : quelle allocation examiner de plus \
près et pourquoi, quelle vérification faire, quelle question trancher. Indique \
toujours l'effet attendu ET la limite.

CE QUE TU NE FAIS JAMAIS
Ne désigne pas une allocation comme « la meilleure » : elle l'est sur le passé, \
ce qui ne se transpose pas.
Ne recommande pas d'acheter ou de vendre une valeur précise.
Ne prédis aucune performance future.

FORME
Cinq à sept phrases, en prose, sans liste ni titre. Commence par le compromis \
principal."""


# ==========================================================================
# Sérialisation et empreinte
# ==========================================================================

def _serialiser(donnees) -> str:
    """Convertit tableaux et dictionnaires en JSON compact et lisible."""
    def _propre(valeur):
        if isinstance(valeur, (np.integer, np.floating)):
            valeur = float(valeur)
        if isinstance(valeur, float):
            return round(valeur, 4) if np.isfinite(valeur) else None
        return valeur

    if isinstance(donnees, pd.DataFrame):
        extrait = donnees.head(25)
        lignes = []
        for index, ligne in extrait.iterrows():
            entree = {"_": str(index)}
            entree.update({str(k): _propre(v) for k, v in ligne.items()})
            lignes.append(entree)
        return json.dumps(lignes, ensure_ascii=False, default=str)

    if isinstance(donnees, pd.Series):
        return json.dumps({str(k): _propre(v) for k, v in donnees.items()},
                          ensure_ascii=False, default=str)

    if isinstance(donnees, dict):
        return json.dumps({str(k): _propre(v) for k, v in donnees.items()},
                          ensure_ascii=False, default=str)

    return json.dumps(donnees, ensure_ascii=False, default=str)


def _empreinte(titre: str, contenu: str, contexte: str) -> str:
    """Signature des donnees : memes chiffres, meme analyse."""
    return hashlib.sha256(
        (titre + contenu + contexte).encode("utf-8")).hexdigest()[:24]


# ==========================================================================
# Appel au modèle
# ==========================================================================

@st.cache_data(ttl=86400, show_spinner=False, max_entries=200)
def _generer(empreinte: str, titre: str, contenu: str, contexte: str,
             modele: str, cle: str, consignes: str = "") -> str:
    """
    Produit l'analyse. Cache 24 h, indexe sur l'empreinte des donnees.

    L'empreinte est le premier argument pour que Streamlit l'utilise comme
    cle de cache : deux tableaux identiques ne declenchent qu'un seul appel.
    """
    try:
        import anthropic
        espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        client = anthropic.Anthropic(
            api_key=cle.strip(),
            default_headers={"anthropic-workspace-id": espace} if espace else None)

        message = (f"Tableau : {titre}\n\n"
                   f"Contexte : {contexte}\n\n"
                   "Cadre : investisseur particulier français, compte-titres "
                   "ordinaire sauf mention contraire — un arbitrage déclenche "
                   "l'imposition des plus-values au prélèvement forfaitaire "
                   "unique de 30 %.\n\n"
                   f"Données :\n```json\n{contenu}\n```")
        reponse = client.messages.create(
            model=modele, max_tokens=700, system=consignes or CONSIGNES,
            messages=[{"role": "user", "content": message}])
        return "".join(b.text for b in reponse.content
                       if getattr(b, "type", "") == "text").strip()
    except Exception as erreur:
        detail = str(erreur)[:200]
        if "Illegal header value" in detail:
            return ("ERREUR : le secret `cle_anthropic` contient un espace ou "
                    "un retour à la ligne. Recolle-le sans saut de ligne.")
        if "workspace-id" in detail:
            return ("ERREUR : clé liée à une identité. Crée une clé dont la "
                    "portée est un espace de travail.")
        return f"ERREUR : {type(erreur).__name__} — {detail}"


# ==========================================================================
# Bloc à insérer sous un tableau
# ==========================================================================

def bloc(titre: str, donnees, contexte: str = "", cle_widget: str = "",
         ouvert: bool = False, automatique: bool = False,
         consignes: str = "") -> None:
    """
    Affiche un bloc d'analyse repliable sous un tableau.

    En mode manuel, l'analyse ne part qu'a la demande. En mode automatique,
    elle se genere a l'ouverture du bloc et reste en cache.
    """
    cle_api = ""
    try:
        cle_api = str(st.secrets.get("cle_anthropic", "")).strip()
    except Exception:
        pass

    if not cle_api:
        return                                   # silencieux : pas de clé, pas de bloc

    automatique = automatique or st.session_state.get("analyses_auto", False)
    modele = st.session_state.get("modele_analyses", MODELE_DEFAUT)

    contenu = _serialiser(donnees)
    if len(contenu) < 10:
        return
    empreinte = _empreinte(titre, contenu, contexte)
    identifiant = cle_widget or empreinte

    with st.expander("🤖 Analyse", expanded=ouvert):
        deja = st.session_state.get(f"analyse_{empreinte}")

        if deja is None and not automatique:
            if not st.button("Générer l'analyse", key=f"btn_{identifiant}",
                             use_container_width=True):
                st.caption(
                    "Environ un centime. L'analyse est ensuite conservée tant "
                    "que les chiffres ne changent pas.")
                return

        with st.spinner("Analyse…"):
            texte = _generer(empreinte, titre, contenu, contexte, modele,
                             cle_api, consignes)
        st.session_state[f"analyse_{empreinte}"] = texte

        if texte.startswith("ERREUR"):
            st.error(texte)
        else:
            st.markdown(texte)
            st.caption(
                "Générée à partir des chiffres ci-dessus, non vérifiée. Aucun "
                "nombre n'est produit par le modèle. Les actions proposées "
                "portent sur la structure du portefeuille et ne constituent "
                "pas un conseil en investissement.")


def reglages_barre_laterale() -> None:
    """Reglages des analyses, a placer dans la barre laterale."""
    try:
        cle = str(st.secrets.get("cle_anthropic", "")).strip()
    except Exception:
        cle = ""
    if not cle:
        return

    st.sidebar.divider()
    st.sidebar.subheader("Analyses")
    st.session_state["analyses_auto"] = st.sidebar.toggle(
        "Générer automatiquement", value=False,
        help="Sans cela, chaque analyse demande un clic. En automatique, elle "
             "se génère à l'ouverture du bloc — environ un centime par tableau, "
             "puis mise en cache tant que les chiffres ne changent pas.")
    st.session_state["modele_analyses"] = st.sidebar.selectbox(
        "Modèle",
        ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5"],
        format_func=lambda m: {"claude-sonnet-5": "Sonnet 5 — équilibré",
                               "claude-haiku-4-5-20251001": "Haiku 4.5 — économique",
                               "claude-opus-5": "Opus 5 — plus fin"}[m])
