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
Ne recommande ni achat ni vente.

CE QUE TU FAIS
Retiens les deux ou trois éléments qui comptent vraiment et laisse le reste. \
Un tableau de vingt lignes contient rarement plus de trois informations utiles.
Privilégie ce qui surprend : un chiffre qui contredit l'intuition, un écart \
inattendu entre deux mesures, une valeur qui sort du lot.
Explique ce que le chiffre implique concrètement, pas ce qu'il est.
Signale une limite quand elle change la lecture : historique court, mesure \
bruitée, donnée manquante.

FORME
Trois à cinq phrases, en prose. Pas de liste à puces, pas de titre, pas de \
formule d'introduction. Commence directement par l'élément le plus important.
Écris comme un analyste qui parle à un collègue, sans jargon inutile."""


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
             modele: str, cle: str) -> str:
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
                   f"Données :\n```json\n{contenu}\n```")
        reponse = client.messages.create(
            model=modele, max_tokens=500, system=CONSIGNES,
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
         ouvert: bool = False) -> None:
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

    automatique = st.session_state.get("analyses_auto", False)
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
            texte = _generer(empreinte, titre, contenu, contexte, modele, cle_api)
        st.session_state[f"analyse_{empreinte}"] = texte

        if texte.startswith("ERREUR"):
            st.error(texte)
        else:
            st.markdown(texte)
            st.caption("Générée à partir des chiffres ci-dessus, non vérifiée. "
                       "Aucun nombre n'est produit par le modèle.")


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
