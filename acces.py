"""
Controle d'acces a l'application.

L'hebergement gratuit de Streamlit ne propose pas d'application privee : toute
URL deployee est publique. La protection se fait donc dans l'application
elle-meme — rien n'est affiche tant que l'identification n'a pas reussi.

Ce que cela protege : l'affichage de tes donnees a un visiteur qui tomberait
sur l'adresse. Ce que cela ne protege pas : le code lui-meme, qui s'execute
cote serveur avant l'ecran de connexion. Suffisant pour un portefeuille
personnel, insuffisant pour des donnees reellement sensibles.

Les mots de passe ne sont jamais stockes en clair : seule leur empreinte
SHA-256 salee figure dans les secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import streamlit as st

DUREE_BLOCAGE = 60          # secondes de blocage apres trop d'echecs
TENTATIVES_MAX = 5


def empreinte(mot_de_passe: str, sel: str) -> str:
    """
    Empreinte SHA-256 salee.

    Le sel evite qu'une meme empreinte corresponde au meme mot de passe d'un
    deploiement a l'autre, et rend inutilisables les tables precalculees.
    """
    return hashlib.sha256((sel + mot_de_passe).encode("utf-8")).hexdigest()


def _comparer(a: str, b: str) -> bool:
    """
    Comparaison a temps constant.

    Une comparaison classique s'arrete au premier caractere different, ce qui
    laisse fuiter de l'information par le temps de reponse.
    """
    return hmac.compare_digest(a.encode(), b.encode())


def _identifiants() -> tuple[dict, str]:
    """Lit les comptes et le sel depuis les secrets Streamlit."""
    try:
        secrets = st.secrets
    except Exception:
        return {}, ""
    comptes = secrets.get("comptes", {})
    sel = secrets.get("sel", "")
    return dict(comptes), str(sel)


def _bloque() -> int:
    """Secondes de blocage restantes apres des echecs repetes."""
    jusqua = st.session_state.get("bloque_jusqua", 0)
    reste = int(jusqua - time.time())
    return max(0, reste)


def verifier(identifiant: str, mot_de_passe: str) -> bool:
    """Confronte le couple saisi aux empreintes enregistrees."""
    comptes, sel = _identifiants()
    attendue = comptes.get(identifiant.strip())
    if not attendue:
        return False
    return _comparer(empreinte(mot_de_passe, sel), str(attendue))


def porte(titre: str = "Analyse de portefeuille") -> bool:
    """
    Affiche l'ecran de connexion et bloque tant qu'il n'est pas franchi.

    Renvoie True une fois l'utilisateur identifie. A appeler tout en haut de
    l'application, avant tout autre affichage.
    """
    if st.session_state.get("authentifie"):
        return True

    comptes, sel = _identifiants()
    if not comptes or not sel:
        st.error(
            "Aucun compte configuré. Dans les paramètres de l'application sur "
            "share.streamlit.io, section Secrets, ajoute un sel et au moins un "
            "compte. Voir la documentation du fichier acces.py."
        )
        st.stop()

    st.title(titre)
    st.caption("Accès réservé.")

    restant = _bloque()
    if restant:
        st.error(f"Trop de tentatives. Réessaie dans {restant} secondes.")
        st.stop()

    with st.form("connexion"):
        identifiant = st.text_input("Identifiant")
        mot_de_passe = st.text_input("Mot de passe", type="password")
        valider = st.form_submit_button("Se connecter", type="primary")

    if valider:
        if verifier(identifiant, mot_de_passe):
            st.session_state.authentifie = True
            st.session_state.utilisateur = identifiant.strip()
            st.session_state.echecs = 0
            st.rerun()
        else:
            st.session_state.echecs = st.session_state.get("echecs", 0) + 1
            if st.session_state.echecs >= TENTATIVES_MAX:
                st.session_state.bloque_jusqua = time.time() + DUREE_BLOCAGE
                st.session_state.echecs = 0
                st.error(f"Trop de tentatives. Attends {DUREE_BLOCAGE} secondes.")
            else:
                restantes = TENTATIVES_MAX - st.session_state.echecs
                st.error(
                    f"Identifiant ou mot de passe incorrect. "
                    f"{restantes} tentative(s) avant blocage temporaire."
                )

    st.stop()


def bouton_deconnexion() -> None:
    """Bouton de deconnexion, a placer dans la barre laterale."""
    if not st.session_state.get("authentifie"):
        return
    st.sidebar.caption(f"Connecté : {st.session_state.get('utilisateur', '—')}")
    if st.sidebar.button("Se déconnecter", use_container_width=True):
        for cle in ("authentifie", "utilisateur", "echecs"):
            st.session_state.pop(cle, None)
        st.rerun()


# ==========================================================================
# Utilitaire de configuration
# ==========================================================================

def generer_secrets(identifiant: str, mot_de_passe: str,
                    sel: str | None = None) -> str:
    """
    Produit le bloc a coller dans les secrets Streamlit.

    A executer en local, jamais sur le serveur : le mot de passe en clair ne
    doit transiter nulle part.
    """
    import secrets as sec
    sel = sel or sec.token_hex(16)
    return (
        f'sel = "{sel}"\n\n'
        f"[comptes]\n"
        f'{identifiant} = "{empreinte(mot_de_passe, sel)}"\n'
    )


if __name__ == "__main__":
    import getpass

    print("Génération d'un bloc de secrets Streamlit.\n")
    nom = input("Identifiant : ").strip()
    mdp = getpass.getpass("Mot de passe : ")
    confirmation = getpass.getpass("Confirme le mot de passe : ")
    if mdp != confirmation:
        raise SystemExit("Les deux saisies diffèrent.")
    if len(mdp) < 8:
        raise SystemExit("Choisis un mot de passe d'au moins 8 caractères.")

    print("\n--- À coller dans Settings → Secrets sur share.streamlit.io ---\n")
    print(generer_secrets(nom, mdp))
    print("--- fin ---")
