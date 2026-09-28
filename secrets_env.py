"""
Acces aux secrets hors Streamlit.

`google_prive` lit ses identifiants dans `st.secrets`. C'est commode dans
l'application, inutilisable dans GitHub Actions, ou Streamlit n'existe pas et
ou les secrets arrivent par variables d'environnement.

Plutot que de dupliquer le code de lecture — c'est exactement ce genre de
duplication qui a fait diverger la veille et l'application — on fabrique un
objet qui presente la meme interface que `st` et puise dans l'environnement.
`google_prive` n'a pas besoin de savoir lequel des deux il manipule.

Le compte de service se depose dans un secret GitHub nomme
GCP_SERVICE_ACCOUNT, contenant le fichier JSON tel quel.
"""

from __future__ import annotations

import json
import os


class Secrets:
    """Presente l'interface de st.secrets : .get() et l'acces par cle."""

    def __init__(self) -> None:
        self._compte = None
        brut = os.environ.get("GCP_SERVICE_ACCOUNT", "").strip()
        if brut:
            try:
                self._compte = json.loads(brut)
            except json.JSONDecodeError as erreur:
                raise ValueError(
                    "Le secret GCP_SERVICE_ACCOUNT n'est pas un JSON valide. "
                    "Colle le contenu du fichier téléchargé depuis Google "
                    f"Cloud, sans le modifier. Détail : {erreur}") from erreur

    def get(self, cle: str, defaut=None):
        if cle == "gcp_service_account":
            return self._compte
        return os.environ.get(cle.upper(), defaut)

    def __getitem__(self, cle: str):
        valeur = self.get(cle)
        if valeur is None:
            raise KeyError(cle)
        return valeur

    def __contains__(self, cle: str) -> bool:
        return self.get(cle) is not None


class Contexte:
    """Objet minimal accepte par google_prive a la place de Streamlit."""

    def __init__(self) -> None:
        self.secrets = Secrets()


def contexte() -> Contexte:
    return Contexte()


def compte_configure() -> bool:
    return bool(os.environ.get("GCP_SERVICE_ACCOUNT", "").strip())
