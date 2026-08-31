"""
Publications de resultats : avant et apres.

Deux moments distincts.

AVANT (J-3) : ce que le marche attend. Chiffre d'affaires, benefice par action,
resultat net et marge implicite, tels que les analystes les prevoient. C'est
la reference contre laquelle les chiffres reels seront juges — un resultat en
hausse mais sous le consensus fait baisser le cours, et l'inverse est vrai.

APRES (jour J) : ce qui a ete publie, confronte a cette reference, avec une
lecture redigee.

Limite a connaitre : yfinance donne les chiffres, pas le communique lui-meme.
La lecture porte donc sur les ecarts au consensus et sur les titres de
depeches primaires du jour, jamais sur le texte integral du communique.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))


# ==========================================================================
# Avant : le consensus
# ==========================================================================

def consensus(ticker: str) -> dict:
    """
    Attentes des analystes pour le trimestre a venir.

    Renvoie le chiffre d'affaires et le benefice par action attendus, la
    croissance impliquee, le nombre d'analystes, et la marge nette implicite
    quand le nombre d'actions permet de la reconstituer.
    """
    sortie = {}
    try:
        valeur = yf.Ticker(ticker)
    except Exception:
        return sortie

    def _nombre(x):
        try:
            y = float(x)
            return y if np.isfinite(y) else None
        except (TypeError, ValueError):
            return None

    # --- Chiffre d'affaires attendu
    try:
        estimation = valeur.revenue_estimate
        if estimation is not None and not estimation.empty \
                and "0q" in estimation.index:
            ligne = estimation.loc["0q"]
            sortie["ca_attendu"] = _nombre(ligne.get("avg"))
            sortie["ca_bas"] = _nombre(ligne.get("low"))
            sortie["ca_haut"] = _nombre(ligne.get("high"))
            sortie["ca_analystes"] = _nombre(ligne.get("numberOfAnalysts"))
            croissance = _nombre(ligne.get("growth"))
            if croissance is not None:
                sortie["ca_croissance_pct"] = croissance * 100
            sortie["ca_an_dernier"] = _nombre(ligne.get("yearAgoRevenue"))
    except Exception:
        pass

    # --- Bénéfice par action attendu
    try:
        estimation = valeur.earnings_estimate
        if estimation is not None and not estimation.empty \
                and "0q" in estimation.index:
            ligne = estimation.loc["0q"]
            sortie["bpa_attendu"] = _nombre(ligne.get("avg"))
            sortie["bpa_bas"] = _nombre(ligne.get("low"))
            sortie["bpa_haut"] = _nombre(ligne.get("high"))
            sortie["bpa_analystes"] = _nombre(ligne.get("numberOfAnalysts"))
            croissance = _nombre(ligne.get("growth"))
            if croissance is not None:
                sortie["bpa_croissance_pct"] = croissance * 100
            sortie["bpa_an_dernier"] = _nombre(ligne.get("yearAgoEps"))
    except Exception:
        pass

    # --- Marge nette implicite, reconstituée depuis le nombre d'actions
    try:
        infos = valeur.fast_info
        actions = _nombre(infos.get("shares"))
        if actions is None:
            actions = _nombre(dict(valeur.info).get("sharesOutstanding"))
        if actions and sortie.get("bpa_attendu") and sortie.get("ca_attendu"):
            resultat_net = sortie["bpa_attendu"] * actions
            sortie["resultat_net_attendu"] = resultat_net
            sortie["marge_nette_attendue_pct"] = (
                resultat_net / sortie["ca_attendu"] * 100)
            sortie["actions"] = actions
    except Exception:
        pass

    # --- Marge du même trimestre un an plus tôt, pour comparaison
    try:
        comptes = valeur.quarterly_income_stmt
        if comptes is not None and not comptes.empty:
            import fondamentaux as fo
            ca = fo.poste(comptes, "chiffre_affaires")
            rn = fo.poste(comptes, "resultat_net")
            if len(ca) >= 4 and len(rn) >= 4 and float(ca.iloc[3]) != 0:
                sortie["marge_nette_an_dernier_pct"] = (
                    float(rn.iloc[3]) / float(ca.iloc[3]) * 100)
    except Exception:
        pass

    return sortie


# ==========================================================================
# Après : les chiffres publiés
# ==========================================================================

def chiffres_publies(ticker: str) -> dict:
    """
    Derniers comptes trimestriels publies.

    Yahoo met a jour ces postes dans les heures ou les jours suivant la
    publication. Un decalage est donc possible le jour meme.
    """
    sortie = {}
    try:
        valeur = yf.Ticker(ticker)
        comptes = valeur.quarterly_income_stmt
        if comptes is None or comptes.empty:
            return sortie

        import fondamentaux as fo
        ca = fo.poste(comptes, "chiffre_affaires")
        rn = fo.poste(comptes, "resultat_net")
        ebitda = fo.ebitda_serie(comptes)
        ebit = fo.poste(comptes, "ebit")

        if not ca.empty:
            sortie["date_trimestre"] = ca.index[0]
            sortie["ca_publie"] = float(ca.iloc[0])
            if len(ca) >= 5 and float(ca.iloc[4]) != 0:
                sortie["ca_croissance_pct"] = (
                    float(ca.iloc[0]) / float(ca.iloc[4]) - 1) * 100
            if not rn.empty:
                sortie["resultat_net_publie"] = float(rn.iloc[0])
                sortie["marge_nette_pct"] = (
                    float(rn.iloc[0]) / float(ca.iloc[0]) * 100)
                if len(rn) >= 5 and len(ca) >= 5 and float(ca.iloc[4]) != 0:
                    sortie["marge_nette_an_dernier_pct"] = (
                        float(rn.iloc[4]) / float(ca.iloc[4]) * 100)
            if not ebitda.empty:
                sortie["marge_ebitda_pct"] = (
                    float(ebitda.iloc[0]) / float(ca.iloc[0]) * 100)
            if not ebit.empty:
                sortie["marge_operationnelle_pct"] = (
                    float(ebit.iloc[0]) / float(ca.iloc[0]) * 100)
    except Exception:
        pass
    return sortie


def confronter(publie: dict, attendu: dict) -> dict:
    """Ecarts entre chiffres publies et consensus."""
    ecarts = {}
    if publie.get("ca_publie") and attendu.get("ca_attendu"):
        ecarts["ecart_ca_pct"] = (
            publie["ca_publie"] / attendu["ca_attendu"] - 1) * 100
    if publie.get("marge_nette_pct") and attendu.get("marge_nette_attendue_pct"):
        ecarts["ecart_marge_points"] = (
            publie["marge_nette_pct"] - attendu["marge_nette_attendue_pct"])
    if (publie.get("marge_nette_pct") is not None
            and publie.get("marge_nette_an_dernier_pct") is not None):
        ecarts["evolution_marge_points"] = (
            publie["marge_nette_pct"] - publie["marge_nette_an_dernier_pct"])
    return ecarts


# ==========================================================================
# Lecture rédigée
# ==========================================================================

CONSIGNES = """Tu commentes la publication de résultats d'une société cotée, \
pour un investisseur français qui suit les marchés de près.

RÈGLES ABSOLUES

Ne calcule rien. Tous les chiffres et écarts sont fournis.

N'invente aucun fait. Tu ne disposes PAS du communiqué de presse intégral : \
tu as les chiffres publiés, le consensus, et éventuellement des titres de \
dépêches du jour. Tu ne sais donc pas ce qu'a dit la direction, quelles \
perspectives elle a données, ni comment le cours a réagi. Ne l'invente jamais.

Si des titres de dépêches sont fournis, tu peux signaler ce qu'ils \
mentionnent, en indiquant qu'il s'agit d'un titre et non du communiqué.

Ne recommande ni achat ni vente.

CE QUE TU DOIS DIRE, dans cet ordre

1. Le verdict sur les attentes : les chiffres dépassent-ils, égalent-ils ou \
manquent-ils le consensus, et de combien. C'est l'écart qui fait le cours, \
pas le niveau absolu.

2. La marge : progresse-t-elle ou s'érode-t-elle par rapport au même trimestre \
un an plus tôt. Un chiffre d'affaires en hausse avec une marge en baisse est \
une croissance achetée.

3. Ce qui manque, le cas échéant. Si les perspectives, la marge ou un poste \
clé ne figurent pas dans les données, dis-le : c'est une information en soi \
de savoir ce qu'on ne sait pas.

FORME

Trois à cinq phrases. Denses, factuelles, sans formule d'introduction. \
Commence directement par le verdict."""


def rediger_lecture(ticker: str, nom: str, publie: dict, attendu: dict,
                    ecarts: dict, titres: list[str] | None = None,
                    cle: str = "", modele: str = "claude-sonnet-5") -> str:
    """
    Fait rediger la lecture des resultats.

    Sans cle ou en cas d'echec, renvoie une chaine vide : l'appelant se
    contentera alors des chiffres bruts, ce qui reste exploitable.
    """
    cle = (cle or os.environ.get("CLE_ANTHROPIC", "")).strip()
    if not cle:
        return ""

    import json
    donnees = {
        "société": nom or ticker,
        "chiffres_publiés": {k: v for k, v in publie.items() if v is not None},
        "consensus_avant_publication": {k: v for k, v in attendu.items()
                                        if v is not None},
        "écarts_calculés": ecarts,
        "titres_de_dépêches_du_jour": titres or [],
        "avertissement": ("Le communiqué intégral n'est pas disponible. "
                          "Seuls les chiffres et les titres ci-dessus le sont."),
    }

    try:
        import anthropic
        espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        client = anthropic.Anthropic(
            api_key=cle,
            default_headers={"anthropic-workspace-id": espace} if espace else None)
        reponse = client.messages.create(
            model=modele, max_tokens=600, system=CONSIGNES,
            messages=[{"role": "user", "content": json.dumps(
                donnees, ensure_ascii=False, indent=1, default=str)}],
        )
        return "".join(b.text for b in reponse.content
                       if getattr(b, "type", "") == "text").strip()
    except Exception as erreur:
        print(f"Lecture IA indisponible — {type(erreur).__name__} : "
              f"{str(erreur)[:200]}", file=sys.stderr)
        return ""


# ==========================================================================
# Mise en forme des montants
# ==========================================================================

def montant(valeur: float | None, devise: str = "") -> str:
    """Formate un montant en milliards ou millions, selon son ordre de grandeur."""
    if valeur is None or not np.isfinite(valeur):
        return "—"
    suffixe = f" {devise}" if devise else ""
    if abs(valeur) >= 1e9:
        return f"{valeur / 1e9:.2f} Md{suffixe}"
    if abs(valeur) >= 1e6:
        return f"{valeur / 1e6:.0f} M{suffixe}"
    return f"{valeur:,.0f}{suffixe}".replace(",", " ")
