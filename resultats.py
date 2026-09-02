"""
Cycle de resultats — phases 2 et 3.

Quatre moments, dans l'ordre du cahier des charges :

  J-15  ouverture du dossier : consensus, historique, ce qui est deja dans le
        prix, et les questions auxquelles la publication doit repondre
  J-5   mise a jour : revisions d'estimations, le message le plus informatif
        du cycle — les analystes finalisent dans les deux dernieres semaines
  J-1   rappel : heure, consensus final, mouvement implicite
  J     message A, les chiffres bruts ; message B, la lecture et le verdict

Separation stricte entre A et B. Aucun chiffre du message A ne transite par un
modele de langage : une seule hallucination sur un chiffre d'affaires detruit
la confiance dans l'ensemble du systeme.

Ce que les donnees gratuites ne donnent pas, et qui est signale comme absent
plutot qu'estime : consensus de resultat operationnel, de marge et de free
cash-flow ; guidance en vigueur ; detail par segment ; ecart-type du consensus.
"""

from __future__ import annotations

import json
import os
import sys


import numpy as np
import pandas as pd
import yfinance as yf

MANQUANT = "n.c."


# ==========================================================================
# Extraction des comptes
# ==========================================================================

ALIAS = {
    "ca": ["Total Revenue", "Operating Revenue", "Revenue"],
    "marge_brute": ["Gross Profit"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "amortissements": ["Reconciled Depreciation", "Depreciation And Amortization"],
    "resultat_net": ["Net Income", "Net Income Common Stockholders"],
    "bpa": ["Diluted EPS", "Basic EPS"],
    "cfo": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "fcf": ["Free Cash Flow"],
    "dette": ["Total Debt"],
    "tresorerie": ["Cash And Cash Equivalents",
                   "Cash Cash Equivalents And Short Term Investments"],
}


def poste(etat: pd.DataFrame, cle: str) -> pd.Series:
    """Recupere un poste comptable par ses intitules possibles."""
    if etat is None or getattr(etat, "empty", True):
        return pd.Series(dtype=float)
    for nom in ALIAS.get(cle, [cle]):
        if nom in etat.index:
            serie = pd.to_numeric(etat.loc[nom], errors="coerce").dropna()
            if not serie.empty:
                return serie
    return pd.Series(dtype=float)


def _val(serie: pd.Series, rang: int = 0):
    return float(serie.iloc[rang]) if len(serie) > rang else None


def _pct(numerateur, denominateur):
    if numerateur is None or not denominateur:
        return None
    return numerateur / denominateur * 100


# ==========================================================================
# Consensus
# ==========================================================================

def consensus(ticker: str) -> dict:
    """
    Attentes des analystes pour le trimestre en cours.

    Yahoo ne diffuse que le chiffre d'affaires et le benefice par action. Le
    resultat operationnel, la marge et le free cash-flow n'ont pas de
    consensus gratuit : ils seront signales absents plutot qu'estimes.
    """
    sortie = {"absents": []}

    def _n(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    try:
        valeur = yf.Ticker(ticker)
    except Exception:
        return sortie

    try:
        estimation = valeur.revenue_estimate
        if estimation is not None and not estimation.empty and "0q" in estimation.index:
            ligne = estimation.loc["0q"]
            sortie.update({
                "ca_attendu": _n(ligne.get("avg")),
                "ca_bas": _n(ligne.get("low")),
                "ca_haut": _n(ligne.get("high")),
                "ca_analystes": _n(ligne.get("numberOfAnalysts")),
                "ca_an_dernier": _n(ligne.get("yearAgoRevenue"))})
            croissance = _n(ligne.get("growth"))
            if croissance is not None:
                sortie["ca_croissance_pct"] = croissance * 100
    except Exception:
        pass

    try:
        estimation = valeur.earnings_estimate
        if estimation is not None and not estimation.empty and "0q" in estimation.index:
            ligne = estimation.loc["0q"]
            sortie.update({
                "bpa_attendu": _n(ligne.get("avg")),
                "bpa_bas": _n(ligne.get("low")),
                "bpa_haut": _n(ligne.get("high")),
                "bpa_analystes": _n(ligne.get("numberOfAnalysts")),
                "bpa_an_dernier": _n(ligne.get("yearAgoEps"))})
            croissance = _n(ligne.get("growth"))
            if croissance is not None:
                sortie["bpa_croissance_pct"] = croissance * 100
    except Exception:
        pass

    # Dispersion : un consensus resserré sur 25 analystes ne se lit pas comme
    # une fourchette large sur 5. À défaut d'écart-type, l'étendue relative.
    if sortie.get("ca_bas") and sortie.get("ca_haut") and sortie.get("ca_attendu"):
        sortie["ca_dispersion_pct"] = (
            (sortie["ca_haut"] - sortie["ca_bas"]) / sortie["ca_attendu"] * 100)
    if sortie.get("bpa_bas") and sortie.get("bpa_haut") and sortie.get("bpa_attendu"):
        sortie["bpa_dispersion_pct"] = (
            (sortie["bpa_haut"] - sortie["bpa_bas"])
            / abs(sortie["bpa_attendu"]) * 100)

    sortie["absents"] = ["consensus de résultat opérationnel",
                         "consensus de marge", "consensus de free cash-flow",
                         "écart-type du consensus", "guidance en vigueur"]
    return sortie


def revisions(ticker: str) -> dict:
    """
    Dynamique de revision du benefice par action.

    Le champ le plus predictif du cycle : la direction des revisions est
    documentee comme predictive, le niveau du consensus ne l'est pas.
    """
    sortie = {}
    try:
        tendance = yf.Ticker(ticker).eps_trend
        if tendance is None or tendance.empty:
            return sortie
        for periode, libelle in [("0q", "trimestre"), ("0y", "exercice")]:
            if periode not in tendance.index:
                continue
            ligne = tendance.loc[periode]
            actuelle = ligne.get("current")
            if actuelle is None or actuelle != actuelle:
                continue
            sortie[f"{libelle}_estimation"] = float(actuelle)
            for jours, cle in [(7, "7j"), (30, "30j"), (90, "90j")]:
                ancienne = ligne.get(f"{jours}daysAgo")
                if ancienne and ancienne == ancienne and ancienne != 0:
                    sortie[f"{libelle}_revision_{cle}_pct"] = (
                        (float(actuelle) - float(ancienne))
                        / abs(float(ancienne)) * 100)
    except Exception:
        pass
    return sortie


# ==========================================================================
# Historique des surprises
# ==========================================================================

def historique(ticker: str, cours: pd.Series | None = None,
               nombre: int = 8) -> dict:
    """
    Les huit derniers trimestres : surprise et reaction du cours a J+1.

    Une societe qui bat de 2 % a chaque trimestre a un consensus guide, pas
    une execution exceptionnelle. Battre le consensus et perdre 4 % signifie
    que le marche attend plus que le consensus affiche.
    """
    try:
        dates = yf.Ticker(ticker).earnings_dates
        if dates is None or dates.empty:
            return {}
        index = pd.to_datetime(dates.index)
        table = dates.copy()
        table.index = index.tz_localize(None) if index.tz is not None else index
        if "Reported EPS" in table.columns:
            table = table[table["Reported EPS"].notna()]
        table = table.sort_index(ascending=False).head(nombre)
        if table.empty:
            return {}

        if cours is not None and getattr(cours.index, "tz", None) is not None:
            cours = cours.copy()
            cours.index = cours.index.tz_localize(None)

        lignes = []
        for date, ligne in table.iterrows():
            surprise = ligne.get("Surprise(%)")
            attendu, publie = ligne.get("EPS Estimate"), ligne.get("Reported EPS")
            if (surprise is None or surprise != surprise) and attendu:
                surprise = (publie - attendu) / abs(attendu) * 100
            reaction = np.nan
            if cours is not None:
                try:
                    apres = cours[cours.index > date]
                    avant = cours[cours.index <= date]
                    if len(apres) and len(avant):
                        reaction = float(apres.iloc[0] / avant.iloc[-1] - 1) * 100
                except Exception:
                    pass
            lignes.append({"date": date, "surprise": surprise,
                           "reaction": reaction})

        surprises = [l["surprise"] for l in lignes
                     if l["surprise"] == l["surprise"]]
        reactions = [l["reaction"] for l in lignes
                     if l["reaction"] == l["reaction"]]
        if not surprises:
            return {}

        resume = {
            "lignes": lignes,
            "trimestres": len(surprises),
            "depassement_pct": sum(1 for s in surprises if s > 0) / len(surprises) * 100,
            "surprise_mediane": float(np.median(surprises)),
        }
        if reactions:
            resume["reaction_mediane"] = float(np.median(reactions))
            resume["amplitude_mediane"] = float(np.median([abs(r) for r in reactions]))

        depassement = resume["depassement_pct"]
        mediane = resume["surprise_mediane"]
        reaction = resume.get("reaction_mediane")
        if depassement >= 75 and reaction is not None and reaction < -1:
            resume["lecture"] = ("dépasse le consensus mais le cours baisse : "
                                 "le marché attend plus que le consensus affiché")
        elif depassement >= 85 and 0 < mediane < 6:
            resume["lecture"] = "consensus guidé, dépassements constants et faibles"
        elif depassement <= 40:
            resume["lecture"] = "manque régulièrement les attentes"
        elif resume.get("amplitude_mediane", 0) >= 6:
            resume["lecture"] = (f"réaction violente à chaque publication "
                                 f"(±{resume['amplitude_mediane']:.0f} %)")
        else:
            resume["lecture"] = "historique sans biais marqué"
        return resume
    except Exception:
        return {}


def mouvement_implicite(ticker: str, date_publication=None) -> dict:
    """
    Amplitude anticipee par le marche, deduite du straddle a la monnaie.

    Dit combien il faut surprendre pour surprendre. Indisponible sur la
    plupart des valeurs europeennes, dont Yahoo ne diffuse pas les options.
    """
    try:
        valeur = yf.Ticker(ticker)
        echeances = valeur.options
        if not echeances:
            return {}
        cible = (pd.Timestamp(date_publication).normalize()
                 if date_publication is not None else pd.Timestamp.now().normalize())
        retenue = next((e for e in echeances if pd.Timestamp(e) >= cible), None)
        if retenue is None:
            return {}

        try:
            spot = float(valeur.fast_info["lastPrice"])
        except Exception:
            spot = float(valeur.history(period="5d")["Close"].iloc[-1])

        chaine = valeur.option_chain(retenue)
        calls, puts = chaine.calls, chaine.puts
        if calls.empty or puts.empty or spot <= 0:
            return {}

        strike = float(calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]["strike"])
        call = calls[calls["strike"] == strike]
        put = puts[puts["strike"] == strike]
        if call.empty or put.empty:
            return {}

        def _prix(ligne):
            achat = float(ligne["bid"].iloc[0] or 0)
            vente = float(ligne["ask"].iloc[0] or 0)
            return ((achat + vente) / 2 if achat and vente
                    else float(ligne["lastPrice"].iloc[0] or 0))

        straddle = _prix(call) + _prix(put)
        if straddle <= 0:
            return {}
        return {"echeance": retenue, "cours": spot,
                "mouvement_pct": straddle / spot * 100}
    except Exception:
        return {}


# ==========================================================================
# Chiffres publiés
# ==========================================================================

def chiffres_publies(ticker: str) -> dict:
    """
    Derniers comptes trimestriels, avec la base de comparaison N-1.

    Yahoo met a jour ces postes dans les heures ou les jours suivant la
    publication : un decalage est possible le jour meme.
    """
    sortie = {"absents": []}
    try:
        valeur = yf.Ticker(ticker)
        comptes = valeur.quarterly_income_stmt
        if comptes is None or comptes.empty:
            sortie["absents"].append("comptes trimestriels")
            return sortie

        ca = poste(comptes, "ca")
        rn = poste(comptes, "resultat_net")
        ebit = poste(comptes, "ebit")
        ebitda = poste(comptes, "ebitda")
        if ebitda.empty and not ebit.empty:
            amortissements = poste(comptes, "amortissements")
            if not amortissements.empty:
                ebitda = ebit + amortissements.reindex(ebit.index)

        if ca.empty:
            sortie["absents"].append("chiffre d'affaires")
            return sortie

        sortie["periode"] = ca.index[0]
        sortie["ca"] = _val(ca)
        sortie["ca_n1"] = _val(ca, 4)
        if sortie["ca_n1"]:
            sortie["ca_croissance_pct"] = (sortie["ca"] / sortie["ca_n1"] - 1) * 100

        sortie["resultat_net"] = _val(rn)
        sortie["resultat_net_n1"] = _val(rn, 4)
        sortie["marge_nette_pct"] = _pct(sortie["resultat_net"], sortie["ca"])
        sortie["marge_nette_n1_pct"] = _pct(_val(rn, 4), _val(ca, 4))

        sortie["ebit"] = _val(ebit)
        sortie["marge_op_pct"] = _pct(_val(ebit), sortie["ca"])
        sortie["marge_op_n1_pct"] = _pct(_val(ebit, 4), _val(ca, 4))
        sortie["marge_ebitda_pct"] = _pct(_val(ebitda), sortie["ca"])

        # Flux de trésorerie et bilan
        try:
            flux = valeur.quarterly_cashflow
            fcf = poste(flux, "fcf")
            if fcf.empty:
                cfo, capex = poste(flux, "cfo"), poste(flux, "capex")
                if not cfo.empty and not capex.empty:
                    fcf = cfo + capex.reindex(cfo.index)
            if not fcf.empty:
                sortie["fcf"] = _val(fcf)
                sortie["fcf_n1"] = _val(fcf, 4)
                if sortie.get("resultat_net"):
                    sortie["conversion_cash"] = sortie["fcf"] / sortie["resultat_net"]
            else:
                sortie["absents"].append("free cash-flow")
        except Exception:
            sortie["absents"].append("free cash-flow")

        try:
            bilan = valeur.quarterly_balance_sheet
            dette, treso = poste(bilan, "dette"), poste(bilan, "tresorerie")
            if not dette.empty:
                sortie["dette_nette"] = _val(dette) - (_val(treso) or 0)
                if sortie.get("marge_ebitda_pct") and sortie.get("ca"):
                    ebitda_annuel = (sortie["ca"] * sortie["marge_ebitda_pct"]
                                     / 100 * 4)
                    if ebitda_annuel:
                        sortie["levier"] = sortie["dette_nette"] / ebitda_annuel
            else:
                sortie["absents"].append("dette nette")
        except Exception:
            sortie["absents"].append("dette nette")

        sortie["absents"].extend(["croissance organique", "détail par segment",
                                  "guidance", "résultat ajusté"])
    except Exception as erreur:
        print(f"Comptes indisponibles pour {ticker} : {type(erreur).__name__}",
              file=sys.stderr)
    return sortie


def dernier_bpa(ticker: str) -> dict:
    """BPA publie et surprise, depuis le calendrier des publications."""
    try:
        dates = yf.Ticker(ticker).earnings_dates
        if dates is None or dates.empty:
            return {}
        index = pd.to_datetime(dates.index)
        table = dates.copy()
        table.index = index.tz_localize(None) if index.tz is not None else index
        if "Reported EPS" in table.columns:
            table = table[table["Reported EPS"].notna()]
        if table.empty:
            return {}
        ligne = table.sort_index().iloc[-1]
        date = table.sort_index().index[-1]
        surprise = ligne.get("Surprise(%)")
        attendu, publie = ligne.get("EPS Estimate"), ligne.get("Reported EPS")
        if (surprise is None or surprise != surprise) and attendu:
            surprise = (publie - attendu) / abs(attendu) * 100
        return {"date": date, "bpa": float(publie),
                "bpa_attendu": float(attendu) if attendu == attendu else None,
                "surprise_pct": float(surprise) if surprise == surprise else None,
                "jours": int((pd.Timestamp.now().normalize() - date).days)}
    except Exception:
        return {}


# ==========================================================================
# Verdict de thèse
# ==========================================================================

# Indicateurs reconnus dans la colonne Note de la feuille
INDICATEURS = {
    "marge nette": "marge_nette_pct",
    "marge op": "marge_op_pct",
    "marge opérationnelle": "marge_op_pct",
    "marge ebitda": "marge_ebitda_pct",
    "croissance": "ca_croissance_pct",
    "levier": "levier",
    "dette": "levier",
    "conversion": "conversion_cash",
}


def analyser_these(note: str) -> list[dict]:
    """
    Extrait les seuils de these d'une note libre.

    Reconnait les formulations du type « marge nette > 42 % », « croissance
    > 5 % », « levier < 2,5x ». Ce sont ces seuils qui permettent de dire si
    la these est validee ou invalidee par la publication.
    """
    import re
    if not note or not str(note).strip():
        return []

    texte = str(note).lower().replace(",", ".")
    regles = []
    motif = re.compile(
        r"([a-zéèêà\s]+?)\s*([<>≤≥]=?)\s*(-?\d+(?:\.\d+)?)\s*(%|x)?")
    for correspondance in motif.finditer(texte):
        libelle = correspondance.group(1).strip()
        operateur = correspondance.group(2)
        seuil = float(correspondance.group(3))

        champ = None
        for cle, valeur in INDICATEURS.items():
            if cle in libelle:
                champ = valeur
                break
        if champ:
            regles.append({"libelle": libelle, "champ": champ,
                           "operateur": operateur, "seuil": seuil})
    return regles


def verdict_these(regles: list[dict], publie: dict) -> list[dict]:
    """Confronte chaque indicateur de la these aux chiffres publies."""
    verdicts = []
    for regle in regles:
        valeur = publie.get(regle["champ"])
        if valeur is None or (isinstance(valeur, float) and valeur != valeur):
            verdicts.append({**regle, "valeur": None, "statut": "indisponible"})
            continue
        if regle["operateur"].startswith(">"):
            valide = valeur >= regle["seuil"]
        else:
            valide = valeur <= regle["seuil"]
        verdicts.append({**regle, "valeur": float(valeur),
                         "statut": "validé" if valide else "invalidé"})
    return verdicts


# ==========================================================================
# Lecture rédigée
# ==========================================================================

CONSIGNES = """Tu commentes la publication de résultats d'une société cotée, \
pour un investisseur français qui suit les marchés de près.

RÈGLES ABSOLUES
Ne calcule rien : tous les chiffres et écarts sont fournis.
N'invente aucun fait. Tu n'as PAS le communiqué intégral : tu ignores ce qu'a \
dit la direction, les perspectives données, la réaction du cours. Ne les \
invente jamais.
Ne recommande ni achat ni vente.

CE QUE TU DIS, dans cet ordre
1. Le verdict sur les attentes : dépasse, égale ou manque le consensus, et de \
combien. C'est l'écart qui fait le cours, pas le niveau absolu.
2. La marge : progresse ou s'érode face au même trimestre un an plus tôt. Un \
chiffre d'affaires en hausse avec une marge en baisse est une croissance achetée.
3. Le verdict de thèse si des indicateurs sont fournis : lesquels sont validés, \
lesquels ne le sont pas.
4. Ce qui manque. Dire ce qu'on ignore fait partie de l'information.

FORME
Trois à cinq phrases, denses, sans formule d'introduction. Commence par le verdict."""


def rediger(ticker: str, publie: dict, attendu: dict, verdicts: list[dict],
            cle: str = "", modele: str = "claude-sonnet-5") -> str:
    """
    Fait rediger la lecture. Renvoie une chaine vide en cas d'echec — les
    chiffres bruts restent exploitables sans elle.
    """
    cle = (cle or os.environ.get("CLE_ANTHROPIC", "")).strip()
    if not cle:
        return ""
    try:
        import anthropic
        espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        client = anthropic.Anthropic(
            api_key=cle,
            default_headers={"anthropic-workspace-id": espace} if espace else None)
        donnees = {
            "société": ticker,
            "chiffres_publiés": {k: v for k, v in publie.items()
                                 if v is not None and k != "absents"},
            "consensus": {k: v for k, v in attendu.items()
                          if v is not None and k != "absents"},
            "verdict_thèse": verdicts,
            "données_absentes": publie.get("absents", []) + attendu.get("absents", []),
        }
        reponse = client.messages.create(
            model=modele, max_tokens=600, system=CONSIGNES,
            messages=[{"role": "user", "content": json.dumps(
                donnees, ensure_ascii=False, indent=1, default=str)}])
        return "".join(b.text for b in reponse.content
                       if getattr(b, "type", "") == "text").strip()
    except Exception as erreur:
        print(f"Lecture IA indisponible — {type(erreur).__name__} : "
              f"{str(erreur)[:200]}", file=sys.stderr)
        return ""
