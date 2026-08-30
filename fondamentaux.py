"""
Analyse fondamentale.

Extrait et retraite les etats financiers fournis par yfinance, puis calcule
les ratios de qualite, de valorisation, de croissance et de solvabilite.

Deux precautions de methode. D'abord, yfinance ne fournit que quatre exercices :
c'est suffisant pour juger un niveau, insuffisant pour juger une tendance de
long terme. Ensuite, les intitules de postes varient selon l'emetteur et le
referentiel comptable — d'ou la recherche par alias plutot que par nom exact.
Tout poste manquant renvoie NaN plutot que de fausser un calcul.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ==========================================================================
# Extraction robuste
# ==========================================================================

ALIAS = {
    "chiffre_affaires": ["Total Revenue", "Operating Revenue", "Revenue"],
    "cout_ventes": ["Cost Of Revenue", "Cost Of Goods Sold",
                    "Reconciled Cost Of Revenue"],
    "marge_brute": ["Gross Profit"],
    "charges_exploitation": ["Operating Expense", "Total Operating Expenses"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "amortissements": ["Reconciled Depreciation",
                       "Depreciation And Amortization",
                       "Depreciation Amortization Depletion"],
    "charges_financieres": ["Interest Expense", "Interest Expense Non Operating"],
    "resultat_avant_impot": ["Pretax Income", "Income Before Tax"],
    "impots": ["Tax Provision", "Income Tax Expense"],
    "resultat_net": ["Net Income", "Net Income Common Stockholders",
                     "Net Income Continuous Operations"],
    "bpa": ["Diluted EPS", "Basic EPS"],
    "actions": ["Diluted Average Shares", "Basic Average Shares",
                "Share Issued", "Ordinary Shares Number"],
    # Bilan
    "actif_total": ["Total Assets"],
    "actif_courant": ["Current Assets", "Total Current Assets"],
    "passif_courant": ["Current Liabilities", "Total Current Liabilities"],
    "stocks": ["Inventory"],
    "tresorerie": ["Cash And Cash Equivalents",
                   "Cash Cash Equivalents And Short Term Investments"],
    "dette_totale": ["Total Debt"],
    "dette_long_terme": ["Long Term Debt"],
    "capitaux_propres": ["Stockholders Equity", "Total Equity Gross Minority Interest",
                         "Common Stock Equity"],
    "capital_investi": ["Invested Capital", "Total Capitalization"],
    "benefices_non_distribues": ["Retained Earnings"],
    # Flux
    "flux_exploitation": ["Operating Cash Flow",
                          "Cash Flow From Continuing Operating Activities"],
    "investissements": ["Capital Expenditure", "Purchase Of PPE"],
    "flux_libre": ["Free Cash Flow"],
    "dividendes_verses": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "rachats_actions": ["Repurchase Of Capital Stock"],
}


def poste(etat: pd.DataFrame, cle: str) -> pd.Series:
    """
    Recherche un poste comptable par ses intitules possibles.

    Renvoie une Series indexee par exercice, du plus recent au plus ancien,
    ou une Series vide si aucun alias ne correspond.
    """
    if etat is None or etat.empty:
        return pd.Series(dtype=float)
    for nom in ALIAS.get(cle, [cle]):
        if nom in etat.index:
            serie = pd.to_numeric(etat.loc[nom], errors="coerce")
            if serie.notna().any():
                return serie.dropna()
    return pd.Series(dtype=float)


def _val(serie: pd.Series, rang: int = 0) -> float:
    """Valeur du n-ieme exercice le plus recent."""
    return float(serie.iloc[rang]) if len(serie) > rang else np.nan


def _div(a: float, b: float) -> float:
    """Division protegee : renvoie NaN si le denominateur est nul ou absent."""
    if b is None or a is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


# ==========================================================================
# Etats retraites
# ==========================================================================

def ebitda_serie(income: pd.DataFrame) -> pd.Series:
    """
    EBITDA publie, ou reconstitue comme EBIT + amortissements.

    Beaucoup d'emetteurs ne publient pas de ligne EBITDA : la reconstituer
    evite de perdre tous les ratios qui en dependent.
    """
    direct = poste(income, "ebitda")
    if not direct.empty:
        return direct
    ebit = poste(income, "ebit")
    amort = poste(income, "amortissements")
    if ebit.empty or amort.empty:
        return pd.Series(dtype=float)
    return ebit + amort.reindex(ebit.index)


def compte_de_resultat(income: pd.DataFrame) -> pd.DataFrame:
    """Compte de resultat condense, en unites de la devise de publication."""
    ca = poste(income, "chiffre_affaires")
    if ca.empty:
        return pd.DataFrame()

    brute = poste(income, "marge_brute")
    cout = poste(income, "cout_ventes")
    if brute.empty and not cout.empty:
        brute = ca - cout.reindex(ca.index)

    ebitda = ebitda_serie(income)
    ebit = poste(income, "ebit")

    lignes = {
        "Chiffre d'affaires": ca,
        "Marge brute": brute,
        "EBITDA": ebitda,
        "EBIT (résultat d'exploitation)": ebit,
        "Résultat avant impôt": poste(income, "resultat_avant_impot"),
        "Résultat net": poste(income, "resultat_net"),
        "BPA dilué": poste(income, "bpa"),
    }
    out = pd.DataFrame({k: v for k, v in lignes.items() if not v.empty})
    return out.T if not out.empty else pd.DataFrame()


def marges(income: pd.DataFrame) -> pd.DataFrame:
    """Marges en pourcentage du chiffre d'affaires, par exercice."""
    cr = compte_de_resultat(income)
    if cr.empty or "Chiffre d'affaires" not in cr.index:
        return pd.DataFrame()

    ca = cr.loc["Chiffre d'affaires"]
    out = {}
    for nom, cible in [("Marge brute", "Marge brute (%)"),
                       ("EBITDA", "Marge d'EBITDA (%)"),
                       ("EBIT (résultat d'exploitation)", "Marge opérationnelle (%)"),
                       ("Résultat net", "Marge nette (%)")]:
        if nom in cr.index:
            out[cible] = cr.loc[nom] / ca.replace(0, np.nan) * 100
    return pd.DataFrame(out).T


def flux_tresorerie(cashflow: pd.DataFrame) -> pd.DataFrame:
    """
    Flux de tresorerie et flux libre.

    Le flux libre est le poste le plus difficile a manipuler comptablement :
    c'est de l'argent reellement encaisse, pas une convention. A privilegier
    sur le resultat net quand les deux divergent durablement.
    """
    cfo = poste(cashflow, "flux_exploitation")
    if cfo.empty:
        return pd.DataFrame()

    capex = poste(cashflow, "investissements")
    fcf = poste(cashflow, "flux_libre")
    if fcf.empty and not capex.empty:
        fcf = cfo + capex.reindex(cfo.index)  # capex est negatif chez Yahoo

    lignes = {
        "Flux d'exploitation": cfo,
        "Investissements (capex)": capex,
        "Flux de trésorerie libre": fcf,
        "Dividendes versés": poste(cashflow, "dividendes_verses"),
        "Rachats d'actions": poste(cashflow, "rachats_actions"),
    }
    out = pd.DataFrame({k: v for k, v in lignes.items() if not v.empty})
    return out.T if not out.empty else pd.DataFrame()


# ==========================================================================
# Ratios
# ==========================================================================

def rentabilite(income: pd.DataFrame, bilan: pd.DataFrame,
                cashflow: pd.DataFrame) -> dict:
    """
    Ratios de rentabilite du capital.

    Le ROIC est le plus important de tous : il mesure ce que l'entreprise
    gagne sur chaque euro immobilise. Une societe dont le ROIC depasse
    durablement son cout du capital cree de la valeur ; en dessous, elle en
    detruit, quelle que soit sa croissance.
    """
    rn = _val(poste(income, "resultat_net"))
    ebit = _val(poste(income, "ebit"))
    impots = _val(poste(income, "impots"))
    rai = _val(poste(income, "resultat_avant_impot"))
    cp = _val(poste(bilan, "capitaux_propres"))
    actifs = _val(poste(bilan, "actif_total"))
    dette = _val(poste(bilan, "dette_totale"))
    treso = _val(poste(bilan, "tresorerie"))
    cfo = _val(poste(cashflow, "flux_exploitation"))
    ca = _val(poste(income, "chiffre_affaires"))

    taux_impot = _div(impots, rai)
    taux_impot = taux_impot if np.isfinite(taux_impot) and 0 <= taux_impot < 0.6 else 0.25
    nopat = ebit * (1 - taux_impot) if np.isfinite(ebit) else np.nan
    capital = (cp + dette - treso) if all(np.isfinite([cp, dette, treso])) else cp

    return {
        "ROE — rentabilité des capitaux propres (%)": _div(rn, cp) * 100,
        "ROA — rentabilité des actifs (%)": _div(rn, actifs) * 100,
        "ROIC — rentabilité du capital investi (%)": _div(nopat, capital) * 100,
        "Rotation des actifs": _div(ca, actifs),
        "Conversion en cash (CFO / résultat net)": _div(cfo, rn),
    }


def solvabilite(income: pd.DataFrame, bilan: pd.DataFrame) -> dict:
    """
    Capacite a honorer les echeances.

    La dette nette rapportee a l'EBITDA est le ratio que surveillent les
    preteurs : au-dela de 3, la marge de manoeuvre se reduit nettement.
    """
    dette = _val(poste(bilan, "dette_totale"))
    treso = _val(poste(bilan, "tresorerie"))
    cp = _val(poste(bilan, "capitaux_propres"))
    ac = _val(poste(bilan, "actif_courant"))
    pc = _val(poste(bilan, "passif_courant"))
    stocks = _val(poste(bilan, "stocks"))
    ebitda = _val(ebitda_serie(income))
    ebit = _val(poste(income, "ebit"))
    interets = abs(_val(poste(income, "charges_financieres")))

    dette_nette = dette - treso if np.isfinite(dette) and np.isfinite(treso) else np.nan
    return {
        "Dette nette / EBITDA": _div(dette_nette, ebitda),
        "Dette / capitaux propres": _div(dette, cp),
        "Couverture des intérêts (EBIT / charges)": _div(ebit, interets),
        "Liquidité générale": _div(ac, pc),
        "Liquidité réduite": _div(ac - stocks, pc) if np.isfinite(stocks) else np.nan,
        "Dette nette": dette_nette,
    }


def croissance(income: pd.DataFrame, cashflow: pd.DataFrame) -> dict:
    """
    Taux de croissance annuels moyens sur l'historique disponible.

    Sur quatre exercices, le chiffre reste fragile : un exercice atypique
    suffit a le fausser. A lire comme un ordre de grandeur.
    """
    def cagr(serie: pd.Series) -> float:
        s = serie.dropna()
        if len(s) < 2:
            return np.nan
        recent, ancien = float(s.iloc[0]), float(s.iloc[-1])
        annees = len(s) - 1
        if ancien <= 0 or recent <= 0:
            return np.nan
        return ((recent / ancien) ** (1 / annees) - 1) * 100

    fcf = poste(cashflow, "flux_libre")
    if fcf.empty:
        cfo = poste(cashflow, "flux_exploitation")
        capex = poste(cashflow, "investissements")
        if not cfo.empty and not capex.empty:
            fcf = cfo + capex.reindex(cfo.index)

    return {
        "Croissance du CA (%/an)": cagr(poste(income, "chiffre_affaires")),
        "Croissance de l'EBITDA (%/an)": cagr(ebitda_serie(income)),
        "Croissance du résultat net (%/an)": cagr(poste(income, "resultat_net")),
        "Croissance du BPA (%/an)": cagr(poste(income, "bpa")),
        "Croissance du flux libre (%/an)": cagr(fcf),
    }


def valorisation(info: dict, income: pd.DataFrame, bilan: pd.DataFrame,
                 cashflow: pd.DataFrame) -> dict:
    """
    Multiples de valorisation.

    L'EV/EBITDA est preferable au PER pour comparer des societes d'endettement
    different : il raisonne sur la valeur d'entreprise, dette comprise, la ou
    le PER ignore la structure financiere.
    """
    capi = info.get("marketCap")
    dette = _val(poste(bilan, "dette_totale"))
    treso = _val(poste(bilan, "tresorerie"))
    ve = (capi + dette - treso) if all(
        v is not None and np.isfinite(v) for v in [capi, dette, treso]) else None

    ca = _val(poste(income, "chiffre_affaires"))
    ebitda = _val(ebitda_serie(income))
    ebit = _val(poste(income, "ebit"))
    cfo = _val(poste(cashflow, "flux_exploitation"))
    capex = _val(poste(cashflow, "investissements"))
    fcf = _val(poste(cashflow, "flux_libre"))
    if not np.isfinite(fcf) and np.isfinite(cfo) and np.isfinite(capex):
        fcf = cfo + capex

    rendement = info.get("dividendYield")
    return {
        "Capitalisation": capi,
        "Valeur d'entreprise": ve,
        "PER (12 derniers mois)": info.get("trailingPE"),
        "PER prévisionnel": info.get("forwardPE"),
        "PEG (PER / croissance)": info.get("pegRatio"),
        "Cours / actif net": info.get("priceToBook"),
        "Cours / chiffre d'affaires": info.get("priceToSalesTrailing12Months"),
        "EV / chiffre d'affaires": _div(ve, ca),
        "EV / EBITDA": _div(ve, ebitda),
        "EV / EBIT": _div(ve, ebit),
        "Rendement du flux libre (%)": _div(fcf, capi) * 100,
        "Rendement du dividende (%)": (rendement * 100) if rendement else np.nan,
        "Taux de distribution (%)": (info.get("payoutRatio") or np.nan) * 100,
    }


# ==========================================================================
# Scores composites
# ==========================================================================

def piotroski(income: pd.DataFrame, bilan: pd.DataFrame,
              cashflow: pd.DataFrame) -> dict:
    """
    Score F de Piotroski, de 0 a 9.

    Neuf tests binaires sur la rentabilite, la structure financiere et
    l'efficacite operationnelle, chacun comparant l'exercice courant au
    precedent. Au-dessus de 7 le profil est solide, en dessous de 3 il est
    fragile. Concu a l'origine pour trier les valeurs decotees, ou il separe
    les societes bon marche des societes en difficulte.
    """
    rn = poste(income, "resultat_net")
    ca = poste(income, "chiffre_affaires")
    brute = poste(income, "marge_brute")
    actifs = poste(bilan, "actif_total")
    dette_lt = poste(bilan, "dette_long_terme")
    ac = poste(bilan, "actif_courant")
    pc = poste(bilan, "passif_courant")
    actions = poste(bilan, "actions")
    cfo = poste(cashflow, "flux_exploitation")

    tests, details = {}, {}

    def deux(s):
        return len(s) >= 2

    roa_n = _div(_val(rn), _val(actifs))
    roa_p = _div(_val(rn, 1), _val(actifs, 1)) if deux(rn) and deux(actifs) else np.nan

    tests["Résultat net positif"] = _val(rn) > 0
    tests["Flux d'exploitation positif"] = _val(cfo) > 0
    tests["ROA en progression"] = roa_n > roa_p if np.isfinite(roa_p) else False
    tests["Flux supérieur au résultat net"] = _val(cfo) > _val(rn)
    tests["Endettement long terme en baisse"] = (
        _val(dette_lt) < _val(dette_lt, 1) if deux(dette_lt) else False)
    tests["Liquidité en amélioration"] = (
        _div(_val(ac), _val(pc)) > _div(_val(ac, 1), _val(pc, 1))
        if deux(ac) and deux(pc) else False)
    tests["Pas de dilution"] = (
        _val(actions) <= _val(actions, 1) * 1.02 if deux(actions) else False)
    tests["Marge brute en progression"] = (
        _div(_val(brute), _val(ca)) > _div(_val(brute, 1), _val(ca, 1))
        if deux(brute) and deux(ca) else False)
    tests["Rotation des actifs en progression"] = (
        _div(_val(ca), _val(actifs)) > _div(_val(ca, 1), _val(actifs, 1))
        if deux(ca) and deux(actifs) else False)

    for k, v in tests.items():
        details[k] = bool(v) if isinstance(v, (bool, np.bool_)) else False
    return {"score": sum(details.values()), "détail": details}


def altman_z(income: pd.DataFrame, bilan: pd.DataFrame,
             capitalisation: float | None) -> dict:
    """
    Score Z d'Altman : probabilite de defaillance a deux ans.

    Au-dessus de 2,99 la situation est saine, entre 1,81 et 2,99 elle est
    incertaine, en dessous de 1,81 le risque de defaut est eleve. Peu
    pertinent pour les banques et les societes financieres, dont la structure
    de bilan est differente.
    """
    actifs = _val(poste(bilan, "actif_total"))
    ac = _val(poste(bilan, "actif_courant"))
    pc = _val(poste(bilan, "passif_courant"))
    reserves = _val(poste(bilan, "benefices_non_distribues"))
    cp = _val(poste(bilan, "capitaux_propres"))
    ebit = _val(poste(income, "ebit"))
    ca = _val(poste(income, "chiffre_affaires"))
    passif_total = (actifs - cp) if np.isfinite(actifs) and np.isfinite(cp) else np.nan

    x1 = _div(ac - pc, actifs)
    x2 = _div(reserves, actifs)
    x3 = _div(ebit, actifs)
    x4 = _div(capitalisation, passif_total) if capitalisation else np.nan
    x5 = _div(ca, actifs)

    composantes = [1.2 * x1, 1.4 * x2, 3.3 * x3, 0.6 * x4, 1.0 * x5]
    if sum(np.isfinite(c) for c in composantes) < 4:
        return {"score": np.nan, "verdict": "Données insuffisantes"}

    z = float(np.nansum(composantes))
    verdict = ("Situation saine" if z > 2.99
               else "Zone d'incertitude" if z > 1.81
               else "Risque de défaillance élevé")
    return {"score": z, "verdict": verdict}


def synthese(info: dict, income: pd.DataFrame, bilan: pd.DataFrame,
             cashflow: pd.DataFrame) -> dict:
    """Tous les ratios rassembles par famille."""
    return {
        "Rentabilité": rentabilite(income, bilan, cashflow),
        "Solvabilité": solvabilite(income, bilan),
        "Croissance": croissance(income, cashflow),
        "Valorisation": valorisation(info, income, bilan, cashflow),
    }
