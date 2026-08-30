"""
Resultats trimestriels, consensus des analystes et actualites.

Precaution de methode sur le consensus : il est structurellement optimiste.
Les objectifs de cours moyens depassent le cours au comptant dans la grande
majorite des cas, et les recommandations de vente sont rares — les analystes
couvrent des societes dont ils suivent aussi les emissions. Ce qui est
reellement exploitable, ce n'est pas le niveau du consensus mais sa REVISION :
un objectif releve ou abaisse porte davantage d'information que sa valeur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Maisons dont les changements d'avis pesent le plus sur les cours
GRANDES_MAISONS = [
    "Goldman Sachs", "Morgan Stanley", "JP Morgan", "JPMorgan",
    "Bank of America", "BofA", "Citigroup", "Citi", "UBS", "Barclays",
    "Deutsche Bank", "Wells Fargo", "Jefferies", "HSBC", "BNP Paribas",
    "Credit Suisse", "RBC", "Evercore", "Bernstein", "Redburn",
]


# ==========================================================================
# Resultats trimestriels
# ==========================================================================

def resultats_trimestriels(q_income: pd.DataFrame) -> pd.DataFrame:
    """
    Comptes trimestriels condenses.

    Les societes europeennes publient souvent au semestre : dans ce cas
    Yahoo ne renvoie que deux periodes par an, ce qui est normal et non un
    defaut de donnees.
    """
    if q_income is None or q_income.empty:
        return pd.DataFrame()

    from fondamentaux import ebitda_serie, poste

    ca = poste(q_income, "chiffre_affaires")
    if ca.empty:
        return pd.DataFrame()

    lignes = {
        "Chiffre d'affaires": ca,
        "Marge brute": poste(q_income, "marge_brute"),
        "EBITDA": ebitda_serie(q_income),
        "EBIT": poste(q_income, "ebit"),
        "Résultat net": poste(q_income, "resultat_net"),
        "BPA dilué": poste(q_income, "bpa"),
    }
    out = pd.DataFrame({k: v for k, v in lignes.items() if not v.empty})
    if out.empty:
        return pd.DataFrame()

    out = out.sort_index(ascending=False)
    out["Marge nette (%)"] = (
        out.get("Résultat net", np.nan) / out["Chiffre d'affaires"].replace(0, np.nan) * 100)
    out["Marge d'EBITDA (%)"] = (
        out.get("EBITDA", np.nan) / out["Chiffre d'affaires"].replace(0, np.nan) * 100)
    return out.T


def croissance_annuelle_glissante(q_income: pd.DataFrame) -> pd.DataFrame:
    """
    Croissance de chaque trimestre face au meme trimestre un an plus tot.

    C'est la seule comparaison valable pour une activite saisonniere : un
    quatrieme trimestre se compare au quatrieme trimestre precedent, jamais
    au troisieme.
    """
    res = resultats_trimestriels(q_income)
    if res.empty:
        return pd.DataFrame()

    postes = [p for p in ["Chiffre d'affaires", "EBITDA", "Résultat net"]
              if p in res.index]
    if not postes:
        return pd.DataFrame()

    sous = res.loc[postes]
    if sous.shape[1] < 5:
        return pd.DataFrame()

    out = {}
    for i in range(sous.shape[1] - 4):
        recent = sous.iloc[:, i]
        ancien = sous.iloc[:, i + 4]
        out[sous.columns[i]] = (recent / ancien.replace(0, np.nan) - 1) * 100
    return pd.DataFrame(out)


# ==========================================================================
# Publications et surprises
# ==========================================================================

def surprises(dates_resultats: pd.DataFrame) -> pd.DataFrame:
    """
    Historique des publications : attendu, publie, ecart.

    Une suite de surprises positives traduit soit une execution solide, soit
    une direction qui guide prudemment les analystes. L'inverse est plus
    rarement innocent.
    """
    if dates_resultats is None or dates_resultats.empty:
        return pd.DataFrame()

    df = dates_resultats.copy()
    renommage = {
        "EPS Estimate": "BPA attendu",
        "Reported EPS": "BPA publié",
        "Surprise(%)": "Surprise (%)",
    }
    df = df.rename(columns={k: v for k, v in renommage.items() if k in df.columns})
    colonnes = [c for c in ["BPA attendu", "BPA publié", "Surprise (%)"]
                if c in df.columns]
    if not colonnes:
        return pd.DataFrame()

    df = df[colonnes].dropna(how="all")
    if "Surprise (%)" not in df.columns and {"BPA attendu", "BPA publié"} <= set(df.columns):
        df["Surprise (%)"] = ((df["BPA publié"] - df["BPA attendu"])
                              / df["BPA attendu"].abs().replace(0, np.nan) * 100)
    return df.sort_index(ascending=False)


def prochaine_publication(dates_resultats: pd.DataFrame) -> dict:
    """Date de la prochaine publication attendue et BPA anticipe."""
    if dates_resultats is None or dates_resultats.empty:
        return {}
    futur = dates_resultats[dates_resultats["Reported EPS"].isna()] \
        if "Reported EPS" in dates_resultats.columns else pd.DataFrame()
    if futur.empty:
        return {}
    ligne = futur.sort_index().iloc[0]
    return {
        "date": futur.sort_index().index[0],
        "bpa_attendu": float(ligne.get("EPS Estimate", np.nan)),
    }


def taux_de_reussite(dates_resultats: pd.DataFrame) -> dict:
    """Part des publications ayant depasse le consensus."""
    s = surprises(dates_resultats)
    if s.empty or "Surprise (%)" not in s.columns:
        return {}
    valides = s["Surprise (%)"].dropna()
    if valides.empty:
        return {}
    return {
        "publications analysées": int(len(valides)),
        "au-dessus du consensus (%)": float((valides > 0).mean() * 100),
        "surprise moyenne (%)": float(valides.mean()),
        "surprise médiane (%)": float(valides.median()),
    }


# ==========================================================================
# Consensus
# ==========================================================================

def objectif_de_cours(info: dict, cours_actuel: float | None = None) -> dict:
    """
    Consensus d'objectif de cours et potentiel implicite.

    A lire avec le nombre d'analystes : un consensus fonde sur deux avis n'a
    pas le poids d'un consensus fonde sur trente.
    """
    actuel = cours_actuel or info.get("currentPrice") or info.get("regularMarketPrice")
    moyen = info.get("targetMeanPrice")
    resultat = {
        "Cours actuel": actuel,
        "Objectif bas": info.get("targetLowPrice"),
        "Objectif moyen": moyen,
        "Objectif médian": info.get("targetMedianPrice"),
        "Objectif haut": info.get("targetHighPrice"),
        "Nombre d'analystes": info.get("numberOfAnalystOpinions"),
    }
    if actuel and moyen:
        resultat["Potentiel implicite (%)"] = (moyen / actuel - 1) * 100
        resultat["Dispersion des objectifs (%)"] = (
            (info["targetHighPrice"] - info["targetLowPrice"]) / moyen * 100
            if info.get("targetHighPrice") and info.get("targetLowPrice") else np.nan)
    return resultat


def distribution_avis(recommandations: pd.DataFrame) -> pd.DataFrame:
    """Repartition des recommandations, du plus recent au plus ancien."""
    if recommandations is None or recommandations.empty:
        return pd.DataFrame()

    renommage = {
        "strongBuy": "Achat fort", "buy": "Achat", "hold": "Conserver",
        "sell": "Vendre", "strongSell": "Vente forte", "period": "Période",
    }
    df = recommandations.rename(columns=renommage)
    colonnes = [c for c in ["Achat fort", "Achat", "Conserver", "Vendre",
                            "Vente forte"] if c in df.columns]
    if not colonnes:
        return pd.DataFrame()
    if "Période" in df.columns:
        df = df.set_index("Période")
    return df[colonnes]


def note_moyenne(info: dict) -> dict:
    """
    Note de synthese Yahoo, de 1 (achat fort) a 5 (vente forte).

    La distribution reelle est fortement asymetrique : la moyenne du marche
    tourne autour de 2,2, si bien qu'une note de 2,5 constitue deja un avis
    tiede plutot que neutre.
    """
    note = info.get("recommendationMean")
    cle = info.get("recommendationKey", "")
    traduction = {
        "strong_buy": "Achat fort", "buy": "Achat", "hold": "Conserver",
        "sell": "Vendre", "strong_sell": "Vente forte", "underperform": "Sous-performance",
        "outperform": "Surperformance", "none": "Aucun avis",
    }
    return {
        "note": float(note) if note else np.nan,
        "avis": traduction.get(cle, cle or "—"),
        "analystes": info.get("numberOfAnalystOpinions"),
    }


# ==========================================================================
# Estimations et revisions
# ==========================================================================

def estimations(estimation_ca: pd.DataFrame,
                estimation_bpa: pd.DataFrame) -> pd.DataFrame:
    """Prévisions de chiffre d'affaires et de bénéfice par période."""
    libelles = {"avg": "moyen", "low": "bas", "high": "haut",
                "numberOfAnalysts": "analystes", "growth": "croissance",
                "yearAgoRevenue": "an dernier", "yearAgoEps": "an dernier"}

    blocs = []
    for source, prefixe in [(estimation_ca, "CA"), (estimation_bpa, "BPA")]:
        if source is None or getattr(source, "empty", True):
            continue
        df = source.copy()
        # Prefixer TOUTES les colonnes, y compris celles non prevues : yfinance
        # fait evoluer ses champs, et deux colonnes homonymes font echouer
        # l'affichage.
        df.columns = [f"{prefixe} — {libelles.get(str(c), str(c))}"
                      for c in df.columns]
        blocs.append(df)

    if not blocs:
        return pd.DataFrame()

    fusion = pd.concat(blocs, axis=1)
    return fusion.loc[:, ~fusion.columns.duplicated()]


def revisions(eps_trend: pd.DataFrame) -> pd.DataFrame:
    """
    Evolution de l'estimation de BPA sur 7, 30, 60 et 90 jours.

    C'est le signal le plus exploitable de toute cette page. Le momentum des
    revisions — le fait que les analystes relevent ou abaissent leurs
    chiffres — est documente comme predicteur, contrairement au niveau du
    consensus lui-meme.
    """
    if eps_trend is None or eps_trend.empty:
        return pd.DataFrame()

    renommage = {
        "current": "Estimation actuelle", "7daysAgo": "Il y a 7 jours",
        "30daysAgo": "Il y a 30 jours", "60daysAgo": "Il y a 60 jours",
        "90daysAgo": "Il y a 90 jours",
    }
    df = eps_trend.rename(columns={k: v for k, v in renommage.items()
                                   if k in eps_trend.columns})
    colonnes = [c for c in renommage.values() if c in df.columns]
    if not colonnes:
        return pd.DataFrame()

    out = df[colonnes].copy()
    if {"Estimation actuelle", "Il y a 90 jours"} <= set(out.columns):
        out["Révision sur 90 jours (%)"] = (
            (out["Estimation actuelle"] - out["Il y a 90 jours"])
            / out["Il y a 90 jours"].abs().replace(0, np.nan) * 100)
    return out


def sens_des_revisions(eps_revisions: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'analystes ayant releve ou abaisse leur estimation."""
    if eps_revisions is None or eps_revisions.empty:
        return pd.DataFrame()
    renommage = {
        "upLast7days": "Relevé (7 j)", "upLast30days": "Relevé (30 j)",
        "downLast7days": "Abaissé (7 j)", "downLast30days": "Abaissé (30 j)",
    }
    df = eps_revisions.rename(columns={k: v for k, v in renommage.items()
                                       if k in eps_revisions.columns})
    colonnes = [c for c in renommage.values() if c in df.columns]
    return df[colonnes] if colonnes else pd.DataFrame()


def changements_davis(upgrades: pd.DataFrame, limite: int = 25,
                      grandes_maisons_seulement: bool = False) -> pd.DataFrame:
    """
    Historique des relevements et abaissements, avec le nom de la maison.

    C'est ici qu'apparaissent nommement Goldman Sachs, Morgan Stanley ou
    JP Morgan. Leur objectif de cours courant n'est pas diffuse gratuitement,
    mais leurs changements d'avis le sont — et ce sont eux qui deplacent
    reellement les cours.
    """
    if upgrades is None or upgrades.empty:
        return pd.DataFrame()

    df = upgrades.copy()
    renommage = {"Firm": "Maison", "ToGrade": "Nouvel avis",
                 "FromGrade": "Avis précédent", "Action": "Sens"}
    df = df.rename(columns={k: v for k, v in renommage.items() if k in df.columns})

    if grandes_maisons_seulement and "Maison" in df.columns:
        motif = "|".join(GRANDES_MAISONS)
        df = df[df["Maison"].astype(str).str.contains(motif, case=False, na=False)]

    if "Sens" in df.columns:
        traduction = {"up": "Relèvement", "down": "Abaissement",
                      "init": "Initiation", "main": "Maintien",
                      "reit": "Réitération"}
        df["Sens"] = df["Sens"].map(lambda x: traduction.get(str(x), str(x)))

    colonnes = [c for c in ["Maison", "Avis précédent", "Nouvel avis", "Sens"]
                if c in df.columns]
    return df[colonnes].sort_index(ascending=False).head(limite)


def solde_des_avis(upgrades: pd.DataFrame, jours: int = 90) -> dict:
    """Relevements moins abaissements sur la periode recente."""
    if upgrades is None or upgrades.empty or "Action" not in upgrades.columns:
        return {}
    recent = upgrades.copy()
    try:
        limite = pd.Timestamp.now(tz=recent.index.tz) - pd.Timedelta(days=jours)
        recent = recent[recent.index >= limite]
    except Exception:
        recent = recent.head(30)
    if recent.empty:
        return {}
    hausses = int((recent["Action"] == "up").sum())
    baisses = int((recent["Action"] == "down").sum())
    return {"relèvements": hausses, "abaissements": baisses,
            "solde": hausses - baisses}


# ==========================================================================
# Actualites
# ==========================================================================

def actualites(news: list, limite: int = 15) -> pd.DataFrame:
    """
    Normalise le flux d'actualites, dont le format varie selon les versions.

    Les titres sont a lire comme un contexte, jamais comme un signal : au
    moment ou une information est publiee, elle est deja dans les cours.
    """
    if not news:
        return pd.DataFrame()

    lignes = []
    for article in news[:limite]:
        contenu = article.get("content", article) if isinstance(article, dict) else {}
        titre = contenu.get("title") or article.get("title")
        if not titre:
            continue
        editeur = (contenu.get("provider", {}).get("displayName")
                   if isinstance(contenu.get("provider"), dict)
                   else article.get("publisher", "—"))
        lien = (contenu.get("canonicalUrl", {}).get("url")
                if isinstance(contenu.get("canonicalUrl"), dict)
                else article.get("link", ""))
        date = contenu.get("pubDate") or article.get("providerPublishTime")
        if isinstance(date, (int, float)):
            date = pd.to_datetime(date, unit="s")
        lignes.append({"Date": date, "Titre": titre,
                       "Source": editeur or "—", "Lien": lien or ""})
    return pd.DataFrame(lignes)
