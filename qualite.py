"""
Controle de qualite des donnees.

Principe : yfinance est un scraper non officiel de Yahoo, sans garantie ni
engagement de service. Plutot que de lui faire confiance, on instrumente.

Ce module ne corrige rien — il signale. Une anomalie detectee n'est pas
forcement une erreur de donnee : un ecart de 30 % en une seance peut etre
reel (resultats, OPA, avertissement). Mais toute anomalie merite une
verification manuelle avant d'engager de l'argent sur le chiffre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Seuils de detection. Volontairement conservateurs : mieux vaut quelques
# fausses alertes qu'une erreur de donnee passee inapercue.
SEUIL_VARIATION = 0.35        # variation quotidienne suspecte
SEUIL_SPLIT = 0.45            # ecart evoquant un fractionnement non ajuste
JOURS_FIGES = 5               # cours identique plusieurs seances de suite
ECART_SOURCES = 0.02          # divergence acceptable entre deux sources


# ==========================================================================
# Controles sur une serie de cours
# ==========================================================================

def valeurs_impossibles(cours: pd.Series) -> pd.DataFrame:
    """Cours nuls ou negatifs — toujours une erreur, jamais un fait de marche."""
    fautifs = cours[(cours <= 0) | cours.isna()]
    if fautifs.empty:
        return pd.DataFrame()
    return pd.DataFrame({"Date": fautifs.index, "Valeur": fautifs.to_numpy(),
                         "Anomalie": "Cours nul, négatif ou manquant"})


def variations_extremes(cours: pd.Series,
                        seuil: float = SEUIL_VARIATION) -> pd.DataFrame:
    """
    Variations quotidiennes hors norme.

    Un ecart superieur a 35 % en une seance sur une grande capitalisation est
    presque toujours un fractionnement ou un dividende exceptionnel mal
    ajuste. Sur une petite valeur, il peut etre reel.
    """
    r = cours.pct_change().dropna()
    suspects = r[r.abs() > seuil]
    if suspects.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Date": suspects.index,
        "Variation (%)": suspects.to_numpy() * 100,
        "Cours avant": cours.shift(1).reindex(suspects.index).to_numpy(),
        "Cours après": cours.reindex(suspects.index).to_numpy(),
        "Anomalie": np.where(
            suspects.abs() > SEUIL_SPLIT,
            "Fractionnement probablement non ajusté",
            "Variation extrême — à vérifier"),
    })


def cours_figes(cours: pd.Series, jours: int = JOURS_FIGES) -> pd.DataFrame:
    """
    Cours identique plusieurs seances consecutives.

    Signale soit une suspension de cotation, soit — plus souvent — un defaut
    d'alimentation ou l'extension mecanique de la derniere valeur connue.
    """
    identique = cours.diff() == 0
    groupes = (~identique).cumsum()
    series = identique.groupby(groupes).sum()
    suspects = series[series >= jours]
    if suspects.empty:
        return pd.DataFrame()

    lignes = []
    for groupe in suspects.index:
        bloc = cours[groupes == groupe]
        lignes.append({
            "Début": bloc.index[0].date(), "Fin": bloc.index[-1].date(),
            "Séances": int(len(bloc)), "Valeur": float(bloc.iloc[0]),
            "Anomalie": "Cours inchangé sur plusieurs séances",
        })
    return pd.DataFrame(lignes)


def seances_manquantes(cours: pd.Series, tolerance: int = 5) -> pd.DataFrame:
    """
    Trous dans l'historique depassant un long week-end ferie.

    Quelques jours d'absence sont normaux (feries locaux). Au-dela d'une
    semaine, l'historique est lacunaire et fausse les calculs de volatilite.
    """
    if len(cours) < 2:
        return pd.DataFrame()
    ecarts = pd.Series(cours.index).diff().dt.days.dropna()
    trous = ecarts[ecarts > tolerance]
    if trous.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Reprise": [cours.index[i].date() for i in trous.index],
        "Jours manquants": trous.to_numpy().astype(int),
        "Anomalie": "Interruption de l'historique",
    })


def profondeur(cours: pd.Series) -> dict:
    """Etendue reelle de l'historique disponible, et sa densite."""
    s = cours.dropna()
    if s.empty:
        return {"séances": 0}
    duree = (s.index[-1] - s.index[0]).days
    attendues = duree / 365.25 * 252
    return {
        "première séance": s.index[0].date(),
        "dernière séance": s.index[-1].date(),
        "séances disponibles": int(len(s)),
        "séances attendues": int(attendues),
        "taux de couverture (%)": float(len(s) / attendues * 100) if attendues > 0 else np.nan,
        "ancienneté de la dernière donnée (jours)":
            int((pd.Timestamp.now().normalize() - s.index[-1].normalize()).days),
    }


def fraicheur(cours: pd.Series, seuil_jours: int = 5) -> dict:
    """
    Verifie que la derniere donnee est recente.

    Un decalage superieur a quelques jours ouvres signale un titre radie, un
    ticker errone, ou une alimentation interrompue.
    """
    s = cours.dropna()
    if s.empty:
        return {"état": "Aucune donnée", "à jour": False}
    retard = (pd.Timestamp.now().normalize() - s.index[-1].normalize()).days
    ouvres = int(np.busday_count(s.index[-1].date(),
                                 pd.Timestamp.now().date()))
    return {
        "dernière donnée": s.index[-1].date(),
        "retard (jours calendaires)": retard,
        "retard (jours ouvrés)": ouvres,
        "à jour": ouvres <= seuil_jours,
        "état": "À jour" if ouvres <= seuil_jours
                else f"Périmé de {ouvres} jours ouvrés",
    }


def controler(cours: pd.Series) -> dict:
    """Batterie complete de controles sur une serie de cours."""
    return {
        "profondeur": profondeur(cours),
        "fraîcheur": fraicheur(cours),
        "valeurs impossibles": valeurs_impossibles(cours),
        "variations extrêmes": variations_extremes(cours),
        "cours figés": cours_figes(cours),
        "séances manquantes": seances_manquantes(cours),
    }


def score_qualite(rapport: dict) -> dict:
    """
    Note synthetique de 0 a 100, pour trier rapidement.

    Le score n'est pas une garantie d'exactitude : il mesure l'absence
    d'anomalies DETECTABLES. Une donnee peut etre fausse et parfaitement
    reguliere — une erreur de retraitement de dividende, par exemple, ne
    laisse aucune trace visible.
    """
    note = 100
    motifs = []

    fr = rapport.get("fraîcheur", {})
    if not fr.get("à jour", True):
        note -= 30
        motifs.append(fr.get("état", "Données périmées"))

    couverture = rapport.get("profondeur", {}).get("taux de couverture (%)", 100)
    if np.isfinite(couverture) and couverture < 90:
        note -= 20
        motifs.append(f"Couverture de {couverture:.0f} % des séances attendues")

    for cle, penalite, libelle in [
        ("valeurs impossibles", 40, "cours nuls ou négatifs"),
        ("variations extrêmes", 15, "variations extrêmes"),
        ("cours figés", 10, "périodes de cours figé"),
        ("séances manquantes", 10, "interruptions d'historique"),
    ]:
        table = rapport.get(cle)
        if isinstance(table, pd.DataFrame) and not table.empty:
            note -= penalite
            motifs.append(f"{len(table)} {libelle}")

    note = max(0, note)
    niveau = ("Fiable" if note >= 85 else
              "Acceptable" if note >= 65 else
              "Douteux" if note >= 40 else "Inexploitable")
    return {"score": note, "niveau": niveau,
            "motifs": motifs or ["Aucune anomalie détectée"]}


# ==========================================================================
# Verification croisee
# ==========================================================================

def charger_stooq(ticker: str) -> pd.Series:
    """
    Cours de cloture depuis Stooq, source independante de Yahoo.

    Sert de contre-verification. Les conventions de tickers different :
    les valeurs americaines prennent le suffixe .US, les francaises .FR.
    """
    equivalences = {".PA": ".FR", ".DE": ".DE", ".L": ".UK", ".AS": ".NL",
                    ".MI": ".IT", ".MC": ".ES", ".SW": ".CH"}
    code = ticker
    for suffixe_yahoo, suffixe_stooq in equivalences.items():
        if ticker.endswith(suffixe_yahoo):
            code = ticker[: -len(suffixe_yahoo)] + suffixe_stooq
            break
    else:
        if "." not in ticker and "=" not in ticker and "^" not in ticker:
            code = ticker + ".US"

    try:
        url = f"https://stooq.com/q/d/l/?s={code.lower()}&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        return pd.to_numeric(df["Close"], errors="coerce").dropna().sort_index()
    except Exception:
        return pd.Series(dtype=float)


def comparer_sources(yahoo: pd.Series, autre: pd.Series,
                     tolerance: float = ECART_SOURCES) -> dict:
    """
    Confronte deux sources independantes sur leurs dates communes.

    Des ecarts de quelques dixiemes de pourcent sont normaux : arrondis,
    heures de cloture, traitement des dividendes. Au-dela de 2 %, l'une des
    deux sources se trompe.
    """
    if yahoo.empty or autre.empty:
        return {"comparable": False,
                "raison": "Source de contrôle indisponible pour ce ticker"}

    commun = pd.concat([yahoo.rename("yahoo"), autre.rename("autre")],
                       axis=1, join="inner").dropna()
    if len(commun) < 20:
        return {"comparable": False,
                "raison": f"Seulement {len(commun)} dates communes"}

    ecart = (commun["yahoo"] / commun["autre"] - 1).abs()
    divergences = commun[ecart > tolerance].copy()
    if not divergences.empty:
        divergences["Écart (%)"] = ecart[ecart > tolerance].to_numpy() * 100

    return {
        "comparable": True,
        "dates comparées": int(len(commun)),
        "écart médian (%)": float(ecart.median() * 100),
        "écart maximal (%)": float(ecart.max() * 100),
        "dates divergentes": int(len(divergences)),
        "part divergente (%)": float(len(divergences) / len(commun) * 100),
        "détail": divergences.tail(15),
        "verdict": ("Les deux sources concordent" if len(divergences) == 0
                    else f"{len(divergences)} dates divergent de plus de "
                         f"{tolerance * 100:.0f} %"),
    }


def controler_univers(cours: pd.DataFrame) -> pd.DataFrame:
    """Score de qualite pour chaque colonne d'un tableau de cours."""
    lignes = []
    for ticker in cours.columns:
        rapport = controler(cours[ticker].dropna())
        score = score_qualite(rapport)
        lignes.append({
            "Ticker": ticker,
            "Score": score["score"],
            "Niveau": score["niveau"],
            "Séances": rapport["profondeur"].get("séances disponibles", 0),
            "Couverture (%)": rapport["profondeur"].get("taux de couverture (%)", np.nan),
            "Dernière donnée": rapport["fraîcheur"].get("dernière donnée"),
            "À jour": rapport["fraîcheur"].get("à jour"),
            "Anomalies": "; ".join(score["motifs"][:3]),
        })
    return pd.DataFrame(lignes).set_index("Ticker").sort_values("Score")
