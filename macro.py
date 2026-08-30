"""
Contexte macroeconomique et calendriers.

Deux sources. Les series economiques viennent de FRED, la base publique de la
Reserve federale de Saint-Louis, accessible sans cle. Les indicateurs de
marche viennent de Yahoo.

Sur le calendrier : aucune API gratuite ne diffuse le calendrier
macroeconomique. Les dates ci-dessous sont donc RECONSTITUEES par regles a
partir des schemas de publication habituels — emploi le premier vendredi,
ISM le premier jour ouvre, et ainsi de suite. Elles sont justes a un jour
pres et doivent etre verifiees sur les sites officiels avant tout usage
engageant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ==========================================================================
# Series FRED
# ==========================================================================

URL_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    # Taux et politique monetaire
    "FEDFUNDS": ("Taux directeur Fed (%)", "Taux"),
    "DGS2": ("Taux 2 ans US (%)", "Taux"),
    "DGS10": ("Taux 10 ans US (%)", "Taux"),
    "DGS30": ("Taux 30 ans US (%)", "Taux"),
    "T10Y2Y": ("Écart 10 ans − 2 ans (%)", "Taux"),
    "T10Y3M": ("Écart 10 ans − 3 mois (%)", "Taux"),
    "ECBDFR": ("Taux de dépôt BCE (%)", "Taux"),
    "IRLTLT01DEM156N": ("Taux 10 ans Allemagne (%)", "Taux"),
    # Inflation
    "CPIAUCSL": ("Indice des prix US", "Inflation"),
    "CPILFESL": ("Inflation sous-jacente US (indice)", "Inflation"),
    "PCEPILFE": ("PCE sous-jacent (indice)", "Inflation"),
    "T5YIE": ("Anticipation d'inflation 5 ans (%)", "Inflation"),
    "T10YIE": ("Anticipation d'inflation 10 ans (%)", "Inflation"),
    # Activite et emploi
    "GDPC1": ("PIB réel US (Md$)", "Activité"),
    "UNRATE": ("Taux de chômage US (%)", "Activité"),
    "PAYEMS": ("Emplois non agricoles (milliers)", "Activité"),
    "ICSA": ("Inscriptions hebdo au chômage", "Activité"),
    "INDPRO": ("Production industrielle (indice)", "Activité"),
    "UMCSENT": ("Confiance des ménages", "Activité"),
    "RSAFS": ("Ventes au détail (M$)", "Activité"),
    # Credit et risque
    "BAMLH0A0HYM2": ("Spread haut rendement US (%)", "Crédit"),
    "BAMLC0A0CM": ("Spread investment grade (%)", "Crédit"),
    "VIXCLS": ("VIX", "Risque"),
    "STLFSI4": ("Indice de stress financier", "Risque"),
    "DTWEXBGS": ("Indice dollar (large)", "Risque"),
    "WALCL": ("Bilan de la Fed (M$)", "Liquidité"),
    "M2SL": ("Masse monétaire M2 (Md$)", "Liquidité"),
}

COURBE = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1,
    "DGS2": 2, "DGS3": 3, "DGS5": 5, "DGS7": 7, "DGS10": 10,
    "DGS20": 20, "DGS30": 30,
}

# Indicateurs de marche, via Yahoo
MARCHE = {
    "^VIX": "VIX (volatilité implicite)",
    "^TNX": "Taux 10 ans US",
    "DX-Y.NYB": "Indice dollar",
    "GC=F": "Or",
    "CL=F": "Pétrole WTI",
    "BZ=F": "Pétrole Brent",
    "HG=F": "Cuivre",
    "^GSPC": "S&P 500",
    "^STOXX50E": "Euro Stoxx 50",
    "EURUSD=X": "EUR/USD",
    "BTC-USD": "Bitcoin",
}

LIENS_OFFICIELS = {
    "Réserve fédérale (FOMC)": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "Bureau of Labor Statistics": "https://www.bls.gov/schedule/news_release/",
    "Bureau of Economic Analysis": "https://www.bea.gov/news/schedule",
    "Banque centrale européenne": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
    "Eurostat": "https://ec.europa.eu/eurostat/news/release-calendar",
    "INSEE": "https://www.insee.fr/fr/information/1405619",
}


def charger_serie(code: str) -> pd.Series:
    """
    Telecharge une serie FRED. Renvoie une Series vide en cas d'echec.

    FRED expose ses donnees en CSV sans authentification, ce qui evite d'avoir
    a gerer une cle d'API.
    """
    try:
        df = pd.read_csv(URL_FRED.format(code), parse_dates=[0], index_col=0)
        serie = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        serie.name = SERIES.get(code, (code,))[0]
        return serie
    except Exception:
        return pd.Series(dtype=float, name=code)


def tableau_de_bord(series: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Derniere valeur de chaque serie, avec ses variations.

    Les variations sont exprimees en ecart absolu pour les taux (points de
    pourcentage) et en pourcentage pour les niveaux — melanger les deux est
    une erreur de lecture frequente.
    """
    lignes = []
    for code, serie in series.items():
        if serie is None or serie.empty:
            continue
        libelle, famille = SERIES.get(code, (code, "Autre"))
        derniere = float(serie.iloc[-1])
        est_taux = famille in ("Taux", "Crédit") or "%" in libelle

        def ecart(mois: int) -> float:
            cible = serie.index[-1] - pd.DateOffset(months=mois)
            passe = serie[serie.index <= cible]
            if passe.empty:
                return np.nan
            ancien = float(passe.iloc[-1])
            if est_taux:
                return derniere - ancien
            return (derniere / ancien - 1) * 100 if ancien else np.nan

        lignes.append({
            "Indicateur": libelle,
            "Famille": famille,
            "Valeur": derniere,
            "1 mois": ecart(1),
            "3 mois": ecart(3),
            "12 mois": ecart(12),
            "Dernière donnée": serie.index[-1].date(),
            "Unité de variation": "points" if est_taux else "%",
        })
    return pd.DataFrame(lignes).set_index("Indicateur") if lignes else pd.DataFrame()


def variation_annuelle(serie: pd.Series, mois: int = 12) -> pd.Series:
    """
    Glissement annuel d'un indice.

    Indispensable pour les prix : l'indice CPI brut ne dit rien, seule sa
    variation sur douze mois constitue l'inflation.
    """
    if serie.empty:
        return serie
    return (serie / serie.shift(mois) - 1).dropna() * 100


def courbe_des_taux(series: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Structure par termes des taux americains.

    Une courbe inversee — le court terme au-dessus du long terme — a precede
    chacune des recessions americaines depuis 1955, avec un delai de six a
    dix-huit mois et un seul faux signal. C'est le meilleur predicteur macro
    connu, et aussi l'un des plus lents.
    """
    points = []
    for code, maturite in COURBE.items():
        serie = series.get(code)
        if serie is None or serie.empty:
            continue
        points.append({"Maturité (années)": maturite,
                       "Taux (%)": float(serie.iloc[-1]),
                       "Échéance": code.replace("DGS", "")})
    return pd.DataFrame(points).sort_values("Maturité (années)")


def diagnostic_courbe(series: dict[str, pd.Series]) -> dict:
    """Lecture synthetique de la pente de la courbe."""
    ecart = series.get("T10Y2Y")
    if ecart is None or ecart.empty:
        return {}
    valeur = float(ecart.iloc[-1])
    if valeur < -0.2:
        etat, lecture = "Nettement inversée", "Signal de récession historiquement fiable"
    elif valeur < 0:
        etat, lecture = "Inversée", "Signal d'alerte, à confirmer"
    elif valeur < 0.5:
        etat, lecture = "Plate", "Cycle en phase de transition"
    else:
        etat, lecture = "Normale", "Configuration d'expansion"
    return {"écart 10-2 ans": valeur, "état": etat, "lecture": lecture,
            "date": ecart.index[-1].date()}


# ==========================================================================
# Calendrier macroeconomique reconstitue
# ==========================================================================

def _nieme_jour_semaine(annee: int, mois: int, jour_semaine: int,
                        rang: int) -> pd.Timestamp:
    """Nieme occurrence d'un jour de la semaine dans un mois. Lundi = 0."""
    debut = pd.Timestamp(annee, mois, 1)
    decalage = (jour_semaine - debut.dayofweek) % 7
    return debut + pd.Timedelta(days=decalage + 7 * (rang - 1))


def _jour_ouvre(annee: int, mois: int, rang: int) -> pd.Timestamp:
    """Nieme jour ouvre du mois."""
    jours = pd.bdate_range(pd.Timestamp(annee, mois, 1),
                           pd.Timestamp(annee, mois, 1) + pd.offsets.MonthEnd(1))
    return jours[min(rang - 1, len(jours) - 1)]


# Regles de publication. Le champ "regle" decrit comment situer la date.
REGLES = [
    ("Emploi américain (NFP)", "États-Unis", "vendredi", 5, 1, "14h30",
     "Le chiffre le plus suivi du mois. Emploi, salaires et taux de chômage."),
    ("ISM manufacturier", "États-Unis", "ouvre", None, 1, "16h00",
     "Enquête auprès des directeurs d'achat. Au-dessus de 50, l'activité progresse."),
    ("ISM services", "États-Unis", "ouvre", None, 3, "16h00",
     "Même logique, sur les services — soit l'essentiel de l'économie américaine."),
    ("Inflation américaine (CPI)", "États-Unis", "jour", 12, None, "14h30",
     "Détermine la trajectoire des taux. Le chiffre sous-jacent compte plus que le brut."),
    ("Prix à la production (PPI)", "États-Unis", "jour", 14, None, "14h30",
     "Précède souvent le CPI dans les retournements d'inflation."),
    ("Ventes au détail", "États-Unis", "jour", 16, None, "14h30",
     "Mesure la consommation, environ 70 % du PIB américain."),
    ("PCE sous-jacent", "États-Unis", "jour", 27, None, "14h30",
     "Mesure d'inflation privilégiée par la Fed, plus que le CPI."),
    ("Inflation zone euro", "Zone euro", "fin", None, None, "11h00",
     "Estimation rapide publiée en fin de mois."),
    ("PMI zone euro", "Zone euro", "jour", 23, None, "10h00",
     "Équivalent européen de l'ISM, publié en version flash."),
]

# Reunions de banques centrales : environ huit par an, a intervalles reguliers.
MOIS_FOMC = [1, 3, 5, 6, 7, 9, 11, 12]
MOIS_BCE = [1, 3, 4, 6, 7, 9, 10, 12]


def calendrier_macro(depuis: pd.Timestamp | None = None,
                     mois_a_venir: int = 3) -> pd.DataFrame:
    """
    Reconstitue le calendrier des prochaines publications.

    Les dates sont estimees a partir des schemas habituels et peuvent varier
    d'un jour. Verifier sur les sites officiels avant toute decision liee a
    une echeance precise.
    """
    depart = depuis or pd.Timestamp.now().normalize()
    evenements = []

    for decalage in range(mois_a_venir + 1):
        ref = (depart + pd.DateOffset(months=decalage)).replace(day=1)
        annee, mois = ref.year, ref.month

        for nom, zone, regle, param, rang, heure, note in REGLES:
            if regle == "vendredi":
                date = _nieme_jour_semaine(annee, mois, 4, rang)
            elif regle == "ouvre":
                date = _jour_ouvre(annee, mois, rang)
            elif regle == "jour":
                date = pd.Timestamp(annee, mois, min(param, 28))
                while date.dayofweek > 4:
                    date += pd.Timedelta(days=1)
            elif regle == "fin":
                date = pd.Timestamp(annee, mois, 1) + pd.offsets.BMonthEnd(0)
            else:
                continue
            evenements.append({"Date": date, "Heure": heure, "Zone": zone,
                               "Événement": nom, "Type": "Statistique",
                               "Commentaire": note})

        if mois in MOIS_FOMC:
            date = _nieme_jour_semaine(annee, mois, 2, 3)  # mercredi, 3e semaine
            evenements.append({
                "Date": date, "Heure": "20h00", "Zone": "États-Unis",
                "Événement": "Décision de la Fed (FOMC)", "Type": "Banque centrale",
                "Commentaire": "Décision de taux et conférence de presse. "
                               "Le ton compte souvent davantage que la décision."})

        if mois in MOIS_BCE:
            date = _nieme_jour_semaine(annee, mois, 3, 2)  # jeudi, 2e semaine
            evenements.append({
                "Date": date, "Heure": "14h15", "Zone": "Zone euro",
                "Événement": "Décision de la BCE", "Type": "Banque centrale",
                "Commentaire": "Décision de taux, suivie de la conférence de presse."})

        # Expiration trimestrielle des dérivés
        if mois in (3, 6, 9, 12):
            evenements.append({
                "Date": _nieme_jour_semaine(annee, mois, 4, 3),
                "Heure": "17h30", "Zone": "Mondial",
                "Événement": "Expiration trimestrielle des dérivés",
                "Type": "Marché",
                "Commentaire": "Volumes et volatilité élevés, mouvements souvent "
                               "techniques et sans portée durable."})

    df = pd.DataFrame(evenements)
    df = df[df["Date"] >= depart].sort_values(["Date", "Heure"])
    return df.reset_index(drop=True)


# ==========================================================================
# Calendrier des publications d'entreprises
# ==========================================================================

def calendrier_resultats(donnees: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Rassemble les prochaines dates de publication des valeurs suivies.

    `donnees` associe chaque ticker au DataFrame `earnings_dates` de yfinance.
    Seules les lignes sans BPA publie correspondent a des dates futures.
    """
    lignes = []
    maintenant = pd.Timestamp.now()

    for ticker, df in donnees.items():
        if df is None or getattr(df, "empty", True):
            continue
        futur = df[df["Reported EPS"].isna()] if "Reported EPS" in df.columns else df
        for date, ligne in futur.iterrows():
            try:
                horodatage = pd.Timestamp(date)
                comparaison = (horodatage.tz_localize(None)
                               if horodatage.tzinfo else horodatage)
                if comparaison < maintenant:
                    continue
                lignes.append({
                    "Date": comparaison.normalize(),
                    "Ticker": ticker,
                    "BPA attendu": float(ligne.get("EPS Estimate", np.nan)),
                    "Dans (jours)": int((comparaison - maintenant).days),
                })
            except Exception:
                continue

    if not lignes:
        return pd.DataFrame()
    return pd.DataFrame(lignes).sort_values("Date").reset_index(drop=True)


def regrouper_par_semaine(calendrier: pd.DataFrame) -> dict:
    """Regroupe les evenements par semaine, pour un affichage en agenda."""
    if calendrier.empty or "Date" not in calendrier.columns:
        return {}
    groupes = {}
    for date, sous in calendrier.groupby(pd.Grouper(key="Date", freq="W-MON")):
        if sous.empty:
            continue
        debut = date - pd.Timedelta(days=6)
        libelle = f"Semaine du {debut.strftime('%d/%m')}"
        groupes[libelle] = sous
    return groupes
