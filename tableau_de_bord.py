"""
FinexResearch — tableau de bord.

Un seul ecran, trois blocs, lisible en trente secondes.

  OU J'EN SUIS      la valeur, les positions, le risque
  CE QUI A CHANGE   les faits survenus depuis la derniere visite
  CE QUI ARRIVE     les publications des dix prochains jours

Le parti pris tient en une phrase : un outil consulte trente secondes par
jour vaut mieux qu'un outil complet consulte dix minutes par trimestre. La
valeur d'un tableau de bord tient a sa frequence d'usage, et la frequence
tient a la rapidite de lecture. Chaque onglet supplementaire ajoute une
decision — ou chercher ? — et chaque decision coute de l'attention.

Le bloc du milieu est vide la plupart du temps. C'est voulu : quand il ne
l'est pas, on sait immediatement pourquoi on est la.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
import acces as ac
import analytics as an
import feuille as fe
import google_prive as gpv

ETAT = Path(__file__).parent / "derniere_visite.json"

# Seuils calibrés en écarts-types plutôt qu'en pourcentage fixe : 5 % est un
# événement sur Coca-Cola et une séance ordinaire sur une valeur volatile.
SIGMA_MOUVEMENT = 2.5
PLANCHER_MOUVEMENT = 3.0
PLAFOND_MOUVEMENT = 12.0
PROXIMITE_SEUIL = 3.0            # % — approche d'un seuil
CONCENTRATION = 25.0             # % — poids maximal d'une ligne


st.set_page_config(page_title="FinexResearch", page_icon="◪", layout="wide")
ac.porte("FinexResearch")


# ==========================================================================
# Mémoire de la dernière visite
# ==========================================================================

def derniere_visite() -> datetime | None:
    """Date de la derniere ouverture, pour dater « ce qui a change »."""
    if not ETAT.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(ETAT.read_text())["date"])
    except Exception:
        return None


def marquer_visite() -> None:
    try:
        ETAT.write_text(json.dumps({"date": datetime.now().isoformat()}))
    except Exception:
        pass


# ==========================================================================
# Chargement
# ==========================================================================

@st.cache_data(ttl=300, show_spinner=False)
def charger_univers(url: str) -> pd.DataFrame:
    """Positions et seuils depuis la feuille Google."""
    if gpv.disponible(st):
        brut = gpv.lire(st, url)
    else:
        brut = None
        for adresse in fe.urls_candidates(url):
            try:
                essai = pd.read_csv(adresse)
                if not essai.empty:
                    brut = essai
                    break
            except Exception:
                continue
        if brut is None:
            raise ValueError("Feuille inaccessible.")

    correspondance = {}
    for colonne in brut.columns:
        nom = fe._sans_accents(str(colonne))
        for cible, variantes in {
                "Ticker": ["ticker", "symbole", "valeur", "code"],
                "Quantité": ["quantite", "qte", "nombre", "titres"],
                "PRU": ["pru", "prix d'achat", "prix dachat", "prix de revient"],
                "Prix entrée": ["prix entree", "prix d'entree", "prix cible",
                                "cible"],
                "Prix sortie": ["prix sortie", "prix de sortie", "seuil vente",
                                "stop"]}.items():
            if cible in correspondance.values():
                continue
            if nom in variantes or any(v in nom for v in variantes):
                correspondance[colonne] = cible
                break
    brut = brut.rename(columns=correspondance)

    if "Ticker" not in brut.columns:
        raise ValueError(
            f"Colonne Ticker introuvable. La feuille contient : "
            f"{', '.join(map(str, brut.columns))}.")

    lignes = []
    for _, ligne in brut.iterrows():
        ticker = str(ligne.get("Ticker", "")).strip().upper()
        if not ticker or ticker in ("NAN", "TICKER"):
            continue
        lignes.append({
            "Ticker": ticker,
            "Quantité": fe._nombre(ligne.get("Quantité")),
            "PRU": fe._nombre(ligne.get("PRU")),
            "Prix entrée": fe._nombre(ligne.get("Prix entrée")),
            "Prix sortie": fe._nombre(ligne.get("Prix sortie"))})
    return pd.DataFrame(lignes).drop_duplicates(subset="Ticker")


@st.cache_data(ttl=900, show_spinner=False)
def charger_cours(tickers: tuple[str, ...], periode: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    brut = yf.download(list(tickers), period=periode, interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    if brut.empty:
        return pd.DataFrame()
    cours = (brut["Close"] if isinstance(brut.columns, pd.MultiIndex)
             else brut[["Close"]].rename(columns={"Close": tickers[0]}))
    return cours.dropna(how="all").ffill()


@st.cache_data(ttl=3600, show_spinner=False)
def taux(source: str, cible: str) -> float:
    if source == cible:
        return 1.0
    try:
        donnees = yf.download(f"{source}{cible}=X", period="5d",
                              progress=False, auto_adjust=True)
        if isinstance(donnees.columns, pd.MultiIndex):
            donnees.columns = donnees.columns.get_level_values(0)
        return float(donnees["Close"].dropna().iloc[-1])
    except Exception:
        return 1.0


@st.cache_data(ttl=3600, show_spinner=False)
def devises(tickers: tuple[str, ...]) -> dict:
    sortie = {}
    for t in tickers:
        try:
            sortie[t] = dict(yf.Ticker(t).fast_info).get("currency", "EUR")
        except Exception:
            sortie[t] = "EUR"
    return sortie


@st.cache_data(ttl=21600, show_spinner=False)
def publications(tickers: tuple[str, ...], jours: int = 10) -> list[dict]:
    """Prochaines publications de resultats."""
    sortie = []
    aujourdhui = pd.Timestamp.now().normalize()
    for t in tickers:
        try:
            dates = yf.Ticker(t).earnings_dates
            if dates is None or dates.empty:
                continue
            index = pd.to_datetime(dates.index)
            table = dates.copy()
            table.index = (index.tz_localize(None) if index.tz is not None
                           else index)
            futures = table[table.index >= aujourdhui]
            if "Reported EPS" in table.columns:
                futures = futures[futures["Reported EPS"].isna()]
            if futures.empty:
                continue
            date = futures.sort_index().index[0]
            delai = int((date - aujourdhui).days)
            if delai <= jours:
                bpa = futures.sort_index().iloc[0].get("EPS Estimate")
                sortie.append({"ticker": t, "date": date, "jours": delai,
                               "bpa": float(bpa) if pd.notna(bpa) else None})
        except Exception:
            continue
    return sorted(sortie, key=lambda p: p["jours"])


# ==========================================================================
# Détection des faits
# ==========================================================================

def seuil_titre(prix: pd.Series) -> float:
    """Seuil de mouvement propre au titre, en pourcentage."""
    r = prix.pct_change().dropna().tail(120)
    if len(r) < 40:
        return PLANCHER_MOUVEMENT * 2
    ecart = float(r.std(ddof=1)) * 100
    return (PLANCHER_MOUVEMENT if ecart <= 0
            else float(np.clip(ecart * SIGMA_MOUVEMENT,
                               PLANCHER_MOUVEMENT, PLAFOND_MOUVEMENT)))


def faits(univers: pd.DataFrame, cours: pd.DataFrame,
          depuis: datetime | None, poids: pd.Series) -> list[dict]:
    """
    Ce qui s'est produit depuis la derniere visite.

    Seuls les faits datables entrent ici : un franchissement, un mouvement
    anormal, un ecart de concentration. Pas d'indicateur, pas d'opinion.
    """
    evenements = []
    if cours.empty:
        return evenements

    seances = 1
    if depuis is not None:
        passees = cours.index[cours.index > pd.Timestamp(depuis)]
        seances = max(len(passees), 1)

    for _, ligne in univers.iterrows():
        t = ligne["Ticker"]
        if t not in cours.columns:
            continue
        prix = cours[t].dropna()
        if len(prix) < seances + 2:
            continue

        actuel = float(prix.iloc[-1])
        reference = float(prix.iloc[-seances - 1])
        variation = (actuel / reference - 1) * 100
        entree, sortie = ligne["Prix entrée"], ligne["Prix sortie"]

        if np.isfinite(sortie) and actuel <= sortie < reference:
            evenements.append({
                "rang": 0, "emoji": "🔴", "ticker": t,
                "titre": "Seuil de vente franchi",
                "detail": f"{actuel:.2f} contre un seuil à {sortie:.2f}"})
        elif np.isfinite(entree) and actuel <= entree < reference:
            evenements.append({
                "rang": 0, "emoji": "🎯", "ticker": t,
                "titre": "Prix d'entrée atteint",
                "detail": f"{actuel:.2f} contre un objectif à {entree:.2f}"})
        else:
            limite = seuil_titre(prix)
            if abs(variation) >= limite:
                evenements.append({
                    "rang": 1, "emoji": "⚠️", "ticker": t,
                    "titre": f"Mouvement de {variation:+.1f} %",
                    "detail": f"{actuel:.2f} — au-delà du seuil de ±{limite:.1f} % "
                              f"propre à ce titre"})
            elif np.isfinite(entree) and 0 < (actuel / entree - 1) * 100 <= PROXIMITE_SEUIL:
                evenements.append({
                    "rang": 2, "emoji": "📉", "ticker": t,
                    "titre": "Approche du prix d'entrée",
                    "detail": f"{actuel:.2f}, soit "
                              f"{(actuel / entree - 1) * 100:+.1f} % de l'objectif"})

    if not poids.empty:
        lourde = poids.idxmax()
        part = float(poids.max()) * 100
        if part > CONCENTRATION:
            evenements.append({
                "rang": 2, "emoji": "⚠️", "ticker": lourde,
                "titre": f"Concentration : {part:.0f} % du portefeuille",
                "detail": f"Une baisse de 30 % sur cette ligne coûterait "
                          f"{part * 0.3:.0f} % du portefeuille"})

    return sorted(evenements, key=lambda e: e["rang"])


# ==========================================================================
# Écran
# ==========================================================================

st.markdown(
    '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px">'
    '<svg width="30" height="30" viewBox="0 0 34 34" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" style="vertical-align:-5px">'
    '<rect x="1.5" y="1.5" width="31" height="31" stroke="#8A6A21" '
    'stroke-width="1.6"/><rect x="1.5" y="17" width="15.5" height="15.5" '
    'fill="#8A6A21"/></svg>'
    '<span style="font-size:30px;font-weight:600;letter-spacing:-0.02em">'
    'Finex<span style="color:#8A6A21">Research</span></span></div>',
    unsafe_allow_html=True)

url = ""
try:
    url = str(st.secrets.get("url_feuille", "")).strip()
except Exception:
    pass

haut = st.columns([3, 1, 1])
if not url:
    url = haut[0].text_input("Adresse de la feuille",
                             label_visibility="collapsed",
                             placeholder="Adresse de ta feuille Google…")
else:
    haut[0].caption("Positions et seuils lus depuis ta feuille Google."
                    + ("  🔒" if gpv.disponible(st) else ""))
devise_base = haut[1].selectbox("Devise", ["EUR", "USD", "CHF", "GBP"],
                                label_visibility="collapsed")
if haut[2].button("Actualiser", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if not url:
    st.info("Renseigne l'adresse de ta feuille pour commencer.")
    st.stop()

try:
    univers = charger_univers(url)
except Exception as erreur:
    st.error(str(erreur))
    st.stop()

if univers.empty:
    st.warning("Aucune ligne exploitable dans la feuille.")
    st.stop()

tickers = tuple(univers["Ticker"])
with st.spinner("Chargement des cours…"):
    cours = charger_cours(tickers)

if cours.empty:
    st.error("Aucun cours disponible. Vérifie tes tickers sur finance.yahoo.com.")
    st.stop()

# --- Valorisation
monnaies = devises(tickers)
lignes = []
for _, ligne in univers.iterrows():
    t = ligne["Ticker"]
    if t not in cours.columns or cours[t].dropna().empty:
        continue
    serie = cours[t].dropna()
    cours_actuel = float(serie.iloc[-1])
    veille = float(serie.iloc[-2]) if len(serie) > 1 else cours_actuel
    change = taux(monnaies.get(t, devise_base), devise_base)
    quantite = ligne["Quantité"] if np.isfinite(ligne["Quantité"]) else 0.0

    lignes.append({
        "Ticker": t,
        "Quantité": quantite,
        "Cours": cours_actuel,
        "Jour (%)": (cours_actuel / veille - 1) * 100,
        "Valeur": cours_actuel * quantite * change,
        "PRU": ligne["PRU"],
        "Gain (%)": ((cours_actuel / ligne["PRU"] - 1) * 100
                     if np.isfinite(ligne["PRU"]) and ligne["PRU"] > 0 else np.nan),
        "Prix entrée": ligne["Prix entrée"],
        "Prix sortie": ligne["Prix sortie"]})

table = pd.DataFrame(lignes).set_index("Ticker")
detenues = table[table["Quantité"] > 0]
total = float(detenues["Valeur"].sum())
poids = (detenues["Valeur"] / total if total > 0
         else pd.Series(dtype=float))

# --- Risque
rdt = cours[[t for t in detenues.index if t in cours.columns]].pct_change().dropna()
part_risque = pd.Series(dtype=float)
if len(rdt.columns) > 1 and len(rdt) > 60:
    decompo = an.decomposition_risque(poids.reindex(rdt.columns).fillna(0),
                                      an.matrice_covariance(rdt))
    part_risque = decompo["Part du risque (%)"]

table["Poids (%)"] = (poids * 100).reindex(table.index)
table["Risque (%)"] = part_risque.reindex(table.index)


# ==========================================================================
# BLOC 1 — Où j'en suis
# ==========================================================================

variation_jour = float((detenues["Valeur"] * detenues["Jour (%)"]).sum()
                       / total) if total > 0 else 0.0
investi = float((detenues["PRU"] * detenues["Quantité"]
                 * [taux(monnaies.get(t, devise_base), devise_base)
                    for t in detenues.index]).sum())

m = st.columns(4)
m[0].metric("Valeur", f"{total:,.0f} {devise_base}".replace(",", " "),
            f"{variation_jour:+.2f} % aujourd'hui")
m[1].metric("Plus-value latente",
            f"{total - investi:+,.0f} {devise_base}".replace(",", " ")
            if investi > 0 else "—",
            f"{(total / investi - 1) * 100:+.1f} %" if investi > 0 else None)
m[2].metric("Lignes", f"{len(detenues)}")
if len(rdt) > 60:
    m[3].metric("Volatilité",
                f"{an.volatilite(an.rendements_portefeuille(rdt, poids.reindex(rdt.columns).fillna(0))) * 100:.1f} %")

colonnes = ["Quantité", "Cours", "Jour (%)", "Valeur", "Gain (%)",
            "Poids (%)", "Risque (%)"]
st.dataframe(
    detenues[[c for c in colonnes if c in detenues.columns]]
    .sort_values("Valeur", ascending=False).round(2),
    use_container_width=True,
    column_config={
        "Jour (%)": st.column_config.NumberColumn(format="%+.2f %%"),
        "Gain (%)": st.column_config.NumberColumn(format="%+.1f %%"),
        "Valeur": st.column_config.NumberColumn(format="%.0f"),
        "Poids (%)": st.column_config.ProgressColumn(
            format="%.1f %%", min_value=0,
            max_value=float(table["Poids (%)"].max() or 100)),
    })


# ==========================================================================
# BLOC 2 — Ce qui a changé
# ==========================================================================

st.divider()
visite = derniere_visite()
entete = st.columns([3, 1])
entete[0].subheader("Ce qui a changé")
if visite:
    entete[1].caption(f"depuis le {visite.strftime('%d/%m à %H:%M')}")

evenements = faits(univers, cours, visite, poids)

if not evenements:
    st.success("Rien à signaler. Aucun seuil franchi, aucun mouvement "
               "inhabituel.", icon="✓")
else:
    for e in evenements:
        with st.container(border=True):
            gauche, droite = st.columns([1, 6])
            gauche.markdown(f"### {e['emoji']}")
            gauche.caption(e["ticker"])
            droite.markdown(f"**{e['titre']}**")
            droite.caption(e["detail"])


# ==========================================================================
# BLOC 3 — Ce qui arrive
# ==========================================================================

st.divider()
st.subheader("Ce qui arrive")

with st.spinner("Calendrier…"):
    prochaines = publications(tuple(detenues.index))

if not prochaines:
    st.caption("Aucune publication annoncée dans les dix prochains jours.")
else:
    for p in prochaines:
        quand = ("aujourd'hui" if p["jours"] == 0
                 else "demain" if p["jours"] == 1
                 else f"dans {p['jours']} jours")
        consensus = (f" · consensus {p['bpa']:.2f} par action"
                     if p["bpa"] is not None else "")
        st.markdown(f"📅 **{p['ticker']}** — résultats {quand}, "
                    f"le {p['date'].strftime('%d/%m')}{consensus}")

marquer_visite()

st.divider()
st.caption(
    f"Cours au {cours.index[-1].strftime('%d/%m/%Y')}, source Yahoo Finance. "
    "Les seuils proviennent des colonnes « Prix entrée » et « Prix sortie » "
    "de ta feuille : sans eux, le bloc du milieu reste vide."
)
