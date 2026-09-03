"""
Analyse de portefeuille et de valeurs.

Lancer avec :  streamlit run app.py
Le portefeuille est sauvegarde dans portefeuille.json a cote du script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import acces as ac
import analyse as ia
import analytics as an
import assistant as asst
import feuille as fe
import fondamentaux as fo
import indicateurs as ind
import journal as jr
import macro as mc
import optimisation as opt
import portefeuille as pf
import previsions as pv
import qualite as ql
import these as th
import univers as univ

st.set_page_config(page_title="FinexResearch", page_icon="◪", layout="wide")

# Porte d'entrée : rien ne s'affiche tant que l'identification n'a pas réussi.
# Désactivable en supprimant ces deux lignes si l'application reste privée.
ac.porte("FinexResearch")

FICHIER_PORTEFEUILLE = Path(__file__).parent / "portefeuille.json"

PORTEFEUILLE_EXEMPLE = [
    {"Ticker": "AAPL", "Quantité": 12.0, "Prix d'achat": 165.20},
    {"Ticker": "MSFT", "Quantité": 8.0, "Prix d'achat": 310.00},
    {"Ticker": "AIR.PA", "Quantité": 25.0, "Prix d'achat": 128.40},
    {"Ticker": "TTE.PA", "Quantité": 40.0, "Prix d'achat": 55.10},
]


# --------------------------------------------------------------------------
# Donnees
# --------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def charger_cours(tickers: tuple[str, ...], periode: str,
                  intervalle: str) -> pd.DataFrame:
    """Cours de cloture ajustes, une colonne par ticker."""
    if not tickers:
        return pd.DataFrame()
    brut = yf.download(list(tickers), period=periode, interval=intervalle,
                       auto_adjust=True, progress=False, group_by="column")
    if brut.empty:
        return pd.DataFrame()
    if isinstance(brut.columns, pd.MultiIndex):
        cours = brut["Close"]
    else:
        cours = brut[["Close"]].rename(columns={"Close": tickers[0]})
    return cours.dropna(how="all").ffill()


@st.cache_data(ttl=900, show_spinner=False)
def charger_ohlcv(ticker: str, periode: str, intervalle: str) -> pd.DataFrame:
    df = yf.download(ticker, period=periode, interval=intervalle,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def devise_du_ticker(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).fast_info
        return (info.get("currency") or "USD").upper()
    except Exception:
        return "USD"


@st.cache_data(ttl=3600, show_spinner=False)
def taux_de_change(source: str, cible: str) -> float:
    """Taux de conversion 1 unite de `source` vers `cible`."""
    if source == cible:
        return 1.0
    try:
        d = yf.download(f"{source}{cible}=X", period="5d",
                        progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return float(d["Close"].dropna().iloc[-1])
    except Exception:
        return np.nan


def charger_portefeuille() -> pd.DataFrame:
    if FICHIER_PORTEFEUILLE.exists():
        try:
            return pd.DataFrame(json.loads(FICHIER_PORTEFEUILLE.read_text()))
        except Exception:
            pass
    return pd.DataFrame(PORTEFEUILLE_EXEMPLE)


def enregistrer_portefeuille(df: pd.DataFrame) -> None:
    FICHIER_PORTEFEUILLE.write_text(
        json.dumps(df.to_dict("records"), indent=2, ensure_ascii=False)
    )


def sans_doublons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes homonymes avant affichage.

    Streamlit passe par Arrow, qui refuse les noms de colonnes dupliqués.
    Les sources externes en produisent parfois sans prévenir.
    """
    if isinstance(df, pd.DataFrame) and df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()]
    return df


def fmt(valeur, decimales: int = 2, unite: str = "") -> str:
    """Formate un nombre, ou renvoie un tiret si la donnée est absente."""
    if valeur is None:
        return "—"
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:,.{decimales}f}".replace(",", " ") + unite


# --------------------------------------------------------------------------
# Panneau lateral
# --------------------------------------------------------------------------

ac.bouton_deconnexion()
st.sidebar.header("Paramètres")

indices = univ.INDICES
nom_indice = st.sidebar.selectbox(
    "Indice de référence", list(indices),
    index=list(indices).index("MSCI World (ETF IWDA)"))
benchmark = indices[nom_indice]

periode = st.sidebar.selectbox(
    "Période d'analyse", ["6mo", "1y", "2y", "3y", "5y", "10y", "max"], index=2
)
intervalle = st.sidebar.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
devise_base = st.sidebar.selectbox("Devise de référence", ["EUR", "USD", "GBP", "CHF"])

taux_sans_risque = st.sidebar.number_input(
    "Taux sans risque annuel (%)", value=2.5, step=0.25, format="%.2f",
    help="Sert de référence aux ratios de Sharpe, Sortino et au calcul de l'alpha.",
) / 100

seuil_var = st.sidebar.select_slider(
    "Seuil de VaR", options=[0.01, 0.05, 0.10], value=0.05,
    format_func=lambda v: f"{int((1 - v) * 100)} %",
)

ia.reglages_barre_laterale()

st.sidebar.divider()
st.sidebar.subheader("Source du portefeuille")
source = st.sidebar.radio(
    "Où sont tes positions ?", ["Saisie dans l'app", "Google Sheets"],
    label_visibility="collapsed",
    help="Google Sheets te permet de modifier ton portefeuille depuis ton "
         "téléphone. L'app le relit à chaque rechargement.",
)

url_feuille = ""
if source == "Google Sheets":
    url_feuille = st.sidebar.text_input(
        "Adresse de la feuille",
        value=st.secrets.get("url_feuille", "") if hasattr(st, "secrets") else "",
        help="Colle l'adresse de ta feuille Google. Elle doit être publiée "
             "sur le web (Fichier → Partager → Publier sur le web).",
    )
    if st.sidebar.button("Recharger la feuille", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Tickers au format Yahoo Finance : AIR.PA (Airbus), MSFT (Microsoft), "
    "IWDA.AS (ETF World)."
)


# --------------------------------------------------------------------------
# Saisie du portefeuille
# --------------------------------------------------------------------------

# Monogramme : un simple carré partiellement rempli, tracé en SVG plutôt
# qu'en image — aucun fichier à héberger, net à toute résolution.
MONOGRAMME = """
<svg width="34" height="34" viewBox="0 0 34 34" fill="none"
     xmlns="http://www.w3.org/2000/svg" style="vertical-align:-7px">
  <rect x="1.5" y="1.5" width="31" height="31" stroke="#8A6A21"
        stroke-width="1.6"/>
  <rect x="1.5" y="17" width="15.5" height="15.5" fill="#8A6A21"/>
</svg>
"""

st.markdown(
    f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
    f'{MONOGRAMME}'
    f'<span style="font-size:34px;font-weight:600;letter-spacing:-0.02em">'
    f'Finex<span style="color:#8A6A21">Research</span></span></div>',
    unsafe_allow_html=True)
st.caption("Poste de travail d'analyse de portefeuille")

if "portefeuille" not in st.session_state:
    st.session_state.portefeuille = charger_portefeuille()

onglets = st.tabs([
    "Portefeuille", "Risque et bêta", "Corrélations", "Optimisation",
    "Simulation et stress", "Backtest", "Analyse d'une valeur", "Données",
    "Screener", "Indicateurs", "Fondamentaux", "Résultats et consensus",
    "Macro et calendrier", "Qualité des données", "Assistant",
    "Valeurs surveillées", "Journal", "Thèses",
])


@st.cache_data(ttl=300, show_spinner=False)
def lire_journal(url: str) -> pd.DataFrame:
    """Journal de transactions depuis Google Sheets. Cache 5 min."""
    return jr.lire(url)


@st.cache_data(ttl=300, show_spinner=False)
def lire_theses(url: str) -> pd.DataFrame:
    """Thèses d'investissement depuis Google Sheets. Cache 5 min."""
    return th.lire(url)


@st.cache_data(ttl=300, show_spinner=False)
def lire_liste_surveillance(url: str) -> list[str]:
    """Liste de surveillance depuis Google Sheets. Cache 5 min."""
    return fe.lire_liste(url)


@st.cache_data(ttl=300, show_spinner=False)
def lire_feuille(url: str) -> pd.DataFrame:
    """Portefeuille depuis Google Sheets. Cache 5 min."""
    return fe.lire(url)


with onglets[0]:
    if source == "Google Sheets":
        if not url_feuille:
            st.info(
                "Colle l'adresse de ta feuille Google dans la barre latérale. "
                "Elle doit contenir trois colonnes en première ligne : "
                "**Ticker**, **Quantité** et **Prix d'achat**."
            )
            st.markdown("**Modèle de feuille**")
            st.dataframe(fe.MODELE, use_container_width=True, hide_index=True)
            st.download_button(
                "Télécharger le modèle (CSV)",
                fe.MODELE.to_csv(index=False).encode("utf-8"),
                "modele_portefeuille.csv", "text/csv",
            )
            st.stop()

        try:
            with st.spinner("Lecture de la feuille…"):
                depuis_feuille = lire_feuille(url_feuille)
        except ValueError as erreur:
            st.error(str(erreur))
            st.stop()

        st.success(f"{len(depuis_feuille)} ligne(s) lue(s) depuis Google Sheets.")
        for alerte in fe.diagnostic(depuis_feuille):
            st.warning(alerte)

        st.caption(
            "Lecture seule. Pour modifier ton portefeuille, ouvre la feuille "
            "dans Google Sheets — depuis ton téléphone si tu veux — puis clique "
            "sur « Recharger la feuille » dans la barre latérale."
        )
        st.dataframe(depuis_feuille, use_container_width=True, hide_index=True)
        st.link_button("Ouvrir la feuille", url_feuille)
        edite = depuis_feuille
        st.session_state.portefeuille = edite

    else:
        st.caption(
            "Modifie directement le tableau. Le bouton du bas ajoute une ligne, "
            "la corbeille en supprime une."
        )
        edite = st.data_editor(
            st.session_state.portefeuille,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Quantité": st.column_config.NumberColumn(
                    "Quantité", min_value=0.0, step=1.0, format="%.4f"),
                "Prix d'achat": st.column_config.NumberColumn(
                    "Prix d'achat unitaire", min_value=0.0, step=0.01,
                    format="%.2f",
                    help="Dans la devise de cotation de la valeur.",
                ),
            },
            key="editeur",
        )
        st.session_state.portefeuille = edite

        col_a, col_b = st.columns([1, 5])
        if col_a.button("Enregistrer", use_container_width=True):
            enregistrer_portefeuille(edite)
            st.success(f"Portefeuille écrit dans {FICHIER_PORTEFEUILLE.name}")
        col_b.caption(
            "Sur l'hébergement en ligne, cet enregistrement est temporaire et "
            "sera effacé au prochain redéploiement. Passe par Google Sheets "
            "dans la barre latérale pour une sauvegarde durable."
        )

    lignes = edite.dropna(subset=["Ticker"]).copy()
    lignes["Ticker"] = lignes["Ticker"].str.strip().str.upper()
    lignes = lignes[lignes["Ticker"] != ""]
    lignes = lignes[lignes["Quantité"].fillna(0) > 0]

    if lignes.empty:
        st.info("Ajoute au moins une ligne pour lancer les calculs.")
        st.stop()

    tickers = tuple(dict.fromkeys(lignes["Ticker"]))
    cours = charger_cours(tickers + (benchmark,), periode, intervalle)

    absents = [t for t in tickers if t not in cours.columns
               or cours[t].dropna().empty]
    if absents:
        st.warning("Cours introuvables pour : " + ", ".join(absents))
        lignes = lignes[~lignes["Ticker"].isin(absents)]
        tickers = tuple(t for t in tickers if t not in absents)

    if lignes.empty or benchmark not in cours.columns:
        st.error("Pas assez de données pour continuer.")
        st.stop()

    # Valorisation, avec conversion dans la devise de reference
    valorisation = []
    for _, ligne in lignes.iterrows():
        t = ligne["Ticker"]
        dernier = float(cours[t].dropna().iloc[-1])
        dev = devise_du_ticker(t)
        fx = taux_de_change(dev, devise_base)
        fx = 1.0 if not np.isfinite(fx) else fx
        pru = float(ligne["Prix d'achat"] or 0)
        valorisation.append({
            "Ticker": t,
            "Quantité": float(ligne["Quantité"]),
            "Devise": dev,
            "Cours": dernier,
            "PRU": pru,
            "Valeur": dernier * float(ligne["Quantité"]) * fx,
            "Investi": pru * float(ligne["Quantité"]) * fx,
            "Taux de change": fx,
        })

    val = pd.DataFrame(valorisation).set_index("Ticker")
    val = val.groupby(level=0).agg({
        "Quantité": "sum", "Devise": "first", "Cours": "last",
        "PRU": "mean", "Valeur": "sum", "Investi": "sum",
        "Taux de change": "last",
    })
    val["Plus-value"] = val["Valeur"] - val["Investi"]
    val["Performance (%)"] = np.where(
        val["Investi"] > 0, val["Plus-value"] / val["Investi"] * 100, np.nan)
    val["Poids (%)"] = val["Valeur"] / val["Valeur"].sum() * 100

    poids = (val["Valeur"] / val["Valeur"].sum())
    total, investi = val["Valeur"].sum(), val["Investi"].sum()
    plus_value = total - investi

    # Series de rendements
    rdt = an.rendements(cours[list(val.index)])
    rdt_bench = an.rendements(cours[benchmark])
    freq = an.frequence_annuelle(rdt.index)
    rdt_ptf = an.rendements_portefeuille(rdt, poids)

    reg = an.regression_marche(rdt_ptf, rdt_bench, taux_sans_risque, freq)
    cov = an.matrice_covariance(rdt, freq)

    st.divider()
    c = st.columns(5)
    c[0].metric("Valeur totale", fmt(total, 0, f" {devise_base}"))
    c[1].metric("Plus-value latente", fmt(plus_value, 0, f" {devise_base}"),
                f"{val['Plus-value'].sum() / investi * 100:+.2f} %"
                if investi > 0 else None)
    c[2].metric("Bêta du portefeuille", fmt(reg["beta"]))
    c[3].metric("Volatilité annualisée",
                fmt(an.volatilite(rdt_ptf, freq) * 100, 2, " %"))
    c[4].metric("Sharpe", fmt(an.sharpe(rdt_ptf, taux_sans_risque, freq)))

    gauche, droite = st.columns([3, 2])
    with gauche:
        st.subheader("Lignes")
        affichage_lignes = val[[
            "Quantité", "Devise", "Cours", "PRU", "Investi", "Valeur",
            "Plus-value", "Performance (%)", "Poids (%)",
        ]].rename(columns={
            "Cours": "Cours (devise cotation)",
            "PRU": "PRU (devise cotation)",
            "Investi": f"Investi ({devise_base})",
            "Valeur": f"Valeur ({devise_base})",
            "Plus-value": f"Plus-value ({devise_base})",
        })
        st.dataframe(affichage_lignes.round(2), use_container_width=True)
        ia.bloc("Lignes du portefeuille", affichage_lignes,
                f"Portefeuille de {len(val)} lignes, {total:,.0f} {devise_base} "
                f"au total. Indice de référence : {nom_indice}.".replace(",", " "),
                "lignes_ptf")

        devises_etrangeres = val[val["Devise"] != devise_base]
        if not devises_etrangeres.empty:
            st.caption(
                f"Cours et PRU sont exprimés dans la devise de cotation de "
                f"chaque valeur ; les montants sont convertis en {devise_base}. "
                "Un calcul de tête sur les deux premières colonnes ne "
                "retrouvera donc pas la plus-value affichée. Taux appliqués : "
                + ", ".join(
                    f"1 {d} = {t:.4f} {devise_base}"
                    for d, t in devises_etrangeres.groupby("Devise")
                    ["Taux de change"].last().items())
                + "."
            )
            st.caption(
                "Conséquence à garder en tête : sur une valeur en dollars, une "
                "partie de ta performance vient du change et non de l'action "
                "elle-même."
            )
    with droite:
        st.subheader("Répartition")
        fig = px.pie(val.reset_index(), names="Ticker", values="Valeur", hole=0.55)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                          showlegend=True)
        fig.update_traces(textinfo="percent")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portefeuille contre indice")
    base = pd.DataFrame({
        "Portefeuille": (1 + rdt_ptf).cumprod() * 100,
        nom_indice: (1 + rdt_bench.reindex(rdt_ptf.index).fillna(0)).cumprod() * 100,
    })
    st.line_chart(base, height=340)

    st.subheader("Pertes depuis le plus haut")
    dd = pd.DataFrame({
        "Portefeuille": an.courbe_drawdown(rdt_ptf) * 100,
        nom_indice: an.courbe_drawdown(rdt_bench.reindex(rdt_ptf.index).fillna(0)) * 100,
    })
    st.area_chart(dd, height=260)

    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Diagnostic")
    st.caption(
        "Ce qui mérite ton attention, classé par gravité. Chaque constat "
        "porte une action possible — un diagnostic sans action n'est qu'un "
        "commentaire."
    )

    @st.cache_data(ttl=86400, show_spinner=False)
    def secteurs_des_lignes(tickers: tuple[str, ...]) -> dict:
        """Secteur de chaque valeur. Cache 24 h."""
        sortie = {}
        for t in tickers:
            try:
                sortie[t] = dict(yf.Ticker(t).info).get("sector") or "Inconnu"
            except Exception:
                sortie[t] = "Inconnu"
        return sortie

    decompo_risque = an.decomposition_risque(poids, cov)
    correlations = rdt.corr()
    lignes_eff = an.nombre_effectif_lignes(poids)
    conditionnel = an.beta_conditionnel(rdt_ptf, rdt_bench)

    with st.spinner("Analyse en cours…"):
        secteurs = secteurs_des_lignes(tuple(val.index))

    constats = pf.diagnostiquer(
        val, decompo_risque, correlations, lignes_eff,
        beta=reg.get("beta"), beta_baissier=conditionnel.get("beta_baisse"),
        secteurs=secteurs)

    if not constats:
        st.success("Aucun déséquilibre notable détecté.")
    else:
        couleurs = {"élevée": "error", "moyenne": "warning", "faible": "info"}
        for constat in constats:
            with st.container(border=True):
                gauche, droite = st.columns([1, 5])
                gauche.markdown(f"**{constat['sujet']}**")
                gauche.caption(constat["gravite"].capitalize())
                droite.markdown(f"**{constat['constat']}**")
                droite.caption(constat["portee"])
                droite.markdown(f"→ {constat['action']}")

    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Performance : le titre ou le change ?")
    st.caption(
        "Une ligne américaine qui gagne 20 % en dollars pendant que l'euro "
        "s'apprécie de 8 % ne rapporte que 11 %. Confondre les deux fait "
        "prendre un pari de change pour une réussite de sélection."
    )

    decompo_perf = pf.decomposer_performance(val)
    colonnes_perf = ["Devise", "PRU", "Cours", "Perf titre (%)"]
    st.dataframe(decompo_perf[colonnes_perf].round(2), use_container_width=True)
    st.caption(
        "La décomposition exacte demanderait le taux de change du jour de "
        "chaque achat. Ajoute une colonne « Date d'achat » à ta feuille et je "
        "pourrai la calculer — sans elle, la performance affichée mélange les "
        "deux effets."
    )

    exp = pf.exposition(val, secteurs)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Exposition par devise**")
        st.dataframe(pd.Series(exp.get("devises", {}), name="%").round(1),
                     use_container_width=True)
    with e2:
        st.markdown("**Exposition par secteur**")
        st.dataframe(pd.Series(exp.get("secteurs", {}), name="%").round(1),
                     use_container_width=True)

    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Que faire")

    onglet_plan, onglet_apport = st.tabs(["Rééquilibrer", "Investir un apport"])

    with onglet_plan:
        st.caption(
            "En France, un arbitrage sur compte-titres déclenche l'imposition "
            "des plus-values. Le coût est immédiat et certain, le gain espéré "
            "ne l'est pas : la comparaison décide."
        )

        methode = st.selectbox(
            "Allocation cible",
            ["Variance minimale", "Parité de risque", "HRP", "Équipondéré"],
            key="cible_reequilibrage")

        cov_robuste, _ = opt.covariance_retrecie(rdt, freq=freq)
        cibles = {
            "Variance minimale": lambda: opt.variance_minimale(cov_robuste),
            "Parité de risque": lambda: opt.parite_de_risque(cov_robuste),
            "HRP": lambda: opt.hrp(cov_robuste),
            "Équipondéré": lambda: pd.Series(1 / len(cov_robuste),
                                             index=cov_robuste.index),
        }
        try:
            poids_cible = cibles[methode]()
        except Exception as erreur:
            st.error(f"Allocation non calculable : {erreur}")
            poids_cible = pd.Series(dtype=float)

        if not poids_cible.empty:
            f1, f2 = st.columns(2)
            taux_fiscal = f1.slider("Taux d'imposition (%)", 0, 40, 30,
                                    help="Prélèvement forfaitaire unique par "
                                         "défaut. Mets 0 pour un PEA.") / 100
            frais_courtier = f2.slider("Frais par ordre (%)", 0.0, 2.0, 0.5,
                                       step=0.1) / 100

            plan = pf.plan_reequilibrage(val, poids_cible,
                                         taux_fiscal=taux_fiscal,
                                         frais=frais_courtier)

            vol_actuelle = an.volatilite_portefeuille(poids, cov_robuste) * 100
            vol_cible = an.volatilite_portefeuille(poids_cible, cov_robuste) * 100
            gain = max(vol_actuelle - vol_cible, 0) * 0.5   # approximation prudente

            synthese = pf.resume_plan(plan, gain_attendu_pct=gain,
                                      valeur_totale=float(total))

            c = st.columns(4)
            c[0].metric("Ordres", synthese.get("ordres", 0))
            c[1].metric("Rotation", fmt(synthese.get("rotation_pct"), 0, " %"))
            c[2].metric("Coût total",
                        fmt(synthese.get("cout_total"), 0, f" {devise_base}"),
                        fmt(synthese.get("cout_pct"), 2, " %"))
            c[3].metric("Volatilité", fmt(vol_cible, 1, " %"),
                        fmt(vol_cible - vol_actuelle, 1, " pt"))

            if synthese.get("verdict"):
                (st.warning if "ne pas rééquilibrer" in synthese["verdict"]
                 else st.info)(synthese["verdict"])

            st.dataframe(
                plan[["Poids actuel (%)", "Poids cible (%)", "Écart (pt)",
                      "Sens", "Montant", "Frais", "Impôt estimé",
                      "Coût total"]].round(2),
                use_container_width=True)
            st.caption(
                "Montants dans ta devise de référence. L'impôt est estimé au "
                "prorata de la plus-value latente cédée ; il sera nul sur un "
                "PEA ou en moins-value."
            )

    with onglet_apport:
        st.caption(
            "La méthode la plus efficace pour un particulier : atteindre la "
            "même cible sans rien vendre, donc sans déclencher un centime "
            "d'impôt. À privilégier tant que tu épargnes."
        )
        montant_apport = st.number_input(
            f"Montant à investir ({devise_base})", min_value=0.0,
            value=1000.0, step=100.0)

        if montant_apport > 0 and not poids_cible.empty:
            repartition = pf.apport_optimal(val, poids_cible, montant_apport)
            if repartition.empty:
                st.info("Le portefeuille est déjà proche de la cible : "
                        "répartis l'apport au prorata des poids visés.")
            else:
                st.dataframe(
                    repartition[["Valeur actuelle", "À acheter",
                                 "Part de l'apport (%)"]].round(0),
                    use_container_width=True)
                st.success(
                    f"Ces {montant_apport:,.0f} {devise_base} rapprochent le "
                    "portefeuille de sa cible sans aucune vente."
                    .replace(",", " "))


with onglets[1]:
    st.subheader("Décomposition du risque")
    st.caption(
        "La contribution marginale mesure l'effet sur la volatilité totale d'une "
        "hausse d'un point du poids. Les contributions au risque, elles, "
        "s'additionnent exactement pour donner la volatilité du portefeuille."
    )
    decompo = an.decomposition_risque(poids, cov)
    st.dataframe(
        decompo.sort_values("Part du risque (%)", ascending=False).round(2),
        use_container_width=True,
    )
    ia.bloc("Décomposition du risque", decompo,
            "La contribution au risque ne suit presque jamais les poids. "
            "L'écart entre les deux colonnes est l'information principale.",
            "decompo_risque")

    e = st.columns(4)
    e[0].metric("Volatilité du portefeuille",
                fmt(an.volatilite_portefeuille(poids, cov) * 100, 2, " %"))
    e[1].metric("Ratio de diversification",
                fmt(an.ratio_diversification(poids, cov)))
    e[2].metric("Lignes effectives", fmt(an.nombre_effectif_lignes(poids), 1),
                help="Inverse de l'indice de Herfindahl : mesure la concentration réelle.")
    e[3].metric("Risque spécifique", fmt(reg["risque_specifique"] * 100, 2, " %"),
                help="Part de la volatilité non expliquée par l'indice.")

    st.subheader("Écart de risque et de poids")
    barres = decompo[["Poids (%)", "Part du risque (%)"]].sort_values("Part du risque (%)")
    fig = go.Figure()
    for col, couleur in [("Poids (%)", "#7F77DD"), ("Part du risque (%)", "#D85A30")]:
        fig.add_bar(y=barres.index, x=barres[col], name=col,
                    orientation="h", marker_color=couleur)
    fig.update_layout(barmode="group", height=60 + 42 * len(barres),
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Mesures détaillées")

    colonnes = {}
    colonnes["Portefeuille"] = an.tableau_metriques(
        rdt_ptf, rdt_bench, taux_sans_risque, freq)
    colonnes[nom_indice] = an.tableau_metriques(
        rdt_bench, rdt_bench, taux_sans_risque, freq)
    for t in val.index:
        colonnes[t] = an.tableau_metriques(
            rdt[t].dropna(), rdt_bench, taux_sans_risque, freq)

    synthese = pd.DataFrame(colonnes)
    synthese.index = [
        {"beta_hausse": "Bêta en marché haussier",
         "beta_baisse": "Bêta en marché baissier",
         "capture_hausse": "Capture de hausse",
         "capture_baisse": "Capture de baisse"}.get(i, i)
        for i in synthese.index
    ]
    st.dataframe(synthese.round(3), use_container_width=True, height=640)
    ia.bloc("Mesures de risque", synthese,
            f"Portefeuille, {nom_indice} et chaque ligne, sur {periode}. "
            "Bêta, alpha, Sharpe, VaR et bêta conditionnel.",
            "mesures_risque")

    st.subheader(f"Perte extrême — VaR {int((1 - seuil_var) * 100)} %")
    v = st.columns(3)
    var_h = an.var_historique(rdt_ptf, seuil_var)
    cvar_h = an.cvar_historique(rdt_ptf, seuil_var)
    v[0].metric("VaR historique (période)", fmt(var_h * 100, 2, " %"),
                fmt(var_h * total, 0, f" {devise_base}"))
    v[1].metric("CVaR (perte moyenne au-delà)", fmt(cvar_h * 100, 2, " %"),
                fmt(cvar_h * total, 0, f" {devise_base}"))
    v[2].metric("VaR gaussienne",
                fmt(an.var_parametrique(rdt_ptf, seuil_var) * 100, 2, " %"),
                help="Un écart marqué avec la VaR historique signale des queues épaisses.")

    fig = px.histogram(rdt_ptf * 100, nbins=60, opacity=0.85)
    fig.add_vline(x=var_h * 100, line_dash="dash", line_color="#E24B4A",
                  annotation_text="VaR")
    fig.add_vline(x=cvar_h * 100, line_dash="dot", line_color="#993C1D",
                  annotation_text="CVaR")
    fig.update_layout(height=300, showlegend=False, bargap=0.02,
                      xaxis_title="Rendement par période (%)",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


with onglets[2]:
    st.subheader("Matrice de corrélation")
    st.caption(
        "Des corrélations élevées entre lignes signifient qu'un portefeuille "
        "de dix positions n'en constitue qu'une seule, déguisée."
    )
    correl = rdt.join(rdt_bench.rename(nom_indice)).corr()
    fig = px.imshow(correl, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(height=120 + 52 * len(correl), coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    ia.bloc("Corrélations", correl,
            "Corrélations entre les lignes et avec l'indice. Au-delà de 0,8, "
            "deux lignes comptent pour une seule position.",
            "correlations")

    st.subheader("Matrice de covariance annualisée")
    st.dataframe(cov.round(4), use_container_width=True)

    st.subheader("Corrélation glissante à l'indice")
    fenetre = st.slider("Fenêtre (périodes)", 20, 250, 60, step=10)
    glissante = pd.DataFrame({
        t: rdt[t].rolling(fenetre).corr(rdt_bench) for t in val.index
    }).dropna(how="all")
    if glissante.empty:
        st.info("Historique trop court pour cette fenêtre.")
    else:
        st.line_chart(glissante, height=320)


with onglets[3]:
    st.subheader("Construction de portefeuille")
    st.caption(
        "L'optimiseur amplifie les erreurs d'estimation : il concentre les poids "
        "exactement là où la covariance est mal mesurée. Le rétrécissement et les "
        "bornes de poids servent à contenir cet effet."
    )

    o = st.columns(4)
    estimateur = o[0].selectbox(
        "Estimateur de covariance", ["retreci", "ewma", "empirique"],
        format_func=lambda k: {"retreci": "Rétréci (Ledoit-Wolf)",
                               "ewma": "EWMA (mémoire courte)",
                               "empirique": "Empirique brut"}[k],
    )
    methode_mu = o[1].selectbox(
        "Rendements attendus", ["retreci", "egal", "ewma", "historique"],
        format_func=lambda k: {"retreci": "Moyenne rétrécie",
                               "egal": "Aucune prévision",
                               "ewma": "Pondérée récente",
                               "historique": "Moyenne historique"}[k],
    )
    poids_max = o[2].slider("Poids maximal par ligne (%)", 10, 100, 40, step=5) / 100
    poids_min = o[3].slider("Poids minimal par ligne (%)", 0, 20, 0, step=1) / 100
    bornes = (poids_min, max(poids_max, poids_min + 0.01))

    if estimateur == "ewma":
        cov_opt = opt.covariance_ewma(rdt, freq=freq)
        note_shrink = None
    elif estimateur == "empirique":
        cov_opt = opt.covariance_empirique(rdt, freq=freq)
        note_shrink = None
    else:
        cov_opt, note_shrink = opt.covariance_retrecie(rdt, freq=freq)

    if note_shrink is not None:
        st.caption(
            f"Intensité de rétrécissement retenue : {note_shrink:.1%} — "
            "plus elle est élevée, moins l'historique est fiable pour optimiser."
        )

    mu_opt = opt.rendements_attendus(rdt, methode_mu, freq)

    allocations = {"Portefeuille actuel": poids.reindex(cov_opt.index).fillna(0)}
    try:
        allocations["Équipondéré"] = pd.Series(1 / len(cov_opt), index=cov_opt.index)
        allocations["Variance minimale"] = opt.variance_minimale(cov_opt, bornes)
        allocations["Parité de risque"] = opt.parite_de_risque(cov_opt)
        allocations["HRP"] = opt.hrp(cov_opt)
        allocations["Diversification max"] = opt.diversification_maximale(cov_opt, bornes)
        allocations["Sharpe maximal"] = opt.sharpe_maximal(
            mu_opt, cov_opt, taux_sans_risque, bornes)
    except Exception as e:
        st.error(f"Optimisation impossible : {e}")

    resume = []
    for nom, w in allocations.items():
        vol = an.volatilite_portefeuille(w, cov_opt)
        rdt_att = float((w * mu_opt.reindex(w.index)).sum())
        realise = an.rendements_portefeuille(rdt, w)
        resume.append({
            "Allocation": nom,
            "Rendement attendu (%)": rdt_att * 100,
            "Volatilité (%)": vol * 100,
            "Sharpe attendu": (rdt_att - taux_sans_risque) / vol if vol else np.nan,
            "Sharpe réalisé": an.sharpe(realise, taux_sans_risque, freq),
            "Drawdown max réalisé (%)": an.drawdown_max(realise) * 100,
            "Lignes effectives": an.nombre_effectif_lignes(w),
            "Écart au portefeuille actuel (%)":
                float((w - allocations["Portefeuille actuel"]).abs().sum()) * 100 / 2,
        })
    tableau_allocations = pd.DataFrame(resume).set_index("Allocation").round(2)
    st.dataframe(tableau_allocations, use_container_width=True)
    ia.bloc("Comparaison des allocations", tableau_allocations,
            "Rendement attendu, volatilité et Sharpe de chaque méthode, face "
            "au portefeuille actuel. Le Sharpe attendu est très sensible aux "
            "erreurs d'estimation des rendements.",
            "allocations")

    st.subheader("Poids par allocation")
    tableau_poids = pd.DataFrame(allocations) * 100
    st.dataframe(tableau_poids.round(2), use_container_width=True)

    st.subheader("Frontière efficiente")
    with st.spinner("Calcul de la frontière…"):
        frontiere = opt.frontiere_efficiente(mu_opt, cov_opt, points=35, bornes=bornes)

    if frontiere.empty:
        st.info("Frontière non calculable avec ces contraintes.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontiere["Volatilité"] * 100, y=frontiere["Rendement"] * 100,
            mode="lines", name="Frontière", line=dict(color="#888780", width=1.5)))
        for nom, w in allocations.items():
            v = an.volatilite_portefeuille(w, cov_opt) * 100
            m_ = float((w * mu_opt.reindex(w.index)).sum()) * 100
            fig.add_trace(go.Scatter(
                x=[v], y=[m_], mode="markers+text", name=nom, text=[nom],
                textposition="top center", marker=dict(size=11)))
        for t in cov_opt.index:
            fig.add_trace(go.Scatter(
                x=[np.sqrt(cov_opt.loc[t, t]) * 100], y=[mu_opt[t] * 100],
                mode="markers", name=t, marker=dict(size=7, symbol="x"),
                showlegend=False, opacity=0.5))
        fig.update_layout(height=460, xaxis_title="Volatilité annualisée (%)",
                          yaxis_title="Rendement attendu (%)",
                          margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Écarts à combler")
    cible_nom = st.selectbox("Allocation cible",
                             [k for k in allocations if k != "Portefeuille actuel"])
    cible = allocations[cible_nom]
    ordres = pd.DataFrame({
        "Poids actuel (%)": allocations["Portefeuille actuel"] * 100,
        "Poids cible (%)": cible * 100,
    })
    ordres["Écart (%)"] = ordres["Poids cible (%)"] - ordres["Poids actuel (%)"]
    ordres[f"Montant ({devise_base})"] = ordres["Écart (%)"] / 100 * total
    ordres["Sens"] = np.where(ordres["Écart (%)"] > 0.5, "Renforcer",
                              np.where(ordres["Écart (%)"] < -0.5, "Alléger", "—"))
    st.dataframe(ordres.round(2).sort_values("Écart (%)"), use_container_width=True)
    st.caption(
        "Montants indicatifs hors frais et fiscalité, et sans tenir compte du "
        "fait qu'un arbitrage déclenche l'imposition des plus-values."
    )


with onglets[4]:
    st.subheader("Projection par bootstrap par blocs")
    st.caption(
        "Rééchantillonne l'historique par blocs de plusieurs séances consécutives, "
        "ce qui conserve l'enchaînement des périodes agitées. Ce n'est pas une "
        "prévision : c'est la dispersion des trajectoires compatibles avec le "
        "comportement passé du portefeuille."
    )

    s = st.columns(3)
    horizon = s[0].slider("Horizon (périodes)", 21, 1260, 252, step=21)
    taille_bloc = s[1].slider("Taille des blocs", 1, 40, 10)
    tirages = s[2].select_slider("Tirages", [1000, 2000, 5000, 10000], value=5000)

    final = opt.bootstrap_horizon(rdt_ptf, horizon, tirages, taille_bloc)

    if len(final) == 0:
        st.info("Historique trop court pour la simulation.")
    else:
        percentiles = np.percentile(final, [5, 25, 50, 75, 95])
        p = st.columns(5)
        for col, (label, v) in zip(p, zip(
                ["5e centile", "1er quartile", "Médiane", "3e quartile", "95e centile"],
                percentiles)):
            col.metric(label, fmt(total * v, 0, f" {devise_base}"),
                       f"{(v - 1) * 100:+.1f} %")

        q = st.columns(3)
        q[0].metric("Probabilité de perte", fmt(float((final < 1).mean()) * 100, 1, " %"))
        q[1].metric("Probabilité de perdre plus de 20 %",
                    fmt(float((final < 0.8).mean()) * 100, 1, " %"))
        q[2].metric("Perte au 5e centile",
                    fmt(total * (percentiles[0] - 1), 0, f" {devise_base}"))

        fig = px.histogram((final - 1) * 100, nbins=70, opacity=0.85)
        fig.add_vline(x=0, line_dash="dash", line_color="#5F5E5A")
        fig.add_vline(x=(percentiles[0] - 1) * 100, line_dash="dot",
                      line_color="#E24B4A", annotation_text="5e centile")
        fig.update_layout(height=320, showlegend=False, bargap=0.02,
                          xaxis_title=f"Performance sur {horizon} périodes (%)",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Pires épisodes réellement traversés")
    f = st.columns(2)
    fenetre_stress = f[0].slider("Longueur de la fenêtre (périodes)", 5, 120, 21)
    pires = opt.stress_historique(rdt_ptf, fenetre_stress, 6)
    if pires.empty:
        st.info("Historique insuffisant.")
    else:
        pires[f"Perte estimée ({devise_base})"] = (
            pires.iloc[:, 1] / 100 * total).round(0)
        st.dataframe(pires.round(2), use_container_width=True, hide_index=True)

    st.subheader("Chocs de marché propagés par le bêta")
    st.caption(
        f"Bêta du portefeuille : {fmt(reg['beta'])} face au {nom_indice}. "
        "L'estimation ignore le risque spécifique et suppose que le bêta reste "
        "stable pendant le choc — en pratique les corrélations montent, "
        "ce qui rend ces chiffres optimistes."
    )
    st.dataframe(
        opt.stress_facteur(reg["beta"], opt.CHOCS_TYPES, total).round(2),
        use_container_width=True, hide_index=True,
    )


with onglets[5]:
    st.subheader("Backtest walk-forward")
    st.caption(
        "À chaque rebalancement, les poids ne sont estimés que sur les données "
        "antérieures, puis appliqués à la période suivante. Les coûts sont "
        "prélevés sur le turnover réel. Une méthode qui brille ici sans être "
        "calibrée sur le futur a une chance de tenir."
    )

    b = st.columns(4)
    fenetre_bt = b[0].slider("Fenêtre d'estimation (périodes)", 60, 756, 252, step=21)
    pas_bt = b[1].slider("Rebalancement tous les (périodes)", 5, 126, 21, step=1)
    cout_bt = b[2].number_input("Coût par transaction (points de base)",
                                0.0, 100.0, 10.0, step=1.0)
    estim_bt = b[3].selectbox("Estimateur", ["retreci", "ewma", "empirique"],
                              key="estim_bt")

    choisies = st.multiselect(
        "Méthodes à comparer", list(opt.METHODES),
        default=["equipondere", "variance_min", "parite_risque", "hrp"],
        format_func=lambda k: opt.METHODES[k],
    )

    @st.cache_data(show_spinner=False)
    def lancer_backtest(donnees, methode, fenetre, pas, cout, rf, f, estim):
        return opt.backtest_walk_forward(donnees, methode, fenetre, pas,
                                         cout, rf, f, estim)

    if not choisies:
        st.info("Sélectionne au moins une méthode.")
    elif len(rdt) <= fenetre_bt + pas_bt:
        st.warning(
            "Historique trop court pour cette fenêtre. Allonge la période "
            "d'analyse dans la barre latérale ou réduis la fenêtre."
        )
    else:
        courbes, mesures, turnovers, poids_hist = {}, [], {}, {}
        with st.spinner("Backtest en cours…"):
            for m in choisies:
                res = lancer_backtest(rdt, m, fenetre_bt, pas_bt, cout_bt,
                                      taux_sans_risque, freq, estim_bt)
                r_bt = res["rendements"]
                if r_bt.empty:
                    continue
                courbes[opt.METHODES[m]] = (1 + r_bt).cumprod() * 100
                turnovers[opt.METHODES[m]] = res["turnover"]
                poids_hist[opt.METHODES[m]] = res["poids"]
                rot = (res["turnover"].mean() * (freq / pas_bt)
                       if len(res["turnover"]) else np.nan)
                mesures.append({
                    "Méthode": opt.METHODES[m],
                    "Rendement annualisé (%)": an.rendement_annualise(r_bt, freq) * 100,
                    "Volatilité (%)": an.volatilite(r_bt, freq) * 100,
                    "Sharpe": an.sharpe(r_bt, taux_sans_risque, freq),
                    "Sortino": an.sortino(r_bt, taux_sans_risque, freq),
                    "Drawdown max (%)": an.drawdown_max(r_bt) * 100,
                    "Calmar": an.calmar(r_bt, freq),
                    "VaR 95 % (%)": an.var_historique(r_bt, 0.05) * 100,
                    "Rotation annuelle (%)": rot * 100,
                })

        if not mesures:
            st.warning("Aucun backtest exploitable.")
        else:
            reference = (1 + rdt_bench.reindex(
                list(courbes.values())[0].index).fillna(0)).cumprod() * 100
            courbes[nom_indice] = reference
            st.line_chart(pd.DataFrame(courbes), height=380)

            tableau_backtest = (pd.DataFrame(mesures).set_index("Méthode")
                                .round(2).sort_values("Sharpe", ascending=False))
            st.dataframe(tableau_backtest, use_container_width=True)
            ia.bloc("Backtest walk-forward", tableau_backtest,
                    "Chaque méthode réestimée à chaque rebalancement sur les "
                    "seules données antérieures, coûts déduits. La rotation "
                    "annuelle est décisive : elle paiera frais et impôts.",
                    "backtest")
            st.caption(
                "La colonne de rotation annuelle est décisive : une méthode qui "
                "fait tourner 400 % du portefeuille par an paiera des frais et "
                "de l'impôt bien au-delà de ce que le backtest suppose."
            )

            detail = st.selectbox("Voir les poids dans le temps",
                                  [opt.METHODES[m] for m in choisies
                                   if opt.METHODES[m] in poids_hist])
            if detail and not poids_hist[detail].empty:
                st.area_chart(poids_hist[detail] * 100, height=300)


with onglets[6]:
    actif = st.selectbox("Valeur", list(val.index), key="valeur_analyse")
    df = charger_ohlcv(actif, periode, intervalle)

    if df.empty:
        st.error("Aucune donnée.")
    else:
        r_actif = an.rendements(df["Close"])
        reg_a = an.regression_marche(r_actif, rdt_bench, taux_sans_risque, freq)
        cond = an.beta_conditionnel(r_actif, rdt_bench)

        m = st.columns(6)
        m[0].metric("Bêta", fmt(reg_a["beta"]))
        m[1].metric("Alpha annualisé", fmt(reg_a["alpha"] * 100, 2, " %"))
        m[2].metric("R²", fmt(reg_a["r2"], 3))
        m[3].metric("Tracking error", fmt(reg_a["tracking_error"] * 100, 2, " %"))
        m[4].metric("Bêta en baisse", fmt(cond.get("beta_baisse")))
        m[5].metric("Capture de hausse", fmt(cond.get("capture_hausse")))

        for n in (20, 50, 200):
            df[f"MM{n}"] = df["Close"].rolling(n).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.04)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=actif), row=1, col=1)
        for n, couleur in ((20, "#7F77DD"), (50, "#EF9F27"), (200, "#378ADD")):
            fig.add_trace(go.Scatter(x=df.index, y=df[f"MM{n}"], name=f"MM{n}",
                                     line=dict(width=1.1, color=couleur)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r_actif.index,
                                 y=an.courbe_drawdown(r_actif) * 100,
                                 name="Drawdown (%)", fill="tozeroy",
                                 line=dict(width=1, color="#E24B4A")), row=2, col=1)
        fig.update_layout(height=620, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Régression sur l'indice")
        paire = pd.concat([r_actif, rdt_bench], axis=1, join="inner").dropna()
        paire.columns = ["actif", "marche"]
        fig = px.scatter(paire * 100, x="marche", y="actif", trendline="ols",
                         opacity=0.55, trendline_color_override="#D85A30")
        fig.update_layout(height=380, xaxis_title=f"{nom_indice} (%)",
                          yaxis_title=f"{actif} (%)",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Pente de la droite = bêta ({fmt(reg_a['beta'])}), "
            f"dispersion autour d'elle = risque spécifique "
            f"({fmt(reg_a['risque_specifique'] * 100, 2, ' %')})."
        )


with onglets[7]:
    st.subheader("Rendements mensuels du portefeuille")
    mensuel = (1 + rdt_ptf).resample("ME").prod() - 1
    tableau = pd.DataFrame({
        "Année": mensuel.index.year, "Mois": mensuel.index.month,
        "Rendement": mensuel.values * 100,
    }).pivot(index="Année", columns="Mois", values="Rendement")
    st.dataframe(tableau.round(2).style.background_gradient(
        cmap="RdYlGn", axis=None), use_container_width=True)

    st.subheader("Séries de cours")
    st.dataframe(cours.sort_index(ascending=False).round(3),
                 use_container_width=True, height=400)

    d = st.columns(3)
    d[0].download_button("Cours (CSV)", cours.to_csv().encode("utf-8"),
                         "cours.csv", "text/csv", use_container_width=True)
    d[1].download_button("Mesures (CSV)", synthese.to_csv().encode("utf-8"),
                         "metriques.csv", "text/csv", use_container_width=True)
    d[2].download_button("Portefeuille (CSV)", val.to_csv().encode("utf-8"),
                         "portefeuille.csv", "text/csv", use_container_width=True)


with onglets[8]:
    st.subheader("Screener")
    st.caption(
        "Balaie un univers entier et classe les valeurs selon tes critères. "
        "Le téléchargement est mis en cache une heure : le premier passage sur "
        "un univers large prend plusieurs minutes, les suivants sont immédiats."
    )

    s1, s2 = st.columns([2, 1])
    choix_univers = s1.multiselect(
        "Univers à balayer", list(univ.UNIVERS),
        default=["CAC 40 (France)"],
    )
    periode_scr = s2.selectbox("Historique", ["1y", "2y", "3y", "5y"], index=1,
                               key="periode_screener")

    liste = sorted({t for u_ in choix_univers for t in univ.UNIVERS[u_]})

    with st.expander("Personnaliser ma liste de surveillance"):
        st.caption(
            "Colle tes tickers séparés par des virgules ou des retours à la "
            "ligne. Cette liste remplace celle sélectionnée ci-dessus. Pour la "
            "rendre permanente, mets-la dans le secret Streamlit "
            "`tickers_surveillance`."
        )
        url_surveillance = ""
        depuis_secrets = ""
        try:
            url_surveillance = st.secrets.get("url_surveillance", "")
            depuis_secrets = st.secrets.get("tickers_surveillance", "")
        except Exception:
            pass

        url_saisie = st.text_input(
            "Adresse de ma feuille de surveillance",
            value=url_surveillance,
            help="Second onglet de ton classeur Google, publié au format CSV. "
                 "Tu le modifies depuis ton téléphone, l'app le relit ici.",
            key="url_surveillance")

        depuis_feuille = []
        if url_saisie.strip():
            col_a, col_b = st.columns([1, 3])
            if col_a.button("Recharger", use_container_width=True,
                            key="recharger_surveillance"):
                lire_liste_surveillance.clear()
            try:
                depuis_feuille = lire_liste_surveillance(url_saisie.strip())
                col_b.success(f"{len(depuis_feuille)} valeur(s) lue(s) "
                              "depuis la feuille.")
            except ValueError as erreur:
                col_b.error(str(erreur))

        saisie_liste = st.text_area(
            "Ou saisis directement tes valeurs",
            value=", ".join(depuis_feuille) if depuis_feuille else depuis_secrets,
            placeholder="AAPL, MSFT, AIR.PA, ASML.AS…", height=120,
            key="liste_perso")

        if saisie_liste.strip():
            perso = [t.strip().upper() for t in
                     saisie_liste.replace(";", ",").replace("\n", ",").split(",")
                     if t.strip()]
            liste = sorted(dict.fromkeys(perso))
            st.success(f"Liste personnalisée retenue : {len(liste)} valeurs.")

        with st.expander("Comment créer la feuille"):
            st.markdown(
                "Dans ton classeur Google, clique sur le **+** en bas à gauche "
                "pour ajouter un onglet. Mets `Ticker` en A1, puis un ticker "
                "par ligne. Ensuite **Fichier → Partager → Publier sur le "
                "web**, choisis cet onglet et le format **CSV**, publie, et "
                "colle l'adresse ci-dessus."
            )
            st.dataframe(fe.MODELE_SURVEILLANCE, use_container_width=True,
                         hide_index=True)

        st.download_button(
            "Exporter la liste sélectionnée",
            ", ".join(liste).encode("utf-8"),
            "surveillance.txt", "text/plain",
            help="À coller dans le secret GitHub TICKERS_SURVEILLANCE "
                 "pour que la veille automatique suive les mêmes valeurs.")

    st.caption(f"{len(liste)} valeurs sélectionnées.")

    if len(liste) > 150:
        st.warning(
            "Au-delà de 150 valeurs, le balayage peut dépasser les limites de "
            "mémoire de l'hébergement gratuit. Procède par univers successifs."
        )

    @st.cache_data(ttl=3600, show_spinner=False)
    def balayer(tickers: tuple[str, ...], periode: str,
                indice: str) -> pd.DataFrame:
        """Indicateurs de synthèse pour chaque valeur de l'univers."""
        cours_u = charger_cours(tickers + (indice,), periode, "1d")
        if cours_u.empty or indice not in cours_u.columns:
            return pd.DataFrame()

        r_indice = cours_u[indice].pct_change()
        lignes = []
        for t in tickers:
            if t not in cours_u.columns:
                continue
            p = cours_u[t].dropna()
            if len(p) < 260:
                continue
            r = p.pct_change().dropna()
            reg_t = an.regression_marche(r, r_indice, taux_sans_risque)
            lignes.append({
                "Ticker": t,
                "Cours": float(p.iloc[-1]),
                "Perf 1 an (%)": float(p.iloc[-1] / p.iloc[-252] - 1) * 100,
                "Perf 3 mois (%)": float(p.iloc[-1] / p.iloc[-63] - 1) * 100,
                "Momentum 12-1 (%)": float(ind.momentum_absolu(p).iloc[-1]) * 100,
                "RSI": float(ind.rsi(p).iloc[-1]),
                "vs MM200 (%)": float(p.iloc[-1] / p.rolling(200).mean().iloc[-1] - 1) * 100,
                "Distance plus haut (%)": float(ind.distance_plus_haut(p).iloc[-1]),
                "Z-score": float(ind.z_score(p).iloc[-1]),
                "Volatilité (%)": an.volatilite(r) * 100,
                "Drawdown max (%)": an.drawdown_max(r) * 100,
                "Sharpe": an.sharpe(r, taux_sans_risque),
                "Bêta": reg_t["beta"],
                "Alpha (%)": reg_t["alpha"] * 100 if np.isfinite(reg_t["alpha"]) else np.nan,
                "Corrélation": reg_t["correlation"],
                "Hurst": ind.hurst(p),
            })
        return pd.DataFrame(lignes).set_index("Ticker")

    if not liste:
        st.info("Choisis au moins un univers.")
    elif st.button("Lancer le balayage", type="primary"):
        with st.spinner(f"Analyse de {len(liste)} valeurs…"):
            st.session_state.screener = balayer(tuple(liste), periode_scr, benchmark)

    if "screener" in st.session_state and not st.session_state.screener.empty:
        scr = st.session_state.screener
        st.success(f"{len(scr)} valeurs analysées.")

        with st.expander("Filtres", expanded=True):
            f1, f2, f3 = st.columns(3)
            rsi_min, rsi_max = f1.slider("RSI", 0, 100, (0, 100))
            beta_max = f2.slider("Bêta maximal", 0.0, 3.0, 3.0, step=0.1)
            vol_max = f3.slider("Volatilité maximale (%)", 0, 100, 100, step=5)
            g1, g2 = st.columns(2)
            mom_min = g1.slider("Momentum 12-1 minimal (%)", -100, 100, -100, step=5)
            mm200 = g2.checkbox("Uniquement au-dessus de la MM200")

        filtre = scr[
            scr["RSI"].between(rsi_min, rsi_max)
            & (scr["Bêta"].fillna(99) <= beta_max)
            & (scr["Volatilité (%)"].fillna(999) <= vol_max)
            & (scr["Momentum 12-1 (%)"].fillna(-999) >= mom_min)
        ]
        if mm200:
            filtre = filtre[filtre["vs MM200 (%)"] > 0]

        tri = st.selectbox("Trier par", list(scr.columns), index=3)
        st.dataframe(filtre.sort_values(tri, ascending=False).round(2),
                     use_container_width=True, height=520)
        ia.bloc("Résultat du screener", filtre.sort_values(tri, ascending=False),
                f"{len(filtre)} valeurs retenues sur {len(scr)} balayées. "
                "Attention au biais de sélection : ces valeurs ont été "
                "choisies en connaissant leur passé.",
                "screener")
        st.caption(f"{len(filtre)} valeurs retenues sur {len(scr)}.")

        st.download_button("Exporter le screener (CSV)",
                           filtre.to_csv().encode("utf-8"),
                           "screener.csv", "text/csv")

        st.subheader("Risque et rendement de l'univers")
        nuage = filtre.dropna(subset=["Volatilité (%)", "Perf 1 an (%)"])
        if not nuage.empty:
            fig = px.scatter(
                nuage.reset_index(), x="Volatilité (%)", y="Perf 1 an (%)",
                text="Ticker", size="Volatilité (%)", color="Bêta",
                color_continuous_scale="RdBu_r",
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)


with onglets[9]:
    st.subheader("Fiche d'indicateurs")

    i1, i2 = st.columns([2, 1])
    saisie_ind = i1.text_input(
        "Ticker", value=list(val.index)[0] if len(val.index) else "AAPL",
        help="Format Yahoo Finance. Voir la table des places ci-dessous.",
    ).strip().upper()
    periode_ind = i2.selectbox("Historique", ["1y", "2y", "5y"], index=1,
                               key="periode_indicateurs")

    df_ind = charger_ohlcv(saisie_ind, periode_ind, "1d")

    if df_ind.empty:
        st.error(f"Aucune donnée pour {saisie_ind}.")
    else:
        serie_marche = cours[benchmark] if benchmark in cours.columns else None
        snap = ind.instantane(df_ind, serie_marche)

        st.caption(
            "Regroupés par famille. Les indicateurs d'une même famille sont "
            "largement redondants — voir la matrice de corrélation en bas."
        )
        for famille, noms in ind.FAMILLES.items():
            presents = [n for n in noms if n in snap and np.isfinite(snap.get(n, np.nan))]
            if not presents:
                continue
            st.markdown(f"**{famille}**")
            cols = st.columns(min(len(presents), 4))
            for j, nom in enumerate(presents):
                cols[j % 4].metric(nom, fmt(snap[nom]))

        st.divider()
        st.subheader("Régime de marché")
        r1, r2, r3 = st.columns(3)
        h = snap.get("Hurst", np.nan)
        r1.metric("Exposant de Hurst", fmt(h, 3),
                  "Persistant" if h > 0.55 else
                  ("Retour à la moyenne" if h < 0.45 else "Marche aléatoire"))
        rv = snap.get("Ratio de variance", np.nan)
        r2.metric("Ratio de variance", fmt(rv, 3),
                  help="1 = marche aléatoire, >1 tendance, <1 retour à la moyenne.")
        adx_val = snap.get("ADX", np.nan)
        r3.metric("ADX", fmt(adx_val),
                  "Tendance nette" if adx_val > 25 else "Sans tendance")

        st.subheader("Historique des indicateurs")
        tout_ind = ind.calculer_tout(df_ind, serie_marche)
        choix_ind = st.multiselect(
            "Indicateurs à tracer", list(tout_ind.columns),
            default=["RSI", "ADX", "Volatilité réalisée"],
        )
        if choix_ind:
            st.line_chart(tout_ind[choix_ind].dropna(how="all"), height=320)

        st.subheader("Redondance entre indicateurs")
        st.caption(
            "Deux indicateurs corrélés à plus de 0,9 apportent la même "
            "information. En suivre plusieurs revient à compter deux fois le "
            "même signal."
        )
        base_corr = tout_ind[choix_ind].dropna() if len(choix_ind) > 1 else pd.DataFrame()
        if len(base_corr.columns) > 1:
            fig = px.imshow(base_corr.corr(), text_auto=".2f", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=120 + 50 * len(base_corr.columns),
                              coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Trouver un ticker — conventions Yahoo Finance"):
            st.dataframe(
                pd.DataFrame(list(univ.EXEMPLES_FORMATS.items()),
                             columns=["Type", "Exemples"]),
                use_container_width=True, hide_index=True)
            st.dataframe(
                pd.DataFrame(list(univ.SUFFIXES.items()),
                             columns=["Suffixe", "Place"]),
                use_container_width=True, hide_index=True, height=300)


with onglets[10]:
    st.subheader("Analyse fondamentale")

    fcol1, fcol2 = st.columns([2, 1])
    ticker_fo = fcol1.text_input(
        "Ticker", value=list(val.index)[0] if len(val.index) else "AAPL",
        key="ticker_fondamentaux",
    ).strip().upper()
    unite = fcol2.selectbox("Unité d'affichage", ["Millions", "Milliards", "Brut"])
    diviseur = {"Millions": 1e6, "Milliards": 1e9, "Brut": 1.0}[unite]

    @st.cache_data(ttl=86400, show_spinner=False)
    def charger_etats(ticker: str):
        """États financiers annuels. Cache 24 h : ils ne bougent qu'aux publications."""
        try:
            t = yf.Ticker(ticker)
            return (t.income_stmt, t.balance_sheet, t.cashflow, dict(t.info))
        except Exception:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    with st.spinner("Chargement des états financiers…"):
        income, bilan_, cash, info_fo = charger_etats(ticker_fo)

    if income is None or income.empty:
        st.error(
            f"Aucun état financier pour {ticker_fo}. La couverture de Yahoo est "
            "bonne aux États-Unis, correcte en Europe, lacunaire ailleurs. "
            "Les ETF, indices et devises n'en ont évidemment pas."
        )
    else:
        st.caption(
            f"{info_fo.get('longName', ticker_fo)} — "
            f"{info_fo.get('sector', '—')} · {info_fo.get('industry', '—')} · "
            f"devise de publication : {info_fo.get('financialCurrency', '—')}"
        )

        familles = fo.synthese(info_fo, income, bilan_, cash)

        st.markdown("**Indicateurs clés**")
        k = st.columns(5)
        cr = fo.compte_de_resultat(income)
        ca_dernier = cr.loc["Chiffre d'affaires"].iloc[0] if "Chiffre d'affaires" in cr.index else np.nan
        ebitda_dernier = fo.ebitda_serie(income)
        mg = fo.marges(income)
        k[0].metric("Chiffre d'affaires", fmt(ca_dernier / diviseur, 0))
        k[1].metric("EBITDA", fmt((ebitda_dernier.iloc[0] if len(ebitda_dernier) else np.nan) / diviseur, 0))
        k[2].metric("Marge d'EBITDA",
                    fmt(mg.loc["Marge d'EBITDA (%)"].iloc[0] if "Marge d'EBITDA (%)" in mg.index else np.nan, 1, " %"))
        k[3].metric("Résultat net",
                    fmt((cr.loc["Résultat net"].iloc[0] if "Résultat net" in cr.index else np.nan) / diviseur, 0))
        k[4].metric("Marge nette",
                    fmt(mg.loc["Marge nette (%)"].iloc[0] if "Marge nette (%)" in mg.index else np.nan, 1, " %"))

        st.divider()
        st.markdown(f"**Compte de résultat** (en {unite.lower()})")
        st.dataframe(sans_doublons(cr / diviseur).round(2), use_container_width=True)
        ia.bloc(f"Compte de résultat — {ticker_fo}", cr / diviseur,
                f"Quatre exercices, en {unite.lower()}. "
                f"Secteur : {info_fo.get('sector', 'inconnu')}.",
                "compte_resultat")

        st.markdown("**Évolution des marges**")
        if not mg.empty:
            st.line_chart(sans_doublons(mg.T).sort_index(), height=280)
            st.caption(
                "Des marges qui s'érodent pendant que le chiffre d'affaires "
                "progresse signalent une croissance achetée par les prix."
            )

        flux = fo.flux_tresorerie(cash)
        if not flux.empty:
            st.markdown(f"**Flux de trésorerie** (en {unite.lower()})")
            st.dataframe(sans_doublons(flux / diviseur).round(2), use_container_width=True)
            st.caption(
                "Le flux libre est le poste le plus difficile à habiller "
                "comptablement. Quand il diverge durablement du résultat net, "
                "c'est lui qu'il faut croire."
            )

        st.divider()
        for nom_famille, valeurs in familles.items():
            st.markdown(f"**{nom_famille}**")
            propres = {k_: v for k_, v in valeurs.items()
                       if v is not None and np.isfinite(v)}
            if not propres:
                st.caption("Données indisponibles.")
                continue
            cols = st.columns(min(len(propres), 4))
            for j, (label, valeur) in enumerate(propres.items()):
                affichage = (fmt(valeur / diviseur, 0) if label in
                             ("Capitalisation", "Valeur d'entreprise", "Dette nette")
                             else fmt(valeur))
                cols[j % 4].metric(label, affichage)

        st.divider()
        st.subheader("Scores de qualité et de solidité")
        sc1, sc2 = st.columns(2)

        with sc1:
            p_score = fo.piotroski(income, bilan_, cash)
            st.metric("Score F de Piotroski", f"{p_score['score']} / 9")
            st.caption(
                "Neuf tests comparant l'exercice au précédent. Au-dessus de 7 "
                "le profil est solide, en dessous de 3 il est fragile."
            )
            for test, reussi in p_score["détail"].items():
                st.write(f"{'✅' if reussi else '⬜️'} {test}")

        with sc2:
            z = fo.altman_z(income, bilan_, info_fo.get("marketCap"))
            st.metric("Score Z d'Altman", fmt(z["score"]), z["verdict"])
            st.caption(
                "Probabilité de défaillance à deux ans. Au-dessus de 2,99 la "
                "situation est saine, en dessous de 1,81 le risque est élevé. "
                "Non pertinent pour les banques et les sociétés financières."
            )
            if info_fo.get("longBusinessSummary"):
                with st.expander("Activité"):
                    st.write(info_fo["longBusinessSummary"])

        st.caption(
            "Yahoo ne fournit que quatre exercices : suffisant pour juger un "
            "niveau, insuffisant pour juger une tendance longue. Les données "
            "peuvent comporter des erreurs — vérifie sur le rapport annuel "
            "avant toute décision engageante."
        )


with onglets[11]:
    st.subheader("Résultats, consensus et actualités")

    ticker_pv = st.text_input(
        "Ticker", value=list(val.index)[0] if len(val.index) else "AAPL",
        key="ticker_previsions",
    ).strip().upper()

    @st.cache_data(ttl=3600, show_spinner=False)
    def charger_consensus(ticker: str):
        """Trimestriels, estimations, avis et actualités. Cache 1 h."""
        blocs = {}
        try:
            t = yf.Ticker(ticker)
            for cle, acces in [
                ("q_income", lambda: t.quarterly_income_stmt),
                ("dates", lambda: t.earnings_dates),
                ("info", lambda: dict(t.info)),
                ("reco", lambda: t.recommendations),
                ("upgrades", lambda: t.upgrades_downgrades),
                ("eps_trend", lambda: t.eps_trend),
                ("eps_revisions", lambda: t.eps_revisions),
                ("est_ca", lambda: t.revenue_estimate),
                ("est_bpa", lambda: t.earnings_estimate),
                ("news", lambda: t.news),
            ]:
                try:
                    blocs[cle] = acces()
                except Exception:
                    blocs[cle] = pd.DataFrame() if cle != "info" else {}
        except Exception:
            pass
        return blocs

    with st.spinner("Chargement des données de consensus…"):
        d = charger_consensus(ticker_pv)

    info_pv = d.get("info", {}) or {}
    if not info_pv:
        st.error(f"Aucune donnée pour {ticker_pv}.")
    else:
        st.caption(info_pv.get("longName", ticker_pv))

        # --- Consensus d'objectif de cours -------------------------------
        st.markdown("**Consensus des analystes**")
        obj = pv.objectif_de_cours(info_pv)
        note = pv.note_moyenne(info_pv)

        c = st.columns(5)
        c[0].metric("Cours actuel", fmt(obj.get("Cours actuel")))
        c[1].metric("Objectif moyen", fmt(obj.get("Objectif moyen")),
                    f"{obj.get('Potentiel implicite (%)', float('nan')):+.1f} %"
                    if np.isfinite(obj.get("Potentiel implicite (%)", np.nan)) else None)
        c[2].metric("Fourchette",
                    f"{fmt(obj.get('Objectif bas'), 0)} – {fmt(obj.get('Objectif haut'), 0)}")
        c[3].metric("Avis moyen", note.get("avis", "—"),
                    f"note {fmt(note.get('note'), 2)}")
        c[4].metric("Analystes", obj.get("Nombre d'analystes") or "—")

        st.caption(
            "Le consensus est structurellement optimiste : les objectifs moyens "
            "dépassent le cours dans la grande majorité des cas et les "
            "recommandations de vente sont rares. Ce qui porte l'information, "
            "c'est la révision du consensus, pas son niveau."
        )

        avis = pv.distribution_avis(d.get("reco"))
        if not avis.empty:
            st.markdown("**Répartition des recommandations**")
            fig = go.Figure()
            couleurs = {"Achat fort": "#1D9E75", "Achat": "#639922",
                        "Conserver": "#EF9F27", "Vendre": "#D85A30",
                        "Vente forte": "#E24B4A"}
            for colonne in avis.columns:
                fig.add_bar(x=avis.index, y=avis[colonne], name=colonne,
                            marker_color=couleurs.get(colonne))
            fig.update_layout(barmode="stack", height=260,
                              margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", y=1.15),
                              xaxis_title="Mois écoulés")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- Estimations et révisions -------------------------------------
        st.markdown("**Estimations de chiffre d'affaires et de bénéfice**")
        est = pv.estimations(d.get("est_ca"), d.get("est_bpa"))
        if est.empty:
            st.caption("Estimations indisponibles pour cette valeur.")
        else:
            st.dataframe(sans_doublons(est).round(3), use_container_width=True)
            st.caption(
                "Les périodes sont codées par Yahoo : 0q le trimestre en cours, "
                "+1q le suivant, 0y l'exercice en cours, +1y le suivant."
            )

        st.markdown("**Révision des estimations de bénéfice**")
        rev = pv.revisions(d.get("eps_trend"))
        if rev.empty:
            st.caption("Historique de révision indisponible.")
        else:
            st.dataframe(sans_doublons(rev).round(3), use_container_width=True)
            st.caption(
                "Le momentum des révisions est le signal le plus exploitable de "
                "cette page : des estimations relevées de façon continue sont "
                "documentées comme prédictives, contrairement au niveau du "
                "consensus."
            )

        sens = pv.sens_des_revisions(d.get("eps_revisions"))
        if not sens.empty:
            st.dataframe(sans_doublons(sens), use_container_width=True)

        st.divider()

        # --- Publications et surprises ------------------------------------
        st.markdown("**Historique des publications**")
        prochaine = pv.prochaine_publication(d.get("dates"))
        if prochaine:
            p1, p2 = st.columns(2)
            p1.metric("Prochaine publication",
                      prochaine["date"].strftime("%d/%m/%Y")
                      if hasattr(prochaine["date"], "strftime") else str(prochaine["date"]))
            p2.metric("BPA attendu", fmt(prochaine.get("bpa_attendu")))

        reussite = pv.taux_de_reussite(d.get("dates"))
        if reussite:
            r = st.columns(3)
            r[0].metric("Au-dessus du consensus",
                        fmt(reussite["au-dessus du consensus (%)"], 0, " %"),
                        f"sur {reussite['publications analysées']} publications")
            r[1].metric("Surprise moyenne", fmt(reussite["surprise moyenne (%)"], 2, " %"))
            r[2].metric("Surprise médiane", fmt(reussite["surprise médiane (%)"], 2, " %"))

        surp = pv.surprises(d.get("dates"))
        if not surp.empty:
            st.dataframe(sans_doublons(surp).round(3), use_container_width=True, height=280)
            st.caption(
                "Une série ininterrompue de surprises positives traduit soit une "
                "exécution solide, soit une direction qui guide prudemment les "
                "analystes pour être sûre de les dépasser."
            )

        st.divider()

        # --- Trimestriels --------------------------------------------------
        st.markdown("**Comptes trimestriels**")
        unite_pv = st.selectbox("Unité", ["Millions", "Milliards", "Brut"],
                                key="unite_previsions")
        div_pv = {"Millions": 1e6, "Milliards": 1e9, "Brut": 1.0}[unite_pv]

        trim = pv.resultats_trimestriels(d.get("q_income"))
        if trim.empty:
            st.caption(
                "Comptes trimestriels indisponibles. Beaucoup de sociétés "
                "européennes ne publient qu'au semestre."
            )
        else:
            affichage = trim.copy()
            monetaires = [i for i in affichage.index if "%" not in i and "BPA" not in i]
            affichage.loc[monetaires] = affichage.loc[monetaires] / div_pv
            st.dataframe(sans_doublons(affichage).round(2), use_container_width=True)

            croiss = pv.croissance_annuelle_glissante(d.get("q_income"))
            if not croiss.empty:
                st.markdown("**Croissance face au même trimestre un an plus tôt (%)**")
                st.dataframe(sans_doublons(croiss).round(1), use_container_width=True)
                st.caption(
                    "Seule comparaison valable pour une activité saisonnière : "
                    "un quatrième trimestre se compare au quatrième trimestre "
                    "précédent, jamais au troisième."
                )

        st.divider()

        # --- Changements d'avis --------------------------------------------
        st.markdown("**Relèvements et abaissements**")
        solde = pv.solde_des_avis(d.get("upgrades"), 90)
        if solde:
            sc = st.columns(3)
            sc[0].metric("Relèvements (90 j)", solde["relèvements"])
            sc[1].metric("Abaissements (90 j)", solde["abaissements"])
            sc[2].metric("Solde", f"{solde['solde']:+d}")

        grandes = st.checkbox("Grandes maisons uniquement", value=False,
                              help="Goldman Sachs, Morgan Stanley, JP Morgan, "
                                   "UBS, Barclays, Bank of America et assimilées.")
        chgt = pv.changements_davis(d.get("upgrades"), 30, grandes)
        if chgt.empty:
            st.caption("Aucun changement d'avis disponible.")
        else:
            st.dataframe(chgt, use_container_width=True, height=320)
            st.caption(
                "L'objectif de cours individuel et courant de chaque maison "
                "n'est pas diffusé gratuitement — seuls Bloomberg et FactSet le "
                "fournissent. Ces changements d'avis nominatifs en sont "
                "l'équivalent le plus proche, et ce sont eux qui déplacent "
                "réellement les cours le jour de leur publication."
            )

        st.divider()

        # --- Actualités ------------------------------------------------------
        st.markdown("**Actualités**")
        actus = pv.actualites(d.get("news"), 15)
        if actus.empty:
            st.caption("Aucune actualité disponible.")
        else:
            st.dataframe(
                actus, use_container_width=True, hide_index=True, height=420,
                column_config={"Lien": st.column_config.LinkColumn(
                    "Lien", display_text="Ouvrir")},
            )
            st.caption(
                "À lire comme un contexte, jamais comme un signal : au moment "
                "où une information est publique, elle est déjà dans les cours."
            )


with onglets[12]:
    st.subheader("Contexte macroéconomique et calendrier")

    sous = st.tabs(["Calendrier", "Indicateurs macro", "Courbe des taux", "Marchés"])

    # ----------------------------------------------------------------------
    with sous[0]:
        st.markdown("**Publications de tes valeurs**")

        @st.cache_data(ttl=21600, show_spinner=False)
        def dates_publications(tickers: tuple[str, ...]) -> dict:
            """Dates de publication à venir. Cache 6 h."""
            out = {}
            for t in tickers:
                try:
                    out[t] = yf.Ticker(t).earnings_dates
                except Exception:
                    out[t] = pd.DataFrame()
            return out

        with st.spinner("Recherche des prochaines publications…"):
            cal_res = mc.calendrier_resultats(dates_publications(tuple(val.index)))

        if cal_res.empty:
            st.caption(
                "Aucune date de publication annoncée pour tes lignes. C'est "
                "fréquent hors des États-Unis, où les calendriers sont publiés "
                "plus tardivement."
            )
        else:
            imminent = cal_res[cal_res["Dans (jours)"] <= 14]
            if not imminent.empty:
                st.warning(
                    f"{len(imminent)} publication(s) dans les 14 jours : "
                    + ", ".join(imminent["Ticker"])
                )
            st.dataframe(
                cal_res.assign(Date=cal_res["Date"].dt.strftime("%d/%m/%Y")),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Les jours de publication concentrent une part importante de la "
                "volatilité annuelle d'un titre. Le sens du mouvement dépend de "
                "l'écart aux attentes, pas de la qualité absolue des chiffres."
            )

        st.divider()
        st.markdown("**Calendrier macroéconomique**")

        m1, m2 = st.columns([1, 2])
        horizon = m1.slider("Horizon (mois)", 1, 6, 2)
        zones = m2.multiselect("Zones", ["États-Unis", "Zone euro", "Mondial"],
                               default=["États-Unis", "Zone euro", "Mondial"])

        cal_macro = mc.calendrier_macro(mois_a_venir=horizon)
        if zones:
            cal_macro = cal_macro[cal_macro["Zone"].isin(zones)]

        st.info(
            "Dates reconstituées à partir des schémas habituels de publication "
            "— premier vendredi pour l'emploi, premier jour ouvré pour l'ISM, "
            "et ainsi de suite. Elles sont justes à un jour près. Vérifie sur "
            "les sites officiels avant toute décision liée à une échéance."
        )

        for semaine, evenements in mc.regrouper_par_semaine(cal_macro).items():
            with st.expander(f"{semaine} — {len(evenements)} événement(s)",
                             expanded=(list(mc.regrouper_par_semaine(cal_macro)).index(semaine) == 0)):
                for _, e in evenements.iterrows():
                    icone = {"Banque centrale": "🏛", "Statistique": "📊",
                             "Marché": "📈"}.get(e["Type"], "•")
                    st.markdown(
                        f"{icone} **{e['Date'].strftime('%A %d/%m')} · {e['Heure']}** — "
                        f"{e['Événement']} ({e['Zone']})"
                    )
                    st.caption(e["Commentaire"])

        with st.expander("Calendriers officiels"):
            for nom, lien in mc.LIENS_OFFICIELS.items():
                st.markdown(f"- [{nom}]({lien})")

    # ----------------------------------------------------------------------
    with sous[1]:
        st.caption(
            "Séries publiques de la Réserve fédérale de Saint-Louis (FRED). "
            "Les variations sont en points de pourcentage pour les taux et en "
            "pourcentage pour les niveaux."
        )

        @st.cache_data(ttl=21600, show_spinner=False)
        def charger_macro(codes: tuple[str, ...]) -> dict:
            return {c: mc.charger_serie(c) for c in codes}

        familles = sorted({f for _, f in mc.SERIES.values()})
        choix_familles = st.multiselect(
            "Familles", familles,
            default=["Taux", "Inflation", "Activité"])

        codes = tuple(c for c, (_, f) in mc.SERIES.items() if f in choix_familles)
        if not codes:
            st.info("Choisis au moins une famille.")
        else:
            with st.spinner("Téléchargement des séries FRED…"):
                series_macro = charger_macro(codes)

            tb = mc.tableau_de_bord(series_macro)
            if tb.empty:
                st.error("Séries indisponibles. FRED est peut-être inaccessible.")
            else:
                st.dataframe(tb.round(2), use_container_width=True, height=440)

                st.markdown("**Évolution d'une série**")
                dispo = {mc.SERIES[c][0]: c for c in codes
                         if not series_macro[c].empty}
                choix_serie = st.selectbox("Série", list(dispo))
                serie = series_macro[dispo[choix_serie]]

                en_glissement = st.checkbox(
                    "Afficher en glissement annuel", value=False,
                    help="Indispensable pour les indices de prix : l'indice brut "
                         "ne dit rien, seule sa variation sur douze mois est "
                         "l'inflation.")
                st.line_chart(
                    mc.variation_annuelle(serie) if en_glissement else serie,
                    height=340)

    # ----------------------------------------------------------------------
    with sous[2]:
        with st.spinner("Chargement de la courbe des taux…"):
            series_courbe = charger_macro(tuple(mc.COURBE) + ("T10Y2Y",))

        courbe = mc.courbe_des_taux(series_courbe)
        diag = mc.diagnostic_courbe(series_courbe)

        if diag:
            dc = st.columns(3)
            dc[0].metric("Écart 10 ans − 2 ans",
                         fmt(diag["écart 10-2 ans"], 2, " pts"), diag["état"])
            dc[1].metric("Lecture", diag["lecture"])
            dc[2].metric("Dernière donnée", str(diag["date"]))

        if courbe.empty:
            st.error("Courbe indisponible.")
        else:
            fig = px.line(courbe, x="Maturité (années)", y="Taux (%)",
                          markers=True, text="Échéance")
            fig.update_traces(textposition="top center", line_color="#7F77DD")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Une courbe inversée — le court terme au-dessus du long terme — "
                "a précédé chacune des récessions américaines depuis 1955, avec "
                "un délai de six à dix-huit mois et un seul faux signal. C'est "
                "le meilleur prédicteur macroéconomique connu, et aussi l'un "
                "des plus lents : l'inversion peut durer plus d'un an avant que "
                "quoi que ce soit ne se produise."
            )

            ecart = series_courbe.get("T10Y2Y")
            if ecart is not None and not ecart.empty:
                st.markdown("**Historique de l'écart 10 ans − 2 ans**")
                st.line_chart(ecart.tail(2600), height=300)

    # ----------------------------------------------------------------------
    with sous[3]:
        st.caption("Indicateurs de marché en temps quasi réel, via Yahoo.")

        @st.cache_data(ttl=900, show_spinner=False)
        def charger_marches(periode: str) -> pd.DataFrame:
            return charger_cours(tuple(mc.MARCHE), periode, "1d")

        periode_m = st.selectbox("Période", ["3mo", "6mo", "1y", "2y", "5y"],
                                 index=2, key="periode_marches")
        with st.spinner("Chargement…"):
            marches = charger_marches(periode_m)

        if marches.empty:
            st.error("Données de marché indisponibles.")
        else:
            lignes = []
            for code, libelle in mc.MARCHE.items():
                if code not in marches.columns:
                    continue
                serie = marches[code].dropna()
                if len(serie) < 30:
                    continue
                lignes.append({
                    "Indicateur": libelle,
                    "Niveau": float(serie.iloc[-1]),
                    "1 jour (%)": float(serie.iloc[-1] / serie.iloc[-2] - 1) * 100,
                    "1 mois (%)": float(serie.iloc[-1] / serie.iloc[-22] - 1) * 100
                    if len(serie) > 22 else np.nan,
                    "Depuis le début (%)": float(serie.iloc[-1] / serie.iloc[0] - 1) * 100,
                    "Volatilité (%)": an.volatilite(serie.pct_change().dropna()) * 100,
                })
            st.dataframe(pd.DataFrame(lignes).set_index("Indicateur").round(2),
                         use_container_width=True)

            st.markdown("**Comparaison en base 100**")
            choix_m = st.multiselect(
                "Indicateurs", [mc.MARCHE[c] for c in marches.columns
                                if c in mc.MARCHE],
                default=[mc.MARCHE[c] for c in ["^GSPC", "^VIX", "GC=F"]
                         if c in marches.columns])
            inverse = {v: k for k, v in mc.MARCHE.items()}
            codes_m = [inverse[n] for n in choix_m if inverse[n] in marches.columns]
            if codes_m:
                base_m = marches[codes_m].dropna()
                base_m = base_m / base_m.iloc[0] * 100
                base_m.columns = [mc.MARCHE[c] for c in codes_m]
                st.line_chart(base_m, height=340)


with onglets[13]:
    st.subheader("Qualité des données")

    st.warning(
        "yfinance est un scraper non officiel de Yahoo Finance, sans garantie "
        "ni engagement de service. Les cours quotidiens des grandes valeurs "
        "sont globalement corrects ; les fondamentaux et le consensus "
        "proviennent de fournisseurs tiers et comportent des erreurs avérées. "
        "Cet onglet ne corrige rien — il signale ce qui est détectable."
    )

    st.markdown("**Contrôle de tes lignes**")
    with st.spinner("Analyse en cours…"):
        tableau_qualite = ql.controler_univers(cours[list(val.index)])

    st.dataframe(tableau_qualite, use_container_width=True)
    ia.bloc("Qualité des données", tableau_qualite,
            "Score sur 100 mesurant l'absence d'anomalies détectables — "
            "ce n'est pas une garantie d'exactitude.",
            "qualite")

    douteux = tableau_qualite[tableau_qualite["Score"] < 65]
    if not douteux.empty:
        st.error(
            "Données douteuses sur : " + ", ".join(douteux.index)
            + ". Vérifie ces valeurs sur finance.yahoo.com avant de te fier "
              "aux calculs qui en dépendent."
        )
    else:
        st.success("Aucune anomalie détectable sur tes lignes.")

    st.divider()
    st.markdown("**Examen détaillé d'une valeur**")

    ticker_q = st.selectbox("Valeur", list(val.index), key="ticker_qualite")
    rapport = ql.controler(cours[ticker_q].dropna())
    score = ql.score_qualite(rapport)

    q = st.columns(4)
    q[0].metric("Score de qualité", f"{score['score']} / 100", score["niveau"])
    q[1].metric("Séances", rapport["profondeur"].get("séances disponibles", 0))
    q[2].metric("Couverture",
                fmt(rapport["profondeur"].get("taux de couverture (%)"), 0, " %"))
    q[3].metric("Fraîcheur", rapport["fraîcheur"].get("état", "—"))

    for cle, titre in [
        ("valeurs impossibles", "Cours nuls ou négatifs"),
        ("variations extrêmes", "Variations extrêmes"),
        ("cours figés", "Périodes de cours figé"),
        ("séances manquantes", "Interruptions d'historique"),
    ]:
        table = rapport.get(cle)
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.markdown(f"**{titre}** — {len(table)} cas")
            st.dataframe(table, use_container_width=True, hide_index=True)

    if score["score"] >= 85:
        st.caption(
            "Aucune anomalie détectable. Attention : cela ne prouve pas que "
            "les données sont exactes. Une erreur de retraitement de dividende "
            "ne laisse aucune trace visible et passera ce contrôle sans être "
            "repérée."
        )

    st.divider()
    st.markdown("**Vérification croisée avec une source indépendante**")
    st.caption(
        "Compare les cours Yahoo à ceux de Stooq, un fournisseur polonais sans "
        "lien avec Yahoo. Deux sources indépendantes qui concordent constituent "
        "une présomption sérieuse ; deux sources qui divergent signalent qu'au "
        "moins l'une des deux se trompe."
    )

    @st.cache_data(ttl=86400, show_spinner=False)
    def croiser(ticker: str, _serie_yahoo: pd.Series) -> dict:
        return ql.comparer_sources(_serie_yahoo, ql.charger_stooq(ticker))

    if st.button("Lancer la vérification croisée"):
        with st.spinner("Interrogation de Stooq…"):
            comparaison = croiser(ticker_q, cours[ticker_q].dropna())

        if not comparaison.get("comparable"):
            st.info(
                f"Comparaison impossible : {comparaison.get('raison')}. "
                "Stooq ne couvre pas les indices, devises et cryptos avec les "
                "mêmes conventions de tickers."
            )
        else:
            v = st.columns(4)
            v[0].metric("Dates comparées", comparaison["dates comparées"])
            v[1].metric("Écart médian", fmt(comparaison["écart médian (%)"], 3, " %"))
            v[2].metric("Écart maximal", fmt(comparaison["écart maximal (%)"], 2, " %"))
            v[3].metric("Dates divergentes", comparaison["dates divergentes"],
                        fmt(comparaison["part divergente (%)"], 1, " %"))

            if comparaison["dates divergentes"] == 0:
                st.success(comparaison["verdict"])
            else:
                st.warning(comparaison["verdict"])
                st.dataframe(comparaison["détail"].round(3),
                             use_container_width=True)

    st.divider()
    with st.expander("Ce que cet outil ne peut pas garantir"):
        st.markdown("""
**Sources par niveau de fiabilité.**

*FRED* (onglet Macro) est la base officielle de la Réserve fédérale de
Saint-Louis, en accès direct. Qualité institutionnelle, aucune réserve.

*Cours quotidiens Yahoo* sur les grandes capitalisations américaines et
européennes : globalement corrects, ajustés des dividendes et fractionnements.
Précision moins constante hors de ces marchés.

*Fondamentaux et consensus* : proviennent de fournisseurs tiers via Yahoo. Des
écarts avec les documents officiels des entreprises sont documentés. À
recouper avec le rapport annuel avant tout engagement significatif.

*Listes de valeurs* (`univers.py`) : figées à la date où je les ai écrites. Les
compositions d'indices changent — entrées, sorties, fusions. À réviser
périodiquement.

**Limites structurelles que rien ne détecte.**

Le biais du survivant : Yahoo ne conserve pas les sociétés radiées ou
faillies. Tout backtest sur une liste actuelle surestime donc mécaniquement
les rendements passés, puisque les disparues ont été retirées de l'échantillon.

Les erreurs de retraitement des dividendes, qui ne laissent aucune trace
visible dans la série.

Les cours différés de plusieurs minutes en séance — sans conséquence pour
l'analyse, disqualifiant pour l'exécution.

**Ce qu'il faudrait pour un usage professionnel.** Un fournisseur officiel
sous contrat : Refinitiv, FactSet, Bloomberg, ou plus modestement Polygon ou
Twelve Data. Comptez de quelques dizaines à plusieurs milliers d'euros par
mois. Cet outil est conçu pour l'analyse personnelle, pas pour gérer l'argent
de tiers.
        """)


with onglets[14]:
    st.subheader("Assistant")

    cle_api = ""
    try:
        cle_api = st.secrets.get("cle_anthropic", "")
    except Exception:
        pass

    if not cle_api:
        st.info(
            "Pour activer l'assistant, ajoute ta clé API dans les secrets "
            "Streamlit :\n\n"
            "```\ncle_anthropic = \"sk-ant-...\"\n```\n\n"
            "Une clé se crée sur console.anthropic.com. Compte environ un "
            "centime par question. Ne mets jamais cette clé dans le code : "
            "des robots parcourent GitHub en permanence à la recherche de clés "
            "publiées, et elles sont exploitées en quelques minutes."
        )
        st.stop()

    st.caption(
        "L'assistant lit les chiffres calculés par l'application et les "
        "explique. Il ne calcule rien lui-même — un modèle de langage est "
        "mauvais en arithmétique — et ne donne aucun conseil d'investissement."
    )

    a1, a2 = st.columns([2, 1])
    modele = a1.selectbox("Modèle", list(asst.MODELES),
                          format_func=lambda m: asst.MODELES[m])
    if a2.button("Effacer la conversation", use_container_width=True):
        st.session_state.pop("discussion", None)
        st.rerun()

    # Contexte : uniquement des résultats déjà calculés par Python
    contexte = asst.construire_contexte(
        valorisation=val[["Quantité", "Devise", "Cours", "PRU", "Valeur",
                          "Plus-value", "Performance (%)", "Poids (%)"]],
        metriques=an.tableau_metriques(rdt_ptf, rdt_bench, taux_sans_risque, freq),
        decomposition=an.decomposition_risque(poids, cov),
        correlations=rdt.corr(),
        parametres={
            "période": periode,
            "intervalle": intervalle,
            "indice_de_référence": nom_indice,
            "devise": devise_base,
            "taux_sans_risque": taux_sans_risque,
            "valeur_totale": round(float(total), 2),
            "nombre_de_lignes": int(len(val)),
        },
        diversification={
            "ratio_de_diversification":
                round(float(an.ratio_diversification(poids, cov)), 3),
            "lignes_effectives":
                round(float(an.nombre_effectif_lignes(poids)), 2),
        },
    )

    with st.expander("Ce que l'assistant reçoit exactement"):
        st.caption(
            "Rien d'autre que ceci. S'il évoque un chiffre absent de ce bloc, "
            "c'est qu'il l'a inventé — signale-le moi."
        )
        st.json(contexte)

    if "discussion" not in st.session_state:
        st.session_state.discussion = []

    for message in st.session_state.discussion:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.discussion:
        st.markdown("**Quelques questions pour commencer**")
        colonnes_s = st.columns(2)
        for i, suggestion in enumerate(asst.SUGGESTIONS):
            if colonnes_s[i % 2].button(suggestion, key=f"suggestion_{i}",
                                        use_container_width=True):
                st.session_state.question_en_attente = suggestion
                st.rerun()

    question = st.chat_input("Pose ta question sur le portefeuille…")
    if "question_en_attente" in st.session_state:
        question = st.session_state.pop("question_en_attente")

    if question:
        st.session_state.discussion.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyse…"):
                try:
                    reponse = asst.interroger(
                        question, contexte, cle_api,
                        historique=st.session_state.discussion[:-1],
                        modele=modele,
                    )
                    st.markdown(reponse)
                    st.session_state.discussion.append(
                        {"role": "assistant", "content": reponse})
                except Exception as erreur:
                    st.error(f"Appel impossible : {erreur}")

    st.divider()
    st.caption(
        "Rappel : cet assistant explique des chiffres, il ne recommande rien. "
        "Les mesures reposent sur des données yfinance et sur des estimations "
        "statistiques imprécises. La décision reste la tienne."
    )


with onglets[15]:
    st.subheader("Valeurs surveillées")

    url_surv = ""
    try:
        url_surv = st.secrets.get("url_surveillance", "")
    except Exception:
        pass

    entete_a, entete_b = st.columns([4, 1])
    url_surv = entete_a.text_input(
        "Feuille de surveillance",
        value=url_surv,
        help="Second onglet de ton classeur Google, publié au format CSV. "
             "Tu le modifies depuis ton téléphone, l'app le relit ici.",
        key="url_surv_onglet",
        label_visibility="collapsed",
        placeholder="Adresse de ta feuille de surveillance…")
    if entete_b.button("Recharger", use_container_width=True,
                       key="recharger_surv_onglet"):
        lire_liste_surveillance.clear()
        st.rerun()

    if not url_surv.strip():
        st.info(
            "Renseigne l'adresse de ta feuille de surveillance, ou ajoute "
            "`url_surveillance` dans les secrets Streamlit pour qu'elle se "
            "remplisse toute seule."
        )
        with st.expander("Créer la feuille"):
            st.markdown(
                "Dans ton classeur Google, ajoute un onglet avec le **+** en "
                "bas à gauche. Mets `Ticker` en A1, puis un ticker par ligne. "
                "Ensuite **Fichier → Partager → Publier sur le web**, "
                "sélectionne **cet onglet** et le format **CSV**."
            )
            st.dataframe(fe.MODELE_SURVEILLANCE, use_container_width=True,
                         hide_index=True)
        st.stop()

    try:
        with st.spinner("Lecture de la feuille…"):
            surveillees = lire_liste_surveillance(url_surv.strip())
    except ValueError as erreur:
        st.error(str(erreur))
        st.stop()

    st.caption(f"{len(surveillees)} valeur(s) suivie(s), lues depuis ta feuille. "
               "Modifie-la depuis ton téléphone puis clique sur Recharger.")

    periode_surv = st.selectbox(
        "Période d'analyse", ["6mo", "1y", "2y", "5y"], index=1,
        key="periode_surveillance")

    @st.cache_data(ttl=1800, show_spinner=False)
    def analyser_surveillance(tickers: tuple[str, ...], periode: str,
                              indice: str) -> pd.DataFrame:
        """Mesures de suivi pour chaque valeur surveillée. Cache 30 min."""
        donnees = charger_cours(tickers + (indice,), periode, "1d")
        if donnees.empty:
            return pd.DataFrame()

        r_indice = (donnees[indice].pct_change()
                    if indice in donnees.columns else pd.Series(dtype=float))
        lignes = []
        for t in tickers:
            if t not in donnees.columns:
                continue
            prix = donnees[t].dropna()
            if len(prix) < 30:
                continue
            r = prix.pct_change().dropna()

            entree = {
                "Ticker": t,
                "Cours": float(prix.iloc[-1]),
                "1 jour (%)": float(prix.iloc[-1] / prix.iloc[-2] - 1) * 100
                if len(prix) > 1 else np.nan,
                "1 mois (%)": float(prix.iloc[-1] / prix.iloc[-22] - 1) * 100
                if len(prix) > 22 else np.nan,
                "Depuis le début (%)": float(prix.iloc[-1] / prix.iloc[0] - 1) * 100,
                "RSI": float(ind.rsi(prix).iloc[-1]),
                "Volatilité (%)": an.volatilite(r) * 100,
                "Drawdown max (%)": an.drawdown_max(r) * 100,
            }
            if len(prix) > 200:
                entree["vs MM200 (%)"] = float(
                    prix.iloc[-1] / prix.rolling(200).mean().iloc[-1] - 1) * 100
            if len(prix) > 252:
                entree["Écart plus haut (%)"] = float(
                    ind.distance_plus_haut(prix).iloc[-1])
                entree["Momentum 12-1 (%)"] = float(
                    ind.momentum_absolu(prix).iloc[-1]) * 100
            if len(r_indice):
                entree["Bêta"] = an.regression_marche(
                    r, r_indice, taux_sans_risque)["beta"]
            lignes.append(entree)

        return pd.DataFrame(lignes).set_index("Ticker") if lignes else pd.DataFrame()

    with st.spinner(f"Analyse de {len(surveillees)} valeurs…"):
        tableau_surv = analyser_surveillance(
            tuple(surveillees), periode_surv, benchmark)

    if tableau_surv.empty:
        st.error("Aucune donnée exploitable. Vérifie tes tickers sur "
                 "finance.yahoo.com.")
        st.stop()

    introuvables = [t for t in surveillees if t not in tableau_surv.index]
    if introuvables:
        st.warning("Tickers sans données : " + ", ".join(introuvables)
                   + ". Vérifie leur orthographe exacte sur finance.yahoo.com.")

    m = st.columns(4)
    m[0].metric("Valeurs suivies", len(tableau_surv))
    if "1 jour (%)" in tableau_surv.columns:
        hausse = int((tableau_surv["1 jour (%)"] > 0).sum())
        m[1].metric("En hausse aujourd'hui", f"{hausse} / {len(tableau_surv)}")
    m[2].metric("Performance médiane 1 mois",
                fmt(tableau_surv["1 mois (%)"].median(), 1, " %")
                if "1 mois (%)" in tableau_surv.columns else "—")
    if "vs MM200 (%)" in tableau_surv.columns:
        au_dessus = int((tableau_surv["vs MM200 (%)"] > 0).sum())
        m[3].metric("Au-dessus de la MM200",
                    f"{au_dessus} / {len(tableau_surv)}",
                    help="Part des valeurs en tendance haussière de fond.")

    tri = st.selectbox("Trier par", list(tableau_surv.columns),
                       index=list(tableau_surv.columns).index("1 mois (%)")
                       if "1 mois (%)" in tableau_surv.columns else 0,
                       key="tri_surveillance")
    st.dataframe(tableau_surv.sort_values(tri, ascending=False).round(2),
                 use_container_width=True, height=560)
    ia.bloc("Valeurs surveillées", tableau_surv.sort_values(tri, ascending=False),
            f"{len(tableau_surv)} valeurs suivies hors portefeuille, sur "
            f"{periode_surv}. Bêta mesuré contre {nom_indice}.",
            "surveillance")

    st.download_button("Exporter (CSV)", tableau_surv.to_csv().encode("utf-8"),
                       "surveillance.csv", "text/csv")

    st.divider()
    st.subheader("Prochaines publications")
    st.caption(
        "Les jours de publication concentrent une part importante de la "
        "volatilité annuelle d'un titre. Le sens du mouvement dépend de "
        "l'écart aux attentes, pas de la qualité absolue des chiffres."
    )

    @st.cache_data(ttl=21600, show_spinner=False)
    def publications_surveillance(tickers: tuple[str, ...]) -> pd.DataFrame:
        """Dates de publication à venir. Cache 6 h."""
        lignes = []
        maintenant = pd.Timestamp.now().normalize()
        for t in tickers:
            try:
                dates = yf.Ticker(t).earnings_dates
                if dates is None or dates.empty:
                    continue
                index = pd.to_datetime(dates.index)
                index = index.tz_localize(None) if index.tz is not None else index
                table = dates.copy()
                table.index = index
                futures = table[table.index >= maintenant]
                if "Reported EPS" in table.columns:
                    futures = futures[futures["Reported EPS"].isna()]
                if futures.empty:
                    continue
                prochaine = futures.sort_index()
                lignes.append({
                    "Ticker": t,
                    "Date": prochaine.index[0],
                    "Dans (jours)": int((prochaine.index[0] - maintenant).days),
                    "BPA attendu": float(prochaine.iloc[0].get(
                        "EPS Estimate", np.nan)),
                })
            except Exception:
                continue
        return (pd.DataFrame(lignes).sort_values("Date").reset_index(drop=True)
                if lignes else pd.DataFrame())

    if st.button("Charger le calendrier des publications"):
        with st.spinner("Interrogation en cours…"):
            st.session_state.publications_surv = publications_surveillance(
                tuple(tableau_surv.index))

    calendrier_surv = st.session_state.get("publications_surv")
    if calendrier_surv is not None:
        if calendrier_surv.empty:
            st.info("Aucune date de publication annoncée. C'est fréquent hors "
                    "des États-Unis, où les calendriers sortent plus tard.")
        else:
            proches = calendrier_surv[calendrier_surv["Dans (jours)"] <= 7]
            if not proches.empty:
                st.warning(f"{len(proches)} publication(s) sous 7 jours : "
                           + ", ".join(proches["Ticker"]))
            st.dataframe(
                calendrier_surv.assign(
                    Date=calendrier_surv["Date"].dt.strftime("%d/%m/%Y")),
                use_container_width=True, hide_index=True)


with onglets[16]:
    st.subheader("Journal de transactions")
    st.caption(
        "La saisie des mouvements est la source unique de vérité : positions, "
        "prix de revient, plus-values et performance en sont dérivés et "
        "recalculés à chaque lecture. C'est ce qui rend possible le calcul du "
        "PRU exact après un split et de la performance neutralisée des apports."
    )

    url_journal = ""
    try:
        url_journal = st.secrets.get("url_journal", "")
    except Exception:
        pass

    ligne_haut = st.columns([4, 1])
    url_journal = ligne_haut[0].text_input(
        "Feuille du journal", value=url_journal,
        placeholder="Adresse CSV de ta feuille de transactions…",
        label_visibility="collapsed", key="url_journal")
    if ligne_haut[1].button("Recharger", use_container_width=True,
                            key="recharger_journal"):
        lire_journal.clear()
        st.rerun()

    if not url_journal.strip():
        st.info("Renseigne l'adresse de ta feuille, ou ajoute `url_journal` "
                "dans les secrets Streamlit.")
        with st.expander("Créer la feuille"):
            st.markdown(
                "Ajoute un onglet à ton classeur avec ces colonnes. Une ligne "
                "par mouvement. Les types reconnus : **Versement**, **Retrait**, "
                "**Achat**, **Vente**, **Dividende**, **Frais**, **Split**.\n\n"
                "Pour un split, mets le ratio dans la colonne Quantité — 2 pour "
                "un deux-pour-un. Pour un versement, le montant dans Prix."
            )
            st.dataframe(jr.MODELE, use_container_width=True, hide_index=True)
        st.stop()

    try:
        with st.spinner("Lecture du journal…"):
            mouvements = lire_journal(url_journal.strip())
    except ValueError as erreur:
        st.error(str(erreur))
        st.stop()

    if mouvements.empty:
        st.warning("Aucun mouvement exploitable dans la feuille.")
        st.stop()

    anomalies = jr.controler(mouvements)
    if anomalies:
        st.error(f"{len(anomalies)} anomalie(s) détectée(s). Les calculs "
                 "ci-dessous en tiennent compte mais peuvent être faussés.")
        for anomalie in anomalies[:8]:
            st.caption(f"Ligne {anomalie['ligne']} — {anomalie['message']}")

    pos_journal, cessions = jr.positions(mouvements)
    mouvements_espece = jr.flux(mouvements)

    m = st.columns(4)
    m[0].metric("Mouvements", len(mouvements))
    m[1].metric("Lignes ouvertes", int((pos_journal["Quantité"] > 0).sum())
                if not pos_journal.empty else 0)
    m[2].metric("Versements nets",
                fmt(float(mouvements_espece.sum()) if not mouvements_espece.empty
                    else 0, 0, f" {devise_base}"))
    m[3].metric("Plus-values réalisées",
                fmt(float(pos_journal["Plus-value réalisée"].sum())
                    if not pos_journal.empty else 0, 0, f" {devise_base}"))

    st.markdown("**Positions reconstituées**")
    st.caption(
        "Prix de revient en coût moyen pondéré, méthode retenue par "
        "l'administration fiscale française. Un split multiplie les quantités "
        "et divise le prix de revient, sans créer de plus-value."
    )
    st.dataframe(pos_journal.round(4), use_container_width=True)
    ia.bloc("Positions reconstituées", pos_journal,
            "Positions dérivées du journal de transactions, avec prix de "
            "revient exact et plus-values réalisées.", "positions_journal")

    if not cessions.empty:
        st.markdown("**Cessions réalisées**")
        st.caption("Base de la déclaration de plus-values. Chaque cession "
                   "porte son prix de revient au moment de la vente.")
        st.dataframe(cessions.round(2), use_container_width=True,
                     hide_index=True)
        annee = st.selectbox(
            "Année fiscale",
            sorted({d.year for d in cessions["Date"]}, reverse=True),
            key="annee_fiscale")
        cessions_annee = cessions[cessions["Date"].dt.year == annee]
        total_pv = float(cessions_annee["Plus-value"].sum())
        st.metric(f"Plus-value nette {annee}",
                  fmt(total_pv, 2, f" {devise_base}"),
                  help="Somme des plus et moins-values de l'année. Les "
                       "moins-values sont imputables sur les plus-values de "
                       "même nature, sur dix ans.")
        st.download_button(
            f"Exporter les cessions {annee} (CSV)",
            cessions_annee.to_csv(index=False).encode("utf-8"),
            f"cessions_{annee}.csv", "text/csv")

    st.divider()
    st.subheader("Performance réelle")
    st.caption(
        "Le rendement pondéré par le temps neutralise l'effet des apports : "
        "c'est la seule mesure comparable à un indice. Le taux de rendement "
        "interne répond à une autre question — combien as-tu réellement gagné, "
        "compte tenu du moment où tu as investi."
    )

    tickers_journal = [t for t in pos_journal.index if t]
    if tickers_journal:
        with st.spinner("Reconstitution de la valeur quotidienne…"):
            cours_journal = charger_cours(tuple(tickers_journal), "5y", "1d")
            serie_valeur = jr.valeurs_quotidiennes(mouvements, cours_journal)

        if len(serie_valeur) > 2:
            rendements_twr = jr.twr(serie_valeur, mouvements_espece)
            perf_twr = float((1 + rendements_twr).prod() - 1) * 100
            annees = len(rendements_twr) / 252
            twr_annualise = (((1 + perf_twr / 100) ** (1 / annees) - 1) * 100
                             if annees > 0.5 else np.nan)
            taux_tri = jr.tri(mouvements_espece,
                              float(serie_valeur.iloc[-1])) * 100

            valeur_finale = float(serie_valeur.iloc[-1])
            verse = float(mouvements_espece.sum()) if not mouvements_espece.empty else 0

            p = st.columns(4)
            p[0].metric("Valeur actuelle", fmt(valeur_finale, 0, f" {devise_base}"))
            p[1].metric("TWR cumulé", fmt(perf_twr, 1, " %"),
                        help="Performance des décisions d'investissement, "
                             "indépendante du calendrier des apports.")
            p[2].metric("TWR annualisé", fmt(twr_annualise, 1, " %"))
            p[3].metric("TRI", fmt(taux_tri, 1, " %"),
                        help="Performance réellement obtenue, sensible au "
                             "moment des versements.")

            if verse > 0:
                apparent = (valeur_finale / verse - 1) * 100
                if abs(apparent - perf_twr) > 3:
                    st.info(
                        f"L'écart mérite attention : rapportée aux versements, "
                        f"la performance semble de {apparent:+.1f} %, alors que "
                        f"le TWR ressort à {perf_twr:+.1f} %. La différence "
                        "vient du calendrier des apports, pas de tes choix.")

            comparaison = pd.DataFrame({
                "Portefeuille": (1 + rendements_twr).cumprod() * 100 - 100})
            if not rdt_bench.empty:
                aligne = rdt_bench.reindex(rendements_twr.index).fillna(0)
                comparaison[nom_indice] = (1 + aligne).cumprod() * 100 - 100
            st.line_chart(comparaison, height=320)
            st.caption("Base 100 au premier jour du journal, en pourcentage.")
        else:
            st.info("Historique trop court pour calculer la performance.")


with onglets[17]:
    st.subheader("Thèses d'investissement")
    st.caption(
        "Une thèse s'écrit avant l'achat, avec des conditions chiffrées de ce "
        "qui la rendrait fausse. Le système les vérifie et signale l'écart au "
        "moment où il apparaît, en rappelant le texte d'origine. Sans ce "
        "rappel, une thèse se reformule inconsciemment pour coller aux faits."
    )

    url_these = ""
    try:
        url_these = st.secrets.get("url_theses", "")
    except Exception:
        pass

    ligne_haut = st.columns([4, 1])
    url_these = ligne_haut[0].text_input(
        "Feuille des thèses", value=url_these,
        placeholder="Adresse CSV de ta feuille de thèses…",
        label_visibility="collapsed", key="url_theses")
    if ligne_haut[1].button("Recharger", use_container_width=True,
                            key="recharger_theses"):
        lire_theses.clear()
        st.rerun()

    if not url_these.strip():
        st.info("Renseigne l'adresse, ou ajoute `url_theses` dans les secrets.")
        with st.expander("Créer la feuille"):
            st.markdown(
                "Une ligne par valeur. Les colonnes **Hypothèses** et "
                "**Invalidation** doivent contenir des seuils chiffrés pour "
                "être vérifiables — par exemple `marge nette > 24 % et "
                "croissance > 5 %`.\n\n"
                "Attention au sens : une hypothèse décrit ce qui doit rester "
                "vrai, une invalidation décrit ce qui ne doit pas arriver."
            )
            st.dataframe(th.MODELE_THESE, use_container_width=True,
                         hide_index=True)
        st.stop()

    try:
        with st.spinner("Lecture des thèses…"):
            theses = lire_theses(url_these.strip())
    except ValueError as erreur:
        st.error(str(erreur))
        st.stop()

    if theses.empty:
        st.warning("Aucune thèse exploitable dans la feuille.")
        st.stop()

    @st.cache_data(ttl=3600, show_spinner=False)
    def mesures_fondamentales(ticker: str) -> dict:
        """Chiffres nécessaires à la vérification des hypothèses. Cache 1 h."""
        import resultats as rs
        publie = rs.chiffres_publies(ticker)
        mesures = {k: v for k, v in publie.items() if isinstance(v, (int, float))}
        try:
            infos = dict(yf.Ticker(ticker).info)
            for cible, source in [("per", "trailingPE"),
                                  ("roe_pct", "returnOnEquity"),
                                  ("marge_brute_pct", "grossMargins"),
                                  ("cours", "currentPrice")]:
                valeur = infos.get(source)
                if valeur is not None:
                    mesures[cible] = (float(valeur) * 100
                                      if cible.endswith("_pct") and abs(valeur) < 5
                                      else float(valeur))
        except Exception:
            pass
        return mesures

    lancer = st.button("Vérifier toutes les thèses", type="primary")
    if lancer or st.session_state.get("theses_verifiees"):
        if lancer:
            resultats_theses = {}
            barre = st.progress(0.0)
            for i, (ticker, ligne) in enumerate(theses.iterrows()):
                mesures = mesures_fondamentales(ticker)
                verdict = th.verifier(ligne.to_dict(), mesures)
                cours_actuel = mesures.get("cours")
                verdict["esperance"] = (th.esperance(ligne.to_dict(), cours_actuel)
                                        if cours_actuel else {})
                verdict["cours"] = cours_actuel
                resultats_theses[ticker] = verdict
                barre.progress((i + 1) / len(theses))
            barre.empty()
            st.session_state["theses_verifiees"] = resultats_theses

        resultats_theses = st.session_state["theses_verifiees"]

        compte = {}
        for verdict in resultats_theses.values():
            compte[verdict["statut"]] = compte.get(verdict["statut"], 0) + 1
        s = st.columns(4)
        s[0].metric("Intactes", compte.get("intacte", 0))
        s[1].metric("Sous surveillance", compte.get("sous surveillance", 0))
        s[2].metric("Invalidées", compte.get("invalidée", 0))
        s[3].metric("Non vérifiables", compte.get("non vérifiable", 0))

        ordre = {"invalidée": 0, "sous surveillance": 1, "intacte": 2,
                 "non vérifiable": 3}
        for ticker in sorted(resultats_theses,
                             key=lambda t: ordre.get(resultats_theses[t]["statut"], 9)):
            verdict = resultats_theses[ticker]
            ligne = theses.loc[ticker]
            pastille = {"intacte": "🟢", "sous surveillance": "🟡",
                        "invalidée": "🔴", "non vérifiable": "⬜️"}[verdict["statut"]]

            with st.expander(f"{pastille} {ticker} — {verdict['statut']}",
                             expanded=verdict["statut"] == "invalidée"):
                if ligne.get("raison"):
                    st.markdown(f"*{ligne['raison']}*")

                esp = verdict.get("esperance") or {}
                if esp:
                    e = st.columns(4)
                    e[0].metric("Espérance", fmt(esp["esperance_pct"], 1, " %"))
                    e[1].metric("Gain espéré", fmt(esp["gain_espere_pct"], 1, " %"))
                    e[2].metric("Perte espérée", fmt(esp["perte_esperee_pct"], 1, " %"))
                    e[3].metric("Asymétrie", fmt(esp.get("asymetrie"), 2))
                    if esp["esperance_pct"] < 8:
                        st.warning(
                            "L'espérance est faible au regard du rendement "
                            "historique d'un indice large : la sélection et le "
                            "risque spécifique ne sont pas rémunérés.")

                if verdict["hypotheses"]:
                    st.markdown("**Hypothèses**")
                    for h in verdict["hypotheses"]:
                        marque = {"vérifiée": "✅", "démentie": "❌",
                                  "non vérifiable": "⬜️"}[h["statut"]]
                        valeur = ("n.c." if h["valeur"] is None
                                  else f"{h['valeur']:.1f}")
                        st.markdown(f"{marque} {h['libelle']} : **{valeur}** "
                                    f"(seuil {h['operateur']}{h['seuil']:g})")

                if verdict["invalidations"]:
                    st.markdown("**Conditions de sortie**")
                    for h in verdict["invalidations"]:
                        declenchee = h["statut"] == "vérifiée"
                        marque = "🔴" if declenchee else "⬜️" if h["valeur"] is None else "🟢"
                        valeur = ("n.c." if h["valeur"] is None
                                  else f"{h['valeur']:.1f}")
                        st.markdown(f"{marque} {h['libelle']} : **{valeur}** "
                                    f"(seuil {h['operateur']}{h['seuil']:g})"
                                    + ("  ← **remplie**" if declenchee else ""))

                if verdict["statut"] == "invalidée":
                    st.error(
                        "Une condition que tu as écrite toi-même est remplie. "
                        f"Texte d'origine : « {ligne.get('invalidation', '')} »"
                        + (f" — écrite le {pd.Timestamp(ligne['date']).strftime('%d/%m/%Y')}."
                           if ligne.get("date") is not None else "."))

                if verdict.get("a_revoir"):
                    st.warning(f"Thèse non mise à jour depuis "
                               f"{verdict['anciennete_jours']} jours.")
