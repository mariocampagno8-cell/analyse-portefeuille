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

import analytics as an
import fondamentaux as fo
import indicateurs as ind
import optimisation as opt
import univers as univ

st.set_page_config(page_title="Portefeuille", page_icon="📊", layout="wide")

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


def fmt(valeur, decimales=2, suffixe=""):
    if valeur is None or (isinstance(valeur, float) and not np.isfinite(valeur)):
        return "—"
    return f"{valeur:,.{decimales}f}{suffixe}".replace(",", " ")


# --------------------------------------------------------------------------
# Panneau lateral
# --------------------------------------------------------------------------

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

st.sidebar.divider()
st.sidebar.caption(
    "Tickers au format Yahoo Finance : AIR.PA (Airbus), MSFT (Microsoft), "
    "IWDA.AS (ETF World)."
)


# --------------------------------------------------------------------------
# Saisie du portefeuille
# --------------------------------------------------------------------------

st.title("Analyse de portefeuille")

if "portefeuille" not in st.session_state:
    st.session_state.portefeuille = charger_portefeuille()

onglets = st.tabs([
    "Portefeuille", "Risque et bêta", "Corrélations", "Optimisation",
    "Simulation et stress", "Backtest", "Analyse d'une valeur", "Données",
    "Screener", "Indicateurs", "Fondamentaux",
])


with onglets[0]:
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
            "Quantité": st.column_config.NumberColumn("Quantité", min_value=0.0,
                                                      step=1.0, format="%.4f"),
            "Prix d'achat": st.column_config.NumberColumn(
                "Prix d'achat unitaire", min_value=0.0, step=0.01, format="%.2f",
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
        })

    val = pd.DataFrame(valorisation).set_index("Ticker")
    val = val.groupby(level=0).agg({
        "Quantité": "sum", "Devise": "first", "Cours": "last",
        "PRU": "mean", "Valeur": "sum", "Investi": "sum",
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
        st.dataframe(
            val[["Quantité", "Devise", "Cours", "PRU", "Valeur",
                 "Plus-value", "Performance (%)", "Poids (%)"]].round(2),
            use_container_width=True,
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
    st.dataframe(pd.DataFrame(resume).set_index("Allocation").round(2),
                 use_container_width=True)

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

            st.dataframe(
                pd.DataFrame(mesures).set_index("Méthode").round(2)
                .sort_values("Sharpe", ascending=False),
                use_container_width=True,
            )
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
    actif = st.selectbox("Valeur", list(val.index))
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
        st.dataframe((cr / diviseur).round(2), use_container_width=True)

        st.markdown("**Évolution des marges**")
        if not mg.empty:
            st.line_chart(mg.T.sort_index(), height=280)
            st.caption(
                "Des marges qui s'érodent pendant que le chiffre d'affaires "
                "progresse signalent une croissance achetée par les prix."
            )

        flux = fo.flux_tresorerie(cash)
        if not flux.empty:
            st.markdown(f"**Flux de trésorerie** (en {unite.lower()})")
            st.dataframe((flux / diviseur).round(2), use_container_width=True)
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
