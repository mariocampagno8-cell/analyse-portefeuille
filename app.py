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
import feuille as fe
import fondamentaux as fo
import indicateurs as ind
import optimisation as opt
import options as op
import portefeuille as pf
import previsions as pv
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

# Ces valeurs sont calculées dans l'onglet Portefeuille et partagées avec les
# autres. Elles sont initialisées ici pour que chaque onglet puisse vérifier
# leur disponibilité plutôt que d'échouer sur une variable inexistante.
val = None
poids = None
rdt = None
rdt_bench = None
freq = None
rdt_ptf = None
cov = None
total = None
cours = None
reg = None
edite = None


def portefeuille_pret() -> bool:
    """Vrai si l'onglet Portefeuille a pu charger et valoriser les positions."""
    return val is not None and not val.empty


def message_prealable() -> None:
    st.info("Charge d'abord ton portefeuille dans l'onglet **Portefeuille** : "
            "les mesures de cet onglet en dépendent.")


onglets = st.tabs([
    "Portefeuille", "Valeur", "Surveillance", "Risque", "Simulateur",
    "Stratégies options",
])


@st.cache_data(ttl=300, show_spinner=False)
def lire_liste_surveillance(url: str) -> list[str]:
    """Liste de surveillance depuis Google Sheets. Cache 5 min."""
    return fe.lire_liste(url)


@st.cache_data(ttl=300, show_spinner=False)
def lire_feuille(url: str) -> pd.DataFrame:
    """Portefeuille depuis Google Sheets. Cache 5 min."""
    return fe.lire(url)


def _onglet_0():
    global val, poids, rdt, rdt_bench, freq, rdt_ptf, cov, total, cours, reg, edite
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
            return

        try:
            with st.spinner("Lecture de la feuille…"):
                depuis_feuille = lire_feuille(url_feuille)
        except ValueError as erreur:
            st.error(str(erreur))
            return

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
        return

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
        return

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

with onglets[0]:
    _onglet_0()






def _onglet_1():
    if not portefeuille_pret():
        message_prealable()
        return

    st.subheader("Analyse d'une valeur")
    st.caption(
        "Tout ce qui concerne une société, au même endroit : le cours et sa "
        "tendance, les comptes, le consensus, et les indicateurs techniques."
    )

    sous_valeur = st.tabs(["Cours", "Fondamentaux", "Résultats", "Indicateurs"])

    with sous_valeur[0]:
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

    with sous_valeur[1]:
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

    with sous_valeur[2]:
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

    with sous_valeur[3]:
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

with onglets[1]:
    _onglet_1()


def _onglet_2():
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
        return

    try:
        with st.spinner("Lecture de la feuille…"):
            surveillees = lire_liste_surveillance(url_surv.strip())
    except ValueError as erreur:
        st.error(str(erreur))
        return

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
        return

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

with onglets[2]:
    _onglet_2()


def _onglet_3():
    if not portefeuille_pret():
        message_prealable()
        return

    st.subheader("Risque")
    st.caption(
        "Trois façons de regarder la même question : d'où vient le risque, "
        "comment les lignes bougent ensemble, et ce que donnerait un choc."
    )

    sous_risque = st.tabs(["Mesures", "Corrélations", "Scénarios"])

    with sous_risque[0]:
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

    with sous_risque[1]:
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

    with sous_risque[2]:
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

with onglets[3]:
    _onglet_3()


def _onglet_4():
    if not portefeuille_pret():
        message_prealable()
        return

    st.subheader("Simulateur de portefeuille")
    st.caption(
        "Construis un portefeuille librement et mesure ce qu'il aurait donné. "
        "Aucun lien avec tes positions réelles : c'est un banc d'essai pour "
        "comparer des allocations avant de s'y engager."
    )

    st.info(
        "Rappel de lecture : une performance passée mesure ce qui s'est "
        "produit, pas ce qui se reproduira. Le risque, lui, est nettement plus "
        "persistant — une allocation volatile hier le restera probablement.",
        icon="⚠️")

    if "simulation" not in st.session_state:
        st.session_state.simulation = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "AIR.PA", "TTE.PA"],
            "Poids (%)": [30.0, 30.0, 20.0, 20.0],
        })

    reglages = st.columns(3)
    periode_sim = reglages[0].selectbox(
        "Période", ["1y", "2y", "3y", "5y", "10y"], index=3, key="periode_sim")
    capital_sim = reglages[1].number_input(
        f"Capital initial ({devise_base})", min_value=1000.0,
        value=10000.0, step=1000.0, key="capital_sim")
    rebalancement = reglages[2].selectbox(
        "Rééquilibrage", ["Aucun", "Trimestriel", "Annuel"], index=2,
        key="rebal_sim",
        help="Sans rééquilibrage, les poids dérivent avec les performances : "
             "le portefeuille devient de plus en plus concentré sur ce qui a "
             "monté.")

    st.markdown("**Composition**")
    st.caption("Ajoute des lignes, modifie les poids. Ils seront normalisés à "
               "100 % au calcul.")

    composition = st.data_editor(
        st.session_state.simulation, num_rows="dynamic",
        use_container_width=True, key="editeur_sim",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Poids (%)": st.column_config.NumberColumn(
                "Poids (%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.1f")})

    modeles = st.columns(4)
    if modeles[0].button("60/40", use_container_width=True):
        st.session_state.simulation = pd.DataFrame({
            "Ticker": ["IWDA.AS", "AGGH.MI"], "Poids (%)": [60.0, 40.0]})
        st.rerun()
    if modeles[1].button("Actions monde", use_container_width=True):
        st.session_state.simulation = pd.DataFrame({
            "Ticker": ["IWDA.AS"], "Poids (%)": [100.0]})
        st.rerun()
    if modeles[2].button("Permanent", use_container_width=True,
                         help="Actions, obligations longues, or, monétaire — "
                              "à parts égales."):
        st.session_state.simulation = pd.DataFrame({
            "Ticker": ["IWDA.AS", "IBTL.AS", "SGLD.MI", "ERNX.MI"],
            "Poids (%)": [25.0, 25.0, 25.0, 25.0]})
        st.rerun()
    if modeles[3].button("Mon portefeuille", use_container_width=True):
        total_actuel = float(val["Valeur"].sum())
        st.session_state.simulation = pd.DataFrame({
            "Ticker": list(val.index),
            "Poids (%)": [float(v) / total_actuel * 100
                          for v in val["Valeur"]]})
        st.rerun()

    propre = composition.dropna(subset=["Ticker"])
    propre = propre[propre["Ticker"].astype(str).str.strip() != ""]

    if propre.empty:
        st.warning("Ajoute au moins une ligne pour lancer la simulation.")
        return

    tickers_sim = [str(t).strip().upper() for t in propre["Ticker"]]
    poids_bruts = pd.Series(
        [float(p) if pd.notna(p) else 0.0 for p in propre["Poids (%)"]],
        index=tickers_sim)
    if poids_bruts.sum() <= 0:
        st.error("La somme des poids doit être positive.")
        return
    poids_sim = poids_bruts / poids_bruts.sum()

    if abs(poids_bruts.sum() - 100) > 0.5:
        st.caption(f"Poids normalisés : la somme saisie était de "
                   f"{poids_bruts.sum():.1f} %.")

    with st.spinner("Téléchargement des cours…"):
        cours_sim = charger_cours(tuple(tickers_sim + [benchmark]),
                                  periode_sim, "1d")

    manquants = [t for t in tickers_sim if t not in cours_sim.columns]
    if manquants:
        st.warning(f"Sans données : {', '.join(manquants)}. Vérifie "
                   "l'orthographe exacte sur finance.yahoo.com.")
        tickers_sim = [t for t in tickers_sim if t in cours_sim.columns]
        if not tickers_sim:
            return
        poids_sim = poids_sim[tickers_sim] / poids_sim[tickers_sim].sum()

    rdt_sim = cours_sim[tickers_sim].pct_change().dropna()
    if len(rdt_sim) < 60:
        st.error("Historique trop court pour une mesure exploitable.")
        return

    # --- Reconstitution de la trajectoire
    if rebalancement == "Aucun":
        # Sans rééquilibrage, les poids dérivent : chaque ligne suit sa
        # propre trajectoire depuis le capital initial.
        parts = (capital_sim * poids_sim) / cours_sim[tickers_sim].iloc[0]
        valeur_sim = (cours_sim[tickers_sim] * parts).sum(axis=1)
        rdt_ptf_sim = valeur_sim.pct_change().dropna()
    else:
        frequence = 63 if rebalancement == "Trimestriel" else 252
        rdt_ptf_sim = pd.Series(index=rdt_sim.index, dtype=float)
        poids_courants = poids_sim.copy()
        for i, (date, ligne) in enumerate(rdt_sim.iterrows()):
            rendement = float((poids_courants * ligne).sum())
            rdt_ptf_sim.loc[date] = rendement
            # Les poids dérivent d'un jour sur l'autre
            poids_courants = poids_courants * (1 + ligne)
            poids_courants /= poids_courants.sum()
            if (i + 1) % frequence == 0:
                poids_courants = poids_sim.copy()
        valeur_sim = capital_sim * (1 + rdt_ptf_sim).cumprod()

    rdt_ref = (cours_sim[benchmark].pct_change().dropna()
               if benchmark in cours_sim.columns else pd.Series(dtype=float))

    # --- Mesures
    annees = len(rdt_ptf_sim) / 252
    perf_totale = float((1 + rdt_ptf_sim).prod() - 1) * 100
    perf_annuelle = ((1 + perf_totale / 100) ** (1 / annees) - 1) * 100 \
        if annees > 0.5 else np.nan
    vol_sim = an.volatilite(rdt_ptf_sim) * 100
    sharpe_sim = an.sharpe(rdt_ptf_sim, taux_sans_risque)
    dd_sim = an.drawdown_max(rdt_ptf_sim) * 100

    m = st.columns(5)
    m[0].metric("Valeur finale", fmt(float(valeur_sim.iloc[-1]), 0,
                                     f" {devise_base}"),
                fmt(perf_totale, 1, " %"))
    m[1].metric("Performance annualisée", fmt(perf_annuelle, 1, " %"))
    m[2].metric("Volatilité", fmt(vol_sim, 1, " %"))
    m[3].metric("Sharpe", fmt(sharpe_sim, 2))
    m[4].metric("Perte maximale", fmt(dd_sim, 1, " %"),
                help="Le pire recul entre un sommet et le creux qui a suivi. "
                     "C'est ce chiffre, pas la volatilité, qui détermine si "
                     "une allocation est tenable psychologiquement.")

    # --- Trajectoire
    trajectoire = pd.DataFrame({"Simulation": valeur_sim})
    if len(rdt_ref):
        aligne = rdt_ref.reindex(rdt_ptf_sim.index).fillna(0)
        trajectoire[nom_indice] = capital_sim * (1 + aligne).cumprod()
    st.line_chart(trajectoire, height=340)
    st.caption(f"Valeur d'un capital de {capital_sim:,.0f} {devise_base} "
               f"investi au début de la période.".replace(",", " "))

    st.markdown("**Pertes depuis le plus haut**")
    st.area_chart(pd.DataFrame({
        "Simulation": an.courbe_drawdown(rdt_ptf_sim) * 100}), height=220)

    # --- Comparaison à l'indice
    if len(rdt_ref) > 60:
        reg_sim = an.regression_marche(rdt_ptf_sim, rdt_ref, taux_sans_risque)
        perf_ref = float((1 + rdt_ref.reindex(rdt_ptf_sim.index)
                          .fillna(0)).prod() - 1) * 100
        c = st.columns(4)
        c[0].metric(f"Écart à {nom_indice}", fmt(perf_totale - perf_ref, 1, " pt"))
        c[1].metric("Bêta", fmt(reg_sim.get("beta"), 2))
        c[2].metric("Alpha annualisé", fmt(reg_sim.get("alpha_annualise", 0) * 100,
                                           1, " %"))
        c[3].metric("Corrélation", fmt(reg_sim.get("correlation"), 2))

    # --- Décomposition du risque
    st.divider()
    st.markdown("**D'où vient le risque**")
    st.caption("La contribution au risque ne suit presque jamais les poids : "
               "c'est l'écart entre les deux colonnes qui informe.")

    cov_sim = an.matrice_covariance(rdt_sim)
    decompo_sim = an.decomposition_risque(poids_sim, cov_sim)
    st.dataframe(decompo_sim.sort_values("Part du risque (%)",
                                         ascending=False).round(2),
                 use_container_width=True)

    d = st.columns(3)
    d[0].metric("Lignes effectives", fmt(an.nombre_effectif_lignes(poids_sim), 1),
                help="Nombre de lignes réellement indépendantes. Très "
                     "inférieur au nombre de positions dès que les lignes "
                     "sont corrélées.")
    correlation_moyenne = rdt_sim.corr().where(
        ~np.eye(len(rdt_sim.columns), dtype=bool)).stack().mean()
    d[1].metric("Corrélation moyenne", fmt(correlation_moyenne, 2))
    d[2].metric("Ligne dominante",
                decompo_sim["Part du risque (%)"].idxmax()
                if not decompo_sim.empty else "—",
                fmt(decompo_sim["Part du risque (%)"].max(), 0, " %")
                if not decompo_sim.empty else None)

    st.divider()
    st.markdown("**Lecture de cette simulation**")
    ia.bloc("Simulation de portefeuille",
            pd.concat([decompo_sim,
                       pd.DataFrame({"Mesure": ["Perf. annualisée", "Volatilité",
                                                "Sharpe", "Perte max"],
                                     "Valeur": [perf_annuelle, vol_sim,
                                                sharpe_sim, dd_sim]}
                                    ).set_index("Mesure")], axis=0),
            f"Portefeuille fictif de {len(tickers_sim)} lignes sur "
            f"{periode_sim}, rééquilibrage {rebalancement.lower()}, capital "
            f"de {capital_sim:.0f} {devise_base}.",
            "simulateur", ouvert=True, automatique=True,
            consignes=ia.CONSIGNES_PORTEFEUILLE)

    st.download_button(
        "Exporter la simulation (CSV)",
        trajectoire.to_csv().encode("utf-8"),
        "simulation.csv", "text/csv")

    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Allocations optimales")
    st.caption(
        "Six façons de répartir les mêmes valeurs, calculées sur la période "
        "choisie, comparées à ta composition."
    )

    st.warning(
        "Avertissement décisif sur ces chiffres. Ces allocations sont "
        "optimisées SUR LA PÉRIODE AFFICHÉE : elles connaissent son résultat. "
        "Leur performance passée est donc flatteuse par construction et ne "
        "dit rien de l'avenir.\n\n"
        "La distinction qui compte : les méthodes qui maximisent le rendement "
        "reposent sur des rendements attendus, quantité que personne ne sait "
        "estimer — une erreur de 1 % sur une espérance déplace les poids de "
        "dizaines de points. Les méthodes qui minimisent le risque ne "
        "dépendent que de la covariance, bien plus stable dans le temps. "
        "C'est pourquoi la variance minimale et la parité de risque tiennent "
        "hors échantillon, là où le Sharpe maximal se dégrade presque "
        "toujours.",
        icon="⚠️")

    if len(tickers_sim) < 2:
        st.info("Au moins deux lignes sont nécessaires pour optimiser.")
    else:
        cov_opt, _ = opt.covariance_retrecie(rdt_sim, freq=252)
        mu_opt = opt.rendements_attendus(rdt_sim, methode="retreci", freq=252)

        allocations = {"Ma composition": poids_sim}
        methodes = [
            ("Variance minimale", lambda: opt.variance_minimale(cov_opt),
             "Le risque le plus faible possible. Ne dépend que de la "
             "covariance, donc la plus robuste hors échantillon."),
            ("Parité de risque", lambda: opt.parite_de_risque(cov_opt),
             "Chaque ligne contribue également au risque total. Bon "
             "compromis entre robustesse et diversification."),
            ("Sharpe maximal", lambda: opt.sharpe_maximal(mu_opt, cov_opt,
                                                          taux_sans_risque),
             "Le meilleur rendement par unité de risque — sur le passé. "
             "Très sensible aux erreurs d'estimation."),
            ("Diversification maximale",
             lambda: opt.diversification_maximale(cov_opt),
             "Maximise l'écart entre la volatilité moyenne des lignes et "
             "celle du portefeuille."),
            ("HRP", lambda: opt.hrp(cov_opt),
             "Répartition hiérarchique par grappes de corrélation. Ne "
             "requiert aucune inversion de matrice, donc stable."),
            ("Équipondéré",
             lambda: pd.Series(1 / len(tickers_sim), index=tickers_sim),
             "Référence difficile à battre, et sans aucun paramètre à "
             "estimer."),
        ]

        explications = {"Ma composition": "Les poids que tu as saisis."}
        with st.spinner("Optimisation…"):
            for nom, calcul, explication in methodes:
                try:
                    allocations[nom] = calcul()
                    explications[nom] = explication
                except Exception as erreur:
                    st.caption(f"{nom} non calculable : {erreur}")

        def _mesurer(poids_test: pd.Series) -> dict:
            """Rejoue la même simulation avec une autre répartition."""
            poids_test = poids_test.reindex(tickers_sim).fillna(0)
            if poids_test.sum() <= 0:
                return {}
            poids_test = poids_test / poids_test.sum()

            if rebalancement == "Aucun":
                parts_test = (capital_sim * poids_test) / cours_sim[tickers_sim].iloc[0]
                serie = (cours_sim[tickers_sim] * parts_test).sum(axis=1)
                r_test = serie.pct_change().dropna()
            else:
                frequence_test = 63 if rebalancement == "Trimestriel" else 252
                r_test = pd.Series(index=rdt_sim.index, dtype=float)
                courants = poids_test.copy()
                for i, (date, ligne) in enumerate(rdt_sim.iterrows()):
                    r_test.loc[date] = float((courants * ligne).sum())
                    courants = courants * (1 + ligne)
                    courants /= courants.sum()
                    if (i + 1) % frequence_test == 0:
                        courants = poids_test.copy()

            duree = len(r_test) / 252
            cumul = float((1 + r_test).prod() - 1) * 100
            return {
                "Perf. annualisée (%)": (((1 + cumul / 100) ** (1 / duree) - 1) * 100
                                         if duree > 0.5 else np.nan),
                "Volatilité (%)": an.volatilite(r_test) * 100,
                "Sharpe": an.sharpe(r_test, taux_sans_risque),
                "Perte max (%)": an.drawdown_max(r_test) * 100,
                "Lignes effectives": an.nombre_effectif_lignes(poids_test),
                "Poids max (%)": float(poids_test.max()) * 100,
            }

        comparaison = {}
        for nom, poids_test in allocations.items():
            mesures = _mesurer(poids_test)
            if mesures:
                comparaison[nom] = mesures

        tableau_opt = pd.DataFrame(comparaison).T
        tri_opt = st.selectbox(
            "Trier par", list(tableau_opt.columns), index=2, key="tri_optim")
        croissant = tri_opt in ("Volatilité (%)", "Poids max (%)")
        st.dataframe(
            tableau_opt.sort_values(tri_opt, ascending=croissant).round(2),
            use_container_width=True)

        meilleure_vol = tableau_opt["Volatilité (%)"].idxmin()
        meilleur_sharpe = tableau_opt["Sharpe"].idxmax()
        meilleure_perf = tableau_opt["Perf. annualisée (%)"].idxmax()

        v = st.columns(3)
        v[0].metric("Risque le plus faible", meilleure_vol,
                    fmt(tableau_opt.loc[meilleure_vol, "Volatilité (%)"], 1, " %"))
        v[1].metric("Meilleur rendement", meilleure_perf,
                    fmt(tableau_opt.loc[meilleure_perf, "Perf. annualisée (%)"],
                        1, " %"))
        v[2].metric("Meilleur rapport", meilleur_sharpe,
                    fmt(tableau_opt.loc[meilleur_sharpe, "Sharpe"], 2))

        if meilleure_perf != meilleur_sharpe:
            st.info(
                f"L'allocation la plus performante ({meilleure_perf}) n'est "
                f"pas celle au meilleur rapport rendement-risque "
                f"({meilleur_sharpe}). Le rendement supplémentaire a été payé "
                "par du risque supplémentaire — reste à savoir si tu l'aurais "
                "supporté.")

        # --- Courbes de progression
        st.markdown("**Progression comparée**")
        st.caption(
            "La trajectoire de chaque allocation sur la période, à partir du "
            "même capital. Ce n'est pas le point d'arrivée qui compte le "
            "plus, mais le chemin : deux courbes finissant au même niveau "
            "n'ont pas été également supportables."
        )

        a_comparer = st.multiselect(
            "Allocations affichées", list(allocations),
            default=[m for m in ("Ma composition", "Variance minimale",
                                 "Sharpe maximal", "Équipondéré")
                     if m in allocations],
            key="comparer_alloc")

        if a_comparer:
            courbes = {}
            for nom in a_comparer:
                poids_c = allocations[nom].reindex(tickers_sim).fillna(0)
                if poids_c.sum() <= 0:
                    continue
                poids_c = poids_c / poids_c.sum()

                if rebalancement == "Aucun":
                    parts_c = (capital_sim * poids_c) / cours_sim[tickers_sim].iloc[0]
                    courbes[nom] = (cours_sim[tickers_sim] * parts_c).sum(axis=1)
                else:
                    frequence_c = 63 if rebalancement == "Trimestriel" else 252
                    serie_c = pd.Series(index=rdt_sim.index, dtype=float)
                    courants_c = poids_c.copy()
                    for i, (date_c, ligne_c) in enumerate(rdt_sim.iterrows()):
                        serie_c.loc[date_c] = float((courants_c * ligne_c).sum())
                        courants_c = courants_c * (1 + ligne_c)
                        courants_c /= courants_c.sum()
                        if (i + 1) % frequence_c == 0:
                            courants_c = poids_c.copy()
                    courbes[nom] = capital_sim * (1 + serie_c).cumprod()

            if courbes:
                st.line_chart(pd.DataFrame(courbes), height=360)
                st.caption(f"Valeur d'un capital de {capital_sim:,.0f} "
                           f"{devise_base}.".replace(",", " "))

                st.markdown("**Pertes depuis le plus haut**")
                st.caption("Le creux le plus profond est ce qui décide si une "
                           "allocation est tenable dans la durée.")
                pertes = pd.DataFrame({
                    nom: an.courbe_drawdown(serie.pct_change().dropna()) * 100
                    for nom, serie in courbes.items()})
                st.line_chart(pertes, height=280)

        # --- Risque à gauche, rendement à droite
        st.markdown("**Risque et rendement, méthode par méthode**")
        st.caption(
            "Le risque s'étend vers la gauche, le rendement vers la droite. "
            "Une barre longue d'un côté sans contrepartie de l'autre signale "
            "un mauvais échange."
        )

        deux_cotes = pd.DataFrame({
            "Risque (volatilité)": -tableau_opt["Volatilité (%)"],
            "Rendement annualisé": tableau_opt["Perf. annualisée (%)"],
        }).sort_values("Rendement annualisé")
        st.bar_chart(deux_cotes, height=320, horizontal=True)
        st.caption(
            "Le risque est porté en négatif pour qu'il s'affiche à gauche : "
            "il s'agit bien d'une volatilité positive."
        )

        pertes_cotes = pd.DataFrame({
            "Perte maximale": tableau_opt["Perte max (%)"],
            "Rendement annualisé": tableau_opt["Perf. annualisée (%)"],
        }).sort_values("Rendement annualisé")
        st.bar_chart(pertes_cotes, height=320, horizontal=True)
        st.caption(
            "Même lecture avec la perte maximale, déjà négative. C'est le "
            "chiffre qui détermine si une allocation se tient jusqu'au bout, "
            "davantage que la volatilité."
        )

        st.markdown("**Détail d'une répartition**")
        detail = st.selectbox("Méthode", list(allocations), key="detail_alloc")
        st.caption(explications.get(detail, ""))
        repartition = (allocations[detail].reindex(tickers_sim).fillna(0) * 100)
        st.bar_chart(repartition.rename("Poids (%)"), height=240)

        if st.button("Reprendre cette allocation dans la simulation"):
            st.session_state.simulation = pd.DataFrame({
                "Ticker": tickers_sim,
                "Poids (%)": [float(repartition[t]) for t in tickers_sim]})
            st.rerun()

        # --- Frontière efficiente
        st.markdown("**Frontière efficiente**")
        st.caption(
            "Chaque point est un portefeuille possible. La courbe joint ceux "
            "qui offrent le meilleur rendement pour un risque donné — sur la "
            "période observée uniquement."
        )
        try:
            frontiere = opt.frontiere_efficiente(mu_opt, cov_opt, points=30)
            if not frontiere.empty:
                nuage = pd.DataFrame({
                    "Volatilité (%)": frontiere["volatilite"] * 100,
                    "Frontière": frontiere["rendement"] * 100}).set_index(
                        "Volatilité (%)")
                positions = pd.DataFrame({
                    "Volatilité (%)": tableau_opt["Volatilité (%)"],
                    "Allocations": tableau_opt["Perf. annualisée (%)"]}
                    ).set_index("Volatilité (%)")
                st.line_chart(nuage.join(positions, how="outer").sort_index(),
                              height=320)
                st.caption(
                    "Les points « Allocations » situent chaque méthode. Une "
                    "allocation nettement sous la courbe laisse du rendement "
                    "sur la table pour le même risque — mais souviens-toi que "
                    "la courbe elle-même est tracée en connaissant le passé."
                )
        except Exception as erreur:
            st.caption(f"Frontière non calculable : {erreur}")

        st.markdown("**Lecture de ces allocations**")
        detail_poids = pd.DataFrame(
            {nom: p.reindex(tickers_sim).fillna(0) * 100
             for nom, p in allocations.items()}).round(1)
        ia.bloc("Allocations optimales",
                {"mesures": tableau_opt.round(2).to_dict("index"),
                 "poids_par_methode": detail_poids.to_dict(),
                 "correlation_moyenne": round(float(correlation_moyenne), 2),
                 "periode": periode_sim,
                 "rebalancement": rebalancement},
                f"Comparaison de {len(comparaison)} répartitions des mêmes "
                f"{len(tickers_sim)} valeurs sur {periode_sim}. Les "
                "optimisations connaissent le résultat de la période : leur "
                "performance passée est flatteuse par construction.",
                "allocations_sim", ouvert=True, automatique=True,
                consignes=ia.CONSIGNES_PORTEFEUILLE)

with onglets[4]:
    _onglet_4()



def _onglet_5():
    st.subheader("Stratégies options")
    st.caption(
        "Quatre structures classiques, calculées sur les chaînes d'options "
        "réelles. L'outil mesure le profil de gain, les seuils et le "
        "rendement — il ne recommande aucune position."
    )

    st.warning(
        "Trois points à connaître avant de commencer. Les options ne sont pas "
        "éligibles au PEA : compte-titres uniquement, primes imposées au "
        "prélèvement forfaitaire. Un contrat porte sur 100 titres, donc une "
        "position minimale de plusieurs milliers d'euros. Et les chaînes "
        "d'options ne sont diffusées que pour les valeurs américaines.",
        icon="⚠️")

    reglages = st.columns([2, 2, 1])
    ticker_op = reglages[0].text_input(
        "Valeur", value="AAPL", key="ticker_options",
        help="Valeurs américaines uniquement.").strip().upper()

    if not ticker_op:
        return

    liste_echeances = op.echeances(ticker_op)
    if not liste_echeances:
        st.error(
            f"Aucune chaîne d'options pour {ticker_op}. C'est le cas de la "
            "quasi-totalité des valeurs européennes : Yahoo ne diffuse pas "
            "leurs options. Essaie une valeur américaine.")
        return

    echeance = reglages[1].selectbox(
        "Échéance", liste_echeances,
        index=min(3, len(liste_echeances) - 1), key="echeance_options")
    contrats = reglages[2].number_input(
        "Contrats", min_value=1, max_value=50, value=1, key="contrats_options",
        help="Un contrat = 100 titres.")

    with st.spinner("Chargement de la chaîne…"):
        donnees = op.chaine(ticker_op, echeance)

    if not donnees or donnees["calls"].empty:
        st.error("Chaîne illisible pour cette échéance.")
        return

    spot = donnees["spot"]
    jours = donnees["jours"]
    quantite = int(contrats) * 100

    entete = st.columns(4)
    entete[0].metric("Cours", fmt(spot, 2))
    entete[1].metric("Échéance", echeance)
    entete[2].metric("Jours restants", jours)
    entete[3].metric("Titres engagés", f"{quantite}")

    choix = st.selectbox(
        "Stratégie",
        list(op.STRATEGIES),
        format_func=lambda c: op.STRATEGIES[c]["nom"],
        key="strategie_options")
    fiche = op.STRATEGIES[choix]

    with st.expander("Comment fonctionne cette stratégie", expanded=False):
        st.markdown(f"**{fiche['resume']}**")
        st.markdown(
            f"- **Tendance visée** : {fiche['tendance']}\n"
            f"- **Gain maximal** : {fiche['gain_max']}\n"
            f"- **Risque maximal** : {fiche['risque_max']}\n"
            f"- **Volatilité** : {fiche['volatilite']}\n"
            f"- **Quand l'employer** : {fiche['quand']}")
        st.warning(f"**Le piège** — {fiche['piege']}")

    calls, puts = donnees["calls"], donnees["puts"]
    jambes = {}

    if choix == "covered_call":
        ecart = st.slider("Strike du call vendu, en % au-dessus du cours",
                          0, 30, 5, key="ecart_cc")
        ligne = op.proche(calls, spot * (1 + ecart / 100))
        if ligne is None:
            st.error("Aucun strike exploitable.")
            return
        jambes = {"call_strike": float(ligne["strike"]),
                  "call_prime": float(ligne["prix"])}
        st.dataframe(pd.DataFrame([{
            "Jambe": "Call vendu", "Strike": ligne["strike"],
            "Prime": round(float(ligne["prix"]), 2),
            "Écart cours (%)": round(float(ligne["moneyness_pct"]), 1),
            "Vol. implicite (%)": round(float(ligne.get("impliedVolatility", 0)) * 100, 1),
            "Fourchette (%)": round(float(ligne.get("ecart_pct", np.nan)), 1),
            "Intérêt ouvert": int(ligne.get("openInterest", 0) or 0),
        }]), use_container_width=True, hide_index=True)

    elif choix == "cash_secured_put":
        ecart = st.slider("Strike du put vendu, en % sous le cours",
                          0, 30, 5, key="ecart_csp")
        ligne = op.proche(puts, spot * (1 - ecart / 100))
        if ligne is None:
            st.error("Aucun strike exploitable.")
            return
        jambes = {"put_strike": float(ligne["strike"]),
                  "put_prime": float(ligne["prix"])}
        st.dataframe(pd.DataFrame([{
            "Jambe": "Put vendu", "Strike": ligne["strike"],
            "Prime": round(float(ligne["prix"]), 2),
            "Écart cours (%)": round(float(ligne["moneyness_pct"]), 1),
            "Vol. implicite (%)": round(float(ligne.get("impliedVolatility", 0)) * 100, 1),
            "Intérêt ouvert": int(ligne.get("openInterest", 0) or 0),
        }]), use_container_width=True, hide_index=True)

    elif choix == "collar":
        deux = st.columns(2)
        haut = deux[0].slider("Call vendu, % au-dessus", 0, 30, 8, key="collar_h")
        bas = deux[1].slider("Put acheté, % en dessous", 0, 30, 8, key="collar_b")
        ligne_call = op.proche(calls, spot * (1 + haut / 100))
        ligne_put = op.proche(puts, spot * (1 - bas / 100))
        if ligne_call is None or ligne_put is None:
            st.error("Strikes indisponibles.")
            return
        jambes = {"call_strike": float(ligne_call["strike"]),
                  "call_prime": float(ligne_call["prix"]),
                  "put_strike": float(ligne_put["strike"]),
                  "put_prime": float(ligne_put["prix"])}
        st.dataframe(pd.DataFrame([
            {"Jambe": "Call vendu", "Strike": ligne_call["strike"],
             "Prime": round(float(ligne_call["prix"]), 2)},
            {"Jambe": "Put acheté", "Strike": ligne_put["strike"],
             "Prime": round(-float(ligne_put["prix"]), 2)},
        ]), use_container_width=True, hide_index=True)
        net = jambes["call_prime"] - jambes["put_prime"]
        (st.success if net >= 0 else st.info)(
            f"Prime nette de {net:+.2f} par titre, soit "
            f"{net * quantite:+,.0f} {devise_base}. "
            .replace(",", " ")
            + ("Le call finance intégralement la protection."
               if net >= 0 else "La protection coûte davantage que la prime."))

    else:  # iron condor
        deux = st.columns(2)
        largeur_interne = deux[0].slider("Ailes vendues, % du cours",
                                         2, 20, 6, key="ic_interne")
        largeur_aile = deux[1].slider("Largeur des ailes, % du cours",
                                      1, 10, 3, key="ic_aile")
        pv = op.proche(puts, spot * (1 - largeur_interne / 100))
        pa = op.proche(puts, spot * (1 - (largeur_interne + largeur_aile) / 100))
        cv = op.proche(calls, spot * (1 + largeur_interne / 100))
        ca = op.proche(calls, spot * (1 + (largeur_interne + largeur_aile) / 100))
        if any(x is None for x in (pv, pa, cv, ca)):
            st.error("Strikes indisponibles pour cette configuration.")
            return
        jambes = {
            "put_vendu": float(pv["strike"]), "put_vendu_prime": float(pv["prix"]),
            "put_achete": float(pa["strike"]), "put_achete_prime": float(pa["prix"]),
            "call_vendu": float(cv["strike"]), "call_vendu_prime": float(cv["prix"]),
            "call_achete": float(ca["strike"]), "call_achete_prime": float(ca["prix"])}
        st.dataframe(pd.DataFrame([
            {"Jambe": "Put acheté", "Strike": pa["strike"],
             "Prime": round(-float(pa["prix"]), 2)},
            {"Jambe": "Put vendu", "Strike": pv["strike"],
             "Prime": round(float(pv["prix"]), 2)},
            {"Jambe": "Call vendu", "Strike": cv["strike"],
             "Prime": round(float(cv["prix"]), 2)},
            {"Jambe": "Call acheté", "Strike": ca["strike"],
             "Prime": round(-float(ca["prix"]), 2)},
        ]), use_container_width=True, hide_index=True)

    # --- Mesures et profil
    resultat = op.profil(choix, spot, jambes, quantite)
    rendement = op.rendements(choix, spot, jambes, jours, quantite)
    if not resultat:
        return

    st.divider()
    m = st.columns(4)
    m[0].metric("Prime encaissée",
                fmt(rendement["prime_encaissee"], 0, f" {devise_base}"))
    m[1].metric("Gain maximal",
                fmt(resultat["gain_max"], 0, f" {devise_base}"))
    m[2].metric("Perte maximale",
                fmt(resultat["perte_max"], 0, f" {devise_base}"),
                help="À l'échéance, dans le pire cas envisagé par le profil.")
    m[3].metric("Capital immobilisé",
                fmt(rendement["capital_immobilise"], 0, f" {devise_base}"))

    r = st.columns(3)
    r[0].metric(f"Rendement sur {jours} jours",
                fmt(rendement["rendement_periode_pct"], 2, " %"))
    r[1].metric("Annualisé",
                fmt(rendement["rendement_annualise_pct"], 1, " %"),
                help="Suppose de reproduire l'opération à l'identique toute "
                     "l'année, ce que la volatilité ne garantit jamais. À "
                     "lire comme une base de comparaison entre échéances.")
    if choix == "covered_call":
        r[2].metric("Si assigné",
                    fmt(rendement["rendement_si_assigne_pct"], 2, " %"),
                    help="Rendement total si le titre est cédé au strike.")
    elif isinstance(resultat["seuil"], tuple):
        r[2].metric("Fourchette de gain",
                    f"{resultat['seuil'][0]:.1f} – {resultat['seuil'][1]:.1f}")
    else:
        r[2].metric("Seuil de rentabilité", fmt(resultat["seuil"], 2))

    profil_gain = pd.DataFrame(
        {"Gain à l'échéance": resultat["gain"]},
        index=pd.Index(resultat["cours"].round(2), name="Cours"))
    st.line_chart(profil_gain, height=320)
    st.caption(
        f"Gain ou perte à l'échéance du {echeance}, selon le cours de "
        f"{ticker_op}. Position de {quantite} titres. Le cours actuel est "
        f"de {spot:.2f}."
    )

    if choix == "covered_call":
        renonce = max(0.0, (spot * 1.3 - jambes["call_strike"]) * quantite)
        if renonce > 0:
            st.info(
                f"Si {ticker_op} gagnait 30 % d'ici l'échéance, tu "
                f"renoncerais à {renonce:,.0f} {devise_base} de plus-value "
                f"pour une prime de {rendement['prime_encaissee']:,.0f}. "
                "C'est l'arbitrage central de cette stratégie."
                .replace(",", " "))

    ia.bloc(f"Stratégie {fiche['nom']} sur {ticker_op}",
            {**jambes, **{k: v for k, v in rendement.items()
                          if isinstance(v, (int, float))},
             "cours": spot, "gain_max": resultat["gain_max"],
             "perte_max": resultat["perte_max"], "jours": jours},
            f"{fiche['nom']} sur {ticker_op}, échéance {echeance}, "
            f"{quantite} titres.", "options")


with onglets[5]:
    _onglet_5()
