"""
Regression sur les cinq modules recents.

Chaque verification faite a la main pendant l'ecriture de ces modules est
rejouee ici. C'est tout l'objet du fichier : un controle qu'on ne refait pas
automatiquement n'a servi qu'une fois, et rien ne signalera le jour ou une
modification cassera un calcul juste.

Les valeurs attendues sont posees en clair, calculees independamment du code
teste. Comparer une fonction au resultat d'une autre fonction du meme fichier
ne prouve que leur coherence mutuelle, pas leur justesse.

Lancement :  python3 test_modules.py
ou           python3 -m pytest test_modules.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import dimension as dm
import mouvements as mo
import niveaux as nv
import performance as pf
import reglages as rg
import validation as val


def proche(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def journal(operations) -> pd.DataFrame:
    """Construit un journal a partir de tuples (date, ticker, sens, q, prix, frais)."""
    return mo.lire(pd.DataFrame([
        {"Date": d, "Ticker": t, "Sens": s, "Quantité": q,
         "Prix unitaire": p, "Devise": "USD", "Frais": f}
        for d, t, s, q, p, f in operations]))


# ==========================================================================
# mouvements — prix de revient et plus-values
# ==========================================================================

NVDA = [
    ("12/01/2025", "NVDA", "ACHAT", 20, 400.0, 5),
    ("03/03/2025", "NVDA", "ACHAT", 10, 460.0, 5),
    ("10/06/2025", "NVDA", "DIVISION", 4, None, 0),
    ("15/09/2025", "NVDA", "VENTE", 40, 140.0, 5),
    ("20/10/2025", "NVDA", "VENTE", 40, 155.0, 5),
    ("05/12/2025", "NVDA", "VENTE", 40, 170.0, 5),
    ("14/03/2026", "NVDA", "ACHAT", 50, 120.0, 5),
]


def test_pru_moyen_pondere():
    """20 x 400 + 5 puis 10 x 460 + 5 = 12 610 pour 30 titres, soit 420,3333."""
    etat, _ = mo.derouler(journal(NVDA[:2]))
    assert proche(etat.at["NVDA", "PRU"], 12610 / 30, 1e-10)
    assert proche(etat.at["NVDA", "Investi"], 12610.0, 1e-10)


def test_division_multiplie_la_quantite_sans_toucher_au_cout():
    """30 titres a 420,3333 deviennent 120 titres a 105,0833."""
    etat, _ = mo.derouler(journal(NVDA[:3]))
    assert proche(etat.at["NVDA", "Quantité"], 120.0)
    assert proche(etat.at["NVDA", "PRU"], 12610 / 120, 1e-10)
    assert proche(etat.at["NVDA", "Investi"], 12610.0, 1e-10)


def test_ventes_en_tranches_au_meme_pru():
    """Le prix de revient ne bouge pas entre deux tranches vendues."""
    pru = 12610 / 120
    piste = mo.piste(journal(NVDA[:6]), "NVDA")
    ventes = piste[piste["Sens"] == "VENTE"]
    assert len(ventes) == 3
    for _, v in ventes.iterrows():
        assert proche(v["PRU appliqué"], pru, 1e-10)
    attendus = [40 * (140 - pru) - 5, 40 * (155 - pru) - 5, 40 * (170 - pru) - 5]
    for obtenu, attendu in zip(ventes["Résultat"], attendus):
        assert proche(obtenu, attendu, 1e-9)


def test_cumul_des_plus_values_et_rachat():
    """Total 6 972,50, et le rachat repart sur une base propre."""
    etat, _ = mo.derouler(journal(NVDA + [
        ("02/04/2026", "NVDA", "ACHAT", 50, 100.0, 0),
        ("02/04/2026", "NVDA", "VENTE", 50, 130.0, 0)]))
    assert proche(etat.at["NVDA", "PV réalisée"], 6972.50, 1e-9)
    assert proche(etat.at["NVDA", "Quantité"], 50.0)
    assert proche(etat.at["NVDA", "PRU"], 110.05, 1e-10)


def test_position_soldee_remet_le_cout_a_zero():
    etat, _ = mo.derouler(journal(NVDA[:6]))
    assert proche(etat.at["NVDA", "Quantité"], 0.0)
    assert proche(etat.at["NVDA", "Investi"], 0.0)


def test_vente_superieure_a_la_position_est_ramenee():
    etat, anomalies = mo.derouler(journal([
        ("05/01/2026", "MU", "ACHAT", 20, 90.0, 0),
        ("06/01/2026", "MU", "VENTE", 25, 120.0, 0)]))
    assert proche(etat.at["MU", "PV réalisée"], 20 * (120 - 90), 1e-9)
    assert any("ramenee" in a for a in anomalies)


def test_vente_sur_position_soldee_est_ignoree():
    etat, anomalies = mo.derouler(journal([
        ("07/01/2026", "MU", "VENTE", 5, 130.0, 0)]))
    assert proche(etat.at["MU", "PV réalisée"], 0.0)
    assert any("soldee" in a or "jamais achetee" in a for a in anomalies)


def test_lignes_illisibles_rejetees_pas_devinees():
    j = mo.lire(pd.DataFrame([
        {"Date": "04/04/2026", "Ticker": "MU", "Sens": "ACHT",
         "Quantité": 10, "Prix unitaire": 90, "Devise": "USD", "Frais": 0},
        {"Date": "", "Ticker": "KO", "Sens": "ACHAT",
         "Quantité": 5, "Prix unitaire": 70, "Devise": "USD", "Frais": 0},
        {"Date": "05/04/2026", "Ticker": "KO", "Sens": "ACHAT",
         "Quantité": 5, "Prix unitaire": 70, "Devise": "USD", "Frais": 0}]))
    assert len(j) == 1
    assert len(j.attrs["rejets"]) == 2


def test_doublons_signales_sans_etre_supprimes():
    j = journal([("05/01/2026", "MU", "ACHAT", 10, 90.07, 0),
                 ("05/01/2026", "MU", "ACHAT", 10, 90.07, 0)])
    assert len(j) == 2
    assert len(mo.doublons(j)) == 1
    etat, _ = mo.derouler(j)
    assert proche(etat.at["MU", "Quantité"], 20.0)


def test_virgule_decimale_francaise():
    j = mo.lire(pd.DataFrame([
        {"Date": "01/01/2026", "Ticker": "X", "Sens": "ACHAT", "Quantité": 10,
         "Prix unitaire": "1 234,56", "Devise": "EUR", "Frais": "2,50"}]))
    assert proche(j.at[0, "Prix unitaire"], 1234.56, 1e-9)
    assert proche(j.at[0, "Frais"], 2.50, 1e-9)


# ==========================================================================
# niveaux — ATR et stops
# ==========================================================================

def test_amplitude_vraie_calculee_a_la_main():
    h = pd.Series([10.0, 10.5, 11.0, 10.8, 12.0, 11.5, 11.2])
    b = pd.Series([9.5, 9.8, 10.2, 10.0, 10.9, 10.8, 10.5])
    c = pd.Series([9.8, 10.3, 10.6, 10.4, 11.8, 11.0, 10.9])
    # max(h-b, |h-c_veille|, |b-c_veille|) seance par seance
    attendu = [0.5, 0.7, 0.8, 0.8, 1.6, 1.0, 0.7]
    ref = [sum(attendu[:3]) / 3]
    for i in range(3, 7):
        ref.append(ref[-1] + (attendu[i] - ref[-1]) / 3)
    obtenu = list(nv.atr(h, b, c, n=3).dropna())
    assert len(obtenu) == len(ref)
    for a, r in zip(obtenu, ref):
        assert proche(a, r, 1e-12)


def test_atr_amorce_sur_la_moyenne_des_n_premieres():
    """Une moyenne exponentielle ordinaire s'amorcerait sur la premiere valeur."""
    h = pd.Series([11.0] * 20)
    b = pd.Series([10.0] * 20)
    c = pd.Series([10.5] * 20)
    # Amplitude constante de 1,0 : l'ATR doit valoir exactement 1,0
    assert proche(float(nv.atr(h, b, c, n=5).dropna().iloc[0]), 1.0, 1e-12)


def bougies(vol, base, derive, n=300, graine=4):
    rng = np.random.default_rng(graine)
    c = base * np.exp(np.cumsum(rng.normal(derive, vol, n)))
    o = np.concatenate([[base], c[:-1]])
    m = np.abs(rng.normal(0, vol * 0.4, n))
    return pd.DataFrame({"High": np.maximum(o, c) * (1 + m),
                         "Low": np.minimum(o, c) * (1 - m), "Close": c})


def test_marge_positive_quand_le_cours_est_au_dessus_du_stop():
    t = nv.niveaux(bougies(0.012, 70, 0.0007))
    for _, r in t.iterrows():
        assert (r["Marge (%)"] > 0) == (not r["Franchi"])


def test_stop_egal_au_plus_haut_moins_k_atr():
    d = bougies(0.015, 100, 0.0005)
    a = float(nv.atr(d["High"], d["Low"], d["Close"]).dropna().iloc[-1])
    t = nv.niveaux(d)
    for _, r in t.iterrows():
        p = nv.HORIZONS[r["Horizon"]]
        attendu = float(d["High"].tail(p["fenetre"]).max()) - p["atr"] * a
        assert proche(r["Stop"], attendu, 1e-9)


def chute_apres_sommet() -> pd.DataFrame:
    """
    Serie deterministe : montee jusqu'a 150, puis chute a 80.

    Une marche aleatoire, meme a derive negative, peut remonter sur ses
    dernieres seances et repasser au-dessus de son stop — un premier essai de
    ce test echouait pour cette raison, sans que le code soit en cause. Une
    trajectoire imposee rend le resultat certain.
    """
    c = np.concatenate([np.linspace(100, 150, 150), np.linspace(150, 80, 150)])
    return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c})


def test_objectif_absent_si_stop_franchi():
    t = nv.niveaux(chute_apres_sommet())
    franchis = t[t["Franchi"]]
    assert len(franchis) == 3           # les trois horizons sont dépassés
    assert franchis["Objectif"].isna().all()
    assert franchis["Potentiel (%)"].isna().all()
    assert (franchis["Marge (%)"] < 0).all()


def test_objectif_est_un_multiple_du_risque():
    t = nv.niveaux(bougies(0.012, 70, 0.0007))
    for _, r in t.iterrows():
        if r["Franchi"]:
            continue
        assert proche(r["Gain visé (%)"],
                      nv.HORIZONS[r["Horizon"]]["gain"] * r["Risque (%)"], 1e-9)


def test_stop_capital_calcule_sur_le_prix_de_revient():
    """PRU 100 et 15 % de perte acceptée : le stop de capital vaut 85."""
    d = bougies(0.012, 70, 0.0007)
    t = nv.niveaux(d, pru=100.0, perte_capital=15.0)
    for _, r in t.iterrows():
        assert proche(r["Stop capital"], 85.0, 1e-9)


def test_seuil_retenu_est_le_plus_haut_des_deux():
    """À la baisse, c'est le seuil le plus haut qui est touché en premier."""
    d = bougies(0.012, 70, 0.0007)
    t = nv.niveaux(d, pru=140.0, perte_capital=15.0)
    for _, r in t.iterrows():
        assert proche(r["Stop"], max(r["Stop marché"], r["Stop capital"]), 1e-9)
        attendue = ("capital" if r["Stop capital"] > r["Stop marché"]
                    else "marché")
        assert r["Origine"] == attendue


def test_sans_prix_de_revient_seul_le_stop_de_marche_existe():
    t = nv.niveaux(bougies(0.012, 70, 0.0007))
    for _, r in t.iterrows():
        assert np.isnan(r["Stop capital"])
        assert r["Origine"] == "marché"
        assert proche(r["Stop"], r["Stop marché"], 1e-12)
        assert np.isnan(r["Résultat au stop (%)"])


def test_resultat_au_stop_rapporte_au_prix_de_revient():
    """Quand le stop de capital l'emporte, la perte vaut exactement le budget."""
    d = bougies(0.012, 70, 0.0007)
    t = nv.niveaux(d, pru=200.0, perte_capital=15.0)
    capitaux = t[t["Origine"] == "capital"]
    assert len(capitaux) == 3          # PRU très au-dessus : le capital mord
    for _, r in capitaux.iterrows():
        assert proche(r["Résultat au stop (%)"], -15.0, 1e-9)


def test_marge_et_resultat_au_stop_ont_des_references_differentes():
    """L'une se rapporte au cours, l'autre au prix de revient."""
    d = bougies(0.012, 70, 0.0007)
    actuel = float(d["Close"].iloc[-1])
    pru = actuel * 0.60                 # acheté bien plus bas
    t = nv.niveaux(d, pru=pru, perte_capital=15.0)
    r = t.iloc[0]
    assert proche(r["Marge (%)"], (1 - r["Stop"] / actuel) * 100, 1e-9)
    assert proche(r["Résultat au stop (%)"], (r["Stop"] / pru - 1) * 100, 1e-9)
    # Une ligne largement gagnante : le stop reste au-dessus du prix payé
    assert r["Résultat au stop (%)"] > 0


def test_historique_trop_court_ne_produit_rien():
    assert nv.niveaux(pd.DataFrame()).empty
    assert nv.niveaux(pd.DataFrame({"High": [1, 2], "Low": [1, 2],
                                    "Close": [1, 2]})).empty


def test_cours_plat_ne_produit_pas_de_stop():
    plat = pd.DataFrame({"High": [10.0] * 200, "Low": [10.0] * 200,
                         "Close": [10.0] * 200})
    assert nv.niveaux(plat).empty


# ==========================================================================
# validation — controles de saisie
# ==========================================================================

def historique(base, n=250, vol=0.02, graine=3):
    rng = np.random.default_rng(graine)
    return pd.Series(base * np.exp(np.cumsum(rng.normal(0, vol, n))),
                     index=pd.bdate_range("2025-10-01", periods=n))


def test_devise_contredite_par_la_place():
    a = val.controler_devise("ALKAL.PA", "USD", "EUR")
    assert len(a) == 1 and a[0]["gravite"] == "bloquant"


def test_devise_conforme_ne_declenche_rien():
    assert val.controler_devise("AAPL", "USD", "USD") == []


def test_cotation_en_centiemes_detectee():
    a = val.controler_devise("FRMI.L", "GBP", "GBp")
    assert any(x["titre"] == "Cotation en centièmes" for x in a)


def test_montant_total_pris_pour_un_prix_unitaire():
    c = historique(50.0)
    a = val.controler_prix("AMD", 493.95, pd.Timestamp("2026-01-15"), c, 10.0)
    assert len(a) == 1 and a[0]["gravite"] == "bloquant"
    assert "divisant par la quantité" in a[0]["detail"]


def test_prix_juste_ne_declenche_rien():
    c = historique(50.0)
    milieu = float(c.median())
    assert val.controler_prix("AMD", milieu, pd.Timestamp("2026-01-15"),
                              c, 10.0) == []


def test_prix_reel_mais_date_fausse_est_serieux_pas_bloquant():
    """Un cours que le titre a vraiment coté n'est pas une erreur de saisie."""
    c = historique(50.0, vol=0.03)
    extreme = float(c.max())
    jour = pd.Timestamp(c.idxmin())          # date ou le cours etait au plus bas
    a = val.controler_prix("X", extreme, jour, c, 1.0)
    assert len(a) == 1
    assert a[0]["gravite"] == "serieux"
    assert a[0]["titre"] == "Date probablement erronée"


def test_ticker_sans_cours_signale():
    a = val.controler_positions(
        pd.DataFrame({"Devise": ["USD"]}, index=["XXXX"]),
        pd.Series({"XXXX": 100.0}), set())
    assert any(x["titre"] == "Aucun cours trouvé" for x in a)


def test_concentration_au_dela_du_seuil():
    a = val.controler_positions(
        pd.DataFrame({"Devise": ["EUR", "EUR"]}, index=["A", "B"]),
        pd.Series({"A": 8000.0, "B": 2000.0}), {"A", "B"})
    concentres = [x for x in a if "Concentration" in x["titre"]]
    assert len(concentres) == 1 and concentres[0]["ticker"] == "A"


def test_seuil_de_concentration_vient_des_reglages():
    assert val.CONCENTRATION_ALERTE == rg.CONCENTRATION


# ==========================================================================
# dimension — taille des positions
# ==========================================================================

def test_taille_calculee_a_la_main():
    """1 % de 29 908 = 299,08 ; perte unitaire 28,88 - 26,94 = 1,94."""
    r = dm.taille(29908.0, 28.88, 26.94, 1.0)
    assert r["titres"] == int(299.08 / 1.94)
    assert proche(r["montant"], int(299.08 / 1.94) * 28.88, 1e-9)
    assert r["limite"] == "risque"


def test_stop_serre_declenche_le_plafond_de_taille():
    r = dm.taille(29908.0, 326.94, 320.55, 1.0, 20.0)
    assert r["limite"] == "taille"
    assert r["part"] <= 20.0 + 1e-9


def test_perte_proportionnelle_au_budget():
    a = dm.taille(100000.0, 100.0, 90.0, 1.0)
    b = dm.taille(100000.0, 100.0, 90.0, 2.0)
    assert proche(b["titres"], 2 * a["titres"], 1.0)


def test_parametres_invalides_ne_produisent_rien():
    for cours, stop in ((100.0, 100.0), (100.0, 110.0), (100.0, 0.0), (0.0, 90.0)):
        assert dm.taille(10000.0, cours, stop, 1.0)["titres"] == 0


def test_budget_respecte_plafonne_la_cible():
    pos = pd.DataFrame({"Valeur": [1725.0], "Cours": [326.94], "Stop": [320.55]},
                       index=["CDNS"])
    b = dm.budget_respecte(pos, 1.0, 20.0)
    assert proche(b.at["CDNS", "Valeur cible"], 1725.0 * 0.20, 1e-9)


def test_ligne_franchie_sans_budget():
    pos = pd.DataFrame({"Valeur": [3157.0], "Cours": [3.59], "Stop": [4.12]},
                       index=["NIO"])
    b = dm.budget_respecte(pos, 1.0)
    assert bool(b.at["NIO", "Franchi"])
    assert np.isnan(b.at["NIO", "Risque (%)"])


def test_kelly_refuse_une_esperance_defavorable():
    k = dm.kelly(0.30, 1.0, 1.0)
    assert not k["favorable"]
    assert proche(k["fractionne"], 0.0)


def test_kelly_formule():
    """f = p - (1 - p) / R, avec R = gain / perte."""
    k = dm.kelly(0.55, 2.0, 1.0)
    assert proche(k["complet"], (0.55 - 0.45 / 2.0) * 100, 1e-9)
    assert proche(k["fractionne"], k["complet"] * dm.FRACTION_KELLY, 1e-9)


# ==========================================================================
# performance — TWR, TRI, contrefactuel
# ==========================================================================

def scenario_doublement_puis_retour():
    """Achat a 10, le cours double, renforcement au plus haut, retour a 10."""
    dates = pd.bdate_range("2025-01-01", periods=262)
    prix = pd.Series(np.concatenate([np.linspace(10, 20, 131),
                                     np.linspace(20, 10, 131)]), index=dates)
    j = journal([(dates[0].strftime("%d/%m/%Y"), "X", "ACHAT", 100, 10.0, 0),
                 (dates[130].strftime("%d/%m/%Y"), "X", "ACHAT", 100, 20.0, 0)])
    q = pf.quantites_quotidiennes(j, dates)
    v = pf.valeur_quotidienne(q, pd.DataFrame({"X": prix}))
    return dates, j, v, pf.flux_externes(j)


def test_twr_neutralise_les_versements():
    """+100 % puis -50 % : 2,0 x 0,5 = 1,0, donc zero."""
    _, _, v, f = scenario_doublement_puis_retour()
    assert proche(pf.twr(v, f)["total"], 0.0, 1e-9)


def test_tri_penalise_un_renforcement_au_mauvais_moment():
    dates, _, v, f = scenario_doublement_puis_retour()
    r = pf.tri(f, float(v.iloc[-1]), dates[-1])
    assert r < -0.30        # 3 000 versés, 2 000 à l'arrivée
    # Verification independante : la valeur actuelle doit s'annuler a ce taux
    origine = f.index.min()
    va = sum(m / (1 + r) ** ((pd.Timestamp(d) - origine).days / pf.JOURS_AN)
             for d, m in f.items())
    va -= float(v.iloc[-1]) / (1 + r) ** ((dates[-1] - origine).days / pf.JOURS_AN)
    assert abs(va) < 1e-4


def test_quantites_reconstituees_jour_par_jour():
    d = pd.bdate_range("2025-01-01", periods=200)
    j = journal([(d[0].strftime("%d/%m/%Y"), "X", "ACHAT", 100, 10.0, 0),
                 (d[99].strftime("%d/%m/%Y"), "X", "VENTE", 100, 15.0, 0),
                 (d[150].strftime("%d/%m/%Y"), "X", "ACHAT", 50, 12.0, 0)])
    q = pf.quantites_quotidiennes(j, d)
    assert proche(q["X"].iloc[0], 100.0)
    assert proche(q["X"].iloc[99], 0.0)
    assert proche(q["X"].iloc[120], 0.0)
    assert proche(q["X"].iloc[-1], 50.0)


def test_division_ne_cree_aucun_flux():
    d = pd.bdate_range("2025-01-01", periods=200)
    j = journal([(d[0].strftime("%d/%m/%Y"), "Y", "ACHAT", 10, 400.0, 0),
                 (d[50].strftime("%d/%m/%Y"), "Y", "DIVISION", 4, None, 0)])
    assert len(pf.flux_externes(j)) == 1
    assert proche(pf.quantites_quotidiennes(j, d)["Y"].iloc[-1], 40.0)


def test_flux_signes_et_frais():
    d = pd.bdate_range("2025-01-01", periods=10)
    j = journal([(d[0].strftime("%d/%m/%Y"), "X", "ACHAT", 100, 10.0, 5),
                 (d[5].strftime("%d/%m/%Y"), "X", "VENTE", 50, 12.0, 5)])
    f = pf.flux_externes(j)
    assert proche(f.iloc[0], 100 * 10 + 5)      # achat : argent entré, frais compris
    assert proche(f.iloc[1], -50 * 12 + 5)      # vente : argent sorti, frais payés


def test_contrefactuel_achete_des_parts_au_cours_du_jour():
    d = pd.bdate_range("2025-01-01", periods=100)
    indice = pd.Series(np.linspace(100.0, 200.0, 100), index=d)
    flux = pd.Series({d[0]: 1000.0})
    c = pf.contrefactuel(flux, indice)
    assert proche(c["parts"], 10.0, 1e-9)       # 1 000 / 100
    assert proche(c["valeur"], 2000.0, 1e-9)    # 10 x 200
    assert proche(c["rendement"], 1.0, 1e-9)


def test_contrefactuel_quotidien_finit_sur_le_total():
    """La courbe et le calcul final doivent se rejoindre au dernier point."""
    d = pd.bdate_range("2025-01-01", periods=100)
    indice = pd.Series(np.linspace(100.0, 200.0, 100), index=d)
    flux = pd.Series({d[0]: 1000.0, d[49]: 1000.0})
    courbe = pf.contrefactuel_quotidien(flux, indice, d)
    assert proche(courbe.iloc[-1], pf.contrefactuel(flux, indice)["valeur"], 1e-6)
    # Premier jour : 1 000 / 100 = 10 parts, soit 10 x 100
    assert proche(courbe.iloc[0], 1000.0, 1e-9)
    # Au versement suivant, les parts s'ajoutent au cours du jour
    p = float(indice.iloc[49])
    assert proche(courbe.iloc[49], (10 + 1000 / p) * p, 1e-9)


def test_versements_cumules_en_escalier():
    d = pd.bdate_range("2025-01-01", periods=100)
    flux = pd.Series({d[0]: 1000.0, d[49]: 500.0, d[80]: -300.0})
    v = pf.versements_cumules(flux, d)
    assert proche(v.iloc[0], 1000.0)
    assert proche(v.iloc[48], 1000.0)
    assert proche(v.iloc[49], 1500.0)
    assert proche(v.iloc[-1], 1200.0)


def test_comparaison_signale_le_sens_de_l_ecart():
    c = {"valeur": 1000.0, "investi": 800.0, "rendement": 0.25, "parts": 1.0}
    assert pf.comparaison(1200.0, c)["gagne"] is True
    assert pf.comparaison(900.0, c)["gagne"] is False


def test_duree_trop_courte_declaree_non_fiable():
    assert not pf.fiable(1.5)
    assert pf.fiable(pf.DUREE_INTERPRETABLE)


def test_entrees_vides_ne_plantent_pas():
    vide = pd.Series(dtype=float)
    assert pf.twr(pd.Series([100.0]), vide) == {}
    assert np.isnan(pf.tri(vide, 100.0, "2025-01-01"))
    assert pf.contrefactuel(vide, vide) == {}


# ==========================================================================
# Coherence entre les deux moteurs
# ==========================================================================

def test_journal_et_performance_donnent_la_meme_position_finale():
    """
    Les quantites reconstituees jour par jour doivent finir sur l'etat du
    journal. Deux chemins de calcul differents, un seul resultat possible.
    """
    d = pd.bdate_range("2025-01-01", periods=400)
    j = journal([(d[0].strftime("%d/%m/%Y"), "NVDA", "ACHAT", 20, 400.0, 5),
                 (d[40].strftime("%d/%m/%Y"), "NVDA", "ACHAT", 10, 460.0, 5),
                 (d[100].strftime("%d/%m/%Y"), "NVDA", "DIVISION", 4, None, 0),
                 (d[200].strftime("%d/%m/%Y"), "NVDA", "VENTE", 40, 140.0, 5)])
    etat, _ = mo.derouler(j)
    q = pf.quantites_quotidiennes(j, d)
    assert proche(q["NVDA"].iloc[-1], float(etat.at["NVDA", "Quantité"]))


def test_seuils_partages_identiques_partout():
    assert dm.RISQUE_DEFAUT == rg.RISQUE_PAR_IDEE
    assert dm.PLAFOND_LIGNE == rg.PLAFOND_LIGNE
    assert val.CONCENTRATION_ALERTE == rg.CONCENTRATION


if __name__ == "__main__":
    import sys
    import traceback

    echecs = []
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for nom, fonction in tests:
        try:
            fonction()
            print(f"  ok    {nom}")
        except AssertionError:
            echecs.append((nom, "résultat faux"))
            print(f"  ÉCHEC {nom}")
            traceback.print_exc(limit=2)
        except Exception as e:
            echecs.append((nom, f"{type(e).__name__}: {e}"))
            print(f"  ERREUR {nom} — {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n{len(tests) - len(echecs)}/{len(tests)} tests passés")
    for nom, raison in echecs:
        print(f"  - {nom} : {raison}")
    sys.exit(1 if echecs else 0)
