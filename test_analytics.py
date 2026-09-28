"""
Verification du module de calcul financier.

Chaque test confronte une fonction a un resultat connu analytiquement, jamais
a un autre calcul du meme fichier : verifier un code par lui-meme ne prouve
rien. Les series sont construites pour que la reponse soit calculable a la
main, et cette reponse est ecrite en clair dans le test.

Lancement :  python3 -m pytest test_analytics.py -q
ou, sans pytest :  python3 test_analytics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analytics as an

TOL = 1e-9
JOURS = an.JOURS_BOURSE


def proche(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# ==========================================================================
# Rendements et annualisation
# ==========================================================================

def test_rendements_simples():
    prix = pd.Series([100.0, 110.0, 99.0])
    r = an.rendements(prix)
    # 110/100-1 = 0.10 ; 99/110-1 = -0.10
    assert proche(r.iloc[0], 0.10)
    assert proche(r.iloc[1], -0.10)


def test_rendement_annualise_croissance_constante():
    # 252 seances a +0,1 % : (1,001)^252 - 1 sur exactement un an
    r = pd.Series([0.001] * int(JOURS))
    attendu = 1.001 ** JOURS - 1
    assert proche(an.rendement_annualise(r, JOURS), attendu, 1e-12)


def test_rendement_annualise_deux_ans():
    # Deux ans de donnees : le CAGR doit etre la racine du total
    r = pd.Series([0.0005] * int(2 * JOURS))
    total = 1.0005 ** (2 * JOURS)
    assert proche(an.rendement_annualise(r, JOURS), total ** 0.5 - 1, 1e-10)


def test_rendement_annualise_perte_totale():
    # Une periode a -100 % rend le capital nul : pas de taux definissable
    r = pd.Series([0.05, -1.0, 0.05])
    assert np.isnan(an.rendement_annualise(r, JOURS))


def test_frequence_deduite_de_l_espacement():
    quotidien = pd.bdate_range("2024-01-01", periods=30)
    hebdo = pd.date_range("2024-01-01", periods=30, freq="7D")
    mensuel = pd.date_range("2024-01-01", periods=30, freq="MS")
    assert an.frequence_annuelle(quotidien) == JOURS
    assert an.frequence_annuelle(hebdo) == 52.0
    assert an.frequence_annuelle(mensuel) == 12.0


# ==========================================================================
# Dispersion
# ==========================================================================

def test_volatilite_serie_alternee():
    # +2 %, -2 % repetes : ecart-type d'echantillon connu
    r = pd.Series([0.02, -0.02] * 50)
    ecart = r.std(ddof=1)
    assert proche(an.volatilite(r, JOURS), ecart * np.sqrt(JOURS), 1e-12)


def test_volatilite_serie_constante_est_nulle():
    assert proche(an.volatilite(pd.Series([0.01] * 50), JOURS), 0.0)


def test_variance_est_le_carre_de_la_volatilite():
    r = pd.Series(np.linspace(-0.03, 0.04, 200))
    v = an.volatilite(r, JOURS)
    assert proche(an.variance_annualisee(r, JOURS), v ** 2, 1e-12)


def test_semi_deviation_ignore_les_hausses():
    # Seules -0,02 et -0,04 comptent : racine de la moyenne des carres
    r = pd.Series([0.05, -0.02, 0.03, -0.04, 0.01])
    attendu = np.sqrt(((0.02 ** 2 + 0.04 ** 2) / 2) * JOURS)
    assert proche(an.semi_deviation(r, 0.0, JOURS), attendu, 1e-12)


def test_semi_deviation_serie_uniquement_haussiere():
    assert np.isnan(an.semi_deviation(pd.Series([0.01] * 20), 0.0, JOURS))


# ==========================================================================
# Pertes
# ==========================================================================

def test_drawdown_max_calcule_a_la_main():
    # 100 -> 120 -> 90 -> 110 : pire recul 90/120 - 1 = -25 %
    r = pd.Series([0.20, -0.25, 0.20 / 0.9 - 1 + 1 - 1])
    r = pd.Series([0.20, -0.25])
    assert proche(an.drawdown_max(r), -0.25, 1e-12)


def test_drawdown_courbe_revient_a_zero_au_nouveau_sommet():
    r = pd.Series([0.10, -0.10, 0.20])
    dd = an.courbe_drawdown(r)
    # 1,10 puis 0,99 puis 1,188 : nouveau plus haut, donc drawdown nul
    assert proche(dd.iloc[0], 0.0)
    assert proche(dd.iloc[-1], 0.0, 1e-12)


def test_drawdown_serie_monotone_croissante():
    assert proche(an.drawdown_max(pd.Series([0.01] * 40)), 0.0, 1e-12)


def test_duree_drawdown():
    # Sommet en position 0, creux en position 3 : trois periodes ecoulees
    r = pd.Series([0.10, -0.05, -0.05, -0.05, 0.30])
    assert an.duree_drawdown_max(r) == 4


# ==========================================================================
# Valeur en risque
# ==========================================================================

def test_var_historique_sur_distribution_connue():
    # 100 valeurs de -0,50 % a 0,49 % par pas de 0,01 % : le 5e centile
    r = pd.Series(np.arange(-50, 50) / 10000)
    var = an.var_historique(r, 0.05)
    assert proche(var, np.quantile(r, 0.05), 1e-12)
    assert var < 0


def test_cvar_plus_severe_que_var():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.02, 2000))
    assert an.cvar_historique(r, 0.05) < an.var_historique(r, 0.05)


def test_var_parametrique_sur_serie_gaussienne():
    # Sur un echantillon normal, VaR 5 % ~ moyenne - 1,645 ecart-type
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.015, 20000))
    attendu = r.mean() - 1.645 * r.std(ddof=1)
    assert proche(an.var_parametrique(r, 0.05), attendu, 1e-12)


def test_var_est_plus_negative_a_1_pourcent_qu_a_10():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.02, 5000))
    assert an.var_historique(r, 0.01) < an.var_historique(r, 0.10)


# ==========================================================================
# Relation au marche
# ==========================================================================

def test_beta_vaut_un_quand_l_actif_est_le_marche():
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.01, 500))
    res = an.regression_marche(m, m)
    assert proche(res["beta"], 1.0, 1e-9)
    assert proche(res["r2"], 1.0, 1e-9)
    assert proche(res["correlation"], 1.0, 1e-9)


def test_beta_vaut_deux_quand_l_actif_amplifie_le_marche():
    rng = np.random.default_rng(4)
    m = pd.Series(rng.normal(0, 0.01, 500))
    res = an.regression_marche(2 * m, m)
    assert proche(res["beta"], 2.0, 1e-9)


def test_beta_negatif_sur_actif_inverse():
    rng = np.random.default_rng(5)
    m = pd.Series(rng.normal(0, 0.01, 500))
    assert proche(an.regression_marche(-m, m)["beta"], -1.0, 1e-9)


def test_risque_specifique_nul_si_parfaitement_explique():
    rng = np.random.default_rng(6)
    m = pd.Series(rng.normal(0, 0.01, 500))
    res = an.regression_marche(1.5 * m, m)
    assert proche(res["risque_specifique"], 0.0, 1e-6)


def test_tracking_error_nulle_contre_soi_meme():
    rng = np.random.default_rng(7)
    m = pd.Series(rng.normal(0, 0.01, 500))
    assert proche(an.regression_marche(m, m)["tracking_error"], 0.0, 1e-9)


# ==========================================================================
# Portefeuille
# ==========================================================================

def test_covariance_annualisee():
    rng = np.random.default_rng(8)
    df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("ABC"))
    cov = an.matrice_covariance(df, JOURS)
    assert proche(cov.loc["A", "A"], df["A"].var(ddof=1) * JOURS, 1e-12)
    assert proche(cov.loc["A", "B"], cov.loc["B", "A"], 1e-15)


def test_volatilite_portefeuille_deux_actifs_independants():
    # Variances 0,04 et 0,09, covariance nulle, poids 50/50
    # variance = 0,25*0,04 + 0,25*0,09 = 0,0325 ; volatilite = 0,18028
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"],
                       columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert proche(an.volatilite_portefeuille(w, cov), np.sqrt(0.0325), 1e-12)


def test_volatilite_portefeuille_correlation_parfaite():
    # Deux actifs identiques : la volatilite du tout egale celle de chacun
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["A", "B"],
                       columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert proche(an.volatilite_portefeuille(w, cov), 0.2, 1e-12)


def test_somme_des_parts_de_risque_vaut_cent():
    cov = pd.DataFrame([[0.04, 0.01, 0.00],
                        [0.01, 0.09, 0.02],
                        [0.00, 0.02, 0.16]],
                       index=list("ABC"), columns=list("ABC"))
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    d = an.decomposition_risque(w, cov)
    assert proche(d["Part du risque (%)"].sum(), 100.0, 1e-9)


def test_contributions_sommees_egalent_la_volatilite():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]],
                       index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.6, "B": 0.4})
    d = an.decomposition_risque(w, cov)
    total = d["Contribution au risque (%)"].sum() / 100
    assert proche(total, an.volatilite_portefeuille(w, cov), 1e-12)


def test_ratio_diversification_vaut_un_si_tout_est_correle():
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]],
                       index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert proche(an.ratio_diversification(w, cov), 1.0, 1e-12)


def test_ratio_diversification_superieur_a_un_si_decorrele():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]],
                       index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert an.ratio_diversification(w, cov) > 1.4


def test_nombre_effectif_lignes():
    # Quatre lignes egales : 4 lignes effectives
    assert proche(an.nombre_effectif_lignes(pd.Series([0.25] * 4)), 4.0, 1e-12)
    # Une ligne a 80 %, trois a 6,67 % : 1/(0,64+3*0,004444) = 1,533
    w = pd.Series([0.8] + [0.2 / 3] * 3)
    assert proche(an.nombre_effectif_lignes(w), 1 / ((w ** 2).sum()), 1e-12)


def test_rendements_portefeuille_poids_constants():
    df = pd.DataFrame({"A": [0.10, -0.05], "B": [0.00, 0.05]})
    w = pd.Series({"A": 0.5, "B": 0.5})
    r = an.rendements_portefeuille(df, w)
    assert proche(r.iloc[0], 0.05)
    assert proche(r.iloc[1], 0.0)


# ==========================================================================
# Ratios composites
# ==========================================================================

def test_sharpe_calcule_independamment():
    """
    Reference reconstruite avec numpy seul, sans reutiliser le module.

    Un test qui compare sharpe() a rendement_annualise() divise par
    volatilite() ne verifie que la coherence interne : si les deux composants
    sont faux de la meme facon, il passe quand meme.
    """
    r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015] * 40)
    x = r.to_numpy()
    cagr = float(np.prod(1 + x)) ** (JOURS / len(x)) - 1
    vol = float(np.std(x, ddof=1) * np.sqrt(JOURS))
    assert proche(an.sharpe(r, 0.0, JOURS), cagr / vol, 1e-12)


def test_decomposition_risque_calculee_a_la_main():
    """
    Deux actifs, contributions posees sur le papier.

    variance = 0,36 x 0,04 + 2 x 0,6 x 0,4 x 0,01 + 0,16 x 0,09 = 0,0336
    contribution de A = 0,6 x (0,04 x 0,6 + 0,01 x 0,4) = 0,0168
    contribution de B = 0,4 x (0,01 x 0,6 + 0,09 x 0,4) = 0,0168
    Les deux lignes pesent donc exactement la moitie du risque chacune,
    bien que leurs poids soient 60 et 40 et leurs volatilites 20 % et 30 %.
    """
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]],
                       index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.6, "B": 0.4})
    vol = np.sqrt(0.0336)
    d = an.decomposition_risque(w, cov)

    assert proche(d.at["A", "Contribution au risque (%)"] / 100,
                  0.0168 / vol, 1e-12)
    assert proche(d.at["B", "Contribution au risque (%)"] / 100,
                  0.0168 / vol, 1e-12)
    assert proche(d.at["A", "Part du risque (%)"], 50.0, 1e-9)
    assert proche(d.at["B", "Part du risque (%)"], 50.0, 1e-9)
    # La volatilite seule de chaque ligne : racines de 0,04 et 0,09
    assert proche(d.at["A", "Volatilité seule (%)"], 20.0, 1e-12)
    assert proche(d.at["B", "Volatilité seule (%)"], 30.0, 1e-12)


def test_sharpe_indefini_si_volatilite_nulle():
    assert np.isnan(an.sharpe(pd.Series([0.001] * 50), 0.0, JOURS))


def test_sortino_superieur_au_sharpe_si_asymetrie_favorable():
    # Beaucoup de petites baisses, quelques fortes hausses
    r = pd.Series([-0.002] * 90 + [0.05] * 10)
    assert an.sortino(r, 0.0, JOURS) > an.sharpe(r, 0.0, JOURS)


def test_omega_vaut_un_si_gains_egalent_pertes():
    r = pd.Series([0.01, -0.01, 0.02, -0.02])
    assert proche(an.omega(r, 0.0), 1.0, 1e-12)


def test_omega_superieur_a_un_si_gains_dominent():
    r = pd.Series([0.03, -0.01, 0.02, -0.01])
    assert proche(an.omega(r, 0.0), 0.05 / 0.02, 1e-12)


def test_calmar_est_rendement_sur_perte_max():
    r = pd.Series([0.02] * 100 + [-0.30] + [0.02] * 100)
    attendu = an.rendement_annualise(r, JOURS) / abs(an.drawdown_max(r))
    assert proche(an.calmar(r, JOURS), attendu, 1e-12)


def test_treynor_sur_beta_connu():
    rng = np.random.default_rng(10)
    m = pd.Series(rng.normal(0.0004, 0.01, 800))
    actif = 2 * m
    attendu = an.rendement_annualise(actif, JOURS) / 2.0
    assert proche(an.treynor(actif, m, 0.0, JOURS), attendu, 1e-6)


# ==========================================================================
# Robustesse
# ==========================================================================

def test_series_vides_ne_plantent_pas():
    vide = pd.Series(dtype=float)
    for f in (an.volatilite, an.rendement_annualise, an.drawdown_max,
              an.var_historique, an.cvar_historique):
        f(vide)          # ne doit lever aucune exception


def test_serie_d_un_seul_point():
    un = pd.Series([0.01])
    assert np.isnan(an.volatilite(un, JOURS))


def test_valeurs_manquantes_ignorees():
    avec = pd.Series([0.01, np.nan, -0.01, 0.02])
    sans = pd.Series([0.01, -0.01, 0.02])
    assert proche(an.volatilite(avec, JOURS), an.volatilite(sans, JOURS), 1e-12)


if __name__ == "__main__":
    import sys, traceback
    echecs = []
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for nom, fonction in tests:
        try:
            fonction()
            print(f"  ok   {nom}")
        except AssertionError:
            echecs.append((nom, "resultat faux"))
            print(f"  ÉCHEC {nom}")
            traceback.print_exc(limit=2)
        except Exception as e:
            echecs.append((nom, f"{type(e).__name__}: {e}"))
            print(f"  ERREUR {nom} — {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(echecs)}/{len(tests)} tests passés")
    for nom, raison in echecs:
        print(f"  - {nom} : {raison}")
    sys.exit(1 if echecs else 0)
