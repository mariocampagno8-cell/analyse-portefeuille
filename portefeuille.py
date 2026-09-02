"""
Analyse de portefeuille orientee decision.

Un tableau de valorisation ne dit pas quoi faire. Ce module repond a trois
questions, dans l'ordre ou elles se posent :

  OU EST MON RISQUE      la contribution au risque, qui ne suit presque jamais
                         les poids
  QU'EST-CE QUI CLOCHE   un diagnostic automatique, hierarchise par gravite
  QUE FAIRE              un plan chiffre, avec son cout fiscal

Deux partis pris expliques.

La performance est decomposee entre le titre et le change. Une ligne
americaine qui gagne 20 % en dollars pendant que l'euro s'apprecie de 8 % ne
rapporte que 11 % : confondre les deux fait prendre un pari de change pour
une reussite de selection.

Le cout fiscal est chiffre. En France, un arbitrage sur compte-titres
declenche l'imposition des plus-values au prelevement forfaitaire unique. Un
rebalancement theoriquement optimal peut couter plus qu'il ne rapporte, et
l'ignorer rend le conseil inutilisable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PFU = 0.30                      # prélèvement forfaitaire unique
FRAIS_ORDRE = 0.005             # 0,5 % par ordre, ordre de grandeur courtier

SEUIL_CONCENTRATION = 25.0      # % du portefeuille sur une ligne
SEUIL_RISQUE = 40.0             # % du risque porté par une ligne
SEUIL_CORRELATION = 0.80        # corrélation moyenne entre deux lignes
SEUIL_LIGNES_EFFECTIVES = 4.0   # diversification réelle minimale
SEUIL_DEVISE = 60.0             # % d'exposition à une devise unique


# ==========================================================================
# Décomposition de la performance
# ==========================================================================

def decomposer_performance(valorisation: pd.DataFrame,
                           taux_achat: dict | None = None) -> pd.DataFrame:
    """
    Separe ce qui vient du titre de ce qui vient du change.

    Sans taux historique fourni, la decomposition n'est pas calculable et la
    colonne est laissee vide plutot qu'estimee : mieux vaut une case vide
    qu'un chiffre faux.
    """
    sortie = valorisation.copy()
    sortie["Perf titre (%)"] = np.where(
        sortie["PRU"] > 0, (sortie["Cours"] / sortie["PRU"] - 1) * 100, np.nan)

    if taux_achat:
        change = []
        for ticker, ligne in sortie.iterrows():
            achat = taux_achat.get(ticker)
            actuel = ligne.get("Taux de change", 1.0)
            change.append((actuel / achat - 1) * 100
                          if achat and achat > 0 else np.nan)
        sortie["Perf change (%)"] = change
        sortie["Perf totale (%)"] = (
            (1 + sortie["Perf titre (%)"] / 100)
            * (1 + sortie["Perf change (%)"].fillna(0) / 100) - 1) * 100
    else:
        sortie["Perf change (%)"] = np.nan
        sortie["Perf totale (%)"] = sortie["Perf titre (%)"]
    return sortie


def exposition(valorisation: pd.DataFrame, secteurs: dict | None = None) -> dict:
    """Repartition par devise et, si connue, par secteur."""
    total = float(valorisation["Valeur"].sum())
    if total <= 0:
        return {}

    devises = (valorisation.groupby("Devise")["Valeur"].sum() / total * 100)
    resultat = {"devises": devises.sort_values(ascending=False).to_dict()}

    if secteurs:
        table = valorisation.copy()
        table["Secteur"] = [secteurs.get(t, "Inconnu") for t in table.index]
        resultat["secteurs"] = (
            table.groupby("Secteur")["Valeur"].sum() / total * 100
        ).sort_values(ascending=False).to_dict()
    return resultat


# ==========================================================================
# Diagnostic
# ==========================================================================

def diagnostiquer(valorisation: pd.DataFrame, decomposition: pd.DataFrame,
                  correlations: pd.DataFrame, lignes_effectives: float,
                  beta: float | None = None,
                  beta_baissier: float | None = None,
                  secteurs: dict | None = None) -> list[dict]:
    """
    Liste hierarchisee de ce qui merite attention.

    Chaque constat porte un niveau de gravite et une action possible. Un
    diagnostic sans action a faire n'est qu'un commentaire.
    """
    constats = []
    total = float(valorisation["Valeur"].sum())
    if total <= 0:
        return constats

    poids = valorisation["Valeur"] / total * 100

    # --- Concentration en capital
    plus_lourde = poids.idxmax()
    part = float(poids.max())
    if part > SEUIL_CONCENTRATION:
        constats.append({
            "gravite": "élevée" if part > 40 else "moyenne",
            "sujet": "Concentration",
            "constat": f"{plus_lourde} pèse {part:.0f} % du portefeuille.",
            "portee": ("Une baisse de 30 % sur cette seule ligne coûterait "
                       f"{part * 0.3:.0f} % du portefeuille."),
            "action": f"Alléger {plus_lourde} pour revenir sous "
                      f"{SEUIL_CONCENTRATION:.0f} %."})

    # --- Concentration en risque : le décalage avec les poids est l'essentiel
    if not decomposition.empty and "Part du risque (%)" in decomposition.columns:
        risque = decomposition["Part du risque (%)"]
        dominante = risque.idxmax()
        part_risque = float(risque.max())
        part_capital = float(poids.get(dominante, 0))
        if part_risque > SEUIL_RISQUE and part_risque > part_capital * 1.3:
            constats.append({
                "gravite": "élevée",
                "sujet": "Risque concentré",
                "constat": f"{dominante} porte {part_risque:.0f} % du risque "
                           f"pour {part_capital:.0f} % du capital.",
                "portee": ("Le comportement du portefeuille est déterminé par "
                           "cette ligne, pas par les plus grosses positions."),
                "action": f"Alléger {dominante} réduit le risque total "
                          f"{part_risque / max(part_capital, 1):.1f} fois plus "
                          "qu'alléger n'importe quelle autre ligne."})

    # --- Diversification réelle
    if lignes_effectives and lignes_effectives < SEUIL_LIGNES_EFFECTIVES:
        constats.append({
            "gravite": "moyenne",
            "sujet": "Diversification",
            "constat": f"{lignes_effectives:.1f} lignes effectives pour "
                       f"{len(valorisation)} positions.",
            "portee": ("La diversification est cosmétique : le portefeuille se "
                       "comporte comme s'il ne comptait que quelques lignes."),
            "action": "Réduire les positions dominantes ou ajouter des actifs "
                      "faiblement corrélés."})

    # --- Corrélations élevées
    if not correlations.empty and len(correlations) > 1:
        paires = []
        for i, a in enumerate(correlations.index):
            for b in correlations.index[i + 1:]:
                valeur = correlations.loc[a, b]
                if np.isfinite(valeur) and valeur >= SEUIL_CORRELATION:
                    paires.append((a, b, float(valeur)))
        if paires:
            paires.sort(key=lambda x: -x[2])
            liste = ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in paires[:3])
            constats.append({
                "gravite": "moyenne",
                "sujet": "Redondance",
                "constat": f"Corrélations supérieures à {SEUIL_CORRELATION:.2f} : "
                           f"{liste}.",
                "portee": "Ces lignes montent et descendent ensemble : elles "
                          "comptent pour une seule position.",
                "action": "Conserver la meilleure de chaque paire plutôt que "
                          "les deux."})

    # --- Exposition devise
    expo = exposition(valorisation, secteurs)
    for devise, part_devise in expo.get("devises", {}).items():
        if part_devise > SEUIL_DEVISE and len(expo["devises"]) > 1:
            constats.append({
                "gravite": "faible",
                "sujet": "Change",
                "constat": f"{part_devise:.0f} % du portefeuille en {devise}.",
                "portee": "Une part de la performance dépend du taux de change, "
                          "indépendamment des sociétés détenues.",
                "action": "Le savoir suffit ; se couvrir coûte plus cher que le "
                          "risque évité sur un portefeuille de cette taille."})
            break

    # --- Asymétrie du bêta
    if (beta is not None and beta_baissier is not None
            and np.isfinite(beta) and np.isfinite(beta_baissier)
            and beta_baissier > beta * 1.15):
        constats.append({
            "gravite": "élevée",
            "sujet": "Bêta asymétrique",
            "constat": f"Bêta de {beta:.2f} en moyenne, mais "
                       f"{beta_baissier:.2f} en marché baissier.",
            "portee": "Le portefeuille suit peu les hausses et amplifie les "
                      "baisses — la pire des combinaisons.",
            "action": "Identifier les lignes responsables dans la colonne "
                      "bêta baissier de l'onglet Risque."})

    if secteurs:
        for secteur, part_secteur in expo.get("secteurs", {}).items():
            if part_secteur > 50 and secteur != "Inconnu":
                constats.append({
                    "gravite": "moyenne",
                    "sujet": "Secteur",
                    "constat": f"{part_secteur:.0f} % en {secteur}.",
                    "portee": "Un choc sectoriel toucherait la moitié du "
                              "portefeuille en même temps.",
                    "action": "Diversifier vers d'autres secteurs lors des "
                              "prochains apports."})
                break

    ordre = {"élevée": 0, "moyenne": 1, "faible": 2}
    return sorted(constats, key=lambda c: ordre.get(c["gravite"], 9))


# ==========================================================================
# Plan de rééquilibrage
# ==========================================================================

def plan_reequilibrage(valorisation: pd.DataFrame, poids_cible: pd.Series,
                       seuil_ecart: float = 2.0,
                       taux_fiscal: float = PFU,
                       frais: float = FRAIS_ORDRE) -> pd.DataFrame:
    """
    Traduit une allocation cible en ordres, avec leur cout reel.

    Le cout fiscal change souvent la conclusion. Vendre une ligne en
    plus-value de 90 % pour rééquilibrer coûte 27 % du gain en impôt : le
    rebalancement doit rapporter davantage que cela pour valoir la peine.
    """
    total = float(valorisation["Valeur"].sum())
    if total <= 0 or poids_cible.empty:
        return pd.DataFrame()

    lignes = []
    for ticker in valorisation.index:
        valeur = float(valorisation.loc[ticker, "Valeur"])
        investi = float(valorisation.loc[ticker, "Investi"])
        actuel = valeur / total * 100
        cible = float(poids_cible.get(ticker, 0)) * 100
        ecart = cible - actuel
        montant = ecart / 100 * total

        if abs(ecart) < seuil_ecart:
            sens = "Conserver"
            impot = frais_ordre = 0.0
        elif ecart > 0:
            sens = "Renforcer"
            impot = 0.0
            frais_ordre = abs(montant) * frais
        else:
            sens = "Alléger"
            # Impôt au prorata de la plus-value latente cédée
            plus_value = valeur - investi
            part_cedee = min(abs(montant) / valeur, 1.0) if valeur > 0 else 0
            impot = max(plus_value * part_cedee, 0) * taux_fiscal
            frais_ordre = abs(montant) * frais

        lignes.append({
            "Ticker": ticker,
            "Poids actuel (%)": actuel,
            "Poids cible (%)": cible,
            "Écart (pt)": ecart,
            "Sens": sens,
            "Montant": montant,
            "Frais": frais_ordre,
            "Impôt estimé": impot,
            "Coût total": frais_ordre + impot,
        })

    plan = pd.DataFrame(lignes).set_index("Ticker")
    return plan.sort_values("Écart (pt)")


def resume_plan(plan: pd.DataFrame, gain_attendu_pct: float | None = None,
                valeur_totale: float = 0) -> dict:
    """
    Synthese chiffree du plan, et verdict sur son interet.

    Le gain attendu est celui de la reduction de volatilite ; le cout est
    immediat et certain. Comparer les deux est la seule facon de decider.
    """
    if plan.empty:
        return {}

    mouvements = plan[plan["Sens"] != "Conserver"]
    cout = float(mouvements["Coût total"].sum())
    resultat = {
        "ordres": int(len(mouvements)),
        "montant_echange": float(mouvements["Montant"].abs().sum()),
        "frais": float(mouvements["Frais"].sum()),
        "impot": float(mouvements["Impôt estimé"].sum()),
        "cout_total": cout,
    }
    if valeur_totale > 0:
        resultat["cout_pct"] = cout / valeur_totale * 100
        resultat["rotation_pct"] = (resultat["montant_echange"]
                                    / valeur_totale * 100)

    if gain_attendu_pct is not None and valeur_totale > 0:
        gain = gain_attendu_pct / 100 * valeur_totale
        resultat["gain_attendu"] = gain
        resultat["verdict"] = (
            "Le coût dépasse le gain attendu : ne pas rééquilibrer."
            if cout >= gain else
            f"Le gain attendu couvre le coût en "
            f"{cout / gain:.1f} fois l'écart annuel.")
    return resultat


def apport_optimal(valorisation: pd.DataFrame, poids_cible: pd.Series,
                   montant: float) -> pd.DataFrame:
    """
    Repartit un apport neuf pour se rapprocher de la cible sans rien vendre.

    C'est la methode la plus efficace pour un particulier : elle atteint le
    meme resultat qu'un rebalancement sans declencher un centime d'impot.
    """
    total = float(valorisation["Valeur"].sum())
    if total <= 0 or montant <= 0 or poids_cible.empty:
        return pd.DataFrame()

    nouveau_total = total + montant
    lignes = []
    for ticker in poids_cible.index:
        valeur = float(valorisation["Valeur"].get(ticker, 0))
        visee = float(poids_cible[ticker]) * nouveau_total
        achat = max(visee - valeur, 0)
        lignes.append({"Ticker": ticker, "Valeur actuelle": valeur,
                       "Cible après apport": visee, "À acheter": achat})

    plan = pd.DataFrame(lignes).set_index("Ticker")
    besoin = float(plan["À acheter"].sum())
    if besoin > 0:
        # On ne peut allouer que le montant disponible : répartition au prorata
        plan["À acheter"] = plan["À acheter"] / besoin * montant
    plan["Part de l'apport (%)"] = plan["À acheter"] / montant * 100
    return plan[plan["À acheter"] > 0].sort_values("À acheter", ascending=False)
