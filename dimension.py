"""
Dimensionnement des positions.

C'est le seul calcul de cet outil qui intervient avant l'achat. Tous les
autres constatent : celui-ci decide.

Le principe tient en une phrase. Le risque d'une ligne n'est pas sa taille,
c'est sa taille multipliee par la distance a son stop. Une position de
10 000 EUR sur un titre dont le stop est a 5 % risque 500 EUR ; la meme somme
sur un titre dont le stop est a 25 % en risque 2 500. Raisonner en euros
investis met donc sur le meme plan des engagements cinq fois differents.

On renverse donc le calcul : on fixe d'abord ce qu'on accepte de perdre, puis
on en deduit la taille. La perte acceptee s'exprime en pourcentage du
portefeuille, parce que c'est la seule unite qui reste comparable d'une idee
a l'autre et d'une annee a l'autre.

Sur le niveau a retenir, la litterature de gestion converge autour de 0,5 a
2 % du capital par idee pour un particulier, sans qu'aucun chiffre precis ne
soit demontre. Ce qui est en revanche solidement etabli, c'est l'asymetrie des
pertes : perdre 50 % oblige a gagner 100 % pour revenir au point de depart.
C'est cette asymetrie, et non une regle magique, qui justifie de plafonner
chaque ligne.

Le critere de Kelly est propose a titre de comparaison. Sa formule donne la
mise qui maximise la croissance a long terme, mais elle exige de connaitre la
probabilite de gain et le rapport gain/perte — deux grandeurs qu'un
investisseur particulier ne peut pas estimer de facon fiable. Kelly integral
est notoirement trop agressif des que ces estimations sont fausses, et elles
le sont toujours. La fraction est donc affichee divisee par quatre, avec la
mise en garde qui s'impose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import reglages as rg

# Plafond de perte acceptee sur une seule idee, en % du portefeuille.
RISQUE_DEFAUT = rg.RISQUE_PAR_IDEE
RISQUE_MAX = 3.0

# Plafond de taille, independamment du stop : une ligne peut avoir un stop
# tres serre et rester deraisonnable en taille, notamment parce qu'un stop
# ne protege pas d'une ouverture en decalage.
PLAFOND_LIGNE = rg.PLAFOND_LIGNE

FRACTION_KELLY = 0.25


def taille(portefeuille: float, cours: float, stop: float,
           risque_pct: float = RISQUE_DEFAUT,
           plafond_pct: float = PLAFOND_LIGNE) -> dict:
    """
    Combien acheter pour ne risquer que `risque_pct` % du portefeuille.

    La perte acceptee divisee par la perte unitaire donne le nombre de titres.
    Le resultat est ensuite ecrete par le plafond de taille, et l'ecretage est
    signale : savoir que la contrainte a joue, et laquelle, importe autant que
    le chiffre.
    """
    vide = {"titres": 0, "montant": 0.0, "perte": 0.0, "part": 0.0,
            "distance": np.nan, "limite": "données insuffisantes"}
    if not all(np.isfinite([portefeuille, cours, stop])) or portefeuille <= 0 \
            or cours <= 0 or stop <= 0 or stop >= cours:
        return vide

    distance = (cours - stop) / cours * 100          # % de baisse jusqu'au stop
    perte_acceptee = portefeuille * risque_pct / 100
    perte_unitaire = cours - stop

    titres_risque = perte_acceptee / perte_unitaire
    titres_plafond = portefeuille * plafond_pct / 100 / cours

    limite = "risque" if titres_risque <= titres_plafond else "taille"
    titres = int(min(titres_risque, titres_plafond))

    montant = titres * cours
    return {
        "titres": titres,
        "montant": montant,
        "perte": titres * perte_unitaire,
        "part": montant / portefeuille * 100,
        "distance": distance,
        "limite": limite}


def risque_actuel(valeur: float, portefeuille: float, cours: float,
                  stop: float) -> dict:
    """
    Ce qu'une position existante risque reellement, et la taille qu'elle
    aurait si elle respectait un budget donne.

    Sert a confronter le portefeuille tel qu'il est au portefeuille tel qu'il
    serait si chaque ligne avait ete dimensionnee.
    """
    if not all(np.isfinite([valeur, portefeuille, cours, stop])) \
            or portefeuille <= 0 or cours <= 0:
        return {}
    if stop >= cours:
        # Stop deja franchi : la perte n'est plus une hypothese.
        return {"perte": np.nan, "part": valeur / portefeuille * 100,
                "risque_pct": np.nan, "franchi": True}
    distance = (cours - stop) / cours
    perte = valeur * distance
    return {"perte": perte, "part": valeur / portefeuille * 100,
            "risque_pct": perte / portefeuille * 100, "franchi": False}


def budget_respecte(positions: pd.DataFrame,
                    risque_pct: float = RISQUE_DEFAUT,
                    plafond_pct: float = PLAFOND_LIGNE) -> pd.DataFrame:
    """
    Compare, ligne par ligne, le risque pris au risque autorise.

    `positions` porte les colonnes Valeur, Cours, Stop. La colonne Écart dit
    de combien il faudrait alleger pour rentrer dans le budget.
    """
    colonnes = ["Valeur", "Poids (%)", "Risque (%)", "Budget (%)",
                "Valeur cible", "À alléger", "Franchi"]
    if positions is None or positions.empty:
        return pd.DataFrame(columns=colonnes)

    total = float(positions["Valeur"].sum())
    lignes = []
    for t, p in positions.iterrows():
        cours, stop, valeur = float(p["Cours"]), float(p["Stop"]), float(p["Valeur"])
        franchi = not np.isfinite(stop) or stop >= cours
        if franchi:
            lignes.append({"Ticker": t, "Valeur": valeur,
                           "Poids (%)": valeur / total * 100,
                           "Risque (%)": np.nan, "Budget (%)": risque_pct,
                           "Valeur cible": np.nan, "À alléger": np.nan,
                           "Franchi": True})
            continue
        distance = (cours - stop) / cours
        risque = valeur * distance / total * 100
        # Le budget de risque seul autoriserait une ligne enorme sur un titre
        # au stop tres serre : a 2 % de distance, 1 % de risque justifierait
        # la moitie du portefeuille. Or un stop proche ne protege pas d'une
        # ouverture en decalage, ou le cours saute par-dessus. Le plafond de
        # taille reste donc actif, et c'est lui qui mord dans ce cas.
        cible = min(total * risque_pct / 100 / distance,
                    total * plafond_pct / 100)
        lignes.append({
            "Ticker": t, "Valeur": valeur,
            "Poids (%)": valeur / total * 100,
            "Risque (%)": risque, "Budget (%)": risque_pct,
            "Valeur cible": cible,
            "À alléger": max(0.0, valeur - cible),
            "Franchi": False})

    return pd.DataFrame(lignes).set_index("Ticker")[colonnes]


def kelly(probabilite: float, gain: float, perte: float,
          fraction: float = FRACTION_KELLY) -> dict:
    """
    Fraction de Kelly, deliberement divisee.

    f = p - (1 - p) / R, ou R est le rapport gain sur perte. La formule
    suppose connues la probabilite de gain et l'amplitude des deux issues ;
    aucune des deux ne l'est. Une surestimation de quelques points de la
    probabilite suffit a rendre Kelly integral ruineux, alors que le sous-
    dimensionnement ne coute qu'un peu de croissance. D'ou la fraction.
    """
    if not all(np.isfinite([probabilite, gain, perte])) or perte <= 0 \
            or not 0 < probabilite < 1:
        return {}
    rapport = gain / perte
    f = probabilite - (1 - probabilite) / rapport
    return {"complet": f * 100, "fractionne": max(0.0, f * fraction) * 100,
            "rapport": rapport,
            "favorable": f > 0}
