"""
Classement sectoriel des valeurs surveillees.

Une contrainte de secteur suppose de savoir a quel secteur appartient chaque
titre. Yahoo expose cette information, mais une par une, lentement, et avec
des libelles qui changent sans prevenir — « Technology » un jour,
« Information Technology » le lendemain. Le classement est donc ecrit ici,
en clair et verifiable.

Deux consequences a assumer. Il est fige : une valeur ajoutee a la liste de
surveillance tombe dans « Non classe » tant que personne ne l'ajoute ici.
Et il reflete un jugement : Vertiv fabrique des systemes de refroidissement
pour centres de donnees, ce qui en fait un industriel exposé a la technologie
plutot qu'une valeur technologique ; Constellation Energy et Oklo vendent de
l'electricite a des centres de donnees sans etre des valeurs technologiques
non plus. Classer autrement serait defendable, et il suffit de modifier la
ligne correspondante.

La distinction « technologie americaine » separe la cotation et le marche
principal, pas le siege social : ASML est neerlandaise, Nebius aussi.
"""

from __future__ import annotations

TECH_US = "Technologie US"
NON_CLASSE = "Non classé"

SECTEURS = {
    # --- Technologie américaine
    "NVDA": TECH_US,   # semi-conducteurs, calcul accéléré
    "AVGO": TECH_US,   # semi-conducteurs et logiciels d'infrastructure
    "MU": TECH_US,     # mémoires
    "MRVL": TECH_US,   # semi-conducteurs pour centres de données
    "SMCI": TECH_US,   # serveurs
    "CRWV": TECH_US,   # infrastructure de calcul en nuage
    "SNPS": TECH_US,   # logiciels de conception de puces
    "CDNS": TECH_US,   # logiciels de conception de puces
    "SNOW": TECH_US,   # entrepôt de données en nuage
    "DDOG": TECH_US,   # supervision applicative
    "NET": TECH_US,    # réseau de diffusion et sécurité
    "FN": TECH_US,     # optique et photonique, cotée aux États-Unis
    "IONQ": TECH_US,   # calcul quantique

    # --- Technologie hors États-Unis
    "ASML.AS": "Technologie Europe",   # lithographie, Pays-Bas
    "NBIS": "Technologie Europe",      # infrastructure de calcul, Pays-Bas

    # --- Santé
    "JNJ": "Santé",

    # --- Consommation de base
    "KO": "Consommation de base",
    "PEP": "Consommation de base",
    "PG": "Consommation de base",
    "CL": "Consommation de base",

    # --- Consommation discrétionnaire
    "MCD": "Consommation discrétionnaire",
    "RACE": "Consommation discrétionnaire",   # Ferrari, Italie
    "NIO": "Consommation discrétionnaire",    # automobile, Chine

    # --- Industrie
    "VRT": "Industrie",    # refroidissement de centres de données
    "FIX": "Industrie",    # génie climatique
    "BE": "Industrie",     # piles à combustible

    # --- Services aux collectivités
    "CEG": "Services aux collectivités",   # électricité, nucléaire
    "VST": "Services aux collectivités",
    "OKLO": "Services aux collectivités",   # petits réacteurs modulaires

    # --- Finance
    "CRCL": "Finance",   # émetteur de stablecoin
    "NU": "Finance",     # banque en ligne, Brésil
}


def secteur(ticker: str) -> str:
    return SECTEURS.get(str(ticker).strip().upper(), NON_CLASSE)


def par_secteur(tickers) -> dict[str, list[str]]:
    """Regroupe une liste de tickers par secteur."""
    groupes: dict[str, list[str]] = {}
    for t in tickers:
        groupes.setdefault(secteur(t), []).append(t)
    return groupes


def liste_secteurs(tickers=None) -> list[str]:
    """Secteurs representes, par ordre alphabetique, Technologie US en tete."""
    presents = (sorted({secteur(t) for t in tickers}) if tickers
                else sorted(set(SECTEURS.values())))
    if TECH_US in presents:
        presents.remove(TECH_US)
        presents.insert(0, TECH_US)
    return presents
