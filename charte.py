"""
Conventions d'affichage FinexResearch.

Parti pris, apres un essai rate : ne pas redecorer Streamlit. Son rendu natif
est coherent ; le recouvrir d'une charte concue pour une application web sur
mesure produit un resultat batard, ni l'un ni l'autre.

Ce module se limite donc a ce qui sert reellement la lecture :

  L'ECRITURE DES CHIFFRES, qui est la partie qui empeche une erreur de
  lecture. Separateur de milliers en espace insecable fine, virgule decimale,
  signe toujours explicite, moins typographique aligne sur le plus, cadratin
  pour une donnee absente plutot qu'un zero trompeur.

  TROIS RETOUCHES visuelles seulement : chiffres tabulaires pour que les
  colonnes s'alignent, laiton a la place du rouge Streamlit sur les elements
  actifs, et classes de couleur pour les valeurs de performance.

Regle conservee du document d'origine, la plus utile de toutes : le vert et le
rouge ne signifient que gain et perte. C'est pourquoi l'accent de l'interface
est un laiton neutre — un onglet actif en rouge, dans une application
financiere, est une faute de lecture.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

# Espace insécable fine, séparateur de milliers de la locale française
FINE = "\u202f"
INSECABLE = "\u00a0"
MOINS = "\u2212"          # moins typographique, aligné sur le plus
CADRATIN = "—"

MOIS_ABREGES = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
                "juil.", "août", "sept.", "oct.", "nov.", "déc."]


# ==========================================================================
# Écriture des chiffres
# ==========================================================================

def _base(valeur: float, decimales: int) -> str:
    """Nombre en convention francaise : espace fine, virgule decimale."""
    texte = f"{abs(valeur):,.{decimales}f}"
    return texte.replace(",", FINE).replace(".", ",")


def nombre(valeur, decimales: int = 2, signe: bool = False) -> str:
    """Nombre brut. Renvoie un cadratin si la donnee est absente."""
    if valeur is None:
        return CADRATIN
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return CADRATIN
    if not np.isfinite(v):
        return CADRATIN

    texte = _base(v, decimales)
    if v < 0:
        return MOINS + texte
    # Une variation nulle ne porte pas de signe : « +0,00 % » suggère une
    # hausse infime là où il ne s'est rien passé.
    if signe and round(v, decimales) != 0:
        return "+" + texte
    return texte


def montant(valeur, devise: str = "€", decimales: int = 2) -> str:
    """Montant : deux decimales, symbole apres, espace insecable."""
    if valeur is None or not np.isfinite(float(valeur or np.nan)):
        return CADRATIN
    return nombre(valeur, decimales) + INSECABLE + devise


def valorisation(valeur, devise: str = "€") -> str:
    """
    Valorisation agregee, abregee au-dela du million.

    Un tableau de positions ou chaque ligne affiche neuf chiffres devient
    illisible ; l'abreviation restitue l'ordre de grandeur immediatement.
    """
    if valeur is None:
        return CADRATIN
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return CADRATIN
    if not np.isfinite(v):
        return CADRATIN

    if abs(v) >= 1e9:
        return nombre(v / 1e9, 2) + INSECABLE + "Md" + devise
    if abs(v) >= 1e6:
        return nombre(v / 1e6, 2) + INSECABLE + "M" + devise
    return montant(v, devise)


def variation(valeur, decimales: int = 2) -> str:
    """
    Variation en pourcentage, signe toujours explicite.

    Le signe n'est jamais porte par la couleur seule : cette regle rend
    l'interface lisible en daltonisme et a l'impression.
    """
    if valeur is None:
        return CADRATIN
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return CADRATIN
    if not np.isfinite(v):
        return CADRATIN
    return nombre(v, decimales, signe=True) + INSECABLE + "%"


def annualise(valeur) -> str:
    """Performance annualisee : une decimale, annualisation mentionnee."""
    if valeur is None or not np.isfinite(float(valeur or np.nan)):
        return CADRATIN
    return variation(valeur, 1) + INSECABLE + "p.a."


def ecart(valeur) -> str:
    """
    Ecart de faible amplitude, exprime en points de base sous 1 %.

    « +35 pb » se lit mieux que « +0,35 % » : sous le pour cent, la decimale
    devient plus difficile a comparer que l'entier.
    """
    if valeur is None:
        return CADRATIN
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return CADRATIN
    if not np.isfinite(v):
        return CADRATIN
    if abs(v) < 1:
        return nombre(v * 100, 0, signe=True) + INSECABLE + "pb"
    return variation(v)


def ratio(valeur) -> str:
    """Ratio : deux decimales, sans unite."""
    return nombre(valeur, 2)


def quantite(valeur, fractionnable: bool = False) -> str:
    """Quantite de titres : entier, sauf fractions d'ETF."""
    return nombre(valeur, 4 if fractionnable else 0)


def date_courte(quand) -> str:
    """Date au format long abrege : 3 sept. 2026."""
    if quand is None:
        return CADRATIN
    try:
        import pandas as pd
        horodatage = pd.Timestamp(quand)
        if horodatage is None or str(horodatage) == "NaT":
            return CADRATIN
        return (f"{horodatage.day}{INSECABLE}"
                f"{MOIS_ABREGES[horodatage.month - 1]}{INSECABLE}"
                f"{horodatage.year}")
    except Exception:
        return CADRATIN


def heure(quand=None, fuseau: str = "CET") -> str:
    """Heure sur 24 h, fuseau toujours affiche."""
    quand = quand or datetime.now()
    return quand.strftime("%H:%M") + INSECABLE + fuseau


def devise_etrangere(valeur_origine, devise_origine: str,
                     valeur_convertie, devise_base: str = "€") -> str:
    """Devise d'origine puis conversion entre parentheses."""
    symboles = {"USD": "$", "GBP": "£", "CHF": "CHF", "JPY": "¥", "EUR": "€"}
    symbole = symboles.get(devise_origine.upper(), devise_origine)
    origine = (symbole + nombre(valeur_origine, 2)
               if symbole in ("$", "£", "¥")
               else nombre(valeur_origine, 2) + INSECABLE + symbole)
    return f"{origine} ({montant(valeur_convertie, devise_base)})"


def perime(texte: str) -> str:
    """Marque une donnee dont la fraicheur n'est pas garantie."""
    return texte + "*"


def signe_couleur(valeur) -> str:
    """Nom du jeton de couleur correspondant au signe d'une valeur."""
    if valeur is None:
        return "flat"
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return "flat"
    if not np.isfinite(v) or v == 0:
        return "flat"
    return "gain" if v > 0 else "loss"


def colorer(valeur, formateur=variation) -> str:
    """
    Nombre colore selon son signe, en HTML.

    Le signe explicite est conserve : la couleur double l'information, elle ne
    la porte pas.
    """
    return (f'<span class="num {signe_couleur(valeur)}">'
            f'{formateur(valeur)}</span>')


def pastille(etat: str) -> str:
    """
    Pastille d'etat de these : pastille ET mot, jamais la couleur seule.

    Un utilisateur daltonien ou un rapport imprime en noir et blanc doivent
    rester lisibles.
    """
    correspondance = {
        "intacte": ("gain", "Intacte"),
        "sous surveillance": ("brass", "À revoir"),
        "à revoir": ("brass", "À revoir"),
        "invalidée": ("loss", "Invalidée"),
        "clôturée": ("flat", "Clôturée"),
        "non vérifiable": ("flat", "Non vérifiable"),
    }
    jeton, libelle = correspondance.get(etat.lower(), ("flat", etat))
    return (f'<span class="pastille pastille-{jeton}">'
            f'<span class="point"></span>{libelle}</span>')


# ==========================================================================
# Feuille de style
# ==========================================================================

CSS = """
<style>
/* ------------------------------------------------------------------
   Retouches minimales.

   Le rendu natif de Streamlit est propre et cohérent : le remplacer par
   une surcouche produit un résultat bâtard, ni l'un ni l'autre. On ne
   touche donc qu'à ce qui sert la lecture des chiffres, et on laisse la
   mise en page tranquille.

   Trois interventions, pas une de plus :
     1. chiffres tabulaires, pour que les colonnes s'alignent
     2. laiton à la place du rouge Streamlit sur les éléments actifs —
        le rouge doit rester disponible pour signifier une perte
     3. classes de couleur sémantique pour les valeurs affichées en HTML
   ------------------------------------------------------------------ */

/* 1. Le chiffre commande la mise en page ---------------------------- */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.num{
  font-variant-numeric:tabular-nums lining-nums;
}
.stNumberInput input{
  text-align:right;
  font-variant-numeric:tabular-nums;
}

/* 2. Laiton pour l'actif : le rouge reste réservé aux données ------- */
.stTabs [aria-selected="true"]{
  color:#8A6A21 !important;
}
.stTabs [data-baseweb="tab-highlight"]{
  background-color:#8A6A21 !important;
}
[data-testid="stMetricDelta"] svg{
  display:none;                    /* la flèche double le signe déjà écrit */
}

/* 3. Couleurs sémantiques, réservées aux valeurs de performance ----- */
.gain{color:#1F6B4D;}
.loss{color:#A93226;}
.flat{color:#797F7A;}

.pastille{
  display:inline-flex;align-items:center;gap:6px;font-size:13px;
}
.pastille .point{
  width:7px;height:7px;border-radius:50%;display:inline-block;
}
.pastille-gain .point{background:#1F6B4D;}
.pastille-loss .point{background:#A93226;}
.pastille-brass .point{background:#8A6A21;}
.pastille-flat .point{background:#797F7A;}

/* Identifiants techniques : ticker, ISIN, horodatage ---------------- */
.ticker,.isin{
  font-family:ui-monospace,"SF Mono",Consolas,monospace;
  font-size:11.5px;opacity:0.7;
}
</style>
"""

CONFIG_TOML = """# Seul l'accent est imposé : le laiton remplace le rouge de Streamlit,
# qui doit rester disponible pour signifier une perte. Les fonds et la
# police restent ceux de Streamlit, qui sont bons.
[theme]
primaryColor = "#8A6A21"

[server]
maxUploadSize = 20
"""


def appliquer(st) -> None:
    """Injecte la feuille de style. A appeler une fois, au demarrage."""
    st.markdown(CSS, unsafe_allow_html=True)
