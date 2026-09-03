"""
Charte graphique FinexResearch v1.0.

Deux apports distincts, et le second compte davantage que le premier.

L'APPARENCE — jetons de couleur, polices, chiffres tabulaires — s'applique par
injection CSS. Streamlit impose son propre rendu pour certains composants :
la hauteur de ligne exacte des tableaux et le filet appuyé sous les en-têtes
ne sont pas atteignables. Ce qui l'est l'est integralement.

L'ECRITURE DES CHIFFRES est la partie qui empeche reellement une erreur de
lecture, et elle s'applique sans reserve : separateur de milliers en espace
insecable fine, virgule decimale, signe toujours explicite, moins
typographique, cadratin pour une donnee absente plutot qu'un zero trompeur.

Regle structurante : le vert et le rouge ne signifient que gain et perte. Ils
n'apparaissent jamais dans un bouton, un en-tete ou une etiquette. C'est
pourquoi la couleur de marque est un laiton neutre, impossible a confondre
avec un signe.
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
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Archivo:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap');

:root{
  --paper:#F0F0EB; --surface:#FFFFFF; --sunk:#E7E8E2;
  --ink:#191C1A;   --ink-2:#4A504C;   --ink-3:#797F7A;
  --rule:#D4D6CF;  --rule-strong:#B3B7AE;
  --brass:#8A6A21; --brass-ink:#6B5119; --brass-soft:#EFE7D1; --info:#2C5A72;
  --gain:#1F6B4D;  --loss:#A93226;    --flat:#797F7A;
  --s1:#123B4D; --s2:#8A6A21; --s3:#4E7A66;
  --s4:#7A4B63; --s5:#35566E; --s6:#A0602E;
  --serif:"Source Serif 4",Georgia,serif;
  --sans:"Archivo","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --r-sm:2px; --r-md:3px; --r-lg:5px;
  --sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:18px; --sp-5:32px; --sp-6:56px;
  --row-h:36px;
}

/* --- Décor général ------------------------------------------------- */
.stApp{background:var(--paper);color:var(--ink);}
section.main .block-container{max-width:1160px;padding-top:2rem;}
html,body,[class*="css"],.stMarkdown,p,li,label,div[data-baseweb]{
  font-family:var(--sans);font-size:15px;line-height:1.55;color:var(--ink);}

/* --- Titres : voix éditoriale de la recherche ---------------------- */
h1{font-family:var(--serif)!important;font-weight:600!important;
   font-size:52px!important;line-height:1.06!important;
   letter-spacing:-0.025em!important;color:var(--ink)!important;}
h2{font-family:var(--serif)!important;font-weight:600!important;
   font-size:26px!important;line-height:1.2!important;
   letter-spacing:-0.02em!important;color:var(--ink)!important;}
h3{font-family:var(--sans)!important;font-weight:600!important;
   font-size:13px!important;letter-spacing:0.01em!important;
   color:var(--ink-3)!important;text-transform:none!important;}

/* --- Chiffres : le nombre commande la mise en page ----------------- */
.num,[data-testid="stMetricValue"],
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th{
  font-variant-numeric:tabular-nums lining-nums;}
[data-testid="stMetricValue"]{
  font-family:var(--serif)!important;font-weight:600!important;
  font-size:38px!important;line-height:1!important;color:var(--ink)!important;}
[data-testid="stMetricLabel"]{
  font-family:var(--sans)!important;font-size:13px!important;
  font-weight:600!important;color:var(--ink-3)!important;}

/* Le delta d'une métrique est une donnée : vert et rouge y sont légitimes */
[data-testid="stMetricDelta"] svg{display:none;}
[data-testid="stMetricDelta"]{font-size:13.5px;font-variant-numeric:tabular-nums;}

/* --- Tableaux ------------------------------------------------------ */
[data-testid="stDataFrame"]{
  border:1px solid var(--rule);border-radius:0;background:var(--surface);}
[data-testid="stDataFrame"] td{font-size:13.5px;}
[data-testid="stDataFrame"] th{
  font-size:12px;font-weight:600;color:var(--ink-3);
  border-bottom:2px solid var(--rule-strong)!important;}

/* --- Onglets : laiton pour l'actif, jamais de vert ni de rouge ----- */
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--rule);}
.stTabs [data-baseweb="tab"]{
  font-family:var(--sans);font-size:14px;font-weight:500;
  color:var(--ink-2);background:transparent;border-radius:0;
  padding:8px 14px;}
.stTabs [aria-selected="true"]{
  color:var(--brass-ink)!important;
  border-bottom:2px solid var(--brass)!important;background:transparent;}

/* --- Boutons : encre en primaire, laiton au survol ----------------- */
.stButton>button{
  font-family:var(--sans);font-size:14px;font-weight:500;
  border-radius:var(--r-md);border:1px solid var(--rule-strong);
  background:var(--surface);color:var(--ink);transition:all 120ms linear;}
.stButton>button:hover{
  border-color:var(--brass);color:var(--brass-ink);background:var(--brass-soft);}
.stButton>button[kind="primary"]{
  background:var(--ink);color:var(--paper);border-color:var(--ink);}
.stButton>button[kind="primary"]:hover{
  background:var(--brass-ink);border-color:var(--brass-ink);color:var(--paper);}

/* --- Champs : focus laiton avec halo ------------------------------- */
.stTextInput input,.stNumberInput input,.stSelectbox>div>div{
  border-radius:var(--r-md)!important;border-color:var(--rule-strong)!important;
  background:var(--paper)!important;font-family:var(--sans)!important;}
.stTextInput input:focus,.stNumberInput input:focus{
  border-color:var(--brass)!important;
  box-shadow:0 0 0 3px var(--brass-soft)!important;}
.stNumberInput input{text-align:right;font-variant-numeric:tabular-nums;}

/* --- Identifiants : mono réservé aux tickers et horodatages -------- */
code,.ticker,.isin{
  font-family:var(--mono)!important;font-size:11.5px!important;
  color:var(--ink-3)!important;background:transparent!important;}

/* --- Couleurs de données : usage sémantique exclusif --------------- */
.gain{color:var(--gain);}
.loss{color:var(--loss);}
.flat{color:var(--flat);}

/* --- Pastilles d'état : pastille + mot ----------------------------- */
.pastille{
  display:inline-flex;align-items:center;gap:6px;font-size:13px;
  font-family:var(--sans);color:var(--ink-2);}
.pastille .point{
  width:7px;height:7px;border-radius:50%;display:inline-block;}
.pastille-gain .point{background:var(--gain);}
.pastille-loss .point{background:var(--loss);}
.pastille-brass .point{background:var(--brass);}
.pastille-flat .point{background:var(--flat);}

/* --- Alertes : filet gauche, jamais de fond coloré vif ------------- */
.stAlert{border-radius:0;border-left:3px solid var(--brass);
  background:var(--surface);}

/* --- Barre latérale ------------------------------------------------ */
[data-testid="stSidebar"]{background:var(--sunk);border-right:1px solid var(--rule);}

/* --- Rien ne bouge sans raison ------------------------------------- */
*{animation-duration:0s!important;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>
"""

CONFIG_TOML = """[theme]
base = "light"
primaryColor = "#8A6A21"
backgroundColor = "#F0F0EB"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#191C1A"
font = "sans serif"

[server]
maxUploadSize = 20
"""


def appliquer(st) -> None:
    """Injecte la feuille de style. A appeler une fois, au demarrage."""
    st.markdown(CSS, unsafe_allow_html=True)
