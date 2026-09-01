"""
Mise en forme des notifications — qualite institutionnelle.

Trois principes non negociables, repris du cahier des charges.

CHIFFRES EN TABLEAU. Un ecart de consensus noye dans une phrase ne se lit pas.
Les donnees chiffrees sont alignees en colonnes a chasse fixe, lisibles sans
derouler sur mobile.

PROVENANCE EXPLICITE. Ce qui est extrait d'un communique et ce qui est redige
par un modele de langage ne se melangent jamais. Le second est encadre et
etiquete. Une hallucination sur un chiffre d'affaires detruit la confiance
dans l'ensemble du systeme ; la seule protection est de ne jamais laisser un
modele produire un chiffre.

DONNEE MANQUANTE SIGNALEE. Une case vide est une information. On ecrit « non
communique », jamais rien.

Note technique : Telegram accepte le HTML. Le Markdown echoue des qu'un
asterisque n'est pas apparie, ce qui arrive constamment dans un texte redige.
Tout contenu variable passe par `echapper()`.
"""

from __future__ import annotations

import html
from datetime import datetime

# --- Repères visuels
FILET = "━━━━━━━━━━━━━━━━━━━━"
FILET_FIN = "──────────────────"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]

# --- Catégories. L'emoji porte le niveau d'urgence, pas la décoration.
CATEGORIES = {
    "resultats": "📊",
    "prepublication": "📅",
    "guidance_baissee": "🔴",
    "guidance_relevee": "🟢",
    "avertissement": "🔴",
    "operation": "⚡️",
    "direction": "👤",
    "macro": "🏛",
    "risque": "⚠️",
    "presse": "📰",
    "digest": "🗂",
    "dirigeant": "💼",
}

MANQUANT = "n.c."


# ==========================================================================
# Primitives
# ==========================================================================

def echapper(texte) -> str:
    """Neutralise les caracteres reserves du HTML dans tout contenu variable."""
    return html.escape(str(texte), quote=False)


def gras(texte) -> str:
    return f"<b>{echapper(texte)}</b>"


def italique(texte) -> str:
    return f"<i>{echapper(texte)}</i>"


def lien(url: str, libelle: str) -> str:
    return f'<a href="{echapper(url)}">{echapper(libelle)}</a>'


def date_fr(quand=None) -> str:
    quand = quand or datetime.now()
    return f"{quand.day} {MOIS[quand.month - 1]} {quand.year}"


def jour_fr(quand) -> str:
    return f"{JOURS[quand.weekday()]} {quand.day} {MOIS[quand.month - 1]}"


def assembler(*blocs: str) -> str:
    return "\n\n".join(b for b in blocs if b and b.strip())


# ==========================================================================
# Nombres
# ==========================================================================

def nombre(valeur, decimales: int = 2, unite: str = "",
           signe: bool = False) -> str:
    """Formate un nombre, ou renvoie la mention de donnee manquante."""
    if valeur is None:
        return MANQUANT
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return MANQUANT
    if v != v:                                    # NaN
        return MANQUANT
    format_signe = "+" if signe else ""
    return f"{v:{format_signe},.{decimales}f}".replace(",", " ") + unite


def montant(valeur, devise: str = "") -> str:
    """Montant en milliards ou millions, selon son ordre de grandeur."""
    if valeur is None:
        return MANQUANT
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return MANQUANT
    if v != v:
        return MANQUANT
    suffixe = devise or ""
    if abs(v) >= 1e9:
        return f"{v / 1e9:,.2f} Md{suffixe}".replace(",", " ")
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.0f} M{suffixe}".replace(",", " ")
    return f"{v:,.0f}{suffixe}".replace(",", " ")


# ==========================================================================
# Tableaux
# ==========================================================================

def tableau(lignes: list[list[str]], entetes: list[str] | None = None,
            largeurs: list[int] | None = None) -> str:
    """
    Tableau a chasse fixe, aligne en colonnes.

    Telegram rend `<pre>` en police monospace, seul moyen d'obtenir un
    alignement fiable sur mobile. Les largeurs sont contraintes pour tenir
    sur un ecran de telephone sans defilement horizontal.
    """
    if not lignes:
        return ""

    colonnes = max(len(l) for l in lignes)
    if entetes:
        colonnes = max(colonnes, len(entetes))

    if largeurs is None:
        largeurs = []
        for i in range(colonnes):
            contenu = [str(l[i]) if i < len(l) else "" for l in lignes]
            if entetes and i < len(entetes):
                contenu.append(str(entetes[i]))
            largeurs.append(min(max(len(c) for c in contenu), 14))

    def ligne_formatee(cellules, aligner_droite=True):
        morceaux = []
        for i in range(colonnes):
            valeur = str(cellules[i]) if i < len(cellules) else ""
            valeur = valeur[:largeurs[i]]
            morceaux.append(valeur.ljust(largeurs[i]) if i == 0
                            else valeur.rjust(largeurs[i]))
        return " ".join(morceaux).rstrip()

    corps = []
    if entetes:
        corps.append(ligne_formatee(entetes))
        corps.append("─" * min(sum(largeurs) + colonnes - 1, 34))
    corps.extend(ligne_formatee(l) for l in lignes)
    return "<pre>" + echapper("\n".join(corps)) + "</pre>"


def liste_valeurs(elements: dict, largeur_cle: int = 22) -> str:
    """Paires libelle / valeur alignees, pour les blocs courts."""
    lignes = []
    for cle, valeur in elements.items():
        etiquette = str(cle)[:largeur_cle].ljust(largeur_cle, ".")
        lignes.append(f"{etiquette} {valeur}")
    return "<pre>" + echapper("\n".join(lignes)) + "</pre>"


# ==========================================================================
# Structure d'un message
# ==========================================================================

def titre(categorie: str, ticker: str, nature: str,
          detenue: bool = False) -> str:
    """
    Ligne de titre : emoji de categorie, ticker, nature de l'evenement.

    Format impose par le cahier des charges. Exemple :
    🔴 ORA — Guidance 2026 abaissee
    """
    emoji = CATEGORIES.get(categorie, "•")
    marque = " 💼" if detenue else ""
    return f"{emoji} <b>{echapper(ticker)}{marque} — {echapper(nature)}</b>"


def sous_titre(texte: str) -> str:
    return f"<b>{echapper(texte)}</b>"


def encadre_ia(texte: str, modele: str = "") -> str:
    """
    Encadre un texte produit par un modele de langage.

    L'etiquette est obligatoire : le lecteur doit pouvoir distinguer en un
    coup d'oeil un chiffre extrait d'un communique d'une phrase generee.
    """
    if not texte or not texte.strip():
        return ""
    etiquette = "🤖 <i>Analyse générée — non vérifiée</i>"
    if modele:
        etiquette = f"🤖 <i>Analyse générée ({echapper(modele)}) — non vérifiée</i>"
    return f"{etiquette}\n{echapper(texte.strip())}"


def source(nom: str, url: str = "", horodatage: bool = True,
           complement: str = "") -> str:
    """Pied de message : source citee, liee quand elle est accessible."""
    morceaux = []
    if url and nom:
        morceaux.append(f'<a href="{echapper(url)}">{echapper(nom)}</a>')
    elif nom:
        morceaux.append(echapper(nom))
    if complement:
        morceaux.append(echapper(complement))
    if horodatage:
        morceaux.append(datetime.now().strftime("%d/%m %H:%M"))
    return f"<i>Source : {' · '.join(morceaux)}</i>" if morceaux else ""


def note_manquant(elements: list[str]) -> str:
    """
    Signale explicitement les donnees absentes.

    Regle du cahier des charges : une donnee manquante est signalee, jamais
    omise en silence. Savoir ce qu'on ignore fait partie de l'information.
    """
    if not elements:
        return ""
    return italique("Non disponible : " + ", ".join(elements) + ".")


# ==========================================================================
# Contrôle
# ==========================================================================

def valider(message: str) -> tuple[bool, str]:
    """Verifie longueur et appariement des balises avant envoi."""
    if len(message) > 4096:
        return False, f"Message trop long ({len(message)} caractères)"

    for balise in ("b", "i", "pre", "code"):
        ouvertes = message.count(f"<{balise}>")
        fermees = message.count(f"</{balise}>")
        if ouvertes != fermees:
            return False, f"Balise <{balise}> déséquilibrée ({ouvertes}/{fermees})"

    if message.count("<a href=") != message.count("</a>"):
        return False, "Balise <a> déséquilibrée"
    return True, ""


def decouper(message: str, limite: int = 3900) -> list[str]:
    """
    Decoupe un message trop long sur les separations de blocs.

    On ne coupe jamais a l'interieur d'un tableau : cela casserait
    l'alignement et les balises.
    """
    if len(message) <= limite:
        return [message]

    morceaux, courant = [], ""
    for bloc in message.split("\n\n"):
        if len(courant) + len(bloc) + 2 > limite and courant:
            morceaux.append(courant.strip())
            courant = bloc
        else:
            courant += ("\n\n" if courant else "") + bloc
    if courant.strip():
        morceaux.append(courant.strip())
    return morceaux
