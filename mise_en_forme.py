"""
Mise en forme des messages Telegram.

Telegram accepte deux syntaxes. Le Markdown echoue des qu'un asterisque ou un
tiret bas n'est pas apparie — ce qui arrive constamment dans un texte redige
librement, et fait perdre le message entier. Le HTML est tolerant a condition
d'echapper `&`, `<` et `>` dans le CONTENU, jamais dans les balises. C'est ce
que fait `echapper()`, applique systematiquement a tout texte variable.

Langage visuel commun a tous les messages :

  - un bandeau en capitales identifie la nature du message d'un coup d'oeil ;
  - le titre en gras porte l'information, jamais le nom de l'emetteur ;
  - un filet separe l'information du contexte ;
  - le pied de message indique la source et l'horodatage.
"""

from __future__ import annotations

import html
from datetime import datetime

FILET = "─────────────"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]


def date_fr(quand: datetime | None = None) -> str:
    """Date en français : Python ne localise pas sans configuration système."""
    quand = quand or datetime.now()
    return (f"{JOURS[quand.weekday()].capitalize()} {quand.day} "
            f"{MOIS[quand.month - 1]}")

BANDEAUX = {
    "signal": ("📈", "SIGNAL"),
    "surveillance": ("👁", "SURVEILLANCE"),
    "urgence": ("🚨", "ALERTE"),
    "echeance": ("📅", "ÉCHÉANCE"),
    "risque": ("⚠️", "RISQUE"),
    "brief": ("☕️", "BRIEF"),
    "semaine": ("🗓", "SEMAINE"),
    "presse": ("📰", "PRESSE"),
    "bilan": ("📊", "BILAN"),
    "macro": ("🏛", "MACRO"),
}


def echapper(texte) -> str:
    """
    Neutralise les caracteres reserves du HTML dans un contenu variable.

    Indispensable : un titre d'article contenant « AT&T » ou « <5 % » ferait
    echouer l'envoi complet sans cette precaution.
    """
    return html.escape(str(texte), quote=False)


def gras(texte) -> str:
    return f"<b>{echapper(texte)}</b>"


def italique(texte) -> str:
    return f"<i>{echapper(texte)}</i>"


def code(texte) -> str:
    return f"<code>{echapper(texte)}</code>"


def lien(url: str, libelle: str) -> str:
    return f'<a href="{echapper(url)}">{echapper(libelle)}</a>'


def bandeau(genre: str, complement: str = "", detenue: bool = False) -> str:
    """
    Ligne d'en-tete identifiant la nature du message.

    Le marqueur de detention compte : une publication sur une ligne detenue
    appelle une decision, la meme sur une valeur simplement suivie n'appelle
    qu'une lecture.
    """
    emoji, libelle = BANDEAUX.get(genre, ("•", genre.upper()))
    texte = f"{emoji} {libelle}"
    if complement:
        texte += f" · {complement}"
    if detenue:
        texte += " · 💼 EN PORTEFEUILLE"
    return f"<b>{texte}</b>"


def pied(source: str = "", horodatage: bool = True) -> str:
    """Pied de message : source et heure, en discret."""
    morceaux = []
    if source:
        morceaux.append(echapper(source))
    if horodatage:
        morceaux.append(datetime.now().strftime("%d/%m %H:%M"))
    return f"<i>{' · '.join(morceaux)}</i>" if morceaux else ""


def assembler(*blocs: str) -> str:
    """Assemble les blocs non vides en les separant d'une ligne blanche."""
    return "\n\n".join(b for b in blocs if b and b.strip())


# ==========================================================================
# Modèles de messages
# ==========================================================================

def message_signal(titre: str, corps: str, ticker: str = "",
                   surveille: bool = False, chiffres: dict | None = None,
                   dates: dict | None = None) -> str:
    """
    Alerte sur une valeur.

    Le titre porte l'information ; le ticker et les chiffres clés viennent
    ensuite, pour permettre une lecture en deux temps — le titre au coup
    d'œil, le détail si l'on s'arrête.
    """
    genre = "surveillance" if surveille else "signal"
    entete = bandeau(genre, ticker)

    ligne_titre = gras(titre)
    texte = echapper(corps)

    bloc_chiffres = ""
    if chiffres:
        propres = [f"{echapper(k)} : {gras(v)}" for k, v in chiffres.items()
                   if v is not None]
        if propres:
            bloc_chiffres = FILET + "\n" + "\n".join(propres)

    bloc_dates = ""
    if dates:
        propres = [f"{echapper(k)} : {gras(v)}" for k, v in dates.items() if v]
        if propres:
            bloc_dates = "🗓 " + gras("Publications") + "\n" + "\n".join(propres)

    return assembler(entete, ligne_titre, texte, bloc_chiffres, bloc_dates,
                     pied("Portefeuille" if not surveille else "Surveillance"))


def message_urgence(titre: str, corps: str, detail: str = "") -> str:
    """Alerte urgente : format resserré, aucun ornement."""
    return assembler(bandeau("urgence"), gras(titre), echapper(corps),
                     italique(detail) if detail else "", pied())


def message_brief(agenda: list[str], marches: list[str],
                  contexte: str = "") -> str:
    """Brief du matin : agenda du jour puis mouvements de la veille."""
    blocs = [bandeau("brief", date_fr())]

    if agenda:
        blocs.append(gras("Aujourd'hui") + "\n"
                     + "\n".join(f"• {echapper(l)}" for l in agenda))
    else:
        blocs.append(italique("Aucune publication majeure aujourd'hui."))

    if marches:
        blocs.append(FILET + "\n" + gras("Marchés") + "\n"
                     + "\n".join(echapper(l) for l in marches))

    if contexte:
        blocs.append(italique(contexte))

    blocs.append(pied())
    return assembler(*blocs)


def message_semaine(jours: dict[str, list[str]], temps_fort: str = "") -> str:
    """Aperçu du lundi : échéances de la semaine, jour par jour."""
    blocs = [bandeau("semaine")]
    for jour, evenements in jours.items():
        blocs.append(gras(jour) + "\n"
                     + "\n".join(f"• {echapper(e)}" for e in evenements))
    if temps_fort:
        blocs.append(FILET + "\n" + gras("Le rendez-vous de la semaine") + "\n"
                     + echapper(temps_fort))
    blocs.append(italique("Dates reconstituées, à vérifier sur les sites officiels."))
    return assembler(*blocs)


def message_article(titre: str, source: str, url: str = "",
                    pourquoi: str = "", libre: bool = True) -> str:
    """Un article, un message."""
    blocs = [bandeau("presse", source)]
    blocs.append(lien(url, titre) if url else gras(titre))
    if pourquoi:
        blocs.append(echapper(pourquoi))
    if not libre:
        blocs.append(italique("Accès abonné probable."))
    blocs.append(pied(horodatage=False))
    return assembler(*blocs)


def message_bilan(performance: dict, faits: list[str],
                  risque: dict, semaine_suivante: list[str]) -> str:
    """Bilan hebdomadaire : le seul message qui prend du recul."""
    blocs = [bandeau("bilan", "Semaine " + datetime.now().strftime("%V"))]

    if performance:
        lignes = [f"{echapper(k)} : {gras(v)}" for k, v in performance.items()]
        blocs.append(gras("Performance") + "\n" + "\n".join(lignes))

    if faits:
        blocs.append(FILET + "\n" + "\n".join(f"• {echapper(f)}" for f in faits))

    if risque:
        lignes = [f"{echapper(k)} : {gras(v)}" for k, v in risque.items()]
        blocs.append(FILET + "\n" + gras("Structure du risque") + "\n"
                     + "\n".join(lignes))

    if semaine_suivante:
        blocs.append(FILET + "\n" + gras("La semaine prochaine") + "\n"
                     + "\n".join(f"• {echapper(e)}" for e in semaine_suivante))

    blocs.append(pied())
    return assembler(*blocs)


def valider(message: str) -> tuple[bool, str]:
    """
    Verifie qu'un message est envoyable.

    Controle la longueur et l'appariement des balises : une balise ouverte non
    fermee fait rejeter tout le message par Telegram.
    """
    if len(message) > 4096:
        return False, f"Message trop long ({len(message)} caractères)"

    for balise in ("b", "i", "code", "a"):
        ouvertes = message.count(f"<{balise}>") + (
            message.count("<a href=") if balise == "a" else 0)
        if balise == "a":
            ouvertes = message.count("<a href=")
        fermees = message.count(f"</{balise}>")
        if ouvertes != fermees:
            return False, f"Balise <{balise}> déséquilibrée ({ouvertes}/{fermees})"
    return True, ""
