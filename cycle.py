"""
Cycle de resultats fonde sur EDGAR.

Choix d'architecture, et il est decisif : le declencheur du jour J est le DEPOT
du formulaire 8-K, pas la mise a jour des comptes chez Yahoo.

La raison tient a la latence. Un 8-K item 2.02 apparait sur EDGAR dans les
minutes qui suivent la publication, avec le lien vers le communique officiel.
Les comptes trimestriels Yahoo arrivent un a trois jours plus tard. Batir
l'alerte du jour J sur Yahoo rendait impossible la contrainte « sous soixante
secondes » du cahier des charges ; la batir sur EDGAR la rend triviale.

Consequence assumee : deux messages au lieu d'un.

  IMMEDIAT   source primaire, latence de quelques minutes. La societe a
             publie, voici le lien vers le communique. Aucun chiffre non
             verifie, aucun texte genere.
  LENDEMAIN  donnees secondaires, une fois Yahoo a jour. Tableau publie
             contre consensus contre N-1, verdict de these, lecture redigee.
             Explicitement etiquete comme secondaire et a recouper.

Asymetrie geographique assumee : EDGAR ne couvre que les Etats-Unis. Sur les
valeurs europeennes, le systeme se limite au calendrier et aux prix. Ce n'est
pas rattrapable gratuitement, et mieux vaut le savoir que le decouvrir.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import officiel as of

SUFFIXES_ETRANGERS = (".PA", ".DE", ".SW", ".AS", ".L", ".MI", ".MC", ".BR",
                      ".ST", ".OL", ".CO", ".HE", ".T", ".HK", ".TO", ".AX")


def est_americaine(ticker: str) -> bool:
    """EDGAR ne couvre que les emetteurs enregistres aux Etats-Unis."""
    return not ticker.upper().endswith(SUFFIXES_ETRANGERS)


def echapper(texte) -> str:
    return html.escape(str(texte), quote=False)


def _tableau(lignes: list[tuple[str, str]], largeur: int = 17) -> str:
    return "<pre>" + "\n".join(
        f"{echapper(k)[:largeur].ljust(largeur, '.')} {echapper(v)}"
        for k, v in lignes) + "</pre>"


def _montant(valeur, devise: str = "", compact: bool = False) -> str:
    """
    Formate un montant.

    En mode compact, l'unite est collee au nombre et le format garanti court :
    une troncature qui transformerait « 17.90 Md » en « 17.90 M » changerait
    le chiffre d'un facteur mille. C'est le genre d'erreur qui decredibilise
    l'ensemble du systeme.
    """
    if valeur is None or (isinstance(valeur, float) and valeur != valeur):
        return "n.c."
    v = float(valeur)
    if compact:
        if abs(v) >= 1e9:
            return f"{v / 1e9:.1f}Md"
        if abs(v) >= 1e6:
            return f"{v / 1e6:.0f}M"
        return f"{v:.0f}"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f} Md{devise}"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.0f} M{devise}"
    return f"{v:,.0f}{devise}".replace(",", " ")


def _pct(valeur, decimales: int = 1, signe: bool = False) -> str:
    if valeur is None or (isinstance(valeur, float) and valeur != valeur):
        return "n.c."
    return f"{float(valeur):{'+' if signe else ''}.{decimales}f} %"


# ==========================================================================
# 1. Alerte immédiate, sur source primaire
# ==========================================================================

def detecter_publications(tickers: list[str], heures: int = 24) -> list[dict]:
    """
    Depots 8-K item 2.02 recents : la societe vient de publier.

    C'est le seul evenement du systeme dont la latence se compte en minutes.
    """
    trouvees = []
    for ticker in tickers:
        if not est_americaine(ticker):
            continue
        for depot in of.depots(ticker, jours=max(1, heures // 24)):
            if depot["formulaire"] == "8-K" and depot["item"] == "2.02":
                trouvees.append(depot)
    return trouvees


def message_publication_immediate(depot: dict, strate: str,
                                  attentes: dict | None = None) -> str:
    """
    Message immediat : la source primaire, sans aucun chiffre non verifie.

    Le consensus est rappele parce qu'il etait connu AVANT la publication et
    n'a donc pas besoin d'etre extrait du communique. Les chiffres publies,
    eux, ne figurent pas : ils viendront le lendemain, une fois verifiables.
    """
    marque = " 💼" if strate == "A" else ""
    blocs = [f"📊 <b>{echapper(depot['ticker'])}{marque} — Résultats publiés</b>",
             f"<b>{depot['date'].strftime('%d/%m/%Y')}</b>"]

    attentes = attentes or {}
    lignes = []
    if attentes.get("ca_attendu"):
        texte = _montant(attentes["ca_attendu"])
        if attentes.get("ca_croissance_pct") is not None:
            texte += f" ({_pct(attentes['ca_croissance_pct'], 1, True)})"
        lignes.append(("CA attendu", texte))
    if attentes.get("bpa_attendu"):
        lignes.append(("BPA attendu", f"{attentes['bpa_attendu']:.2f}"))
    if attentes.get("bpa_analystes"):
        lignes.append(("Analystes", str(int(attentes["bpa_analystes"]))))

    if lignes:
        blocs.append("<b>Ce que le marché attendait</b>\n" + _tableau(lignes))

    blocs.append(f'<a href="{echapper(depot["lien"])}">'
                 f'Communiqué officiel (8-K item 2.02)</a>')
    blocs.append("<i>Source primaire SEC. Les chiffres publiés seront "
                 "confrontés au consensus demain, une fois les comptes "
                 "disponibles.</i>")
    return "\n\n".join(blocs)


# ==========================================================================
# 2. Message enrichi du lendemain
# ==========================================================================

def message_chiffres(ticker: str, strate: str, publie: dict, attendu: dict,
                     bpa: dict, verdicts: list[dict],
                     lecture: str = "") -> str:
    """
    Tableau publie / consensus / N-1, verdict de these, puis lecture.

    Les chiffres sont extraits des comptes ; la lecture est produite par un
    modele de langage et etiquetee comme telle. Les deux ne se melangent
    jamais : une hallucination sur un chiffre d'affaires detruirait la
    confiance dans l'ensemble.
    """
    marque = " 💼" if strate == "A" else ""
    blocs = [f"📊 <b>{echapper(ticker)}{marque} — Chiffres et lecture</b>"]

    if publie.get("periode") is not None:
        blocs.append(f"<b>Trimestre clos le "
                     f"{pd.Timestamp(publie['periode']).strftime('%d/%m/%Y')}</b>")

    # --- Tableau principal
    colonnes = []

    def _ajouter(libelle, valeur, consensus, n1, formateur):
        if valeur is None:
            return
        colonnes.append((libelle, formateur(valeur),
                         formateur(consensus) if consensus is not None else "n.c.",
                         formateur(n1) if n1 is not None else "n.c."))

    compact = lambda v: _montant(v, compact=True)
    _ajouter("CA", publie.get("ca"), attendu.get("ca_attendu"),
             publie.get("ca_n1"), compact)
    _ajouter("BPA", bpa.get("bpa"), bpa.get("bpa_attendu"), None,
             lambda v: f"{v:.2f}")
    _ajouter("Rés. net", publie.get("resultat_net"), None,
             publie.get("resultat_net_n1"), compact)
    _ajouter("Marge n.", publie.get("marge_nette_pct"), None,
             publie.get("marge_nette_n1_pct"), lambda v: f"{v:.1f}%")
    _ajouter("Marge op.", publie.get("marge_op_pct"), None,
             publie.get("marge_op_n1_pct"), lambda v: f"{v:.1f}%")
    _ajouter("FCF", publie.get("fcf"), None, publie.get("fcf_n1"), compact)
    _ajouter("Dette n.", publie.get("dette_nette"), None, None, compact)

    if colonnes:
        # Largeurs calculées sur le contenu réel : aucune troncature possible
        largeur_libelle = max(len(c[0]) for c in colonnes)
        largeurs = [max(len(c[i]) for c in colonnes) for i in (1, 2, 3)]
        largeurs = [max(l, len(t)) for l, t in
                    zip(largeurs, ("Publié", "Cons.", "N-1"))]

        entete = (" " * largeur_libelle + " "
                  + " ".join(t.rjust(l) for t, l in
                             zip(("Publié", "Cons.", "N-1"), largeurs)))
        lignes = [entete, "─" * len(entete)]
        for libelle, valeur, consensus, n1 in colonnes:
            lignes.append(libelle.ljust(largeur_libelle) + " " + " ".join(
                v.rjust(l) for v, l in zip((valeur, consensus, n1), largeurs)))
        blocs.append("<pre>" + echapper("\n".join(lignes)) + "</pre>")

    # --- Écarts
    ecarts = []
    if publie.get("ca") and attendu.get("ca_attendu"):
        ecarts.append(("Écart CA",
                       _pct(publie["ca"] / attendu["ca_attendu"] * 100 - 100, 1, True)))
    if bpa.get("surprise_pct") is not None:
        ecarts.append(("Surprise BPA", _pct(bpa["surprise_pct"], 1, True)))
    if publie.get("ca_croissance_pct") is not None:
        ecarts.append(("Croissance CA", _pct(publie["ca_croissance_pct"], 1, True)))
    if (publie.get("marge_nette_pct") is not None
            and publie.get("marge_nette_n1_pct") is not None):
        ecarts.append(("Marge vs N-1",
                       f"{publie['marge_nette_pct'] - publie['marge_nette_n1_pct']:+.1f} pt"))
    if publie.get("levier") is not None:
        ecarts.append(("Levier", f"{publie['levier']:.1f}x"))
    if ecarts:
        blocs.append("<b>Écarts</b>\n" + _tableau(ecarts))

    # --- Verdict de thèse
    if verdicts:
        lignes = []
        for verdict in verdicts:
            marque_v = {"validé": "✅", "invalidé": "❌",
                        "indisponible": "⬜️"}[verdict["statut"]]
            valeur = ("n.c." if verdict["valeur"] is None
                      else f"{verdict['valeur']:.1f}")
            lignes.append(f"{marque_v} {verdict['libelle'][:20]} : {valeur} "
                          f"(seuil {verdict['operateur']}{verdict['seuil']:g})")
        blocs.append("<b>Verdict de thèse</b>\n" + echapper("\n".join(lignes)))

    # --- Lecture générée, séparée et étiquetée
    if lecture:
        blocs.append("🤖 <i>Analyse générée — non vérifiée</i>\n"
                     + echapper(lecture))

    absents = list(dict.fromkeys(publie.get("absents", [])
                                 + attendu.get("absents", [])))
    if absents:
        blocs.append("<i>Non disponible : " + echapper(", ".join(absents[:6]))
                     + ".</i>")

    blocs.append("<i>Chiffres Yahoo Finance (source secondaire, à recouper "
                 "avec le communiqué). Aucun chiffre généré.</i>")
    return "\n\n".join(blocs)


# ==========================================================================
# 3. Cycle avant publication
# ==========================================================================

def message_ouverture(ticker: str, strate: str, date_publication,
                      jours: int, attendu: dict, revisions: dict,
                      hist: dict, options: dict,
                      performance: dict | None = None) -> str:
    """
    Message J-15 ou J-5.

    A J-5, la dynamique de revision passe en tete : les analystes finalisent
    leurs estimations dans les deux dernieres semaines, et c'est le champ le
    plus informatif du cycle.
    """
    marque = " 💼" if strate == "A" else ""
    titre = "Ouverture du dossier" if jours > 7 else "Mise à jour"
    blocs = [f"📅 <b>{echapper(ticker)}{marque} — {titre} (J-{jours})</b>",
             f"<b>Publication le "
             f"{pd.Timestamp(date_publication).strftime('%d/%m/%Y')}</b>"]

    # --- Révisions en tête à J-5
    lignes_revision = []
    for libelle, prefixe in [("Trimestre", "trimestre"), ("Exercice", "exercice")]:
        for jours_ref, cle in [(30, "30j"), (90, "90j")]:
            valeur = revisions.get(f"{prefixe}_revision_{cle}_pct")
            if valeur is not None:
                lignes_revision.append(
                    (f"{libelle} {jours_ref} j", _pct(valeur, 1, True)))
    if lignes_revision and jours <= 7:
        blocs.append("<b>Révision du BPA</b>\n" + _tableau(lignes_revision))

    # --- Consensus
    lignes = []
    if attendu.get("ca_attendu"):
        texte = _montant(attendu["ca_attendu"])
        if attendu.get("ca_croissance_pct") is not None:
            texte += f" ({_pct(attendu['ca_croissance_pct'], 1, True)})"
        lignes.append(("CA attendu", texte))
    if attendu.get("bpa_attendu"):
        texte = f"{attendu['bpa_attendu']:.2f}"
        if attendu.get("bpa_croissance_pct") is not None:
            texte += f" ({_pct(attendu['bpa_croissance_pct'], 1, True)})"
        lignes.append(("BPA attendu", texte))
    if attendu.get("bpa_analystes"):
        lignes.append(("Analystes", str(int(attendu["bpa_analystes"]))))
    if attendu.get("ca_dispersion_pct") is not None:
        lignes.append(("Dispersion CA", _pct(attendu["ca_dispersion_pct"])))
    if lignes:
        blocs.append("<b>Consensus</b>\n" + _tableau(lignes))

    if lignes_revision and jours > 7:
        blocs.append("<b>Révision du BPA</b>\n" + _tableau(lignes_revision))

    # --- Déjà dans le prix
    lignes_prix = []
    if performance:
        for libelle, cle in [("Perf 1 mois", "perf_1m"), ("Perf 3 mois", "perf_3m")]:
            if performance.get(f"{cle}_pct") is not None:
                texte = _pct(performance[f"{cle}_pct"], 1, True)
                relatif = performance.get(f"{cle}_relative_pct")
                if relatif is not None:
                    texte += f" ({_pct(relatif, 1, True)} rel.)"
                lignes_prix.append((libelle, texte))
    if options.get("mouvement_pct"):
        lignes_prix.append(("Mvt implicite", f"±{options['mouvement_pct']:.1f} %"))
    if lignes_prix:
        blocs.append("<b>Déjà dans le prix</b>\n" + _tableau(lignes_prix))

    # --- Historique
    if hist.get("lignes"):
        lignes_hist = ["Date    Surprise    J+1", "─" * 26]
        for element in hist["lignes"][:8]:
            surprise = ("n.c." if element["surprise"] != element["surprise"]
                        else f"{element['surprise']:+.1f}%")
            reaction = ("n.c." if element["reaction"] != element["reaction"]
                        else f"{element['reaction']:+.1f}%")
            lignes_hist.append(
                f"{element['date'].strftime('%m/%y')}  {surprise.rjust(8)} "
                f"{reaction.rjust(8)}")
        blocs.append("<b>8 derniers trimestres</b>\n<pre>"
                     + echapper("\n".join(lignes_hist)) + "</pre>")
        if hist.get("lecture"):
            blocs.append("<i>" + echapper(hist["lecture"].capitalize()) + "</i>")

    absents = attendu.get("absents", [])
    if absents:
        blocs.append("<i>Non disponible : " + echapper(", ".join(absents[:5]))
                     + ".</i>")
    return "\n\n".join(blocs)


# ==========================================================================
# 4. Communiqués officiels hors résultats
# ==========================================================================

def message_communique(depot: dict, strate: str) -> str:
    """Communique classe par code d'item, avec lien vers la source."""
    emoji = {"P1": "🔴", "P2": "⚡️", "P3": "📄"}.get(depot["priorite"], "•")
    marque = " 💼" if strate == "A" else ""
    blocs = [f"{emoji} <b>{echapper(depot['ticker'])}{marque} — "
             f"{echapper(depot['libelle'])}</b>",
             f"<b>{depot['date'].strftime('%d/%m/%Y')}</b>"]

    details = [("Formulaire", depot["formulaire"])]
    if depot.get("item"):
        details.append(("Item", depot["item"]))
    details.append(("Priorité", depot["priorite"]))
    blocs.append(_tableau(details, largeur=12))

    blocs.append(f'<a href="{echapper(depot["lien"])}">Document officiel</a>')
    if depot.get("note"):
        blocs.append("<i>" + echapper(depot["note"]) + "</i>")
    blocs.append("<i>Source : dépôt SEC (EDGAR)</i>")
    return "\n\n".join(blocs)


def message_macro(publication: dict) -> str:
    """Statistique macroeconomique inhabituelle."""
    blocs = [f"🏛 <b>{echapper(publication['libelle'])}</b>",
             f"<b>{publication['date'].strftime('%d/%m/%Y')}</b>"]
    blocs.append(_tableau([
        ("Valeur", f"{publication['valeur']:.2f}"),
        ("Précédent", f"{publication['precedent']:.2f}"),
        ("Variation", f"{publication['variation']:+.2f} {publication['unite']}"),
        ("Écart", f"{publication['ecarts_types']:+.1f} σ"),
    ]))
    blocs.append("<i>" + echapper(publication["note"]) + "</i>")
    blocs.append("<i>Source : Réserve fédérale de Saint-Louis (FRED)</i>")
    return "\n\n".join(blocs)
