"""
Veille evenementielle — script d'execution.

Envoie un message par evenement, jamais de bloc. S'il ne se passe rien, aucun
message ne part : c'est le fonctionnement normal, pas une panne.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import evenements as ev
import mise_en_forme as mf
import publications as pu

PLAFOND = int(os.environ.get("PLAFOND_MESSAGES", "12"))


def envoyer(message: str) -> bool:
    jeton = os.environ.get("TELEGRAM_JETON", "").strip()
    destinataire = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    if not jeton or not destinataire:
        print("Telegram non configuré.\n" + message, file=sys.stderr)
        return False

    valide, motif = mf.valider(message)
    if not valide:
        print(f"Message écarté : {motif}", file=sys.stderr)
        return False

    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": destinataire, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20)
        if not reponse.ok:
            print(f"Telegram a refusé (code {reponse.status_code}) : "
                  f"{reponse.text[:250]}", file=sys.stderr)
        return reponse.ok
    except Exception as erreur:
        print(f"Échec Telegram : {erreur}", file=sys.stderr)
        return False


def message_publication(publication: dict, attentes: dict | None = None,
                        nom: str = "", detenue: bool = False) -> str:
    """
    Annonce d'une publication a venir, avec ce que le marche attend.

    Le consensus est la reference contre laquelle les chiffres seront jugés :
    un resultat en hausse mais sous les attentes fait baisser le cours.
    """
    jours = publication["jours"]
    quand = ("aujourd'hui" if jours == 0 else
             "demain" if jours == 1 else f"dans {jours} jours")
    attentes = attentes or {}
    devise = attentes.get("devise", "")

    date_publication = publication["date"]
    jour_fr = mf.JOURS[date_publication.dayofweek]
    mois_fr = mf.MOIS[date_publication.month - 1]
    lignes = [f"Publication des résultats {quand}, "
              f"{jour_fr} {date_publication.day} {mois_fr}."]

    consensus = []
    if attentes.get("ca_attendu"):
        texte = f"Chiffre d'affaires : {pu.montant(attentes['ca_attendu'], devise)}"
        if attentes.get("ca_croissance_pct") is not None:
            texte += f" ({attentes['ca_croissance_pct']:+.1f} % sur un an)"
        consensus.append(texte)

    if attentes.get("bpa_attendu"):
        texte = f"Bénéfice par action : {attentes['bpa_attendu']:.2f}"
        if attentes.get("bpa_croissance_pct") is not None:
            texte += f" ({attentes['bpa_croissance_pct']:+.1f} % sur un an)"
        consensus.append(texte)

    if attentes.get("resultat_net_attendu"):
        consensus.append("Résultat net implicite : "
                         + pu.montant(attentes["resultat_net_attendu"], devise))

    if attentes.get("marge_nette_attendue_pct") is not None:
        texte = f"Marge nette implicite : {attentes['marge_nette_attendue_pct']:.1f} %"
        an_dernier = attentes.get("marge_nette_an_dernier_pct")
        if an_dernier is not None:
            ecart = attentes["marge_nette_attendue_pct"] - an_dernier
            texte += f" (contre {an_dernier:.1f} % un an plus tôt, {ecart:+.1f} pt)"
        consensus.append(texte)

    if attentes.get("ca_bas") and attentes.get("ca_haut"):
        consensus.append(
            f"Fourchette des estimations de CA : "
            f"{pu.montant(attentes['ca_bas'])} à {pu.montant(attentes['ca_haut'])}")

    analystes = attentes.get("bpa_analystes") or attentes.get("ca_analystes")

    blocs = [mf.bandeau("echeance", publication["ticker"], detenue),
             mf.gras(publication["date"].strftime("%d/%m/%Y")),
             mf.echapper(" ".join(lignes))]

    if consensus:
        blocs.append(mf.FILET + "\n" + mf.gras("Consensus des analystes") + "\n"
                     + "\n".join(f"• {mf.echapper(c)}" for c in consensus))
    else:
        blocs.append(mf.italique("Consensus détaillé indisponible pour cette "
                                 "valeur."))

    pied = "Source : calendrier de la société"
    if analystes:
        pied += f" · consensus de {int(analystes)} analystes"
    blocs.append(mf.italique(pied))
    return mf.assembler(*blocs)


def message_resultat(resultat: dict, publie: dict | None = None,
                     attendu: dict | None = None, ecarts: dict | None = None,
                     lecture: str = "", nom: str = "",
                     detenue: bool = False) -> str:
    """Resultats publies : les chiffres, les ecarts, puis la lecture."""
    publie, attendu, ecarts = publie or {}, attendu or {}, ecarts or {}
    devise = attendu.get("devise", "")

    surprise = resultat.get("surprise_pct")
    if surprise is None:
        titre = "Résultats publiés"
    else:
        sens = "au-dessus" if surprise > 0 else "en dessous"
        titre = f"Résultats {sens} du consensus ({surprise:+.1f} %)"

    chiffres = []
    if resultat.get("bpa_publie") is not None:
        texte = f"Bénéfice par action : {resultat['bpa_publie']:.2f}"
        if resultat.get("bpa_attendu") is not None:
            texte += f" (attendu {resultat['bpa_attendu']:.2f})"
        chiffres.append(texte)

    if publie.get("ca_publie"):
        texte = f"Chiffre d'affaires : {pu.montant(publie['ca_publie'], devise)}"
        if publie.get("ca_croissance_pct") is not None:
            texte += f" ({publie['ca_croissance_pct']:+.1f} % sur un an)"
        if ecarts.get("ecart_ca_pct") is not None:
            texte += f", {ecarts['ecart_ca_pct']:+.1f} % vs consensus"
        chiffres.append(texte)

    if publie.get("resultat_net_publie"):
        chiffres.append("Résultat net : "
                        + pu.montant(publie["resultat_net_publie"], devise))

    if publie.get("marge_nette_pct") is not None:
        texte = f"Marge nette : {publie['marge_nette_pct']:.1f} %"
        if ecarts.get("evolution_marge_points") is not None:
            texte += f" ({ecarts['evolution_marge_points']:+.1f} pt sur un an)"
        chiffres.append(texte)

    if publie.get("marge_operationnelle_pct") is not None:
        chiffres.append("Marge opérationnelle : "
                        f"{publie['marge_operationnelle_pct']:.1f} %")

    blocs = [mf.bandeau("signal", resultat["ticker"], detenue),
             mf.gras(titre),
             mf.echapper(f"Publié le {resultat['date'].strftime('%d/%m/%Y')}.")]

    if chiffres:
        blocs.append(mf.FILET + "\n"
                     + "\n".join(f"• {mf.echapper(c)}" for c in chiffres))
    if lecture:
        blocs.append(mf.FILET + "\n" + mf.gras("Lecture") + "\n"
                     + mf.echapper(lecture))

    blocs.append(mf.italique(
        "Source : publication officielle · chiffres via Yahoo Finance"))
    return mf.assembler(*blocs)


def message_revision(revision: dict, detenue: bool = False) -> str:
    """Revision : periode concernee et ampleur, sans interpretation."""
    periodes = {"0q": "trimestre en cours", "0y": "exercice en cours",
                "+1y": "exercice prochain", "+1q": "trimestre prochain"}
    periode = periodes.get(revision["periode"], revision["periode"])
    sens = "relevée" if revision["variation_pct"] > 0 else "abaissée"

    return mf.assembler(
        mf.bandeau("signal", revision["ticker"], detenue),
        mf.gras(f"Prévision {sens} de "
                f"{abs(revision['variation_pct']):.1f} % sur un mois"),
        mf.echapper(f"Estimation de bénéfice pour l'{periode} : "
                    f"{revision['estimation']:.2f} par action."),
        mf.italique("Source : consensus des analystes, via Yahoo Finance"))


def message_actualite(article: dict, detenue: bool = False) -> str:
    """Depeche : date et heure, source, puis l'information."""
    horodatage = (article["date"].strftime("%d/%m/%Y à %H:%M")
                  if article.get("date") else "")

    blocs = [mf.bandeau("presse", article["ticker"], detenue)]
    if horodatage:
        blocs.append(mf.gras(horodatage))
    blocs.append(mf.lien(article["lien"], article["titre"])
                 if article.get("lien") else mf.gras(article["titre"]))
    blocs.append(mf.italique(f"Source : {article['source']}"))
    return mf.assembler(*blocs)


def principal() -> int:
    tickers = ev.charger_liste()
    detenues = ev.charger_detenues()
    if not tickers:
        print("Aucune valeur à surveiller. Renseigne TICKERS_SURVEILLANCE.",
              file=sys.stderr)
        return 1

    jours_avant = int(os.environ.get("JOURS_AVANT_PUBLICATION", "3"))
    print(f"{len(tickers)} valeur(s) surveillée(s), publications à "
          f"{jours_avant} jours.")

    resultat = ev.collecter(tickers, jours_avant)
    print(f"{len(resultat['resultats'])} résultat(s) publié(s), "
          f"{len(resultat['publications'])} publication(s) approchante(s), "
          f"{len(resultat['revisions'])} révision(s), "
          f"{len(resultat['actualites'])} dépêche(s) du jour, "
          f"{resultat['echecs']} valeur(s) en échec.")

    # Les échéances passent avant les dépêches : elles appellent une décision
    # Ordre de priorité : ce qui est daté et engageant d'abord, le contexte
    # ensuite. Les résultats publiés priment sur tout : c'est le fait brut.
    evenements = (
        [{"type": "resultat", "donnees": r} for r in resultat["resultats"]]
        + [{"type": "publication", "donnees": p} for p in resultat["publications"]]
        + [{"type": "revision", "donnees": r} for r in resultat["revisions"]]
        + [{"type": "actualite", "donnees": a} for a in resultat["actualites"]])

    # À type d'événement égal, une position détenue passe devant
    rang = {"resultat": 0, "publication": 1, "revision": 2, "actualite": 3}
    evenements.sort(key=lambda e: (
        rang.get(e["type"], 9),
        0 if e["donnees"].get("ticker") in detenues else 1))

    nouveaux = ev.filtrer_nouveaux([e["donnees"] for e in evenements])
    cles_nouvelles = {id(n) for n in nouveaux}
    evenements = [e for e in evenements if id(e["donnees"]) in cles_nouvelles]
    print(f"{len(evenements)} événement(s) non encore transmis.")

    if len(evenements) > PLAFOND:
        print(f"Plafond de {PLAFOND} : {len(evenements) - PLAFOND} "
              f"événement(s) reporté(s).")
        evenements = evenements[:PLAFOND]

    envois = 0
    for evenement in evenements:
        donnees = evenement["donnees"]
        genre = evenement["type"]
        est_detenue = donnees.get("ticker") in detenues

        if genre == "publication":
            # Consensus complet : chiffre d'affaires, bénéfice, marge implicite
            attentes = pu.consensus(donnees["ticker"])
            message = message_publication(donnees, attentes,
                                          detenue=est_detenue)

        elif genre == "resultat":
            ticker = donnees["ticker"]
            publie = pu.chiffres_publies(ticker)
            attendu = pu.consensus(ticker)
            ecarts = pu.confronter(publie, attendu)
            titres = [a["titre"] for a in resultat["actualites"]
                      if a["ticker"] == ticker][:5]
            lecture = pu.rediger_lecture(ticker, ticker, publie, attendu,
                                         ecarts, titres)
            message = message_resultat(donnees, publie, attendu, ecarts,
                                       lecture, detenue=est_detenue)

        elif genre == "revision":
            message = message_revision(donnees, est_detenue)
        else:
            message = message_actualite(donnees, est_detenue)
        if envoyer(message):
            envois += 1
            import time
            time.sleep(1)

    print(f"{envois} message(s) envoyé(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
