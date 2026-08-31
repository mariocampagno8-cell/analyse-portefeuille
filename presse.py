"""
Revue de presse financiere.

Avertissement sur les sources. Bloomberg et Reuters ont supprime leurs flux
RSS publics et placent l'essentiel de leur production derriere un peage. On
passe donc par Google News pour recuperer leurs titres — utile pour savoir ce
qui se dit, frustrant a la lecture puisque le lien butera souvent sur un mur
d'abonnement. Les sources librement lisibles sont signalees par `libre=True`.

Principe de selection : on ne retient pas ce qui est recent, on retient ce qui
est PERTINENT. Un fil d'actualite brut noie l'essentiel sous les depeches de
routine. Chaque article est donc note sur son titre, et seuls les mieux
places sont transmis.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

FICHIER_VUS = Path(__file__).parent / "articles_vus.json"
MEMOIRE_JOURS = 5

EN_TETES = {"User-Agent": "Mozilla/5.0 (compatible; VeillePortefeuille/1.0)"}


# ==========================================================================
# Sources
# ==========================================================================

SOURCES = [
    # --- Francophones
    {"nom": "Les Échos — Marchés", "libre": False,
     "url": "https://services.lesechos.fr/rss/les-echos-finance-marches.xml"},
    {"nom": "La Tribune — Économie", "libre": True,
     "url": "https://www.latribune.fr/rss/rubriques/economie.html"},
    {"nom": "Le Monde — Économie", "libre": False,
     "url": "https://www.lemonde.fr/economie/rss_full.xml"},
    {"nom": "Boursorama", "libre": True,
     "url": "https://www.boursorama.com/rss/actualites"},

    # --- Anglophones en accès libre
    {"nom": "CNBC — Marchés", "libre": True,
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml"
            "?partnerId=wrss01&id=20910258"},
    {"nom": "MarketWatch", "libre": True,
     "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    {"nom": "Investing.com", "libre": True,
     "url": "https://www.investing.com/rss/news_285.rss"},

    # --- Institutions : source primaire, toujours accessible
    {"nom": "BCE — Communiqués", "libre": True,
     "url": "https://www.ecb.europa.eu/rss/press.html"},
    {"nom": "Réserve fédérale", "libre": True,
     "url": "https://www.federalreserve.gov/feeds/press_all.xml"},

    # --- Via Google News : titres accessibles, articles souvent payants
    {"nom": "Reuters (via Google News)", "libre": False,
     "url": "https://news.google.com/rss/search?q=site:reuters.com+"
            "(markets+OR+economy+OR+fed+OR+ecb)+when:2d&hl=fr&gl=FR&ceid=FR:fr"},
    {"nom": "Bloomberg (via Google News)", "libre": False,
     "url": "https://news.google.com/rss/search?q=site:bloomberg.com+"
            "(markets+OR+economy+OR+fed)+when:2d&hl=fr&gl=FR&ceid=FR:fr"},
    {"nom": "Financial Times (via Google News)", "libre": False,
     "url": "https://news.google.com/rss/search?q=site:ft.com+"
            "(markets+OR+economy)+when:2d&hl=fr&gl=FR&ceid=FR:fr"},
]


# Mots-clés de pertinence. Le poids reflète l'importance pour un investisseur,
# pas la fréquence du terme.
POIDS = {
    # Politique monétaire — ce qui déplace le plus les marchés
    "fed": 5, "fomc": 5, "bce": 5, "ecb": 5, "powell": 4, "lagarde": 4,
    "taux directeur": 5, "interest rate": 4, "rate cut": 5, "rate hike": 5,
    "baisse des taux": 5, "hausse des taux": 5, "politique monétaire": 4,
    "monetary policy": 4, "quantitative": 3,
    # Inflation et croissance
    "inflation": 4, "cpi": 4, "pce": 4, "déflation": 4, "récession": 5,
    "recession": 5, "pib": 3, "gdp": 3, "chômage": 3, "unemployment": 3,
    "payrolls": 4, "emploi": 3, "croissance": 2,
    # Marchés
    "obligataire": 3, "bond": 3, "yield": 3, "rendement": 2, "courbe": 3,
    "krach": 5, "crash": 5, "correction": 3, "volatilité": 3, "vix": 3,
    "bear market": 4, "bull market": 3, "record": 2, "rally": 2,
    "sell-off": 4, "plongeon": 3, "chute": 2,
    # Risques systémiques
    "crise": 4, "défaut": 4, "default": 4, "faillite": 3, "bankruptcy": 3,
    "contagion": 4, "liquidité": 3, "banque centrale": 4,
    # Géopolitique et commerce
    "tarif": 3, "tariff": 3, "sanctions": 3, "guerre commerciale": 4,
    "trade war": 4, "pétrole": 2, "oil": 2, "opec": 3, "opep": 3,
    # Entreprises
    "résultats": 2, "earnings": 2, "guidance": 3, "avertissement": 3,
    "profit warning": 4, "fusion": 2, "acquisition": 2, "ipo": 2,
}

# Bruit à écarter : sujets sans portée pour un investisseur
REJET = [
    "horoscope", "recette", "people", "football", "cinéma", "série",
    "météo", "bon plan", "promo", "code promo", "publi-", "sponsorisé",
    "sport", "tennis", "rugby",
]


# ==========================================================================
# Récupération
# ==========================================================================

def _texte(element, chemin: str) -> str:
    trouve = element.find(chemin)
    return html.unescape(trouve.text.strip()) if trouve is not None and trouve.text else ""


def _date(brut: str) -> datetime | None:
    """Analyse les formats de date rencontrés dans les flux RSS."""
    for format_date in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            valeur = datetime.strptime(brut.strip(), format_date)
            return valeur if valeur.tzinfo else valeur.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def lire_flux(source: dict, limite: int = 25) -> list[dict]:
    """Récupère et analyse un flux RSS. Renvoie une liste vide en cas d'échec."""
    try:
        reponse = requests.get(source["url"], headers=EN_TETES, timeout=15)
        if not reponse.ok:
            return []
        racine = ET.fromstring(reponse.content)
    except Exception as erreur:
        print(f"  {source['nom']} : {type(erreur).__name__}", file=sys.stderr)
        return []

    articles = []
    for item in list(racine.iter("item"))[:limite]:
        titre = _texte(item, "title")
        if not titre:
            continue
        articles.append({
            "titre": titre,
            "lien": _texte(item, "link"),
            "date": _date(_texte(item, "pubDate")),
            "source": source["nom"],
            "libre": source["libre"],
            "resume": re.sub(r"<[^>]+>", "", _texte(item, "description"))[:300],
        })
    return articles


def noter(article: dict) -> int:
    """
    Note de pertinence, fondée sur le titre et le résumé.

    Le titre pèse double : c'est là que se trouve l'information, le résumé
    étant souvent une accroche générique.
    """
    titre = article["titre"].lower()
    resume = article.get("resume", "").lower()

    if any(mot in titre for mot in REJET):
        return 0

    note = sum(poids * 2 for mot, poids in POIDS.items() if mot in titre)
    note += sum(poids for mot, poids in POIDS.items() if mot in resume)

    if article.get("libre"):
        note += 2                      # à pertinence égale, préférer le lisible
    if "communiqué" in article["source"].lower() or "Réserve" in article["source"]:
        note += 3                      # source primaire
    return note


def collecter(note_minimale: int = 6, maximum: int = 12) -> list[dict]:
    """Parcourt toutes les sources et renvoie les articles les mieux notés."""
    tous = []
    for source in SOURCES:
        articles = lire_flux(source)
        print(f"  {source['nom']} : {len(articles)} article(s)")
        tous.extend(articles)

    # Déduplication sur les premiers mots du titre : les dépêches circulent
    vus, uniques = set(), []
    for article in tous:
        empreinte = " ".join(re.sub(r"[^\w\s]", "", article["titre"].lower())
                             .split()[:7])
        if empreinte in vus:
            continue
        vus.add(empreinte)
        article["note"] = noter(article)
        if article["note"] >= note_minimale:
            uniques.append(article)

    recents = [a for a in uniques
               if a["date"] is None
               or a["date"] > datetime.now(timezone.utc) - timedelta(days=2)]
    return sorted(recents, key=lambda a: a["note"], reverse=True)[:maximum]


# ==========================================================================
# Mémoire des articles déjà transmis
# ==========================================================================

def _cle(article: dict) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", article["titre"].lower()).split()[:7])


def filtrer_nouveaux(articles: list[dict]) -> list[dict]:
    """Écarte les articles déjà envoyés au cours des derniers jours."""
    vus = {}
    if FICHIER_VUS.exists():
        try:
            vus = json.loads(FICHIER_VUS.read_text())
        except Exception:
            pass

    limite = datetime.now(timezone.utc) - timedelta(days=MEMOIRE_JOURS)
    vus = {k: v for k, v in vus.items()
           if datetime.fromisoformat(v) > limite}

    nouveaux = [a for a in articles if _cle(a) not in vus]
    maintenant = datetime.now(timezone.utc).isoformat()
    for article in nouveaux:
        vus[_cle(article)] = maintenant

    try:
        FICHIER_VUS.write_text(json.dumps(vus, indent=1, ensure_ascii=False))
    except Exception:
        pass
    return nouveaux


# ==========================================================================
# Sélection par Claude
# ==========================================================================

CONSIGNES = """Tu sélectionnes des articles pour un investisseur français qui \
suit les marchés de près.

Parmi la liste fournie, retiens les 2 ou 3 qui comptent VRAIMENT — ceux dont \
la lecture change quelque chose. Écarte les dépêches de routine, les \
commentaires de séance, les articles qui ne font que constater.

Pour chacun, écris UNE phrase disant pourquoi il mérite d'être lu. Pas de \
résumé de l'article : tu n'as que le titre, tu ne sais pas ce qu'il contient. \
Dis en quoi le SUJET importe.

Réponds uniquement par un tableau JSON, sans texte autour ni balises de code :
[{"index": 0, "pourquoi": "…"}, {"index": 3, "pourquoi": "…"}]

Si aucun article ne sort du lot, renvoie un tableau vide."""


def selectionner(articles: list[dict], cle: str = "",
                 modele: str = "claude-sonnet-5") -> list[dict]:
    """
    Demande au modèle de retenir les articles les plus significatifs.

    Sans clé ou en cas d'échec, on retombe sur le classement par note, qui
    reste exploitable.
    """
    cle = cle or os.environ.get("CLE_ANTHROPIC", "")
    if not cle or len(articles) < 3:
        return articles[:3]

    liste = "\n".join(f"{i}. [{a['source']}] {a['titre']}"
                      for i, a in enumerate(articles))
    try:
        import anthropic
        espace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        client = anthropic.Anthropic(
            api_key=cle,
            default_headers={"anthropic-workspace-id": espace} if espace else None)
        reponse = client.messages.create(
            model=modele, max_tokens=700, system=CONSIGNES,
            messages=[{"role": "user", "content": liste}],
        )
        texte = "".join(b.text for b in reponse.content
                        if getattr(b, "type", "") == "text").strip()
        texte = texte.removeprefix("```json").removeprefix("```").removesuffix("```")
        choix = json.loads(texte.strip())

        retenus = []
        for element in choix:
            i = element.get("index")
            if isinstance(i, int) and 0 <= i < len(articles):
                article = dict(articles[i])
                article["pourquoi"] = element.get("pourquoi", "")
                retenus.append(article)
        return retenus or articles[:3]
    except Exception as erreur:
        print(f"Sélection IA indisponible — {type(erreur).__name__} : "
              f"{str(erreur)[:200]}", file=sys.stderr)
        return articles[:3]


def formater(article: dict) -> str:
    """Message Telegram pour un article, un par message."""
    lignes = [f"📰 {article['titre']}"]
    if article.get("pourquoi"):
        lignes.append("")
        lignes.append(article["pourquoi"])
    lignes.append("")
    acces = "" if article.get("libre") else " · accès payant probable"
    lignes.append(f"{article['source']}{acces}")
    if article.get("lien"):
        lignes.append(article["lien"])
    return "\n".join(lignes)
