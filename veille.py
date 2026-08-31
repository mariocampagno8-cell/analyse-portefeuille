"""
Veille automatique et notifications.

Script autonome, sans Streamlit : concu pour tourner sans personne devant
l'ecran, typiquement une fois par jour via GitHub Actions.

Philosophie des alertes. Une alerte n'a d'interet que si elle appelle une
decision que tu ne pouvais pas anticiper. Les signaux techniques n'en font
generalement pas partie : ils se declenchent souvent, se contredisent, et leur
effet documente est d'augmenter la frequence de transaction sans ameliorer le
resultat. Les regles activees par defaut sont donc des regles d'ECHEANCE et de
DERIVE, pas de signal. Les regles de signal existent mais sont desactivees.

Anti-repetition : chaque alerte emise est memorisee dans etat_veille.json.
Une meme alerte ne sera pas renvoyee avant le delai configure — sans quoi une
position sous son seuil declencherait une notification chaque jour pendant
des semaines.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import redaction as rd
import signaux as sg

FICHIER_ETAT = Path(__file__).parent / "etat_veille.json"
DELAI_REPETITION = 7          # jours avant de reemettre une meme alerte


# ==========================================================================
# Configuration
# ==========================================================================

REGLES_DEFAUT = {
    # --- Échéances : tu ne peux pas les deviner, elles appellent une action
    "publication_proche": {
        "actif": True, "jours": 3,
        "libelle": "Publication de résultats imminente",
    },
    # --- Dérive : ton portefeuille ne ressemble plus à ce que tu voulais
    "derive_poids": {
        "actif": True, "seuil_points": 10,
        "libelle": "Une position a fortement dérivé",
    },
    "concentration": {
        "actif": True, "seuil_pct": 40,
        "libelle": "Concentration excessive sur une ligne",
    },
    # --- Risque : le portefeuille sort de ta zone de tolérance
    "drawdown_portefeuille": {
        "actif": True, "seuil_pct": -15,
        "libelle": "Repli du portefeuille au-delà du seuil",
    },
    "volatilite_anormale": {
        "actif": True, "multiple": 2.0,
        "libelle": "Volatilité très supérieure à sa normale",
    },
    # --- Qualité : un chiffre est devenu douteux
    "donnees_perimees": {
        "actif": True, "jours_ouvres": 5,
        "libelle": "Cours non mis à jour",
    },
    # --- Signaux documentés (voir signaux.py pour les références)
    "momentum": {
        "actif": True,
        "libelle": "Position au classement momentum",
    },
    "derive_resultats": {
        "actif": True,
        "libelle": "Dérive post-annonce de résultats",
    },
    "revisions": {
        "actif": True,
        "libelle": "Révision des estimations",
    },
    "plus_haut": {
        "actif": True,
        "libelle": "Position face au plus haut annuel",
    },
    "regime": {
        "actif": True, "indice": "IWDA.AS",
        "libelle": "Changement de régime de marché",
    },
    "dimensionnement": {
        "actif": True,
        "libelle": "Concentration du risque",
    },

    # --- Signaux techniques simples : désactivés, voir l'avertissement en tête
    "rsi_bas": {
        "actif": False, "seuil": 30,
        "libelle": "RSI en zone basse",
    },
    "rsi_haut": {
        "actif": False, "seuil": 70,
        "libelle": "RSI en zone haute",
    },
    "franchissement_mm200": {
        "actif": False,
        "libelle": "Franchissement de la moyenne 200 séances",
    },
    "ecart_plus_haut": {
        "actif": False, "seuil_pct": -25,
        "libelle": "Écart important au plus haut annuel",
    },
}


def charger_config() -> dict:
    """
    Regles depuis la variable d'environnement REGLES_VEILLE, sinon defauts.

    Permet d'ajuster les seuils sans toucher au code ni redeployer.
    """
    brut = os.environ.get("REGLES_VEILLE", "")
    regles = json.loads(json.dumps(REGLES_DEFAUT))
    if brut:
        try:
            for cle, valeurs in json.loads(brut).items():
                if cle in regles:
                    regles[cle].update(valeurs)
        except json.JSONDecodeError:
            print("REGLES_VEILLE illisible, valeurs par défaut retenues.",
                  file=sys.stderr)
    return regles


# ==========================================================================
# Mémoire des alertes déjà émises
# ==========================================================================

def lire_etat() -> dict:
    if FICHIER_ETAT.exists():
        try:
            return json.loads(FICHIER_ETAT.read_text())
        except Exception:
            pass
    return {}


def ecrire_etat(etat: dict) -> None:
    FICHIER_ETAT.write_text(json.dumps(etat, indent=1, ensure_ascii=False))


def deja_signalee(etat: dict, identifiant: str) -> bool:
    """Vrai si cette alerte a deja ete emise recemment."""
    horodatage = etat.get(identifiant)
    if not horodatage:
        return False
    try:
        emise = datetime.fromisoformat(horodatage)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - emise < timedelta(days=DELAI_REPETITION)


# ==========================================================================
# Évaluation des règles
# ==========================================================================

def _rsi(prix: pd.Series, n: int = 14) -> float:
    delta = prix.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    pertes = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    valeur = 100 - 100 / (1 + gains / pertes.replace(0, np.nan))
    return float(valeur.iloc[-1]) if len(valeur.dropna()) else np.nan


def evaluer(portefeuille: pd.DataFrame, cours: pd.DataFrame,
            regles: dict, publications: dict | None = None) -> list[dict]:
    """
    Confronte l'etat du portefeuille aux regles actives.

    Renvoie une liste d'alertes, chacune avec un identifiant stable qui sert
    a l'anti-repetition.
    """
    alertes = []
    if portefeuille.empty or cours.empty:
        return alertes

    valeurs, poids_cible = {}, {}
    for _, ligne in portefeuille.iterrows():
        t = ligne["Ticker"]
        if t in cours.columns and not cours[t].dropna().empty:
            valeurs[t] = float(cours[t].dropna().iloc[-1]) * float(ligne["Quantité"])
            poids_cible[t] = float(ligne.get("Poids cible (%)", np.nan))
    total = sum(valeurs.values())
    if total <= 0:
        return alertes

    # --- Portefeuille dans son ensemble
    lignes_valides = [t for t in valeurs if t in cours.columns]
    if lignes_valides:
        poids = pd.Series({t: valeurs[t] / total for t in lignes_valides})
        rendements = cours[lignes_valides].pct_change().dropna()
        if not rendements.empty:
            r_ptf = (rendements * poids).sum(axis=1)
            cumul = (1 + r_ptf).cumprod()
            drawdown = float((cumul.iloc[-1] / cumul.cummax().iloc[-1] - 1) * 100)

            regle = regles["drawdown_portefeuille"]
            if regle["actif"] and drawdown <= regle["seuil_pct"]:
                alertes.append({
                    "id": f"drawdown|{int(drawdown // 5) * 5}",
                    "titre": regle["libelle"],
                    "corps": f"Le portefeuille est {drawdown:.1f} % sous son "
                             f"plus haut. Seuil fixé à {regle['seuil_pct']} %.",
                    "priorite": "haute",
                })

            regle = regles["volatilite_anormale"]
            if regle["actif"] and len(r_ptf) > 260:
                courte = float(r_ptf.tail(21).std(ddof=1) * np.sqrt(252) * 100)
                longue = float(r_ptf.tail(252).std(ddof=1) * np.sqrt(252) * 100)
                if longue > 0 and courte > longue * regle["multiple"]:
                    alertes.append({
                        "id": f"volatilite|{datetime.now().strftime('%Y-%W')}",
                        "titre": regle["libelle"],
                        "corps": f"Volatilité sur un mois à {courte:.0f} %, "
                                 f"contre {longue:.0f} % sur un an.",
                        "priorite": "haute",
                    })

    # --- Ligne par ligne
    for t, valeur in valeurs.items():
        part = valeur / total * 100
        serie = cours[t].dropna()

        regle = regles["concentration"]
        if regle["actif"] and part > regle["seuil_pct"]:
            alertes.append({
                "id": f"concentration|{t}|{int(part // 5) * 5}",
                "titre": f"{t} — {regle['libelle']}",
                "corps": f"{t} représente {part:.1f} % du portefeuille "
                         f"(seuil {regle['seuil_pct']} %).",
                "priorite": "normale",
            })

        regle = regles["derive_poids"]
        cible = poids_cible.get(t, np.nan)
        if regle["actif"] and np.isfinite(cible) and cible > 0:
            ecart = part - cible
            if abs(ecart) >= regle["seuil_points"]:
                alertes.append({
                    "id": f"derive|{t}|{int(ecart // 5) * 5}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": f"Poids actuel {part:.1f} % contre {cible:.1f} % "
                             f"visé, soit {ecart:+.1f} points.",
                    "priorite": "normale",
                })

        regle = regles["donnees_perimees"]
        if regle["actif"] and len(serie):
            retard = int(np.busday_count(serie.index[-1].date(),
                                         datetime.now().date()))
            if retard > regle["jours_ouvres"]:
                alertes.append({
                    "id": f"perime|{t}|{serie.index[-1].date()}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": f"Dernier cours daté du {serie.index[-1].date()}, "
                             f"soit {retard} jours ouvrés de retard.",
                    "priorite": "normale",
                })

        if len(serie) < 220:
            continue

        regle = regles["rsi_bas"]
        if regle["actif"]:
            valeur_rsi = _rsi(serie)
            if np.isfinite(valeur_rsi) and valeur_rsi < regle["seuil"]:
                alertes.append({
                    "id": f"rsi_bas|{t}|{datetime.now().strftime('%Y-%W')}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": f"RSI à {valeur_rsi:.0f}.",
                    "priorite": "basse",
                })

        regle = regles["rsi_haut"]
        if regle["actif"]:
            valeur_rsi = _rsi(serie)
            if np.isfinite(valeur_rsi) and valeur_rsi > regle["seuil"]:
                alertes.append({
                    "id": f"rsi_haut|{t}|{datetime.now().strftime('%Y-%W')}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": f"RSI à {valeur_rsi:.0f}.",
                    "priorite": "basse",
                })

        regle = regles["franchissement_mm200"]
        if regle["actif"]:
            mm = serie.rolling(200).mean()
            if len(mm.dropna()) > 2:
                avant = serie.iloc[-2] > mm.iloc[-2]
                apres = serie.iloc[-1] > mm.iloc[-1]
                if avant != apres:
                    sens = "au-dessus de" if apres else "sous"
                    alertes.append({
                        "id": f"mm200|{t}|{serie.index[-1].date()}",
                        "titre": f"{t} — {regle['libelle']}",
                        "corps": f"Le cours est repassé {sens} sa moyenne "
                                 f"200 séances.",
                        "priorite": "basse",
                    })

        regle = regles["ecart_plus_haut"]
        if regle["actif"]:
            ecart = float(serie.iloc[-1] / serie.tail(252).max() - 1) * 100
            if ecart <= regle["seuil_pct"]:
                alertes.append({
                    "id": f"ecart_haut|{t}|{int(ecart // 5) * 5}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": f"{ecart:.0f} % sous son plus haut sur un an.",
                    "priorite": "basse",
                })

    # --- Publications à venir
    regle = regles["publication_proche"]
    if regle["actif"] and publications:
        for t, date in publications.items():
            if date is None:
                continue
            jours = (date - datetime.now().date()).days
            if 0 <= jours <= regle["jours"]:
                alertes.append({
                    "id": f"publication|{t}|{date}",
                    "titre": f"{t} — {regle['libelle']}",
                    "corps": (f"Publication attendue le "
                              f"{date.strftime('%d/%m')}"
                              + (" — demain." if jours == 1 else
                                 " — aujourd'hui." if jours == 0 else
                                 f", dans {jours} jours.")),
                    "priorite": "haute",
                })

    ordre = {"haute": 0, "normale": 1, "basse": 2}
    return sorted(alertes, key=lambda a: ordre.get(a["priorite"], 3))


# ==========================================================================
# Envoi des notifications
# ==========================================================================

def envoyer_ntfy(titre: str, corps: str, sujet: str,
                 priorite: str = "normale") -> bool:
    """
    Notification via ntfy.sh — gratuit, sans compte ni inscription.

    Le « sujet » est un identifiant que tu choisis et que tu abonnes dans
    l'application ntfy sur ton telephone. Toute personne qui le connait peut
    y publier : prends quelque chose de long et d'imprevisible.
    """
    niveaux = {"haute": "high", "normale": "default", "basse": "low"}
    try:
        reponse = requests.post(
            f"https://ntfy.sh/{sujet}",
            data=corps.encode("utf-8"),
            headers={
                "Title": titre.encode("utf-8"),
                "Priority": niveaux.get(priorite, "default"),
                "Tags": "chart_with_upwards_trend",
            },
            timeout=15,
        )
        return reponse.ok
    except Exception as erreur:
        print(f"Échec ntfy : {erreur}", file=sys.stderr)
        return False


def envoyer_telegram(message: str, jeton: str, destinataire: str) -> bool:
    """
    Notification via un bot Telegram.

    Envoi en texte brut : le Markdown de Telegram echoue des qu'un asterisque
    ou un tiret bas n'est pas apparie, ce qui arrive constamment dans un
    commentaire redige librement. La mise en forme ne vaut pas le risque de
    perdre le message.
    """
    if not jeton or not destinataire:
        print("Telegram : jeton ou destinataire manquant.", file=sys.stderr)
        return False
    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": str(destinataire).strip(), "text": message,
                  "disable_web_page_preview": True},
            timeout=20,
        )
        if not reponse.ok:
            print(f"Telegram a refusé la requête (code {reponse.status_code}) : "
                  f"{reponse.text[:300]}", file=sys.stderr)
            return False
        return True
    except Exception as erreur:
        print(f"Échec Telegram : {type(erreur).__name__} — {erreur}",
              file=sys.stderr)
        return False


def notifier(alertes: list[dict]) -> int:
    """Envoie les alertes par les canaux configures. Renvoie le nombre d'envois."""
    sujet_ntfy = os.environ.get("NTFY_SUJET", "").strip()
    jeton_tg = os.environ.get("TELEGRAM_JETON", "").strip()
    dest_tg = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    envois = 0

    print(f"Canaux — Telegram : {'configuré' if jeton_tg and dest_tg else 'absent'}"
          f" | ntfy : {'configuré' if sujet_ntfy else 'absent'}")

    if jeton_tg and dest_tg:
        # Telegram limite un message à 4096 caractères : on découpe si besoin
        blocs, courant = [], "📊 VEILLE PORTEFEUILLE\n"
        for a in alertes:
            morceau = f"\n▸ {a['titre']}\n{a['corps']}\n"
            if len(courant) + len(morceau) > 3800:
                blocs.append(courant)
                courant = morceau
            else:
                courant += morceau
        blocs.append(courant)
        for bloc in blocs:
            if envoyer_telegram(bloc, jeton_tg, dest_tg):
                envois += 1

    if sujet_ntfy:
        for alerte in alertes:
            if envoyer_ntfy(alerte["titre"], alerte["corps"],
                            sujet_ntfy, alerte["priorite"]):
                envois += 1

    if not sujet_ntfy and not (jeton_tg and dest_tg):
        print("Aucun canal configuré. Alertes non envoyées :", file=sys.stderr)
        for a in alertes:
            print(f"  [{a['priorite']}] {a['titre']} — {a['corps']}")

    return envois


# ==========================================================================
# Exécution
# ==========================================================================

def charger_portefeuille() -> pd.DataFrame:
    """Portefeuille depuis la feuille Google publiee."""
    url = os.environ.get("URL_FEUILLE", "")
    if not url:
        raise SystemExit("Variable URL_FEUILLE absente.")
    sys.path.insert(0, str(Path(__file__).parent))
    import feuille as fe
    return fe.lire(url)


def principal() -> int:
    regles = charger_config()
    portefeuille = charger_portefeuille()
    tickers = list(dict.fromkeys(portefeuille["Ticker"]))
    print(f"{len(tickers)} valeur(s) suivie(s) : {', '.join(tickers)}")

    donnees = yf.download(tickers, period="2y", interval="1d",
                          auto_adjust=True, progress=False, group_by="column")
    cours = (donnees["Close"] if isinstance(donnees.columns, pd.MultiIndex)
             else donnees[["Close"]].rename(columns={"Close": tickers[0]}))
    cours = cours.dropna(how="all").ffill()

    publications = {}
    if regles["publication_proche"]["actif"]:
        for t in tickers:
            try:
                dates = yf.Ticker(t).earnings_dates
                futures = dates[dates["Reported EPS"].isna()]
                if not futures.empty:
                    prochaine = pd.Timestamp(futures.sort_index().index[0])
                    publications[t] = prochaine.tz_localize(None).date() \
                        if prochaine.tzinfo else prochaine.date()
            except Exception:
                continue

    alertes = evaluer(portefeuille, cours, regles, publications)
    print(f"{len(alertes)} alerte(s) de gestion.")

    # --- Signaux documentés, avec contexte propre à chaque société
    contextes, alertes_signaux = {}, []
    try:
        valeurs = {}
        for _, ligne in portefeuille.iterrows():
            t = ligne["Ticker"]
            if t in cours.columns and not cours[t].dropna().empty:
                valeurs[t] = float(cours[t].dropna().iloc[-1]) * float(ligne["Quantité"])
        total_ptf = sum(valeurs.values())

        # Contribution au risque de chaque ligne
        parts_risque = {}
        if total_ptf > 0 and len(valeurs) > 1:
            w = pd.Series({t: v / total_ptf for t, v in valeurs.items()})
            covariance = cours[list(w.index)].pct_change().dropna().cov() * 252
            vol = float(np.sqrt(w @ covariance.to_numpy() @ w))
            if vol > 0:
                contributions = w.to_numpy() * (covariance.to_numpy() @ w.to_numpy()) / vol
                parts_risque = {t: float(c / vol * 100)
                                for t, c in zip(w.index, contributions)}

        for t, valeur in valeurs.items():
            info, surprises, revisions = {}, None, None
            try:
                ticker_yf = yf.Ticker(t)
                info = dict(ticker_yf.info)
                dates = ticker_yf.earnings_dates
                if dates is not None and not dates.empty and "Surprise(%)" in dates.columns:
                    surprises = dates.rename(
                        columns={"Surprise(%)": "Surprise (%)"})[["Surprise (%)"]].dropna()
                tendance = ticker_yf.eps_trend
                if tendance is not None and not tendance.empty \
                        and {"current", "90daysAgo"} <= set(tendance.columns):
                    actuelle = float(tendance["current"].iloc[0])
                    ancienne = float(tendance["90daysAgo"].iloc[0])
                    if ancienne:
                        revisions = pd.DataFrame({
                            "Révision sur 90 jours (%)":
                                [(actuelle - ancienne) / abs(ancienne) * 100]})
            except Exception:
                pass

            contextes[t] = sg.contexte_societe(
                t, cours[t], poids_pct=valeur / total_ptf * 100,
                part_risque_pct=parts_risque.get(t),
                info=info, surprises=surprises, revisions=revisions)

        indice = None
        if regles["regime"]["actif"]:
            try:
                brut = yf.download(regles["regime"]["indice"], period="2y",
                                   progress=False, auto_adjust=True)
                if isinstance(brut.columns, pd.MultiIndex):
                    brut.columns = brut.columns.get_level_values(0)
                indice = brut["Close"].dropna()
            except Exception:
                pass

        actifs = {c: regles.get(c, {}).get("actif", False)
                  for c in ("momentum", "derive_resultats", "revisions",
                            "plus_haut", "dimensionnement", "regime")}
        alertes_signaux = sg.evaluer_tout(cours, contextes, indice, actifs)
        print(f"{len(alertes_signaux)} signal(aux) déclenché(s).")

        # Rédaction par Claude, avec repli sur les gabarits en cas d'échec
        alertes_signaux = rd.enrichir(alertes_signaux, contextes,
                                      os.environ.get("CLE_ANTHROPIC", ""))
        rediges = sum(1 for a in alertes_signaux if a.get("rédigé_par_ia"))
        print(f"{rediges} commentaire(s) rédigé(s) par IA.")

        for a in alertes_signaux:
            alertes.append({
                "id": f"signal|{a['type']}|{a['ticker']}|"
                      f"{datetime.now().strftime('%Y-%W')}",
                "titre": a["titre"],
                "corps": a["commentaire"],
                "priorite": "normale",
            })
    except Exception as erreur:
        print(f"Signaux indisponibles : {type(erreur).__name__} — {erreur}",
              file=sys.stderr)

    print(f"{len(alertes)} alerte(s) au total.")

    etat = lire_etat()
    nouvelles = [a for a in alertes if not deja_signalee(etat, a["id"])]
    print(f"{len(nouvelles)} nouvelle(s) après filtrage anti-répétition.")

    if not nouvelles:
        return 0

    envois = notifier(nouvelles)
    print(f"{envois} notification(s) envoyée(s).")

    if envois == 0:
        print("Aucun envoi réussi : les alertes ne sont pas mémorisées, "
              "elles seront réessayées au prochain passage.", file=sys.stderr)
        return 1

    maintenant = datetime.now(timezone.utc).isoformat()
    for alerte in nouvelles:
        etat[alerte["id"]] = maintenant

    limite = datetime.now(timezone.utc) - timedelta(days=60)
    etat = {k: v for k, v in etat.items()
            if datetime.fromisoformat(v) > limite}
    ecrire_etat(etat)
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
