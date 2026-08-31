"""
Bilan hebdomadaire, envoye le vendredi soir.

C'est le seul message de la semaine qui prend du recul. Les alertes
quotidiennes signalent des evenements ; celui-ci mesure une trajectoire.

Trois parties : ce qu'a fait le portefeuille, ce qui a change dans sa
structure de risque, et ce qui arrive la semaine prochaine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
import analytics as an
import macro as mc

INDICE_DEFAUT = "IWDA.AS"


def envoyer(message: str) -> bool:
    jeton = os.environ.get("TELEGRAM_JETON", "").strip()
    destinataire = os.environ.get("TELEGRAM_DESTINATAIRE", "").strip()
    if not jeton or not destinataire:
        print("Telegram non configuré. Message :\n" + message, file=sys.stderr)
        return False
    try:
        reponse = requests.post(
            f"https://api.telegram.org/bot{jeton}/sendMessage",
            json={"chat_id": destinataire, "text": message[:4000],
                  "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20)
        if not reponse.ok:
            print(f"Telegram a refusé (code {reponse.status_code}) : "
                  f"{reponse.text[:250]}", file=sys.stderr)
        return reponse.ok
    except Exception as erreur:
        print(f"Échec Telegram : {erreur}", file=sys.stderr)
        return False


def _pourcent(valeur: float, decimales: int = 1) -> str:
    return f"{valeur:+.{decimales}f} %" if np.isfinite(valeur) else "—"


def bilan(portefeuille: pd.DataFrame, cours: pd.DataFrame,
          indice: str) -> str:
    """
    Rédige le bilan.

    La comparaison à l'indice compte davantage que la performance absolue :
    une semaine à +2 % dans un marché à +4 % est une mauvaise semaine.
    """
    valeurs = {}
    for _, ligne in portefeuille.iterrows():
        t = ligne["Ticker"]
        if t in cours.columns and not cours[t].dropna().empty:
            valeurs[t] = float(cours[t].dropna().iloc[-1]) * float(ligne["Quantité"])
    total = sum(valeurs.values())
    if total <= 0:
        return ""

    lignes_ptf = [t for t in valeurs if t in cours.columns]
    poids = pd.Series({t: valeurs[t] / total for t in lignes_ptf})
    rendements = cours[lignes_ptf].pct_change().dropna()
    if rendements.empty:
        return ""

    r_ptf = (rendements * poids).sum(axis=1)
    r_indice = (cours[indice].pct_change().dropna()
                if indice in cours.columns else pd.Series(dtype=float))

    def cumul(serie: pd.Series, jours: int) -> float:
        if len(serie) < jours:
            return np.nan
        return float((1 + serie.tail(jours)).prod() - 1) * 100

    semaine = cumul(r_ptf, 5)
    mois = cumul(r_ptf, 21)
    semaine_indice = cumul(r_indice, 5) if len(r_indice) else np.nan

    lignes = ["📊 BILAN DE LA SEMAINE", ""]

    if np.isfinite(semaine_indice):
        ecart = semaine - semaine_indice
        verdict = ("mieux que l'indice" if ecart > 0.3
                   else "moins bien que l'indice" if ecart < -0.3
                   else "au niveau de l'indice")
        lignes.append(f"Portefeuille {_pourcent(semaine)} sur la semaine, "
                      f"indice {_pourcent(semaine_indice)} — {verdict} "
                      f"({_pourcent(ecart)}).")
    else:
        lignes.append(f"Portefeuille {_pourcent(semaine)} sur la semaine.")

    lignes.append(f"Sur un mois : {_pourcent(mois)}.")
    lignes.append("")

    # --- Ce qui a porté et ce qui a pesé
    contributions = {}
    for t in lignes_ptf:
        serie = cours[t].dropna()
        if len(serie) < 6:
            continue
        variation = float(serie.iloc[-1] / serie.iloc[-6] - 1)
        contributions[t] = variation * float(poids[t]) * 100

    if contributions:
        classement = sorted(contributions.items(), key=lambda x: -x[1])
        meilleur, pire = classement[0], classement[-1]
        lignes.append(f"Meilleure contribution : {meilleur[0]} "
                      f"({_pourcent(meilleur[1], 2)} de performance globale).")
        if pire[1] < 0:
            lignes.append(f"Plus lourd frein : {pire[0]} "
                          f"({_pourcent(pire[1], 2)}).")
        lignes.append("")

    # --- Structure du risque
    if len(lignes_ptf) > 1:
        covariance = an.matrice_covariance(rendements)
        vol = an.volatilite_portefeuille(poids, covariance)
        decompo = an.decomposition_risque(poids, covariance)
        effectives = an.nombre_effectif_lignes(poids)

        if not decompo.empty:
            dominante = decompo["Part du risque (%)"].idxmax()
            part = float(decompo.loc[dominante, "Part du risque (%)"])
            poids_dominante = float(decompo.loc[dominante, "Poids (%)"])
            lignes.append(f"Volatilité annualisée : {vol * 100:.1f} %.")
            lignes.append(f"{dominante} porte {part:.0f} % du risque pour "
                          f"{poids_dominante:.0f} % du capital.")
            lignes.append(f"Diversification réelle : {effectives:.1f} lignes "
                          f"effectives sur {len(lignes_ptf)}.")
            lignes.append("")

    # --- Drawdown courant
    cumule = (1 + r_ptf).cumprod()
    drawdown = float(cumule.iloc[-1] / cumule.cummax().iloc[-1] - 1) * 100
    if drawdown < -3:
        lignes.append(f"Le portefeuille est {abs(drawdown):.1f} % sous son "
                      f"plus haut.")
        lignes.append("")

    # --- La semaine qui vient
    calendrier = mc.calendrier_macro(mois_a_venir=1)
    limite = pd.Timestamp.now().normalize() + pd.Timedelta(days=8)
    prochaine = calendrier[(calendrier["Date"] <= limite)
                           & (calendrier["Type"] != "Marché")]
    jours_fr = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven",
                5: "Sam", 6: "Dim"}
    if not prochaine.empty:
        lignes.append("La semaine prochaine :")
        banques = prochaine[prochaine["Type"] == "Banque centrale"]
        majeures = pd.concat([banques, prochaine.head(4)]).drop_duplicates()
        for _, e in majeures.head(4).iterrows():
            jour = jours_fr[e["Date"].dayofweek]
            lignes.append(f"  • {jour} {e['Date'].strftime('%d/%m')} "
                          f"{e['Événement']}")

    return "\n".join(lignes)


def principal() -> int:
    url = os.environ.get("URL_FEUILLE", "")
    if not url:
        raise SystemExit("Variable URL_FEUILLE absente.")

    sys.path.insert(0, str(Path(__file__).parent))
    import feuille as fe
    portefeuille = fe.lire(url)
    tickers = list(dict.fromkeys(portefeuille["Ticker"]))
    indice = os.environ.get("INDICE_REFERENCE", INDICE_DEFAUT)

    brut = yf.download(tickers + [indice], period="1y", interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    cours = (brut["Close"] if isinstance(brut.columns, pd.MultiIndex)
             else brut[["Close"]].rename(columns={"Close": tickers[0]}))
    cours = cours.dropna(how="all").ffill()

    message = bilan(portefeuille, cours, indice)
    if not message:
        print("Bilan non calculable.")
        return 1

    print(message)
    return 0 if envoyer(message) else 1


if __name__ == "__main__":
    raise SystemExit(principal())
