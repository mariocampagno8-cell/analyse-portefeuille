"""
Bibliotheque d'indicateurs, organisee par famille.

Remarque de methode : la plupart des indicateurs d'une meme famille sont des
transformations de la meme information. RSI, stochastique, Williams %R et CCI
mesurent tous la position du prix dans sa plage recente. Les combiner ne
multiplie pas l'information, cela repete le meme signal. Utiliser au plus un ou
deux indicateurs par famille, et verifier leur correlation avant de conclure.

Toutes les fonctions prennent un DataFrame OHLCV avec les colonnes
Open, High, Low, Close, Volume, et renvoient une Series ou un DataFrame
aligne sur le meme index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JOURS_BOURSE = 252


# ==========================================================================
# Tendance
# ==========================================================================

def sma(prix: pd.Series, n: int = 20) -> pd.Series:
    """Moyenne mobile simple."""
    return prix.rolling(n).mean()


def ema(prix: pd.Series, n: int = 20) -> pd.Series:
    """Moyenne mobile exponentielle : reagit plus vite, retarde moins."""
    return prix.ewm(span=n, adjust=False).mean()


def wma(prix: pd.Series, n: int = 20) -> pd.Series:
    """Moyenne mobile ponderee lineairement."""
    poids = np.arange(1, n + 1)
    return prix.rolling(n).apply(
        lambda x: np.dot(x, poids) / poids.sum(), raw=True)


def hull(prix: pd.Series, n: int = 20) -> pd.Series:
    """
    Moyenne de Hull : reduit fortement le retard sans amplifier le bruit.

    Construite comme 2*WMA(n/2) - WMA(n), relissee sur racine de n.
    """
    demi = wma(prix, max(int(n / 2), 1))
    pleine = wma(prix, n)
    return wma(2 * demi - pleine, max(int(np.sqrt(n)), 1))


def macd(prix: pd.Series, rapide: int = 12, lent: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    Convergence-divergence de moyennes mobiles.

    La ligne MACD est un filtre passe-bande : elle isole les mouvements de
    duree intermediaire. L'histogramme mesure l'acceleration.
    """
    ligne = ema(prix, rapide) - ema(prix, lent)
    ligne_signal = ema(ligne, signal)
    return pd.DataFrame({
        "MACD": ligne,
        "Signal": ligne_signal,
        "Histogramme": ligne - ligne_signal,
    })


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """
    Directional Movement de Wilder.

    L'ADX mesure la FORCE de la tendance, pas son sens : au-dessus de 25 le
    marche tend, en dessous de 20 il oscille. Le sens se lit sur +DI et -DI.
    Utile pour savoir si les indicateurs de tendance ou d'oscillation
    s'appliquent au moment present.
    """
    haut, bas, cloture = df["High"], df["Low"], df["Close"]
    tr = _true_range(df)

    dm_plus = haut.diff()
    dm_moins = -bas.diff()
    dm_plus = dm_plus.where((dm_plus > dm_moins) & (dm_plus > 0), 0.0)
    dm_moins = dm_moins.where((dm_moins > dm_plus.abs()) & (dm_moins > 0), 0.0)

    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / n, adjust=False).mean() / atr_
    di_moins = 100 * dm_moins.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (di_plus - di_moins).abs() / (di_plus + di_moins).replace(0, np.nan)

    return pd.DataFrame({
        "ADX": dx.ewm(alpha=1 / n, adjust=False).mean(),
        "+DI": di_plus, "-DI": di_moins,
    })


def aroon(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Temps ecoule depuis le plus haut et le plus bas de la fenetre."""
    haut = df["High"].rolling(n + 1).apply(lambda x: x.argmax(), raw=True)
    bas = df["Low"].rolling(n + 1).apply(lambda x: x.argmin(), raw=True)
    return pd.DataFrame({
        "Aroon haut": haut / n * 100,
        "Aroon bas": bas / n * 100,
        "Aroon oscillateur": (haut - bas) / n * 100,
    })


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Nuage d'Ichimoku : supports, resistances et tendance en une lecture."""
    def milieu(n):
        return (df["High"].rolling(n).max() + df["Low"].rolling(n).min()) / 2

    tenkan, kijun = milieu(9), milieu(26)
    return pd.DataFrame({
        "Tenkan": tenkan,
        "Kijun": kijun,
        "Senkou A": ((tenkan + kijun) / 2).shift(26),
        "Senkou B": milieu(52).shift(26),
        "Chikou": df["Close"].shift(-26),
    })


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Bande de suivi de tendance construite sur l'ATR."""
    atr_ = atr(df, n)
    milieu = (df["High"] + df["Low"]) / 2
    haute, basse = milieu + mult * atr_, milieu - mult * atr_

    sens = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > haute.iloc[i - 1]:
            sens.iloc[i] = 1
        elif df["Close"].iloc[i] < basse.iloc[i - 1]:
            sens.iloc[i] = -1
        else:
            sens.iloc[i] = sens.iloc[i - 1]
    return pd.DataFrame({"Supertrend": np.where(sens > 0, basse, haute),
                         "Sens": sens})


# ==========================================================================
# Momentum et oscillateurs
# ==========================================================================

def rsi(prix: pd.Series, n: int = 14) -> pd.Series:
    """Force relative de Wilder, bornee entre 0 et 100."""
    delta = prix.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    pertes = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gains / pertes.replace(0, np.nan))


def stochastique(df: pd.DataFrame, n: int = 14, lissage: int = 3) -> pd.DataFrame:
    """Position de la cloture dans la plage haut-bas de la fenetre."""
    bas = df["Low"].rolling(n).min()
    haut = df["High"].rolling(n).max()
    k = 100 * (df["Close"] - bas) / (haut - bas).replace(0, np.nan)
    return pd.DataFrame({"%K": k, "%D": k.rolling(lissage).mean()})


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Stochastique inverse, borne entre -100 et 0."""
    haut = df["High"].rolling(n).max()
    bas = df["Low"].rolling(n).min()
    return -100 * (haut - df["Close"]) / (haut - bas).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Ecart du prix typique a sa moyenne, normalise par l'ecart absolu moyen."""
    typique = (df["High"] + df["Low"] + df["Close"]) / 3
    moyenne = typique.rolling(n).mean()
    ecart = typique.rolling(n).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typique - moyenne) / (0.015 * ecart.replace(0, np.nan))


def roc(prix: pd.Series, n: int = 12) -> pd.Series:
    """Variation en pourcentage sur n periodes."""
    return prix.pct_change(n) * 100


def momentum_absolu(prix: pd.Series, n: int = 252) -> pd.Series:
    """
    Momentum a douze mois, l'anomalie la plus robuste de la litterature.

    Documentee sur plus d'un siecle et sur la plupart des marches. On saute
    generalement le dernier mois, qui presente un effet de retournement.
    """
    return prix.shift(21) / prix.shift(n) - 1


def tsi(prix: pd.Series, lent: int = 25, rapide: int = 13) -> pd.Series:
    """True Strength Index : momentum doublement lisse, peu bruite."""
    delta = prix.diff()
    lisse = delta.ewm(span=lent, adjust=False).mean().ewm(
        span=rapide, adjust=False).mean()
    lisse_abs = delta.abs().ewm(span=lent, adjust=False).mean().ewm(
        span=rapide, adjust=False).mean()
    return 100 * lisse / lisse_abs.replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """RSI pondere par les volumes."""
    typique = (df["High"] + df["Low"] + df["Close"]) / 3
    flux = typique * df["Volume"]
    hausse = flux.where(typique.diff() > 0, 0).rolling(n).sum()
    baisse = flux.where(typique.diff() < 0, 0).rolling(n).sum()
    return 100 - 100 / (1 + hausse / baisse.replace(0, np.nan))


# ==========================================================================
# Volatilite
# ==========================================================================

def _true_range(df: pd.DataFrame) -> pd.Series:
    cloture = df["Close"].shift()
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - cloture).abs(),
        (df["Low"] - cloture).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Amplitude vraie moyenne : sert au dimensionnement des positions."""
    return _true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def bollinger(prix: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    Bandes de Bollinger.

    Le %B situe le prix dans les bandes, la largeur mesure le regime de
    volatilite : un resserrement precede souvent une expansion.
    """
    moyenne = prix.rolling(n).mean()
    ecart = prix.rolling(n).std(ddof=1)
    haute, basse = moyenne + k * ecart, moyenne - k * ecart
    return pd.DataFrame({
        "Bande haute": haute, "Moyenne": moyenne, "Bande basse": basse,
        "%B": (prix - basse) / (haute - basse).replace(0, np.nan),
        "Largeur": (haute - basse) / moyenne.replace(0, np.nan) * 100,
    })


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Canal construit sur l'ATR plutot que sur l'ecart-type."""
    milieu = ema(df["Close"], n)
    a = atr(df, n)
    return pd.DataFrame({"Keltner haut": milieu + mult * a,
                         "Keltner milieu": milieu,
                         "Keltner bas": milieu - mult * a})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Canal des extremes : la base des strategies de cassure."""
    return pd.DataFrame({"Donchian haut": df["High"].rolling(n).max(),
                         "Donchian bas": df["Low"].rolling(n).min()})


def volatilite_realisee(prix: pd.Series, n: int = 21,
                        freq: float = JOURS_BOURSE) -> pd.Series:
    """Ecart-type glissant des rendements, annualise."""
    return prix.pct_change().rolling(n).std(ddof=1) * np.sqrt(freq) * 100


def volatilite_parkinson(df: pd.DataFrame, n: int = 21,
                         freq: float = JOURS_BOURSE) -> pd.Series:
    """
    Estimateur haut-bas de Parkinson.

    Environ cinq fois plus efficace que l'ecart-type des clotures, parce
    qu'il exploite l'amplitude intra-seance au lieu de deux points par jour.
    """
    ratio = np.log(df["High"] / df["Low"]) ** 2
    return np.sqrt(ratio.rolling(n).mean() / (4 * np.log(2)) * freq) * 100


def volatilite_garman_klass(df: pd.DataFrame, n: int = 21,
                            freq: float = JOURS_BOURSE) -> pd.Series:
    """Estimateur OHLC, plus efficace encore que Parkinson."""
    hl = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
    return np.sqrt((hl - co).rolling(n).mean().clip(lower=0) * freq) * 100


def volatilite_rogers_satchell(df: pd.DataFrame, n: int = 21,
                               freq: float = JOURS_BOURSE) -> pd.Series:
    """Estimateur OHLC valable meme en presence de derive."""
    terme = (np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"])
             + np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"]))
    return np.sqrt(terme.rolling(n).mean().clip(lower=0) * freq) * 100


def ulcer(prix: pd.Series, n: int = 14) -> pd.Series:
    """
    Indice d'Ulcer : profondeur ET duree des pertes.

    Contrairement a la volatilite, il ne penalise pas les hausses — ce qui
    correspond mieux au risque tel qu'un investisseur le ressent.
    """
    sommet = prix.rolling(n).max()
    baisse = ((prix - sommet) / sommet * 100) ** 2
    return np.sqrt(baisse.rolling(n).mean())


# ==========================================================================
# Volume
# ==========================================================================

def obv(df: pd.DataFrame) -> pd.Series:
    """Volume cumule signe par le sens de la seance."""
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Volume pondere par la position de la cloture dans la plage du jour."""
    plage = (df["High"] - df["Low"]).replace(0, np.nan)
    multiplicateur = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / plage
    return (multiplicateur * df["Volume"]).fillna(0).cumsum()


def chaikin(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Flux monetaire de Chaikin, borne entre -1 et 1."""
    plage = (df["High"] - df["Low"]).replace(0, np.nan)
    multiplicateur = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / plage
    flux = (multiplicateur * df["Volume"]).fillna(0)
    return flux.rolling(n).sum() / df["Volume"].rolling(n).sum().replace(0, np.nan)


def vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Prix moyen pondere par les volumes."""
    typique = (df["High"] + df["Low"] + df["Close"]) / 3
    return ((typique * df["Volume"]).rolling(n).sum()
            / df["Volume"].rolling(n).sum().replace(0, np.nan))


def volume_relatif(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Volume du jour rapporte a sa moyenne : detecte l'activite anormale."""
    return df["Volume"] / df["Volume"].rolling(n).mean().replace(0, np.nan)


def liquidite_amihud(df: pd.DataFrame, n: int = 21) -> pd.Series:
    """
    Illiquidite d'Amihud : impact prix par euro echange.

    Plus la valeur est elevee, plus le titre bouge pour un faible volume.
    Determinant pour savoir si une position est sortable sans degat.
    """
    echange = (df["Close"] * df["Volume"]).replace(0, np.nan)
    return (df["Close"].pct_change().abs() / echange).rolling(n).mean() * 1e9


# ==========================================================================
# Statistique
# ==========================================================================

def hurst(prix: pd.Series, max_lag: int = 60) -> float:
    """
    Exposant de Hurst.

    Au-dessus de 0,5 la serie persiste (les tendances se prolongent), en
    dessous elle revient a la moyenne, a 0,5 c'est une marche aleatoire.
    Indique si les strategies de suivi ou de retour a la moyenne ont un sens
    sur ce titre. La plupart des actions tournent autour de 0,5.
    """
    x = np.log(prix.dropna().to_numpy())
    if len(x) < max_lag * 3:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = [np.std(x[l:] - x[:-l], ddof=1) for l in lags]
    tau = np.array(tau)
    valides = tau > 0
    if valides.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log(lags[valides]), np.log(tau[valides]), 1)[0])


def autocorrelation(prix: pd.Series, decalages: int = 10) -> pd.Series:
    """
    Autocorrelation des rendements.

    Une autocorrelation positive au decalage 1 signale de la persistance,
    negative du retour a la moyenne. Sur actions liquides elle est presque
    toujours proche de zero — ce qui est la definition d'un marche efficient.
    """
    r = prix.pct_change().dropna()
    return pd.Series({f"Décalage {k}": r.autocorr(k)
                      for k in range(1, decalages + 1)})


def ratio_variance(prix: pd.Series, q: int = 5) -> float:
    """
    Test du ratio de variance de Lo-MacKinlay.

    Vaut 1 pour une marche aleatoire. Au-dessus, la serie tend ; en dessous,
    elle revient a la moyenne.
    """
    r = np.log(prix.dropna()).diff().dropna().to_numpy()
    if len(r) < q * 10:
        return np.nan
    var_1 = r.var(ddof=1)
    agrege = np.add.reduceat(r, np.arange(0, len(r) - len(r) % q, q))
    var_q = agrege.var(ddof=1) / q
    return float(var_q / var_1) if var_1 > 0 else np.nan


def force_relative(prix: pd.Series, prix_marche: pd.Series,
                   n: int = 63) -> pd.Series:
    """Performance du titre moins celle de l'indice sur la fenetre."""
    return (prix.pct_change(n) - prix_marche.pct_change(n)) * 100


def z_score(prix: pd.Series, n: int = 60) -> pd.Series:
    """Ecart a la moyenne en nombre d'ecarts-types : mesure d'extension."""
    moyenne = prix.rolling(n).mean()
    ecart = prix.rolling(n).std(ddof=1)
    return (prix - moyenne) / ecart.replace(0, np.nan)


def distance_plus_haut(prix: pd.Series, n: int = 252) -> pd.Series:
    """
    Ecart au plus haut de la fenetre, en pourcentage.

    Le fait d'etre proche de son plus haut annuel est l'un des rares
    predicteurs simples documentes dans la litterature academique.
    """
    return (prix / prix.rolling(n).max() - 1) * 100


# ==========================================================================
# Synthese
# ==========================================================================

FAMILLES = {
    "Tendance": ["MM20", "MM50", "MM200", "MACD", "ADX", "Aroon oscillateur",
                 "Supertrend"],
    "Momentum": ["RSI", "Stochastique %K", "Williams %R", "CCI", "ROC 12",
                 "Momentum 12 mois", "TSI", "MFI"],
    "Volatilité": ["Volatilité réalisée", "Parkinson", "Garman-Klass",
                   "Rogers-Satchell", "ATR %", "Largeur Bollinger", "Ulcer"],
    "Volume": ["Volume relatif", "Chaikin", "OBV pente", "Amihud"],
    "Statistique": ["Hurst", "Ratio de variance", "Z-score", "Distance au plus haut"],
}


def calculer_tout(df: pd.DataFrame,
                  prix_marche: pd.Series | None = None) -> pd.DataFrame:
    """Calcule l'ensemble des indicateurs sous forme de colonnes."""
    out = pd.DataFrame(index=df.index)
    cloture = df["Close"]

    for n in (20, 50, 200):
        out[f"MM{n}"] = (cloture / sma(cloture, n) - 1) * 100

    m = macd(cloture)
    out["MACD"] = m["Histogramme"]
    a = adx(df)
    out["ADX"] = a["ADX"]
    out["Aroon oscillateur"] = aroon(df)["Aroon oscillateur"]
    out["Supertrend"] = supertrend(df)["Sens"]

    out["RSI"] = rsi(cloture)
    out["Stochastique %K"] = stochastique(df)["%K"]
    out["Williams %R"] = williams_r(df)
    out["CCI"] = cci(df)
    out["ROC 12"] = roc(cloture, 12)
    out["Momentum 12 mois"] = momentum_absolu(cloture) * 100
    out["TSI"] = tsi(cloture)
    out["MFI"] = mfi(df)

    out["Volatilité réalisée"] = volatilite_realisee(cloture)
    out["Parkinson"] = volatilite_parkinson(df)
    out["Garman-Klass"] = volatilite_garman_klass(df)
    out["Rogers-Satchell"] = volatilite_rogers_satchell(df)
    out["ATR %"] = atr(df) / cloture * 100
    out["Largeur Bollinger"] = bollinger(cloture)["Largeur"]
    out["Ulcer"] = ulcer(cloture)

    out["Volume relatif"] = volume_relatif(df)
    out["Chaikin"] = chaikin(df)
    out["OBV pente"] = obv(df).diff(20)
    out["Amihud"] = liquidite_amihud(df)

    out["Z-score"] = z_score(cloture)
    out["Distance au plus haut"] = distance_plus_haut(cloture)

    if prix_marche is not None:
        aligne = prix_marche.reindex(cloture.index).ffill()
        out["Force relative 3 mois"] = force_relative(cloture, aligne)

    return out


def instantane(df: pd.DataFrame,
               prix_marche: pd.Series | None = None) -> dict:
    """Derniere valeur de chaque indicateur, plus les mesures globales."""
    tout = calculer_tout(df, prix_marche)
    if tout.empty:
        return {}
    resultat = tout.iloc[-1].to_dict()
    resultat["Hurst"] = hurst(df["Close"])
    resultat["Ratio de variance"] = ratio_variance(df["Close"])
    return resultat
