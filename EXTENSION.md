# Faire évoluer l'outil

L'architecture est modulaire exprès : chaque type d'ajout se fait dans un seul
fichier, sans toucher au reste. Voici les quatre cas les plus fréquents.

| Ce que tu veux ajouter | Fichier à modifier | Effort |
|---|---|---|
| Un indicateur technique | `indicateurs.py` | 5 lignes |
| Un ratio de risque | `analytics.py` | 8 lignes |
| Des valeurs à screener | `univers.py` | 1 ligne |
| Une méthode d'allocation | `optimisation.py` | 15 lignes |

---

## 1. Ajouter un indicateur

Trois étapes. Prenons l'exemple du **Coppock**, un indicateur de retournement
de long terme.

**Écris la fonction** dans `indicateurs.py`, dans la section de sa famille :

```python
def coppock(prix: pd.Series, n1: int = 14, n2: int = 11, n3: int = 10) -> pd.Series:
    """Somme de deux taux de variation, lissée. Signale les creux majeurs."""
    return wma(roc(prix, n1) + roc(prix, n2), n3)
```

**Déclare-le** dans le dictionnaire `FAMILLES`, en haut de la section synthèse :

```python
"Momentum": ["RSI", "Stochastique %K", ..., "MFI", "Coppock"],
```

**Branche-le** dans `calculer_tout()` :

```python
out["Coppock"] = coppock(cloture)
```

C'est tout. Il apparaîtra automatiquement dans l'onglet Indicateurs, dans les
graphiques, et dans la matrice de corrélation.

---

## 2. Ajouter un ratio de risque

Dans `analytics.py`. Exemple du **ratio de Burke**, qui pénalise l'ensemble
des drawdowns et pas seulement le pire :

```python
def burke(r: pd.Series, freq: float = JOURS_BOURSE) -> float:
    """Rendement rapporté à la racine de la somme des carrés des drawdowns."""
    dd = courbe_drawdown(r)
    denominateur = np.sqrt((dd ** 2).sum())
    if denominateur == 0:
        return np.nan
    return rendement_annualise(r, freq) / denominateur
```

Puis ajoute une ligne dans `tableau_metriques()` :

```python
"Ratio de Burke": burke(r, freq),
```

Il apparaîtra dans le tableau comparatif de l'onglet Risque, calculé pour le
portefeuille, l'indice et chaque ligne.

---

## 3. Ajouter des valeurs

Dans `univers.py`, une simple liste :

```python
BIOTECH_US = ["MRNA", "BNTX", "REGN", "VRTX", "AMGN", "GILD", "BIIB"]
```

Puis une entrée dans le dictionnaire `UNIVERS` :

```python
UNIVERS = {
    ...,
    "Biotech US": BIOTECH_US,
}
```

Elle apparaît immédiatement dans le menu du screener.

---

## 4. Ajouter une méthode d'allocation

Dans `optimisation.py`. Exemple d'une pondération inverse à la volatilité :

```python
def inverse_volatilite(cov: pd.DataFrame) -> pd.Series:
    """Pondère chaque ligne à l'inverse de sa volatilité. Version simplifiée
    de la parité de risque, qui ignore les corrélations."""
    inv = 1 / np.sqrt(np.diag(cov.to_numpy()))
    return pd.Series(inv / inv.sum(), index=cov.index)
```

Trois branchements ensuite. Dans `_poids_methode()` :

```python
if methode == "inverse_vol":
    return inverse_volatilite(cov)
```

Dans le dictionnaire `METHODES` :

```python
"inverse_vol": "Inverse volatilité",
```

Et dans l'onglet Optimisation de `app.py` :

```python
allocations["Inverse volatilité"] = opt.inverse_volatilite(cov_opt)
```

Elle sera alors comparable aux autres, backtestable, et positionnée sur la
frontière efficiente.

---

## Méthode de travail

**Teste avant de brancher.** Chaque module est utilisable seul, sans Streamlit.
Ouvre un terminal Python et vérifie ta fonction sur des données simulées :

```python
import numpy as np, pandas as pd, indicateurs as ind
prix = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500))))
print(ind.coppock(prix).tail())
```

Si le résultat est cohérent, branche. Sinon, corrige avant que l'erreur ne se
propage dans l'interface, où elle sera beaucoup plus difficile à localiser.

**Vérifie la redondance.** Après avoir ajouté un indicateur, regarde sa
corrélation avec les existants dans l'onglet Indicateurs. Au-dessus de 0,9, il
n'apporte rien de nouveau — Williams %R et le stochastique sont corrélés à
1,00, ce sont littéralement le même calcul.

**Backteste avant de croire.** Un indicateur qui a l'air convaincant sur un
graphique ne vaut rien tant qu'il n'a pas été testé hors échantillon. L'onglet
Backtest existe pour ça.

**GitHub garde l'historique.** Chaque dépôt de fichier crée une version. Si un
ajout casse l'application, tu reviens en arrière depuis l'onglet History de ton
dépôt, sans rien perdre.

---

## Les extensions qui en valent vraiment la peine

Par ordre décroissant d'apport réel, d'après ce que la recherche documente :

**L'analyse factorielle.** Régresser tes lignes sur les facteurs Fama-French
(taille, valeur, momentum, rentabilité, investissement) plutôt que sur un seul
indice. Tu découvriras souvent que ce que tu prenais pour du talent est une
exposition passive à un facteur connu. Les données sont téléchargeables
gratuitement depuis la bibliothèque de Kenneth French.

**Les données fondamentales.** yfinance donne accès aux bilans, comptes de
résultat et flux de trésorerie. Les ratios de qualité — rentabilité des
capitaux, marge, endettement, croissance des bénéfices — sont mieux documentés
comme prédicteurs que la plupart des indicateurs techniques.

**Le suivi des décisions.** Enregistrer chaque arbitrage avec sa justification
écrite, puis mesurer après coup lesquelles de tes thèses se vérifient. Peu
spectaculaire, souvent le plus rentable : c'est la seule extension qui
t'apprend quelque chose sur toi plutôt que sur les marchés.

**Le dimensionnement des positions.** Le critère de Kelly fractionnel ou une
pondération par l'ATR. Combien mettre sur une idée compte davantage que le
choix de l'idée elle-même.
