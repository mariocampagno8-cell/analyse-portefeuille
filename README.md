# Analyse quantitative de portefeuille

Application Streamlit d'analyse de portefeuille et de valeurs mondiales.

## Modules

- `app.py` — interface (10 onglets)
- `analytics.py` — mesures de risque, bêta, alpha, VaR, décomposition
- `optimisation.py` — covariance robuste, allocations, backtest walk-forward
- `indicateurs.py` — 31 indicateurs techniques et statistiques
- `univers.py` — 407 valeurs sur 34 places, 24 indices de référence

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Avertissement

Cet outil calcule, il ne conseille pas. Les résultats reposent sur des données
passées et sur des estimations statistiques imprécises. Aucun élément ne
constitue un conseil en investissement.
