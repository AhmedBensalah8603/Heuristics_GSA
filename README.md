# 🌌 Application GSA (Gravitational Search Algorithm)

Cette application Streamlit illustre l'**Algorithme de Recherche Gravitationnelle (GSA)** appliqué à des fonctions benchmark. L'utilisateur peut régler les paramètres et visualiser la convergence en temps réel.

## Fonctionnalités
- Fonctions objectif : **f1 (Sphere, unimodale)** et **f8 (Schwefel-like, multimodale)**
- Paramètres personnalisables : G0, alpha, population N, dimensions D, Tmax, bornes LB/UB
- Visualisations : barre de progression, courbes de convergence, comparaisons multi-tests
- Mode multi-tests pour comparer différentes combinaisons G0/alpha

## Installation
```bash
git clone https://github.com/votre-utilisateur/gsa-streamlit.git
cd gsa-streamlit
pip install -r requirements.txt
