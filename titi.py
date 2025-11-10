# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep

# ========== FONCTIONS OBJECTIF ==========
def f1(x):
    return np.sum(x**2, axis=1)

def f8(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

# ========== IMPLEMENTATION GSA (version K-best) ==========
def run_gsa(G0, alpha, func, N=100, D=30, LB=-500, UB=500, Tmax=300, epsilon=1e-15):
    t = 0
    x = np.random.uniform(LB, UB, (N, D))  # positions
    v = np.zeros((N, D))                    # vitesses
    bestvalues = []

    # Fitness initiale et meilleur global
    fitness = func(x)
    g_index = np.argmin(fitness)
    gbest = x[g_index].copy()
    bestvalues.append(fitness[g_index])

    progress = st.progress(0, text="Exécution de l'optimisation GSA...")

    while t < Tmax:
        G = G0 * np.exp(-alpha * t / Tmax)

        # --- Calcul des masses ---
        worst = np.max(fitness)
        best = np.min(fitness)
        if np.allclose(fitness, fitness[0]):
            mass = np.ones(N) / N
        else:
            m = (fitness - worst) / (best - worst + epsilon)
            mass = m / (np.sum(m) + epsilon)
            mass = np.clip(mass, 0, None)

        # --- Détermination des K meilleurs agents ---
        K = max(1, int(np.ceil(N * (Tmax - t) / Tmax)))  # au moins 1
        Kbest_idx = np.argsort(fitness)[:K]

        # --- Calcul des forces et accélérations ---
        a = np.zeros((N, D))
        for i in range(N):
            F = np.zeros(D)
            for j in Kbest_idx:
                if j != i:
                    diff = x[j] - x[i]
                    dist = np.linalg.norm(diff) + epsilon
                    F += np.random.rand() * G * mass[j] * diff / dist
            a[i] = F

        # --- Mise à jour des vitesses et positions ---
        v = np.random.rand(N, D) * v + a
        x = np.clip(x + v, LB, UB)

        # --- Mise à jour de la fitness et du meilleur global ---
        fitness = func(x)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < func(gbest[np.newaxis, :])[0]:
            gbest = x[current_best_idx].copy()

        bestvalues.append(func(gbest[np.newaxis, :])[0])
        t += 1
        progress.progress(t / Tmax, text=f"Iteration {t}/{Tmax}")

    progress.empty()
    return bestvalues

# ========== MULTI-TESTS POUR BENCHMARKS ==========
def run_tests(tests):
    results = []
    allcurves = {}
    funcs = [(f1, 'f1 Unimodal'), (f8, 'f8 Multimodal')]
    for G0, alpha in tests:
        for func, fname in funcs:
            bestvalues = run_gsa(G0, alpha, func)
            results.append({'Fonction': fname, 'G0': G0, 'alpha': alpha, 'Meilleur': bestvalues[-1]})
            allcurves.setdefault(fname, {})[f"G0={G0}, alpha={alpha}"] = bestvalues
    df = pd.DataFrame(results)
    return df, allcurves

# ========== GRAPHIQUES ==========
def plot_comparisons(df, allcurves):
    st.subheader("📊 Comparaison des meilleurs résultats finaux")
    for fname in df['Fonction'].unique():
        subset = df[df['Fonction'] == fname]
        st.write(f"**Fonction :** {fname}")
        st.bar_chart(subset.set_index(subset.index.astype(str))['Meilleur'])

    st.subheader("📉 Courbes de convergence")
    for fname, curves in allcurves.items():
        st.markdown(f"### {fname}")
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, curve in curves.items():
            ax.plot(curve, label=label)
        ax.set_xlabel("Itérations")
        ax.set_ylabel("Valeur de la fitness")
        ax.set_title(f"Courbes de convergence pour {fname}")
        ax.legend()
        st.pyplot(fig)

# ========== INTERFACE STREAMLIT ==========
st.set_page_config(page_title="Algorithme de Recherche Gravitationnelle", layout="centered")
st.title("🌌 Optimisation par Algorithme de Recherche Gravitationnelle (GSA)")

st.markdown("""
Cette application démontre l'**Algorithme de Recherche Gravitationnelle (GSA)** appliqué à des fonctions de référence classiques.
Ajustez les paramètres ci-dessous et visualisez la convergence !
""")

# --- Contrôles dans la barre latérale ---
st.sidebar.header("⚙️ Paramètres GSA")

func_option = st.sidebar.selectbox("Fonction objectif", ('f1 Unimodale (Sphere)', 'f8 Multimodale (type Schwefel)'))
func = f1 if 'f1' in func_option else f8

G0 = st.sidebar.slider("Constante gravitationnelle initiale (G0)", 1.0, 200.0, 100.0)
alpha = st.sidebar.slider("Taux de décroissance (alpha)", 1.0, 50.0, 20.0)
N = st.sidebar.slider("Taille de population (N)", 10, 200, 100, step=10)
D = st.sidebar.slider("Dimensions (D)", 2, 50, 30, step=2)
Tmax = st.sidebar.slider("Nombre d'itérations (Tmax)", 50, 1000, 300, step=50)
LB = st.sidebar.number_input("Borne inférieure (LB)", value=-500.0)
UB = st.sidebar.number_input("Borne supérieure (UB)", value=500.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Lancer GSA")

# --- Lancer l'optimisation ---
if run_button:
    with st.spinner("Exécution de l'optimisation GSA..."):
        bestvalues = run_gsa(G0, alpha, func, N=N, D=D, LB=LB, UB=UB, Tmax=Tmax)

    st.success("✅ Optimisation terminée !")
    st.write(f"**Meilleure fitness trouvée :** {bestvalues[-1]:.6f}")
    st.line_chart(bestvalues, height=300, use_container_width=True)

# --- Mode test optionnel ---
test_mode = st.checkbox("📈 Comparer plusieurs combinaisons (G0, α)")

if test_mode:
    st.info("Exécution des tests prédéfinis avec différentes valeurs de G0 et alpha...")
    tests = [(100, 20), (50, 10), (150, 30), (200, 40)]
    df, allcurves = run_tests(tests)
    st.dataframe(df)
    plot_comparisons(df, allcurves)

st.markdown("---")
st.caption("Développé par Ahmed Bensalah & Taher Chaltout | © 2025")
