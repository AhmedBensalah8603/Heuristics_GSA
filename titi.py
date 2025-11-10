# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep

# ========== OBJECTIVE FUNCTIONS ==========
def f1(x):
    return np.sum(x**2, axis=1)

def f8(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

# ========== GSA IMPLEMENTATION (K-best version) ==========
def run_gsa(G0, alpha, func, N=100, D=30, LB=-500, UB=500, Tmax=300, epsilon=1e-15):
    t = 0
    x = np.random.uniform(LB, UB, (N, D))  # positions
    v = np.zeros((N, D))                   # velocities
    bestvalues = []

    # Initial fitness and global best
    fitness = func(x)
    g_index = np.argmin(fitness)
    gbest = x[g_index].copy()
    bestvalues.append(fitness[g_index])

    progress = st.progress(0, text="Running GSA optimization...")

    while t < Tmax:
        G = G0 * np.exp(-alpha * t / Tmax)

        # --- Mass calculation ---
        worst = np.max(fitness)
        best = np.min(fitness)
        if np.allclose(fitness, fitness[0]):
            mass = np.ones(N) / N
        else:
            m = (fitness - worst) / (best - worst + epsilon)
            mass = m / (np.sum(m) + epsilon)
            mass = np.clip(mass, 0, None)

        # --- Determine K-best agents ---
        K = max(1, int(np.ceil(N * (Tmax - t) / Tmax)))  # ensure at least 1
        Kbest_idx = np.argsort(fitness)[:K]

        # --- Force and acceleration calculation ---
        a = np.zeros((N, D))
        for i in range(N):
            F = np.zeros(D)
            for j in Kbest_idx:
                if j != i:
                    diff = x[j] - x[i]
                    dist = np.linalg.norm(diff) + epsilon
                    F += np.random.rand() * G * mass[j] * diff / dist
            a[i] = F

        # --- Update velocities and positions ---
        v = np.random.rand(N, D) * v + a
        x = np.clip(x + v, LB, UB)

        # --- Update fitness and global best ---
        fitness = func(x)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < func(gbest[np.newaxis, :])[0]:
            gbest = x[current_best_idx].copy()

        bestvalues.append(func(gbest[np.newaxis, :])[0])
        t += 1
        progress.progress(t / Tmax, text=f"Iteration {t}/{Tmax}")

    progress.empty()
    return bestvalues


# ========== MULTI-TESTS FOR BENCHMARKS ==========
def run_tests(tests):
    results = []
    allcurves = {}
    funcs = [(f1, 'f1 Unimodal'), (f8, 'f8 Multimodal')]
    for G0, alpha in tests:
        for func, fname in funcs:
            bestvalues = run_gsa(G0, alpha, func)
            results.append({'Function': fname, 'G0': G0, 'alpha': alpha, 'Best': bestvalues[-1]})
            allcurves.setdefault(fname, {})[f"G0={G0}, alpha={alpha}"] = bestvalues
    df = pd.DataFrame(results)
    return df, allcurves


# ========== PLOTS ==========
def plot_comparisons(df, allcurves):
    st.subheader("📊 Best Final Values Comparison")
    for fname in df['Function'].unique():
        subset = df[df['Function'] == fname]
        st.write(f"**Function:** {fname}")
        st.bar_chart(subset.set_index(subset.index.astype(str))['Best'])

    st.subheader("📉 Convergence Curves")
    for fname, curves in allcurves.items():
        st.markdown(f"### {fname}")
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, curve in curves.items():
            ax.plot(curve, label=label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Fitness Value")
        ax.set_title(f"Convergence Curves for {fname}")
        ax.legend()
        st.pyplot(fig)


# ========== STREAMLIT INTERFACE ==========
st.set_page_config(page_title="Gravitational Search Algorithm", layout="centered")
st.title("🌌 Gravitational Search Algorithm (GSA) Optimization")

st.markdown("""
This app demonstrates the **Gravitational Search Algorithm (GSA)** applied to common benchmark functions.
Adjust parameters below and visualize convergence!
""")

# --- Sidebar controls ---
st.sidebar.header("⚙️ GSA Parameters")

func_option = st.sidebar.selectbox("Objective Function", ('f1 Unimodal (Sphere)', 'f8 Multimodal (Schwefel-like)'))
func = f1 if 'f1' in func_option else f8

G0 = st.sidebar.slider("Initial gravitational constant (G0)", 1.0, 200.0, 100.0)
alpha = st.sidebar.slider("Decay rate (alpha)", 1.0, 50.0, 20.0)
N = st.sidebar.slider("Population size (N)", 10, 200, 100, step=10)
D = st.sidebar.slider("Dimensions (D)", 2, 50, 30, step=2)
Tmax = st.sidebar.slider("Iterations (Tmax)", 50, 1000, 300, step=50)
LB = st.sidebar.number_input("Lower Bound (LB)", value=-500.0)
UB = st.sidebar.number_input("Upper Bound (UB)", value=500.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run GSA")

# --- Run Optimization ---
if run_button:
    with st.spinner("Running GSA optimization..."):
        bestvalues = run_gsa(G0, alpha, func, N=N, D=D, LB=LB, UB=UB, Tmax=Tmax)

    st.success("✅ Optimization Complete!")
    st.write(f"**Best fitness found:** {bestvalues[-1]:.6f}")
    st.line_chart(bestvalues, height=300, use_container_width=True)

# --- Optional test mode ---
test_mode = st.checkbox("📈 Compare Multiple (G0, α) Combinations")

if test_mode:
    st.info("Running predefined tests with various G0 and alpha values...")
    tests = [(100, 20), (50, 10), (150, 30), (200, 40)]
    df, allcurves = run_tests(tests)
    st.dataframe(df)
    plot_comparisons(df, allcurves)

st.markdown("---")
st.caption("Developed by Sali7a & Titi | © 2025")
