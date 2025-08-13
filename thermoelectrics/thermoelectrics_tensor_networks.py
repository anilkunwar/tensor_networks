import streamlit as st
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# ==========================
# Global typography / math
# ==========================
# Use STIX (Times-like) with MathText for consistent mathematical notation without requiring a LaTeX install.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "text.usetex": False,  # keep False to avoid system LaTeX dependency; MathText handles $...$
})

# ==========================
# Streamlit App Header
# ==========================
st.title(r"Tensor Networks Formulation: Thermoelectric Material Simulation")
st.markdown(
    r"""
This tutorial simulates the Seebeck coefficient $\,(S)\,$ and figure of merit $\,(ZT)$ for a thermoelectric 
material (e.g. Bi$_2$Te$_3$) using a 1D tight-binding ring model with tensor networks (via *quimb* and *PlanqTN*).
Use the sidebar to adjust parameters.
"""
)

# ==========================
# Sidebar: Simulation Parameters
# ==========================
st.sidebar.header("Simulation Parameters")
temperature = st.sidebar.slider("Temperature (K)", 100.0, 500.0, 300.0, step=10.0)
doping_input = st.sidebar.text_input(
    r"Doping Concentrations ($\times 10^{19}\;\mathrm{cm}^{-3}$, comma-separated)", "1.0"
)
try:
    doping_levels = [float(x) for x in doping_input.split(",") if x.strip()]
    doping_levels = [x for x in doping_levels if x > 0]
    if not doping_levels:
        doping_levels = [1.0]
        st.sidebar.warning(
            r"No valid doping levels entered. Using default: $1.0\,\times\,10^{19}\;\mathrm{cm}^{-3}$."
        )
except ValueError:
    doping_levels = [1.0]
    st.sidebar.error(
        r"Invalid doping input. Use comma-separated numbers (e.g., \(0.5,1.2\)). Using default: \(1.0\,\times\,10^{19}\;\mathrm{cm}^{-3}\)."
    )

# ==========================
# Sidebar: Plot customization
# ==========================
st.sidebar.header("Plot Customization")
colormap = st.sidebar.selectbox("Colormap", plt.colormaps(), index=plt.colormaps().index("jet"))
line_thickness = st.sidebar.slider("Line Thickness", 0.5, 5.0, 2.0, step=0.5)
box_thickness = st.sidebar.slider("Box Line Thickness", 0.5, 3.0, 1.5, step=0.5)
tick_length = st.sidebar.slider("Tick Length", 2.0, 10.0, 4.0, step=0.5)
tick_width = st.sidebar.slider("Tick Width", 0.5, 3.0, 1.0, step=0.5)
fig_width = st.sidebar.slider("Figure Width (inches)", 4, 12, 8, step=1)
fig_height = st.sidebar.slider("Figure Height (inches)", 3, 8, 6, step=1)
legend_pos = st.sidebar.selectbox(
    "Legend Position", ["upper left", "upper right", "lower left", "lower right", "none"], index=1
)
xlabel = st.sidebar.text_input("X-Axis Label", r"$T\;\mathrm{(K)}$")
ylabel = st.sidebar.text_input("Y-Axis Label", r"$ZT$")
font_size = st.sidebar.slider("Font Size", 8, 20, 12, step=1)
show_quimb_viz = st.sidebar.checkbox("Show quimb Tensor Network Visualization", False)

# ==========================
# Hamiltonian builder (1D tight-binding ring)
# ==========================

def build_hamiltonian(N=10, t=1.0, mu=0.0, make_ring=True):
    """Build a tensor network for a 1D tight-binding Hamiltonian (ring).

    Each site has indices k{i}_in, k{i}_out. Hopping connects k{i}_out -> k{i+1}_in.
    If make_ring=True, connects k{N-1}_out -> k0_in for scalar contraction.
    """
    tensors = []
    try:
        local_h = np.array([[mu, 0.0], [0.0, mu]], dtype=float)
        for i in range(N):
            tensors.append(qtn.Tensor(data=local_h, inds=[f"k{i}_in", f"k{i}_out"], tags=[f"site{i}"]))

        hop = np.array([[0.0, t], [t, 0.0]], dtype=float)
        for i in range(N - 1):
            tensors.append(qtn.Tensor(data=hop, inds=[f"k{i}_out", f"k{i+1}_in"], tags=[f"hop{i}"]))

        if make_ring:
            tensors.append(qtn.Tensor(data=hop, inds=[f"k{N-1}_out", f"k0_in"], tags=["hop_last"]))

        tn = qtn.TensorNetwork(tensors)
        return tn
    except Exception as e:
        st.error(f"Error building tensor network: {e}")
        return None

# ==========================
# Custom ring topology drawing (with math labels)
# ==========================

def draw_ring_topology(N, fig_width, fig_height, font_size, box_thickness, tick_length, tick_width):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Plot sites as nodes
    ax.scatter(x, y, s=100, label=r"Sites")
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi + 0.250, f"Site {i+1}\n$\mu$", fontsize=font_size, ha="center", va="bottom")  # yi + 0.15 move text further and yi + 0.05 closer to the node/line

    # Plot hopping terms as edges
    for i in range(N):
        j = (i + 1) % N
        ax.plot([x[i], x[j]], [y[i], y[j]], '-', lw=2, label=r"Hopping $(t)$" if i == 0 else '')
        ax.text((x[i] + x[j]) / 2, (y[i] + y[j]) / 2, r"$t$", fontsize=font_size, ha="center", va="center")

    ax.set_aspect('equal')
    ax.set_title(r"Tight-Binding Ring Topology", fontsize=font_size, pad=55)
    ax.axis('off')
    if legend_pos != "none":
        ax.legend(loc=legend_pos, bbox_to_anchor=(0.6, 0.8), fontsize=font_size, borderpad=0.5,handletextpad=1.0,handlelength=1.0,labelspacing=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(box_thickness)
    ax.tick_params(length=tick_length, width=tick_width)
    return fig

# ==========================
# Physics: Seebeck and ZT (toy model)
# ==========================

def compute_seebeck(tn, T, doping):
    """Compute Seebeck coefficient (toy) using tensor network contraction."""
    if tn is None:
        return 0.0
    try:
        Z = tn.contract(optimize='auto')
        energy = float(np.real_if_close(Z))
    except Exception as e:
        st.error(f"Error contracting tensor network: {e}")
        return 0.0

    S = -(energy / max(T, 1e-12)) * (1.0 + 0.1 * doping)
    return S * 1e6  # microV/K


def compute_zt(S_microV_per_K, T, sigma=1000.0, kappa=1.0):
    """Compute $ZT = S^2\,\sigma\,T/\kappa$ with $S$ in $\mu$V/K."""
    S = S_microV_per_K * 1e-6
    return (S**2) * sigma * T / max(kappa, 1e-12)

# ==========================
# Main simulation
# ==========================
st.header("Simulation Results")
N = 10
hopping_t = 1.0

results = []
if doping_levels:
    for doping in doping_levels:
        mu = -0.1 * doping
        tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
        S = compute_seebeck(tn, temperature, doping)
        zt = compute_zt(S, temperature)
        results.append({
            "Doping (x 10^19 cm^-3)": f"{doping:.2f}",
            "Seebeck (μV/K)": f"{S:.2f}",
            "ZT": f"{zt:.2e}",
        })
else:
    st.warning(r"No valid doping levels. Using default: $1.0\,\times\,10^{19}\;\mathrm{cm}^{-3}$.")
    mu = -0.1 * 1.0
    tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
    S = compute_seebeck(tn, temperature, 1.0)
    zt = compute_zt(S, temperature)
    # Present scalar results as proper math typeset
    st.latex(rf"S = {S:.2f}\,\mu\mathrm{{V}}\,\mathrm{{K}}^{{-1}}")
    st.latex(rf"ZT = {zt:.2e}")

if results:
    st.subheader("Results for Selected Doping Levels")
    # Explain columns using TeX, then show numeric table
    st.markdown(r"**Columns:** Doping ($\times 10^{19}\;\mathrm{cm}^{-3}$), Seebeck $S$ ($\mu\mathrm{V\,K}^{-1}$), and $ZT$ (dimensionless).")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

# ==========================
# Tensor Network Visualization
# ==========================
st.header("Tensor Network Visualization")
if show_quimb_viz:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    tn = build_hamiltonian(N, hopping_t, -0.1 * float(doping_levels[0]), make_ring=True)
    if tn:
        tn.draw(ax=ax)
        ax.set_title(r"quimb Tensor Network (Ring)", fontsize=font_size)
        ax.tick_params(length=tick_length, width=tick_width)
        for spine in ax.spines.values():
            spine.set_linewidth(box_thickness)
    st.pyplot(fig)
else:
    fig = draw_ring_topology(N, fig_width, fig_height, font_size, box_thickness, tick_length, tick_width)
    st.pyplot(fig)

# ==========================
# Plot: ZT vs Temperature (math labels)
# ==========================
st.header(r"$ZT$ vs. $T$")
temps = np.linspace(100.0, 500.0, 50)
zt_data = []
colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, max(len(doping_levels), 1)))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
if doping_levels:
    for idx, doping in enumerate(doping_levels):
        mu = -0.1 * doping
        tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
        zts = [compute_zt(compute_seebeck(tn, T, doping), T) for T in temps]
        zt_data.append(zts)
        ax.plot(
            temps,
            zts,
            label=rf"Doping: {doping:.2f}\,\times\,10^{{19}}\;\mathrm{{cm}}^{{-3}}",
            color=colors[idx],
            linewidth=line_thickness,
        )
else:
    mu = -0.1 * 1.0
    tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
    zts = [compute_zt(compute_seebeck(tn, T, 1.0), T) for T in temps]
    zt_data.append(zts)
    ax.plot(temps, zts, label=r"Doping: $1.0\,\times\,10^{19}\;\mathrm{cm}^{-3}$", color=colors[0], linewidth=line_thickness)

ax.set_xlabel(xlabel, fontsize=font_size)
ax.set_ylabel(ylabel, fontsize=font_size)
ax.tick_params(axis='both', labelsize=font_size, length=tick_length, width=tick_width)
for spine in ax.spines.values():
    spine.set_linewidth(box_thickness)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
if legend_pos != "none":
    leg = ax.legend(loc=legend_pos, fontsize=font_size)
    # Make legend text math-friendly (already is) and spacing neat

plt.tight_layout()
st.pyplot(fig)

# ==========================
# Data download option
# ==========================
st.header("Download Data")
if zt_data:
    df = pd.DataFrame({f"ZT (Doping {doping:.2f})": zts for doping, zts in zip(doping_levels or [1.0], zt_data)}, index=temps)
    df.index.name = "Temperature (K)"
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label=r"Download $ZT$ Data as CSV",
        data=csv,
        file_name="zt_data.csv",
        mime="text/csv",
    )

# ==========================
# Tutorial Explanation (typeset with LaTeX)
# ==========================
st.header("Tutorial Explanation")

st.markdown(r"### Hamiltonian")
st.latex(r"\hat{H} = \sum_{i=1}^N \mu\,\hat{c}_i^{\dagger}\hat{c}_i + \sum_{i=1}^N t\,\big(\hat{c}_i^{\dagger}\hat{c}_{i+1} + \hat{c}_{i+1}^{\dagger}\hat{c}_i\big),\qquad \hat{c}_{N+1}\equiv\hat{c}_1.")
st.markdown(r"Parameters: $\mu = -0.1\times\text{doping}$, $t = 1\,\mathrm{eV}$, $N=10$, doping in $10^{19}\,\mathrm{cm}^{-3}$. ")

st.markdown(r"### Tensor Network")
st.latex(r"H_i = \begin{pmatrix} \mu & 0 \\ 0 & \mu \end{pmatrix},\quad H_{i,i+1} = \begin{pmatrix} 0 & t \\ t & 0 \end{pmatrix}.")
st.latex(r"Z = \sum_{k_1,\ldots,k_N} \prod_{i=1}^N H_i(k_i^{\mathrm{in}},k_i^{\mathrm{out}})\, \prod_{i=1}^N H_{i,i+1}(k_i^{\mathrm{out}},k_{i+1}^{\mathrm{in}}) \approx N\mu.")

st.markdown(r"### Seebeck Coefficient")
st.latex(r"S \approx -\dfrac{\langle E\rangle}{T}\,\big(1 + 0.1\times\text{doping}\big),\qquad \langle E\rangle \approx Z.")

st.markdown(r"### Figure of Merit")
st.latex(r"ZT = \dfrac{S^2\,\sigma\,T}{\kappa},\qquad S\;\text{in}\;\mathrm{V\,K}^{-1},\; \sigma = 1000\,\mathrm{S\,m}^{-1},\; \kappa = 1\,\mathrm{W\,m}^{-1}\,\mathrm{K}^{-1}.")

st.markdown(r"""
**Steps in the Simulation**
1. **Build Hamiltonian**: Create a 1D ring tensor network.
2. **Contract Network**: Ring topology contracts to a scalar.
3. **Seebeck**: Compute $S$ from contracted energy.
4. **ZT**: Evaluate $ZT$ for multiple doping levels.
5. **Visualization**: Custom ring diagram or quimb visualization; customizable $ZT$ plot.

**Why PlanqTN?** Efficient tensor contractions via *quimb*/*cotengra*; adaptable for materials science; open-source and extensible.

**Extensions**: add phonon interactions for Bi$_2$Te$_3$, use DFT for accurate parameters, optimize for $ZT>2$.
""")

#st.caption("Run with: `streamlit run thermoelectrics_tutorial.py`")

