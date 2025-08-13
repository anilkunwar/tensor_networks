import streamlit as st
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from mp_api.client import MPRester
from scipy.integrate import trapezoid as trapz

# Constants
k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

# Material-specific parameters (literature defaults)
material_params = {
    "Bi2Te3": {"mp_id": "mp-34202", "t": 0.3, "mu_scale": 0.05, "sigma": 5000.0, "kappa": 1.5, "doping_scale": 0.1},
    "PbTe": {"mp_id": "mp-19717", "t": 0.4, "mu_scale": 0.05, "sigma": 1e4, "kappa": 2.0, "doping_scale": 0.1},
    "SnSe": {"mp_id": "mp-1190", "t": 0.1, "mu_scale": 0.03, "sigma": 1000.0, "kappa": 0.7, "doping_scale": 0.1}
}

# Functions to compute t and mu
def compute_t(bandstructure, carrier_type="n"):
    """Estimate hopping parameter t from band structure bandwidth."""
    try:
        if bandstructure is None or not hasattr(bandstructure, 'bands') or bandstructure.bands is None:
            raise ValueError("Band structure data is unavailable or invalid.")
        energies = bandstructure.bands[0]  # Spin-up
        if energies.size == 0:
            raise ValueError("Band structure energies are empty.")
        # Select conduction or valence bands based on carrier type
        efermi = bandstructure.efermi
        if carrier_type == "n":
            mask = energies > efermi  # Conduction bands
        else:
            mask = energies < efermi  # Valence bands
        if not np.any(mask):
            raise ValueError(f"No {'conduction' if carrier_type == 'n' else 'valence'} band energies available.")
        relevant_energies = energies[mask]
        bandwidth = np.max(relevant_energies) - np.min(relevant_energies)
        if bandwidth <= 0:
            raise ValueError("Invalid bandwidth computed.")
        t_value = bandwidth / 4.0  # For 1D tight-binding, bandwidth ~ 4t
        st.write(f"Computed bandwidth for {carrier_type}-type carriers: {bandwidth:.3f} eV, t = {t_value:.3f} eV")
        return t_value
    except Exception as e:
        st.error(f"Error computing t: {str(e)}")
        return None

def compute_mu(dos, doping_cm3, T, carrier_type="n"):
    """Compute chemical potential mu for given doping and temperature."""
    try:
        if dos is None or not hasattr(dos, 'energies') or not hasattr(dos, 'efermi'):
            raise ValueError("DOS data is unavailable or invalid.")
        energies = np.array(dos.energies) - dos.efermi  # Energies relative to Fermi level
        density = np.array(dos.densities["1"])  # Total DOS
        if len(energies) != len(density):
            raise ValueError("Mismatch between energies and DOS density arrays.")
        fermi_dirac = lambda E, mu: 1 / (1 + np.exp((E - mu) / (k_B * T)))
        volume = dos.structure.volume * 1e-24  # Angstrom^3 to cm^3
        doping_per_cell = doping_cm3 * volume
        
        def carrier_concentration(mu):
            mask = energies > 0 if carrier_type == "n" else energies < 0
            if not np.any(mask):
                raise ValueError(f"No {'positive' if carrier_type == 'n' else 'negative'} energies for {carrier_type}-type carriers.")
            integrand = density[mask] * fermi_dirac(energies[mask], mu)
            integral = trapz(integrand, energies[mask])
            return integral - doping_per_cell
        
        mu_min, mu_max = -1.0, 1.0
        tolerance = 1e-6
        max_iterations = 1000
        iteration = 0
        while mu_max - mu_min > tolerance and iteration < max_iterations:
            mu_mid = (mu_min + mu_max) / 2
            n = carrier_concentration(mu_mid)
            if n > 0:
                mu_max = mu_mid
            else:
                mu_min = mu_mid
            iteration += 1
        if iteration >= max_iterations:
            raise ValueError("Chemical potential computation did not converge.")
        mu_scale = (mu_min + mu_max) / 2 / (doping_cm3 / 1e19)
        st.write(f"Computed mu_scale for doping {doping_cm3/1e19:.2f} x 10^19 cm^-3: {mu_scale:.3f}")
        return mu_scale
    except Exception as e:
        st.error(f"Error computing mu: {str(e)}")
        return None

# Global typography
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "text.usetex": False,
})

# Streamlit App Header
st.title(r"PlanqTN Tutorial: Thermoelectric Material Simulation")
st.markdown(
    r"""
This tutorial simulates the Seebeck coefficient $\,(S)\,$ and figure of merit $\,(ZT)$ for thermoelectric 
materials (Bi$_2$Te$_3$, PbTe, SnSe) using a 1D tight-binding ring model with tensor networks (via *quimb*).
Enter a Materials Project API key to fetch $t$ and $\mu$, or use literature defaults. Adjust parameters in the sidebar.
"""
)

# Sidebar: Simulation Parameters
st.sidebar.header("Simulation Parameters")
material = st.sidebar.selectbox("Material", ["Bi2Te3", "PbTe", "SnSe"], index=0)
api_key = st.sidebar.text_input("Materials Project API Key", type="password", help="Get your API key from https://materialsproject.org")
use_mp = st.sidebar.checkbox("Fetch Parameters from Materials Project", value=False)
temperature = st.sidebar.slider(r"Temperature ($K$)", 100.0, 1000.0, 300.0, step=10.0)
doping_input = st.sidebar.text_input(
    r"Doping Concentrations ($\times 10^{19}\;\mathrm{cm}^{-3}$, comma-separated)", "1.0"
)

# Fetch parameters from Materials Project if API key provided
if use_mp and api_key:
    try:
        with MPRester(api_key) as mpr:
            bandstructure = mpr.get_bandstructure_by_material_id(material_params[material]["mp_id"])
            dos = mpr.get_dos_by_material_id(material_params[material]["mp_id"])
            t_mp = compute_t(bandstructure, carrier_type="n") if bandstructure else None
            mu_scale_mp = compute_mu(dos, 1e19, temperature, carrier_type="n") if dos else None
        if t_mp is not None:
            material_params[material]["t"] = t_mp
        if mu_scale_mp is not None:
            material_params[material]["mu_scale"] = mu_scale_mp
        st.sidebar.success(f"Fetched parameters for {material}: t = {material_params[material]['t']:.3f} eV, mu_scale = {material_params[material]['mu_scale']:.3f}")
    except Exception as e:
        st.sidebar.error(f"Error fetching from Materials Project: {str(e)}. Using literature values: t = {material_params[material]['t']:.3f} eV, mu_scale = {material_params[material]['mu_scale']:.3f}")

# Parameter sliders with defaults from material_params
hopping_t = st.sidebar.slider(r"Hopping Parameter $t$ (eV)", 0.01, 2.0, material_params[material]["t"], step=0.01)
mu_scale = st.sidebar.slider(r"Chemical Potential Scale", -1.0, 1.0, material_params[material]["mu_scale"], step=0.01)
sigma = st.sidebar.slider(r"Electrical Conductivity $\sigma$ (S/m)", 100.0, 1e5, material_params[material]["sigma"], step=100.0)
kappa = st.sidebar.slider(r"Thermal Conductivity $\kappa$ (W/m$\cdot$K)", 0.1, 5.0, material_params[material]["kappa"], step=0.1)
N = st.sidebar.slider("Number of Sites $N$", 5, 50, 10, step=1)
doping_scale = st.sidebar.slider(r"Doping Scale for $S$", 0.01, 1.0, material_params[material]["doping_scale"], step=0.01)

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

# Sidebar: Plot customization
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

# Hamiltonian builder
def build_hamiltonian(N, t, mu, make_ring=True):
    """Build a tensor network for a 1D tight-binding Hamiltonian (ring)."""
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

# Custom ring topology drawing
def draw_ring_topology(N, fig_width, fig_height, font_size, box_thickness, tick_length, tick_width):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    ax.scatter(x, y, s=100, label=r"Sites")
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi + 0.2, f"Site {i+1}\n$\mu$", fontsize=font_size, ha="center", va="bottom")

    for i in range(N):
        j = (i + 1) % N
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', lw=2, label=r"Hopping ($t$)" if i == 0 else '')
        ax.text((x[i] + x[j]) / 2, (y[i] + y[j]) / 2, r"$t$", fontsize=font_size, ha="center", va="center")

    ax.set_aspect('equal')
    ax.set_title(r"Tight-Binding Ring Topology", fontsize=font_size)
    ax.axis('off')
    if legend_pos != "none":
        ax.legend(loc=legend_pos, fontsize=font_size)
    for spine in ax.spines.values():
        spine.set_linewidth(box_thickness)
    ax.tick_params(length=tick_length, width=tick_width)
    return fig

# Physics: Seebeck and ZT
def compute_seebeck(tn, T, doping, doping_scale):
    """Compute Seebeck coefficient (toy) using tensor network contraction."""
    if tn is None:
        return 0.0
    try:
        Z = tn.contract(optimize='auto')
        energy = float(np.real_if_close(Z))
    except Exception as e:
        st.error(f"Error contracting tensor network: {e}")
        return 0.0

    S = -(energy / max(T, 1e-12)) * (1.0 + doping_scale * doping)
    return S * 1e6  # microV/K

def compute_zt(S_microV_per_K, T, sigma, kappa):
    """Compute $ZT = S^2\,\sigma\,T/\kappa$ with $S$ in $\mu$V/K."""
    S = S_microV_per_K * 1e-6
    return (S**2) * sigma * T / max(kappa, 1e-12)

# Main simulation
st.header("Simulation Results")
results = []
if doping_levels:
    for doping in doping_levels:
        mu = mu_scale * doping
        tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
        S = compute_seebeck(tn, temperature, doping, doping_scale)
        zt = compute_zt(S, temperature, sigma, kappa)
        results.append({
            r"Doping ($\times 10^{19}\;\mathrm{cm}^{-3}$)": f"{doping:.2f}",
            r"Seebeck ($S$, $\mu\mathrm{V}\;\mathrm{K}^{-1}$)": f"{S:.2f}",
            r"$ZT$": f"{zt:.2e}",
        })
else:
    st.warning(r"No valid doping levels. Using default: $1.0\,\times\,10^{19}\;\mathrm{cm}^{-3}$.")
    mu = mu_scale * 1.0
    tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
    S = compute_seebeck(tn, temperature, 1.0, doping_scale)
    zt = compute_zt(S, temperature, sigma, kappa)
    st.markdown(rf"Seebeck Coefficient: ${S:.2f}\;\mu\mathrm{{V}}\;\mathrm{{K}}^{{-1}}$")
    st.markdown(rf"Figure of Merit ($ZT$): ${zt:.2e}$")

if results:
    st.subheader(f"Results for {material}")
    st.markdown(r"**Columns**: Doping ($\times 10^{19}\;\mathrm{cm}^{-3}$), Seebeck $S$ ($\mu\mathrm{V}\;\mathrm{K}^{-1}$), $ZT$ (dimensionless).")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

# Tensor Network Visualization
st.header("Tensor Network Visualization")
if show_quimb_viz:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    tn = build_hamiltonian(N, hopping_t, mu_scale * float(doping_levels[0] if doping_levels else 1.0), make_ring=True)
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

# Plot: ZT vs Temperature
st.header(r"$ZT$ vs. $T$")
temps = np.linspace(100.0, 1000.0, 50)
zt_data = []
colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, max(len(doping_levels), 1)))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
if doping_levels:
    for idx, doping in enumerate(doping_levels):
        mu = mu_scale * doping
        tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
        zts = [compute_zt(compute_seebeck(tn, T, doping, doping_scale), T, sigma, kappa) for T in temps]
        zt_data.append(zts)
        ax.plot(
            temps,
            zts,
            label=rf"Doping: {doping:.2f}\;\times\;10^{{19}}\;\mathrm{{cm}}^{{-3}}",
            color=colors[idx],
            linewidth=line_thickness,
        )
else:
    mu = mu_scale * 1.0
    tn = build_hamiltonian(N, hopping_t, mu, make_ring=True)
    zts = [compute_zt(compute_seebeck(tn, T, 1.0, doping_scale), T, sigma, kappa) for T in temps]
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
    ax.legend(loc=legend_pos, fontsize=font_size)
plt.tight_layout()
st.pyplot(fig)

# Data download option
st.header("Download Data")
if zt_data:
    df = pd.DataFrame({f"ZT (Doping {doping:.2f})": zts for doping, zts in zip(doping_levels or [1.0], zt_data)}, index=temps)
    df.index.name = "Temperature (K)"
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label=r"Download $ZT$ Data as CSV",
        data=csv,
        file_name=f"zt_data_{material}.csv",
        mime="text/csv",
    )

# Tutorial Explanation
st.header("Tutorial Explanation")
st.markdown(r"### Hamiltonian")
st.latex(r"\hat{H} = \sum_{i=1}^N \mu\,\hat{c}_i^{\dagger}\hat{c}_i + \sum_{i=1}^N t\,\big(\hat{c}_i^{\dagger}\hat{c}_{i+1} + \hat{c}_{i+1}^{\dagger}\hat{c}_i\big),\qquad \hat{c}_{N+1}\equiv\hat{c}_1.")
st.markdown(rf"Parameters: $\mu = \text{{scale}} \times \text{{doping}}$, $t = {hopping_t:.2f} \, \text{{eV}}$, $N={N}$, doping in $10^{{19}}\,\mathrm{{cm}}^{{-3}}$, material: {material}.")

st.markdown(r"### Tensor Network")
st.latex(r"H_i = \begin{pmatrix} \mu & 0 \\ 0 & \mu \end{pmatrix},\quad H_{i,i+1} = \begin{pmatrix} 0 & t \\ t & 0 \end{pmatrix}.")
st.latex(r"Z = \sum_{k_1,\ldots,k_N} \prod_{i=1}^N H_i(k_i^{\mathrm{in}},k_i^{\mathrm{out}})\, \prod_{i=1}^N H_{i,i+1}(k_i^{\mathrm{out}},k_{i+1}^{\mathrm{in}}) \approx N\mu.")

st.markdown(r"### Seebeck Coefficient")
st.latex(r"S \approx -\dfrac{\langle E\rangle}{T}\,\big(1 + \text{scale} \times \text{doping}\big),\quad \langle E\rangle \approx Z.")

st.markdown(r"### Figure of Merit")
st.latex(r"ZT = \dfrac{S^2\,\sigma\,T}{\kappa},\quad S\;\text{in}\;\mathrm{V\,K}^{-1},\; \sigma\;\text{in}\;\mathrm{S\,m}^{-1},\; \kappa\;\text{in}\;\mathrm{W\,m}^{-1}\mathrm{K}^{-1}.")

st.markdown(r"""
**Steps in the Simulation**
1. **Fetch Parameters**: Use Materials Project API key with mp-api to compute $t$ (from band structure bandwidth, conduction or valence bands) and $\mu$ (from DOS integration) or use literature defaults.
2. **Build Hamiltonian**: Create a 1D ring tensor network for the selected material.
3. **Contract Network**: Ring topology contracts to a scalar.
4. **Seebeck**: Compute $S$ from contracted energy.
5. **ZT**: Evaluate $ZT$ for multiple doping levels.
6. **Visualization**: Custom ring diagram or quimb visualization; customizable $ZT$ plot.
7. **Debugging**: Check computed bandwidth and mu_scale to verify Materials Project data usage.

**Parameters for Materials** (from Materials Project or literature):
- **Bi$_2$Te$_3$ (mp-34202)**: $t=0.3 \, \text{eV}$, $\mu=0.05 \times \text{doping}$, $\sigma=5000 \, \text{S/m}$, $\kappa=1.5 \, \text{W/m·K}$.
- **PbTe (mp-19717)**: $t=0.4 \, \text{eV}$, $\mu=0.05 \times \text{doping}$, $\sigma=10^4 \, \text{S/m}$, $\kappa=2.0 \, \text{W/m·K}$.
- **SnSe (mp-1190)**: $t=0.1 \, \text{eV}$, $\mu=0.03 \times \text{doping}$, $\sigma=1000 \, \text{S/m}$, $\kappa=0.7 \, \text{W/m·K}$.

**Extensions**: Add phonons, use DFT for accurate parameters, optimize for $ZT>2$.
""")
