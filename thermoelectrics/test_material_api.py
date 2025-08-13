import streamlit as st
import numpy as np
from mp_api.client import MPRester
from scipy.integrate import trapezoid as trapz
from pymatgen.electronic_structure.dos import Spin
import pandas as pd

# Constants
k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

# Material-specific parameters (literature defaults)
material_params = {
    "Bi2Te3": {"mp_id": "mp-34202", "t": 0.3, "mu_scale": 0.05},
    "PbTe": {"mp_id": "mp-19717", "t": 0.4, "mu_scale": 0.05},
    "SnSe": {"mp_id": "mp-1190", "t": 0.1, "mu_scale": 0.03},
    "Si": {"mp_id": "mp-149", "t": 0.5, "mu_scale": 0.05},
    "Invalid": {"mp_id": "mp-999999", "t": 0.2, "mu_scale": 0.04}
}

def compute_t(bandstructure, material_name, carrier_type="n"):
    """Estimate hopping parameter t from band structure bandwidth."""
    try:
        if bandstructure is None or not hasattr(bandstructure, 'bands') or bandstructure.bands is None:
            raise ValueError("Band structure data is unavailable or invalid.")
        energies = bandstructure.bands[0]  # Spin-up
        if energies.size == 0:
            raise ValueError("Band structure energies are empty.")
        efermi = bandstructure.efermi
        st.write(f"{material_name}: Fermi level = {efermi:.3f} eV")
        st.write(f"{material_name}: Band energies shape = {energies.shape}")
        # Try conduction band first, fall back to valence band if empty
        mask = energies > efermi if carrier_type == "n" else energies < efermi
        if not np.any(mask):
            st.warning(f"{material_name}: No {'conduction' if carrier_type == 'n' else 'valence'} band energies available. Trying opposite band.")
            mask = energies < efermi if carrier_type == "n" else energies > efermi
            if not np.any(mask):
                raise ValueError("No valid band energies available for either conduction or valence bands.")
        relevant_energies = energies[mask]
        bandwidth = np.max(relevant_energies) - np.min(relevant_energies)
        if bandwidth <= 0:
            raise ValueError("Invalid bandwidth computed (zero or negative).")
        t_value = bandwidth / 4.0  # For 1D tight-binding, bandwidth ~ 4t
        st.write(f"{material_name}: Computed bandwidth = {bandwidth:.3f} eV, t = {t_value:.3f} eV")
        return t_value
    except Exception as e:
        st.error(f"{material_name}: Error computing t: {str(e)}")
        return None

def compute_mu(dos, material_name, doping_cm3, T, carrier_type="n"):
    """Compute chemical potential mu for given doping and temperature."""
    try:
        if dos is None or not hasattr(dos, 'energies') or not hasattr(dos, 'efermi'):
            raise ValueError("DOS data is unavailable or invalid.")
        energies = np.array(dos.energies) - dos.efermi  # Energies relative to Fermi level
        # Handle spin-polarized or non-spin-polarized DOS
        if Spin.up in dos.densities:
            density = np.array(dos.densities[Spin.up])
            if Spin.down in dos.densities:
                density += np.array(dos.densities[Spin.down])
        else:
            raise ValueError("DOS density data is missing or incorrectly formatted.")
        if len(energies) != len(density):
            raise ValueError("Mismatch between energies and DOS density arrays.")
        st.write(f"{material_name}: DOS energy range = [{energies.min():.3f}, {energies.max():.3f}] eV")
        st.write(f"{material_name}: DOS density shape = {density.shape}")
        # Clip energies to avoid overflow in Fermi-Dirac
        energy_clip = 10.0 * k_B * T  # Limit to ~10 kT (~0.258 eV at 300 K)
        mask_clip = np.abs(energies) < energy_clip
        energies = energies[mask_clip]
        density = density[mask_clip]
        if len(energies) < 2:
            raise ValueError("Insufficient energy points after clipping.")
        fermi_dirac = lambda E, mu: 1 / (1 + np.exp(np.clip((E - mu) / (k_B * T), -100, 100)))
        volume = dos.structure.volume * 1e-24  # Angstrom^3 to cm^3
        doping_per_cell = doping_cm3 * volume / 1e19  # Scale to match doping
        def carrier_concentration(mu):
            mask = energies > 0 if carrier_type == "n" else energies < 0
            if not np.any(mask):
                raise ValueError(f"No {'positive' if carrier_type == 'n' else 'negative'} energies for {carrier_type}-type carriers.")
            integrand = density[mask] * fermi_dirac(energies[mask], mu)
            integral = trapz(integrand, energies[mask])
            return integral - doping_per_cell
        mu_min, mu_max = -0.5, 0.5  # Tighter bounds to improve convergence
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
        mu_scale = (mu_min + mu_max) / 2
        st.write(f"{material_name}: Computed mu_scale for doping {doping_cm3/1e19:.2f} x 10^19 cm^-3: {mu_scale:.3f}")
        return mu_scale
    except Exception as e:
        st.error(f"{material_name}: Error computing mu: {str(e)}")
        return None

# Streamlit App Header
st.title("Materials Project API Parameter Tester")
st.markdown(
    """
    This app tests the computation of hopping parameter \( t \) (from band structure) and chemical potential scale \( \mu \)-scale (from DOS) 
    for thermoelectric materials (Bi₂Te₃, PbTe, SnSe), Si, and an invalid material ID. Enter a Materials Project API key to fetch data. 
    Results show data availability, computed values, and defaults.
    """
)

# Sidebar: Input Parameters
st.sidebar.header("Input Parameters")
api_key = st.sidebar.text_input("Materials Project API Key", type="password", help="Get your API key from https://materialsproject.org")
temperature = st.sidebar.slider("Temperature (K)", 100.0, 1000.0, 300.0, step=10.0)
doping_cm3 = st.sidebar.number_input("Doping Concentration (x 10^19 cm^-3)", min_value=0.01, max_value=10.0, value=1.0, step=0.1) * 1e19
carrier_type = st.sidebar.selectbox("Carrier Type", ["n", "p"], index=0)
run_button = st.sidebar.button("Run Tests")

# Main Interface
if run_button and api_key:
    st.header("Test Results")
    results = []
    with MPRester(api_key) as mpr:
        for material, params in material_params.items():
            mp_id = params["mp_id"]
            default_t = params["t"]
            default_mu_scale = params["mu_scale"]
            st.subheader(f"Processing {material} (mp_id: {mp_id})")
            
            # Fetch band structure
            try:
                bandstructure = mpr.get_bandstructure_by_material_id(mp_id)
                st.write(f"{material}: Band structure available = {bool(bandstructure)}")
            except Exception as e:
                st.error(f"{material}: Error fetching band structure: {str(e)}")
                bandstructure = None
            
            # Fetch DOS
            try:
                dos = mpr.get_dos_by_material_id(mp_id)
                st.write(f"{material}: DOS available = {bool(dos)}")
            except Exception as e:
                st.error(f"{material}: Error fetching DOS: {str(e)}")
                dos = None
            
            # Compute t and mu
            t = compute_t(bandstructure, material, carrier_type=carrier_type)
            mu_scale = compute_mu(dos, material, doping_cm3, temperature, carrier_type=carrier_type)
            
            # Store results
            t_result = f"{t:.3f}" if t is not None else f"Default: {default_t:.3f}"
            mu_result = f"{mu_scale:.3f}" if mu_scale is not None else f"Default: {default_mu_scale:.3f}"
            results.append({
                "Material": material,
                "MP ID": mp_id,
                "Band Structure Available": bool(bandstructure),
                "DOS Available": bool(dos),
                "Computed t (eV)": t_result,
                "Computed mu_scale": mu_result
            })
            
            # Display results for this material
            if t is not None:
                st.write(f"{material}: Computed t = {t:.3f} eV (Default: {default_t:.3f} eV)")
            else:
                st.write(f"{material}: Using default t = {default_t:.3f} eV due to computation failure")
            if mu_scale is not None:
                st.write(f"{material}: Computed mu_scale = {mu_scale:.3f} (Default: {default_mu_scale:.3f})")
            else:
                st.write(f"{material}: Using default mu_scale = {default_mu_scale:.3f} due to computation failure")
    
    # Display results table
    if results:
        st.header("Summary of Results")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
else:
    st.info("Enter a valid API key and click 'Run Tests' to start.")
