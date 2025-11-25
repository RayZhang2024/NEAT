"""Core fitting utilities for NEAT."""

import numpy as np
from scipy import special

def fitting_function_1(x, a0, b0):
    return np.exp(-(a0 + b0 * x))

def fitting_function_2(x, a_hkl, b_hkl, a0, b0):
    return np.exp(-((a0) + (b0 * x))) * np.exp(-((a_hkl) + (b_hkl * x)))

def fitting_function_3(
    x, a0, b0, a_hkl, b_hkl,
    s, t, eta,
    hkl_list, 
    min_wavelength, max_wavelength,
    structure_type,        # <--- new
    lattice_params         # <--- new
):
    """
    Fitting function for Region 3, but now capable of using a more general
    formula for x_hkl based on the crystal structure.
    """
    if hkl_list:
        # Use a general function that handles any crystal system
        x_hkl_list = calculate_x_hkl_general(structure_type, lattice_params, hkl_list)
    else:
        # If no hkl_list is provided, fallback to using 'a' directly or skip
        x_hkl_list = [2 * lattice_params.get("a", 0)]  # Fallback for unknown phases

    # Initialize total modeled intensity to zero
    total_intensity = np.zeros_like(x, dtype=float)

    # Iterate over each x_hkl
    for x_hkl in x_hkl_list:
        # Only apply region3 logic if x_hkl is within [min_wavelength, max_wavelength]
        if np.isnan(x_hkl):
            continue
        if (x_hkl < min_wavelength) or (x_hkl > max_wavelength):
            continue


        try:         
            # -------------------------------------------------
            # helper: Lorentzian integral + exp-tail (τ = t > 0)
            # -------------------------------------------------
            def _l_int_exp_tail(x, x_hkl, s, t):
                """
                Integrated Lorentzian convolved with a 1-sided exponential tail.
                Good to <0.3 % for t ≥ s/3 (Thomsen, JAC 52 (2019) 456).
                """
                u      = (x - x_hkl) / (s * np.sqrt(2))
                L      = 0.5 + np.arctan(u) / np.pi                # plain CDF
                shift  = np.exp(-(x - x_hkl) / t)                  # tail weight
                u_s    = (x - x_hkl - s**2 / t) / (s * np.sqrt(2)) # τ–shifted arg
                L_s    = 0.5 + np.arctan(u_s) / np.pi              # shifted CDF
                return L - shift * (L - L_s)
            
            # -------------------------------------------------
            # inside your existing loop / try–block
            # -------------------------------------------------
            g_term = 0.5 * (
                    special.erfc(-(x - x_hkl) / (np.sqrt(2) * s)) -
                    np.exp(-(x - x_hkl) / t + s**2 / (2 * t**2))
                    * special.erfc(-(x - x_hkl)/(np.sqrt(2)*s) + s / t)
            )
            
            l_term = _l_int_exp_tail(x, x_hkl, s, t)        # ← tailed Lorentzian
            
            step   = (1.0 - eta) * g_term + eta * l_term    # pseudo-Voigt mix
            
            pre    = np.exp(-(a_hkl + b_hkl * x))
            edge   = np.exp(-(a0 + b0 * x)) * (pre + (1.0 - pre) * step)
            total_intensity += edge
            

        except Exception as e:
            print(f"Error calculating edge for x_hkl={x_hkl}: {e}")
            continue

    return total_intensity

def calculate_x_hkl_general(structure_type, lattice_params, hkl_list):
    """
    Returns a list of theoretical Bragg edges (lambda_hkl) for the given structure.
    lambda_hkl = 2 * d_hkl. 
    d_hkl depends on the structure and lattice parameters.
    
    Parameters:
        structure_type (str): e.g. "cubic", "tetragonal", "hexagonal", etc.
        lattice_params (dict): e.g. {"a": 3.266, "c": 4.8} for tetragonal, etc.
        hkl_list (list of tuples): (h,k,l) indices.
    """

    def d_spacing(h, k, l, structure, lat):
        if structure in ("cubic", "fcc", "bcc"):
            a = lat["a"]
            return a / np.sqrt(h**2 + k**2 + l**2)

        elif structure == "tetragonal":
            a, c = lat["a"], lat["c"]
            # d_{hkl} = 1 / sqrt( (h^2+k^2)/a^2 + l^2/c^2 )
            return 1.0 / np.sqrt((h**2 + k**2)/(a**2) + (l**2)/(c**2))

        elif structure == "hexagonal":
            a, c = lat["a"], lat["c"]
            # d_{hkl} = 1 / sqrt( (4/3)*((h^2 + h*k + k^2)/a^2) + l^2/c^2 )
            return 1.0 / np.sqrt((4.0/3.0)*((h**2 + h*k + k**2)/(a**2)) + (l**2)/(c**2))

        elif structure == "orthorhombic":
            a, b, c = lat["a"], lat["b"], lat["c"]
            return 1.0 / np.sqrt((h**2)/(a**2) + (k**2)/(b**2) + (l**2)/(c**2))

        # Add more structures if needed...

        # Fallback:
        return np.nan

    x_hkl_list = []
    for (h, k, l) in hkl_list:
        d_hkl = d_spacing(h, k, l, structure_type, lattice_params)
        if d_hkl > 0:
            x_hkl = 2.0 * d_hkl  # Bragg's law (assuming sin(θ)=1 at reflection edge)
            x_hkl_list.append(x_hkl)
        else:
            x_hkl_list.append(np.nan)

    return x_hkl_list

_FCC_HKL_DEFAULT = [
    (1, 1, 1),
    (2, 0, 0),
    (2, 2, 0),
    (3, 1, 1),
    (2, 2, 2),
    (4, 0, 0),
    (3, 3, 1),
    (4, 2, 0),
]

_BCC_HKL_DEFAULT = [
    (1, 1, 0),
    (2, 0, 0),
    (2, 1, 1),
    (2, 2, 0),
    (3, 1, 0),
    (2, 2, 2),
    (3, 2, 1),
    (3, 3, 0),
]

PHASE_DATA = {   
    "Unknown_Phase": {
        "structure": "unknown",  # or "custom"
        "lattice_params": {},    # No known lattice constants
        "hkl_list": []
    },
    "Cu_fcc": {
        "structure": "fcc",
        "lattice_params": {"a": 5.431},  # in Å
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Fe_bcc": {
        "structure": "bcc",
        "lattice_params": {"a": 2.86},   # in Å
        "hkl_list": list(_BCC_HKL_DEFAULT),
    },
    "Fe_fcc": {
        "structure": "fcc",
        "lattice_params": {"a": 3.61},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Al": {
        "structure": "fcc",
        "lattice_params": {"a": 4.05},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Ni_gamma": {
        "structure": "fcc",
        "lattice_params": {"a": 3.60},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "CeO2": {
        "structure": "fcc",
        "lattice_params": {"a": 5.41},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    
    "Ti_Beta": {
        "structure": "tetragonal",
        # For demonstration. Beta-Ti can actually be bcc, but assume tetragonal for example:
        "lattice_params": {"a": 3.266, "c": 4.80},  
        "hkl_list": [(1,1,0), (2,0,0), (2,1,1),
                     (2,2,0), (3,1,0), (2,2,2)]
    },
    "Ti_alpha_hex": {
        "structure": "hexagonal",
        "lattice_params": {"a": 2.95, "c": 4.68},
        "hkl_list": [(1,0,0), (0,0,2), (1,0,1), (1,1,0),
                     (2,0,0), (1,1,2), (2,1,1)]
    },
    
}

__all__ = [
    "fitting_function_1",
    "fitting_function_2",
    "fitting_function_3",
    "calculate_x_hkl_general",
    "PHASE_DATA",
]
