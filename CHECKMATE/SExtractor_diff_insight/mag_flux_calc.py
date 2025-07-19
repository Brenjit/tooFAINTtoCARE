import numpy as np

def mag_to_flux(mag, zeropoint):
    """
    Convert magnitude to flux in arbitrary units (same as flux scale of ZP).
    
    Parameters:
    mag : float
        Magnitude value.
    zeropoint : float
        Zeropoint magnitude.
        
    Returns:
    flux : float
        Flux corresponding to the input magnitude.
    """
    flux = 10**((zeropoint - mag) / 2.5)
    return flux

# Example values
mag = 0.04
zeropoint = 28.08652

flux = mag_to_flux(mag, zeropoint)
print(f"Flux = {flux:.6e}")
