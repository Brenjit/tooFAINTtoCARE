import numpy as np
import matplotlib.pyplot as plt
from read_eazy_binaries import generate_sed_arrays

# Load EAZY binary outputs
MAIN_OUTPUT_FILE = 'nircam6_output'
OUTPUT_DIRECTORY = './OUTPUT'

# Read data
tempfilt, z_grid, obs_sed, templam, temp_sed = generate_sed_arrays(
    MAIN_OUTPUT_FILE=MAIN_OUTPUT_FILE,
    OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
    CACHE_FILE='Same'
)

# Define filter condition: Example: select sources with z > 7
z_min = 7.0
z_max = 15.0
selected_idx = np.where((z_grid >= z_min) & (z_grid <= z_max))[0]

print(f"Found {len(selected_idx)} sources with {z_min} ≤ z ≤ {z_max}")

# Preview the first 5
for i, idx in enumerate(selected_idx[:5]):
    plt.figure(figsize=(8, 5))
    plt.errorbar(tempfilt['lc'], tempfilt['fnu'][:, idx],
                 yerr=tempfilt['efnu'][:, idx],
                 fmt='o', color='black', label='Observed flux')
    plt.plot(tempfilt['lc'], obs_sed[:, idx], 'r--', label='Model photometry')
    plt.plot(templam * (1 + z_grid[idx]), temp_sed[:, idx], 'b-', label=f'Template SED (z = {z_grid[idx]:.2f})')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(3000, 40000)
    plt.ylim(bottom=1e-2)  # Adjust if needed
    plt.xlabel('Observed Wavelength (Å)')
    plt.ylabel('Flux Density (fν)')
    plt.title(f'Object {idx} | z = {z_grid[idx]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ask user if they want to save all
save_all = input(f"\nSave SED plots for all {len(selected_idx)} filtered sources? (y/n): ").strip().lower()
if save_all == 'y':
    outdir = 'SED_Plots'
    import os
    os.makedirs(outdir, exist_ok=True)

    for idx in selected_idx:
        plt.figure(figsize=(8, 5))
        plt.errorbar(tempfilt['lc'], tempfilt['fnu'][:, idx],
                     yerr=tempfilt['efnu'][:, idx],
                     fmt='o', color='black')
        plt.plot(tempfilt['lc'], obs_sed[:, idx], 'r--')
        plt.plot(templam * (1 + z_grid[idx]), temp_sed[:, idx], 'b-')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(3000, 40000)
        plt.ylim(bottom=1e-2)
        plt.title(f'Object {idx} | z = {z_grid[idx]:.2f}')
        plt.tight_layout()
        plt.savefig(f"{outdir}/sed_{idx}_z{z_grid[idx]:.2f}.png")
        plt.close()
    
    print(f"\n✅ Saved all plots in folder: {outdir}")
else:
    print("❌ Skipped saving plots.")

