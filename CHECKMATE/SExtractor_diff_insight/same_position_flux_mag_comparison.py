import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Read the catalog
# --------------------------
cat = pd.read_csv("catalog_samepos_diff_flux.txt", sep='\t')

# --------------------------
# Flux comparison plot
# --------------------------
plt.figure(figsize=(8, 8))

plt.errorbar(
    cat["FLUX_AUTO_mine"], 
    cat["FLUX_AUTO_romeo"], 
    xerr=cat["FLUXERR_AUTO_mine"], 
    yerr=cat["FLUXERR_AUTO_romeo"], 
    fmt='o', color='blue', ecolor='gray', alpha=0.7, markersize=4, label='Sources'
)

max_flux = max(cat["FLUX_AUTO_mine"].max(), cat["FLUX_AUTO_romeo"].max())
plt.loglog([0, max_flux], [0, max_flux], 'k--', label='1:1 Line')
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel("FLUX_AUTO (mine) [counts]")
plt.ylabel("FLUX_AUTO (romeo) [counts]")
plt.title("Flux Comparison (with error bars)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("flux_comparison_plot.png", dpi=300)
plt.show()

# --------------------------
# Magnitude comparison plot
# --------------------------
plt.figure(figsize=(8, 8))

plt.errorbar(
    cat["MAG_AUTO_mine"], 
    cat["MAG_AUTO_romeo"], 
    xerr=cat["MAGERR_AUTO_mine"], 
    yerr=cat["MAGERR_AUTO_romeo"], 
    fmt='o', color='green', ecolor='gray', alpha=0.7, markersize=4, label='Sources'
)

min_mag = min(cat["MAG_AUTO_mine"].min(), cat["MAG_AUTO_romeo"].min())
max_mag = max(cat["MAG_AUTO_mine"].max(), cat["MAG_AUTO_romeo"].max())
plt.loglog([min_mag, max_mag], [min_mag, max_mag], 'k--', label='1:1 Line')
plt.xlabel("MAG_AUTO (mine) [mag]")
plt.ylabel("MAG_AUTO (romeo) [mag]")
plt.xlim(0,35)
plt.ylim(0,35)
plt.title("Magnitude Comparison (with error bars)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mag_comparison_plot.png", dpi=300)
plt.show()
