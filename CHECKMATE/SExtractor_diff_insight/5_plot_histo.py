import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

# === Configure logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Load the catalog ===
filename = 'catalog_same_positions.txt'

try:
    df = pd.read_csv(filename, delim_whitespace=True)
    logging.info(f"✅ Successfully loaded '{filename}' with {len(df)} entries.")
except Exception as e:
    logging.error(f"❌ Error reading the file: {e}")
    raise

# === Compute differences ===
df['delta_mag_auto'] = df['MAG_AUTO_mine'] - df['MAG_AUTO_romeo']
df['delta_mag_aper'] = df['MAG_APER_mine'] - df['MAG_APER_romeo']
df['delta_flux_auto'] = df['FLUX_AUTO_mine'] - df['FLUX_AUTO_romeo']
df['delta_flux_aper'] = df['FLUX_APER_mine'] - df['FLUX_APER_romeo']

# === Log basic stats ===
logging.info(f"🔹 Mean ΔMAG_AUTO:  {df['delta_mag_auto'].mean():.4f}")
logging.info(f"🔹 Std  ΔMAG_AUTO:  {df['delta_mag_auto'].std():.4f}")
logging.info(f"🔹 Mean ΔFLUX_AUTO: {df['delta_flux_auto'].mean():.2f}")
logging.info(f"🔹 Std  ΔFLUX_AUTO: {df['delta_flux_auto'].std():.2f}")

# === Helper: symmetric symlog-style bins including zeros ===
def symmetric_symlog_bins(data, bins=100, linthresh=1e-5):
    finite_data = data[np.isfinite(data)]
    max_val = np.max(np.abs(finite_data))
    log_max = np.log10(max_val)
    
    log_bins = np.linspace(np.log10(linthresh), log_max, bins // 2)
    positive_bins = np.power(10, log_bins)
    negative_bins = -positive_bins[::-1]
    
    return np.concatenate([negative_bins, [-linthresh, linthresh], positive_bins])

# === Helper: log-log histogram plot with central linear bin ===
def plot_loglog_hist(ax, data, bins, color, title, xlabel, linthresh=1e-5):
    data = data[np.isfinite(data)]
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=color, edgecolor='black', align='center')
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Sources (log scale)')
    ax.grid(True, which='both', ls='--', lw=0.5)

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Differences Between Version 2.28.2 and Version 2.28.0 (Log-Log Histograms)", fontsize=16)

linthresh = 1e-5  # Linear region threshold

# ΔMAG_AUTO
bins_mag_auto = symmetric_symlog_bins(df['delta_mag_auto'], bins=100, linthresh=linthresh)
plot_loglog_hist(axs[0, 0], df['delta_mag_auto'], bins_mag_auto, 'cornflowerblue',
                 'ΔMAG_AUTO', 'MAG_AUTO (v2.28.2 - v2.28.0)', linthresh)

# ΔMAG_APER
bins_mag_aper = symmetric_symlog_bins(df['delta_mag_aper'], bins=100, linthresh=linthresh)
plot_loglog_hist(axs[0, 1], df['delta_mag_aper'], bins_mag_aper, 'mediumseagreen',
                 'ΔMAG_APER', 'MAG_APER (v2.28.2 - v2.28.0)', linthresh)

# ΔFLUX_AUTO
bins_flux_auto = symmetric_symlog_bins(df['delta_flux_auto'], bins=100, linthresh=linthresh)
plot_loglog_hist(axs[1, 0], df['delta_flux_auto'], bins_flux_auto, 'salmon',
                 'ΔFLUX_AUTO', 'FLUX_AUTO (v2.28.2 - v2.28.0)', linthresh)

# ΔFLUX_APER
bins_flux_aper = symmetric_symlog_bins(df['delta_flux_aper'], bins=100, linthresh=linthresh)
plot_loglog_hist(axs[1, 1], df['delta_flux_aper'], bins_flux_aper, 'goldenrod',
                 'ΔFLUX_APER', 'FLUX_APER (v2.28.2 - v2.28.0)', linthresh)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("delta_comparison_loglog.png", dpi=300)
plt.show()
