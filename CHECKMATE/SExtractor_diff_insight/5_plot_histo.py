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
def refined_symlog_bins(data, bins_per_side=8, linthresh=1e-5, center_bins=3):
    """
    Creates symmetric log bins on both sides and a few small linear bins near zero.
    """
    finite_data = data[np.isfinite(data)]
    max_val = 100
    
    # Log-spaced bins on each side
    log_edges_pos = np.logspace(np.log10(linthresh), np.log10(max_val), bins_per_side)
    log_edges_neg = -log_edges_pos[::-1]
    
    # Linearly spaced bins around zero (center bins)
    center_edges = np.linspace(-linthresh, linthresh, center_bins + 1)
    
    # Combine
    bins = np.concatenate([log_edges_neg, center_edges[1:-1], log_edges_pos])
    return bins

#linear_bin
def linear_symmetric_bins(data, num_bins=5000):
    finite_data = data[np.isfinite(data)]
    max_abs = np.max(np.abs(finite_data))
    bins = np.linspace(-1, 1, num_bins + 1)
    return bins


# === Helper: log-log histogram plot with central linear bin ===
def plot_loglog_hist(ax, data, bins, color, xlabel, xlim_min,xlim_max, linthresh=1e-5):
    data = data[np.isfinite(data)]
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=color, edgecolor='black', align='center')
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('log')
    ax.set_xlim(xlim_min,xlim_max) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Sources')
    ax.grid(True, which='both', ls='--', lw=0.5)

def plot_lin(ax, data, bins, color, xlabel, xlim_min,xlim_max, linthresh=1e-5):
    data = data[np.isfinite(data)]
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=color, edgecolor='black', align='center')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(xlim_min,xlim_max) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Sources')
    ax.grid(True, which='both', ls='--', lw=0.5)

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ΔMAG_AUTO & ΔMAG_APER Distributions (Linear + Log)", fontsize=16)

linthresh = 1e-5  # Linear region threshold

# Use correct data for each bin calculation!
bins_mag_auto  = linear_symmetric_bins(df['delta_mag_auto'],  num_bins=100)
bins_mag_aper  = linear_symmetric_bins(df['delta_mag_aper'],  num_bins=100)
bins_mag_auto_log = refined_symlog_bins(df['delta_mag_auto'], bins_per_side=8, linthresh=1e-5, center_bins=3)
bins_mag_aper_log = refined_symlog_bins(df['delta_mag_aper'], bins_per_side=8, linthresh=1e-5, center_bins=3)

# Plot
plot_lin(axs[0,0], df['delta_mag_auto'],  bins_mag_auto,  'cornflowerblue',  'ΔMAG_AUTO (linear scale)', -1,1,linthresh)
plot_lin(axs[0,1], df['delta_mag_aper'],  bins_mag_aper,  'cornflowerblue',  'ΔMAG_APER (linear scale)', -1,1,linthresh)
plot_loglog_hist(axs[1,0], df['delta_mag_auto'],  bins_mag_auto_log,  'cornflowerblue',  'ΔMAG_AUTO (log scale)', -200,200,linthresh)
plot_loglog_hist(axs[1,1], df['delta_mag_aper'],  bins_mag_aper_log,  'mediumseagreen',    'ΔMAG_APER (log scale)', -200,200,linthresh)



plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("5_delta_comparison_loglog_updated.png", dpi=300)
plt.show()
