import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import warnings
from astropy.wcs import FITSFixedWarning

# --------------------------
# CONFIGURATION
# --------------------------
image_path = "/Volumes/MY_SSD_1TB/My_work_june_24/Recheck/Images/hlsp_ceers_jwst_nircam_nircam6_f200w_dr0.5_i2d_SCI_BKSUB_c.fits"
catalog_path = "catalog_samepos_diff_flux.txt"
cutout_size_arcsec = 1.5
grid_rows = 10
grid_cols = 6
output_dir = "4_Clipping"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Logging Setup (NO TIMESTAMP)
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "process_log.txt")),
        logging.StreamHandler()
    ]
)

warnings.simplefilter('ignore', FITSFixedWarning)

# --------------------------
# Load Catalog
# --------------------------
logging.info("📂 Loading catalog...")
try:
    cat = pd.read_csv(catalog_path, sep='\t')
    logging.info(f"✅ Catalog loaded successfully: {len(cat)} sources")
except Exception as e:
    logging.error("❌ Failed to read the catalog.")
    raise e

# --------------------------
# Calculate ΔFlux and ΔMag
# --------------------------
logging.info("🔍 Calculating |ΔFlux| and |ΔMag|...")
cat["DELTA_FLUX"] = np.abs(cat["FLUX_AUTO_mine"] - cat["FLUX_AUTO_romeo"])
cat["DELTA_MAG"] = np.abs(cat["MAG_AUTO_mine"] - cat["MAG_AUTO_romeo"])

# --------------------------
# Load FITS Image
# --------------------------
hdu = fits.open(image_path)[0]
image_data = hdu.data
wcs = WCS(hdu.header)
pixscale = np.abs(hdu.header['CDELT1']) * 3600
cutout_half_size = int((cutout_size_arcsec / pixscale) / 2)

# --------------------------
# Cutout Function
# --------------------------
def get_cutout(data, x, y, size):
    x, y = int(x), int(y)
    return data[y-size:y+size, x-size:x+size]

# --------------------------
# Grid Maker Function
# --------------------------
def make_grid(df_chunk, image_data, output_path, rows=10, cols=6):
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()

    for i, row in enumerate(df_chunk.itertuples()):
        x = row.X_IMAGE_mine
        y = row.Y_IMAGE_mine
        id_mine = int(row.ID_mine)
        id_romeo = int(row.ID_romeo)
        delta_flux = row.DELTA_FLUX
        delta_mag = row.DELTA_MAG

        ax = axs[i]
        try:
            cutout = get_cutout(image_data, x, y, cutout_half_size)
            norm = simple_norm(cutout, 'sqrt', percent=99)
            ax.imshow(cutout, cmap='gray', origin='lower', norm=norm)
            ax.set_title(
                f"Mine:{id_mine} R:{id_romeo}\nΔF:{delta_flux:.2f} ΔM:{delta_mag:.2f}",
                fontsize=8
            )
        except Exception as e:
            logging.warning(f"⚠️ Cutout error for ID {id_mine}: {e}")
            ax.text(0.5, 0.5, "Cutout Err", ha='center', va='center', fontsize=8)
        ax.axis('off')

    # Turn off unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# --------------------------
# Processing Modes
# --------------------------
thresholds = [0.1, 0.5, 1.0]
batch_size = grid_rows * grid_cols

clip_modes = {
    "flux": lambda df, t: df[df["DELTA_FLUX"] > t],
    "mag": lambda df, t: df[df["DELTA_MAG"] > t],
    "both": lambda df, t: df[(df["DELTA_FLUX"] > t) & (df["DELTA_MAG"] > t)],
}

for mode, condition in clip_modes.items():
    for thresh in thresholds:
        filtered = condition(cat, thresh).copy()
        tag = f"{mode}_gt_{thresh:.1f}"
        txt_fname = os.path.join(output_dir, f"{tag}.txt")
        filtered.to_csv(txt_fname, sep='\t', index=False)

        logging.info(f"💾 {mode.upper()} > {thresh:.1f}: {len(filtered)} sources → saved to {txt_fname}")

        # Make cutout grids
        cutout_dir = os.path.join(output_dir, f"cutouts_{tag}")
        os.makedirs(cutout_dir, exist_ok=True)

        num_batches = int(np.ceil(len(filtered) / batch_size))
        for i in range(num_batches):
            chunk = filtered.iloc[i * batch_size: (i + 1) * batch_size]
            out_path = os.path.join(cutout_dir, f"cutout_grid_{i+1:02}.png")
            make_grid(chunk, image_data, out_path, rows=grid_rows, cols=grid_cols)

        logging.info(f"🖼️ {num_batches} cutout grid(s) saved in {cutout_dir}")

# --------------------------
# ΔFlux Plot (Log Scale)
# --------------------------
epsilon = 1e-5
cat["DELTA_FLUX"] = cat["DELTA_FLUX"].replace(0, epsilon)
x_idx = np.arange(1, len(cat) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_idx, cat["DELTA_FLUX"], 'o', markersize=4, alpha=0.7, label='|ΔFlux|')
for thresh in thresholds:
    plt.axhline(thresh, linestyle='--', linewidth=1.2, label=f'ΔFlux = {thresh}')
plt.yscale('log')
plt.xlabel("Source Index")
plt.ylabel("|ΔFlux| [counts] (log scale)")
plt.title("Absolute Flux Difference vs Source Index")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "delta_flux_vs_index_log.png"), dpi=300)
plt.close()
logging.info("📊 ΔFlux plot saved.")

# --------------------------
# ΔMag Plot (Log Scale)
# --------------------------
cat["DELTA_MAG"] = cat["DELTA_MAG"].replace(0, epsilon)

plt.figure(figsize=(10, 6))
plt.plot(x_idx, cat["DELTA_MAG"], 'o', markersize=4, alpha=0.7, label='|ΔMag|', color='orange')
for thresh in thresholds:
    plt.axhline(thresh, linestyle='--', linewidth=1.2, label=f'ΔMag = {thresh}')
plt.yscale('log')
plt.xlabel("Source Index")
plt.ylabel("|ΔMag| (log scale)")
plt.title("Absolute Mag Difference vs Source Index")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "delta_mag_vs_index_log.png"), dpi=300)
plt.close()
logging.info("📊 ΔMag plot saved.")

# --------------------------
# Final Message
# --------------------------
logging.info("🎉 All processing completed successfully.")
