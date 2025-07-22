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
from matplotlib.patches import Circle 

# --------------------------
# CONFIGURATION
# --------------------------
image_path = "/Volumes/MY_SSD_1TB/My_work_june_24/Recheck/Images/hlsp_ceers_jwst_nircam_nircam6_f200w_dr0.5_i2d_SCI_BKSUB_c.fits"
catalog_path = "catalog_samepos_diff_flux.txt"
cutout_size_arcsec = 1
grid_rows = 2
grid_cols = 3
output_dir = "custom_cutouts_by_id"
os.makedirs(output_dir, exist_ok=True)

# 👇 Add your target IDs here (ID_mine values)
target_ids = [265,1171,2082, 8232,9362,11240]  # <-- MODIFY THIS LIST

# --------------------------
# Logging Setup
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "cutout_by_id_log.txt")),
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
# Filter by Target IDs
# --------------------------
selected = cat[cat["ID_mine"].isin(target_ids)].copy()
if selected.empty:
    logging.warning("⚠️ No matching IDs found in the catalog!")
    exit()

logging.info(f"🔎 Found {len(selected)} matching IDs in catalog.")

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
# Grid Maker for Selected IDs
# --------------------------
def make_custom_grid(df_chunk, image_data, output_path, rows=10, cols=6):
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axs = axs.flatten()
    
    aper_diam_pix = 10

    for i, row in enumerate(df_chunk.itertuples()):
        x = row.X_IMAGE_mine
        y = row.Y_IMAGE_mine
        id_v2 = int(row.ID_mine)
        id_v0 = int(row.ID_romeo)
        delta_flux = np.abs(row.FLUX_AUTO_mine - row.FLUX_AUTO_romeo)
        delta_mag = np.abs(row.MAG_AUTO_mine - row.MAG_AUTO_romeo)

        ax = axs[i]
        try:
            cutout = get_cutout(image_data, x, y, cutout_half_size)
            norm = simple_norm(cutout, 'sqrt', percent=99)
            ax.imshow(cutout, cmap='gray', origin='lower', norm=norm)

            # Add aperture circle
            center = (cutout_half_size, cutout_half_size)
            radius = aper_diam_pix / 2
            circ = Circle(center, radius, edgecolor='red', facecolor='none', linewidth=1.5)
            ax.add_patch(circ)

            ax.set_title(
                f"ID v.2:{id_v2} v.0:{id_v0}\nΔF:{delta_flux:.4f} ΔM:{delta_mag:.4f}",
                fontsize=8
            )
        except Exception as e:
            logging.warning(f"⚠️ Cutout error for ID {id_v2}: {e}")
            ax.text(0.5, 0.5, "Cutout Err", ha='center', va='center', fontsize=8)
        ax.axis('off')

    # Turn off unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    caption = ("Figure : Cutouts (1\"x1\") for selected galaxy candidates. "
           "Each panel shows the F200W band as the detection image, centered on the source. "
           "Apertures of 10-pixel diameter are shown, as used in SExtractor for photometry. "
           "Fluxes have been measured after background subtraction")
    # Add caption (adjust vertical position with `y`)
    # Reserve space at the bottom for the caption
    # Adjust layout to make room for caption
    plt.subplots_adjust(bottom=0.18)  # 👈 increase space below subplots

    # Add caption
    fig.text(0.5, 0.02, caption, wrap=True, ha='center', va='bottom', fontsize=9)

    # Save and close
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"🖼️ Saved cutout grid: {output_path}")
# --------------------------
# Create Cutouts for Selected IDs
# --------------------------
batch_size = grid_rows * grid_cols
num_batches = int(np.ceil(len(selected) / batch_size))
for i in range(num_batches):
    chunk = selected.iloc[i * batch_size: (i + 1) * batch_size]
    out_path = os.path.join(output_dir, f"cutout_grid_ids_{i+1:02}.png")
    make_custom_grid(chunk, image_data, out_path, rows=grid_rows, cols=grid_cols)

logging.info("✅ All requested cutouts saved successfully.")
