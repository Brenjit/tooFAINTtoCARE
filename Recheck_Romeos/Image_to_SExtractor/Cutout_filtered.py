import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from pathlib import Path

# ==== USER SETTINGS ====
zout_file = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/OUTPUT_Z20/nircam1/nircam1_output.zout"
sextractor_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/z_8_5_12/nircam1"
science_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/nircam1"
detection_image = os.path.join(science_dir, "nircam1_f150w_coadd12.fits")
output_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Cutouts/nircam1"
cutout_size = 50  # pixels

filters = [
    ("f606w", "egs_all_acs_wfc_f606w_030mas_v1.9_nircam1_mef_SCI_BKSUB.fits"),
    ("f814w", "egs_all_acs_wfc_f814w_030mas_v1.9_nircam1_mef_SCI_BKSUB.fits"),
    ("f115w", "hlsp_ceers_jwst_nircam_nircam1_f115w_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f150w", "hlsp_ceers_jwst_nircam_nircam1_f150w_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f200w", "hlsp_ceers_jwst_nircam_nircam1_f200w_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f277w", "hlsp_ceers_jwst_nircam_nircam1_f277w_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f356w", "hlsp_ceers_jwst_nircam_nircam1_f356w_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f410m", "hlsp_ceers_jwst_nircam_nircam1_f410m_dr0.5_i2d_SCI_BKSUB_c.fits"),
    ("f444w", "hlsp_ceers_jwst_nircam_nircam1_f444w_dr0.5_i2d_SCI_BKSUB_c.fits")
]

os.makedirs(output_dir, exist_ok=True)

# ==== STEP 1: Read EAZY output IDs ====
zout = ascii.read(zout_file)
source_ids = list(zout['id'])  # 'id' column from zout

# ==== STEP 2: Read SExtractor catalogs into dict {filter: Table} ====
sextractor_tables = {}
for filt, _ in filters:
    cat_path = os.path.join(sextractor_dir, f"selected_{filt}_catalog.cat")
    if os.path.exists(cat_path):
        sextractor_tables[filt] = ascii.read(cat_path)
    else:
        print(f"WARNING: Missing catalog {cat_path}")

# ==== STEP 3: Preload detection image ====
with fits.open(detection_image) as hdul:
    det_data = hdul[0].data

# ==== STEP 4: Loop over each source and make a single-row PNG ====
for src_id in source_ids:
    fig, axes = plt.subplots(1, len(filters) + 1, figsize=(3*(len(filters)+1), 3))
    
    # --- Get coordinates from reference filter (f115w) ---
    if 'f115w' not in sextractor_tables:
        print("No f115w catalog loaded, skipping source", src_id)
        continue
    
    ref_table = sextractor_tables['f115w']
    row = ref_table[ref_table['NUMBER'] == src_id]
    if len(row) == 0:
        print(f"Source {src_id} not found in f115w catalog")
        continue
    
    x = int(round(row['X_IMAGE'][0]))
    y = int(round(row['Y_IMAGE'][0]))
    half = cutout_size // 2

    # --- First column: detection image ---
    cut_det = det_data[y-half:y+half, x-half:x+half]
    axes[0].imshow(cut_det, origin='lower', cmap='gray',
                   vmin=np.percentile(cut_det, 5), vmax=np.percentile(cut_det, 99))
    axes[0].set_title(f"coadd\nID:{src_id}", fontsize=8)
    axes[0].axis('off')

    # --- Next columns: science images ---
    for i, (filt, fname) in enumerate(filters, start=1):
        img_path = os.path.join(science_dir, fname)
        if not os.path.exists(img_path):
            axes[i].axis('off')
            continue
        
        with fits.open(img_path) as hdul:
            data = hdul[0].data
        
        cut = data[y-half:y+half, x-half:x+half]
        axes[i].imshow(cut, origin='lower', cmap='gray',
                       vmin=np.percentile(cut, 5), vmax=np.percentile(cut, 99))
        axes[i].set_title(f"{filt}\nID:{src_id}", fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"source_{src_id}_cutouts.png"), dpi=200)
    plt.close()

print("Cutouts saved in:", output_dir)
