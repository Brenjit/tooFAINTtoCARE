import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import os

# === CONFIGURATION ===
image_path = "/Volumes/MY_SSD_1TB/My_work_june_24/Recheck/Images/hlsp_ceers_jwst_nircam_nircam6_f200w_dr0.5_i2d_SCI_BKSUB_c.fits"
catalog_path = "/Users/brenjithazarika/Desktop/PhD_computer/Codes/TooFAINTtooCARE/CHECKMATE/SExtractor_diff_insight/catalog_samepos_diff_flux.txt"
output_dir = "cutout_grids_f200w"
cutout_size_arcsec = 1.5   # square cutout of 1.5"
grid_n = 10                # 10x10 grid

os.makedirs(output_dir, exist_ok=True)

# === LOAD IMAGE & WCS ===
hdu = fits.open(image_path)[0]
image_data = hdu.data
wcs = WCS(hdu.header)
pixscale = np.abs(hdu.header['CDELT1']) * 3600  # degrees/pixel → arcsec/pixel
cutout_half_size = int((cutout_size_arcsec / pixscale) / 2)  # half size in pixels

# === LOAD CATALOG ===
df = pd.read_csv(catalog_path, delim_whitespace=True)

# === CUTOUT FUNCTION ===
def get_cutout(data, x, y, size):
    x, y = int(x), int(y)
    return data[y-size:y+size, x-size:x+size]

# === GRID MAKER ===
def make_grid(df_chunk, image_data, output_path, grid_n=10):
    fig, axs = plt.subplots(grid_n, grid_n, figsize=(20, 20))
    axs = axs.flatten()

    for i, row in enumerate(df_chunk.itertuples()):
        x = row.X_IMAGE_mine
        y = row.Y_IMAGE_mine
        id_mine = int(row.ID_mine)
        id_romeo = int(row.ID_romeo)

        ax = axs[i]
        try:
            cutout = get_cutout(image_data, x, y, cutout_half_size)
            norm = simple_norm(cutout, 'sqrt', percent=99)
            ax.imshow(cutout, cmap='gray', origin='lower', norm=norm)
            ax.set_title(f"Mine: {id_mine} | R: {id_romeo}", fontsize=10)
        except:
            ax.text(0.5, 0.5, "Cutout Err", ha='center', va='center', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# === GENERATE ALL GRIDS ===
batch_size = grid_n * grid_n
num_batches = int(np.ceil(len(df) / batch_size))

for i in range(num_batches):
    chunk = df.iloc[i*batch_size : (i+1)*batch_size]
    out_path = os.path.join(output_dir, f"cutout_grid_{i+1:02}.png")
    make_grid(chunk, image_data, out_path, grid_n=grid_n)

print(f"✅ Done! {num_batches} cutout grids saved in: {output_dir}")
