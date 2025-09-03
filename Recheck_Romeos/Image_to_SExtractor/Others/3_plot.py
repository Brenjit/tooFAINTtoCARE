import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import ascii
import pandas as pd

# === Configuration ===
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
catalog_subdir = 'catalogue_z7'
eazy_catalog_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Eazy_catalogue/'
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

# Sources of interest
target_sources = {
    'nircam1': [8272],
    'nircam2': [11316],
    'nircam7': [3581],
    'nircam9': [8076]
}

# Filter mapping for EAZY catalog
filter_mapping = {
    'f606w': 'F606W', 'f814w': 'F814W', 'f115w': 'F115W', 'f150w': 'F150W',
    'f200w': 'F200W', 'f277w': 'F277W', 'f356w': 'F356W', 'f410m': 'F410M', 'f444w': 'F444W'
}

# === Updated scaling factors (shortened for this demo, fill full dict from earlier) ===
scaling_factors = {
    'nircam1': {'f606w': 0.69623, 'f814w': 0.33166, 'f115w': 0.71782, 'f150w': 0.51542,
                'f200w': 0.44576, 'f277w': 0.65191, 'f356w': 0.83187, 'f410m': 0.65723, 'f444w': 0.61749},
    'nircam2': {'f606w': 0.77297, 'f814w': 0.67423, 'f115w': 0.35926, 'f150w': 0.33716,
                'f200w': 0.35920, 'f277w': 0.47949, 'f356w': 0.53669, 'f410m': 0.73322, 'f444w': 0.73174},
    'nircam7': {'f606w': 0.50932, 'f814w': 0.48553, 'f115w': 0.43789, 'f150w': 0.00148,
                'f200w': 0.00108, 'f277w': 0.00112, 'f356w': 0.55363, 'f410m': 0.00101, 'f444w': 0.00083},
    'nircam9': {'f606w': 0.89682, 'f814w': 0.78386, 'f115w': 0.67045, 'f150w': 0.00143,
                'f200w': 0.00158, 'f277w': 0.00114, 'f356w': 0.57630, 'f410m': 0.00065, 'f444w': 0.00057}
}

# === Helpers ===
def read_sextractor_catalog(filepath, pointing, filter_name):
    """Read a single SExtractor catalog and compute SNRs."""
    data = ascii.read(filepath, comment='#', header_start=None, data_start=0,
                      names=['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                             'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                             'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ALPHA_J2000',
                             'DELTA_J2000'])
    with np.errstate(divide='ignore', invalid='ignore'):
        original_snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
    original_snr[np.isinf(original_snr) | np.isnan(original_snr)] = 0
    data['SNR_ORIGINAL'] = original_snr
    scale_factor = scaling_factors[pointing].get(filter_name, 1.0)
    data['SNR'] = original_snr * scale_factor
    return data

def read_eazy_catalog(filepath):
    """Read EAZY catalog (NMAD flux and errors)."""
    df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None)
    df.columns = ['id',
                  'f_F606W','e_F606W','f_F814W','e_F814W','f_F115W','e_F115W',
                  'f_F150W','e_F150W','f_F200W','e_F200W','f_F277W','e_F277W',
                  'f_F356W','e_F356W','f_F410M','e_F410M','f_F444W','e_F444W']
    return df

def calculate_nmad_snr(eazy_data):
    snr_data = {}
    for filt in filters:
        eazy_filt = filter_mapping[filt]
        flux_col, err_col = f'f_{eazy_filt}', f'e_{eazy_filt}'
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = eazy_data[flux_col] / eazy_data[err_col]
        snr[np.isinf(snr) | np.isnan(snr)] = 0
        snr_data[filt] = snr
    return snr_data

# === Main plotting ===
def main():
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("SNR Comparison for Failed Sources", fontsize=18)

    for row_idx, (pointing, ids) in enumerate(target_sources.items()):
        bid = ids[0]  # only one per pointing
        print(f"Processing {pointing} - {bid}")

        # Load SExtractor catalogs
        filter_data = {}
        for filt in filters:
            cat_path = os.path.join(base_dir, pointing, catalog_subdir, f"f150dropout_{filt}_catalog.cat")
            if os.path.exists(cat_path):
                filter_data[filt] = read_sextractor_catalog(cat_path, pointing, filt)

        # Load EAZY catalog
        eazy_file = os.path.join(eazy_catalog_dir, f"{pointing}_eazy_catalogue_54_gal.cat")
        eazy_data = read_eazy_catalog(eazy_file)
        nmad_snr = calculate_nmad_snr(eazy_data)

        # Collect values
        original_vals, scaled_vals, nmad_vals = [], [], []
        for filt in filters:
            val_o, val_s = 0, 0
            if filt in filter_data:
                match = filter_data[filt][filter_data[filt]['NUMBER'] == bid]
                if len(match) == 1:
                    val_o = match['SNR_ORIGINAL'][0]
                    val_s = match['SNR'][0]
            original_vals.append(val_o)
            scaled_vals.append(val_s)

            # NMAD
            val_n = 0
            match_eazy = eazy_data[eazy_data['id'] == bid]
            if len(match_eazy) > 0:
                val_n = nmad_snr[filt].iloc[match_eazy.index[0]]
            nmad_vals.append(val_n)

        # Plot row: Original, Scaled, NMAD
        x_pos = np.arange(len(filters))
        for col, (vals, title, color) in enumerate([
            (original_vals, f"{pointing}-{bid}\nOriginal SNR", "steelblue"),
            (scaled_vals,  f"{pointing}-{bid}\nScaled SNR", "orange"),
            (nmad_vals, f"{pointing}-{bid}\nNMAD SNR", "green")
        ]):
            ax = axes[row_idx, col]
            bars = ax.bar(x_pos, vals, color=color, alpha=0.7)

            # Add value labels above bars
            for bar, val in zip(bars, vals):
                if val != 0:  # only label non-zero values
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.1f}",
                        ha='center', va='bottom',
                        fontsize=9, rotation=90
                    )

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
            ax.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1)
            ax.set_ylabel("SNR")
            ax.set_title(title, fontsize=12)


    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("selected_sources_snr_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
