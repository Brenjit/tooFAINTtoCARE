import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from matplotlib.patches import Circle
import os
import warnings
from astropy.wcs import FITSFixedWarning

# Ignore the specific FITS warning about non-standard keywords
warnings.simplefilter('ignore', FITSFixedWarning)

def load_and_merge_catalogs(eazy_path, sextractor_dir, filters_to_load):
    """
    Loads the EAZY redshift catalog and merges it with SExtractor catalogs 
    to create a master table.

    Args:
        eazy_path (str): Path to the EAZY .zout file.
        sextractor_dir (str): Directory containing the SExtractor .cat files.
        filters_to_load (list): A list of filter names (e.g., 'f115w') to load.

    Returns:
        pandas.DataFrame: A master DataFrame with merged information.
    """
    print("--- Step 1: Loading and Merging Catalogs ---")
    try:
        eazy_df = pd.read_csv(eazy_path, sep='\s+', comment='#', header=None,
                              names=['id', 'z_spec', 'z_a', 'z_m1', 'chi_a', 'l68', 'u68',
                                     'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z',
                                     'z_peak', 'peak_prob', 'z_mc'])
        print(f"✅ Loaded {len(eazy_df)} sources from EAZY catalog.")
    except FileNotFoundError:
        print(f"❌ ERROR: EAZY catalog not found at {eazy_path}")
        return None

    master_filter = 'f150w'
    print(f"ℹ️  Loading primary coordinate catalog: {master_filter.upper()}")
    try:
        master_cat_path = os.path.join(sextractor_dir, f"f150dropout_{master_filter}_catalog.cat")
        master_df = pd.read_csv(master_cat_path, sep='\s+', comment='#', header=None,
                                names=['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                                       'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                                       'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                                       'ALPHA_J2000', 'DELTA_J2000'])
        master_df = master_df[['NUMBER', 'X_IMAGE', 'Y_IMAGE']]
        print(f"✅ Loaded {len(master_df)} sources from {master_filter.upper()} catalog.")
    except FileNotFoundError:
        print(f"❌ ERROR: Primary SExtractor catalog for {master_filter.upper()} not found.")
        return None

    merged_df = pd.merge(eazy_df, master_df, left_on='id', right_on='NUMBER')

    for filt in filters_to_load:
        print(f"ℹ️  Merging photometry for filter: {filt.upper()}")
        try:
            cat_path = os.path.join(sextractor_dir, f"f150dropout_{filt}_catalog.cat")
            phot_df = pd.read_csv(cat_path, sep='\s+', comment='#', header=None,
                                  usecols=[0, 5], names=['NUMBER', f'MAG_APER_{filt.upper()}'])
            merged_df = pd.merge(merged_df, phot_df, on='NUMBER', how='left')
        except FileNotFoundError:
            print(f"  - ⚠️ Warning: Catalog for {filt.upper()} not found. Magnitudes will be NaN.")
            merged_df[f'MAG_APER_{filt.upper()}'] = np.nan

    print("\n✅ Catalog merging complete.")
    return merged_df


def create_cutout_sheets(data_df, image_paths, output_dir, batch_size=10, cutout_size_arcsec=1.0, aper_diam_arcsec=0.32):
    """
    Generates and saves sheets of cutout images for sources in the DataFrame.

    Args:
        data_df (pandas.DataFrame): DataFrame containing the sources to plot.
        image_paths (dict): Dictionary mapping filter names to FITS file paths.
        output_dir (str): Directory to save the output images.
        batch_size (int): Number of sources per output image file.
        cutout_size_arcsec (float): The size of the cutout in arcseconds.
        aper_diam_arcsec (float): The diameter of the aperture circle in arcseconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")

    try:
        with fits.open(image_paths['F150W']) as hdul:
            pixel_scale = abs(hdul[0].header['CD1_1']) * 3600
        print(f"ℹ️  Determined pixel scale: {pixel_scale:.4f} arcsec/pixel")
    except Exception as e:
        print(f"⚠️ Warning: Could not determine pixel scale. Assuming 0.03 arcsec/pixel. Error: {e}")
        pixel_scale = 0.03

    cutout_size_pix = int(cutout_size_arcsec / pixel_scale)
    aper_radius_pix = (aper_diam_arcsec / pixel_scale) / 2.0
    print(f"ℹ️  Cutout size: {cutout_size_pix}x{cutout_size_pix} pixels. Aperture radius: {aper_radius_pix:.2f} pixels.")

    num_sources = len(data_df)
    filters = list(image_paths.keys())
    num_filters = len(filters)

    fits_data = {}
    print("\n--- Step 3: Loading FITS Image Data ---")
    for filt, path in image_paths.items():
        try:
            fits_data[filt] = fits.getdata(path, ext=0)
            print(f"  - ✅ Loaded {filt}")
        except FileNotFoundError:
            print(f"  - ❌ ERROR: Image file not found for {filt} at {path}. It will be skipped.")
            fits_data[filt] = None
    
    for i in range(0, num_sources, batch_size):
        batch_df = data_df.iloc[i:i+batch_size]
        num_in_batch = len(batch_df)
        
        fig, axes = plt.subplots(num_in_batch, num_filters, 
                                 figsize=(num_filters * 2, num_in_batch * 2.2),
                                 squeeze=False)
        fig.patch.set_facecolor('white')

        print(f"\n--- Creating Batch {i//batch_size + 1} ({num_in_batch} sources) ---")

        for row_idx, (source_id, source_data) in enumerate(batch_df.iterrows()):
            x_cen, y_cen = int(source_data['X_IMAGE']), int(source_data['Y_IMAGE'])
            
            axes[row_idx, 0].text(-0.5, 0.5, f"ID: {int(source_data['NUMBER'])}\nz_a = {source_data['z_a']:.2f}",
                                  transform=axes[row_idx, 0].transAxes,
                                  ha='right', va='center', fontsize=10, weight='bold')

            for col_idx, filt in enumerate(filters):
                ax = axes[row_idx, col_idx]
                ax.set_xticks([])
                ax.set_yticks([])

                if fits_data.get(filt) is None:
                    ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{filt}", fontsize=9)
                    continue

                half_size = cutout_size_pix // 2
                cutout = fits_data[filt][y_cen - half_size : y_cen + half_size,
                                         x_cen - half_size : x_cen + half_size]

                if cutout.size == 0:
                    ax.text(0.5, 0.5, 'Out of\nBounds', ha='center', va='center', transform=ax.transAxes)
                else:
                    norm = simple_norm(cutout, 'sqrt', percent=99.)
                    ax.imshow(cutout, origin='lower', cmap='gray_r', norm=norm)
                    
                    # Add aperture circle, inspired by your reference script
                    center = (half_size, half_size)
                    circ = Circle(center, aper_radius_pix, edgecolor='red', facecolor='none', linewidth=1)
                    ax.add_patch(circ)

                mag = source_data.get(f'MAG_APER_{filt.upper()}', float('nan'))
                ax.set_title(f"{filt}\nm={mag:.2f}", fontsize=9)

        plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
        
        output_filename = os.path.join(output_dir, f"cutout_batch_{i//batch_size + 1}.png")
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved batch plot: {output_filename}")


if __name__ == '__main__':
    # --- Define paths and parameters ---
    POINTING = 'nircam1'
    DR_VERSION = 'dr0.5' # Adjust for other pointings if necessary
    
    eazy_catalog_path = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Eazy_catalogue/{POINTING}_eazy_catalogue.cat"
    sextractor_base_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{POINTING}/catalogue_z7"
    image_base_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/{POINTING}"
    output_base_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/High_z_Cutouts"

    image_file_map = {
        'F606W': f"egs_all_acs_wfc_f606w_030mas_v1.9_{POINTING}_mef_SCI_BKSUB.fits",
        'F814W': f"egs_all_acs_wfc_f814w_030mas_v1.9_{POINTING}_mef_SCI_BKSUB.fits",
        'F115W': f"hlsp_ceers_jwst_nircam_{POINTING}_f115w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F150W': f"hlsp_ceers_jwst_nircam_{POINTING}_f150w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F200W': f"hlsp_ceers_jwst_nircam_{POINTING}_f200w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F277W': f"hlsp_ceers_jwst_nircam_{POINTING}_f277w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F356W': f"hlsp_ceers_jwst_nircam_{POINTING}_f356w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F410M': f"hlsp_ceers_jwst_nircam_{POINTING}_f410m_{DR_VERSION}_i2d_SCI_BKSUB_c.fits",
        'F444W': f"hlsp_ceers_jwst_nircam_{POINTING}_f444w_{DR_VERSION}_i2d_SCI_BKSUB_c.fits"
    }
    
    full_image_paths = {filt: os.path.join(image_base_dir, fname) for filt, fname in image_file_map.items()}
    
    master_catalog = load_and_merge_catalogs(eazy_catalog_path, sextractor_base_dir, list(image_file_map.keys()))
    
    if master_catalog is not None:
        print("\n--- Step 2: Filtering for High-Redshift Candidates ---")
        high_z_candidates = master_catalog[master_catalog['z_a'] > 8.5].copy()
        print(f"✅ Found {len(high_z_candidates)} sources with z_a > 8.5.")
        
        if not high_z_candidates.empty:
            create_cutout_sheets(high_z_candidates, full_image_paths, output_base_dir)
            print("\n🎉 All processing completed successfully.")
        else:
            print("\nℹ️ No high-redshift candidates to process.")

