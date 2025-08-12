import numpy as np
from astropy.io import ascii
import os
from astropy.table import Table

def read_sextractor_catalog(filepath):
    """
    Reads a SExtractor catalog file, skipping the commented header.
    
    Args:
        filepath (str): The path to the SExtractor .cat file.
        
    Returns:
        astropy.table.Table: An Astropy Table object containing the catalog data, or None if reading fails.
    """
    try:
        data = ascii.read(
            filepath,
            comment='#',
            header_start=None,
            data_start=0,
            names=[
                'NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER', 'ALPHA_J2000',
                'DELTA_J2000'
            ]
        )
        # Calculate Signal-to-Noise Ratio (S/N)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        snr[np.isinf(snr) | np.isnan(snr)] = 0  # Replace inf/nan with 0
        data['SNR'] = snr
        return data
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def get_mags(data_dict, filter_list):
    """Helper function to extract magnitudes, replacing non-detections."""
    mags = {}
    for filt in filter_list:
        mags[filt] = np.nan_to_num(data_dict[filt]['MAG_AUTO'], nan=99.0, posinf=99.0, neginf=99.0)
    return mags

def apply_f814w_selection(data_dict):
    """Applies F814W-dropout (z=5.5-8.5) selection criteria and returns a boolean mask."""
    print("Applying F814W (z=5.5-8.5) selection...")
    sn_detect = (data_dict['f115w']['SNR'] > 5) & (data_dict['f150w']['SNR'] > 5) & \
                (data_dict['f277w']['SNR'] > 5) & (data_dict['f444w']['SNR'] > 5)
    mags = get_mags(data_dict, ['f814w', 'f115w', 'f150w'])
    color1 = mags['f814w'] - mags['f115w']
    color2 = mags['f115w'] - mags['f150w']
    cut1 = (data_dict['f814w']['SNR'] < 2) | (color1 > 0.75)
    cut2 = (color2 > -1.0) & (color2 < 0.5)
    cut3 = color1 > (0.5 + color2 + 1.0)
    star_galaxy_cut = data_dict['f115w']['CLASS_STAR'] < 0.8
    mask = sn_detect & cut1 & cut2 & cut3 & star_galaxy_cut
    print(f"Found {np.sum(mask)} candidates.")
    return mask

def apply_f115w_selection(data_dict):
    """Applies F115W-dropout (z=8.5-12) selection criteria and returns a boolean mask."""
    print("Applying F115W (z=8.5-12) selection...")
    sn_detect = (data_dict['f150w']['SNR'] > 5) & (data_dict['f277w']['SNR'] > 5) & (data_dict['f444w']['SNR'] > 5)
    veto = data_dict['f814w']['SNR'] < 2
    mags = get_mags(data_dict, ['f115w', 'f150w', 'f277w'])
    color1 = mags['f115w'] - mags['f150w']
    color2 = mags['f150w'] - mags['f277w']
    cut1 = (data_dict['f115w']['SNR'] < 2) | (color1 > 0.75)
    cut2 = (color2 > -1.5) & (color2 < 1.0)
    cut3 = color1 > (0.5 + 0.44 * (color2 + 0.8) + 0.5)
    star_galaxy_cut = data_dict['f150w']['CLASS_STAR'] < 0.8
    mask = sn_detect & veto & cut1 & cut2 & cut3 & star_galaxy_cut
    print(f"Found {np.sum(mask)} candidates.")
    return mask
    
def apply_f150w_selection(data_dict):
    """Applies F150W-dropout (z=12-15) selection criteria and returns a boolean mask."""
    print("Applying F150W (z=12-15) selection...")
    sn_detect = (data_dict['f277w']['SNR'] > 5) & (data_dict['f444w']['SNR'] > 5)
    veto = (data_dict['f814w']['SNR'] < 2) & (data_dict['f115w']['SNR'] < 2)
    mags = get_mags(data_dict, ['f150w', 'f277w', 'f444w'])
    color1 = mags['f150w'] - mags['f277w']
    color2 = mags['f277w'] - mags['f444w']
    cut1 = (data_dict['f150w']['SNR'] < 2) | (color1 > 0.75)
    cut2 = (color2 > -1.0) & (color2 < 0.5)
    cut3 = color1 > (color2 + 1.0)
    star_galaxy_cut = data_dict['f277w']['CLASS_STAR'] < 0.8
    mask = sn_detect & veto & cut1 & cut2 & cut3 & star_galaxy_cut
    print(f"Found {np.sum(mask)} candidates.")
    return mask

def main():
    """Main function to run the filtering process and save results in a structured format."""
    # --- User-defined parameters ---
    base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
    output_base_dir = './selection_results'
    pointings = [f'nircam{i}' for i in range(1, 11)]
    catalog_subdir = 'catalogue_z7'
    
    # Updated list of all available filters
    required_filters = [
        'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 
        'f410m', 'f444w', 'f606w', 'f814w'
    ]
    
    selections = {
        "z_5.5-8.5": apply_f814w_selection,
        "z_8.5-12": apply_f115w_selection,
        "z_12-15": apply_f150w_selection
    }

    # Loop over each pointing directory
    for pointing in pointings:
        print(f"\n--- Processing Pointing: {pointing} ---")
        
        catalog_dir = os.path.join(base_dir, pointing, catalog_subdir)
        if not os.path.isdir(catalog_dir):
            print(f"Warning: Directory not found, skipping: {catalog_dir}")
            continue
            
        filter_data = {}
        all_filters_found = True
        for filt in required_filters:
            cat_filename = f"f150dropout_{filt}_catalog.cat"
            cat_path = os.path.join(catalog_dir, cat_filename)
            if os.path.exists(cat_path):
                data = read_sextractor_catalog(cat_path)
                if data is not None:
                    filter_data[filt] = data
                else:
                    all_filters_found = False; break
            else:
                print(f"Warning: Catalog file not found: {cat_path}")
                all_filters_found = False; break
        
        if not all_filters_found:
            print(f"Skipping pointing {pointing} due to missing files.")
            continue

        if len(set(len(tbl) for tbl in filter_data.values())) > 1:
            print(f"Warning: Catalogs for {pointing} have different lengths. Results may be unreliable. Skipping.")
            continue

        # Apply each selection criterion to the current pointing
        for z_range, selection_func in selections.items():
            
            # Get the boolean mask for the candidates
            candidates_mask = selection_func(filter_data)
            
            # If any candidates are found for this pointing and z-range...
            if np.any(candidates_mask):
                # Create the specific output directory for this result
                output_dir = os.path.join(output_base_dir, z_range, pointing)
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"Saving {np.sum(candidates_mask)} candidates for {pointing} in {z_range} to {output_dir}")
                
                # Loop through the original catalogs and save the filtered version of each
                for filt_name, original_table in filter_data.items():
                    
                    # Apply the mask to the original table
                    selected_sources_table = original_table[candidates_mask]
                    
                    # Define the new output filename
                    output_filename = os.path.join(output_dir, f"selected_{filt_name}_catalog.cat")
                    
                    # Save the filtered table in the same format as the input
                    selected_sources_table.write(output_filename, format='ascii.commented_header', delimiter=' ', overwrite=True)
            else:
                print(f"No candidates found for {pointing} in {z_range}.")

    print("\n--- Processing Complete ---")
    print(f"Output saved in the '{output_base_dir}' directory, structured by redshift and pointing.")

if __name__ == '__main__':
    main()
