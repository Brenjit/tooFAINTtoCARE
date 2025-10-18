import numpy as np
import os
from astropy.io import ascii
import pandas as pd

# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
eazy_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/outputs_z20'
eazy_flux_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Results_22nd_Sept/Eazy_catalogue_of_sept_data'
output_dir = './2_high_z_catalogues_for_nmad'
pointings = [f'nircam{i}' for i in range(1, 11)]

filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
filter_mapping = {
    'f606w': 'F606W', 'f814w': 'F814W', 'f115w': 'F115W', 'f150w': 'F150W',
    'f200w': 'F200W', 'f277w': 'F277W', 'f356w': 'F356W', 'f410m': 'F410M', 'f444w': 'F444W'
}

# Redshift bins for high-z sources
redshift_bins = [7, 12]

def read_eazy_zout(filepath):
    """Reads EAZY zout file and returns DataFrame with redshift information."""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = line.split()
                if len(values) >= 5:
                    row = {
                        'id': int(values[0]),
                        'z_spec': float(values[1]),
                        'z_a': float(values[2]),
                        'z_m1': float(values[3]),
                        'chi_a': float(values[4])
                    }
                    data.append(row)
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading EAZY zout file {filepath}: {e}")
        return None

def bin_redshifts_and_select(eazy_data, bins, n_selected=20):
    """Bin sources by redshift and randomly select n_selected from each bin."""
    binned_sources = {}
    
    # Create bins - for high-z we focus on z > 7
    high_z_data = eazy_data[eazy_data['z_a'] > 7]
    
    if len(high_z_data) == 0:
        print("No high-z sources found (z > 7)")
        return binned_sources
    
    # Bin the high-z sources
    binned_indices = np.digitize(high_z_data['z_a'], bins)
    
    for bin_idx in range(1, len(bins)):
        bin_mask = (binned_indices == bin_idx)
        bin_sources = high_z_data[bin_mask]
        
        if len(bin_sources) > 0:
            # Randomly select sources
            if len(bin_sources) <= n_selected:
                selected_sources = bin_sources
            else:
                selected_sources = bin_sources.sample(n=n_selected, random_state=42)
            
            binned_sources[f'z{int(bins[bin_idx-1])}-{int(bins[bin_idx])}'] = selected_sources
    
    return binned_sources

def read_sextractor_catalog(filepath):
    """Reads a SExtractor catalog file."""
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
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def read_eazy_flux_catalog(filepath):
    """Reads the EAZY flux catalog with flux measurements."""
    try:
        # Read the catalog
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract data lines (skip comments)
        data_lines = []
        for line in lines:
            if not line.startswith('#'):
                data_lines.append(line.strip())
        
        # Parse the data - adjust this based on your actual EAZY flux file format
        data = []
        for line in data_lines:
            values = line.split()
            if len(values) == 19:  # id + 9 filters (flux + error each)
                row = {
                    'id': int(values[0]),
                    'f_F606W': float(values[1]), 'e_F606W': float(values[2]),
                    'f_F814W': float(values[3]), 'e_F814W': float(values[4]),
                    'f_F115W': float(values[5]), 'e_F115W': float(values[6]),
                    'f_F150W': float(values[7]), 'e_F150W': float(values[8]),
                    'f_F200W': float(values[9]), 'e_F200W': float(values[10]),
                    'f_F277W': float(values[11]), 'e_F277W': float(values[12]),
                    'f_F356W': float(values[13]), 'e_F356W': float(values[14]),
                    'f_F410M': float(values[15]), 'e_F410M': float(values[16]),
                    'f_F444W': float(values[17]), 'e_F444W': float(values[18])
                }
                data.append(row)
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading EAZY flux catalog {filepath}: {e}")
        return None

def create_sextractor_like_catalog(sextractor_data, eazy_data, pointing, redshift_bin, output_dir):
    """Create a SExtractor-like catalog with flux measurements for NMAD calculation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the source IDs from EAZY data
    source_ids = eazy_data['id'].tolist()
    
    print(f"Creating catalog for {len(source_ids)} high-z sources for {pointing} in bin {redshift_bin}")
    
    # Initialize the output data
    output_data = []
    
    for src_id in source_ids:
        # Find the source in SExtractor data (use f150w as reference for position info)
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            print(f"⚠️  Source {src_id} not found in SExtractor data for {pointing}")
            continue
        
        sex_idx = sex_idx[0]
        
        # Find the source in EAZY data
        eazy_idx = np.where(eazy_data['id'] == src_id)[0]
        if len(eazy_idx) == 0:
            print(f"Source {src_id} not found in EAZY data for {pointing}")
            continue
        
        eazy_idx = eazy_idx[0]
        
        # Get basic information from SExtractor
        x_image = sextractor_data['f150w']['X_IMAGE'][sex_idx]
        y_image = sextractor_data['f150w']['Y_IMAGE'][sex_idx]
        alpha_j2000 = sextractor_data['f150w']['ALPHA_J2000'][sex_idx]
        delta_j2000 = sextractor_data['f150w']['DELTA_J2000'][sex_idx]
        class_star = sextractor_data['f150w']['CLASS_STAR'][sex_idx]
        
        # Get redshift information from EAZY zout (if available)
        z_a = eazy_data['z_a'].iloc[eazy_idx] if 'z_a' in eazy_data.columns else -1
        
        # Create a row with all filter information
        row = {
            'NUMBER': src_id,
            'X_IMAGE': x_image,
            'Y_IMAGE': y_image,
            'ALPHA_J2000': alpha_j2000,
            'DELTA_J2000': delta_j2000,
            'CLASS_STAR': class_star,
            'REDSHIFT': z_a,
            'REDSHIFT_BIN': redshift_bin,
            'POINTING': pointing
        }
        
        # Add SExtractor flux measurements for each filter
        for filt in filters:
            sex_flux = sextractor_data[filt]['FLUX_AUTO'][sex_idx]
            sex_fluxerr = sextractor_data[filt]['FLUXERR_AUTO'][sex_idx]
            sex_mag = sextractor_data[filt]['MAG_AUTO'][sex_idx]
            sex_magerr = sextractor_data[filt]['MAGERR_AUTO'][sex_idx]
            
            row.update({
                f'SEX_FLUX_{filt.upper()}': sex_flux,
                f'SEX_FLUXERR_{filt.upper()}': sex_fluxerr,
                f'SEX_MAG_{filt.upper()}': sex_mag,
                f'SEX_MAGERR_{filt.upper()}': sex_magerr
            })
        
        # Add EAZY flux measurements for each filter
        for filt in filters:
            eazy_filt = filter_mapping[filt]
            eazy_flux = eazy_data[f'f_{eazy_filt}'].iloc[eazy_idx]
            eazy_fluxerr = eazy_data[f'e_{eazy_filt}'].iloc[eazy_idx]
            
            row.update({
                f'EAZY_FLUX_{filt.upper()}': eazy_flux,
                f'EAZY_FLUXERR_{filt.upper()}': eazy_fluxerr
            })
        
        output_data.append(row)
    
    # Convert to DataFrame and save
    if output_data:
        df = pd.DataFrame(output_data)
        
        # Save detailed catalog
        output_file = os.path.join(output_dir, f'{pointing}_{redshift_bin}_high_z_catalog.txt')
        df.to_csv(output_file, index=False, sep="\t")

        
        # Also save a simplified version with just essential columns for NMAD calculation
        simplified_cols = ['NUMBER', 'REDSHIFT', 'REDSHIFT_BIN', 'POINTING']
        for filt in filters:
            simplified_cols.extend([
                f'SEX_FLUX_{filt.upper()}',
                f'SEX_FLUXERR_{filt.upper()}',
                f'EAZY_FLUX_{filt.upper()}',
                f'EAZY_FLUXERR_{filt.upper()}'
            ])
        
        simplified_df = df[simplified_cols]
        simplified_file = os.path.join(output_dir, f'{pointing}_{redshift_bin}_high_z_simplified.txt')
        simplified_df.to_csv(simplified_file, index=False, sep="\t")

        
        print(f"Saved catalog data for {len(df)} sources")
        return df
    else:
        print(f"No valid data found for {pointing} in bin {redshift_bin}")
        return pd.DataFrame()

def create_combined_filter_catalog(sextractor_data, eazy_data, pointing, redshift_bin, output_dir):
    """Create a catalog organized by filter for easier NMAD calculation per filter."""
    os.makedirs(output_dir, exist_ok=True)
    
    source_ids = eazy_data['id'].tolist()
    
    print(f"Creating filter-wise catalog for {pointing} in bin {redshift_bin}")
    
    # Create separate DataFrames for each filter
    filter_data = {filt: [] for filt in filters}
    
    for src_id in source_ids:
        # Find the source in SExtractor data
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            continue
        
        sex_idx = sex_idx[0]
        
        # Find the source in EAZY data
        eazy_idx = np.where(eazy_data['id'] == src_id)[0]
        if len(eazy_idx) == 0:
            continue
        
        eazy_idx = eazy_idx[0]
        
        # Get basic information
        x_image = sextractor_data['f150w']['X_IMAGE'][sex_idx]
        y_image = sextractor_data['f150w']['Y_IMAGE'][sex_idx]
        alpha_j2000 = sextractor_data['f150w']['ALPHA_J2000'][sex_idx]
        delta_j2000 = sextractor_data['f150w']['DELTA_J2000'][sex_idx]
        z_a = eazy_data['z_a'].iloc[eazy_idx] if 'z_a' in eazy_data.columns else -1
        
        # Add data for each filter
        for filt in filters:
            sex_flux = sextractor_data[filt]['FLUX_AUTO'][sex_idx]
            sex_fluxerr = sextractor_data[filt]['FLUXERR_AUTO'][sex_idx]
            eazy_filt = filter_mapping[filt]
            eazy_flux = eazy_data[f'f_{eazy_filt}'].iloc[eazy_idx]
            eazy_fluxerr = eazy_data[f'e_{eazy_filt}'].iloc[eazy_idx]
            
            row = {
                'NUMBER': src_id,
                'X_IMAGE': x_image,
                'Y_IMAGE': y_image,
                'ALPHA_J2000': alpha_j2000,
                'DELTA_J2000': delta_j2000,
                'REDSHIFT': z_a,
                'REDSHIFT_BIN': redshift_bin,
                'POINTING': pointing,
                'FILTER': filt.upper(),
                'SEX_FLUX': sex_flux,
                'SEX_FLUXERR': sex_fluxerr,
                'EAZY_FLUX': eazy_flux,
                'EAZY_FLUXERR': eazy_fluxerr
            }
            filter_data[filt].append(row)
    
    # Save filter-wise catalogs
    for filt in filters:
        if filter_data[filt]:
            df = pd.DataFrame(filter_data[filt])
            output_file = os.path.join(output_dir, f'{pointing}_{redshift_bin}_{filt}_catalog.txt')
            df.to_csv(output_file, index=False, sep="\t")
            print(f"Saved {filt} catalog with {len(df)} entries")
    
    return filter_data

def main():
    """Main function to create catalogs for high-z sources for NMAD calculation."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating high-z catalogs for NMAD calculation")
    print(f"Redshift bins: {redshift_bins}")
    print(f"Output directory: {output_dir}")
    
    all_catalogs = {}
    
    for pointing in pointings:
        print(f"\n{'='*50}")
        print(f"Processing {pointing}")
        print(f"{'='*50}")
        
        # Read EAZY zout file
        eazy_zout_file = os.path.join(eazy_base_dir, pointing, f"{pointing}_output.zout")
        if not os.path.exists(eazy_zout_file):
            print(f"EAZY zout file not found: {eazy_zout_file}")
            continue
        
        eazy_zdata = read_eazy_zout(eazy_zout_file)
        if eazy_zdata is None or len(eazy_zdata) == 0:
            print(f"Could not read EAZY zout data for {pointing}")
            continue
        
        # Bin high-z sources and select
        binned_sources = bin_redshifts_and_select(eazy_zdata, redshift_bins, n_selected=50)
        
        if not binned_sources:
            print(f"No high-z sources found for {pointing}")
            continue
        
        # Read SExtractor data
        sextractor_data = {}
        catalog_dir = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7')
        
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path)
            if data is None:
                print(f"Missing {filt} data for {pointing}")
                missing_data = True
                break
            sextractor_data[filt] = data
        
        if missing_data:
            continue
        
        # Process each redshift bin
        for redshift_bin, selected_sources in binned_sources.items():
            print(f"\nProcessing redshift bin {redshift_bin} with {len(selected_sources)} sources")
            
            # Read EAZY flux data for these sources
            # Adjust this path based on your EAZY flux file structure
            eazy_flux_file = os.path.join(eazy_flux_dir, f"{pointing}_fluxerr_catalogue.cat")  # Adjust as needed
            if not os.path.exists(eazy_flux_file):
                print(f"EAZY flux file not found: {eazy_flux_file}")
                continue
            
            eazy_flux_data = read_eazy_flux_catalog(eazy_flux_file)
            if eazy_flux_data is None:
                print(f"Could not read EAZY flux data for {pointing}")
                continue
            
            # Filter flux data to only include selected high-z sources
            selected_ids = selected_sources['id'].tolist()
            eazy_flux_selected = eazy_flux_data[eazy_flux_data['id'].isin(selected_ids)]
            
            if len(eazy_flux_selected) == 0:
                print(f"No matching flux data found for selected high-z sources in {redshift_bin}")
                continue
            
            # Create the main catalog
            catalog_df = create_sextractor_like_catalog(
                sextractor_data, eazy_flux_selected, pointing, redshift_bin, output_dir
            )
            
            # Create filter-wise catalogs
            filter_catalogs = create_combined_filter_catalog(
                sextractor_data, eazy_flux_selected, pointing, redshift_bin, 
                os.path.join(output_dir, 'filter_wise')
            )
            
            if len(catalog_df) > 0:
                key = f"{pointing}_{redshift_bin}"
                all_catalogs[key] = catalog_df
    
    # Create a master combined catalog
    if all_catalogs:
        all_data = pd.concat(all_catalogs.values(), ignore_index=True)
        all_data.to_csv(os.path.join(output_dir, 'all_high_z_sources_master_catalog.txt'),
                index=False, sep="\t")
        
        # Summary file
        summary = all_data.groupby(['POINTING', 'REDSHIFT_BIN']).agg({
            'NUMBER': 'count',
            'REDSHIFT': ['min', 'max', 'mean']
        }).round(3)
        
        summary.to_csv(os.path.join(output_dir, 'catalog_summary.txt'), sep="\t")
        
        print(f"\nCatalog creation complete!")
        print(f"Total high-z sources processed: {all_data['NUMBER'].nunique()}")
        print(f"Total entries: {len(all_data)}")
        print(f"Catalogs saved to: {output_dir}")
        
        # Print available files
        print(f"\nAvailable catalog files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.csv'):
                    print(f"  {os.path.join(root, file)}")
    else:
        print("No high-z catalogs were created")

if __name__ == '__main__':
    main()