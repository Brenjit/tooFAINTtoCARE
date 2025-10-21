import numpy as np
import os
from astropy.io import ascii
import pandas as pd
import random


# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
eazy_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/outputs_z20'
output_dir = './50_2_random_sources_catalogues'  # Same output directory as previous code
pointings = [f'nircam{i}' for i in range(1, 11)]

filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

# Target redshift range
target_redshift_min = 7
target_redshift_max = 11

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

def create_zout_catalog(eazy_data, selected_sources, pointing, output_dir):
    """Create a text file with EAZY zout values for selected sources."""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_ids = selected_sources['id'].tolist()
    
    print(f"Creating zout catalog for {len(selected_ids)} high-z sources for {pointing}")
    
    # Filter EAZY data to only include selected sources
    zout_data = eazy_data[eazy_data['id'].isin(selected_ids)]
    
    if len(zout_data) == 0:
        print(f"No zout data found for selected sources in {pointing}")
        return None
    
    # Create output file
    output_file = os.path.join(output_dir, f'{pointing}_zout_values.cat')
    
    # Write header
    with open(output_file, 'w') as f:
        f.write('# EAZY zout values for high-z sources (z=7-11)\n')
        f.write('# Created from EAZY zout files\n')
        f.write('# Format: ID z_spec z_a z_m1 chi_a\n')
        f.write('# ID: Source identification number\n')
        f.write('# z_spec: Spectroscopic redshift (if available)\n')
        f.write('# z_a: Photometric redshift (peak of probability distribution)\n')
        f.write('# z_m1: First redshift peak\n')
        f.write('# chi_a: Reduced chi-squared value\n')
        f.write('#\n')
        f.write('# ID\tz_spec\tz_a\tz_m1\tchi_a\n')
    
    # Write data
    with open(output_file, 'a') as f:
        for _, row in zout_data.iterrows():
            f.write(f"{int(row['id'])}\t{row['z_spec']:.6f}\t{row['z_a']:.6f}\t{row['z_m1']:.6f}\t{row['chi_a']:.6f}\n")
    
    print(f"✓ Saved zout catalog with {len(zout_data)} sources to {output_file}")
    return zout_data

def select_sources_in_redshift_range(eazy_data, z_min=7, z_max=11, n_sources=50):
    """Select sources within specified redshift range."""
    # Filter sources in redshift range
    redshift_filtered = eazy_data[(eazy_data['z_a'] >= z_min) & (eazy_data['z_a'] <= z_max)]
    
    if len(redshift_filtered) == 0:
        print(f"No sources found in redshift range {z_min}-{z_max}")
        return pd.DataFrame()
    
    # Randomly select sources
    if len(redshift_filtered) <= n_sources:
        print(f"Only {len(redshift_filtered)} sources available in z={z_min}-{z_max}, selecting all")
        selected_sources = redshift_filtered
    else:
        selected_sources = redshift_filtered.sample(n=n_sources, random_state=42)
    
    print(f"Selected {len(selected_sources)} sources with z between {z_min} and {z_max}")
    return selected_sources

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

def create_sextractor_like_catalog(sextractor_data, selected_sources, pointing, output_dir):
    """Create a SExtractor-like catalog for selected sources - SAME FORMAT AS PREVIOUS CODE."""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_ids = selected_sources['id'].tolist()
    
    print(f"Creating catalog for {len(selected_ids)} high-z sources for {pointing}")
    
    # Initialize the output data
    output_data = []
    
    for src_id in selected_ids:
        # Find the source in SExtractor data (use f150w as reference for position info)
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            print(f"⚠️  Source {src_id} not found in SExtractor data for {pointing}")
            continue
        
        sex_idx = sex_idx[0]
        
        # Get redshift information
        eazy_idx = np.where(selected_sources['id'] == src_id)[0]
        if len(eazy_idx) == 0:
            continue
        eazy_idx = eazy_idx[0]
        z_a = selected_sources['z_a'].iloc[eazy_idx]
        
        # Get basic information from SExtractor
        x_image = sextractor_data['f150w']['X_IMAGE'][sex_idx]
        y_image = sextractor_data['f150w']['Y_IMAGE'][sex_idx]
        alpha_j2000 = sextractor_data['f150w']['ALPHA_J2000'][sex_idx]
        delta_j2000 = sextractor_data['f150w']['DELTA_J2000'][sex_idx]
        class_star = sextractor_data['f150w']['CLASS_STAR'][sex_idx]
        
        # Create a row with all filter information - SAME FORMAT AS PREVIOUS CODE
        row = {
            'NUMBER': src_id,
            'X_IMAGE': x_image,
            'Y_IMAGE': y_image,
            'ALPHA_J2000': alpha_j2000,
            'DELTA_J2000': delta_j2000,
            'CLASS_STAR': class_star,
            'REDSHIFT': z_a,
            'POINTING': pointing
        }
        
        # Add SExtractor measurements for each filter - SAME FORMAT AS PREVIOUS CODE
        for filt in filters:
            sex_data = sextractor_data[filt]
            filt_idx = np.where(sex_data['NUMBER'] == src_id)[0]
            
            if len(filt_idx) > 0:
                filt_idx = filt_idx[0]
                sex_flux = sex_data['FLUX_AUTO'][filt_idx]
                sex_fluxerr = sex_data['FLUXERR_AUTO'][filt_idx]
                sex_mag = sex_data['MAG_AUTO'][filt_idx]
                sex_magerr = sex_data['MAGERR_AUTO'][filt_idx]
                sex_flux_aper = sex_data['FLUX_APER'][filt_idx]
                sex_fluxerr_aper = sex_data['FLUXERR_APER'][filt_idx]
                sex_mag_aper = sex_data['MAG_APER'][filt_idx]
                sex_magerr_aper = sex_data['MAGERR_APER'][filt_idx]
            else:
                # If source not found in this filter, fill with NaN
                sex_flux = sex_fluxerr = sex_mag = sex_magerr = np.nan
                sex_flux_aper = sex_fluxerr_aper = sex_mag_aper = sex_magerr_aper = np.nan
            
            row.update({
                f'FLUX_AUTO_{filt.upper()}': sex_flux,
                f'FLUXERR_AUTO_{filt.upper()}': sex_fluxerr,
                f'MAG_AUTO_{filt.upper()}': sex_mag,
                f'MAGERR_AUTO_{filt.upper()}': sex_magerr,
                f'FLUX_APER_{filt.upper()}': sex_flux_aper,
                f'FLUXERR_APER_{filt.upper()}': sex_fluxerr_aper,
                f'MAG_APER_{filt.upper()}': sex_mag_aper,
                f'MAGERR_APER_{filt.upper()}': sex_magerr_aper
            })
        
        output_data.append(row)
    
    # Convert to DataFrame and save - SAME FORMAT AS PREVIOUS CODE
    if output_data:
        df = pd.DataFrame(output_data)
        
        # Save the catalog - SAME NAMING CONVENTION AS PREVIOUS CODE
        output_file = os.path.join(output_dir, f'{pointing}_50_random_sources.cat')
        
        # Write header - SAME FORMAT AS PREVIOUS CODE
        with open(output_file, 'w') as f:
            f.write('# SExtractor-like catalog for 50 high-z sources (z=7-11)\n')
            f.write('# Created from original SExtractor catalogs\n')
            f.write('# Format: ID, coordinates, CLASS_STAR, REDSHIFT, and photometry for all filters\n')
        
        # Save data - SAME FORMAT AS PREVIOUS CODE
        df.to_csv(output_file, index=False, sep='\t', mode='a')
        
        print(f"✓ Saved catalog with {len(df)} sources to {output_file}")
        return df
    else:
        print(f"✗ No valid data found for {pointing}")
        return pd.DataFrame()

def create_individual_filter_catalogs(sextractor_data, selected_sources, pointing, output_dir):
    """Create individual filter catalogs in original SExtractor format - SAME AS PREVIOUS CODE."""
    filter_output_dir = os.path.join(output_dir, 'individual_filters', pointing)
    os.makedirs(filter_output_dir, exist_ok=True)
    
    selected_ids = selected_sources['id'].tolist()
    
    for filt in filters:
        print(f"Creating {filt} catalog for {pointing}")
        
        output_data = []
        for src_id in selected_ids:
            sex_data = sextractor_data[filt]
            idx = np.where(sex_data['NUMBER'] == src_id)[0]
            
            if len(idx) > 0:
                idx = idx[0]
                row = {
                    'NUMBER': sex_data['NUMBER'][idx],
                    'X_IMAGE': sex_data['X_IMAGE'][idx],
                    'Y_IMAGE': sex_data['Y_IMAGE'][idx],
                    'MAG_AUTO': sex_data['MAG_AUTO'][idx],
                    'MAGERR_AUTO': sex_data['MAGERR_AUTO'][idx],
                    'MAG_APER': sex_data['MAG_APER'][idx],
                    'MAGERR_APER': sex_data['MAGERR_APER'][idx],
                    'CLASS_STAR': sex_data['CLASS_STAR'][idx],
                    'FLUX_AUTO': sex_data['FLUX_AUTO'][idx],
                    'FLUXERR_AUTO': sex_data['FLUXERR_AUTO'][idx],
                    'FLUX_APER': sex_data['FLUX_APER'][idx],
                    'FLUXERR_APER': sex_data['FLUXERR_APER'][idx],
                    'ALPHA_J2000': sex_data['ALPHA_J2000'][idx],
                    'DELTA_J2000': sex_data['DELTA_J2000'][idx]
                }
                output_data.append(row)
        
        if output_data:
            df = pd.DataFrame(output_data)
            output_file = os.path.join(filter_output_dir, f'random_50_{filt}_catalog.cat')
            
            # Write in SExtractor-like format - SAME AS PREVIOUS CODE
            with open(output_file, 'w') as f:
                f.write(f'# SExtractor catalog for 50 high-z sources (z=7-11) - Filter: {filt}\n')
                f.write('# NUMBER X_IMAGE Y_IMAGE MAG_AUTO MAGERR_AUTO MAG_APER MAGERR_APER CLASS_STAR FLUX_AUTO FLUXERR_AUTO FLUX_APER FLUXERR_APER ALPHA_J2000 DELTA_J2000\n')
            
            df.to_csv(output_file, index=False, sep='\t', mode='a', header=False)
            print(f"✓ Saved {filt} catalog with {len(df)} sources")

def main():
    """Main function to create catalogs for 50 high-z sources (z=7-11) per pointing."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating catalogs for 50 high-z sources (z=7-11) per pointing")
    print(f"Output directory: {output_dir}")
    
    all_catalogs = {}
    all_zout_catalogs = {}
    
    for pointing in pointings:
        print(f"\n{'='*50}")
        print(f"Processing {pointing}")
        print(f"{'='*50}")
        
        # Read EAZY zout file to get redshift information
        eazy_zout_file = os.path.join(eazy_base_dir, pointing, f"{pointing}_output.zout")
        if not os.path.exists(eazy_zout_file):
            print(f"EAZY zout file not found: {eazy_zout_file}")
            continue
        
        eazy_zdata = read_eazy_zout(eazy_zout_file)
        if eazy_zdata is None or len(eazy_zdata) == 0:
            print(f"Could not read EAZY zout data for {pointing}")
            continue
        
        # Select sources in redshift range 7-11
        selected_sources = select_sources_in_redshift_range(
            eazy_zdata, 
            z_min=target_redshift_min, 
            z_max=target_redshift_max, 
            n_sources=50
        )
        
        if len(selected_sources) == 0:
            print(f"No sources found in redshift range {target_redshift_min}-{target_redshift_max} for {pointing}")
            continue
        
        # Create zout catalog for this pointing
        zout_catalog = create_zout_catalog(eazy_zdata, selected_sources, pointing, output_dir)
        if zout_catalog is not None:
            all_zout_catalogs[pointing] = zout_catalog
        
        # Read SExtractor data
        sextractor_data = {}
        catalog_dir = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7')
        
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            if not os.path.exists(cat_path):
                print(f"✗ Catalog not found: {cat_path}")
                missing_data = True
                break
            
            data = read_sextractor_catalog(cat_path)
            if data is None:
                print(f"✗ Could not read {filt} data for {pointing}")
                missing_data = True
                break
            sextractor_data[filt] = data
        
        if missing_data:
            continue
        
        print(f"Selected {len(selected_sources)} high-z sources (z=7-11)")
        
        # Create combined catalog - SAME STRUCTURE AS PREVIOUS CODE
        catalog_df = create_sextractor_like_catalog(
            sextractor_data, selected_sources, pointing, output_dir
        )
        
        # Create individual filter catalogs - SAME STRUCTURE AS PREVIOUS CODE
        create_individual_filter_catalogs(
            sextractor_data, selected_sources, pointing, output_dir
        )
        
        if len(catalog_df) > 0:
            all_catalogs[pointing] = catalog_df
    
    # Create a master combined catalog - SAME AS PREVIOUS CODE
    if all_catalogs:
        all_data = pd.concat(all_catalogs.values(), ignore_index=True)
        master_file = os.path.join(output_dir, 'all_pointings_50_sources_master_catalog.cat')
        all_data.to_csv(master_file, index=False, sep='\t')
        
        # Create master zout catalog
        if all_zout_catalogs:
            all_zout_data = pd.concat(all_zout_catalogs.values(), ignore_index=True)
            master_zout_file = os.path.join(output_dir, 'all_pointings_zout_master_catalog.cat')
            
            with open(master_zout_file, 'w') as f:
                f.write('# Master EAZY zout catalog for all high-z sources (z=7-11)\n')
                f.write('# Combined from all pointings\n')
                f.write('# ID z_spec z_a z_m1 chi_a\n')
            
            all_zout_data.to_csv(master_zout_file, index=False, sep='\t', mode='a', header=False)
            print(f"✓ Saved master zout catalog with {len(all_zout_data)} sources")
        
        # Summary - SAME AS PREVIOUS CODE
        print(f"\n{'='*50}")
        print("CATALOG CREATION COMPLETE!")
        print(f"{'='*50}")
        print(f"Total pointings processed: {len(all_catalogs)}")
        print(f"Total high-z sources (z=7-11) in master catalog: {len(all_data)}")
        print(f"Redshift range: {all_data['REDSHIFT'].min():.2f} - {all_data['REDSHIFT'].max():.2f}")
        print(f"Master catalog: {master_file}")
        if all_zout_catalogs:
            print(f"Master zout catalog: {master_zout_file}")
        print(f"\nOutput structure:")
        print(f"  {output_dir}/")
        print(f"    ├── [pointing]_50_random_sources.cat (combined catalogs)")
        print(f"    ├── [pointing]_zout_values.cat (EAZY zout values)")
        print(f"    ├── all_pointings_50_sources_master_catalog.cat")
        print(f"    ├── all_pointings_zout_master_catalog.cat")
        print(f"    └── individual_filters/")
        print(f"        └── [pointing]/")
        print(f"            └── random_50_[filter]_catalog.cat (individual filter catalogs)")
    else:
        print("✗ No catalogs were created")

if __name__ == '__main__':
    main()