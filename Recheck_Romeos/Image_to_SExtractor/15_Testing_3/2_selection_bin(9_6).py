import numpy as np
import os
from astropy.io import ascii
import pandas as pd
import random

# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
output_dir = './50_2_random_sources_catalogues'
pointings = [f'nircam{i}' for i in range(1, 11)]

filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

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

def select_random_sources(sextractor_data, n_sources=50):
    """Select n_sources randomly from SExtractor catalog."""
    # Use f150w as reference filter to get source IDs
    all_sources = sextractor_data['f150w']
    
    if len(all_sources) <= n_sources:
        print(f"Only {len(all_sources)} sources available, selecting all")
        selected_ids = all_sources['NUMBER']
    else:
        # Randomly select sources
        selected_ids = random.sample(list(all_sources['NUMBER']), n_sources)
    
    return selected_ids

def create_sextractor_like_catalog(sextractor_data, selected_ids, pointing, output_dir):
    """Create a SExtractor-like catalog for selected sources."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating catalog for {len(selected_ids)} random sources for {pointing}")
    
    # Initialize the output data
    output_data = []
    
    for src_id in selected_ids:
        # Find the source in SExtractor data (use f150w as reference for position info)
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            print(f"⚠️  Source {src_id} not found in SExtractor data for {pointing}")
            continue
        
        sex_idx = sex_idx[0]
        
        # Get basic information from SExtractor
        x_image = sextractor_data['f150w']['X_IMAGE'][sex_idx]
        y_image = sextractor_data['f150w']['Y_IMAGE'][sex_idx]
        alpha_j2000 = sextractor_data['f150w']['ALPHA_J2000'][sex_idx]
        delta_j2000 = sextractor_data['f150w']['DELTA_J2000'][sex_idx]
        class_star = sextractor_data['f150w']['CLASS_STAR'][sex_idx]
        
        # Create a row with all filter information
        row = {
            'NUMBER': src_id,
            'X_IMAGE': x_image,
            'Y_IMAGE': y_image,
            'ALPHA_J2000': alpha_j2000,
            'DELTA_J2000': delta_j2000,
            'CLASS_STAR': class_star,
            'POINTING': pointing
        }
        
        # Add SExtractor measurements for each filter
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
    
    # Convert to DataFrame and save
    if output_data:
        df = pd.DataFrame(output_data)
        
        # Save the catalog
        output_file = os.path.join(output_dir, f'{pointing}_50_random_sources.cat')
        
        # Write header
        with open(output_file, 'w') as f:
            f.write('# SExtractor-like catalog for 50 random sources\n')
            f.write('# Created from original SExtractor catalogs\n')
            f.write('# Format: ID, coordinates, CLASS_STAR, and photometry for all filters\n')
        
        # Save data
        df.to_csv(output_file, index=False, sep='\t', mode='a')
        
        print(f"✓ Saved catalog with {len(df)} sources to {output_file}")
        return df
    else:
        print(f"✗ No valid data found for {pointing}")
        return pd.DataFrame()

def create_individual_filter_catalogs(sextractor_data, selected_ids, pointing, output_dir):
    """Create individual filter catalogs in original SExtractor format."""
    filter_output_dir = os.path.join(output_dir, 'individual_filters', pointing)
    os.makedirs(filter_output_dir, exist_ok=True)
    
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
            
            # Write in SExtractor-like format
            with open(output_file, 'w') as f:
                f.write(f'# SExtractor catalog for 50 random sources - Filter: {filt}\n')
                f.write('# NUMBER X_IMAGE Y_IMAGE MAG_AUTO MAGERR_AUTO MAG_APER MAGERR_APER CLASS_STAR FLUX_AUTO FLUXERR_AUTO FLUX_APER FLUXERR_APER ALPHA_J2000 DELTA_J2000\n')
            
            df.to_csv(output_file, index=False, sep='\t', mode='a', header=False)
            print(f"✓ Saved {filt} catalog with {len(df)} sources")

def main():
    """Main function to create catalogs for 50 random sources per pointing."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating catalogs for 50 random sources per pointing")
    print(f"Output directory: {output_dir}")
    
    # Set random seed for reproducibility
    random.seed(56)
    
    all_catalogs = {}
    
    for pointing in pointings:
        print(f"\n{'='*50}")
        print(f"Processing {pointing}")
        print(f"{'='*50}")
        
        # Read SExtractor data for all filters
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
        
        # Select 50 random sources
        selected_ids = select_random_sources(sextractor_data, n_sources=50)
        print(f"Selected {len(selected_ids)} random sources")
        
        # Create combined catalog
        catalog_df = create_sextractor_like_catalog(
            sextractor_data, selected_ids, pointing, output_dir
        )
        
        # Create individual filter catalogs
        create_individual_filter_catalogs(
            sextractor_data, selected_ids, pointing, output_dir
        )
        
        if len(catalog_df) > 0:
            all_catalogs[pointing] = catalog_df
    
    # Create a master combined catalog
    if all_catalogs:
        all_data = pd.concat(all_catalogs.values(), ignore_index=True)
        master_file = os.path.join(output_dir, 'all_pointings_50_sources_master_catalog.cat')
        all_data.to_csv(master_file, index=False, sep='\t')
        
        # Summary
        print(f"\n{'='*50}")
        print("CATALOG CREATION COMPLETE!")
        print(f"{'='*50}")
        print(f"Total pointings processed: {len(all_catalogs)}")
        print(f"Total sources in master catalog: {len(all_data)}")
        print(f"Master catalog: {master_file}")
        print(f"\nOutput structure:")
        print(f"  {output_dir}/")
        print(f"    ├── [pointing]_50_random_sources.cat (combined catalogs)")
        print(f"    └── individual_filters/")
        print(f"        └── [pointing]/")
        print(f"            └── random_50_[filter]_catalog.cat (individual filter catalogs)")
    else:
        print("✗ No catalogs were created")

if __name__ == '__main__':
    main()