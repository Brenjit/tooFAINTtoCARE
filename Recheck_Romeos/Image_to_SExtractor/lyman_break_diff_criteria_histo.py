import numpy as np
from astropy.io import ascii
import os
from astropy.table import Table, join, vstack
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm  # For progress bars

# Global storage for tracking all Brenjit sources
all_brenjit_sources = {}
global_results = {
    'params': [],
    'num_sources': [],
    'recovered_ids': []
}

def read_sextractor_catalog(filepath):
    """Reads a SExtractor catalog file, skipping the commented header."""
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
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        snr[np.isinf(snr) | np.isnan(snr)] = 0
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

def plot_global_recovery(results, output_dir, total_sources):
    """Plots global recovery of all 54 Brenjit sources across parameter combinations."""
    plt.figure(figsize=(15, 6))
    
    # Create combination indices
    combo_indices = np.arange(len(results['num_sources']))
    
    # Plot histogram
    plt.bar(combo_indices, results['num_sources'], width=0.8, color='skyblue')
    
    # Highlight the best performing combination
    best_idx = np.argmax(results['num_sources'])
    plt.bar(best_idx, results['num_sources'][best_idx], color='red', width=0.8)
    
    # Add reference line for total sources
    plt.axhline(y=total_sources, color='green', linestyle='--', 
                label=f'Total Brenjit Sources ({total_sources})')
    
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Number of Brenjit Sources Recovered')
    plt.title(f'Global Recovery of {total_sources} Brenjit Sources Across All Pointings')
    plt.legend()
    
    # Add text showing the best parameters
    best_params = results['params'][best_idx]
    param_text = "\n".join([f"{k}: {v:.2f}" for k, v in best_params.items()])
    plt.annotate(f"Best combination (index {best_idx}):\nRecovered {results['num_sources'][best_idx]}/{total_sources}\n{param_text}",
                 xy=(0.98, 0.85), xycoords='axes fraction',
                 ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'global_recovery_histogram.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved global recovery histogram to {plot_path}")

def apply_f115w_selection_variations(data_dict, brenjit_ids, pointing):
    """
    Applies F115W-dropout selection with systematic parameter variations.
    Tracks recovery of all Brenjit sources globally.
    """
    # Define parameter variations
    variations = {
        'snr_threshold': [2,3,4, 5, 6],
        'color1_min': [0.2,0.3, 0.5, 0.75, 1.0],
        'color2_min': [-1.5, -1.0, -0.5],
        'color2_max': [0.5, 1.0, 1.5],
        'color3_slope': [0.3, 0.44, 0.6],
        'color3_intercept': [0.3, 0.65, 1.0]
    }
    
    # Base parameters
    base_params = {
        'veto_snr': 2,
        'star_cut': 0.8
    }
    
    # Generate all parameter combinations
    param_names = sorted(variations.keys())
    param_values = [variations[name] for name in param_names]
    param_combinations = list(product(*param_values))
    
    # Filter to only Brenjit_ID sources
    brenjit_mask = np.isin(data_dict['f150w']['NUMBER'], brenjit_ids)
    filtered_data = {filt: data[brenjit_mask] for filt, data in data_dict.items()}
    
    # Get photometry for filtered sources
    mags = get_mags(filtered_data, ['f115w', 'f150w', 'f277w'])
    color1 = mags['f115w'] - mags['f150w']
    color2 = mags['f150w'] - mags['f277w']
    
    # Initialize results storage for this pointing
    pointing_results = {
        'params': [],
        'num_sources': [],
        'source_ids': [],
        'combo_index': []
    }
    
    print(f"\nTesting {len(param_combinations)} parameter combinations for {pointing}...")
    
    for idx, combo in enumerate(tqdm(param_combinations)):
        params = dict(zip(param_names, combo))
        params = {**base_params, **params}
        
        # Apply selection
        sn_detect = (filtered_data['f150w']['SNR'] > params['snr_threshold']) & \
                    (filtered_data['f277w']['SNR'] > params['snr_threshold']) & \
                    (filtered_data['f444w']['SNR'] > params['snr_threshold'])
        
        veto = filtered_data['f814w']['SNR'] < params['veto_snr']
        cut1 = (filtered_data['f115w']['SNR'] < params['veto_snr']) | (color1 > params['color1_min'])
        cut2 = (color2 > params['color2_min']) & (color2 < params['color2_max'])
        cut3 = color1 > (params['color3_slope'] * (color2 + 0.8) + params['color3_intercept'])
        star_galaxy_cut = filtered_data['f150w']['CLASS_STAR'] < params['star_cut']
        
        mask = sn_detect & veto & cut1 & cut2 & cut3 & star_galaxy_cut
        selected_ids = filtered_data['f150w']['NUMBER'][mask].data
        
        # Store pointing-level results
        pointing_results['params'].append(params)
        pointing_results['num_sources'].append(len(selected_ids))
        pointing_results['source_ids'].append(selected_ids)
        pointing_results['combo_index'].append(idx)
        
        # Update global tracking
        if idx >= len(global_results['params']):
            global_results['params'].append(params)
            global_results['num_sources'].append(0)
            global_results['recovered_ids'].append(set())
        
        # Track recovered sources globally
        for src_id in selected_ids:
            if src_id not in all_brenjit_sources:
                all_brenjit_sources[src_id] = {
                    'pointing': pointing,
                    'recovered_in': []
                }
            global_results['recovered_ids'][idx].add(src_id)
    
    # Update global counts after processing all combinations
    for idx in range(len(global_results['params'])):
        global_results['num_sources'][idx] = len(global_results['recovered_ids'][idx])
    
    return pointing_results

def main():
    """Main function focusing on global recovery of all Brenjit sources."""
    # Configuration
    base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
    output_base_dir = './brenjit_global_analysis'
    matched_cat_path = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/Drive_cat/final_matched_catalogue.txt'
    pointings = [f'nircam{i}' for i in range(1, 11)]
    catalog_subdir = 'catalogue_z7'
    
    # Load matched catalog
    matched_cat = pd.read_csv(matched_cat_path, delim_whitespace=True)
    total_brenjit_sources = len(matched_cat)
    print(f"\nTotal Brenjit sources to analyze: {total_brenjit_sources}")
    
    # Initialize output
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each pointing
    for pointing in pointings:
        pointing_mask = matched_cat['POINTING'] == pointing
        pointing_ids = matched_cat[pointing_mask]['Brenjit_ID'].values
        
        if len(pointing_ids) == 0:
            print(f"\nNo Brenjit sources in {pointing}, skipping")
            continue
            
        print(f"\nProcessing {len(pointing_ids)} Brenjit sources in {pointing}")
        
        catalog_dir = os.path.join(base_dir, pointing, catalog_subdir)
        if not os.path.isdir(catalog_dir):
            print(f"Directory not found: {catalog_dir}")
            continue
            
        # Load required filters
        required_filters = ['f814w', 'f115w', 'f150w', 'f277w', 'f444w']
        filter_data = {}
        
        for filt in required_filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path)
            if data is not None:
                filter_data[filt] = data
            else:
                print(f"Failed to load {filt} for {pointing}, skipping")
                break
        else:
            # Run analysis for this pointing
            pointing_results = apply_f115w_selection_variations(filter_data, pointing_ids, pointing)
            
            # Save pointing-specific results
            pointing_dir = os.path.join(output_base_dir, pointing)
            os.makedirs(pointing_dir, exist_ok=True)
            
            # Save best parameters
            best_idx = np.argmax(pointing_results['num_sources'])
            with open(os.path.join(pointing_dir, 'best_parameters.txt'), 'w') as f:
                f.write(f"Best combination index: {best_idx}\n")
                f.write(f"Recovered {pointing_results['num_sources'][best_idx]}/{len(pointing_ids)} Brenjit sources\n")
                for k, v in pointing_results['params'][best_idx].items():
                    f.write(f"{k}: {v}\n")
    
    # After processing all pointings, analyze global recovery
    print("\nAnalyzing global recovery of all Brenjit sources...")
    
    # Convert global results to arrays
    global_results['num_sources'] = np.array(global_results['num_sources'])
    
    # Plot global recovery histogram
    plot_global_recovery(global_results, output_base_dir, total_brenjit_sources)
    
    # Save detailed global results
    global_df = pd.DataFrame({
        'combo_index': np.arange(len(global_results['params'])),
        'num_sources': global_results['num_sources'],
        **{k: [p[k] for p in global_results['params']] for k in global_results['params'][0].keys()},
        'recovered_ids': [','.join(map(str, ids)) for ids in global_results['recovered_ids']]
    })
    global_df.to_csv(os.path.join(output_base_dir, 'global_parameter_combinations.csv'), index=False)
    
    # Save source-level recovery information
    source_recovery = []
    for src_id, info in all_brenjit_sources.items():
        source_recovery.append({
            'Brenjit_ID': src_id,
            'pointing': info['pointing'],
            'times_recovered': len(info['recovered_in']),
            'recovery_percentage': 100 * len(info['recovered_in']) / len(global_results['params'])
        })
    
    pd.DataFrame(source_recovery).to_csv(
        os.path.join(output_base_dir, 'source_recovery_stats.csv'), 
        index=False
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Global results saved to: {output_base_dir}")
    print(f"Total Brenjit sources processed: {total_brenjit_sources}")
    print(f"Maximum recovered in any combination: {max(global_results['num_sources'])}")
    print("\nKey Output Files:")
    print("- global_recovery_histogram.png")
    print("- global_parameter_combinations.csv")
    print("- source_recovery_stats.csv")

if __name__ == '__main__':
    main()