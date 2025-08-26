import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import ascii
import os
from astropy.table import Table
from datetime import datetime

import sys
from datetime import datetime

def setup_logging(output_dir):
    """Set up logging to both console and file."""
    log_file = os.path.join(output_dir, "processing_log_with_all_filter_snr_without_star_galaxy.txt")
    # Clear previous log file if it exists
    open(log_file, 'w').close()
    
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
            
    sys.stdout = Logger()
    
    # Write header with timestamp
    print(f"\n{'='*50}")
    print(f"Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")


# Configuration
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
output_base_dir = './snr_and_selection_plots_mine_3'
pointings = [f'nircam{i}' for i in range(1, 11)]
catalog_subdir = 'catalogue_z7'
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

# Brenjit_IDs to highlight
highlight_ids = {
    'nircam1': [2858, 4010, 6802, 8216, 11572, 10899, 8272],
    'nircam2': [1332, 2034, 5726, 11316],
    'nircam3': [9992],
    'nircam4': [315, 1669, 1843, 9859, 9427, 12415, 11659],
    'nircam5': [4547, 9883, 12083, 12779, 12888],
    'nircam6': [3206],
    'nircam7': [7070, 9437, 9959, 11615, 11754, 3581, 13394, 13743, 14097, 14208],
    'nircam8': [1398, 2080, 2288, 4493, 6986, 8954, 9454, 11712, 14604],
    'nircam9': [2646, 3096, 5572, 9061, 8076],
    'nircam10': [1424, 1935, 7847, 9562, 10415]
}

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
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        snr[np.isinf(snr) | np.isnan(snr)] = 0
        data['SNR'] = snr
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_mags(data_dict, filter_list):
    """Helper function to extract magnitudes, replacing non-detections."""
    mags = {}
    for filt in filter_list:
        mags[filt] = np.nan_to_num(data_dict[filt]['MAG_AUTO'], nan=99.0, posinf=99.0, neginf=99.0)
    return mags

def apply_f115w_selection(data_dict):
    """Applies F115W-dropout (z=8.5-12) selection criteria and returns a boolean mask."""
    sn_detect = (data_dict['f150w']['SNR'] > 5) & (data_dict['f200w']['SNR'] > 5) & (data_dict['f277w']['SNR'] > 5) & (data_dict['f356w']['SNR'] > 5) &  (data_dict['f410m']['SNR'] > 5) & (data_dict['f444w']['SNR'] > 5)
    veto = (data_dict['f814w']['SNR'] < 2) & (data_dict['f606w']['SNR'] <2)
    mags = get_mags(data_dict, ['f115w', 'f150w', 'f277w'])
    color1 = mags['f115w'] - mags['f150w']
    color2 = mags['f150w'] - mags['f277w']
    cut1 = (data_dict['f115w']['SNR'] < 2) | (color1 > 0.1)
    cut2 = (color2 > -1.5) & (color2 < 1.0)
    cut3 = color1 > (0.5 + 0.44 * (color2 + 0.8) + 0.5)
    #star_galaxy_cut = data_dict['f150w']['CLASS_STAR'] < 0.8
    mask = sn_detect & veto & cut1
    print(f"Found {np.sum(mask)} candidates.")
    return mask


def plot_snr_values(filter_data, pointing, highlight_ids, output_dir):
    """Generate SNR plots with Brenjit_ID sources marked with green outline if they meet criteria."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get source IDs from f150w catalog (reference)
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Calculate selection criteria
    selection_mask = apply_f115w_selection(filter_data)
    
    # Identify which Brenjit sources meet the criteria
    brenjit_meet_criteria = highlight_mask & selection_mask
    brenjit_fail_criteria = highlight_mask & ~selection_mask
        # Calculate recovery statistics
    total_brenjit = np.sum(highlight_mask)
    recovered_brenjit = np.sum(brenjit_meet_criteria)
    recovery_rate = (recovered_brenjit / total_brenjit) * 100 if total_brenjit > 0 else 0
    
    # Print recovery info
    print(f"Total Brenjit_ID sources: {recovered_brenjit}/{total_brenjit}")
    print(f"Recovery rate: {recovery_rate:.1f}%")
    
    if recovered_brenjit > 0:
        print("Recovered IDs:", 
              ", ".join(map(str, source_ids[brenjit_meet_criteria])))
    if np.sum(brenjit_fail_criteria) > 0:
        print("Missed IDs:", 
              ", ".join(map(str, source_ids[brenjit_fail_criteria])))
    # Plot SNR for each filter
    for filt in filters:
        if filt not in filter_data:
            continue
            
        plt.figure(figsize=(14, 6))
        
        # Get SNR values
        snr_values = filter_data[filt]['SNR']
        
        # Plot all sources (gray)
        plt.scatter(source_ids, snr_values, 
                   alpha=0.3, label='All sources', s=10, color='gray')
        
        # Plot sources that pass selection (green)
        plt.scatter(source_ids[selection_mask & ~highlight_mask], 
                   snr_values[selection_mask & ~highlight_mask],
                   alpha=0.7, label='Meets F150W-dropout', s=15, color='green')
        
        # Highlight Brenjit sources that FAIL criteria (solid red)
        plt.scatter(source_ids[brenjit_fail_criteria], 
                   snr_values[brenjit_fail_criteria],
                   color='red', label='Brenjit_ID (fails criteria)', s=30)
        
        # Highlight Brenjit sources that MEET criteria (red with green edge)
        plt.scatter(source_ids[brenjit_meet_criteria], 
                   snr_values[brenjit_meet_criteria],
                   color='red', edgecolor='green', linewidth=2,
                   label='Brenjit_ID (meets criteria)', s=30)
        
        # Add labels for Brenjit_ID sources
        for src_id in highlight_ids:
            if src_id in source_ids:
                idx = np.where(source_ids == src_id)[0][0]
                x = source_ids[idx]
                y = snr_values[idx]
                
                # Adjust label position to avoid overlap
                offset_x = 0.02 * (max(source_ids) - min(source_ids))
                offset_y = 0.05 * (max(snr_values) - min(snr_values))
                
                plt.annotate(str(src_id), 
                            (x, y),
                            textcoords="offset points",
                            xytext=(10, 10),  # Adjust as needed
                            ha='center',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2',
                                    fc='white',
                                    alpha=0.7,
                                    edgecolor='none'))
        
        # Add horizontal line for min Brenjit SNR
        if any(highlight_mask):
            min_brenjit_snr = np.min(snr_values[highlight_mask])
            plt.axhline(y=min_brenjit_snr, color='darkred', linestyle='--', 
                       linewidth=1.5, alpha=0.7,
                       label=f'Min Brenjit SNR: {min_brenjit_snr:.1f}')
        plt.axhline(y=5, color='blue', linestyle='--', 
                   linewidth=1.5, alpha=0.7,
                   label='SNR = 5 threshold')
        plt.axhline(y=2, color='orange', linestyle='--', 
                   linewidth=1.5, alpha=0.7,
                   label='SNR = 2 threshold')
        plt.yscale("log")
        plt.xlabel('Source NUMBER (ID)')
        plt.ylabel(f'SNR ({filt})')
        plt.title(f'{pointing}: {filt} SNR vs Source ID\n(Brenjit_ID: red=normal, red+green=meets criteria)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to make room for labels
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{pointing}_{filt}_snr_with_selection.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    return total_brenjit, recovered_brenjit


def main():
    """Main function to generate SNR plots with selection highlights."""
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_base_dir)
    
    print(f"Starting processing of {len(pointings)} pointings")
    print(f"Output will be saved to: {output_base_dir}")
    print("Applying F115W (z=8.5-12) selection...")

    total_brenjit_all = 0
    recovered_brenjit_all = 0

    for pointing in pointings:
        print(f"\nProcessing {pointing}...")
        
        # Load catalogs FIRST
        filter_data = {}
        catalog_dir = os.path.join(base_dir, pointing, catalog_subdir)
        
        if not os.path.isdir(catalog_dir):
            print(f"Missing directory: {catalog_dir}")
            continue
            
        # Load all filter catalogs
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path)
            if data is None:
                print(f"Missing {filt} data for {pointing}")
                missing_data = True
                break
            filter_data[filt] = data
            
        if not missing_data:
            # Process data only ONCE
            total, recovered = plot_snr_values(
                filter_data, 
                pointing, 
                highlight_ids.get(pointing, []),
                os.path.join(output_base_dir, pointing)
            )
            total_brenjit_all += total
            recovered_brenjit_all += recovered
    
    # Print final summary
    overall_rate = (recovered_brenjit_all / total_brenjit_all) * 100 if total_brenjit_all > 0 else 0
    print(f"\n{'='*50}")
    print("FINAL RECOVERY SUMMARY:")
    print(f"Total Brenjit_ID sources across all pointings: {total_brenjit_all}")
    print(f"Total recovered after criteria: {recovered_brenjit_all}")
    print(f"Overall recovery rate: {overall_rate:.1f}%")
    print(f"{'='*50}")
    
    print(f"\nProcessing complete. All outputs saved to: {output_base_dir}")
    print(f"Detailed log saved to: {os.path.join(output_base_dir, 'processing_log_with_all_filter_snr_without_star_galaxy.txt')}")


if __name__ == '__main__':
    main()