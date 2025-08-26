import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import os
from astropy.table import Table

# Configuration
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/Drive_cat/Romeos_eazy_cat'
output_base_dir = './snr_plots_romeo'
pointings = [f'nircam{i}' for i in range(1, 11)]
filters = ['f410m']  # Only f410m catalogs are available

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

def plot_snr_values(filter_data, pointing, highlight_ids, output_dir):
    """Generate SNR plots for all filters with max Brenjit SNR lines."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get source IDs from f410m catalog (reference)
    source_ids = filter_data['f410m']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Plot SNR for each filter (only f410m available here)
    for filt in filters:
        if filt not in filter_data:
            continue
            
        plt.figure(figsize=(14, 6))
        
        # Get SNR values
        snr_values = filter_data[filt]['SNR']
        
        # Plot all sources
        plt.scatter(source_ids, snr_values, 
                   alpha=0.5, label='All sources', s=10)
        
        # Highlight Brenjit sources
        plt.scatter(source_ids[highlight_mask], 
                   snr_values[highlight_mask],
                   color='red', label='Brenjit_ID sources', s=30)
        
        # Add horizontal line for max Brenjit SNR
        if any(highlight_mask):
            max_brenjit_snr = np.max(snr_values[highlight_mask])
            plt.axhline(y=max_brenjit_snr, color='darkred', linestyle='--', 
                       linewidth=1.5, alpha=0.7,
                       label=f'Max Brenjit SNR: {max_brenjit_snr:.1f}')
        
        plt.yscale("log")
        plt.xlabel('Source NUMBER (ID)')
        plt.ylabel(f'SNR ({filt})')
        plt.title(f'{pointing}: {filt} SNR vs Source ID')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{pointing}_{filt}_snr.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate SNR plots."""
    os.makedirs(output_base_dir, exist_ok=True)
    
    for pointing in pointings:
        print(f"\nProcessing {pointing}...")
        
        filter_data = {}
        
        # File is directly named like nircam1_f410m_catalog.cat
        for filt in filters:
            cat_path = os.path.join(base_dir, f"{pointing}_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path)
            if data is None:
                print(f"Missing {filt} data for {pointing}")
                break
            filter_data[filt] = data
            
        if filter_data:
            plot_snr_values(
                filter_data, 
                pointing, 
                highlight_ids.get(pointing, []),
                os.path.join(output_base_dir, pointing)
            )
    
    print(f"\nAll SNR plots saved to: {output_base_dir}")

if __name__ == '__main__':
    main()
