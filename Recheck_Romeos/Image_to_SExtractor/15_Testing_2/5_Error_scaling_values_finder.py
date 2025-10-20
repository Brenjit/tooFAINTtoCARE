"""for each pointing in [nircam1..10]:
    for each filter in filters:
        read SExtractor catalog
    read NMAD flux catalog
    compute NMAD SNRs
    match sources by ID
    compare SNRs (diff, ratio)
    save comparison + summary tables
    plot:
        - per source
        - per filter
        - per pointing
collect all results
compute scaling factors (median fits)
save final combined summaries"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import ascii
from astropy.table import Table
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
from scipy import stats

# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
nmad_base_dir = '/home/iit-t/Bren_jit/tooFAINTtoCARE/Recheck_Romeos/Image_to_SExtractor/15_Testing_2/4_Eazy_catalogue_of_sept_data'
output_dir = './5_SNR_scaling_factor_finder'

# Define pointings only (no redshift bins)
pointings = [f'nircam{i}' for i in range(1, 11)]

filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
filter_mapping = {
    'f606w': 'F606W', 'f814w': 'F814W', 'f115w': 'F115W', 'f150w': 'F150W',
    'f200w': 'F200W', 'f277w': 'F277W', 'f356w': 'F356W', 'f410m': 'F410M', 'f444w': 'F444W'
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

def read_nmad_catalog(filepath):
    """Reads the NMAD catalog with flux measurements."""
    try:
        # Read the catalog
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract header information
        header = []
        data_lines = []
        for line in lines:
            if line.startswith('#'):
                header.append(line.strip())
            else:
                data_lines.append(line.strip())
        
        # Parse the data
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
        print(f"Error reading NMAD catalog {filepath}: {e}")
        return None

def calculate_nmad_snr(nmad_data):
    """Calculate SNR from NMAD flux and error measurements."""
    snr_data = {}
    for filt in filters:
        nmad_filt = filter_mapping[filt]
        flux_col = f'f_{nmad_filt}'
        err_col = f'e_{nmad_filt}'
        
        if flux_col in nmad_data.columns and err_col in nmad_data.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = nmad_data[flux_col] / nmad_data[err_col]
            snr[np.isinf(snr) | np.isnan(snr)] = 0
            snr_data[filt] = snr
        else:
            print(f"Warning: Columns {flux_col} or {err_col} not found in NMAD data")
            snr_data[filt] = np.zeros(len(nmad_data))
    
    return snr_data

def get_source_ids_from_nmad(nmad_data):
    """Extract source IDs from NMAD data."""
    return nmad_data['id'].tolist()

def create_comparison_catalog(sextractor_data, nmad_data, nmad_snr, pointing, output_dir):
    """Create a comprehensive catalog comparing both SNR calculation methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all source IDs from NMAD data for this pointing
    source_ids = get_source_ids_from_nmad(nmad_data)
    
    if not source_ids:
        print(f"No source IDs found in NMAD data for {pointing}")
        return None, None
    
    print(f"Found {len(source_ids)} sources in NMAD data for {pointing}")
    
    # Initialize the output table
    output_data = []
    
    for src_id in source_ids:
        # Find the source in SExtractor data
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            print(f"Source {src_id} not found in SExtractor data for {pointing}")
            continue
        
        sex_idx = sex_idx[0]
        
        # Find the source in NMAD data
        nmad_idx = np.where(nmad_data['id'] == src_id)[0]
        if len(nmad_idx) == 0:
            print(f"Source {src_id} not found in NMAD data for {pointing}")
            continue
        
        nmad_idx = nmad_idx[0]
        
        # Get position information from SExtractor
        x_image = sextractor_data['f150w']['X_IMAGE'][sex_idx]
        y_image = sextractor_data['f150w']['Y_IMAGE'][sex_idx]
        alpha_j2000 = sextractor_data['f150w']['ALPHA_J2000'][sex_idx]
        delta_j2000 = sextractor_data['f150w']['DELTA_J2000'][sex_idx]
        class_star = sextractor_data['f150w']['CLASS_STAR'][sex_idx]
        
        # Create a row for each filter
        for filt in filters:
            row = {
                'NUMBER': src_id,
                'X_IMAGE': x_image,
                'Y_IMAGE': y_image,
                'ALPHA_J2000': alpha_j2000,
                'DELTA_J2000': delta_j2000,
                'CLASS_STAR': class_star,
                'FILTER': filt.upper(),
                'SEX_FLUX': sextractor_data[filt]['FLUX_AUTO'][sex_idx],
                'SEX_FLUXERR': sextractor_data[filt]['FLUXERR_AUTO'][sex_idx],
                'SEX_SNR': sextractor_data[filt]['SNR'][sex_idx],
                'NMAD_FLUX': nmad_data[f'f_{filter_mapping[filt]}'].iloc[nmad_idx],
                'NMAD_FLUXERR': nmad_data[f'e_{filter_mapping[filt]}'].iloc[nmad_idx],
                'NMAD_SNR': nmad_snr[filt][nmad_idx],
                'SNR_DIFF': sextractor_data[filt]['SNR'][sex_idx] - nmad_snr[filt][nmad_idx],
                'SNR_RATIO': sextractor_data[filt]['SNR'][sex_idx] / nmad_snr[filt][nmad_idx] if nmad_snr[filt][nmad_idx] > 0 else np.nan,
                'POINTING': pointing
            }
            output_data.append(row)
    
    # Convert to DataFrame and save as TXT
    if not output_data:
        print(f"No valid comparison data for {pointing}")
        return None, None
        
    df = pd.DataFrame(output_data)
    output_file = os.path.join(output_dir, f'{pointing}_snr_comparison.txt')
    df.to_csv(output_file, index=False, sep='\t')
    
    # Also create a summary file with just the key statistics
    summary_df = df.groupby(['NUMBER', 'FILTER']).agg({
        'SEX_SNR': 'first',
        'NMAD_SNR': 'first',
        'SNR_DIFF': 'first',
        'SNR_RATIO': 'first'
    }).reset_index()
    
    summary_file = os.path.join(output_dir, f'{pointing}_snr_summary.txt')
    summary_df.to_csv(summary_file, index=False, sep='\t')
    
    return df, summary_df

def calculate_median_fit(sex_snr, nmad_snr):
    """
    Calculates the intercept for a line with slope 1
    that passes through the median of the log-transformed data.
    """
    # Filter out invalid values
    valid_mask = (sex_snr > 0) & (nmad_snr > 0) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
    
    if np.sum(valid_mask) == 0:
        return None
    
    sex_valid = sex_snr[valid_mask]
    nmad_valid = nmad_snr[valid_mask]
    
    # Take the logarithm of the valid data
    log_sex_snr = np.log10(sex_valid)
    log_nmad_snr = np.log10(nmad_valid)
    
    # Find the median of the log-transformed data
    median_log_x = np.median(log_sex_snr)
    median_log_y = np.median(log_nmad_snr)
    
    # The line is log(y) = 1 * log(x) + c.
    # We find c by plugging in the median point.
    intercept = median_log_y - median_log_x
    
    # The slope is fixed at 1
    slope = 1.0
    
    return {
        'slope': slope,
        'intercept': intercept,
        'n_points': len(sex_valid)
    }

def plot_individual_source_comparison(comparison_data, pointing, output_dir):
    """Generate individual plots for each source showing SExtractor vs NMAD SNR across filters."""
    individual_dir = os.path.join(output_dir, pointing, 'individual_sources')
    os.makedirs(individual_dir, exist_ok=True)
    
    source_ids = comparison_data['NUMBER'].unique()
    
    for src_id in source_ids:
        src_data = comparison_data[comparison_data['NUMBER'] == src_id]
        
        plt.figure(figsize=(12, 8))
        
        for i, filt in enumerate(filters):
            filter_data = src_data[src_data['FILTER'] == filt.upper()]
            if len(filter_data) > 0:
                plt.scatter(filter_data['SEX_SNR'], filter_data['NMAD_SNR'], 
                        color=plt.cm.tab10(i), s=100, label=filt.upper(), alpha=0.8)
        
        max_val = max(src_data['SEX_SNR'].max(), src_data['NMAD_SNR'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        sex_all = src_data['SEX_SNR'].values
        nmad_all = src_data['NMAD_SNR'].values
        fit_result = calculate_median_fit(sex_all, nmad_all)
        
        if fit_result:
            x_fit = np.linspace(0, max_val, 100)
            y_fit = (10**fit_result['intercept']) * x_fit
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Median Fit: y = {10**fit_result["intercept"]:.5f}x')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('SExtractor SNR')
        plt.ylabel('NMAD SNR')
        plt.title(f'{pointing} - Source {src_id}: SExtractor vs NMAD SNR by Filter')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = os.path.join(individual_dir, f'{pointing}_source_{src_id}_snr_comparison.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

def plot_all_sources_comparison(comparison_data, pointing, output_dir):
    """Generate plots showing all sources with different colors."""
    all_sources_dir = os.path.join(output_dir, pointing, 'all_sources_comparison')
    os.makedirs(all_sources_dir, exist_ok=True)
    
    source_ids = comparison_data['NUMBER'].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(source_ids)))
    color_map = dict(zip(source_ids, colors))
    
    # Plot 1: All filters in one plot
    plt.figure(figsize=(12, 8))
    
    for src_id in source_ids:
        src_data = comparison_data[comparison_data['NUMBER'] == src_id]
        plt.scatter(src_data['SEX_SNR'], src_data['NMAD_SNR'], 
                color=color_map[src_id], s=50, alpha=0.7, label=f'ID {src_id}')
    
    max_val = max(comparison_data['SEX_SNR'].max(), comparison_data['NMAD_SNR'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
    
    sex_all = comparison_data['SEX_SNR'].values
    nmad_all = comparison_data['NMAD_SNR'].values
    fit_result = calculate_median_fit(sex_all, nmad_all)
    
    if fit_result:
        x_fit = np.linspace(0, max_val, 100)
        y_fit = (10**fit_result['intercept']) * x_fit
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Median Fit: y = {10**fit_result["intercept"]:.5f}x')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('SExtractor SNR')
    plt.ylabel('NMAD SNR')
    plt.title(f'{pointing}: All Sources - SExtractor vs NMAD SNR')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(all_sources_dir, f'{pointing}_all_sources_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Separate plot for each filter with median fit
    for filt in filters:
        plt.figure(figsize=(12, 8))
        filter_data = comparison_data[comparison_data['FILTER'] == filt.upper()]
        
        for src_id in source_ids:
            src_filter_data = filter_data[filter_data['NUMBER'] == src_id]
            if len(src_filter_data) > 0:
                plt.scatter(src_filter_data['SEX_SNR'], src_filter_data['NMAD_SNR'], 
                        color=color_map[src_id], s=100, label=f'ID {src_id}', alpha=0.8)
        
        max_val = max(filter_data['SEX_SNR'].max(), filter_data['NMAD_SNR'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        sex_filter = filter_data['SEX_SNR'].values
        nmad_filter = filter_data['NMAD_SNR'].values
        fit_result = calculate_median_fit(sex_filter, nmad_filter)
        
        if fit_result:
            x_fit = np.linspace(0, max_val, 100)
            y_fit = (10**fit_result['intercept']) * x_fit
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Median Fit: y = {10**fit_result["intercept"]:.5f}x')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('SExtractor SNR')
        plt.ylabel('NMAD SNR')
        plt.title(f'{pointing}: {filt.upper()} - SExtractor vs NMAD SNR by Source')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(all_sources_dir, f'{pointing}_{filt}_sources_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

def plot_snr_comparison(comparison_data, pointing, output_dir):
    """Generate plots comparing SNR values from both methods."""
    if comparison_data is None or len(comparison_data) == 0:
        print(f"No comparison data for {pointing}")
        return
    
    plot_individual_source_comparison(comparison_data, pointing, output_dir)
    plot_all_sources_comparison(comparison_data, pointing, output_dir)
    
    n_filters = len(filters)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, filt in enumerate(filters):
        filter_data = comparison_data[comparison_data['FILTER'] == filt.upper()]
        
        if len(filter_data) == 0:
            axes[i].set_visible(False)
            continue
        
        source_ids = filter_data['NUMBER'].unique()
        colors = cm.rainbow(np.linspace(0, 1, len(source_ids)))
        
        for j, src_id in enumerate(source_ids):
            src_data = filter_data[filter_data['NUMBER'] == src_id]
            axes[i].scatter(src_data['SEX_SNR'], src_data['NMAD_SNR'], 
                        color=colors[j], s=50, alpha=0.7, label=f'ID {src_id}')
        
        max_val = max(filter_data['SEX_SNR'].max(), filter_data['NMAD_SNR'].max())
        axes[i].plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        sex_filter = filter_data['SEX_SNR'].values
        nmad_filter = filter_data['NMAD_SNR'].values
        fit_result = calculate_median_fit(sex_filter, nmad_filter)
        
        if fit_result:
            x_fit = np.linspace(0, max_val, 100)
            y_fit = (10**fit_result['intercept']) * x_fit
            axes[i].plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Median Fit: y = {10**fit_result["intercept"]:.3f}x')
        
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')
        axes[i].set_xlabel('SExtractor SNR')
        axes[i].set_ylabel('NMAD SNR')
        axes[i].set_title(f'{filt.upper()} SNR Comparison')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, pointing, f'{pointing}_snr_comparison_by_filter.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_fit_summary(all_comparison_data, output_dir, sigma_clip=3.0):
    """
    Save a text file with pointing, filter, 10^intercept values, and average per pointing.
    Uses sigma-clipping to remove high-end deviations before averaging.
    """
    summary_lines = []
    
    for pointing, comp_data in all_comparison_data.items():
        fit_values = []
        summary_lines.append(f"\nPointing: {pointing}")
        
        for filt in filters:
            filter_data = comp_data[comp_data['FILTER'] == filt.upper()]
            sex_snr = filter_data['SEX_SNR'].values
            nmad_snr = filter_data['NMAD_SNR'].values
            
            fit_result = calculate_median_fit(sex_snr, nmad_snr)
            if fit_result:
                scale_factor = 10**fit_result["intercept"]
                fit_values.append(scale_factor)
                summary_lines.append(f"  {filt.upper()}: {scale_factor:.5f}")
            else:
                summary_lines.append(f"  {filt.upper()}: NA")
        
        # Sigma-clip high-end deviations
        if len(fit_values) > 0:
            fit_array = np.array(fit_values)
            clipped, _, _ = stats.sigmaclip(fit_array, low=sigma_clip, high=sigma_clip)
            avg_scale = np.mean(clipped) if len(clipped) > 0 else np.nan
            summary_lines.append(f"  Average (sigma-clipped): {avg_scale:.5f}")
        else:
            summary_lines.append(f"  Average (sigma-clipped): NA")
    
    # Write to txt
    output_file = os.path.join(output_dir, "snr_fit_summary.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"Saved fit summary to {output_file}")

def main():
    """Main function to compare SNR values from SExtractor and NMAD methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting SNR comparison analysis with NMAD data")
    print(f"Output will be saved to: {output_dir}")
    
    all_comparison_data = {}  # Dictionary to store all comparison data
    
    for pointing in pointings:
        print(f"\nProcessing {pointing}")
        
        # Read SExtractor data
        sextractor_data = {}
        catalog_dir = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7')
        
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path)
            if data is None:
                print(f"  Missing {filt} data for {pointing}")
                missing_data = True
                break
            sextractor_data[filt] = data
        
        if missing_data:
            continue
        
        # Read NMAD catalog from new location
        nmad_file = os.path.join(nmad_base_dir, f'{pointing}_eazy_catalogue.cat')
        if not os.path.exists(nmad_file):
            print(f"  NMAD catalog not found: {nmad_file}")
            continue
        
        nmad_data = read_nmad_catalog(nmad_file)
        if nmad_data is None or len(nmad_data) == 0:
            print(f"  Could not read NMAD data for {pointing}")
            continue
        
        print(f"  Found {len(nmad_data)} sources in NMAD catalog")
        
        nmad_snr = calculate_nmad_snr(nmad_data)
        comparison_df, summary_df = create_comparison_catalog(
            sextractor_data, nmad_data, nmad_snr, pointing, output_dir
        )
        
        if comparison_df is not None:
            all_comparison_data[pointing] = comparison_df
            
            # Create plots
            plot_dir = os.path.join(output_dir, pointing)
            os.makedirs(plot_dir, exist_ok=True)
            plot_snr_comparison(comparison_df, pointing, output_dir)
            
            print(f"  Processed {len(comparison_df['NUMBER'].unique())} sources for {pointing}")
    
    if all_comparison_data:
        # Combine all data and save as TXT
        all_data_list = []
        for pointing, data in all_comparison_data.items():
            data['pointing'] = pointing
            all_data_list.append(data)
        
        all_data = pd.concat(all_data_list, ignore_index=True)
        all_data_file = os.path.join(output_dir, 'all_pointings_snr_comparison.txt')
        all_data.to_csv(all_data_file, index=False, sep='\t')
        
        # Save fit summary
        save_fit_summary(all_comparison_data, output_dir)
        
        # Create overall summary statistics as TXT
        overall_summary = all_data.groupby(['FILTER', 'pointing']).agg({
            'SEX_SNR': ['mean', 'std', 'count'],
            'NMAD_SNR': ['mean', 'std', 'count'],
            'SNR_DIFF': ['mean', 'std', 'count'],
            'SNR_RATIO': ['mean', 'std', 'count']
        }).round(3)
        
        overall_summary_file = os.path.join(output_dir, 'overall_snr_summary.txt')
        with open(overall_summary_file, 'w') as f:
            f.write("Overall SNR Summary Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(overall_summary.to_string())
        
        print(f"\nAnalysis complete. Processed {len(all_data)} measurements across {len(all_comparison_data)} pointings")
    else:
        print("No data was processed")

if __name__ == '__main__':
    main()

    