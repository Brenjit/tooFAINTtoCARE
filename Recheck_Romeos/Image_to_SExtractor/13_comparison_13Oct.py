import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results'
nmad_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/10_nmad_selected_15_Oct'
output_dir = './13_SNR_Comparison_Analysis_15_Oct'

# Define all parameters
pointings = [f'nircam{i}' for i in range(1, 11)]
redshift_bins = ['z7-8']  #['z7-8', 'z8-10', 'z10-15']
redshift_paths = {
    'z7-8': 'redshift_z7-8',
    #'z8-10': 'redshift_z8-10', 
    #'z10-15': 'redshift_z10-15'
}
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

def read_sextractor_catalog(filepath):
    """Read SExtractor catalog and extract SNR information."""
    try:
        data = ascii.read(filepath)
        # Calculate SNR from FLUX_AUTO and FLUXERR_AUTO
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'NUMBER': np.array(data['NUMBER']),
            'SNR': np.array(snr),
            'FLUX_AUTO': np.array(data['FLUX_AUTO']),
            'FLUXERR_AUTO': np.array(data['FLUXERR_AUTO'])
        }
    except Exception as e:
        print(f"Error reading SExtractor catalog {filepath}: {e}")
        return None

def read_nmad_catalog(filepath):
    """Read NMAD catalog with flux measurements."""
    try:
        # Read the NMAD catalog
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = line.strip().split()
                if len(values) == 19:  # id + 9 filters * (flux + error)
                    row = {'id': int(values[0])}
                    # Parse flux and error for each filter
                    for i, filt in enumerate(filters):
                        row[f'f_{filt.upper()}'] = float(values[1 + i*2])
                        row[f'e_{filt.upper()}'] = float(values[2 + i*2])
                    data.append(row)
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error reading NMAD catalog {filepath}: {e}")
        return None

def calculate_nmad_snr(nmad_df):
    """Calculate SNR from NMAD flux and error measurements."""
    snr_data = {}
    for filt in filters:
        flux_col = f'f_{filt.upper()}'
        err_col = f'e_{filt.upper()}'
        
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = nmad_df[flux_col] / nmad_df[err_col]
        snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
        snr_data[filt.upper()] = snr
    
    return snr_data

def create_comparison_data(sextractor_data, nmad_df, pointing, redshift_bin):
    """Create comparison data between SExtractor and NMAD SNR for matched sources."""
    comparison_data = []
    
    # For each source in NMAD catalog, find matching source in SExtractor
    for _, nmad_row in nmad_df.iterrows():
        source_id = nmad_row['id']
        
        # Find this source in SExtractor F150W data (reference filter)
        sex_mask = sextractor_data['f150w']['NUMBER'] == source_id
        if not np.any(sex_mask):
            continue  # Skip if source not found in SExtractor
            
        sex_idx = np.where(sex_mask)[0][0]
        
        # Compare SNR for each filter
        for filt in filters:
            filt_upper = filt.upper()
            
            # Get SExtractor SNR
            sex_snr = sextractor_data[filt]['SNR'][sex_idx]
            
            # Get NMAD SNR
            nmad_snr_val = nmad_row[f'f_{filt_upper}'] / nmad_row[f'e_{filt_upper}'] if nmad_row[f'e_{filt_upper}'] > 0 else 0
            
            # Calculate ratio (handle division by zero)
            if nmad_snr_val > 0:
                snr_ratio = sex_snr / nmad_snr_val
            else:
                snr_ratio = np.nan
            
            comparison_data.append({
                'POINTING': pointing,
                'REDSHIFT_BIN': redshift_bin,
                'SOURCE_ID': source_id,
                'FILTER': filt_upper,
                'SEX_SNR': sex_snr,
                'NMAD_SNR': nmad_snr_val,
                'SNR_RATIO': snr_ratio,
                'LOG_SEX_SNR': np.log10(sex_snr) if sex_snr > 0 else np.nan,
                'LOG_NMAD_SNR': np.log10(nmad_snr_val) if nmad_snr_val > 0 else np.nan
            })
    
    return pd.DataFrame(comparison_data)

def create_histogram_plots(comparison_df, output_dir):
    """Create histogram-based diagnostic plots."""
    
    # 1. Histogram of SNR ratios
    plt.figure(figsize=(15, 12))
    
    # Filter out invalid ratios
    valid_ratios = comparison_df['SNR_RATIO'].dropna()
    valid_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios < 100)]  # Reasonable range
    
    plt.subplot(2, 2, 1)
    plt.hist(valid_ratios, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(valid_ratios.median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: {valid_ratios.median():.2f}')
    plt.xlabel('SNR Ratio (SExtractor / NMAD)')
    plt.ylabel('Count')
    plt.title('Distribution of SNR Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Histogram by filter
    plt.subplot(2, 2, 2)
    filter_medians = []
    filter_labels = []
    for filt in filters:
        filt_data = comparison_df[comparison_df['FILTER'] == filt.upper()]['SNR_RATIO'].dropna()
        filt_data = filt_data[(filt_data > 0) & (filt_data < 100)]
        if len(filt_data) > 0:
            filter_medians.append(filt_data.median())
            filter_labels.append(filt.upper())
    
    plt.bar(filter_labels, filter_medians, alpha=0.7, edgecolor='black')
    plt.axhline(np.median(filter_medians), color='red', linestyle='--', 
                label=f'Overall: {np.median(filter_medians):.2f}')
    plt.xlabel('Filter')
    plt.ylabel('Median SNR Ratio')
    plt.title('Median SNR Ratio by Filter')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Histogram by redshift bin
    plt.subplot(2, 2, 3)
    redshift_medians = []
    redshift_labels = []
    for zbin in redshift_bins:
        zbin_data = comparison_df[comparison_df['REDSHIFT_BIN'] == zbin]['SNR_RATIO'].dropna()
        zbin_data = zbin_data[(zbin_data > 0) & (zbin_data < 100)]
        if len(zbin_data) > 0:
            redshift_medians.append(zbin_data.median())
            redshift_labels.append(zbin)
    
    plt.bar(redshift_labels, redshift_medians, alpha=0.7, edgecolor='black')
    plt.xlabel('Redshift Bin')
    plt.ylabel('Median SNR Ratio')
    plt.title('Median SNR Ratio by Redshift Bin')
    plt.grid(True, alpha=0.3)
    
    # 4. Distribution of log SNR values
    plt.subplot(2, 2, 4)
    valid_sex_log = comparison_df['LOG_SEX_SNR'].dropna()
    valid_nmad_log = comparison_df['LOG_NMAD_SNR'].dropna()
    
    plt.hist(valid_sex_log, bins=30, alpha=0.7, label='SExtractor', edgecolor='blue')
    plt.hist(valid_nmad_log, bins=30, alpha=0.7, label='NMAD', edgecolor='red')
    plt.xlabel('log10(SNR)')
    plt.ylabel('Count')
    plt.title('Distribution of log(SNR) Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_one_to_one_plots(comparison_df, output_dir):
    """Create 1:1 comparison plots."""
    
    # Main 1:1 plot
    plt.figure(figsize=(15, 12))
    
    # 1. Overall 1:1 plot
    plt.subplot(2, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(filters)))
    
    for i, filt in enumerate(filters):
        filt_data = comparison_df[comparison_df['FILTER'] == filt.upper()]
        valid_mask = (filt_data['SEX_SNR'] > 0) & (filt_data['NMAD_SNR'] > 0)
        if np.sum(valid_mask) > 0:
            plt.scatter(filt_data['SEX_SNR'][valid_mask], filt_data['NMAD_SNR'][valid_mask],
                       alpha=0.6, label=filt.upper(), s=30, color=colors[i])
    
    # Plot 1:1 line
    max_val = max(comparison_df['SEX_SNR'].max(), comparison_df['NMAD_SNR'].max())
    plt.plot([0.1, max_val], [0.1, max_val], 'k--', alpha=0.7, linewidth=2, label='1:1 line')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('SExtractor SNR')
    plt.ylabel('NMAD SNR')
    plt.title('1:1 SNR Comparison (All Filters)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. SNR Ratio vs SExtractor SNR
    plt.subplot(2, 2, 2)
    valid_data = comparison_df.dropna(subset=['SNR_RATIO'])
    valid_data = valid_data[(valid_data['SNR_RATIO'] > 0) & (valid_data['SNR_RATIO'] < 100)]
    
    plt.scatter(valid_data['SEX_SNR'], valid_data['SNR_RATIO'], alpha=0.6, s=20)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Ratio = 1')
    plt.axhline(valid_data['SNR_RATIO'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {valid_data["SNR_RATIO"].median():.2f}')
    plt.xscale('log')
    plt.xlabel('SExtractor SNR')
    plt.ylabel('SNR Ratio (SExtractor / NMAD)')
    plt.title('SNR Ratio vs SExtractor SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. By redshift bin
    plt.subplot(2, 2, 3)
    for zbin in redshift_bins:
        zbin_data = comparison_df[comparison_df['REDSHIFT_BIN'] == zbin]
        valid_mask = (zbin_data['SEX_SNR'] > 0) & (zbin_data['NMAD_SNR'] > 0)
        if np.sum(valid_mask) > 0:
            plt.scatter(zbin_data['SEX_SNR'][valid_mask], zbin_data['NMAD_SNR'][valid_mask],
                       alpha=0.6, label=zbin, s=30)
    
    plt.plot([0.1, max_val], [0.1, max_val], 'k--', alpha=0.7, linewidth=2, label='1:1 line')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('SExtractor SNR')
    plt.ylabel('NMAD SNR')
    plt.title('1:1 SNR Comparison by Redshift Bin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. SNR distribution boxplot by filter
    plt.subplot(2, 2, 4)
    plot_data = []
    labels = []
    for filt in filters:
        filt_data = comparison_df[comparison_df['FILTER'] == filt.upper()]
        sex_snr = filt_data['SEX_SNR'][filt_data['SEX_SNR'] > 0]
        nmad_snr = filt_data['NMAD_SNR'][filt_data['NMAD_SNR'] > 0]
        if len(sex_snr) > 0 and len(nmad_snr) > 0:
            plot_data.append(sex_snr)
            plot_data.append(nmad_snr)
            labels.extend([f'{filt.upper()}\nSEX', f'{filt.upper()}\nNMAD'])
    
    plt.boxplot(plot_data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('SNR')
    plt.yscale('log')
    plt.title('SNR Distribution by Filter and Method')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'one_to_one_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(comparison_df, output_dir):
    """Generate summary statistics and save to file."""
    
    summary_lines = []
    summary_lines.append("SNR COMPARISON ANALYSIS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Total comparisons: {len(comparison_df)}")
    summary_lines.append(f"Total unique sources: {comparison_df['SOURCE_ID'].nunique()}")
    summary_lines.append(f"Total pointings: {comparison_df['POINTING'].nunique()}")
    summary_lines.append("")
    
    # Overall statistics
    valid_ratios = comparison_df['SNR_RATIO'].dropna()
    valid_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios < 100)]
    
    summary_lines.append("OVERALL STATISTICS:")
    summary_lines.append(f"  Median SNR ratio: {valid_ratios.median():.3f}")
    summary_lines.append(f"  Mean SNR ratio: {valid_ratios.mean():.3f}")
    summary_lines.append(f"  Std SNR ratio: {valid_ratios.std():.3f}")
    summary_lines.append(f"  Min SNR ratio: {valid_ratios.min():.3f}")
    summary_lines.append(f"  Max SNR ratio: {valid_ratios.max():.3f}")
    summary_lines.append("")
    
    # By filter
    summary_lines.append("BY FILTER:")
    for filt in filters:
        filt_data = comparison_df[comparison_df['FILTER'] == filt.upper()]
        filt_ratios = filt_data['SNR_RATIO'].dropna()
        filt_ratios = filt_ratios[(filt_ratios > 0) & (filt_ratios < 100)]
        if len(filt_ratios) > 0:
            summary_lines.append(f"  {filt.upper()}: {filt_ratios.median():.3f} (n={len(filt_ratios)})")
    summary_lines.append("")
    
    # By redshift bin
    summary_lines.append("BY REDSHIFT BIN:")
    for zbin in redshift_bins:
        zbin_data = comparison_df[comparison_df['REDSHIFT_BIN'] == zbin]
        zbin_ratios = zbin_data['SNR_RATIO'].dropna()
        zbin_ratios = zbin_ratios[(zbin_ratios > 0) & (zbin_ratios < 100)]
        if len(zbin_ratios) > 0:
            summary_lines.append(f"  {zbin}: {zbin_ratios.median():.3f} (n={len(zbin_ratios)})")
    
    # Save summary
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Print to console
    print('\n'.join(summary_lines))

def generate_error_comparison_files(comparison_df, output_dir):
    """Generate detailed error comparison files for each pointing."""
    
    # Create a subdirectory for error comparisons
    error_dir = os.path.join(output_dir, 'error_comparisons')
    os.makedirs(error_dir, exist_ok=True)
    
    # Process each pointing separately
    for pointing in comparison_df['POINTING'].unique():
        pointing_data = comparison_df[comparison_df['POINTING'] == pointing]
        
        # Create filename for this pointing
        filename = f"error_comparison_{pointing}.txt"
        filepath = os.path.join(error_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("# Error Comparison: SExtractor vs NMAD\n")
            f.write(f"# Pointing: {pointing}\n")
            f.write(f"# Total sources: {pointing_data['SOURCE_ID'].nunique()}\n")
            f.write("#" * 100 + "\n")
            f.write("# Format: SOURCE_ID REDSHIFT_BIN FILTER SEX_FLUX SEX_FLUXERR NMAD_FLUX NMAD_FLUXERR SEX_SNR NMAD_SNR SNR_RATIO\n")
            f.write("#" * 100 + "\n")
            
            # Get unique source IDs for this pointing
            source_ids = sorted(pointing_data['SOURCE_ID'].unique())
            
            for source_id in source_ids:
                source_data = pointing_data[pointing_data['SOURCE_ID'] == source_id]
                redshift_bin = source_data['REDSHIFT_BIN'].iloc[0]  # Same for all filters of same source
                
                f.write(f"\n# Source ID: {source_id}, Redshift Bin: {redshift_bin}\n")
                
                # Write data for each filter
                for filt in filters:
                    filt_data = source_data[source_data['FILTER'] == filt.upper()]
                    if len(filt_data) > 0:
                        row = filt_data.iloc[0]
                        f.write(f"{source_id:8d} {redshift_bin:6s} {filt.upper():6s} "
                               f"{row['SEX_SNR']:10.4f} {row['NMAD_SNR']:10.4f} "
                               f"{row['SNR_RATIO']:10.4f}\n")
            
            # Add summary statistics for this pointing
            f.write("\n" + "#" * 100 + "\n")
            f.write("# SUMMARY STATISTICS FOR THIS POINTING\n")
            f.write("#" * 100 + "\n")
            
            valid_ratios = pointing_data['SNR_RATIO'].dropna()
            valid_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios < 100)]
            
            if len(valid_ratios) > 0:
                f.write(f"Median SNR ratio: {valid_ratios.median():.4f}\n")
                f.write(f"Mean SNR ratio: {valid_ratios.mean():.4f}\n")
                f.write(f"Std SNR ratio: {valid_ratios.std():.4f}\n")
                f.write(f"Min SNR ratio: {valid_ratios.min():.4f}\n")
                f.write(f"Max SNR ratio: {valid_ratios.max():.4f}\n")
                f.write(f"Number of valid comparisons: {len(valid_ratios)}\n")
            
            # Statistics by filter
            f.write("\n# By Filter:\n")
            for filt in filters:
                filt_data = pointing_data[pointing_data['FILTER'] == filt.upper()]
                filt_ratios = filt_data['SNR_RATIO'].dropna()
                filt_ratios = filt_ratios[(filt_ratios > 0) & (filt_ratios < 100)]
                if len(filt_ratios) > 0:
                    f.write(f"{filt.upper()}: median_ratio = {filt_ratios.median():.4f}, n = {len(filt_ratios)}\n")
        
        print(f"Generated error comparison file: {filename}")

def generate_detailed_error_comparison_files(all_comparison_data, output_dir):
    """Generate more detailed error comparison files with flux and flux error values."""
    
    # Create a subdirectory for detailed error comparisons
    detailed_error_dir = os.path.join(output_dir, 'detailed_error_comparisons')
    os.makedirs(detailed_error_dir, exist_ok=True)
    
    # We need to reprocess the data to get flux values
    for pointing in pointings:
        print(f"Generating detailed error file for {pointing}...")
        
        # Create filename for this pointing
        filename = f"detailed_error_comparison_{pointing}.txt"
        filepath = os.path.join(detailed_error_dir, filename)
        
        with open(filepath, 'w') as f:
            # Write comprehensive header
            f.write("# Detailed Error Comparison: SExtractor vs NMAD\n")
            f.write(f"# Pointing: {pointing}\n")
            f.write("#" * 120 + "\n")
            f.write("# Columns: SOURCE_ID REDSHIFT_BIN FILTER SEX_FLUX SEX_FLUXERR NMAD_FLUX NMAD_FLUXERR SEX_SNR NMAD_SNR SNR_RATIO\n")
            f.write("#" * 120 + "\n")
            
            # Collect all data for this pointing
            pointing_data = []
            for redshift_bin in redshift_bins:
                # Read SExtractor data
                sextractor_data = {}
                sex_catalog_found = True
                
                for filt in filters:
                    cat_path = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7', 
                                          f"f150dropout_{filt}_catalog.cat")
                    if not os.path.exists(cat_path):
                        sex_catalog_found = False
                        break
                    
                    data = read_sextractor_catalog(cat_path)
                    if data is None:
                        sex_catalog_found = False
                        break
                    
                    sextractor_data[filt] = data
                
                if not sex_catalog_found:
                    continue
                
                # Read NMAD catalog
                nmad_path = os.path.join(nmad_base_dir, redshift_paths[redshift_bin], 
                                       pointing, 'catalogs', f"{pointing}_{redshift_bin}_nmad_catalogue.cat")
                
                if not os.path.exists(nmad_path):
                    continue
                
                nmad_df = read_nmad_catalog(nmad_path)
                if nmad_df is None or len(nmad_df) == 0:
                    continue
                
                # Process each source
                for _, nmad_row in nmad_df.iterrows():
                    source_id = nmad_row['id']
                    
                    # Find this source in SExtractor F150W data
                    sex_mask = sextractor_data['f150w']['NUMBER'] == source_id
                    if not np.any(sex_mask):
                        continue
                    
                    sex_idx = np.where(sex_mask)[0][0]
                    
                    # For each filter
                    for filt in filters:
                        filt_upper = filt.upper()
                        
                        # Get SExtractor values
                        sex_flux = sextractor_data[filt]['FLUX_AUTO'][sex_idx]
                        sex_fluxerr = sextractor_data[filt]['FLUXERR_AUTO'][sex_idx]
                        sex_snr = sextractor_data[filt]['SNR'][sex_idx]
                        
                        # Get NMAD values
                        nmad_flux = nmad_row[f'f_{filt_upper}']
                        nmad_fluxerr = nmad_row[f'e_{filt_upper}']
                        nmad_snr = nmad_flux / nmad_fluxerr if nmad_fluxerr > 0 else 0
                        
                        # Calculate ratio
                        snr_ratio = sex_snr / nmad_snr if nmad_snr > 0 else np.nan
                        
                        pointing_data.append({
                            'SOURCE_ID': source_id,
                            'REDSHIFT_BIN': redshift_bin,
                            'FILTER': filt_upper,
                            'SEX_FLUX': sex_flux,
                            'SEX_FLUXERR': sex_fluxerr,
                            'NMAD_FLUX': nmad_flux,
                            'NMAD_FLUXERR': nmad_fluxerr,
                            'SEX_SNR': sex_snr,
                            'NMAD_SNR': nmad_snr,
                            'SNR_RATIO': snr_ratio
                        })
            
            # Write data to file
            if pointing_data:
                # Sort by source ID and filter
                pointing_data.sort(key=lambda x: (x['SOURCE_ID'], x['FILTER']))
                
                current_source = None
                for entry in pointing_data:
                    if entry['SOURCE_ID'] != current_source:
                        current_source = entry['SOURCE_ID']
                        f.write(f"\n# Source ID: {current_source}, Redshift Bin: {entry['REDSHIFT_BIN']}\n")
                    
                    f.write(f"{entry['SOURCE_ID']:8d} {entry['REDSHIFT_BIN']:6s} {entry['FILTER']:6s} "
                           f"{entry['SEX_FLUX']:12.6f} {entry['SEX_FLUXERR']:12.6f} "
                           f"{entry['NMAD_FLUX']:12.6f} {entry['NMAD_FLUXERR']:12.6f} "
                           f"{entry['SEX_SNR']:10.4f} {entry['NMAD_SNR']:10.4f} "
                           f"{entry['SNR_RATIO']:10.4f}\n")
                
                # Add summary
                f.write("\n" + "#" * 120 + "\n")
                f.write("# SUMMARY\n")
                f.write("#" * 120 + "\n")
                f.write(f"Total sources: {len(set([x['SOURCE_ID'] for x in pointing_data]))}\n")
                f.write(f"Total measurements: {len(pointing_data)}\n")
                
                # Calculate statistics
                ratios = [x['SNR_RATIO'] for x in pointing_data if not np.isnan(x['SNR_RATIO']) and x['SNR_RATIO'] > 0 and x['SNR_RATIO'] < 100]
                if ratios:
                    f.write(f"Median SNR ratio: {np.median(ratios):.4f}\n")
                    f.write(f"Mean SNR ratio: {np.mean(ratios):.4f}\n")
                    f.write(f"Std SNR ratio: {np.std(ratios):.4f}\n")
                
                print(f"  Generated detailed file with {len(pointing_data)} measurements for {pointing}")
            else:
                f.write("# No data available for this pointing\n")
                print(f"  No data for {pointing}")

# Update the main function to call these new functions
def main():
    """Main analysis function."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting SNR comparison analysis...")
    print(f"Output directory: {output_dir}")
    
    all_comparison_data = []
    all_detailed_data = []  # For storing detailed flux information
    
    # Process each pointing and redshift bin
    for pointing in pointings:
        for redshift_bin in redshift_bins:
            print(f"Processing {pointing} - {redshift_bin}")
            
            # Read all SExtractor catalogs for this pointing
            sextractor_data = {}
            sex_catalog_found = True
            
            for filt in filters:
                cat_path = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7', 
                                      f"f150dropout_{filt}_catalog.cat")
                if not os.path.exists(cat_path):
                    print(f"  Warning: SExtractor catalog not found: {cat_path}")
                    sex_catalog_found = False
                    break
                
                data = read_sextractor_catalog(cat_path)
                if data is None:
                    print(f"  Warning: Could not read SExtractor catalog: {cat_path}")
                    sex_catalog_found = False
                    break
                
                sextractor_data[filt] = data
            
            if not sex_catalog_found:
                continue
            
            # Read NMAD catalog
            nmad_path = os.path.join(nmad_base_dir, redshift_paths[redshift_bin], 
                                   pointing, 'catalogs', f"{pointing}_{redshift_bin}_nmad_catalogue.cat")
            
            if not os.path.exists(nmad_path):
                print(f"  Warning: NMAD catalog not found: {nmad_path}")
                continue
            
            nmad_df = read_nmad_catalog(nmad_path)
            if nmad_df is None or len(nmad_df) == 0:
                print(f"  Warning: No data in NMAD catalog: {nmad_path}")
                continue
            
            # Create comparison data
            comparison_df = create_comparison_data(sextractor_data, nmad_df, pointing, redshift_bin)
            if comparison_df is not None and len(comparison_df) > 0:
                all_comparison_data.append(comparison_df)
                print(f"  Processed {len(nmad_df)} NMAD sources -> {len(comparison_df)} comparisons")
            else:
                print(f"  No matching sources found")
    
    # Combine all data
    if not all_comparison_data:
        print("No data was processed. Check file paths and formats.")
        return
    
    final_comparison_df = pd.concat(all_comparison_data, ignore_index=True)
    
    # Save the combined data
    final_comparison_df.to_csv(os.path.join(output_dir, 'all_snr_comparisons.csv'), index=False)
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    create_histogram_plots(final_comparison_df, output_dir)
    create_one_to_one_plots(final_comparison_df, output_dir)
    
    # Generate error comparison files
    print("Generating error comparison files...")
    generate_error_comparison_files(final_comparison_df, output_dir)
    
    # Generate detailed error comparison files with flux values
    print("Generating detailed error comparison files...")
    generate_detailed_error_comparison_files(all_comparison_data, output_dir)
    
    # Generate summary
    generate_summary_statistics(final_comparison_df, output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
if __name__ == '__main__':
    main()