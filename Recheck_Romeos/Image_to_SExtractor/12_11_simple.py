import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import ascii
import pandas as pd
from scipy import stats

# Configuration
sextractor_base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
nmad_catalog_base_dir = '/Users/brenjithazarika/Desktop/PhD_computer/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/10_nmad_selected/'
output_dir = './12_SNR_Comparison_Analysis_NewSources'
pointings = [f'nircam{i}' for i in range(1, 11)]

# Redshift bins
redshift_bins = ['z7-8', 'z8-10', 'z10-15']
redshift_paths = {
    'z7-8': 'redshift_z7-8/',
    'z8-10': 'redshift_z8-10/', 
    'z10-15': 'redshift_z10-15/'
}

filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
filter_mapping = {filt: filt.upper() for filt in filters}

def read_sextractor_catalog(filepath):
    """Read SExtractor catalog and calculate SNR."""
    try:
        data = ascii.read(filepath, comment='#')
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        snr[np.isinf(snr) | np.isnan(snr)] = 0
        data['SNR'] = snr
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def read_nmad_catalog(filepath):
    """Read NMAD catalog with flux measurements."""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    values = line.split()
                    if len(values) == 19:
                        row = {'id': int(values[0])}
                        for i, filt in enumerate(filters):
                            row[f'f_{filter_mapping[filt]}'] = float(values[1+i*2])
                            row[f'e_{filter_mapping[filt]}'] = float(values[2+i*2])
                        data.append(row)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading NMAD catalog {filepath}: {e}")
        return None

def calculate_nmad_snr(nmad_data):
    """Calculate SNR from NMAD flux and error measurements."""
    snr_data = {}
    for filt in filters:
        flux_col = f'f_{filter_mapping[filt]}'
        err_col = f'e_{filter_mapping[filt]}'
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = nmad_data[flux_col] / nmad_data[err_col]
        snr[np.isinf(snr) | np.isnan(snr)] = 0
        snr_data[filt] = snr
    return snr_data

def create_comparison_data(sextractor_data, nmad_data, nmad_snr, pointing, redshift_bin):
    """Create comparison data between SExtractor and NMAD SNR."""
    output_data = []
    
    for src_id in nmad_data['id'].unique():
        # Find source in SExtractor data
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            continue
        
        sex_idx = sex_idx[0]
        nmad_idx = np.where(nmad_data['id'] == src_id)[0][0]
        
        # Create row for each filter
        for filt in filters:
            sex_snr = sextractor_data[filt]['SNR'][sex_idx]
            nmad_snr_val = nmad_snr[filt][nmad_idx]
            
            output_data.append({
                'NUMBER': src_id,
                'FILTER': filt.upper(),
                'REDSHIFT_BIN': redshift_bin,
                'SEX_SNR': sex_snr,
                'NMAD_SNR': nmad_snr_val,
                'SNR_RATIO': sex_snr / nmad_snr_val if nmad_snr_val > 0 else np.nan
            })
    
    return pd.DataFrame(output_data) if output_data else None

def calculate_median_fit(sex_snr, nmad_snr):
    """Calculate scaling factor between SExtractor and NMAD SNR."""
    valid_mask = (sex_snr > 0) & (nmad_snr > 0) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
    if np.sum(valid_mask) == 0:
        return None
    
    log_sex = np.log10(sex_snr[valid_mask])
    log_nmad = np.log10(nmad_snr[valid_mask])
    
    intercept = np.median(log_nmad) - np.median(log_sex)
    
    return {
        'intercept': intercept,
        'scale_factor': 10**intercept,
        'n_points': len(log_sex)
    }

def plot_comparison(comparison_data, output_dir):
    """Create main comparison plot."""
    if comparison_data is None or len(comparison_data) == 0:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Color by filter
    for i, filt in enumerate(filters):
        filter_data = comparison_data[comparison_data['FILTER'] == filt.upper()]
        if len(filter_data) > 0:
            plt.scatter(filter_data['SEX_SNR'], filter_data['NMAD_SNR'], 
                       alpha=0.7, label=filt.upper(), s=30)
    
    max_val = max(comparison_data['SEX_SNR'].max(), comparison_data['NMAD_SNR'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
    
    # Calculate and plot fit
    sex_all = comparison_data['SEX_SNR'].values
    nmad_all = comparison_data['NMAD_SNR'].values
    fit_result = calculate_median_fit(sex_all, nmad_all)
    
    if fit_result:
        x_fit = np.linspace(0, max_val, 100)
        y_fit = fit_result['scale_factor'] * x_fit
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Scale: {fit_result["scale_factor"]:.3f}x')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('SExtractor SNR')
    plt.ylabel('NMAD SNR')
    plt.title(f'SNR Comparison: {comparison_data["NUMBER"].nunique()} Sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'snr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_results(all_comparison_data, output_dir):
    """Save all results and calculate combined scaling."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all data
    all_data = pd.concat([data for data in all_comparison_data.values() if data is not None])
    all_data.to_csv(os.path.join(output_dir, 'all_snr_comparison.csv'), index=False)
    
    # Calculate combined scaling
    print("\n" + "="*50)
    print("COMBINED SCALING ANALYSIS")
    print("="*50)
    
    fit_results = []
    summary_lines = []
    
    for filt in filters:
        filter_data = all_data[all_data['FILTER'] == filt.upper()]
        if len(filter_data) > 0:
            fit_result = calculate_median_fit(filter_data['SEX_SNR'].values, 
                                            filter_data['NMAD_SNR'].values)
            if fit_result:
                fit_results.append(fit_result['scale_factor'])
                summary_lines.append(f"{filt.upper()}: {fit_result['scale_factor']:.5f}")
    
    if fit_results:
        avg_scale = np.mean(fit_results)
        std_scale = np.std(fit_results)
        
        summary_lines.append(f"\nOverall scaling: {avg_scale:.5f} ± {std_scale:.5f}")
        summary_lines.append(f"Based on {len(fit_results)} filters")
        
        # Save summary
        with open(os.path.join(output_dir, "scaling_summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))
        
        print(f"Average scaling factor: {avg_scale:.5f}")
        print(f"Standard deviation: {std_scale:.5f}")
        print(f"Total sources: {all_data['NUMBER'].nunique()}")
        print(f"Total measurements: {len(all_data)}")
    
    return all_data

def main():
    """Main analysis function."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting SNR comparison analysis")
    print(f"Output directory: {output_dir}")
    
    all_comparison_data = {}
    
    for pointing in pointings:
        for redshift_bin in redshift_bins:
            print(f"Processing {pointing} - {redshift_bin}")
            
            # Read SExtractor data
            sextractor_data = {}
            for filt in filters:
                cat_path = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7', 
                                      f"f150dropout_{filt}_catalog.cat")
                data = read_sextractor_catalog(cat_path)
                if data is None:
                    print(f"  Missing {filt} data")
                    break
                sextractor_data[filt] = data
            else:
                # Read NMAD data
                base_dir = os.path.join(nmad_catalog_base_dir, redshift_paths[redshift_bin])
                catalog_dir = os.path.join(base_dir, pointing, 'catalogs')
                nmad_file = os.path.join(catalog_dir, f"{pointing}_z10-15_nmad_catalogue.cat")
                
                if os.path.exists(nmad_file):
                    nmad_data = read_nmad_catalog(nmad_file)
                    if nmad_data is not None:
                        nmad_snr = calculate_nmad_snr(nmad_data)
                        comparison_df = create_comparison_data(sextractor_data, nmad_data, 
                                                             nmad_snr, pointing, redshift_bin)
                        if comparison_df is not None:
                            all_comparison_data[(pointing, redshift_bin)] = comparison_df
                            print(f"  Processed {len(nmad_data)} sources")
                        continue
            
            print(f"  No valid data for {pointing} {redshift_bin}")
    
    # Save results and create plots
    if all_comparison_data:
        all_data = save_results(all_comparison_data, output_dir)
        plot_comparison(all_data, output_dir)
        print(f"\nAnalysis complete!")
        print(f"Processed {len(all_data)} measurements")
    else:
        print("No data was processed")

if __name__ == '__main__':
    main()