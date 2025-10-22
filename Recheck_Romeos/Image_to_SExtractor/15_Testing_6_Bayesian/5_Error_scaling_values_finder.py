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
compute scaling factors (Bayesian linear regression)
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
import pymc3 as pm
import arviz as az
import theano.tensor as tt

# Configuration
sextractor_base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
nmad_base_dir = '/home/iit-t/Bren_jit/tooFAINTtoCARE/Recheck_Romeos/Image_to_SExtractor/15_Testing_3/4_Eazy_catalogue_of_sept_data'
output_dir = './5_SNR_scaling_factor_finder_bayesian'

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

def bayesian_linear_fit(sex_snr, nmad_snr, n_samples=2000, n_tune=1000):
    """
    Bayesian linear regression with slope fixed at 1 to find scaling factor.
    Uses Student-T distribution for robust outlier handling.
    
    Returns: dictionary with fit results and trace
    """
    # Filter out invalid values
    valid_mask = (sex_snr > 0) & (nmad_snr > 0) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
    
    if np.sum(valid_mask) < 10:  # Need minimum number of points
        return None
    
    x_data = sex_snr[valid_mask]
    y_data = nmad_snr[valid_mask]
    
    # Log transform for better behaved distributions
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)
    
    with pm.Model() as model:
        # Priors
        # Scaling factor: we expect it to be around 1 (no scaling)
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Intercept in log space
        
        # Error term - using HalfNormal for positive-only
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Degrees of freedom for Student-T (robust to outliers)
        nu = pm.Gamma('nu', alpha=2, beta=0.1)
        
        # Expected value
        mu = log_x + alpha  # Slope fixed at 1
        
        # Likelihood - Student-T for robust regression
        likelihood = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=log_y)
        
        # Sample
        trace = pm.sample(n_samples, tune=n_tune, chains=2, target_accept=0.9, 
                         return_inferencedata=False)
    
    # Compute summary statistics
    summary = pm.summary(trace)
    
    # Extract scaling factor (10^alpha)
    alpha_samples = trace['alpha']
    scaling_samples = 10**alpha_samples
    
    result = {
        'alpha_mean': np.mean(alpha_samples),
        'alpha_std': np.std(alpha_samples),
        'scaling_mean': np.mean(scaling_samples),
        'scaling_std': np.std(scaling_samples),
        'scaling_median': np.median(scaling_samples),
        'scaling_mcse': pm.stats.mcse(scaling_samples),
        'scaling_hdi_95': az.hdi(scaling_samples, hdi_prob=0.95),
        'sigma_mean': np.mean(trace['sigma']),
        'nu_mean': np.mean(trace['nu']),
        'n_points': len(x_data),
        'r_hat': summary.loc['alpha', 'r_hat'],
        'trace': trace,
        'model': model
    }
    
    return result

def calculate_median_fit(sex_snr, nmad_snr):
    """
    Legacy function for backward compatibility.
    Calculates the intercept for a line with slope 1 that passes through the median.
    """
    valid_mask = (sex_snr > 0) & (nmad_snr > 0) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
    
    if np.sum(valid_mask) == 0:
        return None
    
    sex_valid = sex_snr[valid_mask]
    nmad_valid = nmad_snr[valid_mask]
    
    log_sex_snr = np.log10(sex_valid)
    log_nmad_snr = np.log10(nmad_valid)
    
    median_log_x = np.median(log_sex_snr)
    median_log_y = np.median(log_nmad_snr)
    
    intercept = median_log_y - median_log_x
    
    return {
        'slope': 1.0,
        'intercept': intercept,
        'n_points': len(sex_valid)
    }

def plot_bayesian_results(sex_snr, nmad_snr, bayesian_result, pointing, filter_name, output_path):
    """Plot Bayesian fitting results with credible intervals."""
    valid_mask = (sex_snr > 0) & (nmad_snr > 0) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
    x_data = sex_snr[valid_mask]
    y_data = nmad_snr[valid_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Data with fits
    ax1.scatter(x_data, y_data, alpha=0.6, s=30, color='blue', label='Data points')
    
    # Generate prediction lines
    x_plot = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
    
    # Bayesian fit
    scaling = bayesian_result['scaling_mean']
    y_bayes = scaling * x_plot
    ax1.plot(x_plot, y_bayes, 'r-', linewidth=2, 
             label=f'Bayesian: y = {scaling:.3f} x')
    
    # 1:1 line
    ax1.plot(x_plot, x_plot, 'k--', alpha=0.7, label='1:1 line')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('SExtractor SNR')
    ax1.set_ylabel('NMAD SNR')
    ax1.set_title(f'{pointing} - {filter_name}\nBayesian Scaling: {scaling:.3f} ± {bayesian_result["scaling_std"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Posterior distribution of scaling factor
    scaling_samples = 10**bayesian_result['trace']['alpha']
    
    ax2.hist(scaling_samples, bins=50, density=True, alpha=0.7, color='green')
    ax2.axvline(bayesian_result['scaling_mean'], color='red', linestyle='--', 
                label=f'Mean: {bayesian_result["scaling_mean"]:.3f}')
    ax2.axvline(bayesian_result['scaling_median'], color='orange', linestyle='--',
                label=f'Median: {bayesian_result["scaling_median"]:.3f}')
    
    hdi = bayesian_result['scaling_hdi_95']
    ax2.axvspan(hdi[0], hdi[1], alpha=0.2, color='red', label=f'95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]')
    
    ax2.set_xlabel('Scaling Factor')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Posterior Distribution of Scaling Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

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
        
        # Use Bayesian fit
        bayes_result = bayesian_linear_fit(sex_all, nmad_all)
        if bayes_result:
            x_fit = np.linspace(0, max_val, 100)
            y_fit = bayes_result['scaling_mean'] * x_fit
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Bayesian Fit: y = {bayes_result["scaling_mean"]:.3f}x')
        
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
    
    # Use Bayesian fit
    bayes_result = bayesian_linear_fit(sex_all, nmad_all)
    if bayes_result:
        x_fit = np.linspace(0, max_val, 100)
        y_fit = bayes_result['scaling_mean'] * x_fit
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Bayesian Fit: y = {bayes_result["scaling_mean"]:.3f}x')
    
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

def plot_snr_comparison(comparison_data, pointing, output_dir):
    """Generate plots comparing SNR values from both methods."""
    if comparison_data is None or len(comparison_data) == 0:
        print(f"No comparison data for {pointing}")
        return
    
    plot_individual_source_comparison(comparison_data, pointing, output_dir)
    plot_all_sources_comparison(comparison_data, pointing, output_dir)
    
    # Create Bayesian fit plots for each filter
    bayesian_dir = os.path.join(output_dir, pointing, 'bayesian_fits')
    os.makedirs(bayesian_dir, exist_ok=True)
    
    for filt in filters:
        filter_data = comparison_data[comparison_data['FILTER'] == filt.upper()]
        if len(filter_data) == 0:
            continue
            
        sex_snr = filter_data['SEX_SNR'].values
        nmad_snr = filter_data['NMAD_SNR'].values
        
        bayes_result = bayesian_linear_fit(sex_snr, nmad_snr)
        if bayes_result:
            plot_path = os.path.join(bayesian_dir, f'{pointing}_{filt}_bayesian_fit.png')
            plot_bayesian_results(sex_snr, nmad_snr, bayes_result, pointing, filt.upper(), plot_path)

def save_bayesian_fit_summary(all_comparison_data, output_dir):
    """
    Save comprehensive Bayesian fit summary with uncertainties.
    """
    summary_lines = []
    all_fits = []
    
    summary_lines.append("Bayesian SNR Scaling Factor Summary")
    summary_lines.append("=" * 60)
    summary_lines.append("Model: NMAD_SNR = scaling_factor * SEX_SNR")
    summary_lines.append("Prior: scaling_factor ~ LogNormal(0, 1)")
    summary_lines.append("Likelihood: Student-T (robust to outliers)")
    summary_lines.append("")
    
    for pointing, comp_data in all_comparison_data.items():
        summary_lines.append(f"\nPointing: {pointing}")
        summary_lines.append("-" * 40)
        
        pointing_fits = []
        
        for filt in filters:
            filter_data = comp_data[comp_data['FILTER'] == filt.upper()]
            sex_snr = filter_data['SEX_SNR'].values
            nmad_snr = filter_data['NMAD_SNR'].values
            
            bayes_result = bayesian_linear_fit(sex_snr, nmad_snr)
            
            if bayes_result:
                scaling = bayes_result['scaling_mean']
                scaling_std = bayes_result['scaling_std']
                hdi = bayes_result['scaling_hdi_95']
                n_points = bayes_result['n_points']
                r_hat = bayes_result['r_hat']
                
                summary_lines.append(
                    f"  {filt.upper():8s}: {scaling:.5f} ± {scaling_std:.5f} "
                    f"(95% HDI: [{hdi[0]:.5f}, {hdi[1]:.5f}]) "
                    f"n={n_points:3d} R_hat={r_hat:.3f}"
                )
                
                pointing_fits.append(scaling)
                all_fits.append({
                    'pointing': pointing,
                    'filter': filt.upper(),
                    'scaling_mean': scaling,
                    'scaling_std': scaling_std,
                    'scaling_median': bayes_result['scaling_median'],
                    'hdi_lower': hdi[0],
                    'hdi_upper': hdi[1],
                    'n_points': n_points,
                    'r_hat': r_hat,
                    'sigma': bayes_result['sigma_mean'],
                    'nu': bayes_result['nu_mean']
                })
            else:
                summary_lines.append(f"  {filt.upper():8s}: NA - insufficient data")
                pointing_fits.append(np.nan)
        
        # Calculate pointing average with sigma clipping
        valid_fits = [f for f in pointing_fits if not np.isnan(f)]
        if len(valid_fits) > 0:
            clipped, lower, upper = stats.sigmaclip(valid_fits, low=3, high=3)
            avg_scaling = np.mean(clipped)
            summary_lines.append(f"  Average (sigma-clipped): {avg_scaling:.5f}")
        else:
            summary_lines.append(f"  Average (sigma-clipped): NA")
    
    # Save detailed summary
    output_file = os.path.join(output_dir, "bayesian_snr_fit_summary.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(summary_lines))
    
    # Save as CSV for easier analysis
    if all_fits:
        fits_df = pd.DataFrame(all_fits)
        csv_file = os.path.join(output_dir, "bayesian_scaling_factors.csv")
        fits_df.to_csv(csv_file, index=False)
        
        # Overall statistics
        overall_avg = fits_df['scaling_mean'].mean()
        overall_std = fits_df['scaling_mean'].std()
        overall_median = fits_df['scaling_mean'].median()
        
        stats_lines = [
            "\n\nOverall Statistics:",
            f"Mean scaling factor: {overall_avg:.5f} ± {overall_std:.5f}",
            f"Median scaling factor: {overall_median:.5f}",
            f"Number of measurements: {len(fits_df)}"
        ]
        
        with open(output_file, "a") as f:
            f.write("\n".join(stats_lines))
    
    print(f"Saved Bayesian fit summary to {output_file}")

def main():
    """Main function to compare SNR values from SExtractor and NMAD methods using Bayesian inference."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Bayesian SNR comparison analysis with NMAD data")
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
        
        # Save Bayesian fit summary
        save_bayesian_fit_summary(all_comparison_data, output_dir)
        
        print(f"\nBayesian analysis complete. Processed {len(all_data)} measurements across {len(all_comparison_data)} pointings")
        print(f"Check {output_dir}/bayesian_snr_fit_summary.txt for detailed results with uncertainties")
    else:
        print("No data was processed")

if __name__ == '__main__':
    main()