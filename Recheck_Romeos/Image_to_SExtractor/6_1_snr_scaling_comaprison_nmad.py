import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import ascii
from astropy.table import Table
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration (same as before)
sextractor_base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
eazy_catalog_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Eazy_catalogue/'
output_dir = './6_1_SNR_Comparison_scaling_Analysis'
pointings = [f'nircam{i}' for i in range(1, 11)]

# Brenjit_IDs to analyze (same as before)
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

def read_eazy_catalog(filepath):
    """Reads the EAZY catalog with NMAD flux measurements."""
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
        print(f"Error reading EAZY catalog {filepath}: {e}")
        return None

def calculate_nmad_snr(eazy_data):
    """Calculate SNR from NMAD flux and error measurements."""
    snr_data = {}
    for filt in filters:
        eazy_filt = filter_mapping[filt]
        flux_col = f'f_{eazy_filt}'
        err_col = f'e_{eazy_filt}'
        
        if flux_col in eazy_data.columns and err_col in eazy_data.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = eazy_data[flux_col] / eazy_data[err_col]
            snr[np.isinf(snr) | np.isnan(snr)] = 0
            snr_data[filt] = snr
        else:
            print(f"Warning: Columns {flux_col} or {err_col} not found in EAZY data")
            snr_data[filt] = np.zeros(len(eazy_data))
    
    return snr_data

def create_comparison_catalog(sextractor_data, eazy_data, nmad_snr, pointing, output_dir):
    """Create a comprehensive catalog comparing both SNR calculation methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the source IDs for this pointing
    source_ids = highlight_ids[pointing]
    
    # Initialize the output table
    output_data = []
    
    for src_id in source_ids:
        # Find the source in SExtractor data
        sex_idx = np.where(sextractor_data['f150w']['NUMBER'] == src_id)[0]
        if len(sex_idx) == 0:
            print(f"Source {src_id} not found in SExtractor data for {pointing}")
            continue
        
        sex_idx = sex_idx[0]
        
        # Find the source in EAZY data
        eazy_idx = np.where(eazy_data['id'] == src_id)[0]
        if len(eazy_idx) == 0:
            print(f"Source {src_id} not found in EAZY data for {pointing}")
            continue
        
        eazy_idx = eazy_idx[0]
        
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
                'NMAD_FLUX': eazy_data[f'f_{filter_mapping[filt]}'].iloc[eazy_idx],
                'NMAD_FLUXERR': eazy_data[f'e_{filter_mapping[filt]}'].iloc[eazy_idx],
                'NMAD_SNR': nmad_snr[filt][eazy_idx],
                'SNR_DIFF': sextractor_data[filt]['SNR'][sex_idx] - nmad_snr[filt][eazy_idx],
                'SNR_RATIO': sextractor_data[filt]['SNR'][sex_idx] / nmad_snr[filt][eazy_idx] if nmad_snr[filt][eazy_idx] > 0 else np.nan
            }
            output_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(output_data)
    output_file = os.path.join(output_dir, f'{pointing}_snr_comparison.csv')
    df.to_csv(output_file, index=False)
    
    # Also create a summary file with just the key statistics
    summary_df = df.groupby(['NUMBER', 'FILTER']).agg({
        'SEX_SNR': 'first',
        'NMAD_SNR': 'first',
        'SNR_DIFF': 'first',
        'SNR_RATIO': 'first'
    }).reset_index()
    
    summary_file = os.path.join(output_dir, f'{pointing}_snr_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    return df, summary_df

def perform_linear_fitting(comparison_data, pointing, output_dir):
    """Perform linear regression to find scaling relationship between SExtractor and NMAD SNR."""
    scaling_results = {}
    
    # Create directory for scaling analysis
    scaling_dir = os.path.join(output_dir, pointing, 'scaling_analysis')
    os.makedirs(scaling_dir, exist_ok=True)
    
    # Fit for each filter separately
    for filt in filters:
        filter_data = comparison_data[comparison_data['FILTER'] == filt.upper()]
        
        if len(filter_data) < 2:
            print(f"Not enough data for {filt} in {pointing}")
            continue
        
        # Extract SNR values
        sex_snr = filter_data['SEX_SNR'].values
        nmad_snr = filter_data['NMAD_SNR'].values
        
        # Filter out zeros and very small values
        valid_mask = (sex_snr > 0.1) & (nmad_snr > 0.1) & np.isfinite(sex_snr) & np.isfinite(nmad_snr)
        
        if np.sum(valid_mask) < 2:
            print(f"Not enough valid data for {filt} in {pointing}")
            continue
        
        sex_snr_valid = sex_snr[valid_mask]
        nmad_snr_valid = nmad_snr[valid_mask]
        
        # Perform linear regression in log space (since we expect multiplicative scaling)
        log_sex = np.log10(sex_snr_valid)
        log_nmad = np.log10(nmad_snr_valid)
        
        # Fit linear model: log10(NMAD_SNR) = slope * log10(SEX_SNR) + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sex, log_nmad)
        
        # Convert back to linear space: NMAD_SNR = 10^intercept * (SEX_SNR)^slope
        scaling_factor = 10**intercept
        power_law_exponent = slope
        
        # Also fit simple linear model in linear space: NMAD_SNR = a * SEX_SNR + b
        slope_linear, intercept_linear, r_value_linear, p_value_linear, std_err_linear = stats.linregress(
            sex_snr_valid, nmad_snr_valid
        )
        
        # Calculate goodness of fit metrics
        predicted_log = slope * log_sex + intercept
        predicted_linear = slope_linear * sex_snr_valid + intercept_linear
        
        r2_log = r_value**2
        r2_linear = r_value_linear**2
        
        mae_log = np.mean(np.abs(10**log_nmad - 10**predicted_log))
        mae_linear = np.mean(np.abs(nmad_snr_valid - predicted_linear))
        
        # Store results
        scaling_results[filt] = {
            'log_fit': {
                'slope': slope,
                'intercept': intercept,
                'scaling_factor': scaling_factor,
                'power_law_exponent': power_law_exponent,
                'r_value': r_value,
                'r2': r2_log,
                'p_value': p_value,
                'std_err': std_err,
                'mae': mae_log
            },
            'linear_fit': {
                'slope': slope_linear,
                'intercept': intercept_linear,
                'r_value': r_value_linear,
                'r2': r2_linear,
                'p_value': p_value_linear,
                'std_err': std_err_linear,
                'mae': mae_linear
            },
            'n_points': len(sex_snr_valid)
        }
        
        # Create plot for this filter
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        plt.scatter(sex_snr_valid, nmad_snr_valid, alpha=0.6, s=50, label='Data points')
        
        # Plot 1:1 line
        max_val = max(sex_snr_valid.max(), nmad_snr_valid.max())
        plt.plot([0.1, max_val], [0.1, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        # Plot log fit
        x_fit = np.logspace(np.log10(0.1), np.log10(max_val), 100)
        y_fit_log = scaling_factor * (x_fit ** power_law_exponent)
        plt.plot(x_fit, y_fit_log, 'r-', linewidth=2, 
                label=f'Power law fit: y = {scaling_factor:.3f} * x^{power_law_exponent:.3f}\nR² = {r2_log:.3f}')
        
        # Plot linear fit
        y_fit_linear = slope_linear * x_fit + intercept_linear
        plt.plot(x_fit, y_fit_linear, 'b-', linewidth=2, 
                label=f'Linear fit: y = {slope_linear:.3f}x + {intercept_linear:.3f}\nR² = {r2_linear:.3f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('SExtractor SNR')
        plt.ylabel('NMAD SNR')
        plt.title(f'{pointing} - {filt.upper()}: SNR Scaling Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(scaling_dir, f'{pointing}_{filt}_scaling_fit.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    return scaling_results


def plot_individual_source_comparison(comparison_data, pointing, output_dir):
    """Generate individual plots for each source showing SExtractor vs NMAD SNR across filters."""
    individual_dir = os.path.join(output_dir, pointing, 'individual_sources')
    os.makedirs(individual_dir, exist_ok=True)
    
    # Get unique source IDs
    source_ids = comparison_data['NUMBER'].unique()
    
    for src_id in source_ids:
        src_data = comparison_data[comparison_data['NUMBER'] == src_id]
        
        plt.figure(figsize=(12, 8))
        
        # Plot each filter for this source
        for i, filt in enumerate(filters):
            filter_data = src_data[src_data['FILTER'] == filt.upper()]
            if len(filter_data) > 0:
                plt.scatter(filter_data['SEX_SNR'], filter_data['NMAD_SNR'], 
                          color=plt.cm.tab10(i), s=100, label=filt.upper(), alpha=0.8)
        
        # Add 1:1 line
        max_val = max(src_data['SEX_SNR'].max(), src_data['NMAD_SNR'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
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
    
    # Create a color map for sources
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
    
    # Plot 2: Separate plot for each filter
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


def plot_scaling_summary(scaling_results, pointing, output_dir):
    """Create summary plots of scaling factors across filters."""
    if not scaling_results:
        return
    
    scaling_dir = os.path.join(output_dir, pointing, 'scaling_analysis')
    
    # Extract scaling factors
    filters_plot = []
    scaling_factors = []
    power_law_exponents = []
    linear_slopes = []
    r2_values_log = []
    r2_values_linear = []
    
    for filt, results in scaling_results.items():
        filters_plot.append(filt.upper())
        scaling_factors.append(results['log_fit']['scaling_factor'])
        power_law_exponents.append(results['log_fit']['power_law_exponent'])
        linear_slopes.append(results['linear_fit']['slope'])
        r2_values_log.append(results['log_fit']['r2'])
        r2_values_linear.append(results['linear_fit']['r2'])
    
    # Plot scaling factors
    plt.figure(figsize=(12, 8))
    
    x_pos = np.arange(len(filters_plot))
    
    plt.subplot(2, 1, 1)
    plt.bar(x_pos - 0.2, scaling_factors, width=0.4, label='Scaling Factor (10^intercept)', alpha=0.7)
    plt.bar(x_pos + 0.2, linear_slopes, width=0.4, label='Linear Slope', alpha=0.7)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No scaling (1:1)')
    plt.xticks(x_pos, filters_plot, rotation=45)
    plt.ylabel('Scaling Parameter')
    plt.title(f'{pointing}: Scaling Parameters by Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.bar(x_pos - 0.2, r2_values_log, width=0.4, label='Power Law R²', alpha=0.7)
    plt.bar(x_pos + 0.2, r2_values_linear, width=0.4, label='Linear R²', alpha=0.7)
    plt.xticks(x_pos, filters_plot, rotation=45)
    plt.ylabel('R² Value')
    plt.xlabel('Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(scaling_dir, f'{pointing}_scaling_summary.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot power law exponents
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, power_law_exponents, alpha=0.7)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No scaling (exponent=1)')
    plt.xticks(x_pos, filters_plot, rotation=45)
    plt.ylabel('Power Law Exponent')
    plt.title(f'{pointing}: Power Law Exponents by Filter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(scaling_dir, f'{pointing}_power_law_exponents.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def save_scaling_parameters(scaling_results, pointing, output_dir):
    """Save scaling parameters to a CSV file."""
    scaling_dir = os.path.join(output_dir, pointing, 'scaling_analysis')
    
    scaling_data = []
    for filt, results in scaling_results.items():
        row = {
            'filter': filt.upper(),
            'log_slope': results['log_fit']['slope'],
            'log_intercept': results['log_fit']['intercept'],
            'scaling_factor': results['log_fit']['scaling_factor'],
            'power_law_exponent': results['log_fit']['power_law_exponent'],
            'log_r_value': results['log_fit']['r_value'],
            'log_r2': results['log_fit']['r2'],
            'log_p_value': results['log_fit']['p_value'],
            'log_std_err': results['log_fit']['std_err'],
            'log_mae': results['log_fit']['mae'],
            'linear_slope': results['linear_fit']['slope'],
            'linear_intercept': results['linear_fit']['intercept'],
            'linear_r_value': results['linear_fit']['r_value'],
            'linear_r2': results['linear_fit']['r2'],
            'linear_p_value': results['linear_fit']['p_value'],
            'linear_std_err': results['linear_fit']['std_err'],
            'linear_mae': results['linear_fit']['mae'],
            'n_points': results['n_points']
        }
        scaling_data.append(row)
    
    scaling_df = pd.DataFrame(scaling_data)
    scaling_file = os.path.join(scaling_dir, f'{pointing}_scaling_parameters.csv')
    scaling_df.to_csv(scaling_file, index=False)
    
    return scaling_df

def plot_snr_comparison(comparison_data, pointing, output_dir):
    """Generate plots comparing SNR values from both methods with scaling fits."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform linear fitting analysis
    scaling_results = perform_linear_fitting(comparison_data, pointing, output_dir)
    
    # Save scaling parameters
    if scaling_results:
        scaling_df = save_scaling_parameters(scaling_results, pointing, output_dir)
        plot_scaling_summary(scaling_results, pointing, output_dir)
    
    # Create individual source plots (existing function)
    plot_individual_source_comparison(comparison_data, pointing, output_dir)
    
    # Create all sources comparison plots (existing function)
    plot_all_sources_comparison(comparison_data, pointing, output_dir)
    
    # Filter-based plots with scaling fits
    pointing_data = comparison_data[comparison_data['pointing'] == pointing]
    if len(pointing_data) == 0:
        print(f"No data found for {pointing}")
        return
    
    n_filters = len(filters)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, filt in enumerate(filters):
        filter_data = pointing_data[pointing_data['FILTER'] == filt.upper()]
        
        if len(filter_data) == 0:
            axes[i].set_visible(False)
            continue
        
        # Color by source ID
        source_ids = filter_data['NUMBER'].unique()
        colors = cm.rainbow(np.linspace(0, 1, len(source_ids)))
        
        for j, src_id in enumerate(source_ids):
            src_data = filter_data[filter_data['NUMBER'] == src_id]
            axes[i].scatter(src_data['SEX_SNR'], src_data['NMAD_SNR'], 
                          color=colors[j], s=50, alpha=0.7, label=f'ID {src_id}')
        
        # Add 1:1 line
        max_val = max(filter_data['SEX_SNR'].max(), filter_data['NMAD_SNR'].max())
        axes[i].plot([0.1, max_val], [0.1, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        # Add scaling fit if available
        if filt in scaling_results:
            fit = scaling_results[filt]['log_fit']
            x_fit = np.logspace(np.log10(0.1), np.log10(max_val), 100)
            y_fit = fit['scaling_factor'] * (x_fit ** fit['power_law_exponent'])
            axes[i].plot(x_fit, y_fit, 'r-', linewidth=2, 
                       label=f'Fit: y={fit["scaling_factor"]:.2f}*x^{fit["power_law_exponent"]:.2f}')
        
        axes[i].set_yscale('log')
        axes[i].set_xscale('log')
        axes[i].set_xlabel('SExtractor SNR')
        axes[i].set_ylabel('NMAD SNR')
        axes[i].set_title(f'{filt.upper()} SNR Comparison')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pointing}_snr_comparison_with_fits.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to compare SNR values from SExtractor and NMAD methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting SNR comparison analysis with scaling fits")
    print(f"Output will be saved to: {output_dir}")
    
    all_comparison_data = []
    all_scaling_results = {}
    
    for pointing in pointings:
        print(f"\nProcessing {pointing}")
        
        # Skip if no highlight IDs for this pointing
        if pointing not in highlight_ids or len(highlight_ids[pointing]) == 0:
            print(f"No highlight IDs for {pointing}, skipping")
            continue
        
        # Load SExtractor catalogs (existing code)
        # Load EAZY catalog (existing code)
        # Calculate NMAD SNR (existing code)
        # Create comparison catalog (existing code)
        
        # Add scaling analysis
        scaling_results = perform_linear_fitting(comparison_df, pointing, output_dir)
        if scaling_results:
            all_scaling_results[pointing] = scaling_results
            save_scaling_parameters(scaling_results, pointing, output_dir)
            plot_scaling_summary(scaling_results, pointing, output_dir)
        
        # Add pointing information to the dataframe
        comparison_df['pointing'] = pointing
        all_comparison_data.append(comparison_df)
        
        # Generate plots with scaling fits
        plot_snr_comparison(comparison_df, pointing, os.path.join(output_dir, pointing))
        
        print(f"Processed {len(highlight_ids[pointing])} sources for {pointing}")
    
    # Combine all data and create overall summary
    if all_comparison_data:
        all_data = pd.concat(all_comparison_data, ignore_index=True)
        all_data.to_csv(os.path.join(output_dir, 'all_pointings_snr_comparison.csv'), index=False)
        
        # Create overall scaling summary
        if all_scaling_results:
            overall_scaling_data = []
            for pointing, scaling_dict in all_scaling_results.items():
                for filt, results in scaling_dict.items():
                    row = {
                        'pointing': pointing,
                        'filter': filt.upper(),
                        'scaling_factor': results['log_fit']['scaling_factor'],
                        'power_law_exponent': results['log_fit']['power_law_exponent'],
                        'linear_slope': results['linear_fit']['slope'],
                        'log_r2': results['log_fit']['r2'],
                        'linear_r2': results['linear_fit']['r2'],
                        'n_points': results['n_points']
                    }
                    overall_scaling_data.append(row)
            
            overall_scaling_df = pd.DataFrame(overall_scaling_data)
            overall_scaling_file = os.path.join(output_dir, 'overall_scaling_parameters.csv')
            overall_scaling_df.to_csv(overall_scaling_file, index=False)
            
            # Create summary statistics by filter
            scaling_summary = overall_scaling_df.groupby('filter').agg({
                'scaling_factor': ['mean', 'std', 'count'],
                'power_law_exponent': ['mean', 'std'],
                'linear_slope': ['mean', 'std'],
                'log_r2': ['mean', 'std'],
                'linear_r2': ['mean', 'std']
            }).round(3)
            
            scaling_summary.to_csv(os.path.join(output_dir, 'scaling_summary_by_filter.csv'))
        
        print(f"\nAnalysis complete. Processed {len(all_data)} measurements across {len(pointings)} pointings")
        print(f"Scaling parameters saved for future use")
    else:
        print("No data was processed")

if __name__ == '__main__':
    main()