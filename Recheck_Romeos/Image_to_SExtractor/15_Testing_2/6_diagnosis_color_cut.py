# Same code as 6_2_py_original_Diagnostic_Analysis_scaled.py
# Just removed the color_cut_3
# cut3 = color1 > (0.5 + 0.44 * (color2 + 0.8) + 0.5)  # Additional color cut

# Diagnosis criteria from the paper
# Sextractor errors are scaled based on the NMAD-SExtractor SNR scaling
# The scaling values are in the 6_1_SNR_comparison_analysis/snr_fit_summary..txt
# The code is inpired by 3_z_Diagnosis

import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import ascii
import os
from astropy.table import Table
from datetime import datetime
import pandas as pd
from scipy import stats
print("Running 6_2 without 410m final summaruy")
# Configuration (same as your original)
base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
output_base_dir = './6_Diagnostic_Analysis_scaled_NMAD'
pointings = [f'nircam{i}' for i in range(1, 11)]
catalog_subdir = 'catalogue_z7'
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

# Path to the new scaling factors file
scaling_file_path = '/home/iit-t/Bren_jit/tooFAINTtoCARE/Recheck_Romeos/Image_to_SExtractor/15_Testing_2/5_SNR_scaling_factor_finder/snr_fit_summary.txt'

# Brenjit_IDs to highlight (same as your original)
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

def parse_scaling_factors(file_path):
    """Parse scaling factors from the snr_fit_summary.txt file."""
    scaling_factors = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        current_pointing = None
        
        for line in lines:
            line = line.strip()
            
            # Check for pointing line
            if line.startswith('Pointing:'):
                current_pointing = line.split(':')[1].strip()
                
                if current_pointing not in scaling_factors:
                    scaling_factors[current_pointing] = {}
            
            # Check for filter scaling factors
            elif line.startswith('F') and ':' in line:
                if current_pointing:
                    filter_part, value_part = line.split(':', 1)
                    filter_name = filter_part.strip().lower()
                    try:
                        scaling_value = float(value_part.strip())
                        scaling_factors[current_pointing][filter_name] = scaling_value
                    except ValueError:
                        print(f"Warning: Could not parse scaling value for {filter_name}: {value_part}")
            
            # Check for average line (we can use this as fallback)
            elif line.startswith('Average') and current_pointing:
                if 'average' not in scaling_factors[current_pointing]:
                    try:
                        avg_value = float(line.split(':')[1].strip())
                        scaling_factors[current_pointing]['average'] = avg_value
                    except ValueError:
                        pass
        
        print(f"Successfully parsed scaling factors for {len(scaling_factors)} pointings")
        return scaling_factors
        
    except Exception as e:
        print(f"Error parsing scaling factors file: {e}")
        return {}

def get_scaling_factor(scaling_factors, pointing, filter_name):
    """Get the scaling factor for a specific pointing and filter."""
    try:
        if (pointing in scaling_factors and 
            filter_name in scaling_factors[pointing]):
            return scaling_factors[pointing][filter_name]
        else:
            # Fallback: use average for the pointing
            if (pointing in scaling_factors and 
                'average' in scaling_factors[pointing]):
                print(f"Using average scaling factor for {pointing} {filter_name}")
                return scaling_factors[pointing]['average']
            else:
                # Final fallback: use 1.0 (no scaling)
                print(f"Warning: No scaling factor found for {pointing} {filter_name}, using 1.0")
                return 1.0
    except Exception as e:
        print(f"Error getting scaling factor for {pointing} {filter_name}: {e}")
        return 1.0

def setup_logging(output_dir):
    """Set up logging to both console and file."""
    log_file = os.path.join(output_dir, f"diagnostic_analysis_SNR_gt_4_veto_lst_4.txt")
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
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC ANALYSIS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

def read_sextractor_catalog(filepath, pointing, filter_name, scaling_factors):
    """Reads a SExtractor catalog file and applies scaling to get NMAD-equivalent SNR."""
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
        
        # Calculate original SExtractor SNR
        with np.errstate(divide='ignore', invalid='ignore'):
            original_snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        original_snr[np.isinf(original_snr) | np.isnan(original_snr)] = 0
        
        # Apply scaling to get NMAD-equivalent SNR
        scale_factor = get_scaling_factor(scaling_factors, pointing, filter_name)
        data['SNR'] = original_snr * scale_factor
        
        print(f"Applied scaling factor {scale_factor:.6f} to {filter_name} SNR for {pointing}")
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

def get_mags_single(filter_data, idx, filter_list):
    """Get magnitudes for a single source."""
    mags = {}
    for filt in filter_list:
        mag_val = filter_data[filt]['MAG_AUTO'][idx]
        if np.isnan(mag_val) or np.isinf(mag_val):
            mag_val = 99.0
        mags[filt] = mag_val
    return mags

def apply_f115w_selection(data_dict):
    """Applies F115W-dropout (z=8.5-12) selection criteria and returns a boolean mask."""
    # SNR detection in red filters
    sn_detect = (data_dict['f150w']['SNR'] > 4) & \
                (data_dict['f200w']['SNR'] > 4) & \
                (data_dict['f277w']['SNR'] > 4) & \
                (data_dict['f356w']['SNR'] > 4) & \
                (data_dict['f444w']['SNR'] > 4)     #removed (data_dict['f410m']['SNR'] > 5) & \
    # Veto in blue filters (no detection)
    veto = (data_dict['f814w']['SNR'] < 4) & (data_dict['f606w']['SNR'] < 4)
    
    # Get colors
    mags = get_mags(data_dict, ['f115w', 'f150w', 'f277w'])
    color1 = mags['f115w'] - mags['f150w']  # F115W - F150W
    color2 = mags['f150w'] - mags['f277w']  # F150W - F277W
    
    # Color cuts
    cut1 = (data_dict['f115w']['SNR'] < 2) | (color1 > 0.1)  # No detection or red color
    cut2 = (color2 > -1.5) & (color2 < 1.0)  # Reasonable continuum slope
    #cut3 = color1 > (0.5 + 0.44 * (color2 + 0.8) + 0.5)  # Additional color cut
    
    # Combine all criteria
    mask = sn_detect & veto & cut1 & cut2
    print(f"Found {np.sum(mask)} candidates.")
    return mask

def plot_snr_values(filter_data, pointing, highlight_ids, output_dir, selection_mask=None):
    """Generate SNR plots with Brenjit_ID sources marked with green outline if they meet criteria."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get source IDs from f150w catalog (reference)
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Calculate selection criteria if not provided
    if selection_mask is None:
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
        print("Recovered IDs:", ", ".join(map(str, source_ids[brenjit_meet_criteria])))
    if np.sum(brenjit_fail_criteria) > 0:
        print("Missed IDs:", ", ".join(map(str, source_ids[brenjit_fail_criteria])))
    
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
                
                plt.annotate(str(src_id), 
                            (x, y),
                            textcoords="offset points",
                            xytext=(10, 10),
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

def analyze_missed_galaxies_detailed(filter_data, highlight_ids, pointing, selection_mask=None):
    """Detailed analysis of why specific galaxies are being missed."""
    
    print(f"\n{'='*60}")
    print(f"DETAILED DIAGNOSTIC ANALYSIS FOR {pointing.upper()}")
    print(f"{'='*60}")
    
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Apply current selection criteria if not provided
    if selection_mask is None:
        selection_mask = apply_f115w_selection(filter_data)
    
    # Identify which sources meet/don't meet criteria
    brenjit_meet_criteria = highlight_mask & selection_mask
    brenjit_fail_criteria = highlight_mask & ~selection_mask
    
    # Create detailed analysis table
    analysis_results = []
    
    for gal_id in highlight_ids:
        if gal_id not in source_ids:
            analysis_results.append({
                'ID': gal_id,
                'Status': 'Not found in catalog',
                'Reason': 'Source ID not present'
            })
            continue
            
        idx = np.where(source_ids == gal_id)[0][0]
        meets_criteria = gal_id in source_ids[brenjit_meet_criteria]
        
        # Check each criterion in detail
        reasons = []
        details = {}
        
        # 1. SNR Detection criteria (>4)
        detection_filters = ['f150w', 'f200w', 'f277w', 'f356w', 'f444w']
        detection_snrs = {}
        detection_failed = []
        for filt in detection_filters:
            snr_val = filter_data[filt]['SNR'][idx]
            detection_snrs[filt] = snr_val
            if snr_val <= 4:
                detection_failed.append(f"{filt}({snr_val:.1f})")
        
        if detection_failed:
            reasons.append(f"Low SNR in detection bands: {', '.join(detection_failed)}")
        
        # 2. Veto criteria (must BOTH be < 4)
        veto_filters = ['f606w', 'f814w']
        veto_snrs = {}
        veto_failed = []
        for filt in veto_filters:
            snr_val = filter_data[filt]['SNR'][idx]
            veto_snrs[filt] = snr_val
            if snr_val >= 4:
                veto_failed.append(f"{filt}({snr_val:.1f})")
        
        if veto_failed:
            reasons.append(f"High SNR in veto bands: {', '.join(veto_failed)}")
        
        # 3. Color criteria
        mags = get_mags_single(filter_data, idx, ['f115w', 'f150w', 'f277w'])
        color1 = mags['f115w'] - mags['f150w']
        color2 = mags['f150w'] - mags['f277w']
        f115w_snr = filter_data['f115w']['SNR'][idx]
        
        # Color criterion 1: (f115w SNR < 2) OR (color1 > 0.1)
        color1_ok = (f115w_snr < 2) or (color1 > 0.1)
        if not color1_ok:
            reasons.append(f"Color1 failed: f115w SNR={f115w_snr:.1f}, color1={color1:.2f}")
        
        # Color criterion 2: -1.5 < color2 < 1.0
        color2_ok = (-1.5 < color2 < 1.0)
        if not color2_ok:
            reasons.append(f"Color2 out of range: {color2:.2f} not in (-1.5, 1.0)")
        
        # Compile results
        status = "PASS" if meets_criteria else "FAIL"
        if not reasons and not meets_criteria:
            reasons = ["Unknown reason (check implementation)"]
        
        analysis_results.append({
            'ID': gal_id,
            'Status': status,
            'Reasons': '; '.join(reasons) if reasons else 'All criteria passed',
            'f115w_SNR': f115w_snr,
            'f150w_SNR': detection_snrs['f150w'],
            'f200w_SNR': detection_snrs['f200w'],
            'f277w_SNR': detection_snrs['f277w'],
            'f356w_SNR': detection_snrs['f356w'],
            #'f410m_SNR': detection_snrs['f410m'],
            'f444w_SNR': detection_snrs['f444w'],
            'f606w_SNR': veto_snrs.get('f606w', 0),
            'f814w_SNR': veto_snrs.get('f814w', 0),
            'Color1': color1,
            'Color2': color2,
        })
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(analysis_results)
    
    # Print summary statistics
    total = len(highlight_ids)
    passed = len(df[df['Status'] == 'PASS'])
    failed = len(df[df['Status'] == 'FAIL'])
    not_found = len(df[df['Status'] == 'Not found in catalog']) if 'Not found in catalog' in df['Status'].values else 0
    
    print(f"\nSUMMARY FOR {pointing}:")
    print(f"Total Brenjit sources: {total}")
    print(f"Pass selection: {passed}")
    print(f"Fail selection: {failed}")
    if not_found > 0:
        print(f"Not found in catalog: {not_found}")
    
    # Print detailed table
    print(f"\nDETAILED ANALYSIS:")
    print("-" * 120)
    for _, row in df.iterrows():
        print(f"ID {row['ID']:4d}: {row['Status']:4s} - {row['Reasons']}")
        if row['Status'] == 'FAIL' and 'Low SNR' in row['Reasons']:
            print(f"      SNR values: f150w={row['f150w_SNR']:5.1f}, f200w={row['f200w_SNR']:5.1f}, "
                  f"f277w={row['f277w_SNR']:5.1f}, f356w={row['f356w_SNR']:5.1f}, "
                  f"f444w={row['f444w_SNR']:5.1f}")
    
    # Create visual diagnostics
    create_diagnostic_plots(filter_data, df, highlight_ids, pointing)
    
    return df

def create_diagnostic_plots(filter_data, analysis_df, highlight_ids, pointing):
    """Create diagnostic plots for visual analysis."""
    
    output_dir = os.path.join(output_base_dir, pointing, 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Plot 1: SNR distribution comparison
    plt.figure(figsize=(12, 8))
    
    detection_filters = ['f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
    colors = plt.cm.Set3(np.linspace(0, 1, len(detection_filters)))
    
    for i, filt in enumerate(detection_filters):
        snr_values = filter_data[filt]['SNR']
        plt.scatter(source_ids[highlight_mask], snr_values[highlight_mask], 
                   color=colors[i], alpha=0.7, label=filt, s=50)
    
    plt.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='SNR=5 threshold')
    plt.yscale('log')
    plt.xlabel('Source ID')
    plt.ylabel('SNR (log scale)')
    plt.title(f'{pointing}: SNR in Detection Bands for Brenjit Sources')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pointing}_detection_snr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Color-color diagram with criteria
    plt.figure(figsize=(10, 8))
    
    # Get colors for all sources
    mags_all = get_mags(filter_data, ['f115w', 'f150w', 'f277w'])
    color1_all = mags_all['f115w'] - mags_all['f150w']
    color2_all = mags_all['f150w'] - mags_all['f277w']
    
    # Plot all sources (background)
    plt.scatter(color2_all, color1_all, alpha=0.1, s=10, color='gray', label='All sources')
    
    # Plot Brenjit sources
    for gal_id in highlight_ids:
        if gal_id in source_ids:
            idx = np.where(source_ids == gal_id)[0][0]
            row = analysis_df[analysis_df['ID'] == gal_id].iloc[0]
            
            color1 = row['Color1']
            color2 = row['Color2']
            
            if row['Status'] == 'PASS':
                marker = 'o'
                color = 'green'
                label = 'Pass'
            else:
                marker = 'x'
                color = 'red'
                label = 'Fail'
            
            plt.scatter(color2, color1, marker=marker, color=color, s=100, 
                       label=label if gal_id == highlight_ids[0] else "")
            plt.annotate(str(gal_id), (color2, color1), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    # Plot selection boundaries
    plt.axvline(x=-1.5, color='orange', linestyle=':', linewidth=2, label='Color2 lower limit')
    plt.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, label='Color2 upper limit')
    plt.axhline(y=0.1, color='purple', linestyle=':', linewidth=2, label='Color1 limit')
    
    plt.xlabel('F150W - F277W')
    plt.ylabel('F115W - F150W')
    plt.title(f'{pointing}: Color-Color Diagram for Brenjit Sources')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pointing}_color_color.png'), dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_report(all_analysis_results):
    """Generate a comprehensive summary report across all pointings."""
    
    summary_file = os.path.join(output_base_dir, f"diagnostic_summary_report.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"COMPREHENSIVE DIAGNOSTIC SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        total_galaxies = 0
        total_passed = 0
        total_failed = 0
        
        failure_reasons = {}
        
        for pointing, df in all_analysis_results.items():
            f.write(f"\n{pointing.upper()}:\n")
            f.write("-" * 30 + "\n")
            
            passed = len(df[df['Status'] == 'PASS'])
            failed = len(df[df['Status'] == 'FAIL'])
            not_found = len(df[df['Status'] == 'Not found in catalog']) if 'Not found in catalog' in df['Status'].values else 0
            
            f.write(f"Passed: {passed}, Failed: {failed}, Not found: {not_found}\n")
            
            total_galaxies += (passed + failed + not_found)
            total_passed += passed
            total_failed += failed
            
            # Analyze failure reasons
            for _, row in df.iterrows():
                if row['Status'] == 'FAIL':
                    reasons = row['Reasons'].split('; ')
                    for reason in reasons:
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        f.write(f"\nOVERALL SUMMARY:\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total galaxies analyzed: {total_galaxies}\n")
        f.write(f"Total passed selection: {total_passed}\n")
        f.write(f"Total failed selection: {total_failed}\n")
        if total_galaxies > 0:
            f.write(f"Overall success rate: {total_passed/total_galaxies*100:.1f}%\n")
        
        f.write(f"\nFAILURE REASONS (by frequency):\n")
        f.write("=" * 30 + "\n")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{reason}: {count} galaxies\n")
    
    print(f"Summary report saved to: {summary_file}")

def create_filtered_catalogs(filter_data, selection_mask, pointing, output_base_dir):
    """Create filtered catalogs for sources that meet the selection criteria."""
    catalog_dir = os.path.join(output_base_dir, 'catalogue_lyman_filtered_final', pointing)
    os.makedirs(catalog_dir, exist_ok=True)
    
    print(f"Creating filtered catalogs for {pointing} in {catalog_dir}")
    
    # For each filter, create a filtered catalog
    for filt in filters:
        if filt not in filter_data:
            continue
            
        # Get the original catalog
        catalog = filter_data[filt]
        
        # Apply the selection mask
        filtered_catalog = catalog[selection_mask]
        
        # Define the output path
        output_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
        
        # Write the filtered catalog
        try:
            ascii.write(filtered_catalog, output_path, format='commented_header')
            print(f"  Created {output_path} with {len(filtered_catalog)} sources")
        except Exception as e:
            print(f"  Error writing {output_path}: {e}")
            
def generate_final_summary(all_analysis_results):
    """Generate a final summary table with counts and percentages and detailed criteria failure information."""
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    
    total_selected = 0
    total_galaxies = 0
    
    # Initialize criteria counters and detailed failure information
    criteria_failures = {
        'Low SNR in detection bands': {'count': 0, 'details': {}},
        'High SNR in veto bands': {'count': 0, 'details': {}},
        'Color1 failed': {'count': 0, 'details': {}},
        'Color2 out of range': {'count': 0, 'details': {}},
        'Not found in catalog': {'count': 0, 'details': {}}
    }
    
    # Print header
    print(f"{'Pointing':<10} {'Selected':<10} {'Total':<10} {'Percentage':<10} {'Criteria Failed':<20}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
    
    # Process each pointing
    for pointing, df in all_analysis_results.items():
        total = len(df)
        selected = len(df[df['Status'] == 'PASS'])
        percentage = (selected / total) * 100 if total > 0 else 0
        
        # Count criteria failures for this pointing
        pointing_failures = {
            'Low SNR in detection bands': 0,
            'High SNR in veto bands': 0,
            'Color1 failed': 0,
            'Color2 out of range': 0,
            'Not found in catalog': 0
        }
        
        for _, row in df.iterrows():
            if row['Status'] == 'FAIL':
                reasons = row['Reasons'].split('; ')
                for reason in reasons:
                    # Categorize the failure reason and store details with pointing info
                    if 'Low SNR in detection bands' in reason:
                        pointing_failures['Low SNR in detection bands'] += 1
                        criteria_failures['Low SNR in detection bands']['count'] += 1
                        # Extract which filters failed
                        failed_filters = []
                        if row['f150w_SNR'] <= 4: failed_filters.append(f"F150W({row['f150w_SNR']:.1f})")
                        if row['f200w_SNR'] <= 4: failed_filters.append(f"F200W({row['f200w_SNR']:.1f})")
                        if row['f277w_SNR'] <= 4: failed_filters.append(f"F277W({row['f277w_SNR']:.1f})")
                        if row['f356w_SNR'] <= 4: failed_filters.append(f"F356W({row['f356w_SNR']:.1f})")
                        if row['f444w_SNR'] <= 4: failed_filters.append(f"F444W({row['f444w_SNR']:.1f})")
                        # Store with pointing information
                        key = f"{pointing}_{row['ID']}"
                        criteria_failures['Low SNR in detection bands']['details'][key] = {
                            'pointing': pointing,
                            'id': row['ID'],
                            'details': failed_filters
                        }
                        
                    elif 'High SNR in veto bands' in reason:
                        pointing_failures['High SNR in veto bands'] += 1
                        criteria_failures['High SNR in veto bands']['count'] += 1
                        # Extract which veto filters failed
                        failed_veto = []
                        if row['f606w_SNR'] >= 4: failed_veto.append(f"F606W({row['f606w_SNR']:.1f})")
                        if row['f814w_SNR'] >= 4: failed_veto.append(f"F814W({row['f814w_SNR']:.1f})")
                        # Store with pointing information
                        key = f"{pointing}_{row['ID']}"
                        criteria_failures['High SNR in veto bands']['details'][key] = {
                            'pointing': pointing,
                            'id': row['ID'],
                            'details': failed_veto
                        }
                        
                    elif 'Color1 failed' in reason:
                        pointing_failures['Color1 failed'] += 1
                        criteria_failures['Color1 failed']['count'] += 1
                        # Store with pointing information
                        key = f"{pointing}_{row['ID']}"
                        criteria_failures['Color1 failed']['details'][key] = {
                            'pointing': pointing,
                            'id': row['ID'],
                            'details': f"F115W SNR={row['f115w_SNR']:.1f}, Color1={row['Color1']:.2f}"
                        }
                        
                    elif 'Color2 out of range' in reason:
                        pointing_failures['Color2 out of range'] += 1
                        criteria_failures['Color2 out of range']['count'] += 1
                        # Store with pointing information
                        key = f"{pointing}_{row['ID']}"
                        criteria_failures['Color2 out of range']['details'][key] = {
                            'pointing': pointing,
                            'id': row['ID'],
                            'details': f"Color2={row['Color2']:.2f} (range: -1.5 to 1.0)"
                        }
                        
            elif row['Status'] == 'Not found in catalog':
                pointing_failures['Not found in catalog'] += 1
                criteria_failures['Not found in catalog']['count'] += 1
                # Store with pointing information
                key = f"{pointing}_{row['ID']}"
                criteria_failures['Not found in catalog']['details'][key] = {
                    'pointing': pointing,
                    'id': row['ID'],
                    'details': "Source ID not present in catalog"
                }
        
        # Create criteria string for this pointing
        criteria_str = ""
        for criterion, count in pointing_failures.items():
            if count > 0:
                if criteria_str:
                    criteria_str += ", "
                criteria_str += f"{criterion[:1]}:{count}"
        
        print(f"{pointing:<10} {selected:<10} {total:<10} {percentage:>9.1f}% {criteria_str:<20}")
        
        total_selected += selected
        total_galaxies += total
    
    # Print total
    if total_galaxies > 0:
        total_percentage = (total_selected / total_galaxies) * 100
        print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
        print(f"{'TOTAL':<10} {total_selected:<10} {total_galaxies:<10} {total_percentage:>9.1f}%")
        
        # Print detailed criteria failure summary
        print(f"\n{'DETAILED CRITERIA FAILURE ANALYSIS':<50}")
        print(f"{'='*50}")
        
        for criterion, data in criteria_failures.items():
            if data['count'] > 0:
                percentage = (data['count'] / total_galaxies) * 100
                print(f"\n{criterion}: {data['count']} galaxies ({percentage:.1f}%)")
                print("-" * 50)
                
                for key, failure_info in data['details'].items():
                    pointing = failure_info['pointing']
                    gal_id = failure_info['id']
                    details = failure_info['details']
                    
                    if isinstance(details, list):
                        # For multiple filter failures
                        print(f"  Source {gal_id} in {pointing}: {', '.join(details)}")
                    else:
                        # For single value failures
                        print(f"  Source {gal_id} in {pointing}: {details}")
    
    # Also add detailed criteria information to the summary report
    summary_file = os.path.join(output_base_dir, f"diagnostic_summary_report.txt")
    with open(summary_file, 'a') as f:
        f.write(f"\n{'DETAILED CRITERIA FAILURE ANALYSIS':<50}\n")
        f.write("=" * 50 + "\n")
        
        for criterion, data in criteria_failures.items():
            if data['count'] > 0:
                percentage = (data['count'] / total_galaxies) * 100
                f.write(f"\n{criterion}: {data['count']} galaxies ({percentage:.1f}%)\n")
                f.write("-" * 50 + "\n")
                
                for key, failure_info in data['details'].items():
                    pointing = failure_info['pointing']
                    gal_id = failure_info['id']
                    details = failure_info['details']
                    
                    if isinstance(details, list):
                        f.write(f"  Source {gal_id} in {pointing}: {', '.join(details)}\n")
                    else:
                        f.write(f"  Source {gal_id} in {pointing}: {details}\n")
                        
def main():
    """Main function with detailed diagnostic analysis."""
    # Parse scaling factors first
    scaling_factors = parse_scaling_factors(scaling_file_path)
    if not scaling_factors:
        print("ERROR: Could not parse scaling factors. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print(f"PROCESSING ALL POINTINGS")
    print(f"{'='*80}")
    
    # Use the single output directory defined at the top
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_base_dir)
    
    print(f"Starting detailed diagnostic analysis of {len(pointings)} pointings")
    print(f"Output will be saved to: {output_base_dir}")
    print("Applying SNR scaling factors from NMAD calibration...")
    
    all_analysis_results = {}
    
    for pointing in pointings:
        print(f"\n{'='*80}")
        print(f"ANALYZING {pointing.upper()}")
        print(f"{'='*80}")
        
        # Load catalogs
        filter_data = {}
        catalog_dir = os.path.join(base_dir, pointing, catalog_subdir)
        
        if not os.path.isdir(catalog_dir):
            print(f"Missing directory: {catalog_dir}")
            continue
            
        # Load all filter catalogs with scaling
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path, pointing, filt, scaling_factors)
            if data is None:
                print(f"Missing {filt} data for {pointing}")
                missing_data = True
                break
            filter_data[filt] = data
            
        if not missing_data:
            # Apply selection criteria to get the mask
            selection_mask = apply_f115w_selection(filter_data)
            
            # Create filtered catalogs
            create_filtered_catalogs(filter_data, selection_mask, pointing, output_base_dir)
            
            # Run detailed analysis - pass the selection_mask to avoid recalculating
            analysis_df = analyze_missed_galaxies_detailed(
                filter_data, 
                highlight_ids.get(pointing, []),
                pointing,
                selection_mask
            )
            all_analysis_results[pointing] = analysis_df
            
            # Also run the original plotting for comparison - pass the selection_mask
            total, recovered = plot_snr_values(
                filter_data, 
                pointing, 
                highlight_ids.get(pointing, []),
                os.path.join(output_base_dir, pointing),
                selection_mask
            )
    
    # Generate summary report across all pointings
    generate_summary_report(all_analysis_results)
    
    # Generate final summary table
    generate_final_summary(all_analysis_results)
    
    print(f"\nDiagnostic analysis complete. All outputs saved to: {output_base_dir}")

if __name__ == '__main__':
    main()