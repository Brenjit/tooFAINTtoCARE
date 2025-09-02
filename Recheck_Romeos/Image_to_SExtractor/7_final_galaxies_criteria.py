# In this code, I am trying to make a new diagnostic plot for the 54 galaxies
# Modifications were made after looking into the galaxy plots of 4_snr_color_all_54_galaxy_analysis plots manually
# We removed the second color cut criteria F150-F277 > 0.5

import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import ascii
import os
from astropy.table import Table
from datetime import datetime
import pandas as pd
import logging

# Configuration
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
output_base_dir = './7_final_Diagnostic_Analysis'
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

def setup_logging(output_dir):
    """Set up proper logging to both console and file."""
    log_file = os.path.join(output_dir, "diagnostic_analysis_log.txt")
    
    # Clear previous log file
    with open(log_file, 'w') as f:
        f.write(f"DIAGNOSTIC ANALYSIS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # Create logger
    logger = logging.getLogger('DiagnosticAnalysis')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def read_sextractor_catalog(filepath, logger):
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
        logger.error(f"Error reading {filepath}: {e}")
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

def apply_f115w_selection(data_dict, logger):
    """Applies F115W-dropout (z=8.5-12) selection criteria WITHOUT cut3 and returns a boolean mask."""
    # SNR detection in red filters
    sn_detect = (data_dict['f150w']['SNR'] > 5) & \
                (data_dict['f200w']['SNR'] > 5) & \
                (data_dict['f277w']['SNR'] > 5) & \
                (data_dict['f356w']['SNR'] > 5) & \
                (data_dict['f444w']['SNR'] > 5)
    
    # Veto in blue filters (no detection)
    veto = (data_dict['f814w']['SNR'] < 5) & (data_dict['f606w']['SNR'] < 5)
    
    # Get colors
    mags = get_mags(data_dict, ['f115w', 'f150w', 'f277w'])
    color1 = mags['f115w'] - mags['f150w']  # F115W - F150W
    color2 = mags['f150w'] - mags['f277w']  # F150W - F277W
    
    # Color cuts (without cut3)
    cut1 = (data_dict['f115w']['SNR'] < 2) | (color1 > 0.1)  # No detection or red color
    cut2 = (color2 > -1.5) & (color2 < 1.0)  # Reasonable continuum slope
    
    # Combine all criteria (EXCLUDING cut3)
    mask = sn_detect & veto & cut1 & cut2
    logger.info(f"Found {np.sum(mask)} candidates (selection without cut3).")
    return mask

def plot_snr_values(filter_data, pointing, highlight_ids, output_dir, logger):
    """Generate SNR plots with Brenjit_ID sources marked with green outline if they meet criteria."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get source IDs from f150w catalog (reference)
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Calculate selection criteria
    selection_mask = apply_f115w_selection(filter_data, logger)
    
    # Identify which Brenjit sources meet the criteria
    brenjit_meet_criteria = highlight_mask & selection_mask
    brenjit_fail_criteria = highlight_mask & ~selection_mask
    
    # Calculate recovery statistics
    total_brenjit = np.sum(highlight_mask)
    recovered_brenjit = np.sum(brenjit_meet_criteria)
    recovery_rate = (recovered_brenjit / total_brenjit) * 100 if total_brenjit > 0 else 0
    
    # Log recovery info
    logger.info(f"Total Brenjit_ID sources: {recovered_brenjit}/{total_brenjit}")
    logger.info(f"Recovery rate: {recovery_rate:.1f}%")
    
    if recovered_brenjit > 0:
        logger.info("Recovered IDs: " + ", ".join(map(str, source_ids[brenjit_meet_criteria])))
    if np.sum(brenjit_fail_criteria) > 0:
        logger.info("Missed IDs: " + ", ".join(map(str, source_ids[brenjit_fail_criteria])))
    
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

def analyze_missed_galaxies_detailed(filter_data, highlight_ids, pointing, logger):
    """Detailed analysis of why specific galaxies are being missed (WITHOUT cut3)."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DETAILED DIAGNOSTIC ANALYSIS FOR {pointing.upper()} (NO CUT3)")
    logger.info(f"{'='*60}")
    
    source_ids = filter_data['f150w']['NUMBER']
    highlight_mask = np.isin(source_ids, highlight_ids)
    
    # Apply current selection criteria (without cut3)
    selection_mask = apply_f115w_selection(filter_data, logger)
    
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
        
        # 1. SNR Detection criteria (REMOVED F410M requirement)
        detection_filters = ['f150w', 'f200w', 'f277w', 'f356w', 'f444w']  # Removed 'f410m'
        detection_snrs = {}
        detection_failed = []
        for filt in detection_filters:
            snr_val = filter_data[filt]['SNR'][idx]
            detection_snrs[filt] = snr_val
            if snr_val <= 5:
                detection_failed.append(f"{filt}({snr_val:.1f})")
        
        if detection_failed:
            reasons.append(f"Low SNR in detection bands: {', '.join(detection_failed)}")
        
        # 2. Veto criteria (must BOTH be < 5)
        veto_filters = ['f606w', 'f814w']
        veto_snrs = {}
        veto_failed = []
        for filt in veto_filters:
            snr_val = filter_data[filt]['SNR'][idx]
            veto_snrs[filt] = snr_val
            if snr_val >= 5:
                veto_failed.append(f"{filt}({snr_val:.1f})")
        
        if veto_failed:
            reasons.append(f"High SNR in veto bands: {', '.join(veto_failed)}")
        
        # 3. Color criteria (without cut3)
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
            'f444w_SNR': detection_snrs['f444w'],
            'f410m_SNR': filter_data['f410m']['SNR'][idx],  # Still include for info, but not for criteria
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
    
    logger.info(f"\nSUMMARY FOR {pointing} (NO CUT3):")
    logger.info(f"Total Brenjit sources: {total}")
    logger.info(f"Pass selection: {passed}")
    logger.info(f"Fail selection: {failed}")
    if not_found > 0:
        logger.info(f"Not found in catalog: {not_found}")
    
    # Print detailed table
    logger.info(f"\nDETAILED ANALYSIS:")
    logger.info("-" * 120)
    for _, row in df.iterrows():
        status_msg = f"ID {row['ID']:4d}: {row['Status']:4s} - {row['Reasons']}"
        
        # Add color information for all galaxies
        if row['Status'] != 'Not found in catalog':
            status_msg += f" | Colors: F115W-F150W={row['Color1']:.2f}, F150W-F277W={row['Color2']:.2f}"
        
        logger.info(status_msg)
        
        if row['Status'] == 'FAIL' and 'Low SNR' in row['Reasons']:
            logger.info(f"      SNR values: f150w={row['f150w_SNR']:5.1f}, f200w={row['f200w_SNR']:5.1f}, "
                  f"f277w={row['f277w_SNR']:5.1f}, f356w={row['f356w_SNR']:5.1f}")
    
    # Create visual diagnostics (without cut3 boundary)
    create_diagnostic_plots(filter_data, df, highlight_ids, pointing, logger)
    
    return df

def create_diagnostic_plots(filter_data, analysis_df, highlight_ids, pointing, logger):
    """Create diagnostic plots for visual analysis (without cut3)."""
    
    output_dir = os.path.join(output_base_dir, pointing, 'diagnostics_no_cut3')
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
    plt.title(f'{pointing}: SNR in Detection Bands for Brenjit Sources (No Cut3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pointing}_detection_snr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Color-color diagram with criteria (without cut3 boundary)
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
    
    # Plot selection boundaries (without cut3)
    plt.axvline(x=-1.5, color='orange', linestyle=':', linewidth=2, label='Color2 lower limit')
    plt.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, label='Color2 upper limit')
    plt.axhline(y=0.1, color='purple', linestyle=':', linewidth=2, label='Color1 limit')
    
    plt.xlabel('F150W - F277W')
    plt.ylabel('F115W - F150W')
    plt.title(f'{pointing}: Color-Color Diagram (Selection WITHOUT Cut3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pointing}_color_diagram_no_cut3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Diagnostic plots saved to: {output_dir}")

def generate_summary_report(all_analysis_results, output_base_dir, logger):
    """Generate a comprehensive summary report across all pointings."""
    
    summary_file = os.path.join(output_base_dir, "diagnostic_summary_report.txt")
    
    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE DIAGNOSTIC SUMMARY REPORT\n")
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
        f.write(f"Overall success rate: {total_passed/total_galaxies*100:.1f}%\n")
        
        f.write(f"\nFAILURE REASONS (by frequency):\n")
        f.write("=" * 30 + "\n")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{reason}: {count} galaxies\n")
    
    logger.info(f"Summary report saved to: {summary_file}")

def main():
    """Main function with detailed diagnostic analysis (without cut3)."""
    # Change output directory to indicate no cut3
    global output_base_dir
    output_base_dir = './7_final_Diagnostic_Analysis_NO_CUT3'
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_base_dir)
    
    logger.info(f"Starting detailed diagnostic analysis of {len(pointings)} pointings (NO CUT3)")
    logger.info(f"Output will be saved to: {output_base_dir}")
    
    all_analysis_results = {}
    
    for pointing in pointings:
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING {pointing.upper()} (NO CUT3)")
        logger.info(f"{'='*80}")
        
        # Load catalogs
        filter_data = {}
        catalog_dir = os.path.join(base_dir, pointing, catalog_subdir)
        
        if not os.path.isdir(catalog_dir):
            logger.error(f"Missing directory: {catalog_dir}")
            continue
            
        # Load all filter catalogs
        missing_data = False
        for filt in filters:
            cat_path = os.path.join(catalog_dir, f"f150dropout_{filt}_catalog.cat")
            data = read_sextractor_catalog(cat_path, logger)
            if data is None:
                logger.error(f"Missing {filt} data for {pointing}")
                missing_data = True
                break
            filter_data[filt] = data
            
        if not missing_data:
            # Run detailed analysis
            analysis_df = analyze_missed_galaxies_detailed(
                filter_data, 
                highlight_ids.get(pointing, []),
                pointing,
                logger
            )
            all_analysis_results[pointing] = analysis_df
            
            # Also run the original plotting for comparison
            total, recovered = plot_snr_values(
                filter_data, 
                pointing, 
                highlight_ids.get(pointing, []),
                os.path.join(output_base_dir, pointing),
                logger
            )
    
    # Generate summary report across all pointings
    generate_summary_report(all_analysis_results, output_base_dir, logger)
    
    logger.info(f"\nDiagnostic analysis complete. All outputs saved to: {output_base_dir}")

if __name__ == '__main__':
    main()