import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import ascii
import pandas as pd
import re

# === Configuration ===
base_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
catalog_subdir = 'catalogue_z7'
eazy_catalog_dir = '/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Eazy_catalogue/'
pointings = [f'nircam{i}' for i in range(1, 11)]
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
print("Running code for all SNRs (SExtractor, Scaled, NMAD)")

# File containing scaling factors
scaling_file_path = '/media/iit-t/MY_SSD_1TB/Work_PhD/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/14_SNR_Comparison_Analysis_NMAD/snr_fit_summary.txt'

# Filter mapping for EAZY catalog
filter_mapping = {
    'f606w': 'F606W', 'f814w': 'F814W', 'f115w': 'F115W', 'f150w': 'F150W',
    'f200w': 'F200W', 'f277w': 'F277W', 'f356w': 'F356W', 'f410m': 'F410M', 'f444w': 'F444W'
}

# Function to read scaling factors from file
def read_scaling_factors(file_path):
    """Read scaling factors from the specified file."""
    scaling_factors = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by pointing sections
        sections = re.split(r'Pointing:\s*(\w+),', content)
        
        # The first element is empty or header, so we start from index 1
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            pointing = sections[i].strip().lower()
            section_content = sections[i + 1]
            
            if pointing not in scaling_factors:
                scaling_factors[pointing] = {}
            
            # Extract filter scaling factors using regex
            for filt in filters:
                pattern = rf'{filt.upper()}:\s*([\d.]+)'
                match = re.search(pattern, section_content)
                if match:
                    try:
                        value = float(match.group(1))
                        scaling_factors[pointing][filt] = value
                    except ValueError:
                        print(f"Warning: Could not parse scaling factor for {filt} in {pointing}")
        
        print("Successfully read scaling factors from file:")
        for pointing in scaling_factors:
            print(f"  {pointing}: {scaling_factors[pointing]}")
        
        return scaling_factors
    
    except Exception as e:
        print(f"Error reading scaling factors file: {e}")
        # Return default structure with 1.0 for all
        return {pointing: {filt: 1.0 for filt in filters} for pointing in pointings}

# Read scaling factors from file
scaling_factors = read_scaling_factors(scaling_file_path)

# If no scaling factors were read properly, use default values of 1.0
if not scaling_factors or all(len(v) == 0 for v in scaling_factors.values()):
    print("Using default scaling factors of 1.0")
    scaling_factors = {pointing: {filt: 1.0 for filt in filters} for pointing in pointings}

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

# Create output directory
output_dir = './14_2_Scaled_SNR_54_galaxy_analysis'
os.makedirs(output_dir, exist_ok=True)

# === Function to read catalog and add scaled SNR ===
def read_sextractor_catalog(filepath, pointing, filter_name):
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
        
        # Calculate original SNR
        with np.errstate(divide='ignore', invalid='ignore'):
            original_snr = data['FLUX_AUTO'] / data['FLUXERR_AUTO']
        original_snr[np.isinf(original_snr) | np.isnan(original_snr)] = 0
        data['SNR_ORIGINAL'] = original_snr
        
        # Apply scaling to get NMAD-equivalent SNR
        # Use scaling factor from file, or default to 1.0 if not found
        scale_factor = scaling_factors.get(pointing, {}).get(filter_name, 1.0)
        data['SNR'] = original_snr * scale_factor
        
        print(f"Applied scaling factor {scale_factor:.6f} to {filter_name} SNR for {pointing}")
        
        return data
    except Exception as e:
        print(f"Error reading catalog {filepath}: {e}")
        return None

# === Function to read EAZY catalog with NMAD flux measurements ===
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

# === Function to calculate NMAD SNR from EAZY data ===
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

# === Function to create comparison plots for each pointing ===
def create_comparison_plots_for_pointing(pointing, filter_data, eazy_data, nmad_snr, all_snr_data, output_dir):
    """
    Create comparison plots for a pointing with three columns:
    Left: Original SExtractor SNR for highlighted galaxies
    Middle: Scaled SNR for highlighted galaxies
    Right: NMAD SNR for highlighted galaxies
    """
    if pointing not in all_snr_data or not all_snr_data[pointing]:
        return
    
    # Get the galaxy IDs for this pointing
    galaxy_ids = list(all_snr_data[pointing].keys())
    n_galaxies = len(galaxy_ids)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_galaxies, 3, figsize=(18, 3*n_galaxies))
    fig.suptitle(f'SNR Comparison - {pointing.upper()}', fontsize=16)
    
    # If only one galaxy, make axes 2D for consistent indexing
    if n_galaxies == 1:
        axes = np.array([axes])
    
    # Plot each galaxy
    for idx, galaxy_id in enumerate(galaxy_ids):
        # Get original, scaled, and NMAD SNR values
        original_snr_values = []
        scaled_snr_values = []
        nmad_snr_values = []
        
        for filt in filters:
            if filt in filter_data and filter_data[filt] is not None:
                data = filter_data[filt]
                match = data[data['NUMBER'] == galaxy_id]
                if len(match) == 1:
                    original_snr_values.append(match['SNR_ORIGINAL'][0])
                    scaled_snr_values.append(match['SNR'][0])
                else:
                    original_snr_values.append(0)
                    scaled_snr_values.append(0)
            else:
                original_snr_values.append(0)
                scaled_snr_values.append(0)
            
            # Get NMAD SNR
            if eazy_data is not None:
                eazy_match = eazy_data[eazy_data['id'] == galaxy_id]
                if len(eazy_match) > 0:
                    nmad_snr_values.append(nmad_snr[filt].iloc[eazy_match.index[0]])
                else:
                    nmad_snr_values.append(0)
            else:
                nmad_snr_values.append(0)
        
        # Left: Original SExtractor SNR
        ax_left = axes[idx, 0]
        x_pos = np.arange(len(filters))
        bars_left = ax_left.bar(x_pos, original_snr_values, alpha=0.7, color='steelblue')
        
        # Add value labels on top of bars
        for i, v in enumerate(original_snr_values):
            if v > 0:
                ax_left.text(i, v + max(original_snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=8)
        
        ax_left.set_title(f'Galaxy ID: {galaxy_id} - Original SNR', fontsize=10)
        ax_left.set_xlabel('Filter')
        ax_left.set_ylabel('SNR (Original)')
        ax_left.set_xticks(x_pos)
        ax_left.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
        ax_left.grid(True, alpha=0.3, axis='y')
        ax_left.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1)
        
        # Middle: Scaled SNR
        ax_middle = axes[idx, 1]
        bars_middle = ax_middle.bar(x_pos, scaled_snr_values, alpha=0.7, color='orange')
        
        # Add value labels on top of bars
        for i, v in enumerate(scaled_snr_values):
            if v > 0:
                ax_middle.text(i, v + max(scaled_snr_values)*0.02, f'{v:.1f}', 
                             ha='center', va='bottom', fontsize=8)
        
        ax_middle.set_title(f'Galaxy ID: {galaxy_id} - Scaled SNR', fontsize=10)
        ax_middle.set_xlabel('Filter')
        ax_middle.set_ylabel('SNR (Scaled)')
        ax_middle.set_xticks(x_pos)
        ax_middle.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
        ax_middle.grid(True, alpha=0.3, axis='y')
        ax_middle.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1)
        
        # Right: NMAD SNR
        ax_right = axes[idx, 2]
        bars_right = ax_right.bar(x_pos, nmad_snr_values, alpha=0.7, color='green')
        
        # Add value labels on top of bars
        for i, v in enumerate(nmad_snr_values):
            if v > 0:
                ax_right.text(i, v + max(nmad_snr_values)*0.02, f'{v:.1f}', 
                             ha='center', va='bottom', fontsize=8)
        
        ax_right.set_title(f'Galaxy ID: {galaxy_id} - NMAD SNR', fontsize=10)
        ax_right.set_xlabel('Filter')
        ax_right.set_ylabel('SNR (NMAD)')
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for suptitle
    
    # Save the figure
    plot_filename = os.path.join(output_dir, f"{pointing}_SNR_comparison.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created SNR comparison plot for {pointing}")

# === Function to create individual galaxy plots ===
def create_individual_galaxy_plots(all_snr_data, all_original_snr_data, all_nmad_snr_data, output_dir):
    """
    Create individual plots for each highlighted galaxy showing original, scaled, and NMAD SNR
    """
    individual_plots_dir = os.path.join(output_dir, "individual_galaxies")
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    for pointing, snr_dict in all_snr_data.items():
        for galaxy_id, snr_values in snr_dict.items():
            original_snr_values = all_original_snr_data[pointing][galaxy_id]
            nmad_snr_values = all_nmad_snr_data[pointing][galaxy_id]
            
            plt.figure(figsize=(18, 6))
            
            # Create three subplots
            plt.subplot(1, 3, 1)
            x_pos = np.arange(len(filters))
            bars = plt.bar(x_pos, original_snr_values, alpha=0.7, color='steelblue')
            
            # Add value labels on top of bars
            for i, v in enumerate(original_snr_values):
                if v > 0:
                    plt.text(i, v + max(original_snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Original SNR - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=12)
            plt.xlabel('Filter')
            plt.ylabel('SNR (Original)')
            plt.xticks(x_pos, [f.upper() for f in filters], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            
            plt.subplot(1, 3, 2)
            bars = plt.bar(x_pos, snr_values, alpha=0.7, color='orange')
            
            # Add value labels on top of bars
            for i, v in enumerate(snr_values):
                if v > 0:
                    plt.text(i, v + max(snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Scaled SNR - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=12)
            plt.xlabel('Filter')
            plt.ylabel('SNR (Scaled)')
            plt.xticks(x_pos, [f.upper() for f in filters], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            
            plt.subplot(1, 3, 3)
            bars = plt.bar(x_pos, nmad_snr_values, alpha=0.7, color='green')
            
            # Add value labels on top of bars
            for i, v in enumerate(nmad_snr_values):
                if v > 0:
                    plt.text(i, v + max(nmad_snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title(f'NMAD SNR - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=12)
            plt.xlabel('Filter')
            plt.ylabel('SNR (NMAD)')
            plt.xticks(x_pos, [f.upper() for f in filters], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            
            plt.tight_layout()
            
            # Save the individual plot
            plot_filename = os.path.join(individual_plots_dir, f"{pointing}_galaxy_{galaxy_id}_SNR_comparison.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()

# === Main process ===
def main():
    output_file = os.path.join(output_dir, "brenjit_ID_flux_snr_table.txt")

    # Store all SNR data for plotting
    all_snr_data = {}
    all_original_snr_data = {}  # Store original SNR values
    all_nmad_snr_data = {}  # Store NMAD SNR values
    all_filter_data = {}  # Store all filter data

    with open(output_file, "w") as f:
        # Write header
        header = ["pointing", "ID"]
        for filt in filters:
            header += [f"{filt}_FLUX", f"{filt}_FLUXERR", f"{filt}_SNR"]
        f.write("\t".join(header) + "\n")

        # Loop over pointings
        for pointing in pointings:
            cat_dir = os.path.join(base_dir, pointing, catalog_subdir)
            if not os.path.isdir(cat_dir):
                print(f"Skipping {pointing}, directory not found.")
                continue

            # Load all filter catalogs with scaling
            filter_data = {}
            for filt in filters:
                cat_path = os.path.join(cat_dir, f"f150dropout_{filt}_catalog.cat")
                if not os.path.exists(cat_path):
                    print(f"Missing {filt} for {pointing}")
                    filter_data[filt] = None
                    continue
                filter_data[filt] = read_sextractor_catalog(cat_path, pointing, filt)
            
            # Store filter data for comparison plots
            all_filter_data[pointing] = filter_data
            
            # Read EAZY catalog for NMAD SNR
            eazy_file = os.path.join(eazy_catalog_dir, f"{pointing}_eazy_catalogue_54_gal.cat")
            eazy_data = None
            nmad_snr = None
            
            if os.path.exists(eazy_file):
                eazy_data = read_eazy_catalog(eazy_file)
                if eazy_data is not None:
                    nmad_snr = calculate_nmad_snr(eazy_data)
                else:
                    print(f"Could not read EAZY data for {pointing}")
            else:
                print(f"EAZY catalog not found: {eazy_file}")
            
            # Initialize SNR data for this pointing
            all_snr_data[pointing] = {}
            all_original_snr_data[pointing] = {}
            all_nmad_snr_data[pointing] = {}
            
            # Process only Brenjit IDs for this pointing
            for bid in highlight_ids.get(pointing, []):
                row_values = [pointing, str(bid)]
                snr_values = []
                original_snr_values = []
                nmad_snr_values = []
                
                for filt in filters:
                    if filt not in filter_data or filter_data[filt] is None:
                        row_values += ["NA", "NA", "NA"]
                        snr_values.append(0)
                        original_snr_values.append(0)
                        nmad_snr_values.append(0)
                        continue
                        
                    data = filter_data[filt]
                    match = data[data['NUMBER'] == bid]
                    if len(match) == 1:
                        row_values += [
                            f"{match['FLUX_AUTO'][0]:.3e}",
                            f"{match['FLUXERR_AUTO'][0]:.3e}",
                            f"{match['SNR'][0]:.2f}"
                        ]
                        snr_values.append(match['SNR'][0])
                        original_snr_values.append(match['SNR_ORIGINAL'][0])
                    else:
                        # No match → fill with blanks
                        row_values += ["NA", "NA", "NA"]
                        snr_values.append(0)
                        original_snr_values.append(0)
                    
                    # Get NMAD SNR
                    if eazy_data is not None and nmad_snr is not None:
                        eazy_match = eazy_data[eazy_data['id'] == bid]
                        if len(eazy_match) > 0:
                            nmad_snr_values.append(nmad_snr[filt].iloc[eazy_match.index[0]])
                        else:
                            nmad_snr_values.append(0)
                    else:
                        nmad_snr_values.append(0)
                
                f.write("\t".join(row_values) + "\n")
                all_snr_data[pointing][bid] = snr_values
                all_original_snr_data[pointing][bid] = original_snr_values
                all_nmad_snr_data[pointing][bid] = nmad_snr_values
            
            # Create comparison plots for this pointing
            create_comparison_plots_for_pointing(pointing, filter_data, eazy_data, nmad_snr, all_snr_data, output_dir)

    print(f"✅ Done! Output saved to {output_file}")

    # === Create individual galaxy plots ===
    print("Creating individual galaxy plots...")
    create_individual_galaxy_plots(all_snr_data, all_original_snr_data, all_nmad_snr_data, output_dir)

    print(f"✅ All plots saved to {output_dir}")

if __name__ == "__main__":
    main()