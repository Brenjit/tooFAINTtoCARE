import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import ascii

# === Configuration ===
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
catalog_subdir = 'catalogue_z7'
pointings = [f'nircam{i}' for i in range(1, 11)]
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
print("Running code 1 again")

# Scaling factors for each pointing and filter (from fit: NMAD_SNR = k × SEXTRACTOR_SNR)
scaling_factors = {
    'nircam1': {
        'f606w': 0.6962, 'f814w': 0.3317, 'f115w': 0.7178, 'f150w': 0.5154,
        'f200w': 0.4458, 'f277w': 0.6519, 'f356w': 0.8319, 'f410m': 0.6572, 'f444w': 0.6175
    },
    'nircam2': {
        'f606w': 0.7730, 'f814w': 0.6742, 'f115w': 0.3593, 'f150w': 0.3372,
        'f200w': 0.3592, 'f277w': 0.4795, 'f356w': 0.5367, 'f410m': 0.7332, 'f444w': 0.7317
    },
    'nircam3': {
        'f606w': 1.0, 'f814w': 0.6090, 'f115w': 0.5086, 'f150w': 0.4157,
        'f200w': 0.4917, 'f277w': 0.4523, 'f356w': 0.5159, 'f410m': 0.8866, 'f444w': 0.8432
    },
    'nircam4': {
        'f606w': 0.7337, 'f814w': 0.3977, 'f115w': 0.4239, 'f150w': 0.3499,
        'f200w': 0.4636, 'f277w': 0.4427, 'f356w': 0.5696, 'f410m': 0.6960, 'f444w': 0.6342
    },
    'nircam5': {
        'f606w': 1.0, 'f814w': 0.7252, 'f115w': 0.7634, 'f150w': 0.0020,
        'f200w': 0.0016, 'f277w': 0.0012, 'f356w': 0.7181, 'f410m': 0.0010, 'f444w': 0.0006
    },
    'nircam6': {
        'f606w': 1.0, 'f814w': 1.0, 'f115w': 0.4591, 'f150w': 0.4421,
        'f200w': 0.3999, 'f277w': 0.4696, 'f356w': 0.5911, 'f410m': 0.5325, 'f444w': 0.9262
    },
    'nircam7': {
        'f606w': 0.5093, 'f814w': 0.4855, 'f115w': 0.4379, 'f150w': 0.0015,
        'f200w': 0.0011, 'f277w': 0.0011, 'f356w': 0.5536, 'f410m': 0.0010, 'f444w': 0.0008
    },
    'nircam8': {
        'f606w': 0.3593, 'f814w': 0.4383, 'f115w': 0.5701, 'f150w': 0.0015,
        'f200w': 0.0015, 'f277w': 0.0010, 'f356w': 0.5066, 'f410m': 0.0009, 'f444w': 0.0006
    },
    'nircam9': {
        'f606w': 0.8968, 'f814w': 0.7839, 'f115w': 0.6705, 'f150w': 0.0014,
        'f200w': 0.0016, 'f277w': 0.0011, 'f356w': 0.5763, 'f410m': 0.0006, 'f444w': 0.0006
    },
    'nircam10': {
        'f606w': 0.5280, 'f814w': 0.5690, 'f115w': 0.3774, 'f150w': 0.3263,
        'f200w': 0.3247, 'f277w': 0.4234, 'f356w': 0.5022, 'f410m': 0.6355, 'f444w': 0.5614
    }
}

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
output_dir = './2_1_Scaled_SNR_54_galaxy_analysis'
os.makedirs(output_dir, exist_ok=True)

# === Function to read catalog and add scaled SNR ===
def read_sextractor_catalog(filepath, pointing, filter_name):
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
    scale_factor = scaling_factors[pointing].get(filter_name, 1.0)
    data['SNR'] = original_snr * scale_factor
    
    print(f"Applied scaling factor {scale_factor:.6f} to {filter_name} SNR for {pointing}")
    
    return data

# === Function to create comparison plots for each pointing ===
def create_comparison_plots_for_pointing(pointing, filter_data, all_snr_data, output_dir):
    """
    Create comparison plots for a pointing with two sections:
    Left: Original SExtractor SNR for highlighted galaxies
    Right: Scaled SNR for highlighted galaxies
    """
    if pointing not in all_snr_data or not all_snr_data[pointing]:
        return
    
    # Get the galaxy IDs for this pointing
    galaxy_ids = list(all_snr_data[pointing].keys())
    n_galaxies = len(galaxy_ids)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_galaxies, 2, figsize=(12, 3*n_galaxies))
    fig.suptitle(f'SNR Comparison - {pointing.upper()}', fontsize=16)
    
    # If only one galaxy, make axes 2D for consistent indexing
    if n_galaxies == 1:
        axes = np.array([axes])
    
    # Plot each galaxy
    for idx, galaxy_id in enumerate(galaxy_ids):
        # Get original and scaled SNR values
        original_snr_values = []
        scaled_snr_values = []
        
        for filt in filters:
            if filt in filter_data:
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
        
        # Right: Scaled SNR
        ax_right = axes[idx, 1]
        bars_right = ax_right.bar(x_pos, scaled_snr_values, alpha=0.7, color='orange')
        
        # Add value labels on top of bars
        for i, v in enumerate(scaled_snr_values):
            if v > 0:
                ax_right.text(i, v + max(scaled_snr_values)*0.02, f'{v:.1f}', 
                             ha='center', va='bottom', fontsize=8)
        
        ax_right.set_title(f'Galaxy ID: {galaxy_id} - Scaled SNR', fontsize=10)
        ax_right.set_xlabel('Filter')
        ax_right.set_ylabel('SNR (NMAD equivalent)')
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
def create_individual_galaxy_plots(all_snr_data, output_dir):
    """
    Create individual plots for each highlighted galaxy showing both original and scaled SNR
    """
    individual_plots_dir = os.path.join(output_dir, "individual_galaxies")
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    for pointing, snr_dict in all_snr_data.items():
        for galaxy_id, snr_values in snr_dict.items():
            # We need to get the original SNR values for this galaxy
            # This will be handled in the main function
            
            # For now, just create a placeholder
            plt.figure(figsize=(12, 6))
            
            # Create two subplots
            plt.subplot(1, 2, 1)
            x_pos = np.arange(len(filters))
            bars = plt.bar(x_pos, snr_values, alpha=0.7, color='steelblue')
            
            # Add value labels on top of bars
            for i, v in enumerate(snr_values):
                if v > 0:
                    plt.text(i, v + max(snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Original SNR - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=12)
            plt.xlabel('Filter')
            plt.ylabel('SNR (Original)')
            plt.xticks(x_pos, [f.upper() for f in filters], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(x_pos, snr_values, alpha=0.7, color='orange')
            
            # Add value labels on top of bars
            for i, v in enumerate(snr_values):
                if v > 0:
                    plt.text(i, v + max(snr_values)*0.02, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Scaled SNR - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=12)
            plt.xlabel('Filter')
            plt.ylabel('SNR (NMAD equivalent)')
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
                    continue
                filter_data[filt] = read_sextractor_catalog(cat_path, pointing, filt)
            
            # Store filter data for comparison plots
            all_filter_data[pointing] = filter_data
            
            # Initialize SNR data for this pointing
            all_snr_data[pointing] = {}
            all_original_snr_data[pointing] = {}
            
            # Process only Brenjit IDs for this pointing
            for bid in highlight_ids.get(pointing, []):
                row_values = [pointing, str(bid)]
                snr_values = []
                original_snr_values = []
                
                for filt in filters:
                    if filt not in filter_data:
                        row_values += ["NA", "NA", "NA"]
                        snr_values.append(0)
                        original_snr_values.append(0)
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
                
                f.write("\t".join(row_values) + "\n")
                all_snr_data[pointing][bid] = snr_values
                all_original_snr_data[pointing][bid] = original_snr_values

    print(f"✅ Done! Output saved to {output_file}")

    # === Create SNR comparison plots ===
    print("Creating SNR comparison plots...")
    for pointing in all_filter_data:
        create_comparison_plots_for_pointing(pointing, all_filter_data[pointing], all_snr_data, output_dir)

    # === Create individual galaxy plots ===
    print("Creating individual galaxy plots...")
    create_individual_galaxy_plots(all_snr_data, output_dir)

    print(f"✅ All plots saved to {output_dir}")

if __name__ == "__main__":
    main()