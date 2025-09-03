import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import ascii

# === Configuration ===
base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
catalog_subdir = 'catalogue_z7'
pointings = [f'nircam{i}' for i in range(1, 11)]
filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
print("running code 1")
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
output_dir = './2_SNR_54_galaxy_analysis'
os.makedirs(output_dir, exist_ok=True)

# === Function to read catalog and add SNR ===
def read_sextractor_catalog(filepath):
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

# === Main process ===
output_file = os.path.join(output_dir, "brenjit_ID_flux_snr_table.txt")

# Store all SNR data for plotting
all_snr_data = {}

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

        # Load all filter catalogs
        filter_data = {}
        for filt in filters:
            cat_path = os.path.join(cat_dir, f"f150dropout_{filt}_catalog.cat")
            if not os.path.exists(cat_path):
                print(f"Missing {filt} for {pointing}")
                break
            filter_data[filt] = read_sextractor_catalog(cat_path)
        else:
            # Initialize SNR data for this pointing
            all_snr_data[pointing] = {}
            
            # Process only Brenjit IDs for this pointing
            for bid in highlight_ids.get(pointing, []):
                row_values = [pointing, str(bid)]
                snr_values = []
                
                for filt in filters:
                    data = filter_data[filt]
                    match = data[data['NUMBER'] == bid]
                    if len(match) == 1:
                        row_values += [
                            f"{match['FLUX_AUTO'][0]:.3e}",
                            f"{match['FLUXERR_AUTO'][0]:.3e}",
                            f"{match['SNR'][0]:.2f}"
                        ]
                        snr_values.append(match['SNR'][0])
                    else:
                        # No match → fill with blanks
                        row_values += ["NA", "NA", "NA"]
                        snr_values.append(0)  # Use 0 for plotting
                
                f.write("\t".join(row_values) + "\n")
                all_snr_data[pointing][bid] = snr_values

print(f"✅ Done! Output saved to {output_file}")

# === Create SNR plots ===
print("Creating SNR plots...")

# Create a plot for each pointing
for pointing, snr_dict in all_snr_data.items():
    if not snr_dict:
        continue
        
    # Determine layout for subplots
    n_galaxies = len(snr_dict)
    n_cols = min(3, n_galaxies)
    n_rows = (n_galaxies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'SNR by Filter - {pointing.upper()}', fontsize=16)
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Plot each galaxy
    for idx, (galaxy_id, snr_values) in enumerate(snr_dict.items()):
        ax = axes[idx]
        
        # Create the plot
        x_pos = np.arange(len(filters))
        bars = ax.bar(x_pos, snr_values, alpha=0.7, color='steelblue')
        
        # Add value labels on top of bars
        for i, v in enumerate(snr_values):
            if v > 0:  # Only label non-zero values
                ax.text(i, v + max(snr_values)*0.02, f'{v:.1f}', 
                       ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_title(f'Galaxy ID: {galaxy_id}', fontsize=10)
        ax.set_xlabel('Filter')
        ax.set_ylabel('SNR')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add a horizontal line at SNR=5 for reference
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(len(filters)-1, 5.2, 'SNR=5', color='r', fontsize=8)
    
    # Hide any empty subplots
    for idx in range(n_galaxies, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust for suptitle
    
    # Save the figure
    plot_filename = os.path.join(output_dir, f"{pointing}_SNR_plots.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created plot for {pointing}")

# Create individual plots for each galaxy
individual_plots_dir = os.path.join(output_dir, "individual_galaxies")
os.makedirs(individual_plots_dir, exist_ok=True)

for pointing, snr_dict in all_snr_data.items():
    for galaxy_id, snr_values in snr_dict.items():
        plt.figure(figsize=(10, 6))
        
        # Create the plot
        x_pos = np.arange(len(filters))
        bars = plt.bar(x_pos, snr_values, alpha=0.7, color='steelblue')
        
        # Add value labels on top of bars
        for i, v in enumerate(snr_values):
            if v > 0:  # Only label non-zero values
                plt.text(i, v + max(snr_values)*0.02, f'{v:.1f}', 
                        ha='center', va='bottom', fontsize=10)
        
        # Customize the plot
        plt.title(f'SNR by Filter - {pointing.upper()} Galaxy ID: {galaxy_id}', fontsize=14)
        plt.xlabel('Filter')
        plt.ylabel('SNR')
        plt.xticks(x_pos, [f.upper() for f in filters], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add a horizontal line at SNR=5 for reference
        plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.text(len(filters)-1, 5.2, 'SNR=5', color='r', fontsize=10)
        
        # Save the individual plot
        plot_filename = os.path.join(individual_plots_dir, f"{pointing}_galaxy_{galaxy_id}_SNR.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

print(f"✅ All plots saved to {output_dir}")
print("Individual galaxy plots saved to:", individual_plots_dir)