import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits, ascii
import pandas as pd
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, aperture_photometry
from skimage.morphology import disk
from scipy.ndimage import binary_erosion
import matplotlib.cm as cm
from astropy.stats import sigma_clipped_stats
from pathlib import Path

# Configuration
sextractor_base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
eazy_catalog_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/Eazy_catalogue/'
image_base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/'
segmentation_base_dir = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/'
output_dir = './9_4_SNR_Method_Comparison'
pointings = [f'nircam{i}' for i in range(1, 11)]

# Brenjit_IDs to analyze
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

# HST parameters
hst_photflam_lamda = {
    'F606W': {'photflam': 7.79611686490250e-20, 'lamda': 5920.405914484679},
    'F814W': {'photflam': 7.05123239329114e-20, 'lamda': 8046.166645443039}
}

def nmad(data):
    """Normalized Median Absolute Deviation"""
    return 1.48 * np.median(np.abs(data - np.median(data)))

def hst_flux_to_ujy(sextractor_flux, photflam, lamda):
    """Convert HST flux to microJanskys"""
    flux_erg = sextractor_flux * photflam
    flux_jy = 3.34e4 * (lamda ** 2) * flux_erg
    flux_ujy = flux_jy * 1e6
    return flux_ujy

def flux_conversion_scale(fits_file):
    """Get conversion scale from JWST FITS file"""
    with fits.open(fits_file) as hdul:
        pixar_sr = hdul[0].header['PIXAR_SR']
    return pixar_sr

def read_eazy_catalog_compatible(filepath):
    """Read EAZY catalog with lowercase column names (compatible with your format)"""
    try:
        # Read the entire file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header to get column order
        header_line = lines[0].strip()
        columns = header_line.split()[1:]  # Skip 'id'
        
        data = []
        for line in lines[1:]:  # Skip header
            if line.strip() and not line.startswith('#'):
                values = line.split()
                if len(values) == len(columns) + 1:  # +1 for id
                    row = {'id': int(values[0])}
                    
                    # Map columns based on header
                    for i, col_name in enumerate(columns):
                        row[col_name] = float(values[i + 1])
                    
                    data.append(row)
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading EAZY catalog {filepath}: {e}")
        return None

def calculate_nmad_error(image_data, x, y, seg_map, num_apertures=200, aperture_radius=5):
    """Calculate error using NMAD method (your original approach)"""
    try:
        # Find empty regions
        empty_region_mask = (seg_map == 0) & (image_data != 0)
        kernel = disk(5)
        eroded_empty_mask = binary_erosion(empty_region_mask, structure=kernel)
        empty_pixels = np.argwhere(eroded_empty_mask)
        
        if len(empty_pixels) == 0:
            return np.nan
        
        empty_pixel_coords = empty_pixels[:, [1, 0]]
        kdtree = cKDTree(empty_pixel_coords)
        
        # Find closest empty apertures
        distances, indices = kdtree.query([x, y], k=min(num_apertures, len(empty_pixel_coords)))
        closest_empty_apertures = empty_pixel_coords[indices]
        
        # Perform aperture photometry
        apertures = CircularAperture(closest_empty_apertures, r=aperture_radius)
        phot_table = aperture_photometry(image_data, apertures)
        fluxes = phot_table['aperture_sum']
        
        return nmad(fluxes)
    except Exception as e:
        print(f"Error in NMAD calculation: {e}")
        return np.nan

def estimate_error_local_rms(image_data, x, y, aperture_radius=5, annulus_radius=10):
    """Estimate error using local background RMS"""
    try:
        y_int, x_int = int(y), int(x)
        
        # Define annulus around the source
        yy, xx = np.ogrid[-annulus_radius:annulus_radius+1, -annulus_radius:annulus_radius+1]
        annulus_mask = (xx**2 + yy**2) >= aperture_radius**2
        annulus_mask &= (xx**2 + yy**2) <= annulus_radius**2
        
        # Extract local background
        y_slice = slice(max(0, y_int-annulus_radius), min(image_data.shape[0], y_int+annulus_radius+1))
        x_slice = slice(max(0, x_int-annulus_radius), min(image_data.shape[1], x_int+annulus_radius+1))
        
        local_background = image_data[y_slice, x_slice]
        
        # Apply annulus mask to the local background
        mask_y, mask_x = annulus_mask.shape
        local_y, local_x = local_background.shape
        
        # Center the annulus mask on the local background
        start_y = (local_y - mask_y) // 2
        start_x = (local_x - mask_x) // 2
        end_y = start_y + mask_y
        end_x = start_x + mask_x
        
        if start_y >= 0 and end_y <= local_y and start_x >= 0 and end_x <= local_x:
            local_annulus_mask = annulus_mask
            local_background_annulus = local_background[start_y:end_y, start_x:end_x][local_annulus_mask]
        else:
            # Fallback: use entire local background
            local_background_annulus = local_background.flatten()
        
        if len(local_background_annulus) == 0:
            return np.nan
        
        local_rms = np.std(local_background_annulus)
        n_pixels = np.pi * aperture_radius**2
        return local_rms * np.sqrt(n_pixels)
    except Exception as e:
        print(f"Error in local RMS calculation: {e}")
        return np.nan

def estimate_error_weighted_rms(rms_map, x, y, aperture_radius=5):
    """Estimate error using weighted RMS map"""
    try:
        y_int, x_int = int(y), int(x)
        
        # Extract local RMS values
        y_min = max(0, y_int - aperture_radius)
        y_max = min(rms_map.shape[0], y_int + aperture_radius + 1)
        x_min = max(0, x_int - aperture_radius)
        x_max = min(rms_map.shape[1], x_int + aperture_radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return np.nan
            
        local_rms = rms_map[y_min:y_max, x_min:x_max]
        
        if local_rms.size == 0:
            return np.nan
        
        # Create aperture mask
        yy, xx = np.ogrid[:local_rms.shape[0], :local_rms.shape[1]]
        center_y, center_x = local_rms.shape[0] // 2, local_rms.shape[1] // 2
        aperture_mask = (xx - center_x)**2 + (yy - center_y)**2 <= aperture_radius**2
        
        aperture_rms = local_rms[aperture_mask]
        if len(aperture_rms) == 0:
            return np.nan
            
        # Sum in quadrature over the aperture
        aperture_pixels = (aperture_rms**2).sum()
        return np.sqrt(aperture_pixels)
    except Exception as e:
        print(f"Error in weighted RMS calculation: {e}")
        return np.nan

def estimate_error_global_stats(image_data, seg_map, aperture_radius=5):
    """Estimate error using global background statistics"""
    try:
        # Calculate background statistics once per image
        bkg_mask = (seg_map == 0) & (image_data != 0)
        bkg_pixels = image_data[bkg_mask]
        
        if len(bkg_pixels) == 0:
            return np.nan
        
        bkg_std = np.std(bkg_pixels)
        n_pixels = np.pi * aperture_radius**2
        return bkg_std * np.sqrt(n_pixels)
    except Exception as e:
        print(f"Error in global stats calculation: {e}")
        return np.nan

def read_sextractor_catalog(filepath):
    """Read SExtractor catalog"""
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
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_image_paths(pointing, filter_name, dr_version='dr0.5'):
    """Get paths to image files"""
    base_image_dir = os.path.join(image_base_dir, pointing)
    base_segmentation_dir = os.path.join(segmentation_base_dir, pointing, 'segmentations_z7')
    
    if filter_name.upper() in ['F606W', 'F814W']:
        # HST filters
        fits_path = os.path.join(base_image_dir, 
                               f"egs_all_acs_wfc_{filter_name.lower()}_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits")
        rms_path = os.path.join(base_image_dir,
                              f"egs_all_acs_wfc_{filter_name.lower()}_030mas_v1.9_{pointing}_mef_RMS.fits")
        seg_path = os.path.join(base_segmentation_dir, f"f150dropout_{filter_name.lower()}_segmentation.fits")
    else:
        # JWST filters
        fits_path = os.path.join(base_image_dir,
                               f"hlsp_ceers_jwst_nircam_{pointing}_{filter_name.lower()}_{dr_version}_i2d_SCI_BKSUB_c.fits")
        rms_path = os.path.join(base_image_dir,
                              f"hlsp_ceers_jwst_nircam_{pointing}_{filter_name.lower()}_{dr_version}_i2d_RMS.fits")
        seg_path = os.path.join(base_segmentation_dir, f"f150dropout_{filter_name.lower()}_segmentation.fits")
    
    return fits_path, rms_path, seg_path

def get_nmad_flux_error(pointing, source_id, filter_name):
    """Get NMAD flux and error from the EAZY catalog"""
    eazy_file = os.path.join(eazy_catalog_dir, f"{pointing}_eazy_catalogue_54_gal.cat")
    
    if not os.path.exists(eazy_file):
        return None, None
    
    eazy_data = read_eazy_catalog_compatible(eazy_file)
    if eazy_data is None or len(eazy_data) == 0:
        return None, None
    
    source_data = eazy_data[eazy_data['id'] == source_id]
    if len(source_data) == 0:
        return None, None
    
    # Use lowercase column names as in your file
    flux_col = f'f_{filter_mapping[filter_name.lower()]}'.lower()
    err_col = f'e_{filter_mapping[filter_name.lower()]}'.lower()
    
    if flux_col in source_data.columns and err_col in source_data.columns:
        flux = source_data[flux_col].iloc[0]
        error = source_data[err_col].iloc[0]
        return flux, error
    
    return None, None

def calculate_snr_all_methods(pointing, source_id, filter_name):
    """Calculate SNR using all methods for a single source and filter"""
    try:
        # Get source position from SExtractor catalog
        catalog_dir = os.path.join(sextractor_base_dir, pointing, 'catalogue_z7')
        cat_path = os.path.join(catalog_dir, f"f150dropout_{filter_name.lower()}_catalog.cat")
        
        sex_data = read_sextractor_catalog(cat_path)
        if sex_data is None:
            return None
        
        source_mask = sex_data['NUMBER'] == source_id
        if not np.any(source_mask):
            return None
        
        source_idx = np.where(source_mask)[0][0]
        x, y = sex_data['X_IMAGE'][source_idx], sex_data['Y_IMAGE'][source_idx]
        sex_flux = sex_data['FLUX_APER'][source_idx]
        sex_fluxerr = sex_data['FLUXERR_APER'][source_idx]
        sex_snr = sex_flux / sex_fluxerr if sex_fluxerr > 0 else 0
        
        # Get NMAD flux and error from catalog
        nmad_flux, nmad_error = get_nmad_flux_error(pointing, source_id, filter_name)
        if nmad_flux is None or nmad_error is None:
            nmad_snr = 0
        else:
            nmad_snr = nmad_flux / nmad_error if nmad_error > 0 else 0
        
        # Get image data for other error methods
        fits_path, rms_path, seg_path = get_image_paths(pointing, filter_name)
        
        if not all(os.path.exists(path) for path in [fits_path, rms_path, seg_path]):
            # If images not available, still return NMAD results
            return {
                'pointing': pointing,
                'source_id': source_id,
                'filter': filter_name.upper(),
                'x_image': x,
                'y_image': y,
                'flux_ujy': np.nan,
                'sex_snr': sex_snr,
                'nmad_snr': nmad_snr,
                'local_rms_snr': np.nan,
                'weighted_rms_snr': np.nan,
                'global_stats_snr': np.nan,
                'sex_flux': sex_flux,
                'nmad_flux': nmad_flux if nmad_flux else np.nan,
                'nmad_error': nmad_error if nmad_error else np.nan
            }
        
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
        with fits.open(rms_path) as hdul:
            rms_data = hdul[0].data
        with fits.open(seg_path) as hdul:
            seg_data = hdul[0].data
        
        # Calculate errors using different methods
        local_rms_error = estimate_error_local_rms(image_data, x, y)
        weighted_rms_error = estimate_error_weighted_rms(rms_data, x, y)
        global_stats_error = estimate_error_global_stats(image_data, seg_data)
        
        # Convert to microJanskys and calculate SNR
        if filter_name.upper() in hst_photflam_lamda:
            photflam = hst_photflam_lamda[filter_name.upper()]['photflam']
            lamda = hst_photflam_lamda[filter_name.upper()]['lamda']
            
            flux_ujy = hst_flux_to_ujy(sex_flux, photflam, lamda)
            local_rms_snr = flux_ujy / hst_flux_to_ujy(local_rms_error, photflam, lamda) if local_rms_error > 0 else 0
            weighted_rms_snr = flux_ujy / hst_flux_to_ujy(weighted_rms_error, photflam, lamda) if weighted_rms_error > 0 else 0
            global_stats_snr = flux_ujy / hst_flux_to_ujy(global_stats_error, photflam, lamda) if global_stats_error > 0 else 0
        else:
            pixar_sr = flux_conversion_scale(fits_path)
            flux_ujy = sex_flux * pixar_sr * 1e12
            local_rms_snr = flux_ujy / (local_rms_error * pixar_sr * 1e12) if local_rms_error > 0 else 0
            weighted_rms_snr = flux_ujy / (weighted_rms_error * pixar_sr * 1e12) if weighted_rms_error > 0 else 0
            global_stats_snr = flux_ujy / (global_stats_error * pixar_sr * 1e12) if global_stats_error > 0 else 0
        
        return {
            'pointing': pointing,
            'source_id': source_id,
            'filter': filter_name.upper(),
            'x_image': x,
            'y_image': y,
            'flux_ujy': flux_ujy,
            'sex_snr': sex_snr,
            'nmad_snr': nmad_snr,
            'local_rms_snr': local_rms_snr,
            'weighted_rms_snr': weighted_rms_snr,
            'global_stats_snr': global_stats_snr,
            'sex_flux': sex_flux,
            'nmad_flux': nmad_flux if nmad_flux else np.nan,
            'nmad_error': nmad_error if nmad_error else np.nan,
            'local_rms_error': local_rms_error,
            'weighted_rms_error': weighted_rms_error,
            'global_stats_error': global_stats_error
        }
        
    except Exception as e:
        print(f"Error processing {pointing} source {source_id} filter {filter_name}: {e}")
        return None

# [Keep the plotting functions create_comparison_plots and create_per_filter_plots the same as before]

def create_comparison_plots(comparison_data, output_dir):
    """Create comprehensive comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ['nmad_snr', 'local_rms_snr', 'weighted_rms_snr', 'global_stats_snr']
    method_names = ['NMAD', 'Local RMS', 'Weighted RMS', 'Global Stats']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Filter out invalid SNR values
    valid_data = comparison_data[
        (comparison_data['sex_snr'] > 0) & 
        (comparison_data['sex_snr'] < 1000) &
        (comparison_data['nmad_snr'] > 0) &
        (comparison_data['nmad_snr'] < 1000)
    ].copy()
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # 1. Overall comparison histogram
    plt.figure(figsize=(15, 10))
    
    for i, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(2, 3, i+1)
        snr_values = valid_data[method]
        valid_snr = snr_values[snr_values > 0]
        
        if len(valid_snr) > 0:
            plt.hist(valid_snr, bins=30, alpha=0.7, color=color, label=method_name)
            plt.xlabel('SNR')
            plt.ylabel('Frequency')
            plt.title(f'{method_name} SNR Distribution\nN={len(valid_snr)}')
        else:
            plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    for method, method_name, color in zip(methods, method_names, colors):
        snr_values = valid_data[method]
        valid_snr = snr_values[snr_values > 0]
        if len(valid_snr) > 0:
            plt.hist(valid_snr, bins=30, alpha=0.5, color=color, label=method_name)
    
    plt.xlabel('SNR')
    plt.ylabel('Frequency')
    plt.title('All Methods SNR Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    sex_snr = valid_data['sex_snr']
    valid_sex = sex_snr[sex_snr > 0]
    if len(valid_sex) > 0:
        plt.hist(valid_sex, bins=30, alpha=0.7, color='purple', label='SExtractor')
        plt.xlabel('SNR')
        plt.ylabel('Frequency')
        plt.title(f'SExtractor SNR Distribution\nN={len(valid_sex)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_distributions_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 1:1 comparison plots
    n_methods = len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
        method_snr = valid_data[method]
        sex_snr = valid_data['sex_snr']
        
        # Filter valid points
        valid_mask = (method_snr > 0) & (sex_snr > 0)
        valid_method = method_snr[valid_mask]
        valid_sex = sex_snr[valid_mask]
        
        if len(valid_method) == 0:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        axes[i].scatter(valid_sex, valid_method, alpha=0.6, color=color, s=30)
        
        # Plot 1:1 line
        max_val = max(valid_sex.max(), valid_method.max())
        axes[i].plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        # Calculate statistics
        diff = valid_method - valid_sex
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        median_ratio = np.median(valid_method / valid_sex)
        
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel('SExtractor SNR')
        axes[i].set_ylabel(f'{method_name} SNR')
        axes[i].set_title(f'{method_name} vs SExtractor SNR\nN={len(valid_method)}, Mean diff: {mean_diff:.2f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_1to1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to compare all SNR calculation methods"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting comprehensive SNR method comparison...")
    all_results = []
    
    # Process each pointing and source
    for pointing in pointings:
        if pointing not in highlight_ids:
            continue
            
        print(f"\nProcessing {pointing}...")
        
        for source_id in highlight_ids[pointing]:
            print(f"  Source {source_id}...")
            
            for filter_name in filters:
                result = calculate_snr_all_methods(pointing, source_id, filter_name)
                if result is not None:
                    all_results.append(result)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(all_results)
    
    if len(comparison_df) == 0:
        print("No data was processed successfully!")
        return
    
    # Save results
    comparison_df.to_csv(os.path.join(output_dir, 'snr_method_comparison_results.csv'), index=False)
    
    # Create summary statistics
    summary_stats = comparison_df.groupby('filter').agg({
        'sex_snr': ['mean', 'std', 'count'],
        'nmad_snr': ['mean', 'std', 'count'],
        'local_rms_snr': ['mean', 'std', 'count'],
        'weighted_rms_snr': ['mean', 'std', 'count'],
        'global_stats_snr': ['mean', 'std', 'count']
    }).round(3)
    
    summary_stats.to_csv(os.path.join(output_dir, 'snr_method_summary_stats.csv'))
    
    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(comparison_df, output_dir)
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS:")
    print("="*60)
    
    methods = ['sex_snr', 'nmad_snr', 'local_rms_snr', 'weighted_rms_snr', 'global_stats_snr']
    method_names = ['SExtractor', 'NMAD', 'Local RMS', 'Weighted RMS', 'Global Stats']
    
    for method, name in zip(methods, method_names):
        valid_snr = comparison_df[method][comparison_df[method] > 0]
        if len(valid_snr) > 0:
            print(f"{name:12} - Mean: {valid_snr.mean():.2f}, Median: {valid_snr.median():.2f}, "
                  f"Std: {valid_snr.std():.2f}, N: {len(valid_snr)}")
    
    print(f"\nTotal measurements: {len(comparison_df)}")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()