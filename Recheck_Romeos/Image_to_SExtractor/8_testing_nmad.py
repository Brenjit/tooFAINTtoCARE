import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.io import fits
from skimage.morphology import disk
from scipy.ndimage import binary_erosion
import pandas as pd
import os
import glob

# Filter-specific parameters for HST
hst_photflam_lamda = {
    'F606W': {'photflam': 7.79611686490250e-20, 'lamda': 5920.405914484679},
    'F814W': {'photflam': 7.05123239329114e-20, 'lamda': 8046.166645443039}
}

# Aperture corrections for each filter
aperture_corrections = {
    'F606W': 1.2676,
    'F814W': 1.3017,
    'F115W': 1.2250,
    'F150W': 1.2168,
    'F200W': 1.2241,
    'F277W': 1.4198,
    'F356W': 1.5421,
    'F410M': 1.6162,
    'F444W': 1.6507
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

def jwst_flux_to_ujy(sextractor_flux, pixar_sr):
    """Convert JWST flux to microJanskys"""
    return sextractor_flux * pixar_sr * 1e12

def flux_conversion_scale(fits_file):
    """Get PIXAR_SR conversion factor from FITS header"""
    with fits.open(fits_file) as hdul:
        pixar_sr = hdul[0].header['PIXAR_SR']
    return pixar_sr

def get_source_coordinates(catalog_path, source_number=1):
    """Extract coordinates of a specific source from SExtractor catalog"""
    try:
        df = pd.read_csv(catalog_path, sep=r'\s+', comment='#', header=None)
        df.columns = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                     'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                     'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                     'ALPHA_J2000', 'DELTA_J2000', 'SNR']
        
        if source_number > len(df):
            print(f"Source number {source_number} not found in catalog. Using first source.")
            source_number = 1
            
        source = df.iloc[source_number - 1]
        return source['X_IMAGE'], source['Y_IMAGE'], source['NUMBER']
    except Exception as e:
        print(f"Error reading catalog {catalog_path}: {e}")
        return None, None, None

def analyze_error_for_source(pointing, dr_version, source_number=1, output_base_dir="./7_", max_apertures=300):
    """
    Analyze error estimation for one source across all filters in a pointing
    
    Parameters:
    -----------
    pointing : str
        The pointing name (e.g., 'nircam1')
    dr_version : str
        Data release version (e.g., 'dr0.5')
    source_number : int
        Which source to analyze (default: first source)
    output_base_dir : str
        Base directory for output
    max_apertures : int
        Maximum number of apertures to test
    """
    
    # Define directory paths
    base_image_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/{pointing}"
    base_catalog_dir = f"/Users/brenjithazarika/Desktop/PhD_computer/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/5_new_Diagnostic_Analysis/catalogue_lyman_filtered/{pointing}"
    base_segmentation_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/segmentations_z7"
    
    # Create output directory for this pointing
    output_dir = os.path.join(output_base_dir, f"{pointing}_source_{source_number}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define catalog files for all filters
    catalog_files = {  
        "F606W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f606w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f606w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_RMS.fits")
        },          
        
        "F814W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f814w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f814w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_RMS.fits")
        },
        
        "F115W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f115w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f115w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_RMS.fits")
        },
        
        "F150W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f150w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f150w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_RMS.fits")
        },
        
        "F200W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f200w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f200w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_RMS.fits")
        },
        
        "F277W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f277w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f277w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_RMS.fits")
        },
        
        "F356W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f356w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f356w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_RMS.fits")
        },
        
        "F410M": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f410m_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f410m_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_RMS.fits")
        },
        
        "F444W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f444w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f444w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_RMS.fits")
        }
    }
    
    # Store results for all filters
    all_results = {}
    
    # Process each filter
    for filter_name, paths in catalog_files.items():
        print(f"\nProcessing {filter_name} filter...")
        
        # Check if files exist
        missing_files = []
        for file_type, path in paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{file_type}: {path}")
        
        if missing_files:
            print(f"Skipping {filter_name} - missing files: {', '.join(missing_files)}")
            continue
        
        # Get source coordinates
        x, y, actual_source_number = get_source_coordinates(paths['catalog'], source_number)
        if x is None or y is None:
            print(f"Skipping {filter_name} - could not get source coordinates")
            continue
        
        print(f"Analyzing source {actual_source_number} at ({x:.2f}, {y:.2f}) in {filter_name}")
        
        # Perform error estimation analysis
        try:
            aperture_counts, error_estimates, error_estimates_ujy = analyze_single_filter(
                x, y, paths['fits'], paths['segmentation'], paths['rms'],
                filter_name, max_apertures=max_apertures, step_size=10, aperture_radius=5,
                output_dir=output_dir, source_number=actual_source_number
            )
            
            all_results[filter_name] = {
                'aperture_counts': aperture_counts,
                'error_estimates': error_estimates,  # Raw flux errors
                'error_estimates_ujy': error_estimates_ujy,  # Errors in µJy
                'source_x': x,
                'source_y': y,
                'source_number': actual_source_number
            }
            
        except Exception as e:
            print(f"Error analyzing {filter_name}: {e}")
            continue
    
    # Create summary plots comparing all filters
    create_summary_plots(all_results, output_dir, pointing, source_number)
    
    # Save comprehensive results
    save_comprehensive_results(all_results, output_dir, pointing, source_number)
    
    return all_results

def analyze_single_filter(source_x, source_y, fits_path, segmentation_path, rms_path,
                         filter_name, max_apertures=300, step_size=10, aperture_radius=5,
                         output_dir="./", source_number=1):
    """
    Analyze error estimation for a single filter with proper flux conversion
    """
    
    # Load the data
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data
    with fits.open(segmentation_path) as hdul:
        seg_map = hdul[0].data
    with fits.open(rms_path) as hdul:
        rms_map = hdul[0].data
    
    # Get aperture correction for this filter
    aperture_correction = aperture_corrections.get(filter_name, 1.0)
    print(f"Using aperture correction {aperture_correction} for {filter_name}")
    
    # Get conversion factors for flux to µJy
    if filter_name in hst_photflam_lamda:
        # HST filters
        photflam = hst_photflam_lamda[filter_name]['photflam']
        lamda = hst_photflam_lamda[filter_name]['lamda']
        print(f"HST filter {filter_name}: photflam={photflam}, lambda={lamda}")
    else:
        # JWST filters
        pixar_sr = flux_conversion_scale(fits_path)
        print(f"JWST filter {filter_name}: PIXAR_SR={pixar_sr}")
    
    # Create empty region mask (RMS file helps identify valid regions)
    empty_region_mask = (seg_map == 0) & (rms_map != 0)
    kernel = disk(5)
    eroded_empty_mask = binary_erosion(empty_region_mask, structure=kernel)
    empty_pixels = np.argwhere(eroded_empty_mask)
    
    if len(empty_pixels) == 0:
        raise ValueError("No empty regions found for error estimation")
    
    empty_pixel_coords = empty_pixels[:, [1, 0]]
    kdtree = cKDTree(empty_pixel_coords)
    
    # Find empty apertures around the source
    distances, indices = kdtree.query([source_x, source_y], k=min(len(empty_pixel_coords), 10000))
    
    # Filter out apertures too close to the source
    mask = distances > 10  # Exclude apertures within 10 pixels of the source
    valid_indices = indices[mask]
    valid_coords = empty_pixel_coords[valid_indices]
    
    if len(valid_coords) == 0:
        raise ValueError("No valid empty apertures found around the source")
    
    # Calculate error estimates for different numbers of apertures
    aperture_counts = list(range(step_size, min(max_apertures, len(valid_coords)) + 1, step_size))
    error_estimates = []  # Raw flux errors
    error_estimates_ujy = []  # Errors in µJy
    
    for n_apertures in aperture_counts:            
        # Select the first n_apertures coordinates
        selected_coords = valid_coords[:n_apertures]
        
        # Perform aperture photometry
        apertures = CircularAperture(selected_coords, r=aperture_radius)
        phot_table = aperture_photometry(image_data, apertures)
        fluxes = phot_table['aperture_sum']
        
        # Calculate error estimate (raw flux)
        error_estimate = nmad(fluxes)
        error_estimates.append(error_estimate)
        
        # Apply aperture correction
        error_corrected = error_estimate * aperture_correction
        
        # Convert to µJy
        if filter_name in hst_photflam_lamda:
            # HST conversion
            error_ujy = hst_flux_to_ujy(error_corrected, photflam, lamda)
        else:
            # JWST conversion
            error_ujy = jwst_flux_to_ujy(error_corrected, pixar_sr)
        
        error_estimates_ujy.append(error_ujy)
    
    # Create visualization for this filter
    create_single_filter_plots(aperture_counts, error_estimates, error_estimates_ujy, 
                              output_dir, filter_name, source_number, source_x, source_y)
    
    return aperture_counts, error_estimates, error_estimates_ujy

def create_single_filter_plots(aperture_counts, error_estimates, error_estimates_ujy, 
                              output_dir, filter_name, source_number, source_x, source_y):
    """Create plots for a single filter showing both raw and converted errors"""
    
    # Plot 1: Raw flux errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(aperture_counts, error_estimates, 'b-', linewidth=2)
    ax1.set_xlabel('Number of Empty Apertures')
    ax1.set_ylabel('Raw Flux Error (NMAD)')
    ax1.set_title(f'{filter_name} - Raw Flux Error vs Number of Apertures')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Converted errors in µJy
    ax2.plot(aperture_counts, error_estimates_ujy, 'r-', linewidth=2)
    ax2.set_xlabel('Number of Empty Apertures')
    ax2.set_ylabel('Error Estimate (µJy)')
    ax2.set_title(f'{filter_name} - Error in µJy vs Number of Apertures')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"error_estimation_{filter_name}_source_{source_number}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create convergence plot (relative change)
    if len(error_estimates_ujy) > 1:
        plt.figure(figsize=(10, 6))
        relative_changes = np.abs(np.diff(error_estimates_ujy) / error_estimates_ujy[:-1]) * 100
        plt.plot(aperture_counts[1:], relative_changes, 'g-', linewidth=2)
        plt.xlabel('Number of Empty Apertures')
        plt.ylabel('Relative Change in Error (%)')
        plt.title(f'{filter_name} - Convergence of Error Estimate')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='1% threshold')
        plt.legend()
        
        conv_plot_filename = os.path.join(output_dir, f"convergence_{filter_name}_source_{source_number}.png")
        plt.savefig(conv_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results to file
    results_filename = os.path.join(output_dir, f"error_estimation_{filter_name}_source_{source_number}.txt")
    with open(results_filename, 'w') as f:
        f.write(f"Error Estimation Results for {filter_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source: {source_number} at ({source_x:.2f}, {source_y:.2f})\n")
        f.write(f"Aperture correction: {aperture_corrections.get(filter_name, 1.0)}\n")
        
        if filter_name in hst_photflam_lamda:
            f.write(f"HST conversion: photflam={hst_photflam_lamda[filter_name]['photflam']}, ")
            f.write(f"lambda={hst_photflam_lamda[filter_name]['lamda']}\n")
        else:
            f.write(f"JWST conversion: PIXAR_SR={flux_conversion_scale(fits_path)}\n")
        
        f.write(f"Final raw error: {error_estimates[-1]:.6f}\n")
        f.write(f"Final error in µJy: {error_estimates_ujy[-1]:.6f}\n\n")
        f.write("N_Apertures\tRaw_Error\tError_µJy\n")
        for n, err, err_ujy in zip(aperture_counts, error_estimates, error_estimates_ujy):
            f.write(f"{n}\t{err:.6f}\t{err_ujy:.6f}\n")

def create_summary_plots(all_results, output_dir, pointing, source_number):
    """Create summary plots comparing all filters"""
    if not all_results:
        return
    
    # Plot 1: Raw flux errors
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for i, (filter_name, results) in enumerate(all_results.items()):
        aperture_counts = results['aperture_counts']
        error_estimates = results['error_estimates']
        
        plt.plot(aperture_counts, error_estimates, 
                label=filter_name, color=colors[i], linewidth=2)
    
    plt.xlabel('Number of Empty Apertures')
    plt.ylabel('Raw Flux Error (NMAD)')
    plt.title(f'Raw Error Comparison - {pointing} Source {source_number}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    raw_plot = os.path.join(output_dir, f"raw_error_comparison_{pointing}_source_{source_number}.png")
    plt.savefig(raw_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Converted errors in µJy
    plt.figure(figsize=(12, 8))
    
    for i, (filter_name, results) in enumerate(all_results.items()):
        aperture_counts = results['aperture_counts']
        error_estimates_ujy = results['error_estimates_ujy']
        
        plt.plot(aperture_counts, error_estimates_ujy, 
                label=filter_name, color=colors[i], linewidth=2)
    
    plt.xlabel('Number of Empty Apertures')
    plt.ylabel('Error Estimate (µJy)')
    plt.title(f'Error in µJy Comparison - {pointing} Source {source_number}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ujy_plot = os.path.join(output_dir, f"ujy_error_comparison_{pointing}_source_{source_number}.png")
    plt.savefig(ujy_plot, dpi=300, bbox_inches='tight')
    plt.close()

def save_comprehensive_results(all_results, output_dir, pointing, source_number):
    """Save comprehensive results for all filters"""
    results_filename = os.path.join(output_dir, f"comprehensive_results_{pointing}_source_{source_number}.txt")
    
    with open(results_filename, 'w') as f:
        f.write(f"Comprehensive Error Estimation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Pointing: {pointing}\n")
        f.write(f"Source: {source_number}\n\n")
        
        for filter_name, results in all_results.items():
            f.write(f"Filter: {filter_name}\n")
            f.write(f"Source coordinates: ({results['source_x']:.2f}, {results['source_y']:.2f})\n")
            f.write(f"Aperture correction: {aperture_corrections.get(filter_name, 1.0)}\n")
            
            if filter_name in hst_photflam_lamda:
                f.write(f"HST conversion parameters: ")
                f.write(f"photflam={hst_photflam_lamda[filter_name]['photflam']}, ")
                f.write(f"lambda={hst_photflam_lamda[filter_name]['lamda']}\n")
            else:
                f.write(f"JWST conversion: PIXAR_SR={flux_conversion_scale(fits_path)}\n")
            
            f.write(f"Final raw error: {results['error_estimates'][-1]:.6f}\n")
            f.write(f"Final error in µJy: {results['error_estimates_ujy'][-1]:.6f}\n")
            
            if len(results['error_estimates_ujy']) > 1:
                rel_change = np.abs((results['error_estimates_ujy'][-1] - results['error_estimates_ujy'][-2]) / 
                                   results['error_estimates_ujy'][-2]) * 100
                f.write(f"Relative change: {rel_change:.2f}%\n")
            
            f.write("-" * 40 + "\n\n")

# Main execution
if __name__ == "__main__":
    # Define which pointings use which dr_version
    pointing_versions = {
        'nircam1': 'dr0.5',
        'nircam2': 'dr0.5',
        'nircam3': 'dr0.5',
        'nircam4': 'dr0.6',
        'nircam5': 'dr0.6',
        'nircam6': 'dr0.5',
        'nircam7': 'dr0.6',
        'nircam8': 'dr0.6',
        'nircam9': 'dr0.6',
        'nircam10': 'dr0.6'
    }
    
    # Create main output directory
    output_base_dir = "./8_nmad_testing"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each pointing
    for pointing, dr_version in pointing_versions.items():
        print(f"\n{'='*60}")
        print(f"Processing pointing: {pointing} with {dr_version}")
        print(f"{'='*60}")
        
        try:
            results = analyze_error_for_source(
                pointing=pointing,
                dr_version=dr_version,
                source_number=1,  # Analyze the first source
                output_base_dir=output_base_dir,
                max_apertures=300
            )
            print(f"Completed processing for {pointing}")
        except Exception as e:
            print(f"Error processing {pointing}: {e}")
            continue
    
    print(f"\nAll processing completed. Results saved to {output_base_dir}")