import pandas as pd
from astropy.io import fits
import numpy as np
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, aperture_photometry
import os
from skimage.morphology import disk
from scipy.ndimage import binary_erosion
from matplotlib.patches import Circle
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import threading
import queue
import psutil

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

def get_input_with_timeout(prompt, timeout, default_value):
    """Modified to immediately return default value without waiting"""
    print(f"{prompt} [Auto-selected: {default_value}]")
    return default_value

def get_optimal_process_count():
    """Helper function to determine optimal number of processes"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
    memory = psutil.virtual_memory()
    
    # Use 75% of available cores, minimum 2
    optimal = max(2, int(cpu_count * 0.75))
    
    # Adjust based on available memory (rough estimate)
    if memory.available < 4 * 1024 * 1024 * 1024:  # Less than 4GB available
        optimal = min(optimal, 2)
    
    return optimal

def flux_conversion_scale(fits_file):
    with fits.open(fits_file) as hdul:
        pixar_sr = hdul[0].header['PIXAR_SR']
    print(f"Loaded PIXAR_SR from {fits_file}: {pixar_sr}")
    return pixar_sr

def nmad(data):
    return 1.48 * np.median(np.abs(data - np.median(data)))

def hst_flux_to_ujy(sextractor_flux, photflam, lamda):
    flux_erg = sextractor_flux * photflam
    flux_jy = 3.34e4 * (lamda ** 2) * flux_erg
    flux_ujy = flux_jy * 1e6
    return flux_ujy

def find_closest_empty_apertures(x, y, kdtree, num_apertures=200, initial_search_radius=8000, min_distance=7, max_attempts=20):
    selected_apertures = []
    attempt = 0
    search_radius = initial_search_radius
    
    print(f"Starting search for empty apertures around ({x}, {y})")
    
    while len(selected_apertures) < num_apertures and attempt < max_attempts:
        print(f"\nAttempt {attempt + 1}: Searching within radius {search_radius}")
        distances, indices = kdtree.query([x, y], k=search_radius)
        print(f"Found {len(distances)} apertures in the search area")
        
        mask_outside_radius = distances > 10
        valid_indices = indices[mask_outside_radius]
        print(f"Valid indices count after applying mask: {len(valid_indices)}")
        
        for idx in valid_indices:
            candidate_aperture = kdtree.data[idx]
            is_valid = True
            
            for aperture in selected_apertures:
                distance_to_existing = np.linalg.norm(candidate_aperture - aperture)
                if distance_to_existing < min_distance:
                    is_valid = False
                    break
            
            if is_valid:
                selected_apertures.append(candidate_aperture)
            
            if len(selected_apertures) >= num_apertures:
                print("Reached required number of apertures, stopping search.")
                break
                
        if len(selected_apertures) < num_apertures:
            search_radius += 200
            attempt += 1
            print(f"Increasing search radius to {search_radius} pixels")
    
    print(f"\nFinal selection: {len(selected_apertures)} apertures found with minimal overlap for source at ({x}, {y})")
    return np.array(selected_apertures)

def load_high_z_catalog(catalog_file):
    """Load the high-z catalog file"""
    print(f"Loading high-z catalog from: {catalog_file}")
    try:
        # Try reading as CSV first
        df = pd.read_csv(catalog_file, sep='\t')
        print(f"Successfully loaded catalog with {len(df)} sources")
        return df
    except Exception as e:
        print(f"Error loading catalog as CSV: {e}")
        try:
            # Try reading as space-separated
            df = pd.read_csv(catalog_file, sep='\s+')
            print(f"Successfully loaded catalog with {len(df)} sources")
            return df
        except Exception as e2:
            print(f"Error loading catalog: {e2}")
            return None

def create_output_directory_structure(base_output_dir, redshift_bin, pointing):
    """Create organized directory structure for outputs"""
    # Create main directories
    redshift_dir = os.path.join(base_output_dir, f"redshift_{redshift_bin}")
    pointing_dir = os.path.join(redshift_dir, pointing)
    results_dir = os.path.join(pointing_dir, "results")
    cutouts_dir = os.path.join(pointing_dir, "cutouts")
    catalogs_dir = os.path.join(pointing_dir, "catalogs")
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cutouts_dir, exist_ok=True)
    os.makedirs(catalogs_dir, exist_ok=True)
    
    return {
        'redshift_dir': redshift_dir,
        'pointing_dir': pointing_dir,
        'results_dir': results_dir,
        'cutouts_dir': cutouts_dir,
        'catalogs_dir': catalogs_dir
    }

def process_single_source(source_data, filter_name, paths, perform_empty_aperture, generate_cutouts, hst_photflam_lamda, aperture_corrections, cutouts_dir):
    """Process a single source for NMAD calculation - FIXED VERSION"""
    print(f"Processing source {source_data['NUMBER']} for filter {filter_name}")
    
    source_number = int(source_data['NUMBER'])
    x_image = source_data['X_IMAGE']
    y_image = source_data['Y_IMAGE']
    
    # Get redshift bin and pointing with proper validation
    redshift_bin = source_data.get('REDSHIFT_BIN', 'unknown') if isinstance(source_data, dict) else getattr(source_data, 'REDSHIFT_BIN', 'unknown')
    pointing = source_data.get('POINTING', 'unknown') if isinstance(source_data, dict) else getattr(source_data, 'POINTING', 'unknown')
    
    correction = aperture_corrections[filter_name]
    print(f"Aperture correction for {filter_name}: {correction}")
    
    # FIRST: Read the actual SExtractor flux values for this source
    try:
        # Read SExtractor catalog for this filter
        sex_catalog = pd.read_csv(paths['catalog'], sep=r'\s+', comment='#', header=None)
        sex_catalog.columns = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                             'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                             'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                             'ALPHA_J2000', 'DELTA_J2000']
        
        # Find this source in SExtractor catalog
        source_mask = sex_catalog['NUMBER'] == source_number
        if not source_mask.any():
            print(f"Warning: Source {source_number} not found in SExtractor catalog for {filter_name}")
            return None
            
        sex_row = sex_catalog[source_mask].iloc[0]
        sex_flux_aper = sex_row['FLUX_APER']
        sex_fluxerr_aper = sex_row['FLUXERR_APER']
        
        # Apply flux conversion
        if filter_name in hst_photflam_lamda:
            photflam = hst_photflam_lamda[filter_name]['photflam']
            lamda = hst_photflam_lamda[filter_name]['lamda']
            sex_flux_aper_uJy = hst_flux_to_ujy(sex_flux_aper, photflam, lamda) * correction
            sex_fluxerr_aper_uJy = hst_flux_to_ujy(sex_fluxerr_aper, photflam, lamda) * correction
        else:
            pixar_sr = flux_conversion_scale(paths['fits'])
            sex_flux_aper_uJy = sex_flux_aper * pixar_sr * 1e12 * correction
            sex_fluxerr_aper_uJy = sex_fluxerr_aper * pixar_sr * 1e12 * correction
            
        print(f"Source {source_number} - {filter_name}: FLUX_APER = {sex_flux_aper_uJy:.6f} µJy")
        
    except Exception as e:
        print(f"Error reading SExtractor flux for source {source_number}, filter {filter_name}: {e}")
        return None
    
    # SECOND: Calculate NMAD error if requested
    nmad_error_uJy = 0
    if perform_empty_aperture == 'yes':
        print("Performing empty aperture calculation...")
        
        try:
            with fits.open(paths['segmentation']) as hdul:
                seg_map = hdul[0].data
            with fits.open(paths['fits']) as hdul:
                image_data = hdul[0].data
            with fits.open(paths['rms']) as hdul:
                rms_map = hdul[0].data
            
            empty_region_mask = (seg_map == 0) & (rms_map != 0)
            
            kernel = disk(5)
            eroded_empty_mask = binary_erosion(empty_region_mask, structure=kernel)
            empty_pixels = np.argwhere(eroded_empty_mask)
            empty_pixel_coords = empty_pixels[:, [1, 0]]
            kdtree = cKDTree(empty_pixel_coords)
            
            closest_empty_apertures = find_closest_empty_apertures(
                x_image, y_image, kdtree, num_apertures=200
            )
            
            apertures = CircularAperture(closest_empty_apertures, r=5)
            phot_table = aperture_photometry(image_data, apertures)
            fluxes = phot_table['aperture_sum']
            photometric_error = nmad(fluxes)
            
            # Convert NMAD error to µJy
            if filter_name in hst_photflam_lamda:
                photflam = hst_photflam_lamda[filter_name]['photflam']
                lamda = hst_photflam_lamda[filter_name]['lamda']
                nmad_error_uJy = hst_flux_to_ujy(photometric_error, photflam, lamda)
            else:
                pixar_sr = flux_conversion_scale(paths['fits'])
                nmad_error_uJy = photometric_error * pixar_sr * 1e12
                
            print(f"NMAD error for {filter_name}: {nmad_error_uJy:.6f} µJy")
            
            # Generate cutouts if requested
            if generate_cutouts == 'yes':
                generate_cutout_image(source_number, filter_name, x_image, y_image, image_data, 
                                    seg_map, apertures, nmad_error_uJy, cutouts_dir)
                
        except Exception as e:
            print(f"Error in NMAD calculation for source {source_number}, filter {filter_name}: {e}")
            nmad_error_uJy = sex_fluxerr_aper_uJy  # Fallback to SExtractor error
    
    else:
        # If not doing NMAD, use SExtractor error
        nmad_error_uJy = sex_fluxerr_aper_uJy
    
    # Return both the actual flux and the proper error
    results = {
        'source_number': source_number,
        'filter': filter_name,
        'x_image': x_image,
        'y_image': y_image,
        'sex_flux_ujy': sex_flux_aper_uJy,  # ACTUAL flux value
        'nmad_error_ujy': nmad_error_uJy,   # PROPER error value
        'redshift_bin': redshift_bin,
        'pointing': pointing
    }
    
    print(f"Processing for source {source_number} in filter {filter_name} completed.")
    return results

def generate_cutout_image(source_number, filter_name, x_image, y_image, image_data, seg_map, apertures, nmad_error, cutouts_dir):
    """Generate cutout image - separated for clarity"""
    print(f"Creating cutouts for source {source_number}")
    
    cutout_size = 150
    x, y = x_image, y_image
    x_min, x_max = int(x - cutout_size // 2), int(x + cutout_size // 2)
    y_min, y_max = int(y - cutout_size // 2), int(y + cutout_size // 2)
    
    # Ensure cutout coordinates are within image bounds
    y_min = max(0, y_min)
    y_max = min(image_data.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(image_data.shape[1], x_max)
    
    cutout_image = image_data[y_min:y_max, x_min:x_max]
    cutout_seg = seg_map[y_min:y_max, x_min:x_max]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    titles = ["Image Data", "Segmentation Map"]
    cutouts = [cutout_image, cutout_seg]
    cmaps = ["gray", "gray"]
    
    for i, ax in enumerate(axes):
        ax.imshow(cutouts[i], cmap=cmaps[i], origin="lower", 
                interpolation="nearest", aspect="equal")
        ax.set_title(titles[i], fontsize=14)
        ax.axis("off")
        
        # Source aperture
        source_x = cutout_size // 2 if (x_min <= x_image <= x_max and y_min <= y_image <= y_max) else -100
        source_y = cutout_size // 2 if (x_min <= x_image <= x_max and y_min <= y_image <= y_max) else -100
        if source_x >= 0 and source_y >= 0:
            aperture = Circle((source_x, source_y), 5, edgecolor='red', facecolor='none', lw=2)
            ax.add_patch(aperture)
        
        # Empty apertures
        for coord in apertures.positions:
            aperture_x = coord[0] - x_min
            aperture_y = coord[1] - y_min
            if 0 <= aperture_x < cutout_size and 0 <= aperture_y < cutout_size:
                aperture = Circle((aperture_x, aperture_y), 5, 
                                edgecolor='green', facecolor='none', 
                                lw=1, linestyle='-')
                ax.add_patch(aperture)
    
    flux_info = f"NMAD: {nmad_error:.5f} µJy"
    plt.suptitle(f"Source {source_number} - {filter_name} - {flux_info}", fontsize=12)
    output_path = os.path.join(cutouts_dir, f"cutout_{filter_name}_source_{source_number}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved cutout: {output_path}")

def process_single_filter_for_high_z(filter_name, paths, redshift_catalog, perform_empty_aperture, generate_cutouts, hst_photflam_lamda, aperture_corrections, cutouts_dir):
    """Process all high-z sources for a single filter"""
    print(f"Processing filter: {filter_name} for high-z sources")
    results = []
    
    if len(redshift_catalog) == 0:
        print(f"No high-z sources found for filter {filter_name}")
        return filter_name, results
    
    print(f"Processing {len(redshift_catalog)} high-z sources for filter {filter_name}")
    
    for idx, source_data in redshift_catalog.iterrows():
        try:
            result = process_single_source(source_data, filter_name, paths, perform_empty_aperture, generate_cutouts, hst_photflam_lamda, aperture_corrections, cutouts_dir)
            results.append(result)
        except Exception as e:
            print(f"Error processing source {source_data['NUMBER']} for filter {filter_name}: {e}")
            continue
    
    print(f"Processing for filter {filter_name} completed. Processed {len(results)} sources.")
    return filter_name, results

def create_nmad_catalog(all_results, redshift_pointing_catalog, catalog_files, catalogs_dir, pointing, redshift_bin):
    """Create NMAD catalog with proper flux and error values - FIXED VERSION"""
    print("Creating NMAD catalog in EAZY format...")
    
    # Group results by source and filter
    source_results = {}
    for result in all_results:
        if result is None:
            continue
        source_num = result['source_number']
        filter_name = result['filter']
        
        if source_num not in source_results:
            source_results[source_num] = {}
        
        source_results[source_num][filter_name] = {
            'flux': result['sex_flux_ujy'],
            'error': result['nmad_error_ujy']
        }
    
    # Prepare catalog data
    catalog_data = []
    
    for source_number in source_results.keys():
        catalog_row = [source_number]
        
        for filter_name in catalog_files.keys():
            if filter_name in source_results[source_number]:
                # Use the actual calculated values
                flux_val = source_results[source_number][filter_name]['flux']
                error_val = source_results[source_number][filter_name]['error']
            else:
                # Source not found in this filter, use zeros
                flux_val = 0
                error_val = 0
            
            catalog_row.extend([flux_val, error_val])
        
        catalog_data.append(catalog_row)
    
    # Save NMAD catalog
    nmad_catalogue_file = os.path.join(catalogs_dir, f"{pointing}_{redshift_bin}_nmad_catalogue.cat")
    try:
        with open(nmad_catalogue_file, 'w') as f:
            # Write header
            f.write('# id ' + ' '.join([f'f_{filter_name} e_{filter_name}' for filter_name in catalog_files.keys()]) + '\n')
            
            # Write data
            for row in catalog_data:
                f.write(f"{int(row[0])} ")
                # Format numbers properly
                formatted_values = []
                for i, val in enumerate(row[1:]):
                    if i % 2 == 0:  # Flux values
                        formatted_values.append(f"{val:.8f}")
                    else:  # Error values  
                        formatted_values.append(f"{val:.8f}")
                f.write(' '.join(formatted_values) + '\n')
        
        print(f"NMAD catalog saved to {nmad_catalogue_file}")
        
        # Print first few lines for verification
        print("First 3 lines of catalog:")
        with open(nmad_catalogue_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 4:  # Header + 3 data lines
                    print(line.strip())
                else:
                    break
                    
    except Exception as e:
        print(f"Error saving NMAD catalog to {nmad_catalogue_file}: {e}")

def process_redshift_bin_pointing(redshift_bin, pointing, pointing_catalog, dr_version, base_output_dir, perform_empty_aperture, generate_cutouts):
    """Process a specific redshift bin and pointing combination - FIXED VERSION"""
    print(f"Processing redshift bin {redshift_bin} for pointing {pointing}")
    
    # Filter catalog for current redshift bin and pointing WITH VALIDATION
    if 'REDSHIFT_BIN' in pointing_catalog.columns and 'POINTING' in pointing_catalog.columns:
        redshift_pointing_catalog = pointing_catalog[
            (pointing_catalog['REDSHIFT_BIN'] == redshift_bin) & 
            (pointing_catalog['POINTING'] == pointing)
        ]
    else:
        # If columns don't exist, use all sources (fallback)
        print(f"Warning: REDSHIFT_BIN or POINTING columns not found. Using all {len(pointing_catalog)} sources.")
        redshift_pointing_catalog = pointing_catalog
    
    if len(redshift_pointing_catalog) == 0:
        print(f"No sources found for redshift bin {redshift_bin} in pointing {pointing}")
        return None
    
    print(f"Found {len(redshift_pointing_catalog)} sources for processing")
    
    # Create output directory structure
    dirs = create_output_directory_structure(base_output_dir, redshift_bin, pointing)
    
    # Updated directory paths
    base_image_dir = f"/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/{pointing}"
    base_catalog_dir = f"/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/catalogue_z7"
    base_segmentation_dir = f"/media/iit-t/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/segmentations_z7"
    
    catalog_files = {  
        "F606W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f606w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f606w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_RMS.fits"),
        },
        
        "F814W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f814w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f814w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_RMS.fits"),
        },
        
        "F115W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f115w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f115w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_RMS.fits"),
        },
        
        "F150W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f150w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f150w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_RMS.fits"),
        },
        
        "F200W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f200w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f200w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_RMS.fits"),
        },
        
        "F277W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f277w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f277w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_RMS.fits"),
        },
        
        "F356W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f356w_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f356w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_RMS.fits"),
        },
        
        "F410M": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f410m_catalog.cat"),
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f410m_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_RMS.fits"),
        },
        
        "F444W": {
            "catalog": os.path.join(base_catalog_dir, f"f150dropout_f444w_catalog.cat"),  # FIXED
            "fits": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_SCI_BKSUB_c.fits"),
            "segmentation": os.path.join(base_segmentation_dir, f"f150dropout_f444w_segmentation.fits"),
            "rms": os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_RMS.fits"),
        }
    }
    
    print(f"Catalog files set up for filters: {', '.join(catalog_files.keys())}")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    num_processes = min(optimal_processes, len(catalog_files))
    print(f"Using {num_processes} processors")
    
    # Initialize multiprocessing pool
    print(f"Initializing multiprocessing pool with {num_processes} processes...")
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for filter_name, paths in catalog_files.items():
        parallel_args.append((filter_name, paths, redshift_pointing_catalog, perform_empty_aperture, generate_cutouts, 
                            hst_photflam_lamda, aperture_corrections, dirs['cutouts_dir']))
    
    # Process filters in parallel
    print("Processing filters in parallel for high-z sources...")
    try:
        results_list = pool.starmap(process_single_filter_for_high_z, parallel_args)
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        pool.close()
        pool.join()
        return None
    
    # Close the pool
    pool.close()
    pool.join()
    print("Multiprocessing pool closed.")
    
    # Combine results from all filters
    print("Combining results from all filters...")
    all_results = []
    for filter_name, filter_results in results_list:
        all_results.extend(filter_results)
    
    # Save results
    if all_results:
        # Filter out None results
        valid_results = [r for r in all_results if r is not None]
        if not valid_results:
            print("No valid results to save.")
            return None
            
        results_df = pd.DataFrame(valid_results)
        
        # Save detailed results
        output_file = os.path.join(dirs['results_dir'], f"{pointing}_{redshift_bin}_nmad_results.txt")
        try:
            results_df.to_csv(output_file, sep='\t', index=False)
            print(f"NMAD results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
        
        # Create NMAD catalog using the dedicated function
        create_nmad_catalog(valid_results, redshift_pointing_catalog, catalog_files, dirs['catalogs_dir'], pointing, redshift_bin)
        
        return results_df
    else:
        print("No results to save.")
        return None

def process_high_z_nmad():
    """Main function to process NMAD for high-z sources with organized directory structure"""
    # Load high-z catalog
    high_z_catalog_file = "/media/iit-t/MY_SSD_1TB/Work_PhD/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/9_6_high_z_catalogues_for_nmad/all_high_z_sources_master_catalog.txt"
    high_z_catalog = load_high_z_catalog(high_z_catalog_file)
    
    if high_z_catalog is None:
        print("Failed to load high-z catalog. Exiting.")
        return
    
    # Base output directory
    base_output_dir = "/media/iit-t/MY_SSD_1TB/Work_PhD/Codes/TooFAINTtooCARE/Recheck_Romeos/Image_to_SExtractor/10_nmad_selected_15_Oct"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get processing preferences
    perform_empty_aperture = get_input_with_timeout(
        "Do you want to perform the empty aperture method? (yes/no, default: yes): ",
        5,
        "yes"
    ).lower()
    print(f"Empty aperture method: {perform_empty_aperture}")
    
    generate_cutouts = get_input_with_timeout(
        "Do you want to generate cutouts? (yes/no, default: no): ",
        5,
        "no"
    ).lower()
    print(f"Generate cutouts: {generate_cutouts}")
    
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
    
    # Get unique redshift bins and pointings
    redshift_bins = high_z_catalog['REDSHIFT_BIN'].unique() if 'REDSHIFT_BIN' in high_z_catalog.columns else ['z7-8', 'z8-10', 'z10-15'] #['z7-8', 'z8-10', 'z10-15']
    pointings = ['nircam1','nircam2','nircam3','nircam4','nircam5','nircam6','nircam7','nircam8','nircam9','nircam10']
    
    print(f"Found redshift bins: {redshift_bins}")
    print(f"Found pointings: {pointings}")
    
    # Process each combination of redshift bin and pointing
    all_results = {}
    
    for redshift_bin in redshift_bins:
        for pointing in pointings:
            print(f"\n{'='*60}")
            print(f"Processing: Redshift Bin {redshift_bin} | Pointing {pointing}")
            print(f"{'='*60}")
            
            dr_version = pointing_versions.get(pointing, 'dr0.5')
            
            try:
                result = process_redshift_bin_pointing(
                    redshift_bin, pointing, high_z_catalog, dr_version, 
                    base_output_dir, perform_empty_aperture, generate_cutouts
                )
                
                if result is not None:
                    key = f"{redshift_bin}_{pointing}"
                    all_results[key] = result
                    print(f"✅ Completed processing for {key}")
                else:
                    print(f"❌ No results for {redshift_bin}_{pointing}")
                    
            except Exception as e:
                print(f"❌ Error processing {redshift_bin}_{pointing}: {e}")
                continue
    
    # Create summary file
    if all_results:
        summary_file = os.path.join(base_output_dir, "processing_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("NMAD Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for key, result_df in all_results.items():
                redshift_bin, pointing = key.split('_', 1)
                f.write(f"Redshift Bin: {redshift_bin}\n")
                f.write(f"Pointing: {pointing}\n")
                f.write(f"Number of sources processed: {len(result_df['source_number'].unique())}\n")
                f.write(f"Number of measurements: {len(result_df)}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nProcessing complete! Summary saved to: {summary_file}")
        print(f"Output directory structure:")
        print(f"  {base_output_dir}/")
        print(f"    redshift_[bin]/")
        print(f"      [pointing]/")
        print(f"        results/")
        print(f"        cutouts/")
        print(f"        catalogs/")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    process_high_z_nmad()
   