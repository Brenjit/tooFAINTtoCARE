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
    """Helper function to get input with timeout"""
    user_input = queue.Queue()
    
    def get_input():
        try:
            text = input(prompt)
            user_input.put(text)
        except:
            user_input.put(None)
    
    # Start input thread
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    
    # Wait for input or timeout
    try:
        user_response = user_input.get(timeout=timeout)
        return user_response if user_response else default_value
    except queue.Empty:
        print(f"\nNo input received within {timeout} seconds. Using default value: {default_value}")
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

def process_single_filter(filter_name, paths, df, perform_empty_aperture, generate_cutouts, hst_photflam_lamda, aperture_corrections):
    print(f"Processing filter: {filter_name}")
    results = {}
    
    correction = aperture_corrections[filter_name]
    print(f"Aperture correction for {filter_name}: {correction}")
    
    if filter_name in hst_photflam_lamda:
        photflam = hst_photflam_lamda[filter_name]['photflam']
        lamda = hst_photflam_lamda[filter_name]['lamda']
        print(f"Using HST photflam and lamda for {filter_name}: {photflam}, {lamda}")
        
        df['FLUX_APER'] = df['FLUX_APER'].apply(lambda flux: hst_flux_to_ujy(flux, photflam, lamda)) * correction
        df['FLUXERR_APER'] = df['FLUXERR_APER'].apply(lambda flux_err: hst_flux_to_ujy(flux_err, photflam, lamda)) * correction
        print(f"Fluxes and flux errors converted to uJy for {filter_name}")
    else:
        pixar_sr = flux_conversion_scale(paths['fits'])
        print(f"Using Pixar scale factor: {pixar_sr}")
        
        df['FLUX_APER'] = df['FLUX_APER'] * pixar_sr * 1e12 * correction
        df['FLUXERR_APER'] = df['FLUXERR_APER'] * pixar_sr * 1e12 * correction
        print(f"Fluxes and flux errors converted to µJy for {filter_name}")
    
    df['SNR'] = df['FLUX_APER'] / df['FLUXERR_APER']
    print(f"Signal-to-Noise Ratio (SNR) calculated for {filter_name}")
    
    if perform_empty_aperture == 'yes':
        print("Performing empty aperture calculation...")
        
        with fits.open(paths['segmentation']) as hdul:
            seg_map = hdul[0].data
        with fits.open(paths['fits']) as hdul:
            image_data = hdul[0].data
        with fits.open(paths['rms']) as hdul:
            rms_map = hdul[0].data
        
        print("Segmentation map, image data, and RMS map loaded.")
        
        empty_region_mask = (seg_map == 0) & (rms_map != 0)
        print(f"Empty region mask created. Shape of mask: {empty_region_mask.shape}")
        
        kernel = disk(5)
        eroded_empty_mask = binary_erosion(empty_region_mask, structure=kernel)
        empty_pixels = np.argwhere(eroded_empty_mask)
        empty_pixel_coords = empty_pixels[:, [1, 0]]
        kdtree = cKDTree(empty_pixel_coords)
        
        print("Empty pixels eroded and KDTree created.")
        
        for idx, row in df.iterrows():
            source_number = int(row['NUMBER'])
            print(f"Processing source {source_number}")
            x_image = row['X_IMAGE']
            y_image = row['Y_IMAGE']
            
            closest_empty_apertures = find_closest_empty_apertures(
                x_image, y_image, kdtree, num_apertures=200
            )
            
            apertures = CircularAperture(closest_empty_apertures, r=5)
            phot_table = aperture_photometry(image_data, apertures)
            fluxes = phot_table['aperture_sum']
            photometric_error = nmad(fluxes)
            
            if generate_cutouts == 'yes':
                print(f"Creating cutouts for source {source_number}")
                output_dir = "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/EAZY/eazy-photoz/cutout_nmad"
                os.makedirs(output_dir, exist_ok=True)
                
                cutout_size = 150
                x, y = x_image, y_image
                x_min, x_max = int(x - cutout_size // 2), int(x + cutout_size // 2)
                y_min, y_max = int(y - cutout_size // 2), int(y + cutout_size // 2)
                
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
                    
                    aperture = Circle((cutout_size // 2, cutout_size // 2), 
                                    5, edgecolor='red', facecolor='none', lw=2)
                    ax.add_patch(aperture)
                    
                    for coord in apertures.positions:
                        aperture_x = coord[0] - (x_image - cutout_size // 2)
                        aperture_y = coord[1] - (y_image - cutout_size // 2)
                        aperture = Circle((aperture_x, aperture_y), 5, 
                                        edgecolor='green', facecolor='none', 
                                        lw=1, linestyle='-')
                        ax.add_patch(aperture)
                
                if filter_name in hst_photflam_lamda:
                    photometric_error_uJy = hst_flux_to_ujy(photometric_error, photflam, lamda)
                    flux_info = f"HST NMAD: {photometric_error_uJy:.5f} µJy"
                else:
                    photometric_error_uJy = photometric_error * pixar_sr * 1e12
                    flux_info = f"JWST NMAD: {photometric_error_uJy:.5f} µJy"
                
                plt.suptitle(f"Source {source_number} - {flux_info}", fontsize=12)
                output_path = os.path.join(output_dir, f"cutout_{filter_name}_source_{source_number}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Saved cutout: {output_path}")
            
            if filter_name in hst_photflam_lamda:
                photometric_error_uJy = hst_flux_to_ujy(photometric_error, photflam, lamda)
            else:
                photometric_error_uJy = photometric_error * pixar_sr * 1e12
            
            results[source_number] = {
                'x_image': row['X_IMAGE'],
                'y_image': row['Y_IMAGE'],
                'class_star': row['CLASS_STAR'],
                f'mag_aper_{filter_name}': row['MAG_APER'],
                f'magerr_aper_{filter_name}': row['MAGERR_APER'],
                f'flux_aper_{filter_name}': row['FLUX_APER'],
                f'fluxerr_aper_{filter_name}': row['FLUXERR_APER'],
                f'snr_{filter_name}': row['SNR'],
                f'nmad_{filter_name}': photometric_error_uJy
            }
    else:
        print("Skipping empty aperture calculation.")
        
        for _, row in df.iterrows():
            source_number = int(row['NUMBER'])
            results[source_number] = {
                'x_image': row['X_IMAGE'],
                'y_image': row['Y_IMAGE'],
                'class_star': row['CLASS_STAR'],
                f'mag_aper_{filter_name}': row['MAG_APER'],
                f'magerr_aper_{filter_name}': row['MAGERR_APER'],
                f'flux_aper_{filter_name}': row['FLUX_APER'],
                f'fluxerr_aper_{filter_name}': row['FLUXERR_APER'],
                f'snr_{filter_name}': row['SNR']
            }
    
    print(f"Processing for filter {filter_name} completed.")
    return filter_name, results

def process_dropouts(dropout_type='z~12', pointing='', dr_version=''):
    print(f"Processing for {dropout_type} dropout")
    
    output_file = f'/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/EAZY/result/{pointing}/{pointing}_output.txt'
    fluxerr_catalogue_file = f'/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/EAZY/eazy-photoz/inputs/{pointing}_fluxerr_catalogue.cat'
    eazy_catalogue_file = f'/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/EAZY/eazy-photoz/inputs/{pointing}_eazy_catalogue.cat'
    
    catalog_files = {  
        "F606W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f606w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f606w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_RMS.fits"
        },
        
        "F814W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f814w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f814w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_RMS.fits"
        },
        
        "F115W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f115w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f115w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_RMS.fits"
        },
        
        "F150W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f150w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f150w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f150w_{dr_version}_i2d_RMS.fits"
        },
        
        "F200W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f200w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f200w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f200w_{dr_version}_i2d_RMS.fits"
        },
        
        "F277W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f277w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f277w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f277w_{dr_version}_i2d_RMS.fits"
        },
        
        "F356W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f356w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f356w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_RMS.fits"
        },
        
        "F410M": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f410m_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f410m_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f410m_{dr_version}_i2d_RMS.fits"
        },
        
        "F444W": {
            "catalog": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/catalogue_z8/f444w_catalog.cat",
            "fits": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_SCI_BKSUB_c.fits",
            "segmentation": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/SEP_JWST/Results/nircam6/segmentations_z8/f444w_segmentation.fits",
            "rms": f"/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/nircam6/hlsp_ceers_jwst_nircam_{pointing}_f444w_{dr_version}_i2d_RMS.fits"
        }
    }
    
    print(f"Catalog files set up for filters: {', '.join(catalog_files.keys())}")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    process_prompt = f"Enter number of processors to use (optimal is {optimal_processes}, press Enter for optimal): "
    num_processes = get_input_with_timeout(process_prompt, 30, str(optimal_processes))
    try:
        num_processes = int(num_processes)
    except ValueError:
        num_processes = optimal_processes
    print(f"Using {num_processes} processors")
    
    # Get empty aperture preference with 30s timeout
    perform_empty_aperture = get_input_with_timeout(
        "Do you want to perform the empty aperture method? (yes/no, default: yes): ",
        30,
        "yes"
    ).lower()
    print(f"Empty aperture method: {perform_empty_aperture}")
    
    # Get cutout preference with 30s timeout
    generate_cutouts = get_input_with_timeout(
        "Do you want to generate cutouts? (yes/no, default: no): ",
        30,
        "no"
    ).lower()
    print(f"Generate cutouts: {generate_cutouts}")
    
    # Initialize multiprocessing pool
    print(f"Initializing multiprocessing pool with {num_processes} processes...")
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for filter_name, paths in catalog_files.items():
        print(f"Reading catalog file: {paths['catalog']}")
        try:
            df = pd.read_csv(paths['catalog'], sep=r'\s+', comment='#', header=None)
            df.columns = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                         'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                         'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                         'ALPHA_J2000', 'DELTA_J2000']
            print(f"Catalog file {paths['catalog']} read successfully with {len(df)} entries.")
        except Exception as e:
            print(f"Error reading catalog {paths['catalog']}: {e}")
            continue
        
        parallel_args.append((filter_name, paths, df, perform_empty_aperture, generate_cutouts, 
                            hst_photflam_lamda, aperture_corrections))
    
    # Process filters in parallel
    print("Processing filters in parallel...")
    try:
        results_list = pool.starmap(process_single_filter, parallel_args)
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        pool.close()
        pool.join()
        return
    
    # Close the pool
    pool.close()
    pool.join()
    print("Multiprocessing pool closed.")
    
    # Combine results from all filters
    print("Combining results from all filters...")
    combined_results = {}
    for filter_name, filter_results in results_list:
        for source_number, source_data in filter_results.items():
            if source_number not in combined_results:
                combined_results[source_number] = {}
            combined_results[source_number].update(source_data)
    
    # Convert results to DataFrame and save
    print("Converting combined results to DataFrame...")
    results_df = pd.DataFrame.from_dict(combined_results, orient='index').reset_index()
    results_df.rename(columns={'index': 'source_number'}, inplace=True)
    try:
        results_df.to_csv(output_file, sep='\t', index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")
    
    # Create EAZY or fluxerr catalog based on perform_empty_aperture
    if perform_empty_aperture == 'yes':
        catalog_file = eazy_catalogue_file
        error_key = 'nmad'
    else:
        catalog_file = fluxerr_catalogue_file
        error_key = 'fluxerr_aper'
    
    # Prepare catalog data
    print(f"Preparing catalog data with {error_key}...")
    catalog_columns = ['# id']
    for filter_name in catalog_files.keys():
        catalog_columns.extend([f'f_{filter_name}', f'e_{filter_name}'])
    
    catalog_data = []
    for source_number, data in combined_results.items():
        catalog_row = [source_number]
        for filter_name in catalog_files.keys():
            catalog_row.append(data.get(f'flux_aper_{filter_name}', 0))
            catalog_row.append(data.get(f'{error_key}_{filter_name}', 0))
        catalog_data.append(catalog_row)
    
    # Save catalog
    print(f"Saving catalog to {catalog_file}...")
    try:
        catalog_df = pd.DataFrame(catalog_data, columns=catalog_columns)
        with open(catalog_file, 'w') as f:
            f.write('# id ' + ' '.join([f'f_{filter_name} e_{filter_name}' for filter_name in catalog_files.keys()]) + '\n')
            for idx, row in catalog_df.iterrows():
                f.write(f"{int(row['# id'])} ")
                f.write(' '.join(f"{value}" for value in row[1:].values) + '\n')
        print(f"Catalog saved to {catalog_file}")
    except Exception as e:
        print(f"Error saving catalog to {catalog_file}: {e}")

if __name__ == "__main__":
    # Customize these values as needed:
    dropout_type = 'z~12'
    pointing = 'nircam6'  # Change this to the correct pointing name you are using
    dr_version = 'dr0.5'   # Set this to your actual DR version

    process_dropouts(dropout_type=dropout_type, pointing=pointing, dr_version=dr_version)
    print("Processing completed.")