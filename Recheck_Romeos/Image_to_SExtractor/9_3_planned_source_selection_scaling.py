import pandas as pd
from astropy.io import fits, ascii
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
import psutil
import random
from astropy.stats import sigma_clipped_stats

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

# Number of sources to select per bin
SOURCES_PER_BIN = 20

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
        distances, indices = kdtree.query([x, y], k=search_radius)
        
        mask_outside_radius = distances > 10
        valid_indices = indices[mask_outside_radius]
        
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
                break
                
        if len(selected_apertures) < num_apertures:
            search_radius += 200
            attempt += 1
    
    print(f"Final selection: {len(selected_apertures)} apertures found for source at ({x}, {y})")
    return np.array(selected_apertures)

def load_photometric_redshifts(pointing, z_max=8):
    """Load photometric redshifts from EAZY output with correct column names"""
    photoz_file = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/outputs_z{z_max}/{pointing}/{pointing}_output.zout"
    
    if not os.path.exists(photoz_file):
        print(f"Photoz file not found: {photoz_file}")
        return None
    
    try:
        # Read EAZY output - skip the header row with column descriptions
        photoz_df = ascii.read(photoz_file, data_start=2).to_pandas()
        
        # The columns are: id, z_spec, z_a, z_m1, chi_a, l68, u68, l95, u95, l99, u99, nfilt, q_z, z_peak, peak_prob, z_mc
        # We want z_a (best-fit redshift) and chi_a (chi-squared)
        
        # Basic quality cuts
        photoz_df = photoz_df[photoz_df['z_a'] > 0]  # Remove negative redshifts
        photoz_df = photoz_df[photoz_df['chi_a'] < 10]  # Reasonable chi-squared
        
        print(f"Loaded {len(photoz_df)} sources with photometric redshifts for {pointing}")
        print(f"Redshift range: {photoz_df['z_a'].min():.2f} - {photoz_df['z_a'].max():.2f}")
        return photoz_df
    except Exception as e:
        print(f"Error loading photoz file: {e}")
        # Try alternative reading method
        try:
            photoz_df = pd.read_csv(photoz_file, sep='\s+', comment='#', header=0)
            print(f"Successfully loaded with alternative method: {len(photoz_df)} sources")
            photoz_df = photoz_df[photoz_df['z_a'] > 0]
            photoz_df = photoz_df[photoz_df['chi_a'] < 10]
            return photoz_df
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return None

def create_redshift_bins(photoz_df, bin_edges=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15]):
    """Create redshift bins and assign sources to bins"""
    photoz_df['z_bin'] = pd.cut(photoz_df['z_a'], bins=bin_edges, labels=False)
    
    bin_counts = photoz_df['z_bin'].value_counts().sort_index()
    print("Redshift bin distribution:")
    for bin_idx, count in bin_counts.items():
        z_min = bin_edges[bin_idx] if bin_idx < len(bin_edges) else bin_edges[-1]
        z_max = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else bin_edges[-1]
        print(f"  z={z_min}-{z_max}: {count} sources")
    
    return photoz_df, bin_edges

def analyze_spatial_bias(image_data, rms_data, seg_data):
    """Analyze spatial patterns in the image to identify biased regions"""
    height, width = image_data.shape
    spatial_masks = {}
    
    # Center vs Edge regions
    center_fraction = 0.6
    edge_fraction = 0.2
    
    center_mask = np.zeros((height, width), dtype=bool)
    y_center, x_center = height // 2, width // 2
    y_radius, x_radius = int(height * center_fraction / 2), int(width * center_fraction / 2)
    center_mask[y_center-y_radius:y_center+y_radius, x_center-x_radius:x_center+x_radius] = True
    
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_width = int(min(height, width) * edge_fraction)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True
    
    spatial_masks['center'] = center_mask
    spatial_masks['edge'] = edge_mask
    spatial_masks['middle'] = ~(center_mask | edge_mask)
    
    # Noise regions
    rms_median = np.median(rms_data[rms_data > 0])
    rms_std = np.std(rms_data[rms_data > 0])
    spatial_masks['high_noise'] = rms_data > (rms_median + 2 * rms_std)
    spatial_masks['low_noise'] = rms_data < (rms_median - rms_std)
    
    # Background level analysis
    bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(image_data[seg_data == 0])
    spatial_masks['high_background'] = image_data > (bkg_median + 2 * bkg_std)
    
    return spatial_masks

def assign_spatial_categories(df, spatial_masks):
    """Assign each source to spatial categories based on its position"""
    df['spatial_category'] = 'normal'
    
    for idx, row in df.iterrows():
        x, y = int(row['X_IMAGE'] - 1), int(row['Y_IMAGE'] - 1)
        
        categories = []
        if y < spatial_masks['center'].shape[0] and x < spatial_masks['center'].shape[1]:
            if spatial_masks['center'][y, x]:
                categories.append('center')
            elif spatial_masks['edge'][y, x]:
                categories.append('edge')
            else:
                categories.append('middle')
            
            if spatial_masks['high_noise'][y, x]:
                categories.append('high_noise')
            elif spatial_masks['low_noise'][y, x]:
                categories.append('low_noise')
            
            if spatial_masks['high_background'][y, x]:
                categories.append('high_background')
        
        df.at[idx, 'spatial_category'] = '_'.join(categories) if categories else 'normal'
    
    return df

def select_balanced_sample(photoz_df, sources_per_bin=SOURCES_PER_BIN, random_state=42):
    """Select a balanced sample considering both redshift and spatial distribution"""
    selected_sources = []
    
    z_bins = photoz_df['z_bin'].unique()
    z_bins = z_bins[~np.isnan(z_bins)]
    spatial_categories = photoz_df['spatial_category'].unique()
    
    for z_bin in z_bins:
        bin_sources = photoz_df[photoz_df['z_bin'] == z_bin].copy()
        
        if len(bin_sources) == 0:
            continue
            
        spatial_counts = bin_sources['spatial_category'].value_counts()
        total_in_bin = len(bin_sources)
        
        for category in spatial_categories:
            if category in spatial_counts.index:
                proportion = spatial_counts[category] / total_in_bin
                n_select = max(1, int(sources_per_bin * proportion))
                
                category_sources = bin_sources[bin_sources['spatial_category'] == category]
                if len(category_sources) > 0:
                    selected = category_sources.sample(
                        n=min(n_select, len(category_sources)), 
                        random_state=random_state
                    )
                    selected_sources.append(selected)
    
    if selected_sources:
        balanced_sample = pd.concat(selected_sources, ignore_index=True)
        
        if len(balanced_sample) < sources_per_bin * len(z_bins):
            remaining_needed = sources_per_bin * len(z_bins) - len(balanced_sample)
            remaining_sources = photoz_df[~photoz_df.index.isin(balanced_sample.index)]
            if len(remaining_sources) > 0:
                supplemental = remaining_sources.sample(
                    n=min(remaining_needed, len(remaining_sources)), 
                    random_state=random_state
                )
                balanced_sample = pd.concat([balanced_sample, supplemental], ignore_index=True)
        
        print(f"Selected {len(balanced_sample)} sources with balanced distribution")
        return balanced_sample
    else:
        n_select = min(sources_per_bin * len(z_bins), len(photoz_df))
        return photoz_df.sample(n=n_select, random_state=random_state)

def process_single_filter_balanced(filter_name, paths, balanced_sample, perform_empty_aperture, 
                                 generate_cutouts, hst_photflam_lamda, aperture_corrections, pointing):
    """Process a single filter using balanced sample selection"""
    print(f"Processing filter: {filter_name} with balanced sample")
    results = {}
    
    if len(balanced_sample) == 0:
        print(f"No sources to process for filter {filter_name} in pointing {pointing}")
        return filter_name, results
    
    try:
        original_df = pd.read_csv(paths['catalog'], sep=r'\s+', comment='#', header=None)
        original_df.columns = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                             'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                             'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                             'ALPHA_J2000', 'DELTA_J2000']
        
        # Merge using the id column from EAZY (which corresponds to NUMBER in SExtractor catalog)
        df = original_df[original_df['NUMBER'].isin(balanced_sample['id'])].copy()
        print(f"Processing {len(df)} balanced sources for {filter_name}")
        
    except Exception as e:
        print(f"Error reading catalog {paths['catalog']}: {e}")
        return filter_name, results
    
    correction = aperture_corrections[filter_name]
    print(f"Aperture correction for {filter_name}: {correction}")
    
    if filter_name in hst_photflam_lamda:
        photflam = hst_photflam_lamda[filter_name]['photflam']
        lamda = hst_photflam_lamda[filter_name]['lamda']
        
        df['FLUX_APER'] = df['FLUX_APER'].apply(lambda flux: hst_flux_to_ujy(flux, photflam, lamda)) * correction
        df['FLUXERR_APER'] = df['FLUXERR_APER'].apply(lambda flux_err: hst_flux_to_ujy(flux_err, photflam, lamda)) * correction
    else:
        pixar_sr = flux_conversion_scale(paths['fits'])
        df['FLUX_APER'] = df['FLUX_APER'] * pixar_sr * 1e12 * correction
        df['FLUXERR_APER'] = df['FLUXERR_APER'] * pixar_sr * 1e12 * correction
    
    df['SNR'] = df['FLUX_APER'] / df['FLUXERR_APER']
    
    if perform_empty_aperture == 'yes':
        print("Performing empty aperture calculation...")
        
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
        
        for idx, row in df.iterrows():
            source_number = int(row['NUMBER'])
            x_image = row['X_IMAGE']
            y_image = row['Y_IMAGE']
            
            closest_empty_apertures = find_closest_empty_apertures(x_image, y_image, kdtree)
            apertures = CircularAperture(closest_empty_apertures, r=5)
            phot_table = aperture_photometry(image_data, apertures)
            fluxes = phot_table['aperture_sum']
            photometric_error = nmad(fluxes)
            
            if generate_cutouts == 'yes':
                output_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/cutout_nmad_balanced/{pointing}"
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

def process_dropouts_balanced(dropout_type='z~12', pointing='nircam1', dr_version='dr0.5', z_max=8):
    """Main function that uses balanced sample selection"""
    print(f"Processing for {dropout_type} dropout in {pointing} with {dr_version} (z_max={z_max})")
    
    # Load photometric redshifts
    photoz_df = load_photometric_redshifts(pointing, z_max)
    if photoz_df is None or len(photoz_df) == 0:
        print(f"No photometric redshifts available for {pointing}, using fallback method")
        return process_dropouts_fallback(pointing, dr_version)
    
    # Create redshift bins
    photoz_df, bin_edges = create_redshift_bins(photoz_df)
    
    # Analyze spatial bias
    base_image_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/{pointing}"
    base_segmentation_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/segmentations_z7"
    
    f356w_fits = os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_SCI_BKSUB_c.fits")
    f356w_rms = os.path.join(base_image_dir, f"hlsp_ceers_jwst_nircam_{pointing}_f356w_{dr_version}_i2d_RMS.fits")
    f356w_seg = os.path.join(base_segmentation_dir, f"f150dropout_f356w_segmentation.fits")
    
    try:
        with fits.open(f356w_fits) as hdul:
            image_data = hdul[0].data
        with fits.open(f356w_rms) as hdul:
            rms_data = hdul[0].data
        with fits.open(f356w_seg) as hdul:
            seg_data = hdul[0].data
        
        spatial_masks = analyze_spatial_bias(image_data, rms_data, seg_data)
        
        # We need to add X_IMAGE and Y_IMAGE to photoz_df for spatial analysis
        # This requires loading the original SExtractor catalog to get positions
        base_catalog_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/catalogue_z7"
        f356w_catalog = os.path.join(base_catalog_dir, f"f150dropout_f356w_catalog.cat")
        
        if os.path.exists(f356w_catalog):
            catalog_df = pd.read_csv(f356w_catalog, sep=r'\s+', comment='#', header=None)
            catalog_df.columns = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                                 'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                                 'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                                 'ALPHA_J2000', 'DELTA_J2000']
            
            # Merge photoz with catalog to get positions
            photoz_df = photoz_df.merge(catalog_df[['NUMBER', 'X_IMAGE', 'Y_IMAGE']], 
                                      left_on='id', right_on='NUMBER', how='left')
            
            photoz_df = assign_spatial_categories(photoz_df, spatial_masks)
            print("Spatial category distribution:")
            print(photoz_df['spatial_category'].value_counts())
        else:
            print("Catalog file not found, skipping spatial analysis")
            photoz_df['spatial_category'] = 'normal'
        
    except Exception as e:
        print(f"Error in spatial analysis: {e}")
        photoz_df['spatial_category'] = 'normal'
    
    # Select balanced sample
    balanced_sample = select_balanced_sample(photoz_df)
    
    # Directory setup
    base_catalog_dir = f"/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/SEP_JWST/Results/{pointing}/catalogue_z7"
    output_dir = f"./9_3_balanced_results_z{z_max}/{pointing}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{pointing}_output.txt")
    fluxerr_catalogue_file = os.path.join(output_dir, f"{pointing}_fluxerr_catalogue_balanced.cat")
    eazy_catalogue_file = os.path.join(output_dir, f"{pointing}_eazy_catalogue_balanced.cat")
    
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
    
    # Get processing preferences
    optimal_processes = get_optimal_process_count()
    num_processes = get_input_with_timeout(
        f"Enter number of processors to use (optimal is {optimal_processes}): ",
        5, str(optimal_processes)
    )
    try:
        num_processes = int(num_processes)
    except ValueError:
        num_processes = optimal_processes
    
    perform_empty_aperture = get_input_with_timeout(
        "Do you want to perform the empty aperture method? (yes/no): ",
        5, "yes"
    ).lower()
    
    generate_cutouts = get_input_with_timeout(
        "Do you want to generate cutouts? (yes/no): ",
        5, "no"
    ).lower()
    
    # Process filters in parallel
    pool = multiprocessing.Pool(processes=num_processes)
    parallel_args = []
    
    for filter_name, paths in catalog_files.items():
        if os.path.exists(paths['catalog']):
            parallel_args.append((filter_name, paths, balanced_sample, perform_empty_aperture, 
                                generate_cutouts, hst_photflam_lamda, aperture_corrections, pointing))
    
    try:
        results_list = pool.starmap(process_single_filter_balanced, parallel_args)
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        pool.close()
        pool.join()
        return
    
    pool.close()
    pool.join()
    
    # Combine results
    combined_results = {}
    for filter_name, filter_results in results_list:
        for source_number, source_data in filter_results.items():
            if source_number not in combined_results:
                combined_results[source_number] = {}
            combined_results[source_number].update(source_data)
    
    # Save results
    results_df = pd.DataFrame.from_dict(combined_results, orient='index').reset_index()
    results_df.rename(columns={'index': 'source_number'}, inplace=True)
    results_df.to_csv(output_file, sep='\t', index=False)
    
    # Create output catalog
    if perform_empty_aperture == 'yes':
        catalog_file = eazy_catalogue_file
        error_key = 'nmad'
    else:
        catalog_file = fluxerr_catalogue_file
        error_key = 'fluxerr_aper'
    
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
    
    with open(catalog_file, 'w') as f:
        f.write('# id ' + ' '.join([f'f_{filter_name} e_{filter_name}' for filter_name in catalog_files.keys()]) + '\n')
        for row in catalog_data:
            f.write(f"{int(row[0])} ")
            f.write(' '.join(f"{value}" for value in row[1:]) + '\n')
    
    print(f"Balanced processing completed for {pointing} with z_max={z_max}")

def process_dropouts_fallback(pointing, dr_version):
    """Fallback function if photometric redshifts are not available"""
    print(f"Using fallback random selection method for {pointing}")
    # You can implement your original random selection method here
    # For now, we'll just print a message
    print("Fallback method would use random selection instead of balanced selection")
    return None

def process_all_pointings_balanced(z_max_values=[8]):
    """Process all pointings with balanced sample selection"""
    dropout_type = 'z~12'
    
    pointing_versions = {
        'nircam1': 'dr0.5', 'nircam2': 'dr0.5', 'nircam3': 'dr0.5',
        'nircam4': 'dr0.6', 'nircam5': 'dr0.6', 'nircam6': 'dr0.5',
        'nircam7': 'dr0.6', 'nircam8': 'dr0.6', 'nircam9': 'dr0.6', 'nircam10': 'dr0.6'
    }
    
    for z_max in z_max_values:
        print(f"\n{'='*60}")
        print(f"PROCESSING WITH Z_MAX = {z_max}")
        print(f"{'='*60}")
        
        for pointing_num in range(1, 2):
            pointing = f"nircam{pointing_num}"
            dr_version = pointing_versions[pointing]
            
            print(f"\n{'='*50}")
            print(f"Processing {pointing} with {dr_version} (z_max={z_max})")
            print(f"{'='*50}")
            
            try:
                process_dropouts_balanced(
                    dropout_type=dropout_type, 
                    pointing=pointing, 
                    dr_version=dr_version, 
                    z_max=z_max
                )
            except Exception as e:
                print(f"Error processing {pointing} with z_max={z_max}: {e}")
                continue

if __name__ == "__main__":
    random.seed(42)
    process_all_pointings_balanced(z_max_values=[8])
    print("All balanced processing completed!")