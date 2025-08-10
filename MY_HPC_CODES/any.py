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
import matplotlib.pyplot as plt
import concurrent.futures
import psutil
import json
import time
from datetime import datetime
import traceback

# ======================== Configuration ========================
STATUS_FILE = "/home/ph25d002/ceers_data/processing_runs/latest_status.json"
os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)

# Filter-specific parameters
hst_photflam_lamda = {
    'F606W': {'photflam': 7.79611686490250e-20, 'lamda': 5920.405914484679},
    'F814W': {'photflam': 7.05123239329114e-20, 'lamda': 8046.166645443039}
}

aperture_corrections = {
    'F606W': 1.2676, 'F814W': 1.3017, 'F115W': 1.2250, 'F150W': 1.2168,
    'F200W': 1.2241, 'F277W': 1.4198, 'F356W': 1.5421, 'F410M': 1.6162, 
    'F444W': 1.6507
}

BASE_DIR = "/home/ph25d002/ceers_data"
CUTOUT_DIR = "/home/ph25d002/ceers_data/EAZY/eazy-photoz/cutout_nmad"

# ======================== Helper Functions ========================
def update_status(pointing, status, details=None):
    """Update status JSON file with processing information"""
    try:
        status_data = {
            "pointing": pointing,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        temp_file = f"{STATUS_FILE}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        os.replace(temp_file, STATUS_FILE)
    except Exception as e:
        print(f"Status update failed: {str(e)}")

def get_input_with_timeout(prompt, timeout, default):
    """Get user input with timeout (simplified version)"""
    print(f"{prompt} [Default: {default}]")
    return default

def get_optimal_process_count():
    """Calculate optimal number of processes based on system resources"""
    cpu_count = psutil.cpu_count(logical=False)
    mem = psutil.virtual_memory()
    return max(2, min(cpu_count, int(mem.available / (2 * 1024**3)))  # 2GB per process

def flux_conversion_scale(fits_file):
    """Get pixel area scaling from FITS header"""
    with fits.open(fits_file) as hdul:
        return hdul[0].header['PIXAR_SR']

def nmad(data):
    """Normalized Median Absolute Deviation"""
    return 1.48 * np.median(np.abs(data - np.median(data)))

def hst_flux_to_ujy(flux, photflam, lamda):
    """Convert HST flux to microjanskys"""
    return 3.34e4 * (lamda**2) * (flux * photflam) * 1e6

# ======================== Core Processing Functions ========================
def find_closest_empty_apertures(x, y, kdtree, num_apertures=200, 
                               search_radius=8000, min_dist=7, max_attempts=20):
    """Find background apertures avoiding sources"""
    apertures = []
    attempt = 0
    
    while len(apertures) < num_apertures and attempt < max_attempts:
        dists, indices = kdtree.query([x, y], k=search_radius)
        valid_indices = indices[dists > 10]  # Exclude very close apertures
        
        for idx in valid_indices:
            candidate = kdtree.data[idx]
            if all(np.linalg.norm(candidate - a) >= min_dist for a in apertures):
                apertures.append(candidate)
                if len(apertures) >= num_apertures:
                    break
        
        if len(apertures) < num_apertures:
            search_radius += 200
            attempt += 1
    
    return np.array(apertures)

def process_single_filter(filter_name, paths, df, config):
    """Process data for a single filter"""
    pointing = config['pointing']
    update_status(pointing, "filter_start", {"filter": filter_name})
    
    # Initialize results storage
    results = {}
    correction = aperture_corrections[filter_name]
    
    # Flux conversion
    if filter_name in hst_photflam_lamda:
        photflam = hst_photflam_lamda[filter_name]['photflam']
        lamda = hst_photflam_lamda[filter_name]['lamda']
        flux_conv = lambda f: hst_flux_to_ujy(f, photflam, lamda) * correction
    else:
        pixar_sr = flux_conversion_scale(paths['fits'])
        flux_conv = lambda f: f * pixar_sr * 1e12 * correction
    
    df['FLUX_APER'] = df['FLUX_APER'].apply(flux_conv)
    df['FLUXERR_APER'] = df['FLUXERR_APER'].apply(flux_conv)
    df['SNR'] = df['FLUX_APER'] / df['FLUXERR_APER']
    
    # Empty aperture processing if requested
    if config['empty_aperture']:
        with fits.open(paths['segmentation']) as hdul:
            seg = hdul[0].data
        with fits.open(paths['fits']) as hdul:
            img = hdul[0].data
        with fits.open(paths['rms']) as hdul:
            rms = hdul[0].data
        
        # Create mask of empty regions
        empty_mask = binary_erosion((seg == 0) & (rms != 0), structure=disk(5))
        kdtree = cKDTree(np.argwhere(empty_mask)[:, [1, 0]])
        
        for _, row in df.iterrows():
            src_id = int(row['NUMBER'])
            x, y = row['X_IMAGE'], row['Y_IMAGE']
            
            # Find and measure background apertures
            bg_apertures = find_closest_empty_apertures(x, y, kdtree)
            phot_table = aperture_photometry(img, CircularAperture(bg_apertures, r=5))
            nmad_val = nmad(phot_table['aperture_sum'])
            
            # Convert to µJy
            bg_error = (hst_flux_to_ujy(nmad_val, photflam, lamda) if filter_name in hst_photflam_lamda 
                       else nmad_val * pixar_sr * 1e12)
            
            # Generate cutouts if requested
            if config['generate_cutouts']:
                os.makedirs(CUTOUT_DIR, exist_ok=True)
                size = 150
                x1, x2 = int(x-size/2), int(x+size/2)
                y1, y2 = int(y-size/2), int(y+size/2)
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for ax, data, title in zip(axes, 
                                          [img[y1:y2, x1:x2], seg[y1:y2, x1:x2]], 
                                          ["Image", "Segmentation"]):
                    ax.imshow(data, cmap='gray', origin='lower')
                    ax.set_title(title)
                    ax.axis('off')
                
                plt.suptitle(f"Source {src_id} - {filter_name}")
                plt.savefig(f"{CUTOUT_DIR}/cutout_{filter_name}_src_{src_id}.png", 
                           bbox_inches='tight', dpi=150)
                plt.close()
            
            # Store results
            results[src_id] = {
                'x': x, 'y': y,
                'class_star': row['CLASS_STAR'],
                f'mag_{filter_name}': row['MAG_APER'],
                f'magerr_{filter_name}': row['MAGERR_APER'],
                f'flux_{filter_name}': row['FLUX_APER'],
                f'fluxerr_{filter_name}': row['FLUXERR_APER'],
                f'snr_{filter_name}': row['SNR'],
                f'nmad_{filter_name}': bg_error
            }
    else:
        # Simple processing without empty apertures
        for _, row in df.iterrows():
            results[int(row['NUMBER'])] = {
                'x': row['X_IMAGE'], 'y': row['Y_IMAGE'],
                'class_star': row['CLASS_STAR'],
                f'mag_{filter_name}': row['MAG_APER'],
                f'magerr_{filter_name}': row['MAGERR_APER'],
                f'flux_{filter_name}': row['FLUX_APER'],
                f'fluxerr_{filter_name}': row['FLUXERR_APER'],
                f'snr_{filter_name}': row['SNR']
            }
    
    update_status(pointing, "filter_complete", {"filter": filter_name})
    return filter_name, results

def process_pointing(pointing, dr_version):
    """Process all filters for a single pointing"""
    start_time = time.time()
    update_status(pointing, "started", {
        "dr_version": dr_version,
        "start_time": datetime.now().isoformat()
    })
    
    try:
        # Get processing configuration
        num_proc = int(get_input_with_timeout(
            f"Processes for {pointing}? (Optimal: {get_optimal_process_count()}): ",
            5, str(get_optimal_process_count()))
        )
        do_empty = get_input_with_timeout(
            "Perform empty aperture method? (yes/no): ", 5, "yes").lower() == 'yes'
        do_cutouts = get_input_with_timeout(
            "Generate cutouts? (yes/no): ", 5, "no").lower() == 'yes'
        
        config = {
            'pointing': pointing,
            'empty_aperture': do_empty,
            'generate_cutouts': do_cutouts
        }
        
        # Define all data paths
        img_dir = f"{BASE_DIR}/Romeo_s_data/{pointing}"
        cat_dir = f"{BASE_DIR}/SEP_JWST/Results/{pointing}/catalogue_z7"
        seg_dir = f"{BASE_DIR}/SEP_JWST/Results/{pointing}/segmentations_z7"
        out_dir = f"{BASE_DIR}/EAZY/eazy-photoz/inputs/Eazy_catalogue"
        os.makedirs(out_dir, exist_ok=True)
        
        # All filters to process
        filters = {
            'F606W': {
                'catalog': f"{cat_dir}/f150dropout_f606w_catalog.cat",
                'fits': f"{img_dir}/egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits",
                'seg': f"{seg_dir}/f150dropout_f606w_segmentation.fits",
                'rms': f"{img_dir}/egs_all_acs_wfc_f606w_030mas_v1.9_{pointing}_mef_RMS.fits"
            },
            'F814W': {
                'catalog': f"{cat_dir}/f150dropout_f814w_catalog.cat",
                'fits': f"{img_dir}/egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_SCI_BKSUB.fits",
                'seg': f"{seg_dir}/f150dropout_f814w_segmentation.fits",
                'rms': f"{img_dir}/egs_all_acs_wfc_f814w_030mas_v1.9_{pointing}_mef_RMS.fits"
            },
            'F115W': {
                'catalog': f"{cat_dir}/f150dropout_f115w_catalog.cat",
                'fits': f"{img_dir}/hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_SCI_BKSUB_c.fits",
                'seg': f"{seg_dir}/f150dropout_f115w_segmentation.fits",
                'rms': f"{img_dir}/hlsp_ceers_jwst_nircam_{pointing}_f115w_{dr_version}_i2d_RMS.fits"
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
        
        
        # Process in parallel
        pool = multiprocessing.Pool(processes=num_proc)
        tasks = []
        
        for fname, paths in filters.items():
            try:
                df = pd.read_csv(paths['catalog'], sep=r'\s+', comment='#', header=None)
                df.columns = [
                    'NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO',
                    'MAG_APER', 'MAGERR_APER', 'CLASS_STAR', 'FLUX_AUTO',
                    'FLUXERR_AUTO', 'FLUX_APER', 'FLUXERR_APER',
                    'ALPHA_J2000', 'DELTA_J2000'
                ]
                tasks.append((fname, paths, df, config))
            except Exception as e:
                update_status(pointing, "error", {
                    "filter": fname, "error": str(e)
                })
                continue
        
        # Process all filters
        results = pool.starmap(process_single_filter, tasks)
        pool.close()
        pool.join()
        
        # Combine results
        combined = {}
        for fname, fdata in results:
            for src_id, sdata in fdata.items():
                if src_id not in combined:
                    combined[src_id] = {'id': src_id}
                combined[src_id].update(sdata)
        
        # Save output files
        pd.DataFrame.from_dict(combined, orient='index').to_csv(
            f"{out_dir}/{pointing}_results.txt", sep='\t', index=False)
        
        # Create EAZY catalog
        catalog_type = 'eazy' if do_empty else 'fluxerr'
        with open(f"{out_dir}/{pointing}_{catalog_type}_catalog.cat", 'w') as f:
            f.write('# id ' + ' '.join(f'f_{f} e_{f}' for f in filters.keys()) + '\n')
            for src in combined.values():
                f.write(f"{src['id']} " + ' '.join(
                    f"{src.get(f'flux_{f}', 0)} {src.get(f'nmad_{f}' if do_empty else f'fluxerr_{f}', 0)}"
                    for f in filters.keys()
                ) + '\n')
        
        update_status(pointing, "completed", {
            "runtime": time.time() - start_time,
            "sources": len(combined)
        })
        
    except Exception as e:
        update_status(pointing, "failed", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise

# ======================== Main Execution ========================
if __name__ == "__main__":
    plt.switch_backend('Agg')  # Non-interactive backend
    
    # Pointing versions mapping
    pointing_versions = {
        f'nircam{i}': 'dr0.5' if i in [1,2,3,6] else 'dr0.6'
        for i in range(1, 11)
    }
    
    # Process all pointings
    update_status("all", "started", {
        "total_pointings": 10,
        "start_time": datetime.now().isoformat()
    })
    
    for pnum in range(1, 11):
        pointing = f"nircam{pnum}"
        update_status("all", "pointing_start", {
            "number": pnum, "pointing": pointing
        })
        
        try:
            process_pointing(pointing, pointing_versions[pointing])
            update_status("all", "pointing_complete", {
                "pointing": pointing
            })
        except Exception as e:
            update_status("all", "pointing_failed", {
                "pointing": pointing,
                "error": str(e)
            })
    
    update_status("all", "completed", {
        "end_time": datetime.now().isoformat()
    })
    print("All processing completed successfully!")