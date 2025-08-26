import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("matching_log.txt"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Paths
base_path = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/Drive_cat"
z9_file = os.path.join(base_path, "z9_catalogue_2.txt")
romeo_cat_dir = os.path.join(base_path, "Romeos_eazy_cat")
brenjit_cat_dir = os.path.join(base_path, "Brenjit_eazy_cat")

# Load final redshift catalogue
colnames = ["POINTING", "Romeo_ID", "REDSHIFT", "RA", "DEC", "M_UV", "m_UV"]
z9 = pd.read_csv(z9_file, delim_whitespace=True, comment="#", names=colnames)

# Function to load a sextractor catalogue with logging
def load_sextractor_cat(file):
    cols = ["NUMBER", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "MAGERR_AUTO",
            "MAG_APER", "MAGERR_APER", "CLASS_STAR", "FLUX_AUTO", "FLUXERR_AUTO",
            "FLUX_APER", "FLUXERR_APER", "RA", "DEC"]
    try:
        cat = pd.read_csv(file, delim_whitespace=True, comment="#", names=cols)
        logger.info(f"Successfully loaded catalog: {os.path.basename(file)} with {len(cat)} sources")
        return cat
    except Exception as e:
        logger.error(f"Failed to load catalog {file}: {str(e)}")
        return None

# Matching parameters
position_match_radius = 0.5 * u.arcsec  # for position verification
flux_tolerance = 0.1  # 10% flux difference allowed

final_rows = []
match_stats = {
    'total': 0,
    'romeo_not_found': 0,
    'brenjit_flux_match': 0,
    'brenjit_position_match': 0,
    'no_match': 0
}

for idx, row in z9.iterrows():
    pointing = row["POINTING"].lower()
    romeo_id = int(row["Romeo_ID"])
    redshift = row["REDSHIFT"]
    match_stats['total'] += 1

    logger.info(f"\nProcessing source #{idx+1}:")
    logger.info(f"  From z9 catalog - POINTING: {pointing}, Romeo_ID: {romeo_id}, z: {redshift}")

    # Initialize match metrics
    flux_diff = np.nan
    pos_diff_arcsec = np.nan
    flux_match_flag = 0
    pos_match_flag = 0
    
    # File names
    romeo_file = os.path.join(romeo_cat_dir, f"{pointing}_f410m_catalog.cat")
    brenjit_file = os.path.join(brenjit_cat_dir, f"{pointing}_f410m_catalog.cat")

    # Load catalogs
    romeo_cat = load_sextractor_cat(romeo_file)
    if romeo_cat is None:
        match_stats['romeo_not_found'] += 1
        final_rows.append([pointing, romeo_id, -1, redshift, flux_diff, pos_diff_arcsec, 
                         flux_match_flag, pos_match_flag])
        continue

    brenjit_cat = load_sextractor_cat(brenjit_file)
    if brenjit_cat is None:
        match_stats['romeo_not_found'] += 1
        final_rows.append([pointing, romeo_id, -1, redshift, flux_diff, pos_diff_arcsec,
                         flux_match_flag, pos_match_flag])
        continue

    # Find source in Romeo's catalog
    r_source = romeo_cat.loc[romeo_cat["NUMBER"] == romeo_id]
    if r_source.empty:
        logger.warning(f"  Romeo ID {romeo_id} not found in {os.path.basename(romeo_file)}")
        match_stats['romeo_not_found'] += 1
        final_rows.append([pointing, romeo_id, -1, redshift, flux_diff, pos_diff_arcsec,
                         flux_match_flag, pos_match_flag])
        continue

    romeo_flux = float(r_source["FLUX_AUTO"])
    romeo_ra = float(r_source["RA"])
    romeo_dec = float(r_source["DEC"])
    
    logger.info(f"  Found in Romeo's catalog:")
    logger.info(f"    RA: {romeo_ra:.6f}, DEC: {romeo_dec:.6f}")
    logger.info(f"    FLUX_AUTO: {romeo_flux:.2f}")

    # Find matching flux in Brenjit's catalog
    flux_diffs = np.abs((brenjit_cat["FLUX_AUTO"] - romeo_flux) / romeo_flux)
    flux_matches = flux_diffs < flux_tolerance
    n_flux_matches = flux_matches.sum()

    if n_flux_matches > 0:
        best_match_idx = flux_diffs.idxmin()
        brenjit_id = int(brenjit_cat.iloc[best_match_idx]["NUMBER"])
        brenjit_flux = float(brenjit_cat.iloc[best_match_idx]["FLUX_AUTO"])
        brenjit_ra = float(brenjit_cat.iloc[best_match_idx]["RA"])
        brenjit_dec = float(brenjit_cat.iloc[best_match_idx]["DEC"])
        
        # Calculate differences
        flux_diff = (brenjit_flux - romeo_flux) / romeo_flux  # fractional difference
        flux_match_flag = 1
        
        # Verify position match
        romeo_coord = SkyCoord(romeo_ra*u.deg, romeo_dec*u.deg)
        brenjit_coord = SkyCoord(brenjit_ra*u.deg, brenjit_dec*u.deg)
        sep = romeo_coord.separation(brenjit_coord)
        pos_diff_arcsec = sep.to('arcsec').value
        
        logger.info(f"  Found {n_flux_matches} flux matches in Brenjit's catalog")
        logger.info(f"  Best flux match: Brenjit ID {brenjit_id}")
        logger.info(f"    Brenjit FLUX_AUTO: {brenjit_flux:.2f} (diff: {100*flux_diff:.1f}%)")
        logger.info(f"    Brenjit position: RA {brenjit_ra:.6f}, DEC {brenjit_dec:.6f}")
        logger.info(f"    Position separation: {pos_diff_arcsec:.2f} arcsec")

        if sep < position_match_radius:
            logger.info("  GOOD POSITION MATCH")
            pos_match_flag = 1
            match_stats['brenjit_position_match'] += 1
        else:
            logger.warning("  WARNING: Flux match but large position difference")
            match_stats['brenjit_flux_match'] += 1
    else:
        brenjit_id = -1
        logger.warning("  No flux match found in Brenjit's catalog")
        match_stats['no_match'] += 1

    final_rows.append([pointing, romeo_id, brenjit_id, redshift, 
                      flux_diff, pos_diff_arcsec, 
                      flux_match_flag, pos_match_flag])

# Save final catalogue
final_cols = ["POINTING", "Romeo_ID", "Brenjit_ID", "REDSHIFT",
              "FLUX_DIFF", "POS_DIFF_ARCSEC",
              "FLUX_MATCH_FLAG", "POS_MATCH_FLAG"]
final_df = pd.DataFrame(final_rows, columns=final_cols)
final_outfile = os.path.join(base_path, "final_matched_catalogue.txt")
final_df.to_csv(final_outfile, sep=" ", index=False, float_format="%.6f")

# Print summary statistics
logger.info("\nMatching Statistics:")
logger.info(f"Total sources processed: {match_stats['total']}")
logger.info(f"Romeo sources not found: {match_stats['romeo_not_found']}")
logger.info(f"Good flux+position matches: {match_stats['brenjit_position_match']}")
logger.info(f"Flux matches but position differs: {match_stats['brenjit_flux_match']}")
logger.info(f"No matches found: {match_stats['no_match']}")
logger.info(f"\nFinal catalogue saved to {final_outfile}")