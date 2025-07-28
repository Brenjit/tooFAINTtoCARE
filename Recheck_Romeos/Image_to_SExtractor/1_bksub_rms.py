import os
import numpy as np
from astropy.io import fits
from pathlib import Path

# Configuration
base_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data"
output_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/final_inputs"
required_filters = ['f606w', 'f814w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_pointing(pointing):
    pointing_dir = os.path.join(base_dir, pointing)
    if not os.path.exists(pointing_dir):
        print(f"⚠️ Pointing directory {pointing_dir} not found. Skipping.")
        return
    
    print(f"\n🔍 Processing pointing: {pointing}")
    
    # Create output subdirectory for this pointing
    pointing_output_dir = os.path.join(output_dir, pointing)
    os.makedirs(pointing_output_dir, exist_ok=True)
    
    # Process each required filter
    for filt in required_filters:
        print(f"\n  🎯 Looking for filter: {filt}")
        
        # Determine which file pattern to look for based on filter
        if filt in ['f606w', 'f814w']:
            # ACS/WFC3 filters
            file_pattern = f"egs_all_*_{filt}_030mas_v1.9_{pointing}_mef.fits"
        else:
            # JWST NIRCam filters
            file_pattern = f"hlsp_ceers_jwst_nircam_{pointing}_{filt}_dr0.5_i2d.fits"
        
        # Find matching files
        matching_files = list(Path(pointing_dir).glob(file_pattern))
        
        if not matching_files:
            print(f"    ❌ No file found for filter {filt} in {pointing}")
            continue
            
        input_file = str(matching_files[0])
        print(f"    ✅ Found file: {os.path.basename(input_file)}")
        
        try:
            with fits.open(input_file) as hdul:
                # For ACS/WFC3 files (multiple extensions)
                if 'mef.fits' in input_file:
                    # Get background-subtracted science image
                    sci_bksub_hdu = hdul['SCI_BKSUB']
                    
                    # Get RMS directly
                    rms_hdu = hdul['RMS']
                    
                    # Create new HDU for RMS
                    rms_data = rms_hdu.data
                    new_rms_hdu = fits.ImageHDU(data=rms_data, header=rms_hdu.header)
                    new_rms_hdu.header['BUNIT'] = 'RMS'
                
                # For JWST NIRCam files (single extension)
                elif 'i2d.fits' in input_file:
                    # Get science image (already background subtracted)
                    sci_bksub_hdu = hdul['SCI']
                    
                    # Get weight map and convert to RMS
                    wht_hdu = hdul['WHT']
                    wht_data = wht_hdu.data
                    
                    # Convert weight (1/variance) to RMS (sqrt(variance))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rms_data = np.sqrt(1.0 / wht_data)
                        rms_data[~np.isfinite(rms_data)] = 0.0  # replace inf/nan with 0
                    
                    # Create new HDU for RMS
                    new_rms_hdu = fits.ImageHDU(data=rms_data, header=wht_hdu.header)
                    new_rms_hdu.header['BUNIT'] = 'RMS'
                    new_rms_hdu.header.comments['BUNIT'] = 'Root Mean Square'
                
                input_filename = os.path.basename(input_file)
                root, ext = os.path.splitext(input_filename)
                # Save the background-subtracted science image
                output_sci_file = os.path.join(pointing_output_dir, f"{root}_SCI_BKSUB{ext}")
                fits.HDUList([fits.PrimaryHDU(), sci_bksub_hdu]).writeto(output_sci_file, overwrite=True)
                print(f"    💾 Saved SCI_BKSUB to {os.path.basename(output_sci_file)}")
                
                # Save the RMS image
                output_rms_file = os.path.join(pointing_output_dir, f"{root}_RMS{ext}")
                fits.HDUList([fits.PrimaryHDU(), new_rms_hdu]).writeto(output_rms_file, overwrite=True)
                print(f"    💾 Saved RMS to {os.path.basename(output_rms_file)}")
                    
        except Exception as e:
            print(f"    ❌ Error processing {input_file}: {str(e)}")

# Process all pointings from nircam1 to nircam6
for pointing_num in range(1, 4):
    pointing = f"nircam{pointing_num}"
    process_pointing(pointing)

print("\n✅ Processing complete!")