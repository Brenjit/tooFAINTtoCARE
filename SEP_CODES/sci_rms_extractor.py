import os
from astropy.io import fits
import numpy as np

# List of filenames
file_list = [
    "egs_all_acs_wfc_f606w_030mas_v1.9_nircam3_mef.fits",
    "egs_all_acs_wfc_f814w_030mas_v1.9_nircam3_mef.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f115w_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f150w_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f200w_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f277w_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f356w_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f410m_dr0.5_i2d.fits",
    "hlsp_ceers_jwst_nircam_nircam3_f444w_dr0.5_i2d.fits"
]

# Input directory
input_dir = "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2/nircam3"

# Output directories
output_dir_bksub = os.path.join(input_dir, "BKSUB")
output_dir_rms = os.path.join(input_dir, "RMS")

os.makedirs(output_dir_bksub, exist_ok=True)
os.makedirs(output_dir_rms, exist_ok=True)

for fname in file_list:
    fpath = os.path.join(input_dir, fname)
    with fits.open(fpath) as hdul:
        # Decide file type
        if "hlsp_ceers" in fname:
            # CEERS files
            bksub_data = hdul["SCI_BKSUB"].data
            bksub_header = hdul["SCI_BKSUB"].header

            wht_data = hdul["WHT"].data
            # Avoid zero or negative weights
            wht_data = np.where(wht_data > 0, wht_data, 0)
            rms_data = np.zeros_like(wht_data)
            mask = wht_data > 0
            rms_data[mask] = 1.0 / np.sqrt(wht_data[mask])

        elif "egs_all" in fname:
            # egs files
            bksub_data = hdul["SCI_BKSUB"].data
            bksub_header = hdul["SCI_BKSUB"].header

            rms_data = hdul["RMS"].data

        else:
            print(f"Skipping unknown file type: {fname}")
            continue

        # Save BKSUB
        bksub_hdu = fits.PrimaryHDU(data=bksub_data, header=bksub_header)
        bksub_out_path = os.path.join(output_dir_bksub, fname.replace(".fits", "_bksub.fits"))
        bksub_hdu.writeto(bksub_out_path, overwrite=True)

        # Save RMS
        rms_hdu = fits.PrimaryHDU(data=rms_data, header=bksub_header)
        rms_out_path = os.path.join(output_dir_rms, fname.replace(".fits", "_rms.fits"))
        rms_hdu.writeto(rms_out_path, overwrite=True)

        print(f"✅ Processed: {fname}")

print("🎉 Done! All BKSUB and RMS files created.")

