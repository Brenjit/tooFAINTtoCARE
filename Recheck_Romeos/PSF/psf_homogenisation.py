import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
import os
import gc  # for garbage collection to free memory

# Define output directory to save the convolved images
output_dir = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1"
os.makedirs(output_dir, exist_ok=True)

# Function to load FITS data from a specific extension
def load_fits_data(filepath, ext):
    print(f"Loading FITS data from {filepath}, extension {ext}...")
    try:
        with fits.open(filepath) as hdul:
            data = hdul[ext].data
        print(f"Successfully loaded data from {filepath}.")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise

# Function to save FITS data
def save_fits_data(filepath, data, header=None):
    print(f"Saving FITS data to {filepath}...")
    try:
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(filepath, overwrite=True)
        print(f"Successfully saved data to {filepath}.")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        raise

# Function to process one image and kernel pair
def process_image(img_path, kernel_path):
    try:
        # Load the image data from extension 0 and the kernel data from extension 0
        print(f"\nProcessing image: {img_path} with kernel: {kernel_path}...")
        image_data = load_fits_data(img_path, ext=0)
        kernel_data = load_fits_data(kernel_path, ext=0)

        # Perform the convolution
        print(f"Performing convolution between the image and kernel...")
        convolved_data = convolve_fft(
            image_data,
            kernel_data,
            boundary='wrap',
            normalize_kernel=True,
            allow_huge=True
        )
        print("Convolution completed successfully.")

        # Save the convolved image
        img_name = os.path.basename(img_path).replace(".fits", "_c.fits")
        output_path = os.path.join(output_dir, img_name)

        print(f"Loading header from {img_path}...")
        with fits.open(img_path) as hdul:
            header = hdul[0].header  # Taking the header from the image extension

        print(f"Saving the convolved image to {output_path}...")
        save_fits_data(output_path, convolved_data, header=header)

        # Force garbage collection to free memory
        del image_data, kernel_data, convolved_data
        gc.collect()
        print(f"Successfully processed {img_path} and saved to {output_path}")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")



# List of image files and corresponding kernel files
image_kernel_pairs = [
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f115w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F115W_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f150w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F150W_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f200w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F200W_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f277w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F277W_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f356w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F356W_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f410m_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F410M_to_F444W_kernel.fits"),
    ("/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/nircam1/hlsp_ceers_jwst_nircam_nircam10_f444w_dr0.6_i2d_SCI_BKSUB.fits", "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/highzgalaxies-main/psf/pypher/F444W_to_F444W_kernel.fits"),
]
# Loop through each image and its corresponding kernel
for img_path, kernel_path in image_kernel_pairs:
    process_image(img_path, kernel_path)