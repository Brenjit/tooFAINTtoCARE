#!/bin/bash

# ✅ Change to correct working directory
cd /Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2/SEP_JWST || { echo "Failed to change directory"; exit 1; }

nircam_version='nircam3'
dr_version='dr0.5'

LOG_FILE="./${nircam_version}_run_log.txt"
echo "Log file created at: ${LOG_FILE}" > "${LOG_FILE}"

DETECTION_IMAGES=(
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/${nircam_version}_coadd.fits"
)

FILTER_IMAGES=(
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f115w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f150w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f200w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f277w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f356w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f410m_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/hlsp_ceers_jwst_nircam_${nircam_version}_f444w_${dr_version}_i2d_SCI_BKSUB_c.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/egs_all_acs_wfc_f606w_030mas_v1.9_nircam6_mef_SCI_BKSUB.fits"
    "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images_2
/${nircam_version}/egs_all_acs_wfc_f814w_030mas_v1.9_nircam6_mef_SCI_BKSUB.fits"
)

OUTPUT_DIR="./Results/${nircam_version}/catalogue"
CHECKIMAGE_DIR="./Results/${nircam_version}/segmentations"

mkdir -p "${OUTPUT_DIR}" || { echo "Failed to create output directory"; exit 1; }
mkdir -p "${CHECKIMAGE_DIR}" || { echo "Failed to create check image directory"; exit 1; }

extract_filter_name() {
    local filename="$1"
    if [[ "$filename" == *"hlsp_ceers_jwst_nircam"* ]]; then
        echo "$(basename "${filename}" | awk -F'_' '{print $6}')"
    elif [[ "$filename" == *"egs_all_acs"* ]]; then
        echo "$(basename "${filename}" | grep -o 'f606w\|f814w')"
    else
        echo "unknown"
    fi
}

run_sextractor() {
    local detection_image="$1"
    local filter_image="$2"
    local filter_name
    filter_name=$(extract_filter_name "${filter_image}")
    local config_file="./${filter_name}_default.sex"
    local param_file="./default.param"

    echo "Processing detection image: ${detection_image} with filter image: ${filter_image}" | tee -a "${LOG_FILE}"
    echo "Using .sex file: ${config_file}" | tee -a "${LOG_FILE}"
    echo "Using .param file: ${param_file}" | tee -a "${LOG_FILE}"
    echo "Output catalog: ${OUTPUT_DIR}/${filter_name}_catalog.cat" | tee -a "${LOG_FILE}"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Configuration file not found: $config_file" | tee -a "${LOG_FILE}"
        return
    fi

    if [[ ! -f "$param_file" ]]; then
        echo "ERROR: Parameter file not found: $param_file" | tee -a "${LOG_FILE}"
        return
    fi

    {
        echo "----- SExtractor started on $(date)"
        sex "${detection_image},${filter_image}" \
            -c "${config_file}" \
            -PARAMETERS_NAME "${param_file}" \
            -CATALOG_NAME "${OUTPUT_DIR}/${filter_name}_catalog.cat" \
            -CHECKIMAGE_TYPE SEGMENTATION \
            -CHECKIMAGE_NAME segmentation.fits
    } 2>&1 | tee -a "${LOG_FILE}"

    if ! mv segmentation.fits "${CHECKIMAGE_DIR}/${filter_name}_segmentation.fits"; then
        echo "Failed to move check image for ${filter_name}." | tee -a "${LOG_FILE}"
    fi
}

for detection_image in "${DETECTION_IMAGES[@]}"; do
    for filter_image in "${FILTER_IMAGES[@]}"; do
        run_sextractor "${detection_image}" "${filter_image}"
    done
done

echo "✅ SExtractor runs completed." | tee -a "${LOG_FILE}"
