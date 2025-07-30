import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------
# Function to read .cat files
# --------------------------
def read_cat(file):
    """
    Reads a SExtractor .cat file, skipping comment lines.
    Returns DataFrame with appropriate columns.
    """
    data = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            data.append([float(x) for x in line.strip().split()])
    data = np.array(data)
    return pd.DataFrame(data, columns=[
        "ID", "X_IMAGE", "Y_IMAGE", "MAG_AUTO", "MAGERR_AUTO",
        "MAG_APER", "MAGERR_APER", "CLASS_STAR", "FLUX_AUTO", "FLUXERR_AUTO",
        "FLUX_APER", "FLUXERR_APER", "RA", "DEC"
    ])

# --------------------------
# Load your catalogs
# --------------------------
cat_mine = read_cat("/Users/brenjithazarika/Downloads/9th_juoy_coma/28_0/Final_f200w_catalog_mine.cat")
cat_romeo = read_cat("/Users/brenjithazarika/Downloads/9th_juoy_coma/28_0/Final_f200w_catalog_romeo.cat")

# --------------------------
# Matching tolerance (in pixels)
# --------------------------
tolerance = 1e-4

# Lists to store matched and unmatched sources
matched = []
unmatched_mine = []
unmatched_romeo_ids = set(cat_romeo["ID"].astype(int).tolist())

print("🔎 Matching sources by X_IMAGE and Y_IMAGE...")
for i, row in tqdm(cat_mine.iterrows(), total=len(cat_mine)):
    # Find matching source in Romeo catalog
    match = cat_romeo[
        (np.isclose(cat_romeo["X_IMAGE"], row["X_IMAGE"], atol=tolerance)) &
        (np.isclose(cat_romeo["Y_IMAGE"], row["Y_IMAGE"], atol=tolerance))
    ]
    if not match.empty:
        romeo_row = match.iloc[0]

        # Combine all columns from mine and romeo, with suffixes
        combined = {}
        for col in cat_mine.columns:
            combined[f"{col}_mine"] = row[col]
        for col in cat_romeo.columns:
            combined[f"{col}_romeo"] = romeo_row[col]

        matched.append(combined)
        unmatched_romeo_ids.discard(int(romeo_row["ID"]))
    else:
        unmatched_mine.append({
            "X_IMAGE": row["X_IMAGE"],
            "Y_IMAGE": row["Y_IMAGE"],
            "My_ID": int(row["ID"])
        })

# --------------------------
# Create DataFrame for matched sources
# --------------------------
matched_df = pd.DataFrame(matched)

# --------------------------
# Check flux differences
# --------------------------
flux_diff_mask = np.abs(matched_df["FLUX_AUTO_mine"] - matched_df["FLUX_AUTO_romeo"]) > 1e-3
matched_diff_flux = matched_df[flux_diff_mask]

# --------------------------
# Save catalogs
# --------------------------
matched_df.to_csv("2_catalog_same_positions.txt", sep='\t', index=False)
matched_diff_flux.to_csv("2_catalog_samepos_diff_flux.txt", sep='\t', index=False)
pd.DataFrame(unmatched_mine).to_csv("2_catalog_unmatched_2.28.2.txt", sep='\t', index=False)
pd.DataFrame(list(unmatched_romeo_ids), columns=["Romeo_ID"]).to_csv("2_catalog_unmatched_2.28.0.txt", sep='\t', index=False)

# --------------------------
# Print summary
# --------------------------
print("\n📊 Matching summary:")
print(f"Total sources in 2.28.2 catalog           : {len(cat_mine)}")
print(f"Total sources in 2.28.0 catalog          : {len(cat_romeo)}")
print(f"Number of matched sources               : {len(matched_df)}")
print(f"Number of matched with flux difference  : {len(matched_diff_flux)}")
print(f"Number of 2.28.2 sources unmatched        : {len(unmatched_mine)}")
print(f"Number of 2.28.0 sources unmatched       : {len(unmatched_romeo_ids)}")
print("\n✅ Output files saved:")
print(" - 2_catalog_same_positions.txt")
print(" - 2_catalog_samepos_diff_flux.txt")
print(" - 2_catalog_unmatched_2.28.2.txt")
print(" - 2_catalog_unmatched_2.28.0.txt")


#done