import os
from astropy.table import Table

# === CONFIG ===
base_dir = "/Volumes/MY_SSD_1TB/My_work_june_24/CEERS_NIRCam_Images/EAZY/eazy-photoz/inputs/OUTPUT"
input_file = os.path.join(base_dir, "nircam6_output.zout")

# Create 'filtered' folder if it doesn't exist
filtered_dir = os.path.join(base_dir, "filtered")
os.makedirs(filtered_dir, exist_ok=True)

output_file = os.path.join(filtered_dir, "filtered_galaxies_z8p5_9p5.cat")

# === Read zout file ===
zout = Table.read(input_file, format="ascii")

# === Filter criteria ===
filtered = zout[
    (zout["z_peak"] >= 8.5) &
    (zout["z_peak"] <= 9.5) &
    (zout["chi_a"] < 30)
]

print(f"✅ Selected {len(filtered)} galaxies with 8.5 ≤ z_peak ≤ 9.5 and chi² < 10.")

# === Write filtered table ===
filtered.write(output_file, format="ascii.commented_header", overwrite=True)
print(f"💾 Saved to: {output_file}")

