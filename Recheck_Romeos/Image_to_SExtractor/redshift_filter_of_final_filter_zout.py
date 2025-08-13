import os

# Path to your zout file
zout_path = "/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/EAZY/eazy-photoz/inputs/OUTPUT_Z20/nircam1/nircam1_output.zout"
output_path = "nircam1_zgt8p5.log"

# Open the zout file
with open(zout_path, "r") as f:
    lines = f.readlines()

header_lines = []
data_lines = []

# Separate header and data
for line in lines:
    if line.startswith("#") or line.strip() == "":
        header_lines.append(line.rstrip("\n"))
    else:
        data_lines.append(line.strip())

filtered_lines = []

for line in data_lines:
    parts = line.split()
    try:
        z_a = float(parts[2])  # z_a is column 3 (index 2)
        if z_a > 8.5:
            filtered_lines.append(line)
    except (ValueError, IndexError):
        continue  # Skip malformed lines

# Write results with proper numbering/log
with open(output_path, "w") as f:
    for h in header_lines:
        f.write(h + "\n")
    f.write(f"# Filter: z_a > 8.5 | Total sources: {len(filtered_lines)}\n\n")

    for idx, line in enumerate(filtered_lines, start=1):
        f.write(f"{idx:03d}  {line}\n")

print(f"Filtered {len(filtered_lines)} sources with z_a > 8.5.")
print(f"Saved to {output_path}")
