# Define file paths
input_file = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/Drive_cat/final_matched_catalogue.txt'
output_file = '/Volumes/MY_SSD_1TB/Work_PhD/July-August/CEERS_data/Romeo_s_data/Drive_cat/extracted_columns.txt'

# Open input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Process each line
    for line in infile:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Split the line into columns
        columns = line.split()
        
        # Make sure we have enough columns
        if len(columns) >= 3:
            # Write POINTING and Brenjit_ID to output file
            outfile.write(f"{columns[0]} {columns[2]}\n")

print(f"Extracted columns saved to: {output_file}")