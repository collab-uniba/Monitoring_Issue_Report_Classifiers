import os
import csv
import sys
import re

# Base Path CSV
base_path = "CSV/"
# Path Cleaned files
cleaned_folder = os.path.join(base_path, "CLEANED/")
# Path Bugs files
bugs_folder = os.path.join(base_path, "BUGS/")
# Create the 'BUGS/' directory if it doesn't exist
os.makedirs(bugs_folder, exist_ok=True)
# Output file path
output_file_path = os.path.join(bugs_folder, "matching_labels.csv")
# CSV extension
csv_extension = ".csv"

# Set the field limit to a larger value
csv.field_size_limit(min(2147483647, sys.maxsize))

# Create a list to store the labels found
matching_labels = []

# Get the list of files in the CLEANED folder
input_files = [f for f in os.listdir(cleaned_folder) if f.endswith(csv_extension)]

# Iterate over each file in the CLEANED folder
for filename_input in input_files:
    input_csv_filename = os.path.join(cleaned_folder, filename_input)

    with open(input_csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        # Extract the header
        header = next(csv_reader)
        # Find the index of the 'label' column
        label_index = header.index('label')

        # Iterate over the lines of the CSV file
        for row in csv_reader:
            # Check if the 'label' column contains 'bug' but not 'report' and is not exactly "Bug"
            if label_index < len(row) and re.search(r'\bbug\b', row[label_index], flags=re.IGNORECASE) and 'report' not in row[label_index].lower() and row[label_index].lower() != 'bug':
                matching_labels.append({
                    'File': filename_input,
                    'Issue ID': row[0],  # TODO - replace with actual id column
                    'Label': row[label_index]
                })

# Write the results to the output file
with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)

    # Write the heading
    csv_writer.writerow(['File', 'Issue ID', 'Label'])

    # Write the data
    for label_info in matching_labels:
        csv_writer.writerow([label_info['File'], label_info['Issue ID'], label_info['Label']])

print(f"Matching labels saved to: {output_file_path}")