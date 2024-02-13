import csv
import os
import re
import sys
import logging

# Set the global logging level to INFO
logging.getLogger().setLevel(logging.INFO)
# Handler for terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Set the field limit to a larger value
csv.field_size_limit(min(2147483647, sys.maxsize))
# CSV extension
csv_extension = ".csv"
# Base Path
base_path = 'CSV/'
# CSV Path
csv_path = os.path.join(base_path,'STANDARD/')
# Path Cleaned files
cleaned_folder = os.path.join(csv_path, "CLEANED/")
# Path Mapped files
mapped_folder = os.path.join(csv_path, "MAPPED/")
# Create the 'mapped_folder' directory if it doesn't exist
os.makedirs(mapped_folder, exist_ok=True)

# Create a dictionary for substitutions
label_mapping = {
    re.compile(r'.*\b(bug|report)\b.*', flags=re.IGNORECASE): 'Bug',
    re.compile(r'.*\b(new feature|improvement|suggestion|feature request|enhancement)\b.*', flags=re.IGNORECASE): 'Enhancement',
    re.compile(r'.*\b(support request|question)\b.*', flags=re.IGNORECASE): 'Question'
}

# Get list of files in cleaned folder
input_files = [f for f in os.listdir(cleaned_folder) if f.endswith(csv_extension)]

# Iterate over each file in the folder
for filename_input in input_files:
    # Build input and output file paths
    input_csv_filename = os.path.join(cleaned_folder, filename_input)
    output_csv_filename = os.path.join(mapped_folder, filename_input)

    # Open the input CSV file
    with open(input_csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
        # Open the input CSV file
        csv_reader = csv.reader(csvfile)
        
        # Extract the first row (header)
        header = next(csv_reader)
        
        # Find the index of the 'label' column
        label_index = header.index('label')

        # Open the output CSV file
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csv_outputfile:
            # Write to the output CSV file
            csv_writer = csv.writer(csv_outputfile)
            
            # Write the heading
            csv_writer.writerow(header)

            # Iterate over the lines of the input CSV file
            for row in csv_reader:
                # Set a flag to indicate if the row has been mapped
                row_mapped = False

                # Perform the replacement only if the 'label' column is present
                if label_index < len(row):
                    # Perform "like" based substitution using regular expressions
                    current_label = row[label_index]
                    for pattern, replacement in label_mapping.items():
                        if re.match(pattern, current_label):
                            row[label_index] = replacement
                            row_mapped = True
                            break

                # Write the line to the output CSV file only if it has been mapped
                if row_mapped:
                    csv_writer.writerow(row)

    logging.info(f'The data in "{filename_input}" have been successfully exported to {output_csv_filename}')

logging.info("Processing of all files is complete.")
