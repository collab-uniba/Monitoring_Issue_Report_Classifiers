import csv
import os
import sys
import logging

# Set the global logging level to INFO
logging.getLogger().setLevel(logging.INFO)
# Handler for terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
# Create the 'logs/' directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
# Add a handler to write WARNING level messages to a file
file_handler = logging.FileHandler('logs/warning_clean.log')
file_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(file_handler)

# Set the field limit to a larger value
csv.field_size_limit(min(2147483647, sys.maxsize))
# Base Path CSV
base_path = "CSV/"
# Output folder for clean files
cleaned_folder = "CSV/CLEANED/"
# Extension CSV
csv_extension = ".csv"
# Create the 'cleaned_folder' directory if it doesn't exist
os.makedirs(cleaned_folder, exist_ok=True)

# Get list of files in base folder
input_files = [f for f in os.listdir(base_path) if f.endswith(csv_extension)]

# Iterate over each file in the folder
for filename_input in input_files:
    # Build input and output file paths
    input_csv_filename = os.path.join(base_path, filename_input)
    output_csv_filename = os.path.join(cleaned_folder, filename_input)

    # Open the input CSV file
    with open(input_csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
        # Read the CSV file
        csv_reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=',')
        # Extract the first row (header)
        header = next(csv_reader)

        # Find indexes of desired columns
        summary = header.index('fields.summary')
        description = header.index('fields.description')
        name = header.index('fields.issuetype.name')
        created = header.index('fields.created')

        # Open the output CSV file
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csv_outputfile:
            # Write to the output CSV file
            csv_writer = csv.writer(csv_outputfile)
            # Write the heading
            csv_writer.writerow(['title', 'body', 'label', 'date'])

            # Iterate over the lines of the input CSV file
            for row in csv_reader:
                # Extract the desired data
                title = row[summary]
                body = row[description]
                
                # Check if the label exists in the row
                if name < len(row):
                    label = row[name]
                    date = row[created]
                    # Write the line to the output CSV file
                    csv_writer.writerow([title, body, label, date])
                else:
                    logging.warning(f'Skipping row without label in file {input_csv_filename}: {row}')

    logging.info(f'The data has been successfully exported to the file {output_csv_filename}')

logging.info("All CSV files have been cleaned")