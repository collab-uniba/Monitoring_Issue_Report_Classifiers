import csv

def remove_duplicates(input_file, output_file):
    unique_rows = set()

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            unique_rows.add(tuple(row))

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(unique_rows)
        
if __name__ == "__main__":
    # Replace with the path to your generic input CSV file
    input_csv = "logs/warning_mapping.csv"
    # Replace with the desired path to the generic output CSV file
    output_csv = "label_not_mapped.csv"

    remove_duplicates(input_csv, output_csv)
    print("Duplicate rows successfully removed.")
