import os
import pandas as pd

# Base Path CSV
base_path = "CSV/"
# Path Mapped files
mapped_folder = os.path.join(base_path, "MAPPED/")

# Initializes a list to store information for each file
file_info_list = []

# Total sums initialized to zero
total_bug_count = 0
total_enhancement_count = 0
total_question_count = 0

# Iterate over each file in the folder
for filename_input in os.listdir(mapped_folder):
    # Build the full path to the file
    input_csv_filename = os.path.join(mapped_folder, filename_input)

    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv_filename)

    # Converts the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Extract the minimum year from the 'date' column excluding NaT (Not a Time) values
    max_year = df['date'].dt.year.min(skipna=True)

    # Calculate label occurrences for the entire file
    bug_count = (df['label'].str.lower() == 'bug').sum()
    enhancement_count = (df['label'].str.lower() == 'enhancement').sum()
    question_count = (df['label'].str.lower() == 'question').sum()

    # Update totals
    total_bug_count += bug_count
    total_enhancement_count += enhancement_count
    total_question_count += question_count

    # Add the file information to the list
    file_info_list.append({
        'Jira Name': os.path.splitext(filename_input)[0],
        'Year': int(max_year),
        'Bug': bug_count,
        'Enhancement': enhancement_count,
        'Question': question_count
    })

# Add a new row to the list with total sums
file_info_list.insert(0, {
    'Jira Name': 'Total',
    'Year': '', 
    'Bug': total_bug_count,
    'Enhancement': total_enhancement_count,
    'Question': total_question_count
})

# Create a DataFrame from the list information
distribution_table = pd.DataFrame(file_info_list)

# Create the 'DISTRIBUTION/' directory if it doesn't exist
os.makedirs("DISTRIBUTION", exist_ok=True)
# Save the distribution table to a new CSV file
distribution_table.to_csv('DISTRIBUTION/distribution_table.csv', index=False)

# View the distribution table
print(distribution_table)