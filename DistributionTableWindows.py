import os
import pandas as pd

# Base Path CSV
base_path = "CSV/"
# Path Mapped files
mapped_folder = os.path.join(base_path, "MAPPED/")

# Create the 'DISTRIBUTION/' directory if it doesn't exist
os.makedirs("DISTRIBUTION", exist_ok=True)

# Iterate over each file in the folder
for filename_input in os.listdir(mapped_folder):
    # Build the full path to the file
    input_csv_filename = os.path.join(mapped_folder, filename_input)

    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv_filename)

    # Converts the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Group by year and label, and calculate counts
    grouped_data = df.groupby([df['date'].dt.year, 'label']).size().unstack(fill_value=0)

    # Create a list to store dictionaries for each row
    distribution_data = []

    # Iterate over the unique years in the data
    for year in grouped_data.index:
        # Get counts for each label in the current year
        bug_count = grouped_data.loc[year, 'Bug'] if 'Bug' in grouped_data.columns else 0
        enhancement_count = grouped_data.loc[year, 'Enhancement'] if 'Enhancement' in grouped_data.columns else 0
        question_count = grouped_data.loc[year, 'Question'] if 'Question' in grouped_data.columns else 0

        # Add a dictionary to the list for the current year
        distribution_data.append({
            'Year': str(int(year)),
            'Bug': bug_count,
            'Enhancement': enhancement_count,
            'Question': question_count
        })

    # Convert the list of dictionaries to a DataFrame
    project_distribution_table = pd.DataFrame(distribution_data)

    # Save the project distribution table to a new CSV file in the 'DISTRIBUTION' folder
    project_csv_filename = os.path.join("DISTRIBUTION", f'{os.path.splitext(filename_input)[0]}_distribution.csv')
    project_distribution_table.to_csv(project_csv_filename, index=False)

print("Script completed successfully.")
