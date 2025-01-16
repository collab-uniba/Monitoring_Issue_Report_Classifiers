import os
import pandas as pd
import argparse

def generate_overall_distribution(mapped_folder):
    # Initializes a list to store information for each file
    file_info_list = []

    # Total sums initialized to zero
    total_bug_count = 0
    total_feature_count = 0
    total_question_count = 0

    # Iterate over each file in the folder
    for filename_input in os.listdir(mapped_folder):
        # Build the full path to the file
        input_csv_filename = os.path.join(mapped_folder, filename_input)

        # Load the CSV into a DataFrame
        df = pd.read_csv(input_csv_filename)

        # Converts the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

        # Extract the minimum and maximum year from the 'date' column excluding NaT (Not a Time) values
        min_year = df['date'].dt.year.min(skipna=True)
        max_year = df['date'].dt.year.max(skipna=True)

        # Calculate label occurrences for the entire file
        bug_count = (df['label'].str.lower() == 'bug').sum()
        feature_count = (df['label'].str.lower() == 'feature').sum()
        question_count = (df['label'].str.lower() == 'question').sum()

        # Update totals
        total_bug_count += bug_count
        total_feature_count += feature_count
        total_question_count += question_count

        # Add the file information to the list
        file_info_list.append({
            'Jira Name': os.path.splitext(filename_input)[0],
            'Year First Issue': int(min_year),
            'Year Last Issue': int(max_year),
            'Bug': bug_count,
            'Feature': feature_count,
            'Question': question_count
        })

    # Add a new row to the list with total sums
    file_info_list.insert(0, {
        'Jira Name': 'Total',
        'Year First Issue': '', 
        'Year Last Issue': '', 
        'Bug': total_bug_count,
        'Feature': total_feature_count,
        'Question': total_question_count
    })

    # Create a DataFrame from the list information
    distribution_table = pd.DataFrame(file_info_list)

    # Create the 'DISTRIBUTION-TABLE/' directory if it doesn't exist
    os.makedirs("distribution-table", exist_ok=True)
    # Save the distribution table to a new CSV file
    distribution_table.to_csv('distribution-table/all_projects.csv', index=False)

    # View the distribution table
    print(distribution_table)

def generate_per_project_distribution(mapped_folder):
    # Create the 'DISTRIBUTION-TABLE/' directory if it doesn't exist
    os.makedirs("distribution-table", exist_ok=True)
    # Create the 'DISTRIBUTION-TABLE/WINDOWS' directory if it doesn't exist
    os.makedirs("distribution-table/windows", exist_ok=True)

    # Iterate over each file in the folder
    for filename_input in os.listdir(mapped_folder):
        # Build the full path to the file
        input_csv_filename = os.path.join(mapped_folder, filename_input)

        # Load the CSV into a DataFrame
        df = pd.read_csv(input_csv_filename)

        # Converts the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

        # Extract the minimum and maximum year from the 'date' column excluding NaT (Not a Time) values
        min_year = df['date'].dt.year.min(skipna=True)
        max_year = df['date'].dt.year.max(skipna=True)

        # Group by year and label, and calculate counts
        grouped_data = df.groupby([df['date'].dt.year, 'label']).size().unstack(fill_value=0)

        # Create a list to store dictionaries for each row
        distribution_data = []

        # Add a row for the first and last year
        distribution_data.append({
            'Year': 'Year First Issue',
            'Bug': '',
            'Feature': '',
            'Question': '',
            'Year Value': int(min_year)
        })
        distribution_data.append({
            'Year': 'Year Last Issue',
            'Bug': '',
            'Feature': '',
            'Question': '',
            'Year Value': int(max_year)
        })

        # Iterate over the unique years in the data
        for year in grouped_data.index:
            # Get counts for each label in the current year
            bug_count = grouped_data.loc[year, 'Bug'] if 'Bug' in grouped_data.columns else 0
            feature_count = grouped_data.loc[year, 'Feature'] if 'Feature' in grouped_data.columns else 0
            question_count = grouped_data.loc[year, 'Question'] if 'Question' in grouped_data.columns else 0

            # Add a dictionary to the list for the current year
            distribution_data.append({
                'Year': str(int(year)),
                'Bug': bug_count,
                'Feature': feature_count,
                'Question': question_count,
                'Year Value': int(year)
            })

        # Convert the list of dictionaries to a DataFrame
        project_distribution_table = pd.DataFrame(distribution_data)

        # Sort the DataFrame by 'Year Value' to ensure proper ordering
        project_distribution_table = project_distribution_table.sort_values(by='Year Value')

        # Drop the 'Year Value' column as it's no longer needed
        project_distribution_table = project_distribution_table.drop(columns=['Year Value'])

        # Save the project distribution table to a new CSV file in the 'DISTRIBUTION' folder
        project_csv_filename = os.path.join("distribution-table/windows", f'{os.path.splitext(filename_input)[0]}.csv')
        project_distribution_table.to_csv(project_csv_filename, index=False)

    print("Script completed successfully.")


def main():
    parser = argparse.ArgumentParser(description='Generate distribution tables based on the provided mode.')
    parser.add_argument('--mode', type=str, required=True, choices=['overall', 'per-project'],
                        help='Mode to generate distribution tables: "overall" for a single table, "per-project" for yearly tables per project.')
    
    args = parser.parse_args()

    # Base Path data
    base_path = "data/"
    # Path Mapped files
    mapped_folder = os.path.join(base_path, "mapped")

    if args.mode == 'overall':
        generate_overall_distribution(mapped_folder)
    elif args.mode == 'per-project':
        generate_per_project_distribution(mapped_folder)

if __name__ == "__main__":
    main()