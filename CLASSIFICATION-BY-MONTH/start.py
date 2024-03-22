import papermill as pm
import glob
import os

def run_notebook(yaml_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, os.path.basename(yaml_file).replace('.yaml', '') + '_output.ipynb')

    pm.execute_notebook(
        './classification-from-model-month.ipynb',  # Path to the source notebook
        output_path,                           # Path to the output notebook
        parameters={'config_path': yaml_file}  # Overrides the parameter in the notebook
    )

output_folder = './output'

yaml_files = glob.glob('./*.yaml')

for yaml_file in yaml_files:
    run_notebook(yaml_file, output_folder)
