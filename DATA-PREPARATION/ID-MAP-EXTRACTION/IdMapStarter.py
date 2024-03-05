import subprocess

# Base Path
base_path = 'SCRIPTS/ID-MAP-EXTRACTION/processes/'
# Paths
mongo_export = base_path + 'MongoExport.py'
clean_csv = base_path + 'CleanCSV.py'
label_mapping = base_path + 'LabelMapping.py'

# Run scripts
subprocess.run(['python', mongo_export])
subprocess.run(['python', clean_csv])
subprocess.run(['python', label_mapping])