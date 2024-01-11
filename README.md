# JiraRepos Database Tools

This set of tools facilitates the extraction and preprocessing of data from the JiraRepos MongoDB database. Follow the steps below to efficiently manage and analyze your Jira data.

## Prerequisites
- MongoDB Compass installed
- Python 3.x installed

## Steps

### 1. Extract Data from MongoDB Compass

1. Open MongoDB Compass.
2. Run the `MongoExport.py` tool for each collection in the JiraRepos database. The tool automates the extraction process and uses the following query in the "Project" field:

    ```json
    {
        "_id": 0,
        "fields.summary": 1,
        "fields.description": 1,
        "fields.issuetype.name": 1,
        "fields.created": 1
    }
    ```

### 2. Clean and Map Labels

1. Run the `CleanCSV.py` tool to preprocess the CSV files.
   
    ```bash
    python CleanCSV.py
    ```

2. Subsequently, execute the `LabelMapping.py` tool to apply label mappings based on predefined rules.
   
    ```bash
    python LabelMapping.py
    ```

### 3. Generate Distribution Table

1. Run the `DistributionTable.py` tool to generate the distribution table.
   
    ```bash
    python DistributionTable.py
    ```

## Additional Information

- The cleaned and labeled CSV files will be stored in the "CSV/CLEANED" and "CSV/MAPPED" folders, respectively.
- The distribution table will be created and placed in the "DISTRIBUTION" folder.
- Review the log files generated during the execution for detailed information on the process.
  - Only logs with warning messages are stored to minimize clutter.
  
Now, you have organized, preprocessed, and generated a distribution table for your Jira data, making it ready for further analysis or integration into your workflow.

**Author:** Simone Le Noci
