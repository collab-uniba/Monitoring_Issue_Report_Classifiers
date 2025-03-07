#!/bin/bash
# Enable error handling
set -e
default_config_dir="classification"
# Define available configurations
declare -a configs=(
    "config-apache-y1.yaml"
    "config-jira-y1.yaml"
    "config-mongodb-y1.yaml"
    "config-redhat-y1.yaml"
    "config-apache-m1.yaml"
    "config-jira-m1.yaml"
    "config-redhat-m1.yaml"
    "config-mongodb-m1.yaml"
)
# Check for mode argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <classify|similarity|stat-test>"
    exit 1
fi
mode="$1"
case "$mode" in
    classify)
        for config in "${configs[@]}"; do
            echo "Running $mode with $config..."
            python "classification/classify.py" --config "$default_config_dir/$config"
            if [ $? -ne 0 ]; then
                echo "Error encountered while running $config. Exiting."
                exit 1
            fi
        done
        ;;
    similarity)
        for config in "${configs[@]}"; do
            echo "Running $mode with $config..."
            python "classification/similarity.py" --config "$default_config_dir/$config" --include-distribution --mode "mean"
            if [ $? -ne 0 ]; then
                echo "Error encountered while running $config. Exiting."
                exit 1
            fi
        done
        ;;
    stat-test)
        for config in "${configs[@]}"; do
            echo "Running $mode with $config..."
            python "statistical-tests/stat-test.py" --config "$default_config_dir/$config --normalize"
            if [ $? -ne 0 ]; then
                echo "Error encountered while running $config. Exiting."
                exit 1
            fi
        done
        ;;
    *)
        echo "Invalid mode: $mode. Use 'classify', 'similarity' or 'stat-test'."
        exit 1
        ;;
esac
echo "All $mode operations completed successfully!"