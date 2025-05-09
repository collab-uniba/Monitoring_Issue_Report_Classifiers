#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <classify|similarity|stat-test> <config-file-name>"
    exit 1
fi

mode="$1"
config_file="$2"
default_config_dir="config"

case "$mode" in
    classify)
        echo "Running $mode with $config_file..."
        python "classification/classify.py" --config "$default_config_dir/$config_file"
        ;;
    similarity)
        echo "Running $mode with $config_file..."
        python "classification/similarity.py" --config "$default_config_dir/$config_file" --include-distribution --mode "mean"
        ;;
    stat-test)
        echo "Running $mode with $config_file..."
        python "statistical-tests/stat-test.py" --config "$default_config_dir/$config_file" --normalize
        ;;
    *)
        echo "Invalid mode: $mode. Use 'classify', 'similarity' or 'stat-test'."
        exit 1
        ;;
esac
echo "Finished running $mode with $config_file."