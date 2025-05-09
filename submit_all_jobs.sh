#!/bin/bash

# Define all config files
configs=(
    #"config-apache-y1.yaml"
    #"config-jira-y1.yaml"
    #"config-redhat-y1.yaml"
    "config-apache-y1-setfit.yaml"
    "config-jira-y1-setfit.yaml"
    "config-redhat-y1-setfit.yaml"
    #"config-apache-m1-setfit.yaml"
    #"config-jira-m1-setfit.yaml"
    #"config-redhat-m1-setfit.yaml"
    #"config-apache-m1.yaml"
    #"config-jira-m1.yaml"
    #"config-redhat-m1.yaml"

)

mode=$1

if [ -z "$mode" ]; then
    echo "Usage: $0 <classify|similarity|stat-test>"
    exit 1
fi

mkdir -p logs/$mode

for config in "${configs[@]}"; do
    jobname="${mode}_$(basename "$config" .yaml)"
    log_out="logs/$mode/${jobname}.out"
    log_err="logs/$mode/${jobname}.err"

    sbatch --job-name="$jobname" \
           --output="$log_out" \
           --error="$log_err" \
           --export=MODE="$mode",CONFIG="$config" \
           slurm_job_template.sh
done
echo "All jobs submitted for mode: $mode"
