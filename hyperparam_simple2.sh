#!/bin/bash

# Define your commands or scripts for each GPU


commands=(
    "CUDA_VISIBLE_DEVICES=6 python hyperparam_1.py -dn 102 -id 6  > ./vili_jobs_log/sens6_output.txt"
    "CUDA_VISIBLE_DEVICES=7 python hyperparam_1.py -dn 102 -id 7 > ./vili_jobs_log/sens7_output.txt"
    "CUDA_VISIBLE_DEVICES=0 python hyperparam_1.py -dn 102 -id 8  > ./vili_jobs_log/sens8_output.txt"
)

# Launch each command in the background
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd &"
done

# Wait for all background jobs to finish
wait
echo "All jobs completed."