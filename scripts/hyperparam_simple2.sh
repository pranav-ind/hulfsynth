#!/bin/bash

# Define your commands or scripts for each GPU

#IXI dataset
commands=(
    
    "CUDA_VISIBLE_DEVICES=5 python hyperparam_1.py -dn 0023"
    "CUDA_VISIBLE_DEVICES=6 python hyperparam_1.py -dn 0027"
    "CUDA_VISIBLE_DEVICES=7 python hyperparam_1.py -dn 0035"
    # "CUDA_VISIBLE_DEVICES=3 python hyperparam_1.py -dn 0011"
    # "CUDA_VISIBLE_DEVICES=4 python hyperparam_1.py -dn 0015"
)

# Launch each command in the background
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd &"
done

# Wait for all background jobs to finish
wait
echo "All jobs completed."