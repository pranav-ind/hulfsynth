#!/bin/bash

# This script finds all available GPUs with a minimum amount of free memory
# and launches a Weights & Biases agent for each, allowing for parallel
# hyperparameter tuning within the same sweep.

#Note: This script is generated using AI; so double check before execution
# ---------------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------------

# 1. Provide the W&B sweep ID here.
SWEEP_ID="s7mtfsom"

update_gpu_info() {
  # 2. Define the minimum free memory threshold in MiB (20 GB = 20 * 1024 MiB)
  MIN_MEMORY_MB=47104

  # ---------------------------------------------------------------------------------
  # SCRIPT LOGIC
  # ---------------------------------------------------------------------------------

  # echo "Searching for GPUs with more than $MIN_MEMORY_MB MiB (~20 GB) of free memory..."
  

  # Get a list of all GPU IDs and their free memory.
  GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  # Check if the command was successful
  if [ -z "$GPU_INFO" ]; then
    echo "Error: Could not retrieve GPU information. Please ensure 'nvidia-smi' is working."
    exit 1
  fi

  # Declare SUITABLE_GPUS as a local array to prevent it from leaking outside the function's scope.
  local -a SUITABLE_GPUS=()

  # Read the GPU information line by line
  while read -r line; do
    # Extract GPU ID and Free Memory from the CSV output
    GPU_ID=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    GPU_FREE_MEMORY=$(echo "$line" | cut -d',' -f2 | tr -d ' ')

    if (( GPU_FREE_MEMORY > MIN_MEMORY_MB )); then
      # echo "Found GPU $GPU_ID with $((GPU_FREE_MEMORY / 1024)) GB of free memory."
      SUITABLE_GPUS+=("$GPU_ID")
    fi
  done <<< "$GPU_INFO"

  

  if [ ${#SUITABLE_GPUS[@]} -eq 0 ]; then
    echo "No GPUs found that meet the memory criteria."
  fi

  # Return the list of suitable GPU IDs by echoing them, one per line.
  # The "${SUITABLE_GPUS[@]}" syntax ensures all elements are printed.
  # echo "num_devices are: ${#SUITABLE_GPUS[@]}"
  echo "${SUITABLE_GPUS[@]}"
}
# gpu_id=5
# PIDS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$gpu_id")
# echo $PIDS

# if [[ -z "$PIDS" ]]; then
#     echo "GPU $gpu_id is FREE (no active processes found)."
#     return 0 # Success (GPU is free)
# fi
num_jobs=10



# returned_array=($(update_gpu_info))

# echo "Elements of returned_array:"
# for item in "${returned_array[@]}"; do
#   echo "$item"
# done
# echo "first element ${returned_array[0]}"

while [[ $num_jobs -gt 0 ]]; do
  
  suitable_gpus=($(update_gpu_info))
  echo "Suitable GPUS: $suitable_gpus and total num_devices are: "${#suitable_gpus[@]}""
  device_id="${suitable_gpus[0]}"
  
  # CUDA_VISIBLE_DEVICES=$device_id wandb agent "$SWEEP_ID" --count 1&
  echo "Allocating Device: $device_id and waiting till the process starts execution"
  

  ((num_jobs--))
  echo "Remaining number of jobs: $num_jobs"
done

  for gpu_id in "${SUITABLE_GPUS[@]}"; do
    echo "Starting agent for sweep $SWEEP_ID on GPU $gpu_id"
    
    # The key to parallelization: setting CUDA_VISIBLE_DEVICES for this specific process
    # and running it in the background (&).
    CUDA_VISIBLE_DEVICES=$gpu_id wandb agent "$SWEEP_ID" --count 3&
  done
  
#   echo ""
#   echo "All agents launched. They will run in the background."
#   echo "Waiting for all agents to complete..."
# fi

# # The 'wait' command ensures the main script doesn't exit until all
# # background agents have completed their sweep or been manually stopped.
# wait
# echo "All W&B agents have finished or been terminated."



