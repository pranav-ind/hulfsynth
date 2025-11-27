#!/bin/bash

# This script processes an array of elements, executing a command for each element
# on an available GPU (>= 30 GB free VRAM). It uses a round-robin approach
# to distribute the workload across all suitable GPUs.

# ---------------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------------

# 1. Define your array of elements here.


declare -a l1_list=( "300" "400" "500" "600" "700" "800")
declare -a l3_list=( "30" "40" "50" "60" "70" "80")
LOG_DIR = "/its/home/pi58/projects/hulfsynth/hulfsynth/vili_jobs_log/"
MIN_MEMORY_MB=30720



echo "Searching for GPUs with more than $MIN_MEMORY_MB MiB (~20 GB) of free memory..."
echo "----------------------------------------------------"

# Query all GPUs for their free memory in MiB.
GPU_FREE_MEMORY_LIST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | cut -d' ' -f2 | tr ',' '\n')

# Check if the command was successful
if [ -z "$GPU_FREE_MEMORY_LIST" ]; then
  echo "Error: Could not retrieve GPU information. Please ensure that 'nvidia-smi' is working and the NVIDIA drivers are loaded."
  exit 1
fi

# Get the count of GPUs to iterate through
GPU_COUNT=$(echo "$GPU_FREE_MEMORY_LIST" | wc -l)
declare -a SUITABLE_GPUS=()

for i in $(seq 0 $((GPU_COUNT - 1))); do
  GPU_FREE_MEMORY=$(echo "$GPU_FREE_MEMORY_LIST" | sed -n "$((i+1))p")

  if (( GPU_FREE_MEMORY > MIN_MEMORY_MB )); then
    echo "Found GPU $i with $((GPU_FREE_MEMORY / 1024)) GB of free memory."
    SUITABLE_GPUS+=($i)
  fi
done

echo "----------------------------------------------------"

if [ ${#SUITABLE_GPUS[@]} -eq 0 ]; then
  echo "No GPUs found with more than $((MIN_MEMORY_MB / 1024)) GB of free memory. No script will be executed."
else
  NUM_GPUS=${#SUITABLE_GPUS[@]}
  echo "Found $NUM_GPUS suitable GPU(s)."

  # Loop through the array of elements
  for i in "${!l1_list[@]}"; do
    l1="${l1_list[$i]}"
    l3="${l3_list[$i]}"
    
    
    gpu_id_index=$((i % NUM_GPUS)) # Use modulo to cycle through the available GPUs
    gpu_id=${SUITABLE_GPUS[$gpu_id_index]}
    
    echo "Processing element l1='${l1}', l3='${l3}' on GPU $gpu_id..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id python main_ixi.py -l1 ${l1} -l3 ${l3} -l4 0.1 -l5 0.1 -ep 10000 > "${LOG_DIR}/gpu_${gpu_id}.log" 2>&1 &
    
    
    
    
  done
  
  
fi



