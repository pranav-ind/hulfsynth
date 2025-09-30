#!/bin/bash

# This script runs a fixed number of jobs in parallel, dynamically detecting
# available GPUs with sufficient free VRAM and throttling execution.

# --- USER CONFIGURATION ---
# 1. Define the number of total jobs you want to run
JOB_COUNT=10
SWEEP_ID="s7mtfsom"

# 2. Define the minimum free memory threshold in MiB (20 GB = 20 * 1024 MiB)
MIN_MEMORY_MB=20480

# --- SCRIPT LOGIC: DYNAMIC GPU DETECTION ---
echo "Searching for GPUs with more than $((MIN_MEMORY_MB / 1024)) GB of free memory..."
echo "--------------------------------------------------------"

# Query all GPUs for their index and free memory in MiB.
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

if [ -z "$GPU_INFO" ]; then
  echo "Error: Could not retrieve GPU information. Is 'nvidia-smi' available?"
  exit 1
fi

declare -a SUITABLE_GPUS=()

# Filter GPUs based on the free memory threshold
while read -r line; do
  GPU_ID=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
  GPU_FREE_MEMORY=$(echo "$line" | cut -d',' -f2 | tr -d ' ')

  if (( GPU_FREE_MEMORY > MIN_MEMORY_MB )); then
    echo "Found suitable GPU $GPU_ID with $((GPU_FREE_MEMORY / 1024)) GB of free memory."
    SUITABLE_GPUS+=("$GPU_ID")
  fi
done <<< "$GPU_INFO"

# Set the number of available devices dynamically
NUM_DEVICES=${#SUITABLE_GPUS[@]}

if [ $NUM_DEVICES -eq 0 ]; then
    echo "No GPUs found that meet the memory criteria. Exiting."
    exit 0
fi

echo "--------------------------------------------------------"
echo "Starting parallel execution of $JOB_COUNT jobs on $NUM_DEVICES suitable GPU(s)."

# --- SCRIPT LOGIC: PARALLEL EXECUTION WITH THROTTLING ---

# Initialize a variable to track the current job number (0-indexed for device cycling)
current_job_number=0

# Loop until the total number of jobs is launched
while [ $current_job_number -lt $JOB_COUNT ]; do
    
    # 1. Determine the device ID using a round-robin approach
    # The device_index cycles from 0 up to NUM_DEVICES-1
    device_index=$((current_job_number % NUM_DEVICES))
    
    # Get the actual physical GPU ID from the SUITABLE_GPUS array
    physical_device_id=${SUITABLE_GPUS[$device_index]}

    # 2. Launch the job in the background
    job_id=$((current_job_number + 1))
    echo "Launching Job $job_id on physical GPU $physical_device_id..."

    # The command to execute in the background
    # Replace "your_command_here" with your actual Python script or binary
    # The "&" symbol is crucial for background execution
    # CUDA_VISIBLE_DEVICES=$physical_device_id your_command_here --job-id="$job_id" &
    CUDA_VISIBLE_DEVICES=$gpu_id wandb agent "$SWEEP_ID" --count 1&
    # 3. Wait for a device to free up if all devices are now busy
    # This check ensures we don't exceed the number of available GPUs.
    # We check if the number of currently running background jobs equals the number of devices.
    if [ $(jobs -r | wc -l) -ge $NUM_DEVICES ]; then
        echo "All $NUM_DEVICES devices are busy. Waiting for a job to complete (wait -n)..."
        wait -n
    fi

    # Increment the job number
    current_job_number=$((current_job_number + 1))
done

# Wait for all remaining background jobs to complete
echo "--------------------------------------------------------"
echo "All $JOB_COUNT jobs have been submitted. Waiting for final jobs to finish..."
wait

echo "All jobs have successfully completed."
