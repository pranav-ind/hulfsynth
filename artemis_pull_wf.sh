#!/bin/bash

# ssh artemis
source ~/.bashrc
echo "sourcing bash..."

 
cd ./projects/hulfsynth/
echo "Navigating to the repository directory ..."

conda activate mri_recon
echo "Pulling from origin main..."
git pull origin main