#!/bin/bash

#this workflow is meant to be executed on artemis
#this workflow pushes to respective git repo; 
# Navigate to your repository directory (optional)
cd /Users/pi58/Library/CloudStorage/Box-Box/PhD/MPhil/Projects/Hulf_Synth/

echo "Adding all changes..."
git add .

echo "Committing changes..."
read -p "Enter your commit message: " commit_message
git commit -m "$commit_message"

echo "Pushing to origin main..."
git push origin main
echo "Git push complete."

echo "logging into artemis..."
ssh -t artemis 'cd /its/home/pi58/projects/hulfsynth/; git pull origin main; bash -l'

