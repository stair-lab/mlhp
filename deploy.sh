#!/bin/bash
set -e

# Activate conda environment
source /lfs/local/0/sttruong/miniconda3/etc/profile.d/conda.sh
conda activate mlhp

# Navigate to project
cd /lfs/skampere2/0/sttruong/mlhp

# Pull latest changes
git pull

# Build the book
quarto render --to html --profile html

# Deploy to www
rsync -av --delete _book/ /afs/cs/group/koyejolab/mlhp/www/

echo "Deployed successfully!"
