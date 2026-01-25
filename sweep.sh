#!/bin/bash

# Script to run all configurations for FAST model training
# Configurations:
#   - Model: transformer, mamba2
#   - Spatial Projection: True, False

# use USE_NINJA=1 MAX_JOBS=8 uv sync to build deps

# Ensure the script stops on error
set -e

if [ ! -f "Processed/BCIC2020Track3.h5" ]; then
    echo "Missing Processed/BCIC2020Track3.h5. Run:"
    echo "  uv run python BCIC2020Track3_preprocess.py"
    exit 1
fi

# Loop through models
for model in "transformer" "mamba2"; do
    # Loop through spatial projection settings
    for proj in "True" "False"; do
        echo "----------------------------------------------------------------"
        echo "Running configuration: Model=$model, Spatial Projection=$proj"
        echo "----------------------------------------------------------------"
        
        # Run the training script
        # Using default accelerator (mps) as per train.py default, or override if needed.
        # Assuming running on the machine where train.py is located.
        # except for model "mamba2", use_spatial_projection is True
        # we skip that one for now
        if [ "$model" == "mamba2" ] && [ "$proj" == "False" ]; then
            continue
        fi
        uv run python train.py \
            --model "$model" \
            --use_spatial_projection "$proj" \
            --folds "0-15" \
            --accelerator "gpu"
            
        echo "Finished configuration: Model=$model, Spatial Projection=$proj"
        echo ""
    done
done

echo "All configurations completed successfully."
