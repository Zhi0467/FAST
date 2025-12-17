#!/bin/bash

# Script to run all configurations for FAST model training
# Configurations:
#   - Model: transformer, mamba2
#   - Spatial Projection: True, False

# Ensure the script stops on error
set -e

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
        python train.py \
            --model "$model" \
            --use_spatial_projection "$proj" \
            --folds "0-15" \
            --accelerator "gpu"
            
        echo "Finished configuration: Model=$model, Spatial Projection=$proj"
        echo ""
    done
done

echo "All configurations completed successfully."

